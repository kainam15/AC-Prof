"""Storage statistics keep Docker accounting separate from filesystem availability."""
import json
import subprocess
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.images import inventory as image_management
from acprof.host.images.inventory import DockerConnection, ImageManagementError


class TestDockerStorage:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.connection = DockerConnection(("--context", "local-test"), "local-test")
        self.info = {"ID": "daemon-a", "DockerRootDir": "/srv/docker", "Name": "local-machine",
                     "OperatingSystem": "Linux"}
        self.endpoint = "unix:///var/run/docker.sock"
        self.rows = [
            {"Type": "Images", "Size": "2.147GB", "Reclaimable": "1.074GB (50%)"},
            {"Type": "Containers", "Size": "12kB", "Reclaimable": "0B (0%)"},
            {"Type": "Local Volumes", "Size": "3MB", "Reclaimable": "1MB (33%)"},
            {"Type": "Build Cache", "Size": "4MB", "Reclaimable": "4MB"},
        ]
        self.fail_df = False
        self.commands = []
        patcher = patch("acprof.host.images.inventory.run_command", side_effect=self.run_docker)
        patcher.start()
        self._request.addfinalizer(partial(patcher.stop))
        patcher = patch("os.uname", return_value=SimpleNamespace(nodename="local-machine"))
        patcher.start()
        self._request.addfinalizer(partial(patcher.stop))
        patcher = patch("os.statvfs", return_value=SimpleNamespace(
            f_frsize=4096, f_blocks=1000, f_bfree=300, f_bavail=250))
        self.statvfs = patcher.start()
        self._request.addfinalizer(partial(patcher.stop))

    def run_docker(self, command, **kwargs):
        self.commands.append(command)
        assert not (kwargs.get("shell"))
        assert (kwargs["timeout"]) > (0)
        assert (command[:3]) == (["docker", "--context", "local-test"])
        args = command[3:]
        if args[0] == "info":
            output = self.info["ID"] if args[-1] == "{{.ID}}" else json.dumps(self.info)
        elif args[:2] == ["context", "inspect"]:
            output = self.endpoint
        elif args[:2] == ["system", "df"]:
            assert ("-v") not in (args)
            if self.fail_df:
                return subprocess.CompletedProcess(command, 1, "", "permission denied")
            output = "\n".join(json.dumps(row) for row in self.rows)
        else:
            pytest.fail(f"Unexpected Docker command: {command}")
        return subprocess.CompletedProcess(command, 0, output, "")

    def read(self):
        return image_management.read_storage(self.connection, daemon_id="daemon-a")

    def test_summary_converts_decimal_units_and_preserves_reserved_blocks(self):
        result = self.read()
        assert (result.root_dir) == ("/srv/docker")
        self.statvfs.assert_called_once_with("/srv/docker")
        assert result.disk is not None
        assert ((result.disk.total, result.disk.used, result.disk.available)) == ((4096000, 2867200, 1024000))
        assert (result.total_bytes) == (2154012000)
        assert (result.reclaimable_bytes) == (1079000000)
        assert (result.usage[1].reclaimable_bytes) == (0)

    @pytest.mark.parametrize('rows_case', range(3), ids=['rows[:3]', "[*rows[:3], {**rows[3], 'Size': 'N/A'}]", '[*rows, rows[0]]'])
    def test_missing_or_invalid_category_is_unknown_not_zero_or_partial_total(self, rows_case):
        rows = tuple((self.rows[:3], [*self.rows[:3], {**self.rows[3], 'Size': 'N/A'}], [*self.rows, self.rows[0]]))[rows_case]
        self.rows = rows
        result = self.read()
        assert (result.total_bytes) is None
        assert (result.warnings)
        assert (result.disk) is not None

    @pytest.mark.parametrize('endpoint_case', range(3), ids=["('ssh://remote', 'local-machine', 'Linux')", "(endpoint, 'docker-desktop', 'Docker Desktop')", "(endpoint, 'other-machine', 'Linux')"])
    def test_remote_or_vm_daemon_never_reports_the_client_disk(self, endpoint_case):
        (endpoint, name, system) = tuple((('ssh://remote', 'local-machine', 'Linux'), (self.endpoint, 'docker-desktop', 'Docker Desktop'), (self.endpoint, 'other-machine', 'Linux')))[endpoint_case]
        self.endpoint = endpoint
        self.info.update(Name=name, OperatingSystem=system)
        result = self.read()
        assert (result.disk) is None
        assert (result.total_bytes) == (2154012000)
        assert (result.warnings)
        self.statvfs.assert_not_called()

    def test_df_failure_keeps_disk_and_marks_docker_totals_unknown(self):
        self.fail_df = True
        result = self.read()
        assert (result.disk) is not None
        assert (result.total_bytes) is None
        assert (result.reclaimable_bytes) is None
        assert (result.warnings)

    def test_unreadable_root_does_not_fall_back_to_another_filesystem(self):
        self.statvfs.side_effect = PermissionError("permission denied")
        result = self.read()
        assert (result.disk) is None
        assert (result.total_bytes) == (2154012000)
        self.statvfs.assert_called_once_with("/srv/docker")

    def test_disk_permission_failure_preserves_docker_statistics_and_reason(self):
        self.statvfs.side_effect = PermissionError("denied /srv/docker")
        result = self.read()
        assert result.total_bytes == 2154012000
        assert result.disk_error.code == "disk_permission"
        assert "denied /srv/docker" in result.disk_error.detail

    def test_df_permission_failure_preserves_disk_and_socket_diagnosis(self):
        self.fail_df = True
        result = self.read()
        assert result.disk is not None
        assert result.usage_error.code == "permission"
        assert "permission denied" in result.usage_error.detail

    @pytest.mark.parametrize(("detail", "code"), [
        ("permission denied while connecting to /var/run/docker.sock", "permission"),
        ("Cannot connect to the Docker daemon. Is the docker daemon running?", "daemon"),
        ('context "remote": context not found', "context"),
    ])
    def test_info_failure_explains_cause(self, detail, code):
        with patch("acprof.host.images.inventory.run_command", return_value=
                   subprocess.CompletedProcess([], 1, "", detail)):
            with pytest.raises(ImageManagementError) as caught:
                self.read()
        assert caught.value.code == code
        assert detail in caught.value.detail
        self.statvfs.assert_not_called()

    def test_changed_daemon_rejects_mixing_inventory_and_storage(self):
        self.info["ID"] = "different-daemon"
        with pytest.raises(ImageManagementError, match="Docker 环境已改变"):
            self.read()
        self.statvfs.assert_not_called()

    def test_context_inspection_failure_does_not_hide_usage(self):
        original = self.run_docker

        def fail_context(command, **kwargs):
            if "inspect" in command:
                return subprocess.CompletedProcess(command, 1, "", "context local-test is inaccessible")
            return original(command, **kwargs)

        with patch("acprof.host.images.inventory.run_command", side_effect=fail_context):
            result = self.read()
        assert result.disk is None
        assert result.disk_error.code == "context"
        assert result.total_bytes == 2154012000
        self.statvfs.assert_not_called()

    @pytest.mark.parametrize("part", ["info", "df"])
    def test_query_timeout_retains_its_cause(self, part):
        original = self.run_docker

        def time_out(command, **kwargs):
            if part in command:
                raise subprocess.TimeoutExpired(command, 30)
            return original(command, **kwargs)

        with patch("acprof.host.images.inventory.run_command", side_effect=time_out):
            if part == "info":
                with pytest.raises(ImageManagementError) as caught:
                    self.read()
                error = caught.value
            else:
                result = self.read()
                assert result.disk is not None
                error = result.usage_error
        assert error.code == "timeout"

    @pytest.mark.parametrize("info", [{}, {"ID": 42}, {"ID": ""}, []])
    def test_invalid_info_never_provides_a_trusted_identity(self, info):
        self.info = info
        with pytest.raises(ImageManagementError) as caught:
            self.read()
        assert caught.value.code == "invalid"
        self.statvfs.assert_not_called()

    def test_invalid_root_does_not_discard_valid_usage(self):
        self.info["DockerRootDir"] = None
        result = self.read()
        assert result.disk is None
        assert result.disk_error.code == "invalid"
        assert result.total_bytes == 2154012000

    def test_malformed_category_retains_other_categories(self):
        self.rows[1] = ["invalid"]
        result = self.read()
        assert result.usage[0].size_bytes == 2147000000
        assert result.usage[1].size_bytes is None
        assert result.usage[2].size_bytes == 3000000
        assert result.total_bytes is None
        assert result.usage_error.code == "invalid"

    def test_nonfinite_size_is_unknown(self):
        self.rows[0]["Size"] = "9" * 400 + "GB"
        result = self.read()
        assert result.usage[0].size_bytes is None
        assert result.usage[1].size_bytes == 12000
        assert result.total_bytes is None

    def test_daemon_change_during_optional_query_rejects_snapshot(self):
        original = self.run_docker

        def switch_daemon(command, **kwargs):
            result = original(command, **kwargs)
            if "df" in command:
                self.info["ID"] = "daemon-b"
            return result

        with patch("acprof.host.images.inventory.run_command", side_effect=switch_daemon):
            with pytest.raises(ImageManagementError) as caught:
                self.read()
        assert caught.value.code == "identity"

    def test_active_context_is_resolved_again_before_publishing(self):
        other = DockerConnection(("--context", "other"), "other")
        with patch("acprof.host.images.inventory._connection", side_effect=[self.connection, other]):
            with pytest.raises(ImageManagementError) as caught:
                image_management.read_storage()
        assert caught.value.code == "identity"
