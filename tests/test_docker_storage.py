"""Storage statistics keep Docker accounting separate from filesystem availability."""

import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host import image_management
from acprof.host.image_management import DockerConnection, ImageManagementError


class DockerStorageTests(unittest.TestCase):
    def setUp(self):
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
        patcher = patch("acprof.host.image_management.subprocess.run", side_effect=self.run_docker)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("os.uname", return_value=SimpleNamespace(nodename="local-machine"))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("os.statvfs", return_value=SimpleNamespace(
            f_frsize=4096, f_blocks=1000, f_bfree=300, f_bavail=250))
        self.statvfs = patcher.start()
        self.addCleanup(patcher.stop)

    def run_docker(self, command, **kwargs):
        self.commands.append(command)
        self.assertFalse(kwargs.get("shell"))
        self.assertGreater(kwargs["timeout"], 0)
        self.assertEqual(command[:3], ["docker", "--context", "local-test"])
        args = command[3:]
        if args[0] == "info":
            output = json.dumps(self.info)
        elif args[:2] == ["context", "inspect"]:
            output = self.endpoint
        elif args[:2] == ["system", "df"]:
            self.assertNotIn("-v", args)
            if self.fail_df:
                return subprocess.CompletedProcess(command, 1, "", "permission denied")
            output = "\n".join(json.dumps(row) for row in self.rows)
        else:
            self.fail(f"Unexpected Docker command: {command}")
        return subprocess.CompletedProcess(command, 0, output, "")

    def read(self):
        return image_management.read_storage(self.connection, daemon_id="daemon-a")

    def test_summary_converts_decimal_units_and_preserves_reserved_blocks(self):
        result = self.read()
        self.assertEqual(result.root_dir, "/srv/docker")
        self.statvfs.assert_called_once_with("/srv/docker")
        assert result.disk is not None
        self.assertEqual((result.disk.total, result.disk.used, result.disk.available),
                         (4096000, 2867200, 1024000))
        self.assertEqual(result.total_bytes, 2154012000)
        self.assertEqual(result.reclaimable_bytes, 1079000000)
        self.assertEqual(result.usage[1].reclaimable_bytes, 0)

    def test_missing_or_invalid_category_is_unknown_not_zero_or_partial_total(self):
        for rows in (self.rows[:3], [*self.rows[:3], {**self.rows[3], "Size": "N/A"}],
                     [*self.rows, self.rows[0]]):
            with self.subTest(rows=rows):
                self.rows = rows
                result = self.read()
                self.assertIsNone(result.total_bytes)
                self.assertTrue(result.warnings)
                self.assertIsNotNone(result.disk)

    def test_remote_or_vm_daemon_never_reports_the_client_disk(self):
        for endpoint, name, system in (("ssh://remote", "local-machine", "Linux"),
                                       (self.endpoint, "docker-desktop", "Docker Desktop"),
                                       (self.endpoint, "other-machine", "Linux")):
            with self.subTest(endpoint=endpoint, name=name):
                self.endpoint = endpoint
                self.info.update(Name=name, OperatingSystem=system)
                result = self.read()
                self.assertIsNone(result.disk)
                self.assertEqual(result.total_bytes, 2154012000)
                self.assertTrue(result.warnings)
        self.statvfs.assert_not_called()

    def test_df_failure_keeps_disk_and_marks_docker_totals_unknown(self):
        self.fail_df = True
        result = self.read()
        self.assertIsNotNone(result.disk)
        self.assertIsNone(result.total_bytes)
        self.assertIsNone(result.reclaimable_bytes)
        self.assertTrue(result.warnings)

    def test_unreadable_root_does_not_fall_back_to_another_filesystem(self):
        self.statvfs.side_effect = PermissionError("permission denied")
        result = self.read()
        self.assertIsNone(result.disk)
        self.assertEqual(result.total_bytes, 2154012000)
        self.statvfs.assert_called_once_with("/srv/docker")

    def test_changed_daemon_rejects_mixing_inventory_and_storage(self):
        self.info["ID"] = "different-daemon"
        with self.assertRaisesRegex(ImageManagementError, "Docker 环境已改变"):
            self.read()
        self.statvfs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
