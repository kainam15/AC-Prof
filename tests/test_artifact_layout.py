"""Result layout contracts, including interrupted and historical experiments."""
import csv
import io
import json
import tempfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from client_fixtures import patch_client

from acprof.artifact_layout import ArtifactLayout, case_sidecar
from acprof.host.client import ClientRunner
from acprof.host.client_config import ClientConfig
from acprof.host.run_state import RunState


@pytest.mark.parametrize('invalid', ('oversized', 'nonfinite'))
def test_manifest_uses_bounded_finite_json_reader(tmp_path, invalid):
    from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
    ArtifactLayout.for_new_run(tmp_path).initialize()
    manifest = tmp_path / 'result_manifest.json'
    if invalid == 'oversized':
        content = ' ' * (MAX_JSON_ARTIFACT_BYTES + 1) + manifest.read_text(encoding='utf-8')
        reason = 'read limit'
    else:
        content = manifest.read_text(encoding='utf-8').rstrip()[:-1] + ', "extra": NaN}'
        reason = 'non-finite'
    manifest.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match=reason):
        ArtifactLayout.discover(tmp_path)


class TestArtifactLayout:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        self.runner = ClientRunner(ClientConfig())

    def test_client_uses_the_experiment_slo_from_nested_case_directory(self):
        from acprof.host import client
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            layout = ArtifactLayout.for_new_run(root)
            layout.initialize()
            case = layout.case("org/model", 1, 4, "off")
            case.csv.parent.mkdir(parents=True)
            (root / "static_meta.json").write_text(json.dumps({
                "schema_version": 7, "latency_slo": {"threshold_s": 0.2, "source": "user"},
            }))
            settings = {
                "OUT_CSV": str(case.csv), "PROFILING_MODE": "full", "GPU_MODE": "off",
                "WARMUP": 0, "REPEAT": 1, "REPEAT_IN_WINDOW": 2,
                "USE_ENERGY": False, "USE_MIPS": False, "IDLE_DEBUG": False,
                "IDLE_SECONDS": 0.0, "IDLE_COOLDOWN_SECONDS": 0.0, "SNIFF_GROUPS_PATH": "",
                "energy_mod": None, "cpu_energy_mod": None, "resource_usage_mod": None,
                "COMPUTE_PROFILE_PLAN_FILE": "", "EXECUTION_PROFILE_PLAN_FILE": "",
                "input_scale_entries": [{"input_scale": 1.0, "scale_label": "one", "payload": {}}],
            }
            for name, value in settings.items():
                stack.enter_context(patch_client(self.runner, name, value))
            stack.enter_context(patch.object(client.requests, "get",
                return_value=SimpleNamespace(status_code=200, text="ok")))
            stack.enter_context(patch_client(self.runner, "_one_request", side_effect=[
                {"latency_app_s": 0.1, "effective_input_scale": 1.0},
                {"latency_app_s": 0.4, "effective_input_scale": 1.0},
            ]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.runner.main()
            with case.csv.open() as stream:
                row = next(csv.DictReader(stream))
            assert (float(row["latency_app_slow_ratio"])) == (0.5)
            assert (json.loads(case.requests.read_text())["latency_app_s"]) == ([0.1, 0.4])

    def test_flat_run_state_still_resumes_without_creating_a_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            root.mkdir()
            (root / "run_state.json").write_text(json.dumps({
                "schema_version": 1, "run_id": "legacy", "options": {}, "host": {},
                "status": "interrupted", "cases": {}, "artifacts": {}, "attempts": [],
            }))
            with patch("acprof.host.run_state.host_identity", return_value={}), \
                 patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", Path(temporary)):
                state = RunState(root, {}, resume=True, project_dir=temporary)
                assert (state.path) == (root / "run_state.json")
                state.close()
            assert not ((root / "result_manifest.json").exists())
            assert not ((root / ".acprof").exists())

    def test_request_publication_does_not_move_a_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = ArtifactLayout.for_new_run(root)
            layout.initialize()
            case = layout.case("org/model", 1, 4, "off")
            case.csv.parent.mkdir(parents=True)
            target = root / "unrelated.txt"
            target.write_text("preserve")
            case.requests.symlink_to(target)
            with pytest.raises(ValueError):
                case.retain_requests()
            assert (target.read_text()) == ("preserve")
            assert (case.requests.is_symlink())
            assert not (case.retained_requests.exists())

    def test_legacy_discovery_is_read_only_and_keeps_flat_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input_scale_plan.json").write_text("{}")
            layout = ArtifactLayout.discover(root)
            assert (layout.layout_version) == (1)
            assert (layout.path("input_scale_plan.json")) == (root / "input_scale_plan.json")
            assert (layout.path("posthoc_backups")) == (root / "posthoc_backups")
            assert (layout.case("org/model", 4, 8, "on").csv.name) == ("result_case_org--model_4c_8g_on.csv")
            assert ([p.name for p in root.iterdir()]) == (["input_scale_plan.json"])

    @pytest.mark.parametrize('payload_case', range(5), ids=["{**original, 'layout_version': 99}", "{**original, 'schema_version': True}", "{**original, 'metadata': '../elsewhere'}", "{'layout_version': 2}", '[]'])
    def test_unknown_or_unsafe_manifest_does_not_fall_back_to_flat_layout(self, payload_case):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ArtifactLayout.for_new_run(root).initialize()
            path = root / "result_manifest.json"
            original = json.loads(path.read_text())
            payload = tuple(({**original, 'layout_version': 99}, {**original, 'schema_version': True}, {**original, 'metadata': '../elsewhere'}, {'layout_version': 2}, []))[payload_case]
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError):
                ArtifactLayout.discover(root)

    def test_new_layout_rejects_occupied_directory_without_moving_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result_all.csv"
            result.write_bytes(b"existing data")
            with pytest.raises(ValueError, match="已有实验产物"):
                ArtifactLayout.for_new_run(root).initialize()
            assert (result.read_bytes()) == (b"existing data")
            assert not ((root / "result_manifest.json").exists())

    @pytest.mark.parametrize('name', ('input_scale_plan.json', '../outside.json', '/outside.json'))
    def test_symlinked_metadata_and_escaping_paths_are_rejected(self, name):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            layout = ArtifactLayout.for_new_run(root)
            layout.initialize()
            (root / "metadata").rmdir()
            (root / "metadata").symlink_to(Path(temporary), target_is_directory=True)
            with pytest.raises(ValueError):
                layout.path(name)

    def test_startup_error_is_written_directly_into_case_work_directory(self):
        from acprof.host.detect import TaskInfo
        from acprof.host.orchestrator import run_single_case
        from acprof.host.runtime_images import ImageInfo
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ArtifactLayout.for_new_run(root).initialize()
            with patch("acprof.host.orchestrator.start_container_session", side_effect=RuntimeError("startup failed")), \
                 redirect_stdout(io.StringIO()):
                output = run_single_case(TaskInfo("org/model", "fill-mask", "nlp", "transformers_pipeline", "transformers", "a" * 40, "manual"),
                                         1, 4, "off", ImageInfo(tag="sha256:" + "b" * 64), str(root), temporary,
                                         warmup=0, repeat=1, input_scales="64", profiling_mode="basic")
            assert (Path(output)) == (root / ".acprof/work/cases/1c_4g_off/result.csv")
            assert (list(root.glob("result_case_*"))) == ([])
            with Path(output).open() as stream:
                rows = list(csv.DictReader(stream))
            assert (len(rows)) == (1)
            assert (rows[0]["status"]) == ("error")

    def test_packet_merge_uses_root_metadata_and_nested_request_samples(self):
        from acprof.packet.merge_packet_latency import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = ArtifactLayout.for_new_run(root)
            layout.initialize()
            case = layout.case("org/model", 2, 8, "on")
            case.csv.parent.mkdir(parents=True)
            fields = ["status", "error", "latency_s", "throughput_samples_per_s", "batch_size"]
            with case.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerow({"status": "ok", "error": "", "latency_s": "nan", "batch_size": "nan"})
            (root / "static_meta.json").write_text(json.dumps({"schema_version": 7, "batch_size": 3}))
            case.sidecar("sniff_groups").write_text('{"sniff_group_id":"window"}\n')
            case.requests.write_text(json.dumps({"schema_version": 1, "sniff_group_id": "window", "latency_app_s": [0.3, 0.4]}) + "\n")
            case.latency.write_text(json.dumps({"schema_version": 2, "requests": {"window:0": {"latency_s": 0.25}}}))
            merged = Path(str(case.csv) + ".merged")
            main([str(case.csv), str(case.latency), str(merged)])
            assert (json.loads(case.requests.read_text())["latency_packet_s"]) == ([0.25, None])
            with merged.open() as stream:
                row = next(csv.DictReader(stream))
            assert (float(row["throughput_samples_per_s"])) == (12)
            assert (case_sidecar(case.csv, "requests")) == (case.requests)

    def test_ncu_report_reference_is_relative_to_the_experiment_root(self):
        from acprof.host.profilers.ncu import _ncu_report_reference
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ArtifactLayout.for_new_run(root).initialize()
            directory = root / "raw/compute_profiles"
            assert (_ncu_report_reference(str(directory), str(directory / "ncu_scale_8.csv"))) == ("raw/compute_profiles/ncu_scale_8.csv")

    def test_new_run_publishes_manifest_and_keeps_state_internal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            with patch("acprof.host.run_state.host_identity", return_value={}), \
                 patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", Path(temporary)):
                state = RunState(root, {}, resume=False, project_dir=temporary)
                try:
                    assert ((root / "result_manifest.json").is_file())
                    manifest = json.loads((root / "result_manifest.json").read_text())
                    assert (manifest["layout_version"]) == (2)
                    assert ((root / ".acprof/run_state.json").is_file())
                    assert ((root / ".acprof/result.lock").is_file())
                    assert not ((root / "run_state.json").exists())
                    assert not ((root / ".acprof-result.lock").exists())
                finally:
                    state.close()
