import csv
import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from run_recovery_fixtures import RunRecoveryFixture


class TestRunRecovery(RunRecoveryFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self.build(request, tmp_path)
    def test_cpu_set_reaches_cases_and_changed_set_refuses_resume(self):
        seen = []
        def capture(**kwargs):
            seen.append(kwargs["cpuset_cpus"])
            return self.write_case(**kwargs)
        with patch("acprof.cli.run.os.sched_getaffinity", return_value={0, 1, 2}):
            self.invoke("--cpuset-cpus", "2,0-1", case=capture)
            assert (seen) == (["0-2", "0-2"])
            self.invoke("--cpuset-cpus", "0-2", "--resume")
            with pytest.raises(SystemExit):
                self.invoke("--cpuset-cpus", "0-1", "--resume")

    def test_resume_archives_promoted_samples_and_idle_without_name_collisions(self):
        def interrupted(**kwargs):
            path = self.write_case(**kwargs)
            if kwargs["cpu"] == 2:
                raw = self.directory / "raw/requests/2c_4g_off.jsonl"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_text('{"previous_attempt":true}\n')
                idle = self.directory / "debug/idle/2c_4g_off.jsonl"
                idle.parent.mkdir(parents=True, exist_ok=True)
                idle.write_text('{"idle_diagnostic":true}\n')
                raise KeyboardInterrupt()
            return path
        with pytest.raises(KeyboardInterrupt):
            self.invoke(case=interrupted)
        self.calls.clear()
        self.invoke("--resume")
        assert (self.calls) == ([2])
        backups = list((self.directory / ".acprof/recovery/interrupted_cases").rglob("2c_4g_off.jsonl"))
        assert (len(backups)) == (2)
        assert ({p.relative_to(p.parents[2]).as_posix(): json.loads(p.read_text()) for p in backups}) == ({"raw/requests/2c_4g_off.jsonl": {"previous_attempt": True},
                          "debug/idle/2c_4g_off.jsonl": {"idle_diagnostic": True}})

    def test_completed_v2_result_keeps_only_primary_files_and_retains_request_samples(self):
        def with_samples(**kwargs):
            path = Path(self.write_case(**kwargs))
            path.with_name("requests.jsonl").write_text('{"schema_version":1,"latency_app_s":[0.1]}\n')
            path.with_name("sniff_groups.jsonl").write_text('{"sniff_group_id":"fixture"}\n')
            return str(path)
        self.invoke(case=with_samples)
        assert ({p.name for p in self.directory.iterdir() if p.is_file()}) == (
            {"result_all.csv", "static_meta.json", "capability_report.json",
             "result_manifest.json", "quality_checks.json", "result_layers.json",
             "summary.csv", "performance.csv", "resources.csv", "energy.csv", "network.csv"})
        from acprof.result_layers import read_result_layers
        fields, rows = read_result_layers(self.directory)
        assert len(rows) == 2
        assert {"latency_s", "gpu_energy_total_j", "cpu_cores"} <= set(fields)
        for cpu in (1, 2):
            sample = self.directory / f"raw/requests/{cpu}c_4g_off.jsonl"
            assert (json.loads(sample.read_text())["latency_app_s"]) == ([0.1])
        assert (list(self.directory.glob(".acprof/work/cases/*"))) == ([])
        from acprof.analysis.audit import audit_result
        report = audit_result(self.directory)
        assert (report["valid"]), report["issues"]
        assert (report["completion"]) == ("complete")
        assert (report["coverage"]["missing"]) == (0)

    def test_seeded_resume_reuses_frozen_order_without_plan_generation(self):
        seen = []
        def interrupted(**kwargs):
            seen.append(kwargs['cpu'])
            if len(seen) == 2:
                raise KeyboardInterrupt()
            return self.write_case(**kwargs)
        with pytest.raises(KeyboardInterrupt):
            self.invoke('--matrix-order', 'seeded', '--matrix-seed', '37', case=interrupted)
        path = self.directory / 'metadata/matrix_plan.json'
        original = path.read_bytes()
        plan = json.loads(original)
        assert (seen) == ([c['cpu_cores'] for c in plan['cases']])
        with patch('acprof.host.matrix_plan.build_matrix_plan', side_effect=AssertionError('reshuffle')):
            self.invoke('--matrix-order', 'seeded', '--matrix-seed', '37', '--resume')
        assert (path.read_bytes()) == (original)
        assert (self.calls) == ([c['cpu_cores'] for c in plan['cases']])

    def test_resume_rejects_tampered_frozen_plan(self):
        self.invoke()
        path = self.directory / 'metadata/matrix_plan.json'
        path.write_text(path.read_text() + ' ')
        with pytest.raises(SystemExit):
            self.invoke('--resume')

    def test_existing_results_are_not_overwritten_without_resume(self):
        self.directory.mkdir()
        metadata = self.directory / "static_meta.json"
        original = b'{"cgroup_version":"v2","preserve":"original"}'
        metadata.write_bytes(original)
        with pytest.raises((RuntimeError, SystemExit)):
            self.invoke()
        assert (metadata.read_bytes()) == (original)
        assert (self.calls) == ([])

    def test_full_rejects_required_measurement_missing_after_preflight(self):
        self.assert_missing_required_measurements_rejected("full")

    def test_basic_rejects_required_measurement_missing_after_preflight(self):
        self.assert_missing_required_measurements_rejected("basic")

    def test_resume_keeps_completed_case_and_restarts_interrupted_case(self):
        from acprof.analysis.audit import audit_result
        from acprof.failures import Failure
        failure = Failure("predict", "inference_failed", "fixture failure").to_dict()
        def interrupted(**kwargs):
            path = self.write_case(**kwargs)
            if kwargs["cpu"] == 2:
                Path(path).with_name("requests.jsonl").write_text('{"partial":true}\n')
                Path(path).with_name("runtime_failures.json").write_text(json.dumps({"failures": [failure]}))
                Path(path).with_name("cleanup_error.json").write_text('{"status":"incomplete","final_state":"unknown"}')
                raise KeyboardInterrupt()
            return path
        with pytest.raises(KeyboardInterrupt):
            self.invoke(case=interrupted)
        assert (audit_result(self.directory)["failures"]) == ([failure])
        original_meta = (self.directory / "static_meta.json").read_bytes()
        self.calls.clear()
        self.invoke("--resume")
        assert (self.calls) == ([2])
        assert ((self.directory / "static_meta.json").read_bytes()) == (original_meta)
        with (self.directory / "result_all.csv").open() as stream:
            assert ([row["cpu_cores"] for row in csv.DictReader(stream)]) == (["1", "2"])
        assert (list((self.directory / ".acprof/recovery/interrupted_cases").rglob("*.csv")))
        archived_samples = list((self.directory / ".acprof/recovery/interrupted_cases").rglob("requests.jsonl"))
        assert (len(archived_samples)) == (1)
        assert (archived_samples[0].read_text()) == ('{"partial":true}\n')
        assert (audit_result(self.directory)["failures"]) == ([])
        archived_failures = list((self.directory / ".acprof/recovery/interrupted_cases").rglob("runtime_failures.json"))
        assert (json.loads(archived_failures[0].read_text())["failures"]) == ([failure])
        archived_cleanup = list((self.directory / ".acprof/recovery/interrupted_cases").rglob("cleanup_error.json"))
        assert (len(archived_cleanup)) == (1)
        assert (json.loads(archived_cleanup[0].read_text())["final_state"]) == ("unknown")
        assert (list((self.directory / ".acprof/work/cases").rglob("cleanup_error.json"))) == ([])

    def test_resume_rejects_changed_measurement_parameters(self):
        self.interrupt_after_first()
        before = (self.directory / ".acprof/run_state.json").read_bytes()
        with pytest.raises(SystemExit):
            self.invoke("--resume", "--repeat", "2")
        assert (self.calls) == ([])
        assert ((self.directory / ".acprof/run_state.json").read_bytes()) == (before)

    def test_resume_rejects_changed_input_plan_without_overwriting_it(self):
        self.interrupt_after_first()
        path = self.directory / "metadata/input_scale_plan.json"
        path.write_text('{"changed":true}')
        with pytest.raises(SystemExit):
            self.invoke("--resume")
        assert (path.read_text()) == ('{"changed":true}')
        assert (self.calls) == ([])

    def test_completed_resume_does_not_repeat_measurements_or_rewrite_result(self):
        self.invoke()
        result = self.directory / "result_all.csv"
        before = result.read_bytes(), result.stat().st_mtime_ns
        self.calls.clear()
        self.invoke("--resume")
        assert (self.calls) == ([])
        assert ((result.read_bytes(), result.stat().st_mtime_ns)) == (before)

    @pytest.mark.skipif(os.name != "posix", reason="directory fsync is a POSIX durability contract")
    def test_preparation_backup_syncs_directory_entries(self):
        import stat

        from acprof.host.run_state import RunState

        state = RunState(
            self.directory,
            {},
            resume=False,
            project_dir=str(Path(__file__).resolve().parents[1]),
        )
        self._request.addfinalizer(state.close)
        evidence = state.layout.path("static_meta.json")
        evidence.write_text('{"fixture":true}\n', encoding="utf-8")
        synced_types = []
        synced_directories = []
        real_fsync = os.fsync

        def track_fsync(fd):
            file_type = stat.S_IFMT(os.fstat(fd).st_mode)
            synced_types.append(file_type)
            if file_type == stat.S_IFDIR:
                synced_directories.append(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve())
            real_fsync(fd)

        with patch("os.fsync", side_effect=track_fsync):
            backup = state._archive_preparation()

        backup_path = (self.directory / backup).resolve()
        assert (backup_path / "static_meta.json").is_file()
        assert stat.S_IFREG in synced_types
        assert any(path == backup_path or path.is_relative_to(backup_path) for path in synced_directories)

    @pytest.mark.skipif(os.name != "posix", reason="directory fsync is a POSIX durability contract")
    def test_interrupted_case_backup_syncs_directory_entries(self):
        import stat

        from acprof.host.run_state import RunState

        state = RunState(
            self.directory,
            {},
            resume=False,
            project_dir=str(Path(__file__).resolve().parents[1]),
        )
        self._request.addfinalizer(state.close)
        case = state.layout.case("", 1, 4, "off")
        case.csv.parent.mkdir(parents=True, exist_ok=True)
        case.csv.write_text("partial\n", encoding="utf-8")
        synced_types = []
        synced_directories = []
        real_fsync = os.fsync

        def track_fsync(fd):
            file_type = stat.S_IFMT(os.fstat(fd).st_mode)
            synced_types.append(file_type)
            if file_type == stat.S_IFDIR:
                synced_directories.append(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve())
            real_fsync(fd)

        with patch("os.fsync", side_effect=track_fsync):
            assert state.prepare_case("unused.csv", 1, 4, "off") is None

        backups = list(state.layout.path("interrupted_cases").rglob("result.csv"))
        backup_root = state.layout.path("interrupted_cases").resolve()
        assert len(backups) == 1
        assert stat.S_IFREG in synced_types
        assert any(path == backup_root or path.is_relative_to(backup_root) for path in synced_directories)

    def test_changed_completed_case_is_rejected(self):
        self.interrupt_after_first()
        source = self.directory / ".acprof/work/cases/1c_4g_off/result.csv"
        source.write_bytes(source.read_bytes() + b"corrupt,row\n")
        before = source.read_bytes()
        with pytest.raises(SystemExit):
            self.invoke("--resume")
        assert (source.read_bytes()) == (before)
        assert (self.calls) == ([])

    def test_directory_lock_rejects_second_writer_and_releases_on_exit(self):
        from acprof.host.run_state import ResultDirectoryLock, RunStateError
        with ResultDirectoryLock(self.directory):
            with pytest.raises(RunStateError):
                with ResultDirectoryLock(self.directory):
                    pytest.fail("two writers acquired the same directory")
        with ResultDirectoryLock(self.directory):
            pass

    @pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
    def test_directory_lock_rejects_link_swapped_after_path_validation(self, link_kind):
        from acprof.host.run_state import ResultDirectoryLock, RunStateError

        lock = ResultDirectoryLock(self.directory)
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.directory.parent / f"external-{link_kind}.lock"
        target.write_text("")
        if link_kind == "symlink":
            lock.path.symlink_to(target.resolve())
        else:
            os.link(target, lock.path)

        with pytest.raises(RunStateError):
            with lock:
                pytest.fail("result-directory lock followed a swapped link")

    def test_different_output_directories_cannot_measure_concurrently(self):
        from acprof.host.run_state import RunState, RunStateError
        first = RunState(self.directory, {}, resume=False, project_dir=str(Path(__file__).resolve().parents[1]))
        self._request.addfinalizer(partial(first.close))
        with pytest.raises(RunStateError):
            second = RunState(self.directory.parent / "other", {}, resume=False,
                              project_dir=str(Path(__file__).resolve().parents[1]))
            second.close()
