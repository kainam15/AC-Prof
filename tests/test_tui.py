import csv
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.tui.commands import (
    PendingLaunch,
    build_probe_command,
    build_profile_command,
    format_command,
    parse_slash_command,
)
from acprof.tui.diagnostics import _readable_rapl_paths, summarize_result_csv
from acprof.tui.log import SelectableLog
from acprof.tui.progress import RunProgressTracker
from acprof.tui.views import ConfirmActionScreen, StatusCheckbox

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_default_config_disables_compute_profiler():
    config = RunConfig(model="demo/model")
    command = build_run_command(
        config,
        project_dir=PROJECT_DIR,
        python_executable="python",
    )

    assert (config.compute_profile_tool) == ("none")
    assert (config.request_timeout_seconds) == (300.0)
    assert (command[command.index("--compute-profile-tool") + 1]) == ("none")
    assert (float(command[command.index("--request-timeout-seconds") + 1])) == (300.0)

def test_smoke_command_delegates_to_existing_run_entrypoint():
    config = RunConfig.smoke("google-bert/bert-base-uncased")
    command = build_run_command(
        config,
        project_dir=PROJECT_DIR,
        python_executable=PROJECT_DIR / ".venv/bin/python",
    )

    assert (command[0]) == (str(PROJECT_DIR / ".venv/bin/python"))
    assert (command[1:5]) == (["-u", "-m", "acprof", "run"])
    assert (command[command.index("--cpus") + 1]) == ("1")
    assert (command[command.index("--mems") + 1]) == ("4")
    assert ("--input-scales") not in (command)
    assert (command[command.index("--input-scale-policy") + 1]) == ("minimal")
    assert (command[command.index("--profiling-mode") + 1]) == ("basic")
    assert (command[command.index("--gpus") + 1]) == ("off")
    assert (command[command.index("--notify") + 1]) == ("none")
    assert (command[command.index("--compute-profile-tool") + 1]) == ("none")
    assert (command[command.index("--request-timeout-seconds") + 1]) == ("300.0")
    assert ("--allow-cgroup-v1") not in (command)
    assert ("acprof run") in (format_command(command))

def test_probe_command_uses_matrix_bounds_without_collection_options():
    config = RunConfig(
        model="demo/model",
        cpus="1,4",
        mems="2,8",
        gpus="off,on",
        input_scales="64,512",
        warmup=9,
        repeat=11,
        idle_seconds=99,
        request_timeout_seconds=999,
        skip_build=True,
    )

    command = build_probe_command(
        config,
        project_dir=PROJECT_DIR,
        python_executable="python",
    )

    assert (command[1:5]) == (["-u", "-m", "acprof", "probe"])
    assert (command[command.index("--cpus") + 1]) == ("1,4")
    assert (command[command.index("--mems") + 1]) == ("2,8")
    assert (command[command.index("--input-scales") + 1]) == ("64,512")
    assert ("--skip-build") in (command)
    assert ("--warmup") not in (command)
    assert ("--repeat") not in (command)
    assert ("--idle-seconds") not in (command)
    assert ("--compute-profile-tool") not in (command)
    assert ("--request-timeout-seconds") not in (command)
    assert ("--timeout-seconds") not in (command)
    assert ("acprof probe") in (format_command(command))

def test_invalid_matrix_is_rejected_before_launch():
    with pytest.raises(RunConfigError) as context:
        RunConfig(
            model="demo/model",
            cpus="1,1",
            mems="4,8",
            gpus="off",
        ).validate(project_dir=PROJECT_DIR)
    assert ("CPU 列表不能重复") in (str(context.value))

def test_invalid_request_timeout_is_rejected_before_launch():
    with pytest.raises(RunConfigError) as context:
        RunConfig(
            model="demo/model",
            request_timeout_seconds=0,
        ).validate(project_dir=PROJECT_DIR)
    assert ("单请求超时秒数必须是大于 0 的有限数字") in (str(context.value))

def test_optional_overrides_and_flags_are_preserved():
    config = RunConfig(
        model="demo/model",
        task="text-generation",
        task_family="nlp",
        backend="transformers_pipeline",
        cpus="2",
        mems="8",
        gpus="off",
        input_scales="64,128",
        request_timeout_seconds=123.5,
        skip_build=True,
        prune_startup_oom=False,
        idle_debug=True,
    )
    command = build_run_command(
        config,
        project_dir=PROJECT_DIR,
        python_executable="python",
    )
    assert ("--task") in (command)
    assert ("--task-family") in (command)
    assert ("--backend") in (command)
    assert ("--skip-build") in (command)
    assert ("--no-prune-startup-oom") in (command)
    assert ("--idle-debug") in (command)
    assert (command[command.index("--request-timeout-seconds") + 1]) == ("123.5")

def test_progress_tracker_marks_measurement_and_completion():
    tracker = RunProgressTracker()
    tracker.feed("Resource matrix: 1 CPUs x 1 MEMs x 1 GPUs = 1 cases")
    tracker.feed("# Case 1/1: CPU=2, MEM=8GB, GPU=off")
    measuring = tracker.feed("[case] Running workload...")
    assert (measuring.measurement_active)
    assert (measuring.stage) == ("正式测量")

    tracker.feed("[case][WARN] recoverable warning")
    tracker.feed("[case] Stopping container...")
    tracker.feed("[case] Done. Output: result.csv")
    tracker.feed("[merge] Final CSV: /tmp/result_all.csv (3 rows)")
    complete = tracker.feed("Profiling complete!")

    assert not (complete.measurement_active)
    assert (complete.completed_cases) == (1)
    assert (complete.warnings) == (1)
    assert (complete.final_csv) == ("/tmp/result_all.csv")
    assert (complete.stage) == ("已完成")

def test_progress_tracker_reports_largest_scale_probe_timings():
    tracker = RunProgressTracker()
    starting = tracker.feed(
        "[largest-probe] Starting minimum configuration: "
        "CPU=1, MEM=2GB, GPU=off, input_scale=512"
    )
    assert (starting.stage) == ("启动探测容器")
    assert ((starting.cpu, starting.mem, starting.gpu)) == (("1", "2", "off"))
    measuring = tracker.feed(
        "[largest-probe] Running one largest-scale request..."
    )
    assert (measuring.measurement_active)

    complete = tracker.feed(
        "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=2 "
        "gpu=off cold_start_s=12.500000 request_s=4.321000 "
        "ready_plus_request_s=16.821000"
    )
    complete = tracker.feed(
        "[largest-probe] Summary JSON: /tmp/probes/largest_scale_probe.json"
    )

    assert not (complete.measurement_active)
    assert (complete.stage) == ("探测完成")
    assert (complete.completed_cases) == (1)
    assert ("单次请求 4.321s") in (complete.detail)
    assert (complete.probe_summary) == ("/tmp/probes/largest_scale_probe.json")

def test_progress_tracker_shows_memory_scan_before_timing():
    tracker = RunProgressTracker()
    scan = tracker.feed(
        "[largest-probe] MEMORY_SCAN cpu=1 gpu=off "
        "candidates=2,4,8 input_scale=512"
    )
    assert (scan.stage) == ("准备内存探测")
    assert (scan.total_cases) == (3)

    tracker.feed(
        "[largest-probe] MEMORY_TRY current=1 total=3 "
        "cpu=1 mem=2 gpu=off input_scale=512"
    )
    first = tracker.feed(
        "[largest-probe] MEMORY_RESULT mem=2 status=startup_oom"
    )
    assert not (first.measurement_active)
    assert ("启动 OOM") in (first.detail)
    assert (first.completed_cases) == (1)

    tracker.feed(
        "[largest-probe] MEMORY_TRY current=2 total=3 "
        "cpu=1 mem=4 gpu=off input_scale=512"
    )
    measuring = tracker.feed(
        "[largest-probe] Running one largest-scale request..."
    )
    assert (measuring.measurement_active)
    found = tracker.feed(
        "[largest-probe] MEMORY_RESULT mem=4 status=ok"
    )
    assert (found.stage) == ("找到最低可用内存")
    assert ("4GB") in (found.detail)

    complete = tracker.feed(
        "[largest-probe] RESULT status=ok input_scale=512 cpu=1 mem=4 "
        "gpu=off cold_start_s=5 request_s=2 "
        "ready_plus_request_s=7"
    )
    assert (complete.stage) == ("探测完成")
    assert (complete.total_cases) == (2)
    assert (complete.completed_cases) == (2)
    assert ("最低可用内存 4GB") in (complete.detail)

def test_result_summary_and_profile_dry_run():
    with tempfile.TemporaryDirectory() as temporary_dir:
        result_dir = Path(temporary_dir)
        result_csv = result_dir / "result_all.csv"
        with result_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "status",
                    "warmup",
                    "cpu_cores",
                    "mem_cap_gb",
                    "gpu_mode",
                ),
            )
            writer.writeheader()
            writer.writerows(
                (
                    {
                        "status": "ok",
                        "warmup": "1",
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                    },
                    {
                        "status": "error",
                        "warmup": "0",
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                    },
                )
            )

        summary = summarize_result_csv(result_csv)
        assert (summary.rows) == (2)
        assert (summary.ok_rows) == (1)
        assert (summary.error_rows) == (1)
        assert (summary.warmup_rows) == (1)
        assert (summary.cases) == (1)

        profile = build_profile_command(
            result_dir,
            tools="torch,ncu",
            dry_run=True,
            project_dir=PROJECT_DIR,
            python_executable="python",
        )
        assert (profile[-1]) == ("--dry-run")

def test_summarize_result_csv_with_latencies():
    with tempfile.TemporaryDirectory() as temporary_dir:
        result_csv = Path(temporary_dir) / "result_all.csv"
        with result_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "status",
                    "warmup",
                    "cpu_cores",
                    "mem_cap_gb",
                    "gpu_mode",
                    "latency_app_s", "input_scale", "repeat_idx",
                ),
            )
            writer.writeheader()
            writer.writerows(
                (
                    {
                        "status": "ok", "input_scale": "64", "repeat_idx": "0",
                        "warmup": "1",
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                        "latency_app_s": "0.100",
                    },
                    {
                        "status": "ok", "input_scale": "64", "repeat_idx": "0",
                        "warmup": "0",
                        "cpu_cores": "1",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                        "latency_app_s": "0.050",
                    },
                    {
                        "status": "ok", "input_scale": "64", "repeat_idx": "0",
                        "warmup": "0",
                        "cpu_cores": "2",
                        "mem_cap_gb": "4",
                        "gpu_mode": "off",
                        "latency_app_s": "0.030",
                    },
                )
            )
        summary = summarize_result_csv(result_csv)
        assert (summary.rows) == (3)
        assert (summary.ok_rows) == (3)
        assert (summary.warmup_rows) == (1)
        assert (summary.cases) == (2)
        assert ({group["cpu_cores"]: group["mean"] for group in summary.groups}) == ({1: 0.050, 2: 0.030})
        assert (all(group["n_windows"] == 1 for group in summary.groups))
        assert (all(group["ci_low"] is None for group in summary.groups))

def test_rapl_check_does_not_follow_cyclic_sysfs_links():
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        package = root / "intel-rapl:0"
        package.mkdir()
        (package / "name").write_text("package-0\n", encoding="utf-8")
        (package / "energy_uj").write_text("123\n", encoding="utf-8")
        (package / "device").symlink_to(package, target_is_directory=True)
        subdomain = root / "intel-rapl:0:0"
        subdomain.mkdir()
        (subdomain / "name").write_text("core\n", encoding="utf-8")
        (subdomain / "energy_uj").write_text("456\n", encoding="utf-8")

        paths = _readable_rapl_paths(root)

        assert (paths) == ([str(package / "energy_uj")])

def test_slash_command_parser_does_not_execute_shell_text():
    command, args = parse_slash_command('/plot "results/a b/result_all.csv"')
    assert (command) == ("plot")
    assert (args) == (["results/a b/result_all.csv"])
    with pytest.raises(RunConfigError):
        parse_slash_command("plot result.csv")


class TestTuiApp:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        xdg_patch = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(temporary)})
        xdg_patch.start()
        self._request.addfinalizer(partial(xdg_patch.stop))

    async def test_app_mounts_and_requires_confirmation_before_run(self):
        temporary = tempfile.TemporaryDirectory()
        self._request.addfinalizer(partial(temporary.cleanup))
        app = AcprofTui(
            RunConfig.smoke("google-bert/bert-base-uncased"),
            settings_path=Path(temporary.name) / "tui.json",
        )
        async with app.run_test(size=(150, 52)) as pilot:
            await pilot.pause()
            assert not (app.ENABLE_COMMAND_PALETTE)
            assert not (app.use_command_palette)
            assert not (app.query_one("HeaderIcon").display)
            assert not (app.query("#allow-cgroup-v1"))
            command_bar = app.query_one("#slash-command-bar")
            assert (command_bar.styles.padding.top) == (1)
            assert (command_bar.styles.padding.right) == (2)
            assert (command_bar.styles.padding.bottom) == (1)
            assert (command_bar.styles.padding.left) == (2)
            slash_command = app.query_one("#slash-command", Input)
            assert (slash_command.parent.id) == ("slash-command-bar")
            assert (slash_command.placeholder.startswith("快捷命令："))
            assert ("/help") in (slash_command.placeholder)
            assert (command_bar.region.bottom) == (app.size.height)
            assert (slash_command.region.y - command_bar.region.y) == (1)
            assert (command_bar.region.bottom - slash_command.region.bottom) == (1)
            preview = str(app.query_one("#command-preview", Static).render())
            assert (preview.startswith("acprof run ")), preview
            assert ("--warmup 0") in (preview)
            assert ("--idle-seconds 0.0") in (preview)
            assert ("--idle-cooldown-seconds 0.0") in (preview)
            assert ("--request-timeout-seconds 300.0") in (preview)
            assert (app.query_one("#request-timeout-seconds", Input).value) == ("300")
            with pytest.raises(NoMatches):
                app.query_one("#preview-command")
            with pytest.raises(NoMatches):
                app.query_one("#back-config")
            with pytest.raises(NoMatches):
                app.query_one("#advanced-settings")
            assert (app.query_one("#settings-tab")) is not None
            assert (app.query_one("#command-details").collapsed)
            assert (str(app.query_one("#probe-largest").label)) == ("探测最大输入")

            prune = app.query_one("#prune-startup-oom", StatusCheckbox)
            reuse = app.query_one("#skip-build", StatusCheckbox)
            assert (StatusCheckbox.BUTTON_INNER) == ("✓")
            assert (prune.value)
            assert (prune.has_class("-on"))
            assert not (reuse.value)
            assert not (reuse.has_class("-on"))

            app.query_one("#cpus", Input).value = "1,3"
            await pilot.pause(0.1)
            auto_preview = str(app.query_one("#command-preview", Static).render())
            assert ("--cpus 1,3") in (auto_preview)
            assert (app.query_one("#run-preset").value) == ("smoke")

            app.query_one("#run-preset", Select).value = "main"
            await pilot.pause(0.1)
            preset_preview = str(app.query_one("#command-preview", Static).render())
            assert (app.query_one("#cpus", Input).value) == ("1,2,4,8")
            assert ("--cpus 1,2,4,8") in (preset_preview)
            assert ("--compute-profile-tool none") in (preset_preview)

            app.action_request_probe()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert (app._pending_launch) is not None
            assert (app._pending_launch.kind) == ("probe")
            assert ("内存候选：2GB,4GB,8GB,16GB（从小到大）") in (app.screen.message)
            assert ("OOM 时自动尝试下一档") in (app.screen.message)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)

            app.action_quick_check()
            app.action_request_run()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            assert not (app._is_busy())
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmActionScreen)

    async def test_subprocess_progress_is_event_driven(self):
        script = "\n".join(
            (
                "print('Resource matrix: x = 1 cases', flush=True)",
                "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                "print('[case] Running workload...', flush=True)",
                "print('routine measurement detail', flush=True)",
                "print('[case] Stopping container...', flush=True)",
                "print('[case] Done. Output: result.csv', flush=True)",
                "print('Profiling complete!', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    RunConfig.smoke("demo/model"),
                )
            )
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            assert not (app._is_busy())
            assert (app._latest_snapshot.stage) == ("失败")
            assert (app._latest_snapshot.completed_cases) == (0)

    async def test_measurement_failure_keeps_multiline_diagnostic(self):
        lines = [
            '[case] Running workload...',
            'routine sample output',
            '[mips][ERROR] Error:',
            'No supported events found.',
            'Access to performance monitoring is limited.',
            'Recovery: configure CAP_PERFMON',
            '[case] Stopping container...',
        ]
        script = (
            "from acprof.progress_events import emit_event\n"
            "emit_event('case_started', 'fixture')\n"
            "emit_event('measurement_started', 'fixture')\n"
            + '\n'.join(f'print({line!r}, flush=True)' for line in lines)
            + "\nemit_event('measurement_stopped', 'fixture')\n"
            "emit_event('case_finished', 'fixture', status='error')\nraise SystemExit(1)"
        )
        app = AcprofTui(RunConfig.smoke('demo/model'))
        async with app.run_test(size=(120, 30)) as pilot:
            app._launch(PendingLaunch((sys.executable, '-u', '-c', script), 'run'))
            for _ in range(60):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            assert not (app._is_busy())
            log = app.query_one('#run-log', SelectableLog).text.splitlines()
            for line in lines[2:6]:
                assert (line) in (log)
            assert ('routine sample output') not in (log)

    async def test_probe_subprocess_keeps_timing_result_in_monitor(self):
        script = "\n".join(
            (
                "print('[largest-probe] MEMORY_SCAN cpu=1 gpu=off "
                "candidates=2,4 input_scale=512', flush=True)",
                "print('[largest-probe] MEMORY_TRY current=1 total=2 "
                "cpu=1 mem=2 gpu=off input_scale=512', flush=True)",
                "print('[largest-probe] MEMORY_RESULT mem=2 "
                "status=startup_oom', flush=True)",
                "print('[largest-probe] MEMORY_TRY current=2 total=2 "
                "cpu=1 mem=4 gpu=off input_scale=512', flush=True)",
                "print('[largest-probe] Running one largest-scale request...', flush=True)",
                "print('[largest-probe] MEMORY_RESULT mem=4 status=ok', flush=True)",
                "print('[largest-probe] RESULT status=ok input_scale=512 "
                "cpu=1 mem=4 gpu=off cold_start_s=12.5 request_s=4.321 "
                "ready_plus_request_s=16.821', flush=True)",
                "print('[largest-probe] Summary JSON: /tmp/probe.json', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "probe",
                    RunConfig.smoke("demo/model"),
                )
            )
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break

            assert not (app._is_busy())
            assert (app._latest_snapshot.stage) == ("探测完成")
            assert ("最低可用内存 4GB") in (app._latest_snapshot.detail)
            assert ("单次请求 4.321s") in (app._latest_snapshot.detail)
            assert (app._latest_snapshot.probe_summary) == ("/tmp/probe.json")

    async def test_elapsed_timer_skips_during_measurement(self):
        """Elapsed ticker must not cause widget updates during measurement."""
        script = "\n".join(
            (
                "import time",
                "print('Resource matrix: x = 1 cases', flush=True)",
                "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                "print('[case] Running workload...', flush=True)",
                "time.sleep(0.2)",
                "print('[case] Stopping container...', flush=True)",
                "print('[case] Done. Output: result.csv', flush=True)",
                "print('Profiling complete!', flush=True)",
            )
        )
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    RunConfig.smoke("demo/model"),
                )
            )
            # The elapsed timer should be active during the run.
            assert (app._elapsed_timer) is not None
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            # After completion the timer is stopped.
            assert (app._elapsed_timer) is None
            assert not (app._is_busy())

    async def test_run_updates_status_and_log_without_monitor_charts(self):
        """Removing monitor charts must preserve launch and completion updates."""
        config = RunConfig.smoke("demo/model")
        app = AcprofTui(config)
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            assert (len(app.query("#case-progress, #matrix-board, #matrix-table"))) == (0)
            script = "\n".join(
                (
                    "print('Resource matrix: x = 1 cases', flush=True)",
                    "print('# Case 1/1: CPU=1, MEM=4GB, GPU=off', flush=True)",
                    "print('[case] Running workload...', flush=True)",
                    "print('[case] Stopping container...', flush=True)",
                    "print('[case] Done. Output: result.csv', flush=True)",
                    "print('Profiling complete!', flush=True)",
                )
            )
            app._launch(
                PendingLaunch(
                    (sys.executable, "-u", "-c", script),
                    "run",
                    config,
                )
            )
            await pilot.pause(0.1)
            for _ in range(40):
                await pilot.pause(0.05)
                if not app._is_busy():
                    break
            assert not (app._is_busy())
            assert (app.query_one("#status-stage", Static).content) == ("失败")
            assert (app.query_one("#status-case", Static).content) == ("当前 1 · 已完成 0/1")
            assert ("Profiling complete!") in (app.query_one("#run-log", SelectableLog).text)
            assert (len(app.query("#case-progress, #matrix-board, #matrix-table"))) == (0)

    async def test_stage_gets_color_class(self):
        """The status-stage widget should receive CSS classes for visual state."""
        from acprof.tui.progress import ProgressSnapshot
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            stage_widget = app.query_one("#status-stage", Static)
            app._render_snapshot(ProgressSnapshot(stage="已完成"))
            await pilot.pause()
            assert (stage_widget.has_class("stage-success"))
            app._render_snapshot(ProgressSnapshot(stage="失败"))
            await pilot.pause()
            assert (stage_widget.has_class("stage-error"))
            assert not (stage_widget.has_class("stage-success"))
            app._render_snapshot(ProgressSnapshot(stage="正式测量"))
            await pilot.pause()
            assert (stage_widget.has_class("stage-measuring"))

    @pytest.mark.parametrize('command', ('matrix', 'board'))
    async def test_removed_matrix_commands_are_rejected_and_absent_from_help(self, command):
        app = AcprofTui(RunConfig.smoke("demo/model"))
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#slash-command", Input)
            with patch.object(app, "notify") as notify:
                input_widget.focus()
                input_widget.value = f"/{command}"
                await pilot.press("enter")
                await pilot.pause()
                notify.assert_called_once()
                assert (app.tr(notify.call_args.args[0])) == (f"未知快捷命令：/{command}")
                assert (notify.call_args.kwargs["severity"]) == ("error")
            input_widget.focus()
            input_widget.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            help_text = app.query_one("#run-log", SelectableLog).text
            assert ("/status") in (help_text)
            assert ("/matrix") not in (help_text)
            assert ("/board") not in (help_text)
