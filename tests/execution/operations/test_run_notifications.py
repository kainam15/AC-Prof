import csv
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.application import notifications as notices, run
from acprof.cli import run as run_cli
from acprof.experiments.configuration import RunConfig
from acprof.host import cases, input_plan, orchestrator
from acprof.host.detect import TaskInfo
from acprof.host.images import runtime as runtime_images
from acprof.host.profiling.progress import ProfilerProgress
from acprof.notifications import NotificationConfigError, NotificationEvent


class TestRunNotificationLifecycle:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        selection = patch('acprof.host.gpu_device.resolve_gpu_device', return_value={
            'uuid': 'GPU-fixture', 'index': 1, 'name': 'Fixture', 'memory_total_bytes': 8 * 1024 ** 3,
            'pci_bus_id': '00000000:02:00.0',
        })
        selection.start()
        self._request.addfinalizer(partial(selection.stop))
        self.notices = notices.RunNotifications()
        with patch("acprof.application.run.RunNotifications", return_value=self.notices):
            yield

    @staticmethod
    def _context(notifier, *, total_cases=2):
        return notices.RunNotificationContext(
            notifier=notifier,
            model_id="org/model",
            output_dir="/tmp/results/org--model",
            started_at=time.perf_counter() - 1.0,
            run_command="acprof run --model org/model",
            total_cases=total_cases,
        )

    @staticmethod
    def _success_event() -> NotificationEvent:
        return NotificationEvent(
            status="success",
            model_id="org/model",
            output_dir="/tmp/results/org--model",
            elapsed_seconds=1.0,
            total_cases=2,
            completed_cases=2,
            result_rows=4,
            error_rows=0,
            final_csv="/tmp/results/org--model/result_all.csv",
            host="test-host",
        )

    def test_default_mode_auto_enables_configured_wecom(self) -> None:
        notifier = Mock()
        assert ("auto") == ("auto")

        with patch(
            "acprof.application.notifications.WeComWebhookNotifier.from_env",
            return_value=notifier,
        ) as from_env:
            self.notices.activate(
                provider="auto",
                model_id="org/model",
                output_dir="/tmp/results/org--model",
                started_at=time.perf_counter(),
                run_command="acprof run --model org/model",
            )

        from_env.assert_called_once_with()
        assert (self.notices.context) is not None
        assert (self.notices.context.notifier) is (notifier)

    def test_default_mode_stays_silent_without_wecom_config(self) -> None:
        stderr = io.StringIO()
        with patch(
            "acprof.application.notifications.WeComWebhookNotifier.from_env",
            side_effect=NotificationConfigError("missing"),
        ), redirect_stderr(stderr):
            self.notices.activate(
                provider="auto",
                model_id="org/model",
                output_dir="/tmp/results/org--model",
                started_at=time.perf_counter(),
                run_command="acprof run --model org/model",
            )

        assert (self.notices.context) is None
        assert (stderr.getvalue()) == ("")

    def test_start_notification_contains_current_command(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        self.notices.context = self._context(CapturingNotifier())
        self.notices.notify_started()

        assert (len(captured)) == (1)
        event = captured[0]
        assert (event.status) == ("started")
        assert (event.run_command) == ("acprof run --model org/model")
        assert ("环境预检") in (event.detail)
        assert (event.elapsed_seconds) >= (0.0)
        assert (self.notices.context.event) is None

    @pytest.mark.parametrize('allowed', [False, True])
    def test_policy_and_runtime_preflight_precede_start_notification(self, allowed) -> None:
        argv = [
            "acprof run",
            "--model",
            "org/model with space",
            "--cpus",
            "1",
        ]
        order = []

        def activate(**kwargs):
            order.append(("activate", kwargs["run_command"]))

        def resolve(**kwargs):
            order.append(("resolution", kwargs["model_id"]))
            return TaskInfo(
                model_id=kwargs["model_id"], pipeline_tag="fill-mask", task_family="nlp",
                runtime_backend="transformers_pipeline", library_name="transformers",
                model_revision="1" * 40, detection_method="unit",
            )

        def policy(*, profiling_mode, compute_tool, execution_tool, dram_energy):
            assert ((profiling_mode, compute_tool, execution_tool, dram_energy)) == (("full", "none", "none", "auto"))
            order.append(("policy", None))
            if not allowed:
                raise RuntimeError("stop after ordering check")

        def preflight():
            order.append(("preflight", None))
            raise RuntimeError("stop after ordering check")

        order.clear()
        with patch.object(sys, "argv", argv), patch(
            "acprof.application.run.bootstrap_project_env"
        ), patch(
            "acprof.application.notifications.RunNotifications.activate",
            side_effect=activate,
        ), patch(
            "acprof.application.run.start_terminal_log",
            side_effect=lambda *_args: order.append(("tmux", None)),
        ), patch(
            "acprof.application.notifications.RunNotifications.notify_started",
            side_effect=lambda: order.append(("started", None)),
        ), patch(
            "acprof.host.detect.detect_task",
            side_effect=resolve,
        ), patch(
            "acprof.application.run.require_collection_host",
            side_effect=policy,
        ), patch(
            "acprof.host.preflight.require_native_docker",
            side_effect=preflight,
        ):
            with pytest.raises(RuntimeError, match="ordering check"):
                run_cli.main()

            assert (order) == ([("policy", None)] + ([
                ("preflight", None),
            ] if allowed else []))

    def test_completion_marks_error_rows_as_partial(self) -> None:
        notifier = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            result_csv = os.path.join(tmp, "result_all.csv")
            with open(result_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("status", "error"))
                writer.writeheader()
                writer.writerow({"status": "ok", "error": ""})
                writer.writerow({"status": "error", "error": "timeout"})

            self.notices.context = self._context(notifier)
            self.notices.record_completion(
                final_csv=result_csv,
                completed_cases=2,
            )

        event = self.notices.context.event
        assert (event) is not None
        assert (event.status) == ("partial")
        assert (event.result_rows) == (2)
        assert (event.error_rows) == (1)
        notifier.send.assert_not_called()

    def test_case_progress_reports_current_case_after_csv_is_complete(self) -> None:
        captured = []

        class CapturingNotifier:
            def send(self, event):
                captured.append(event)

        with tempfile.TemporaryDirectory() as tmp:
            result_csv = os.path.join(tmp, "result_case.csv")
            with open(result_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("status", "error"))
                writer.writeheader()
                writer.writerow({"status": "ok", "error": ""})

            self.notices.context = self._context(CapturingNotifier())
            self.notices.notify_case(
                cases.MatrixProgress(
                    completed_cases=1,
                    total_cases=2,
                    cpu=2,
                    mem=4,
                    gpu="off",
                    result_csv=result_csv,
                )
            )

        assert (len(captured)) == (1)
        event = captured[0]
        assert (event.status) == ("progress")
        assert (event.completed_cases) == (1)
        assert (event.total_cases) == (2)
        assert (event.result_rows) == (1)
        assert (event.error_rows) == (0)
        assert ("CPU=2, MEM=4GB, GPU=off") in (event.detail)

    @pytest.mark.parametrize('status_case', range(4))
    def test_profiler_completion_preserves_status_counts_and_final_event(self, status_case) -> None:
        cases = (
            ("success", 3, 0, ""),
            ("partial", 3, 1, "one scale failed"),
            ("failed", 3, 3, "probe failed"),
            ("no_results", 0, 0, ""),
        )
        (status, total_samples, error_samples, detail) = tuple(cases)[status_case]
        notifier = Mock()
        context = self._context(notifier)
        final_event = self._success_event()
        context.event = final_event
        self.notices.context = context

        self.notices.notify_profiler(
            ProfilerProgress(
                profiler="CPU Torch",
                status=status,
                elapsed_seconds=0.25,
                total_samples=total_samples,
                error_samples=error_samples,
                detail=detail,
            )
        )

        notifier.send.assert_called_once()
        event = notifier.send.call_args.args[0]
        assert (event.status) == (f"profiler_{status}")
        assert (event.profiler) == ("CPU Torch")
        assert (event.profile_elapsed_seconds) == (0.25)
        assert (event.profile_samples) == (total_samples)
        assert (event.profile_error_samples) == (error_samples)
        assert (event.detail) == (detail or None)
        assert (event.model_id) == (context.model_id)
        assert (event.output_dir) == (context.output_dir)
        assert (event.elapsed_seconds) >= (1.0)
        assert (context.event) is (final_event)

    def test_profiler_completion_is_silent_without_active_notifier(self) -> None:
        self.notices.context = None
        with patch("acprof.application.notifications._send_notification_event") as send:
            self.notices.notify_profiler(
                ProfilerProgress("NCU", "success", 2.0, 1, 0)
            )

        send.assert_not_called()

    def test_profiler_notification_failure_preserves_final_event_and_hides_secret(self) -> None:
        unsafe_key = "synthetic-profiler-notification-secret"
        notifier = Mock()
        notifier.send.side_effect = RuntimeError(
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/"
            f"send?key={unsafe_key}"
        )
        context = self._context(notifier)
        final_event = self._success_event()
        context.event = final_event
        self.notices.context = context
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            self.notices.notify_profiler(
                ProfilerProgress("Nsys", "success", 2.0, 1, 0)
            )

        notifier.send.assert_called_once()
        assert (context.event) is (final_event)
        assert ("RuntimeError") in (stderr.getvalue())
        assert (unsafe_key) not in (stderr.getvalue())

    def test_main_wires_profiler_notifications_before_matrix_with_resolved_model(self) -> None:
        self._assert_main_profiler_notifications(provider="auto")

    def test_main_disables_both_profiler_callbacks_when_notify_is_none(self) -> None:
        self._assert_main_profiler_notifications(provider="none")

    def _assert_main_profiler_notifications(self, *, provider: str) -> None:
        task_info = TaskInfo(
            model_id="org/resolved-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        order = []
        events = []
        notifier = Mock()

        def send(event):
            order.append(event.profiler or event.status)
            events.append(event)

        notifier.send.side_effect = send

        def collect(profilers, **kwargs):
            callback = kwargs.get("progress_callback")
            if callback is not None:
                for profiler in profilers:
                    callback(ProfilerProgress(profiler, "success", 2.0, 1, 0))
            return os.path.join(kwargs["output_dir"], "mock_profile_plan.json")

        def run_matrix(_request):
            order.append("matrix")
            return []

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof run", "--model", "org/requested-model",
                "--compute-profile-tool", "both", "--execution-profile-tool", "both",
                "--cpus", "1", "--mems", "4", "--gpus", "off,on",
                "--notify", provider, "--output-dir", tmp_dir,
            ],
        ), patch.multiple(
            run,
            bootstrap_project_env=Mock(),
            require_collection_host=Mock(),
            require_cgroup_prerequisites=Mock(return_value="v2"),
            require_packet_latency_prerequisites=Mock(),
            require_cpu_energy_prerequisites=Mock(),
            require_mips_prerequisites=Mock(),
            start_terminal_log=Mock(return_value=None),
        ), patch("acprof.host.preflight.require_native_docker"), patch(
            "acprof.application.notifications.WeComWebhookNotifier.from_env", return_value=notifier,
        ) as from_env, patch(
            "acprof.host.detect.detect_task", return_value=task_info,
        ), patch(
            "acprof.host.preparation.interface_probe.probe_interface", return_value={"status": "ok"},
        ), patch(
            "acprof.host.images.runtime.configure_runtime_profile",
        ), patch(
            "acprof.host.images.runtime.prepare_image",
            return_value=runtime_images.ImageInfo(tag="acprof-nlp-org--resolved-model:latest"),
        ), patch(
            "acprof.host.metadata.static.collect_static_meta", return_value=SimpleNamespace(),
        ), patch.multiple(
            "acprof.host.metadata.static",
            enrich_static_meta_from_input_plan=Mock(side_effect=lambda meta, planned: meta),
            enrich_static_meta=Mock(side_effect=lambda meta, values: meta),
        ), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}
        ), patch(
            "acprof.host.metadata.static.write_static_meta_json",
        ), patch(
            "acprof.host.input_plan.plan_input_scales",
            return_value=input_plan.PlannedInputScales(
                scales=[1.0], source="unit", plan_file=None,
            ),
        ), patch(
            "acprof.host.profiling.compute.collect_compute_profile_plan",
            side_effect=lambda **kwargs: collect(("CPU Torch", "GPU Torch", "NCU"), **kwargs),
        ) as compute, patch(
            "acprof.host.profiling.execution.collect_execution_profile_plan",
            side_effect=lambda request, options, **kwargs: collect(
                ("Massif", "Nsys"), output_dir=request.output_dir, **kwargs,
            ),
        ) as execution, patch(
            "acprof.host.orchestrator.run_matrix", side_effect=run_matrix,
        ), redirect_stdout(io.StringIO()):
            run_cli.main()

        compute.assert_called_once()
        execution.assert_called_once()
        expected_callback = (
            None if provider == "none" else self.notices.notify_profiler
        )
        for collector in (compute, execution):
            assert (collector.call_args.kwargs["progress_callback"]) == (expected_callback)
        if provider == "none":
            from_env.assert_not_called()
            notifier.send.assert_not_called()
            assert (order) == (["matrix"])
        else:
            from_env.assert_called_once_with()
            assert (order) == (["started", "CPU Torch", "GPU Torch", "NCU", "Massif", "Nsys", "matrix", "no_results"])
            for event in events[1:]:
                assert (event.model_id) == (task_info.model_id)
                assert (event.output_dir) == (os.path.join(tmp_dir, "org--resolved-model"))
            assert (all(event.status == "profiler_success" for event in events[1:-1]))
            assert (events[-1].total_cases) == (2)

    def test_matrix_progress_callback_runs_after_each_case(self) -> None:
        order = []
        progress_events = []

        def fake_case(request, **kwargs):
            order.append(("case", request.cpu, request.gpu))
            return f"/tmp/{request.cpu}-{request.gpu}.csv"

        def progress(event):
            order.append(("progress", event.cpu, event.gpu))
            progress_events.append(event)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.orchestrator.run_single_case",
            side_effect=fake_case,
        ):
            result_csvs = orchestrator.run_matrix(cases.MatrixRequest(
                task_info=Mock(model_id="org/model", model_revision="main", task_family="nlp", model_source="huggingface", selected_tasks=[]),
                image_info=Mock(tag="sha256:" + "b" * 64),
                matrix_order="declared",
                cpu_list=[1, 2],
                mem_list=[4],
                gpu_list=["off", "on"],
                output_dir=tmp,
                project_dir=tmp,
                progress_callback=progress,
            ))

        assert (len(result_csvs)) == (4)
        assert (len(progress_events)) == (4)
        assert (order) == ([
                ("case", 1, "off"),
                ("progress", 1, "off"),
                ("case", 1, "on"),
                ("progress", 1, "on"),
                ("case", 2, "off"),
                ("progress", 2, "off"),
                ("case", 2, "on"),
                ("progress", 2, "on"),
            ])
        assert (progress_events[-1].completed_cases) == (4)
        assert (progress_events[-1].total_cases) == (4)

    def test_matrix_ignores_progress_callback_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.orchestrator.run_single_case",
            return_value="/tmp/result.csv",
        ), redirect_stderr(io.StringIO()):
            result_csvs = orchestrator.run_matrix(cases.MatrixRequest(
                task_info=Mock(model_id="org/model", model_revision="main", task_family="nlp", model_source="huggingface", selected_tasks=[]),
                image_info=Mock(tag="sha256:" + "b" * 64),
                matrix_order="declared",
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                output_dir=tmp,
                project_dir=tmp,
                progress_callback=Mock(side_effect=RuntimeError("notify failed")),
            ))

        assert (result_csvs) == (["/tmp/result.csv"])

    def test_application_finalizes_tmux_before_sending_success(self) -> None:
        order = []
        captured = []
        class CapturingNotifier:
            def send(self, event):
                order.append("notify")
                captured.append(event)
        terminal_log = ("%3", "/tmp/tmux_all.log.part", "/tmp/tmux_all.log")
        def stop_log(value):
            assert value == terminal_log
            order.append("tmux")
            return True
        application = run.RunApplication(notifications=self.notices, terminal_stop=stop_log)
        def finish_run(*args, **kwargs):
            context = self._context(CapturingNotifier())
            context.event = self._success_event()
            self.notices.context = context
            application.terminal_log = terminal_log
            return run.RunOutcome("ok")
        with patch.object(application, "_execute", side_effect=finish_run):
            application.run(RunConfig.smoke("org/model"))
        assert order == ["tmux", "notify"]
        assert captured[0].terminal_log == terminal_log[2]

    @pytest.mark.parametrize("failure", [RuntimeError("profiling failed"), SystemExit(7)])
    def test_application_preserves_failure_and_notifies(self, failure) -> None:
        captured = []
        class CapturingNotifier:
            def send(self, event):
                captured.append(event)
        application = run.RunApplication(notifications=self.notices)
        def fail_run(*args, **kwargs):
            self.notices.context = self._context(CapturingNotifier())
            raise failure
        with patch.object(application, "_execute", side_effect=fail_run):
            with pytest.raises(type(failure)) as raised:
                application.run(RunConfig.smoke("org/model"))
        assert raised.value is failure
        assert len(captured) == 1
        assert captured[0].status == "failed"
        assert type(failure).__name__ in captured[0].detail

    def test_unknown_delivery_error_does_not_change_success_result(self) -> None:
        unsafe_key = "secret-webhook-key"
        class FailingNotifier:
            def send(self, _event):
                raise RuntimeError("failed at https://qyapi.weixin.qq.com/cgi-bin/webhook/"
                                   f"send?key={unsafe_key}")
        application = run.RunApplication(notifications=self.notices)
        outcome = run.RunOutcome("ok", "/tmp/result.csv", 2)
        def finish_run(*args, **kwargs):
            context = self._context(FailingNotifier())
            context.event = self._success_event()
            self.notices.context = context
            return outcome
        stderr = io.StringIO()
        with patch.object(application, "_execute", side_effect=finish_run), redirect_stderr(stderr):
            result = application.run(RunConfig.smoke("org/model"))
        assert result is outcome
        assert "RuntimeError" in stderr.getvalue()
        assert unsafe_key not in stderr.getvalue()
