"""User decisions return to the same subprocess without another Start click."""
import asyncio
import io
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Select

from acprof.events.preparation import encode_event
from acprof.experiments.configuration import RunConfig
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.process import ProcessLifecycle
from acprof.tui.run.progress import RunProgressTracker
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_automatic_interface_selection_reviews_parameters_before_runtime(tmp_path, language):
    script = textwrap.dedent('''
        from types import SimpleNamespace
        from acprof.host.preparation.collection_workflow import PreparationWorkflow
        from acprof.host.preparation.inference_configuration import prepare_inference_configuration
        from acprof.host.detect import TaskInfo
        from acprof.models.candidates import apply_model_candidate, discover_model_candidates
        task = TaskInfo("fixture/model", "unknown", "unknown", "unknown", "transformers",
                        "a" * 40, "config_infer", model_config={"model_type": "bert",
                        "architectures": ["BertForMaskedLM"]},
                        repository_files=("config.json", "model.safetensors"))
        task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
        args = SimpleNamespace(task="", tasks="", backend="", task_family="", model_spec="",
                               revision="", batch_size=1, gpus="off")
        workflow = PreparationWorkflow(interactive=True)
        selected = workflow.resolve(args, initial=task)
        assert len(selected.selected_tasks) == 1
        selected = prepare_inference_configuration(selected, args, workflow, devices=["off"])
        assert selected.inference_configuration["state"] == "frozen"
        workflow.emit("runtime", "running")
        print("AUTO_INTERFACE_SELECTED", flush=True)
    ''')
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    original = ProcessLifecycle.start
    def launch(owner, _command, **kwargs):
        return original(owner, [sys.executable, "-c", script], **kwargs)
    with patch("acprof.tui.run.process.ProcessLifecycle.start", autospec=True, side_effect=launch):
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            assert await pilot.click("#start-run")
            await wait_for_condition(
                pilot, lambda: app.preflight.preparation_screen is not None,
                description="the inference configuration review", timeout=15,
            )
            # The request reference is set before mounting and layout finish.
            await pilot.pause()
            screen = app.preflight.preparation_screen
            assert screen is not None
            assert screen.is_mounted
            assert app.screen is screen
            assert screen.event["request"]["purpose"] == "inference_configuration"
            assert await pilot.click("#preparation-apply")
            await wait_for_condition(
                pilot, lambda: (app.preflight.preparation_screen is not None
                                and app.preflight.preparation_screen.event["request"].get("purpose")
                                == "inference_confirmation"),
                description="the final inference confirmation", timeout=15,
            )
            await pilot.pause()
            confirmation = app.preflight.preparation_screen
            assert confirmation is not None
            assert len(confirmation.event["request"]["confirmation_digest"]) == 64
            app.logs.flush()
            assert "AUTO_INTERFACE_SELECTED" not in app.query_one("#run-log", SelectableLog).text
            assert await pilot.click("#preparation-continue")
            await wait_for_condition(
                pilot, lambda: app.execution.lifecycle.process is None,
                description="the confirmed fixture process to finish", timeout=15,
            )
            await pilot.pause()
            app.logs.flush()
            assert "AUTO_INTERFACE_SELECTED" in app.query_one("#run-log", SelectableLog).text
            assert app.preflight.preparation_screen is None
            assert not app.query_one("#start-run", Button).disabled


async def test_confirmed_stop_reaps_child_releases_lock_and_removes_bundle(tmp_path):
    import os

    from textual.widgets import TabbedContent

    from acprof.host.run_state import MEASUREMENT_LOCK_NAMESPACE, MeasurementLock
    script = textwrap.dedent('''
        import json, signal, sys, time
        from pathlib import Path
        from acprof.host.detect import TaskInfo
        from acprof.host import run_state
        from acprof.host.preparation.source_bundle import source_bundle
        from acprof.events.preparation import encode_event
        signal.signal(signal.SIGINT, signal.default_int_handler)
        task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers", "transformers", "a" * 40, "fixture")
        run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
        run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
        with run_state.MeasurementLock(), source_bundle(task) as bundle:
            Path(sys.argv[1], "bundle.json").write_text(json.dumps(str(bundle.root)))
            print(encode_event("runtime", "running"), flush=True)
            time.sleep(120)
    ''')
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    original_start = ProcessLifecycle.start
    def launch_child(owner, _command, **kwargs):
        return original_start(owner, [sys.executable, "-c", script, str(tmp_path), MEASUREMENT_LOCK_NAMESPACE.hex()], **kwargs)
    with patch("acprof.tui.run.process.ProcessLifecycle.start", autospec=True, side_effect=launch_child):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#start-run")
            await wait_for_condition(
                pilot, lambda: (tmp_path / "bundle.json").exists(),
                description="the child to create its source bundle", timeout=10,
            )
            await pilot.pause()
            process = app.execution.lifecycle.process
            pid = process.pid
            assert app.preflight.preparation_screen is None
            assert await pilot.click("#stop-run")
            await pilot.pause()
            assert await pilot.click("#confirm-yes")
            await asyncio.wait_for(asyncio.shield(app.workers.wait_for_complete()), timeout=45)
            await pilot.pause()
            assert app.execution.lifecycle.process is None
            assert app.query_one(TabbedContent).active == "monitor-tab"
            log_text = app.query_one("#run-log", SelectableLog).raw_text
            assert "任务已由用户终止" in log_text
            assert not app.query_one("#start-run", Button).disabled
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    import json
    assert not Path(json.loads((tmp_path / "bundle.json").read_text())).exists(), f"returncode={process.returncode}\n{log_text}"
    with patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", tmp_path), MeasurementLock():
        pass


def test_runtime_failure_retains_successful_resolution():
    tracker = RunProgressTracker()
    tracker.feed(encode_event("resolution", "passed"))
    state = tracker.feed(encode_event("runtime", "failed"))
    assert (getattr(state, "interface_status", None)) == ("passed")
    assert (getattr(state, "runtime_status", None)) == ("failed")
    assert (getattr(state, "measurement_status", None)) == ("not_started")
    assert not (state.measurement_active)


async def test_start_button_keeps_real_child_alive_through_answer_and_retry():
    script = textwrap.dedent('''
            import os
            from acprof.host.preparation.collection_workflow import PreparationWorkflow
            workflow = PreparationWorkflow.from_environment("unused")
            reply = workflow.ask("resolution", "review", questions=[
                {"path": "task", "options": ["fill-mask", "text-generation"], "value": None}])
            assert reply["answers"] == {"task": "fill-mask"}
            workflow.emit("resolution", "passed")
            attempts = 0
            def validate():
                global attempts
                attempts += 1
                if attempts == 1:
                    raise FileNotFoundError("ultravox_config.py")
            workflow.run("runtime", validate)
            print("WORKFLOW_DONE", os.getpid(), flush=True)
        ''')
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=Path(directory, "settings.json"))
        original_start = ProcessLifecycle.start
        def launch_child(owner, _command, **kwargs):
            return original_start(owner, [sys.executable, "-c", script], **kwargs)
        with patch("acprof.tui.run.process.ProcessLifecycle.start", autospec=True, side_effect=launch_child) as launch:
            async with app.run_test(size=(80, 24)) as pilot:
                async def wait_for_request(request_id):
                    await wait_for_condition(
                        pilot, lambda: app.preflight.preparation_request is not None
                        and app.preflight.preparation_request[1] == request_id,
                        description=f"preparation request {request_id}", timeout=15,
                    )
                    await pilot.pause()

                assert (await pilot.click("#start-run"))
                await pilot.pause()
                await wait_for_request(1)
                pid = app.execution.lifecycle.process.pid
                app.screen.query_one("#preparation-answer-0", Select).value = "fill-mask"
                assert (await pilot.click("#preparation-apply"))
                await wait_for_request(2)
                assert (app.execution.lifecycle.process.pid) == (pid)
                assert (app.execution.session.snapshot.interface_status) == ("passed")
                assert (app.execution.session.snapshot.runtime_status) == ("failed")
                assert (app.execution.session.snapshot.measurement_status) == ("not_started")
                assert (await pilot.click("#preparation-continue"))
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert (app.execution.lifecycle.process) is None
                assert (app.execution.session.snapshot.runtime_status) == ("passed")
                assert (f"WORKFLOW_DONE {pid}") in (app.query_one("#run-log", SelectableLog).text)
                assert ("ACPROF_PREPARATION") not in (app.query_one("#run-log", SelectableLog).text)
                launch.assert_called_once()

@pytest.mark.parametrize('language', ['zh', 'en'])
@pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
async def test_review_and_error_actions_reply_without_relaunch_at_all_sizes(language, size):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=Path(directory, "settings.json"))
        process = Mock(stdin=io.StringIO())
        process.poll.return_value = None
        async with app.run_test(size=size) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            app.execution.lifecycle.process = process
            try:
                with patch.object(app, 'launch') as launch:
                    handler = app.preflight.preparation_event
                    handler({"version": 1, "stage": "resolution", "status": "waiting", "request": {
                        "id": 1, "kind": "review", "detail": "", "questions": [
                            {"path": "task", "value": None, "options": ["fill-mask", "text-generation"], "reason": "two candidates"},
                        ],
                    }})
                    await pilot.pause()
                    choice = app.screen.query_one("#preparation-answer-0", Select)
                    choice.value = "fill-mask"
                    assert (await pilot.click("#preparation-apply"))
                    await pilot.pause()
                    assert ('"action": "answer"') in (process.stdin.getvalue())
                    assert ('"task": "fill-mask"') in (process.stdin.getvalue())
                    handler({"version": 1, "stage": "runtime", "status": "failed", "request": {
                        "id": 2, "kind": "error", "detail": "FileNotFoundError: ultravox_config.py\n" + "diagnostic detail\n" * 40,
                    }})
                    await pilot.pause()
                    assert not (list(app.screen.query(Select)))
                    detail = app.screen.query_one("#preparation-traceback", SelectableLog).text
                    assert isinstance(detail, str)
                    assert ("ultravox_config.py") in (detail)
                    button = app.screen.query_one("#preparation-continue", Button)
                    assert (str(button.label)) == ("重试" if language == "zh" else "Retry")
                    assert (await pilot.click(button))
                    await pilot.pause()
                    assert ('"action": "retry"') in (process.stdin.getvalue())
                    launch.assert_not_called()
            finally:
                app.execution.lifecycle.process = None
async def test_model_lookup_error_is_translated_in_preparation_dialog():
    from acprof.host.preparation.model_errors import ModelLookupError
    from acprof.tui.run.preparation import PreparationScreen

    error = ModelLookupError("asdf", "repository_unavailable", detail="RepositoryNotFoundError: fixture")
    event = {"stage": "resolution", "request": {"id": 1, "kind": "error", "detail": str(error),
                                               "model_error": error.to_dict()}}
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig(model="asdf"), settings_path=Path(directory, "settings.json"))
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = "en"
            await pilot.pause()
            await app.push_screen(PreparationScreen(event))
            await pilot.pause()
            detail = "\n".join(
                str(node.content)
                for node in app.screen.query("#preparation-banner Static, #preparation-callout Static")
            )
            assert ("not found") in (detail)
            assert ("asdf") in (detail)
            assert ("未找到") not in (detail)
            assert ("SystemExit") not in (detail)


@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
@pytest.mark.parametrize("completion", ["runtime", "process"])
async def test_preparation_completion_preserves_stop_confirmation(tmp_path, size, completion):
    from acprof.tui.ui.dialogs import ConfirmActionScreen

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    process = Mock(stdin=io.StringIO())
    process.poll.return_value = None
    async with app.run_test(size=size) as pilot:
        base = app.screen
        app.execution.lifecycle.process = process
        app.execution.session.kind = "run"
        try:
            app.set_busy(True)
            app.activate_tab("monitor-tab")
            stop = app.query_one("#stop-run", Button)
            stop.focus()
            app.preflight.show_preparation({"stage": "runtime", "status": "running"})
            await pilot.pause()
            preparation = app.screen
            await pilot.press("ctrl+x", "tab")
            await pilot.pause()
            confirmation = app.screen
            assert isinstance(confirmation, ConfirmActionScreen)
            focused = confirmation.focused
            assert focused is not None

            if completion == "runtime":
                app.preflight.preparation_event({"stage": "runtime", "status": "passed"})
            else:
                app.execution.lifecycle.process = None
                app.process_finished("run", 1, None, "")
            await pilot.pause()

            assert app.screen is confirmation
            assert confirmation.focused is focused
            assert not app.execution.session.stop_requested
            assert await pilot.click("#confirm-no")
            await pilot.pause()
            assert app.screen is base
            assert preparation not in app.screen_stack
            assert confirmation not in app.screen_stack
            assert app.preflight.preparation_screen is None
            assert app.screen.focused is not None
            assert app.screen.focused.screen is base
            if completion == "runtime":
                assert app.screen.focused is stop
                assert app.execution.lifecycle.process is process
            else:
                assert not app.query_one("#start-run", Button).disabled
        finally:
            app.execution.lifecycle.process = None
            app.execution.session.kind = ""


async def test_completed_preparation_preserves_nested_dialog_replies_and_new_preparation(tmp_path):
    from acprof.tui.ui.dialogs import ConfirmActionScreen

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    responses = []
    async with app.run_test(size=(80, 24)) as pilot:
        base = app.screen
        app.preflight.show_preparation({"stage": "runtime", "status": "running"})
        await pilot.pause()
        previous = app.screen
        first = ConfirmActionScreen("First", "Keep this decision", "Confirm")
        await app.push_screen(first, lambda answer: responses.append(("first", answer)))
        second = ConfirmActionScreen("Second", "Keep this decision too", "Confirm")
        await app.push_screen(second, lambda answer: responses.append(("second", answer)))
        await pilot.pause()
        app.preflight.close_preparation()
        app.preflight.close_preparation()
        await pilot.pause()
        assert app.screen is second
        assert responses == []

        # A completed screen can still have queued messages while covered.
        previous.update_event({"stage": "runtime", "status": "running"})
        previous.action_cancel()
        assert not app.execution.session.stop_requested
        app.preflight.show_preparation({"stage": "resolution", "status": "running"})
        await pilot.pause()
        current = app.screen
        assert current is not previous
        app.preflight.close_preparation()
        await pilot.pause()
        assert app.screen is second
        assert await pilot.click("#confirm-yes")
        await pilot.pause()
        assert app.screen is first
        assert responses == [("second", True)]
        assert await pilot.click("#confirm-no")
        await pilot.pause()
        assert responses == [("second", True), ("first", False)]
        assert app.screen is base
        assert previous not in app.screen_stack
        assert current not in app.screen_stack


@pytest.mark.parametrize("outcome", [pytest.param("image_error", id="continue"), "cancel", "matching", "different"])
async def test_configured_probe_reviews_in_its_own_subprocess(tmp_path, outcome):
    import json
    from dataclasses import replace

    from acprof.tui.i18n import translate
    from acprof.tui.run.stages import Stage

    request = tmp_path / "requests.json"
    request.write_text('{"schema_version":1}')
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", output_dir=str(tmp_path),
                     input_scales="32", inference_config=str(request), confirm_inference="a" * 64)
    script = textwrap.dedent('''
        import json, sys
        from pathlib import Path
        from unittest.mock import Mock
        from acprof.cli import probe
        from acprof.host import detect, largest_scale_probe, runtime_validation
        from acprof.host.docker_runtime import RunningContainer
        from acprof.host.images.runtime import ImageInfo
        from acprof.host.input_plan import families, probe as input_probe
        from acprof.models.inference_configuration import configuration_identity, freeze_inference_configuration
        from acprof.models.candidates import apply_model_candidate, discover_model_candidates
        root, outcome = Path(sys.argv[1]), sys.argv[2]
        task = detect.TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_model", "transformers",
                               "a" * 40, "config_infer", model_config={"model_type": "bert",
                               "architectures": ["BertForMaskedLM"]},
                               repository_files=("config.json", "model.safetensors"))
        task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
        detect.detect_task = Mock(return_value=task)
        probe.bootstrap_project_env = probe.require_collection_host = probe.require_native_docker = Mock()
        probe.require_cgroup_prerequisites = Mock(return_value="v2")
        def capture(task, *args, **kwargs):
            (root / "confirmed.json").write_text(json.dumps(task.inference_configuration))
            if outcome == "image_error":
                raise RuntimeError("offline fixture reached image preparation")
            if outcome == "different":
                # A different, internally valid configuration reaches both the plan and the summary.
                candidate = json.loads((root / "confirmed.json").read_text())
                candidate["state"] = "resolved"
                loading = candidate["loading"]["off"]
                loading["dtype"] = "BF16" if loading.get("dtype") == "FP32" else "FP32"
                candidate["identity_sha256"] = configuration_identity(candidate)
                task.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
            return ImageInfo(tag="fixture-probe")
        probe.prepare_image = capture
        runtime_validation.validate_runtime = Mock()
        input_probe.start_service_session = families.stop_service_session = Mock()
        families.post_probe_payload = lambda session, payload, desc: {
            "payload": payload, "truncated_by_limit": False, "effective_input_scale": 32,
        }
        largest_scale_probe.start_container_session = Mock(return_value=RunningContainer(
            "fixture", "http://127.0.0.1:8002", 8002, 0.1))
        largest_scale_probe.stop_container_session = Mock()
        response = Mock(status_code=200)
        response.json.return_value = {"effective_input_scale": 32}
        largest_scale_probe.requests.post = Mock(return_value=response)
        code = probe.main(sys.argv[3:])
        (root / "exit-code").write_text(str(code))
        sys.exit(code)
    ''')
    original = ProcessLifecycle.start

    def launch(owner, command, **kwargs):
        return original(owner, [sys.executable, "-u", "-c", script, str(tmp_path), outcome,
                                *command[command.index("probe") + 1:]], **kwargs)

    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(ProcessLifecycle, "start", autospec=True, side_effect=launch) as start:
        async with app.run_test(size=(120, 40)) as pilot:
            app.experiments.request_probe()
            await pilot.pause()
            assert "按审阅后的推理配置取最大值" in app.screen.message
            assert await pilot.click("#confirm-yes")

            async def wait_for_review(purpose):
                await wait_for_condition(
                    pilot, lambda: (app.preflight.preparation_screen is not None and
                                    app.preflight.preparation_screen.event["request"]["purpose"] == purpose),
                    description=purpose, timeout=10,
                )
                await pilot.pause()
                return app.preflight.preparation_screen.event["request"]

            await wait_for_review("inference_configuration")
            assert await pilot.click("#preparation-apply")
            confirmation = await wait_for_review("inference_confirmation")
            digest = confirmation["confirmation_digest"]
            assert digest != config.confirm_inference
            assert not (tmp_path / "confirmed.json").exists()
            assert await pilot.click("#preparation-cancel" if outcome == "cancel" else "#preparation-continue")
            await wait_for_condition(pilot, lambda: (tmp_path / "exit-code").is_file()
                                     and not app.is_busy() and not app.execution.session.kind,
                                     description="the reviewed probe to exit", timeout=10)
            start.assert_called_once()
            code = (tmp_path / "exit-code").read_text()
            if outcome == "cancel":
                assert code == "130"
                assert not (tmp_path / "confirmed.json").exists()
                return

            frozen = json.loads((tmp_path / "confirmed.json").read_text())
            assert frozen["identity_sha256"] == digest
            assert frozen["state"] == "frozen"
            if outcome == "image_error":
                assert code == "2"
                return

            from acprof.host.largest_scale_probe import read_probe_summary

            assert code == "0"
            snapshot = app.execution.session.snapshot
            summary = read_probe_summary(snapshot.probe_summary)
            if outcome == "matching":
                assert summary["inference_configuration_sha256"] == digest
                assert snapshot.stage is Stage.PROBE_COMPLETED
            else:
                assert summary["inference_configuration_sha256"] != digest
                assert snapshot.stage is Stage.PROBE_FAILED
                app.logs.flush()
                assert "SHA-256 与本次确认值不一致" in app.query_one("#run-log", SelectableLog).text
                assert "differs from the one you confirmed" in translate(snapshot.detail, "en")


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_blocked_model_returns_to_editing_after_child_exit(tmp_path, language):
    from dataclasses import replace

    from textual.widgets import Input, Static, TabbedContent

    from acprof.tui.ui.structured import FactGrid

    script = textwrap.dedent(f"""
        from pathlib import Path
        from acprof.experiments.arguments import build_parser
        from acprof.host.detect import TaskInfo
        from acprof.host.preparation.collection_workflow import PreparationInterrupted, PreparationWorkflow
        from acprof.models.candidates import apply_model_candidate, discover_model_candidates

        task = TaskInfo("fixture/coreml", "automatic-speech-recognition", "audio", "unknown", "whisperkit",
                        "a" * 40, "hub", hub_metadata={{"pipeline_tag": "automatic-speech-recognition"}},
                        repository_files=("config.json", "AudioEncoder.mlmodelc/weights/weight.bin"))
        task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
        args = build_parser().parse_args(["--model", task.model_id, "--revision", task.model_revision, "--gpus", "off"])
        try:
            PreparationWorkflow(interactive=True).resolve(args, initial=task)
        except PreparationInterrupted as exc:
            Path({str(tmp_path / "reply")!r}).write_text(exc.reason_code)
    """)
    config = replace(RunConfig.smoke("fixture/coreml"), repeat=3, output_dir=str(tmp_path / "output"))
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    start = ProcessLifecycle.start

    def launch(owner, _command, **kwargs):
        return start(owner, [sys.executable, "-c", script], **kwargs)

    with patch.object(ProcessLifecycle, "start", autospec=True, side_effect=launch) as launches:
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            before = app.configuration.collect_config()
            assert await pilot.click("#start-run")
            await wait_for_condition(pilot, lambda: app.preflight.preparation_request is not None,
                                     description="blocked model request", timeout=15)
            await pilot.pause()
            screen = app.screen
            labels = [str(b.label) for b in screen.query("#preparation-actions Button")]
            assert labels == (["返回选择模型", "查看诊断"] if language == "zh"
                              else ["Back to model selection", "View diagnostics"])
            assert not list(screen.query(Input))
            facts = screen.query_one("#preparation-fields", FactGrid).section.rows
            assert [row.value for row in facts] == [
                config.model, "a" * 40, "automatic-speech-recognition", "whisperkit",
            ]
            assert any(("仓库声明的任务" if language == "zh" else "Repository task") in str(node.render())
                       for node in screen.query(".fact-label"))
            assert "Core ML" in str(screen.query_one("#preparation-problem", Static).content)
            assert await pilot.click("#preparation-view-diagnostics")
            await pilot.pause()
            detail = str(app.screen.query_one("#fact-detail-body Static", Static).render())
            assert all(value in detail for value in ("artifact_layout_unsupported", "AudioEncoder.mlmodelc", "Traceback"))
            assert not (tmp_path / "reply").exists()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert await pilot.click("#preparation-cancel")
            await wait_for_condition(pilot, lambda: app.focused is app.query_one("#model", Input),
                                     description="child exit and focus restored to model ID", timeout=15)
            assert app.execution.lifecycle.process is None
            assert app.preflight.preparation_request is None
            assert app.query_one("#main-tabs", TabbedContent).active == "run-tab"
            assert app.configuration.collect_config() == before
            assert (tmp_path / "reply").read_text() == "cancelled_by_user"
            launches.assert_called_once()
