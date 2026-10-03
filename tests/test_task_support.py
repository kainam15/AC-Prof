import io
import json
import sys
import tempfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static
from tui_fixtures import AcprofTui

from acprof.cli import probe, run
from acprof.experiment import RunConfig
from acprof.host import detect
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.tui.commands import PendingLaunch
from acprof.tui.i18n import translate
from acprof.tui.log import SelectableLog
from acprof.tui.progress import RunProgressTracker


def task_info(tag="image-text-to-text", family="unknown"):
    return detect.TaskInfo(
        model_id="example/caption-model",
        pipeline_tag=tag,
        task_family=family,
        runtime_backend="diffusers" if family == "diffusion" else "chronos" if family == "timeseries" else "transformers_pipeline",
        library_name="transformers",
        model_revision="test-revision",
        detection_method="hub_api",
    )


@pytest.mark.parametrize('tag,family,batch_size', (('image-to-text', 'nlp', 1), ('video-classification', 'unknown', 1), ('image-to-text', 'cv', 2)))
@pytest.mark.parametrize('module_case', range(2), ids=['run', 'probe'])
def test_run_and_probe_reject_unsupported_tasks_before_build_or_results(tag, family, batch_size, module_case):
    module = tuple((run, probe))[module_case]
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        existing = root / "example--caption-model"
        existing.mkdir()
        csv = existing / "result_all.csv"
        csv.write_text("existing measurement\n", encoding="utf-8")
        stack.enter_context(patch.object(module, "bootstrap_project_env"))
        for guard in ("require_collection_host", "require_native_docker", "require_cgroup_prerequisites"):
            stack.enter_context(patch.object(module, guard, return_value="v2"))
        if module is run:
            for guard in ("require_packet_latency_prerequisites", "require_cpu_energy_prerequisites", "require_mips_prerequisites", "start_terminal_log"):
                stack.enter_context(patch.object(run, guard, return_value=None))
        stack.enter_context(patch("acprof.host.detect.detect_task", return_value=task_info(tag, family)))
        build_target = "acprof.host.runtime_images.prepare_image" if module is run else "acprof.cli.probe.prepare_image"
        build = stack.enter_context(patch(build_target, side_effect=AssertionError("unsupported task reached image preparation")))
        stderr = stack.enter_context(redirect_stderr(io.StringIO()))
        stack.enter_context(redirect_stdout(io.StringIO()))
        argv = ["--model", "example/caption-model", "--output-dir", tmp, "--skip-build", "--batch-size", str(batch_size)]
        if module is run:
            stack.enter_context(patch.object(sys, "argv", ["acprof run", *argv, "--notify", "none"]))
            with pytest.raises(SystemExit) as caught:
                run.main()
            code = caught.value.code
        else:
            code = probe.main(argv)
        assert (code) == (2)
        build.assert_not_called()
        output = stderr.getvalue()
        assert ("[task-support][ERROR]") in (output)
        assert (tag) in (output)
        assert ("example/caption-model") in (output)
        assert ("原因") in (output)
        assert ("解决办法") in (output)
        assert ("README.md") in (output)
        assert ("Traceback") not in (output)
        assert ("startup_oom") not in (output)
        if batch_size != 1:
            assert ("--batch-size 1") in (output)
        elif tag == "image-to-text":
            assert ("--task-family") in (output)
        assert (csv.read_text(encoding="utf-8")) == ("existing measurement\n")
        assert (sorted(str(p.relative_to(root)) for p in root.rglob("*"))) == (["example--caption-model", "example--caption-model/result_all.csv"])

def test_hub_task_without_adapter_is_preserved_instead_of_guessed_as_nlp():
    hub = SimpleNamespace(pipeline_tag="unregistered-vision-task", library_name="transformers", sha="rev")
    with patch("huggingface_hub.HfApi.model_info", return_value=hub), patch.object(
        detect, "_detect_from_config", return_value=task_info("text2text-generation", "nlp")
    ) as fallback:
        info = detect.detect_task("example/multimodal")
    assert (info.pipeline_tag) == ("unregistered-vision-task")
    assert (info.task_family) == ("unknown")
    fallback.assert_not_called()

def test_manual_override_can_correct_hub_metadata():
    hub = SimpleNamespace(pipeline_tag="unregistered-task", library_name="transformers", sha="rev")
    with patch("huggingface_hub.HfApi.model_info", return_value=hub):
        info = detect.detect_task("example/model", override_tag="image-classification")
    assert ((info.pipeline_tag, info.task_family)) == (("image-classification", "cv"))

@pytest.mark.parametrize('architecture', ('BlipForConditionalGeneration', 'Blip2ForConditionalGeneration', 'InstructBlipForConditionalGeneration', 'VisionEncoderDecoderModel'))
def test_blip_architecture_fallback_does_not_treat_images_as_nlp(architecture):
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "snapshots" / ("a" * 40) / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"architectures": [architecture]}), encoding="utf-8")
        with patch("huggingface_hub.hf_hub_download", return_value=str(config)):
            info = detect._detect_from_config("example/caption-model")
    assert ((info.pipeline_tag, info.task_family)) == (("image-to-text", "cv"))

@pytest.mark.parametrize('task,family', (('fill-mask', 'nlp'), ('text-generation', 'nlp'), ('image-classification', 'cv'), ('object-detection', 'cv'), ('image-to-text', 'cv'), ('automatic-speech-recognition', 'audio'), ('time-series-forecasting', 'timeseries'), ('text-to-image', 'diffusion')))
def test_supported_families_remain_available_and_wrong_family_is_actionable(task, family):
    require_task_support(task_info(task, family))
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task_info("image-classification", "nlp"))
    assert ("任务族 nlp 与之不匹配") in (str(caught.value))
    assert ("--task-family") in (str(caught.value))

def test_manual_caption_task_is_supported_but_requires_single_image_batch():
    with patch.object(detect, "_detect_from_hub", return_value=task_info("fill-mask", "nlp")):
        info = detect.detect_task("example/model", override_tag="image-to-text")
    require_task_support(info)
    with pytest.raises(TaskSupportError, match="--batch-size 1"):
        require_task_support(info, batch_size=2)

def test_progress_preserves_unsupported_task_and_points_to_remedies():
    tracker = RunProgressTracker()
    state = tracker.feed("[task-support][ERROR] Unsupported collection task: image-text-to-text")
    assert (state.stage) == ("任务不支持")
    assert ("image-text-to-text") in (state.detail)
    assert ("解决办法见日志") in (state.detail)
    assert not (state.measurement_active)
    assert (state.errors) == (1)
    state = tracker.feed("  原因：当前项目尚未登记该任务类型的采集适配。")
    assert (state.stage) == ("任务不支持")
    assert ("image-text-to-text") in (translate(state.detail, "en"))
    assert ("log") in (translate(state.detail, "en"))


@pytest.mark.parametrize('language', ('zh', 'en'))
@pytest.mark.parametrize('kind', ('run', 'probe'))
@pytest.mark.parametrize('size', ((80, 24), (120, 30)))
async def test_run_and_probe_keep_actionable_failure_visible_in_both_languages(language, kind, size):
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task_info())
    lines = str(caught.value).splitlines()
    with tempfile.TemporaryDirectory() as tmp:
        config = replace(RunConfig.smoke("example/caption-model"), output_dir=tmp)
        old_csv = config.result_csv(Path.cwd())
        old_csv.parent.mkdir(parents=True)
        old_csv.write_text("existing measurement\n", encoding="utf-8")
        app = AcprofTui(config, settings_path=Path(tmp) / "tui.json")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_language()
            with patch.object(app, "_execute_command"):
                app._launch(PendingLaunch(("unused",), kind, config))
            await pilot.pause()
            tracker = RunProgressTracker()
            for line in lines:
                previous = tracker.snapshot
                state = tracker.feed(line)
                app._consume_process_line(line, state, previous != state)
            with patch.object(app, "notify") as notify, patch.object(
                app, "_update_result_summary"
            ) as read_old_results:
                app._process_finished(kind, 2, tracker.snapshot, "")
            await pilot.pause()
            assert (app._latest_snapshot.stage) == ("任务不支持")
            assert (app.query_one("#status-stage", Static).content) == ("任务不支持" if language == "zh" else "Unsupported task")
            stage = app.query_one("#status-stage", Static)
            assert (stage.region.width) > (0)
            assert (stage.region.height) > (0)
            assert (stage.region.bottom) < (size[1])
            assert (stage.is_on_screen)
            assert ("image-text-to-text") in (app._latest_snapshot.detail)
            assert ("解决办法") in (app.query_one("#run-log", SelectableLog).text)
            assert ("处理办法" if language == "zh" else "Remedies:") in (app.query_one("#run-log", SelectableLog).text.splitlines()[-1])
            assert ("image-text-to-text") in (str(notify.call_args.args[0]))
            assert not (app.query_one("#start-run", Button).disabled)
            assert not (app.query_one("#probe-largest", Button).disabled)
            assert (app._latest_snapshot.completed_cases) == (0)
            assert not (app._latest_snapshot.measurement_active)
            read_old_results.assert_not_called()
            assert (old_csv.read_text(encoding="utf-8")) == ("existing measurement\n")
