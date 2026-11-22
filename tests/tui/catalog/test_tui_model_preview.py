"""The form inspects completed edits using the same static CLI and measurement exclusion."""
import asyncio
import sys

import pytest
from textual.widgets import Input, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.run_state import RunStateError
from acprof.tui.models import preview
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def test_completed_edits_replace_old_inspection_and_show_failure(tmp_path, monkeypatch):
    app = AcprofTui(RunConfig.smoke("original/model"), settings_path=tmp_path / "settings.json")
    calls = []
    first_cancelled = asyncio.Event()

    async def inspect(arguments, python, cwd):
        calls.append(arguments)
        if len(calls) == 1:
            try:
                await asyncio.Event().wait()
            finally:
                first_cancelled.set()
        return {"source": arguments[2], "revision": "c" * 40, "formats": ["CoreML"],
                "platform": "Apple Core ML", "status": "blocked", "reason": "No compatible loader"}

    monkeypatch.setattr("acprof.tui.models.controller.inspect_summary", inspect)
    async with app.run_test(size=(100, 36)) as pilot:
        model = app.query_one("#model", Input)
        widget = app.query_one("#model-preview", Static)
        assert not calls
        assert not widget.display
        model.focus()
        model.value = "test/model-a"
        await pilot.pause()
        assert not calls
        await pilot.press("tab")
        await wait_for_condition(pilot, lambda: bool(calls), description="blur starts metadata read", timeout=5)
        assert calls == [("test/model-a", "--model-source", "huggingface", "--batch-size", "1", "--gpus", "off")]
        assert not model.disabled
        assert not app.operation_state().allows("run")
        model.focus()
        model.value = "test/model-b"
        app.query_one("#revision", Input).value = "v2"
        app.query_one("#model-variant", Input).value = "small/"
        app.query_one("#backend", Input).value = "transformers_pipeline"
        app.query_one("#model-source", Select).value = "modelscope"
        await wait_for_condition(pilot, lambda: "No compatible loader" in str(widget.content),
                                 description="show current repository diagnosis", timeout=5)
        assert first_cancelled.is_set()
        assert calls[-1] == ("test/model-b", "--model-source", "modelscope", "--revision", "v2",
                             "--model-variant", "small/", "--backend", "transformers_pipeline", "--batch-size", "1", "--gpus", "off")
        assert "c" * 40 in str(widget.content)
        assert "CoreML" in str(widget.content)
        assert "Apple Core ML" in str(widget.content)
        model.value = ""
        await pilot.pause()
        assert not widget.display


async def test_external_measurement_suppresses_inspection_and_enter_retries(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from acprof.host import run_state

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    calls = []

    @contextmanager
    def busy_lock():
        raise RunStateError("fixture measurement in progress")
        yield

    async def inspect(arguments, python, cwd):
        calls.append(arguments)
        return {"architecture": "WhisperForConditionalGeneration", "formats": ["safetensors"]}

    monkeypatch.setattr("acprof.tui.models.controller.inspect_summary", inspect)
    async with app.run_test() as pilot:
        widget = app.query_one("#model-preview", Static)
        app.query_one("#model", Input).focus()
        with monkeypatch.context() as guard:
            guard.setattr(run_state, "offline_analysis_lock", busy_lock)
            await pilot.press("enter")
            await wait_for_condition(pilot, lambda: "fixture measurement" in str(widget.content),
                                     description="measurement exclusion", timeout=5)
            assert not calls
        await pilot.press("enter")
        await wait_for_condition(pilot, lambda: "WhisperForConditionalGeneration" in str(widget.content),
                                 description="retry after lock release", timeout=5)
        assert len(calls) == 1
        assert "safetensors" in str(widget.content)


async def test_cancelled_summary_reaps_its_cli(tmp_path, monkeypatch):
    from pathlib import Path

    ready = tmp_path / "pid"
    command = "import os, pathlib, time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)"
    monkeypatch.setattr(preview, "cli_command", lambda *a, **kw: [sys.executable, "-c", command])
    worker = asyncio.create_task(preview.inspect_summary(("example/model",), Path(sys.executable), tmp_path))
    try:
        async with asyncio.timeout(5):
            while not ready.is_file() or not ready.read_text():
                await asyncio.sleep(.01)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)


@pytest.mark.parametrize(("source", "offline_key"), [
    ("huggingface", "HF_HUB_OFFLINE"), ("modelscope", "HF_HUB_OFFLINE"), ("modelscope", "MODELSCOPE_OFFLINE"),
])
async def test_static_preview_obeys_project_offline_policy_before_any_network(tmp_path, monkeypatch, source, offline_key):
    import os
    from pathlib import Path

    marker = tmp_path / "attempted-network"
    (tmp_path / "sitecustomize.py").write_text(
        "import pathlib, socket\n"
        "def deny(*args, **kwargs):\n"
        "    pathlib.Path('attempted-network').touch()\n"
        "    raise AssertionError('unexpected network')\n"
        "socket.socket.connect = socket.socket.connect_ex = deny\n"
    )
    git = tmp_path / "git"
    git.write_text("#!/bin/sh\ntouch attempted-network\nexit 1\n")
    git.chmod(0o755)
    (tmp_path / ".env.local").write_text(f"{offline_key}=1\nHF_ENDPOINT=https://example.invalid\n")
    for key in ("HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.setenv("PATH", os.pathsep.join((str(tmp_path), os.environ["PATH"])))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", ""))))
    summary = await preview.inspect_summary(
        ("offline-fixture/missing", "--model-source", source, "--model-store", str(tmp_path / "store")),
        Path(sys.executable), tmp_path,
    )
    assert not marker.exists()
    assert summary["lookup_error"]["reason_code"] == "offline"
    text = str(preview.preview_text(summary))
    assert "离线模式" in text
    assert "本地缓存" in text
    assert "Traceback" not in text


async def test_gguf_file_picker_pins_revision_and_rechecks_selected_file(tmp_path, monkeypatch):
    repo = "audio-cpp/audio.cpp-gguf"
    commit = "a" * 40
    first, second = "ASR/tiny-q8_0.gguf", "ASR/small-f16.gguf"
    unsupported = "TTS/unsupported.gguf"
    calls = []

    async def inspect(arguments, _python, _cwd):
        calls.append(arguments)
        variant = arguments[arguments.index("--model-variant") + 1] if "--model-variant" in arguments else first
        return {
            "revision": commit, "source": "huggingface", "repository_task": "text-to-speech",
            "backend": "audio_cpp", "platform": "CPU", "formats": ["GGUF"], "compatibility": "supported",
            "model_variant": variant, "gguf_candidates": [
                {"path": path, "family": "whisper", "task": "automatic-speech-recognition",
                 "quantization": quant, "selectable": True} for path, quant in ((first, "Q8_0"), (second, "F16"))
            ] + [{"path": unsupported, "selectable": False}],
        }

    monkeypatch.setattr("acprof.tui.models.controller.inspect_summary", inspect)
    app = AcprofTui(RunConfig.smoke(repo), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#model", Input).focus()
        await pilot.press("enter")
        picker = app.query_one("#gguf-file", Select)
        summary = app.query_one("#model-preview", Static)
        await wait_for_condition(pilot, lambda: picker.display, description="GGUF candidates available", timeout=5)
        text = str(summary.content)
        assert all(value in text for value in (first, "audio_cpp", "text-to-speech", "automatic-speech-recognition",
                                               "Q8_0", "3", "2"))
        assert second not in text
        assert unsupported not in text
        picker.scroll_visible(immediate=True, animate=False)
        await pilot.pause()
        assert await pilot.click("#gguf-file SelectCurrent")
        await pilot.press("end", "enter")
        await wait_for_condition(pilot, lambda: second in str(summary.content),
                                 description="selected file is inspected at the reviewed revision", timeout=5)
        assert calls[-1][calls[-1].index("--revision") + 1] == commit
        config = app.configuration.collect_config()
        assert (config.model, config.model_variant, config.revision) == (repo, second, commit)
        assert "F16" in str(summary.content)
        app.query_one("#model", Input).value = "new/model"
        await pilot.pause()
        assert not picker.display
        assert not app.configuration.input("model-variant")
        assert not app.configuration.input("revision")


async def test_preparation_gguf_review_offers_auto_choice_and_validates_reply(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from acprof.events.preparation import encode_event, parse_event
    from acprof.host.preparation.task_selection import review_checkpoint_variant
    from acprof.tui.run.preparation import PreparationScreen

    paths = ["a/asr-small-q8_0.gguf", "b/asr-tiny-f16.gguf"]
    task = SimpleNamespace(
        model_id="audio-cpp/audio.cpp-gguf", model_revision="b" * 40, model_variant=paths[0],
        metadata_errors=[], model_spec={}, model_resolution={
            "checkpoint_variants": paths, "gguf_candidates": [
                {"path": path, "selectable": True} for path in paths
            ] + [{"path": "c/tts.gguf", "selectable": False}],
        },
    )
    workflow = Mock()
    workflow.ask.return_value = {"answers": {"model_variant": "c/tts.gguf"}}
    with pytest.raises(ValueError, match="supported GGUF file"):
        review_checkpoint_variant(task, workflow)
    stage, kind = workflow.ask.call_args.args
    request = {"id": 3, "kind": kind, **workflow.ask.call_args.kwargs}
    event = parse_event(encode_event(stage, "waiting", request=request))
    replies = []
    app = AcprofTui(RunConfig.smoke(task.model_id), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        selector = screen.query_one("#preparation-answer-0", Select)
        assert selector.value == paths[0]
        selector.scroll_visible(immediate=True, animate=False)
        await pilot.pause()
        assert await pilot.click("#preparation-answer-0 SelectCurrent")
        await pilot.press("end", "enter")
        assert selector.value == paths[1]
        assert await pilot.click("#preparation-apply")
        assert replies == [{"action": "answer", "answers": {"model_variant": paths[1]}}]
        workflow.ask.return_value = replies[0]
        assert review_checkpoint_variant(task, workflow) == paths[1]
