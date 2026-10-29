"""Summary statistics and quality must describe one unchanged source snapshot."""
import asyncio
import csv
import fcntl
import hashlib
import json
import os
import shutil
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest
from tests.results.storage.test_sequential_result_reads import artifact_hashes, snapshot_store

from acprof.analysis import audit
from acprof.config import CSV_FIELDS
from acprof.result_layers import LAYER_FILES, publish_result_rows
from acprof.tui import diagnostics


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "result_all.csv"
    fields = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx",
              "status", "latency_app_s", "environment_class", "note"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, value in enumerate(("0.01", "0.03")):
            writer.writerow(dict(zip(fields, ("2", "8", "off", "32", "0", str(index), "ok",
                                              value, "native_linux", '诊断,"quoted"\r\nnext'), strict=True)))
    return path


def test_summary_parses_once_and_only_rereads_source_for_verification(source, monkeypatch):
    original = source.read_bytes()
    opened = []
    real_open = Path.open

    def observe(path, *args, **kwargs):
        if path == source:
            opened.append(args[0] if args else kwargs.get("mode", "r"))
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", observe)
        summary = diagnostics.summarize_result_csv(source)
    assert opened == ["rb", "rb"]
    assert summary.rows == summary.ok_rows == 2
    assert summary.groups[0]["mean"] == pytest.approx(0.02)
    assert source.read_bytes() == original


def test_audit_receives_the_exact_snapshot_used_by_statistics(source, monkeypatch):
    original = source.read_bytes()
    real_audit = audit.audit_result
    observed = []

    def check_snapshot(path, **kwargs):
        snapshot = kwargs.get("result_snapshot")
        assert snapshot is not None, "audit opened an independent result snapshot"
        assert snapshot.sha256 == hashlib.sha256(original).hexdigest()
        assert [row["latency_app_s"] for row in snapshot.rows] == ["0.01", "0.03"]
        observed.append(snapshot)
        return real_audit(path, **kwargs)

    monkeypatch.setattr(audit, "audit_result", check_snapshot)
    diagnostics.summarize_result_csv(source)
    assert len(observed) == 1


def publish_v1_manifest(root, manifest):
    for layer, info in manifest["layers"].items():
        destination = root / LAYER_FILES[layer]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / info["path"], destination)
        info["path"] = LAYER_FILES[layer]
    manifest["schema_version"] = 1
    manifest.pop("generation")
    (root / "result_layers.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("directory_input", [False, True])
def test_layered_summary_uses_rows_without_reencoding_csv(tmp_path, monkeypatch, version, directory_input):
    layout, rows = snapshot_store(tmp_path / "current", count=2, repetitions=1)
    source = layout.result_csv
    if version != 3:
        root = tmp_path / "snapshot"
        manifest = publish_result_rows(list(CSV_FIELDS), rows, root)
        source = root / "result_layers.json"
        if version == 1:
            publish_v1_manifest(root, manifest)
    original = artifact_hashes(layout)

    def forbidden(_text):
        pytest.fail("summary re-encoded parsed layers as another CSV")

    monkeypatch.setattr("acprof.result_csv._CsvRowEncoder.write", staticmethod(forbidden))
    summary = diagnostics.summarize_result_csv(source.parent if directory_input else source)
    assert summary.rows == summary.ok_rows == 2
    assert artifact_hashes(layout) == original


@pytest.mark.parametrize("operation", ["overwrite", "replace", "remove"])
@pytest.mark.parametrize("phase", ["statistics", "audit"])
def test_source_changes_are_rejected_before_returning_summary(source, monkeypatch, phase, operation):
    stamp = source.stat()
    replacement = source.read_bytes().replace(b"0.01", b"9.99")
    owner, name = (diagnostics, "summarize_windows") if phase == "statistics" else (audit, "audit_result")
    original = getattr(owner, name)

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if operation == "remove":
            source.unlink()
        elif operation == "replace":
            temporary = source.with_suffix(".new")
            temporary.write_bytes(replacement)
            temporary.replace(source)
        else:
            source.write_bytes(replacement)
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return result

    monkeypatch.setattr(owner, name, mutate)
    with pytest.raises(ValueError, match="快照|snapshot|changed"):
        diagnostics.summarize_result_csv(source)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layer_changes_are_rejected_with_unchanged_manifest(tmp_path, monkeypatch, version):
    layout, rows = snapshot_store(tmp_path / "current", count=2, repetitions=1)
    source = layout.result_csv
    if version != 3:
        root = tmp_path / "snapshot"
        manifest = publish_result_rows(list(CSV_FIELDS), rows, root)
        source = root / "result_layers.json"
        if version == 1:
            publish_v1_manifest(root, manifest)
        layer = root / manifest["layers"]["summary"]["path"]
    else:
        layer = layout.layer_csv("summary")
    original_manifest = source.read_bytes()
    summarize = diagnostics.summarize_windows

    def change_layer(*args, **kwargs):
        result = summarize(*args, **kwargs)
        layer.write_bytes(b"corrupted after statistics were computed\n")
        return result

    monkeypatch.setattr(diagnostics, "summarize_windows", change_layer)
    with pytest.raises(ValueError, match="快照|snapshot|changed"):
        diagnostics.summarize_result_csv(source)
    assert source.read_bytes() == original_manifest


def test_cancelled_summary_does_not_open_source(source, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("cancelled summary still started I/O")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(CancelledError):
        diagnostics.summarize_result_csv(source, cancelled=lambda: True)


def test_retargeted_source_symlink_does_not_label_an_old_snapshot_as_current(source, tmp_path, monkeypatch):
    alias = tmp_path / "selected.csv"
    alias.symlink_to(source.name)
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(source.read_bytes().replace(b"0.01", b"9.99"))
    summarize = diagnostics.summarize_windows

    def retarget(*args, **kwargs):
        result = summarize(*args, **kwargs)
        alias.unlink()
        alias.symlink_to(replacement.name)
        return result

    monkeypatch.setattr(diagnostics, "summarize_windows", retarget)
    with pytest.raises(ValueError, match="快照|snapshot|changed"):
        diagnostics.summarize_result_csv(alias)


@pytest.mark.parametrize("phase", ["statistics", "audit"])
def test_cancelled_work_does_not_produce_a_summary(source, monkeypatch, phase):
    cancelled = Event()
    owner, name = (diagnostics, "summarize_windows") if phase == "statistics" else (audit, "audit_result")
    original = getattr(owner, name)

    def cancel_after(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(owner, name, cancel_after)
    if phase == "statistics":
        def forbidden(*_args, **_kwargs):
            pytest.fail("audit continued after cancellation")
        monkeypatch.setattr(audit, "audit_result", forbidden)
    with pytest.raises(CancelledError):
        diagnostics.summarize_result_csv(source, cancelled=cancelled.is_set)


@pytest.mark.parametrize(("language", "failed"), [("zh", "结果快照"), ("en", "result snapshot")])
async def test_changed_source_error_and_retry_remain_usable_at_80x24(
    source, tmp_path, monkeypatch, language, failed,
):
    from textual.widgets import Button, Input, Static
    from tests.support.tui_fixtures import AcprofTui

    from acprof.tui.settings import TuiSettings, UiPreferences, save_settings

    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    summarize = diagnostics.summarize_windows

    def change_result(*args, **kwargs):
        result = summarize(*args, **kwargs)
        source.write_bytes(source.read_bytes().replace(b"0.01", b"0.05"))
        return result

    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("results-tab")
        app.query_one("#result-csv", Input).value = str(source)
        await pilot.pause()
        button = app.query_one("#summarize-results", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        with monkeypatch.context() as scoped:
            scoped.setattr(diagnostics, "summarize_windows", change_result)
            assert await pilot.click(button)
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert failed in str(app.query_one("#result-summary", Static).content)
        assert not app.is_busy()
        assert not button.disabled
        assert app.query_one("#result-csv", Input).value == str(source)
        assert await pilot.click(button)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "40 ms" in str(app.query_one("#result-summary", Static).content)
        assert not app.is_busy()
        assert not button.disabled


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_selection_change_releases_a_waiting_layer_reader_at_80x24(source, tmp_path, monkeypatch, language):
    from tests.runtime.containers.test_layer_read_cancellation import source_hashes
    from tests.results.storage.test_result_row_assembly import row_store
    from textual.widgets import Button, Input, Static
    from tests.support.tui_fixtures import AcprofTui

    from acprof.artifact_layout import ArtifactLayout
    from acprof.result_store.protocol import publication_lock
    from acprof.tui.settings import TuiSettings, UiPreferences, save_settings

    blocked = tmp_path / "blocked"
    row_store(blocked, 3, count=1, width=1)
    before = source_hashes(blocked)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    attempted = Event()
    original_flock = fcntl.flock

    def observe(descriptor, flags):
        if flags & fcntl.LOCK_SH:
            attempted.set()
        return original_flock(descriptor, flags)

    monkeypatch.setattr(fcntl, "flock", observe)
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("results-tab")
        selector = app.query_one("#result-csv", Input)
        selector.value = str(blocked)
        await pilot.pause()
        button = app.query_one("#summarize-results", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        with publication_lock(ArtifactLayout(blocked), exclusive=True):
            assert await pilot.click(button)
            assert await asyncio.to_thread(attempted.wait, 2)
            assert app.read_jobs
            selector.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(selector)
            await pilot.press("ctrl+a")
            await pilot.press(*str(source))
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert not app.read_jobs
            assert not app.is_busy()
            assert selector.value == str(source)
            assert not button.disabled
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(button)
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=2)
            await pilot.pause()
            assert "20 ms" in str(app.query_one("#result-summary", Static).content)
            assert not app.is_busy()
    assert source_hashes(blocked) == before
