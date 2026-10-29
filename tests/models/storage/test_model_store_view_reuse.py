"""Published loader views remain usable without the SDK's redundant snapshot index."""
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.widgets import Input

from acprof.container.download_model import verify_download
from acprof.container.model_files import PLAN_FILENAME, plan_download, seal_plan
from acprof.experiment import RunConfig
from acprof.host.model_store import store as model_store
from acprof.host.network_preflight import preflight
from acprof.network_policy import DownloadPolicyError
from acprof.tui.experiment_picker import SearchPickerScreen
from acprof.tui.model_candidates import cached_candidates
from acprof.tui.settings import TuiSettings, UiPreferences, save_settings
from tests.support.tui_fixtures import AcprofTui

CONTENT = b"{}"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("ACPROF_MAX_DOWNLOAD", "0")
    monkeypatch.setenv("ACPROF_MODEL_MAX_DOWNLOAD", "0")
    plan = None
    snapshots = {}
    for model_id, revision in (("example/model", "a" * 40), ("example/base", "b" * 40)):
        repo = plan_download(model_id=model_id, revision=revision, family="nlp",
            backend="transformers_pipeline", policy="full",
            files={"config.json": {"size": 2, "lfs_sha256": DIGEST}}, read_json=lambda _: {})
        repo["endpoint"] = "https://huggingface.co"
        seal_plan(repo)
        if plan is None:
            plan = repo
        else:
            plan["dependencies"] = [{"repo_id": model_id, "revision": revision, "download": repo}]
            plan["total_selected_bytes"] = 4
            seal_plan(plan)
        cache = tmp_path / "hf" / ("models--" + model_id.replace("/", "--"))
        blob = cache / "blobs" / DIGEST
        blob.parent.mkdir(parents=True)
        blob.write_bytes(CONTENT)
        snapshot = cache / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").symlink_to(os.path.relpath(blob, snapshot))
        snapshots[model_id] = snapshot
    task = SimpleNamespace(model_id=plan["model_id"], model_revision=plan["model_revision"],
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default",
        model_download_policy="full", model_resolution={},
        model_spec={"dependencies": [{"repo_id": "example/base", "revision": "b" * 40}]})
    with patch("huggingface_hub.snapshot_download", side_effect=lambda **kw: str(snapshots[kw["repo_id"]])):
        receipt = model_store.prepare_model(task, plan, tmp_path)
    for snapshot in snapshots.values():
        (snapshot / "config.json").unlink()
    model_store.verify_entry({"model_store": receipt}, tmp_path)
    return task, receipt, tmp_path


def test_preflight_credits_complete_view_without_source_index_or_model_requests(prepared):
    task, receipt, root = prepared
    plan = copy.deepcopy(receipt["model_download"])
    with patch("acprof.host.download_probe.probe_hf_file", side_effect=AssertionError("unexpected probe")) as probe:
        report = preflight(task, None, root, plan, root=root, runtime_plan=([], {"kind": "process"}))
    probe.assert_not_called()
    assert report["expected_download_bytes"] == 0
    assert report["model"]["cached_bytes"] == 4
    assert report["disk"]["required_bytes"] == 0
    assert report["network_probe"] == []
    assert [(row["cache_status"], row["cached_bytes"]) for row in report["sources"]] == [("hit", 2), ("hit", 2)]
    assert plan == receipt["model_download"]


def test_preparation_verifies_actual_view_once_without_sdk_or_republication(prepared, capsys):
    task, receipt, root = prepared
    entry = root / "entries" / receipt["entry_id"]
    before = (entry / PLAN_FILENAME).read_bytes()
    capsys.readouterr()
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected SDK")) as sdk, \
            patch.object(model_store, "publish_view", side_effect=AssertionError("unexpected publication")) as publish, \
            patch("acprof.container.download_model.verify_download", wraps=verify_download) as verify:
        reused = model_store.prepare_model(task, receipt["model_download"], root, planned_download_bytes=0)
    sdk.assert_not_called()
    publish.assert_not_called()
    assert reused == receipt
    assert (entry / PLAN_FILENAME).read_bytes() == before
    assert [Path(call.args[0]) for call in verify.call_args_list] == [
        entry / "hf/models--example--model/snapshots" / ("a" * 40),
        entry / "hf/models--example--base/snapshots" / ("b" * 40),
    ]
    output = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("[network-download] "))
    stats = json.loads(output.split(" ", 1)[1])
    assert stats["verified_new_payload_bytes"] == 0
    assert stats["cache_savings_bytes"] == 4
    assert stats["wire_bytes"] is None


def test_candidate_scan_uses_recorded_entry_id_and_stays_read_only(prepared):
    task, receipt, root = prepared
    entry = root / "entries" / receipt["entry_id"]
    historical_key = "f" * 64
    entry.rename(root / "entries" / historical_key)
    with patch.object(model_store, "store_lock", side_effect=AssertionError("unexpected lock")), \
            patch("acprof.container.download_model.verify_download", side_effect=AssertionError("unexpected payload hash")):
        candidates, warnings = cached_candidates(root)
    assert warnings == ()
    assert [(item.model_id, item.status) for item in candidates] == [(task.model_id, "cached_unverified")]
    assert json.loads(candidates[0].detail)["cache_entry"] == str(root / "entries" / historical_key)
    assert json.loads(candidates[0].detail)["inference_verified"] is False


@pytest.mark.parametrize("field", ["source", "model_revision", "endpoint"])
@pytest.mark.parametrize("reseal", [False, True])
def test_different_plan_cannot_claim_another_entry_view(prepared, field, reseal):
    _, receipt, root = prepared
    plan = copy.deepcopy(receipt["model_download"])
    plan[field] = {"source": "modelscope", "model_revision": "c" * 40, "endpoint": "https://hf-mirror.com"}[field]
    if reseal:
        seal_plan(plan)
    sources = model_store.model_sources(plan, root, entry_id=receipt["entry_id"])
    assert [source.estimated_bytes for source in sources] == [2, 2]
    assert [source.cached_bytes for source in sources] == [0, 0]


def test_missing_view_payload_is_rechecked_before_zero_budget_preparation(prepared):
    task, receipt, root = prepared
    report = preflight(task, None, root, receipt["model_download"], root=root, runtime_plan=([], {"kind": "process"}))
    assert report["expected_download_bytes"] == 0
    entry = root / "entries" / receipt["entry_id"]
    link = entry / "hf/models--example--base/snapshots" / ("b" * 40) / "config.json"
    link.resolve().unlink()
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected SDK")) as sdk:
        with pytest.raises(DownloadPolicyError):
            model_store.prepare_model(task, receipt["model_download"], root,
                                      planned_download_bytes=report["expected_download_bytes"])
    sdk.assert_not_called()
    assert model_store.read_entry(receipt["entry_id"], root) == receipt["model_download"]


def test_complete_view_still_rejects_same_size_corrupt_payload(prepared):
    task, receipt, root = prepared
    entry = root / "entries" / receipt["entry_id"]
    link = entry / "hf/models--example--base/snapshots" / ("b" * 40) / "config.json"
    link.resolve().write_bytes(b"xx")
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected SDK")) as sdk:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            model_store.prepare_model(task, receipt["model_download"], root, planned_download_bytes=0)
    sdk.assert_not_called()
    assert model_store.read_entry(receipt["entry_id"], root) == receipt["model_download"]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_cached_view_can_be_selected_at_80x24_without_source_index(prepared, language):
    task, _, root = prepared
    settings = root / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), root)
    config = replace(RunConfig.smoke("example/new"), output_dir=str(root / "results"), model_store=str(root))
    app = AcprofTui(config, settings_path=settings)
    with patch("acprof.tui.model_candidates.current_context", return_value={"warnings": []}):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.query_one("#model", Input).focus()
            await pilot.press("f4")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, SearchPickerScreen)
            assert len(app.screen.choices) == 1
            app.screen.query_one("#picker-search", Input).focus()
            await pilot.press("ctrl+a", "backspace")
            await pilot.pause()
            candidate = app.screen.filtered[0].value
            assert (candidate.model_id, candidate.status) == (task.model_id, "cached_unverified")
            assert await pilot.click("#picker-use")
            await pilot.pause()
            assert app.query_one("#model", Input).value == task.model_id
            assert app.query_one("#revision", Input).value == task.model_revision
