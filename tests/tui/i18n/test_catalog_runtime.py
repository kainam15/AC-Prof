"""Keep runtime lookup unambiguous and independent of UI or inference imports."""
import subprocess
import sys
from pathlib import Path

import pytest

from acprof.tui.catalog import MESSAGES, merge_messages, source_indexes
from acprof.tui.catalog.entry import Entry
from scripts.check_tui_catalog import check_catalog


def test_runtime_merge_rejects_cross_module_duplicate_keys():
    left = {"common.action.cancel": Entry(zh="取消", en="Cancel")}
    right = {"common.action.cancel": Entry(zh="撤销", en="Undo")}
    with pytest.raises(ValueError, match="Duplicate message key: common.action.cancel"):
        merge_messages(left, right)


def test_source_indexes_reject_ambiguous_english_translation():
    messages = {
        "files.action.open": Entry(zh="打开", en="Open"),
        "panel.action.expand": Entry(zh="打开", en="Expand"),
    }
    with pytest.raises(ValueError, match="Conflicting source translations"):
        source_indexes(messages)


def test_distinct_contexts_can_share_the_same_translation():
    messages = {
        "files.action.open": Entry(zh="打开", en="Open"),
        "logs.action.open": Entry(zh="打开", en="Open"),
    }
    source_to_id, english = source_indexes(messages)
    assert source_to_id["打开"] in messages
    assert english == {"打开": "Open"}
    assert set(merge_messages(messages)) == set(messages)


def test_all_catalog_modules_are_included_in_runtime_lookup():
    root = Path(__file__).resolve().parents[3]
    entries = check_catalog(root).entries
    assert set(MESSAGES) == set(entries)
    assert {key: (entry.zh, entry.en) for key, entry in MESSAGES.items()} == {
        key: (entry.zh, entry.en) for key, entry in entries.items()
    }


def test_catalog_import_does_not_load_tui_or_inference_dependencies():
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
import acprof.tui.catalog
import acprof.tui.i18n
heavy = {name.split('.')[0] for name in sys.modules} & {'textual', 'torch', 'numpy', 'pandas'}
assert not heavy, sorted(heavy)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True,
        text=True, check=False, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_bindings_and_error_entries_translate_cleanly_to_english():
    from acprof.tui.i18n import message_id, translate

    for source in ("全选文本", "快捷命令", "快捷键", "基本/高级配置", "全选日志"):
        rendered = translate(source, "en")
        assert rendered != source
        assert all(ord(ch) < 128 for ch in rendered)

    for msg in (
        message_id("run.recovery.missing_config"),
        message_id("results.profile.layered_dir_only"),
        message_id("images.dependencies.query_failed", "OSError"),
        message_id("validation.settings.unsupported_version_detail", "'3'", 4),
        message_id("validation.settings.retired_cgroup_v1"),
    ):
        rendered = translate(msg, "en")
        assert rendered != str(msg)
        assert not any("\u3400" <= ch <= "\u9fff" for ch in rendered)
