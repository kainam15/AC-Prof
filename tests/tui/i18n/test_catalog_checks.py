"""Exercise catalog failures before application imports can hide dictionary duplicates."""
from pathlib import Path

import pytest

from scripts.check_tui_catalog import check_catalog, main


def write(root, path, source):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def catalog(root, entries, module="common"):
    write(root, f"acprof/tui/catalog/{module}.py", f"MESSAGES = {{\n{entries}\n}}\n")


def codes(report):
    return {issue.code for issue in report.errors}


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ('"ui.1234": Entry(zh="取消", en="Cancel"),', "invalid-key"),
        ('"common.action.cancel": Entry(zh="取消"),', "entry-shape"),
        ('"common.action.cancel": Entry(zh="取消", en=""),', "empty-translation"),
        ('"common.action.cancel": Entry(zh="取消", en="  "),', "empty-translation"),
        ('"run.progress.done": Entry(zh="完成 {0:.2f}", en="Done {0:.3f}"),',
         "format-mismatch"),
        ('"run.progress.done": Entry(zh="完成 {0}", en="Done {1}"),', "format-mismatch"),
        ('"run.progress.done": Entry(zh="完成 {", en="Done"),', "invalid-format"),
        ('"run.progress.done": Entry(zh="完成 {0!z}", en="Done {0!z}"),', "invalid-format"),
        ('"run.progress.done": Entry(zh="完成 {} {0}", en="Done {} {0}"),', "invalid-format"),
    ],
)
def test_invalid_catalog_entries_fail(tmp_path, entry, code):
    catalog(tmp_path, entry)
    report = check_catalog(tmp_path)
    issue = next(issue for issue in report.errors if issue.code == code)
    assert issue.path == Path("acprof/tui/catalog/common.py")
    assert issue.line == 2


def test_missing_and_empty_catalogs_fail(tmp_path):
    assert "missing-catalog" in codes(check_catalog(tmp_path))
    write(tmp_path, "acprof/tui/catalog/entry.py", "class Entry: pass\n")
    write(tmp_path, "acprof/tui/catalog/__init__.py", "MESSAGES = merge_messages()\n")
    assert "empty-catalog" in codes(check_catalog(tmp_path))


@pytest.mark.parametrize("other_module", [False, True])
def test_duplicate_keys_are_detected_before_dictionary_overwrite(tmp_path, other_module):
    entry = '"common.action.cancel": Entry(zh="取消", en="Cancel"),'
    catalog(tmp_path, entry if other_module else f"{entry}\n{entry}")
    if other_module:
        catalog(tmp_path, entry, "settings")
    report = check_catalog(tmp_path)
    duplicate = next(issue for issue in report.errors if issue.code == "duplicate-key")
    assert "common.action.cancel" in duplicate.detail
    assert "first defined" in duplicate.detail


def test_source_ambiguity_fails_but_equal_translation_is_valid(tmp_path):
    catalog(tmp_path, '"common.action.open": Entry(zh="打开", en="Open"),')
    catalog(tmp_path, '"logs.file.open": Entry(zh="打开", en="Expand"),', "logs")
    assert "source-conflict" in codes(check_catalog(tmp_path))
    catalog(tmp_path, '"logs.file.open": Entry(zh="打开", en="Open"),', "logs")
    assert not check_catalog(tmp_path).errors


def test_catalog_order_is_checked(tmp_path):
    catalog(
        tmp_path,
        '"common.action.save": Entry(zh="保存", en="Save"),\n'
        '"common.action.cancel": Entry(zh="取消", en="Cancel"),',
    )
    assert "unsorted-keys" in codes(check_catalog(tmp_path))


def test_aliases_and_explicit_maps_are_checked_without_matching_unrelated_functions(tmp_path):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", '''
from acprof.tui.i18n import message_id as label
from acprof.tui import i18n as translated
LABELS = {"ok": "run.stage.done", "bad": "run.stage.missing"}
def render(state):
    key = LABELS.get(state)
    return label(key)
translated.message_id("run.stage.another_missing")
''')
    write(tmp_path, "acprof/unrelated.py", '''
def message_id(value):
    return value
message_id("not.a.translation")
''')
    report = check_catalog(tmp_path)
    assert {issue.detail for issue in report.errors} == {
        "run.stage.missing", "run.stage.another_missing",
    }
    assert not report.dynamic
    assert not report.unused


def test_unused_analysis_counts_source_fallback_but_not_tests_or_docstrings(tmp_path):
    catalog(
        tmp_path,
        '"common.action.cancel": Entry(zh="取消", en="Cancel"),\n'
        '"common.action.save": Entry(zh="保存", en="Save"),\n'
        '"run.progress.done": Entry(zh="完成 {0}", en="Done {0}"),',
    )
    write(tmp_path, "acprof/tui/panel.py", '''
"""common.action.save"""
from acprof.tui.i18n import message_id
message_id("common.action.cancel")
''')
    write(tmp_path, "acprof/validation.py", '''
from acprof.messages import message
def validate():
    """保存"""
    return message("完成 {0}", 3)
''')
    write(tmp_path, "tests/test_old.py", 'assert "common.action.save" not in []\n')
    report = check_catalog(tmp_path)
    assert not report.errors
    assert [entry.key for entry in report.unused] == ["common.action.save"]


@pytest.mark.parametrize("call", ['label("未登记的新文案")', 'self.tr("未登记的新文案")'])
def test_unregistered_tui_source_literal_fails(tmp_path, call):
    catalog(tmp_path, '"common.action.cancel": Entry(zh="取消", en="Cancel"),')
    write(tmp_path, "acprof/tui/panel.py",
          f"from acprof.messages import message as label\n{call}\n")
    assert "literal-ui-message" in codes(check_catalog(tmp_path))


def test_raw_values_and_language_neutral_acronyms_are_allowed(tmp_path):
    catalog(tmp_path, '"common.resource.gpu": Entry(zh="GPU", en="GPU"),')
    write(tmp_path, "acprof/tui/panel.py", '''
def render(self, user_input):
    self.log.write("用户输入和原始日志")
    return self.tr(user_input)
''')
    assert not check_catalog(tmp_path).errors


def test_unresolved_dynamic_keys_are_visible_and_never_auto_deleted(tmp_path, capsys):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", '''
from acprof.tui.i18n import message_id
def render(state):
    return message_id(f"run.stage.{state}")
''')
    report = check_catalog(tmp_path)
    assert not report.errors
    assert len(report.dynamic) == 1
    before = (tmp_path / "acprof/tui/catalog/common.py").read_bytes()
    assert main(["--root", str(tmp_path), "--unused"]) == 0
    output = capsys.readouterr().out
    assert "dynamic-reference" in output
    assert "unused-candidate: run.stage.done" in output
    assert (tmp_path / "acprof/tui/catalog/common.py").read_bytes() == before


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('from acprof.tui.i18n import message_id as label\n'
         'label(msgid="run.stage.missing")\n', "unknown-key"),
        ('from acprof.messages import message\n'
         'message(template="未登记的新文案")\n', "literal-ui-message"),
        ('def render(self):\n'
         '    return self.tr(source="未登记的新文案")\n', "literal-ui-message"),
    ],
)
def test_keyword_arguments_are_checked(tmp_path, source, expected):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", source)
    assert expected in codes(check_catalog(tmp_path))


@pytest.mark.parametrize(
    "expression",
    [
        'LABELS["ok"]',
        'LABELS.get("ok", "not.a.translation")',
        'LABELS.get("missing", "run.stage.done")',
        '["run.stage.done", "not.a.translation"][0]',
    ],
)
def test_constant_lookups_only_check_reachable_values(tmp_path, expression):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", f'''
from acprof.tui.i18n import message_id
LABELS = {{"ok": "run.stage.done", "metadata": "not.a.translation"}}
message_id({expression})
''')
    report = check_catalog(tmp_path)
    assert not report.errors
    assert not report.dynamic


def test_repeated_assignments_are_explicitly_unresolved(tmp_path):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", '''
from acprof.tui.i18n import message_id
def render(failed):
    if failed:
        key = "run.stage.missing"
    else:
        key = "run.stage.done"
    return message_id(key)
''')
    report = check_catalog(tmp_path)
    assert not report.errors
    assert len(report.dynamic) == 1


def test_function_parameters_shadow_imports_and_mapping_constants(tmp_path):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", '''
from acprof.tui.i18n import message_id
LABELS = {"ok": "run.stage.done"}
def unrelated(message_id):
    return message_id("not.a.translation")
def render(LABELS):
    return message_id(LABELS["ok"])
''')
    report = check_catalog(tmp_path)
    assert not report.errors
    assert len(report.dynamic) == 1


def test_function_imports_do_not_leak_into_other_scopes(tmp_path):
    catalog(tmp_path, '"run.stage.done": Entry(zh="完成", en="Done"),')
    write(tmp_path, "acprof/tui/panel.py", '''
def render():
    from acprof.tui.i18n import message_id as label
    return label("run.stage.missing")
def unrelated(label):
    return label("not.a.translation")
''')
    report = check_catalog(tmp_path)
    assert [(issue.code, issue.detail) for issue in report.errors] == [
        ("unknown-key", "run.stage.missing"),
    ]


def test_automatic_placeholders_are_rejected_before_reordering_changes_types(tmp_path):
    catalog(
        tmp_path,
        '"run.progress.done": Entry(zh="文件 {}，耗时 {:.2f}", en="Elapsed {:.2f}; file {}"),',
    )
    assert "invalid-format" in codes(check_catalog(tmp_path))


def test_cli_returns_failure_for_invalid_catalog(tmp_path, capsys):
    catalog(tmp_path, '"common.action.cancel": Entry(zh="取消", en=""),')
    assert main(["--root", str(tmp_path)]) == 1
    assert "empty-translation" in capsys.readouterr().out


def test_project_catalog_has_no_hard_errors():
    root = Path(__file__).resolve().parents[3]
    report = check_catalog(root)
    assert report.entries
    assert not report.errors, "\n".join(map(str, report.errors))
