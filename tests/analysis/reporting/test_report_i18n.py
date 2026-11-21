"""Regression checks for offline HTML report i18n reason-label coverage."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORT_JS = ROOT / "acprof/plotting/report.js"
REPORT_I18N_JS = ROOT / "acprof/plotting/report_i18n.js"
_CJK = re.compile(r"[\u3400-\u9fff]")


def _extract_report_reason_labels() -> dict[str, str]:
    source = REPORT_JS.read_text(encoding="utf-8")
    match = re.search(r"const reasonLabels = \{(.*?)\};", source, re.DOTALL)
    assert match is not None, "reasonLabels definition missing from report.js"
    labels = dict(re.findall(r'([a-z0-9_]+)\s*:\s*"([^"]+)"', match.group(1)))
    assert len(labels) == 29
    return labels


def test_every_report_reason_label_has_an_english_translation() -> None:
    reason_labels = _extract_report_reason_labels()
    i18n_source = REPORT_I18N_JS.read_text(encoding="utf-8")
    missing = [
        f"{key}={label!r}"
        for key, label in reason_labels.items()
        if f'"{label}":' not in i18n_source
    ]
    assert not missing, f"Missing report_i18n.js translations: {missing}"


def test_report_i18n_translates_all_bare_counted_and_joined_reasons_in_node() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js runtime unavailable")

    reason_labels = list(_extract_report_reason_labels().values())
    counted = [f"{label} {index}" for index, label in enumerate(reason_labels, start=1)]
    joined = [
        "窗口不足 2 · 测量失败 1 · Process Runtime 不支持 4",
        "缺失窗口打断连续块 · 统计版本或口径不匹配",
    ]
    untouched = [
        "Qwen/Qwen3-0.6B",
        "foo · bar 1",
        "custom-实验-tag",
    ]
    inputs = [*reason_labels, *counted, *joined, *untouched]

    script = """
const fs = require("fs");
const vm = require("vm");
const [i18nPath, rawInputs] = process.argv.slice(1);
const inputs = JSON.parse(rawInputs);
const nodes = inputs.map(text => ({
  nodeValue: text,
  parentElement: { closest: () => null },
}));
global.window = global;
global.NodeFilter = { SHOW_TEXT: 4 };
global.document = {
  documentElement: { lang: "zh-CN" },
  body: {},
  createTreeWalker() {
    let idx = -1;
    return {
      currentNode: null,
      nextNode() {
        idx += 1;
        if (idx < nodes.length) {
          this.currentNode = nodes[idx];
          return true;
        }
        return false;
      },
    };
  },
};
vm.runInThisContext(fs.readFileSync(i18nPath, "utf8"), { filename: i18nPath });
window.ACProfI18n.apply("en");
const english = nodes.map(n => n.nodeValue);
window.ACProfI18n.apply("zh");
const restored = nodes.map(n => n.nodeValue);
process.stdout.write(JSON.stringify({ english, restored }));
"""
    result = subprocess.run(
        [node, "-e", script, str(REPORT_I18N_JS), json.dumps(inputs, ensure_ascii=False)],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)
    english = payload["english"]
    assert payload["restored"] == inputs

    reason_end = len(reason_labels) + len(counted) + len(joined)
    for original, translated in zip(inputs[:reason_end], english[:reason_end], strict=True):
        assert translated != original, f"Reason text stayed untranslated: {original}"
        assert not _CJK.search(translated), f"CJK leaked into English translation: {translated}"
    for label, translated in zip(reason_labels, english[len(reason_labels):len(reason_labels) + len(counted)], strict=True):
        assert translated.endswith(tuple(f" {i}" for i in range(1, len(reason_labels) + 1)))
        assert label not in translated

    assert english[reason_end:] == untouched
