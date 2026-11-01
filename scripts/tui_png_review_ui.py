"""Static review controls for the TUI PNG report (no GitHub credentials in HTML)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

REPO_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class ReviewContext:
    repository: str
    run_id: int
    head_sha: str

    @classmethod
    def from_environment(cls) -> ReviewContext | None:
        """Only a main-branch CI run can offer a GitHub approval request."""
        if os.environ.get("GITHUB_REF") != "refs/heads/main" or os.environ.get(
            "GITHUB_EVENT_NAME"
        ) not in {"push", "workflow_dispatch"}:
            return None
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        head_sha = os.environ.get("GITHUB_SHA", "")
        if (
            REPO_PATTERN.fullmatch(repository) is None
            or not run_id.isascii()
            or not run_id.isdecimal()
            or int(run_id) <= 0
            or SHA_PATTERN.fullmatch(head_sha) is None
        ):
            return None
        return cls(repository=repository, run_id=int(run_id), head_sha=head_sha)


REVIEW_STYLE = """
.golden-review{margin:16px 0;padding:18px;border:1px solid #674b86;background:#262037;border-radius:10px}
.golden-review h3{margin:0 0 8px;color:#e2c2ff}
.golden-review p{margin:8px 0;color:#d2cee0}
.golden-review .review-buttons{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.golden-review button{padding:8px 12px;border-radius:7px;background:#423451;color:#f1e9fc;border:1px solid #8f6cad;cursor:pointer}
.golden-review button.primary{background:#6c4393;color:white}
.golden-review button:disabled{opacity:.45;cursor:not-allowed}
.golden-choice{display:flex;align-items:center;gap:9px;font-weight:600;color:#e9d0ff;cursor:pointer}
.golden-choice input{width:18px;height:18px;accent-color:#b68af1}
.golden-choice-wrap{margin:0 18px 16px}
.golden-review small{display:block;color:#bdb2ca}
"""

REVIEW_SCRIPT = """
const goldenPanel = document.querySelector('[data-golden-review-panel]');
if (goldenPanel) {
  const choices = Array.from(document.querySelectorAll('[data-golden-scene]'));
  const submit = goldenPanel.querySelector('[data-golden-submit]');
  const count = goldenPanel.querySelector('[data-golden-count]');
  const refresh = () => {
    const n = choices.filter(input => input.checked).length;
    count.textContent = String(n);
    submit.disabled = n === 0;
  };
  choices.forEach(input => input.addEventListener('change', refresh));
  goldenPanel.querySelector('[data-golden-select-all]').addEventListener('click', () => {
    choices.forEach(input => { input.checked = true; });
    refresh();
  });
  goldenPanel.querySelector('[data-golden-clear]').addEventListener('click', () => {
    choices.forEach(input => { input.checked = false; });
    refresh();
  });
  submit.addEventListener('click', () => {
    const scenes = choices.filter(input => input.checked).map(input => input.dataset.goldenScene).sort();
    if (!scenes.length) return;
    const repository = goldenPanel.dataset.repository;
    const runId = Number(goldenPanel.dataset.runId);
    const headSha = goldenPanel.dataset.headSha;
    const request = {version: 1, run_id: runId, head_sha: headSha, scenes};
    const body = '<!-- acprof-tui-golden-review-v1\\n' +
      JSON.stringify(request) +
      '\\n-->\\n\\nI reviewed and approve these Golden PNG changes: ' +
      scenes.map(s => '`' + s + '`').join(', ') +
      '\\n\\nCI evidence: https://github.com/' + repository + '/actions/runs/' + runId +
      '\\n\\nAn owner-authenticated workflow will verify the source run and open a PR. ' +
      'No baselines are changed by merely creating this link.';
    const url = new URL('https://github.com/' + repository + '/issues/new');
    url.searchParams.set('title', '[TUI Golden] Approve selected screenshots');
    url.searchParams.set('body', body);
    window.open(url.toString(), '_blank', 'noopener,noreferrer');
  });
  refresh();
}
"""
