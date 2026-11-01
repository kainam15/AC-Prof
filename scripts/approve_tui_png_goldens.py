"""Turn an owner-approved GitHub Issue into a scoped Golden PNG update PR.

Only trusted main-branch CI artifacts are accepted. The public HTML never receives
credentials and no Issue text is interpolated into a shell command.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ISSUE_TITLE = "[TUI Golden] Approve selected screenshots"
ISSUE_MARKER = "<!-- acprof-tui-golden-review-v1"
SOURCE_WORKFLOW = ".github/workflows/tui-png.yml"
ARTIFACT_NAME = "tui-png-visual"
SCENE_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
VALID_STATUSES = {"visual-regression", "size-mismatch"}
MAX_PNG_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class ApprovalRequest:
    issue_number: int
    run_id: int
    head_sha: str
    scenes: tuple[str, ...]


def parse_issue(event: dict[str, Any], repository: str) -> ApprovalRequest:
    """Reject everyone but the repository owner, even on a public repository."""
    if (
        event.get("action") != "opened"
        or event.get("repository", {}).get("full_name") != repository
    ):
        raise ValueError("Not an opened Issue in the requested repository")
    if "/" not in repository:
        raise ValueError("Invalid repository")
    owner = repository.split("/", 1)[0]
    issue = event.get("issue", {})
    author = issue.get("user", {}).get("login")
    if not isinstance(author, str) or author.casefold() != owner.casefold():
        raise ValueError("Only the repository owner may approve Golden snapshots")
    if "pull_request" in issue or issue.get("title") != ISSUE_TITLE:
        raise ValueError("Not a Golden approval Issue")
    number = issue.get("number")
    if type(number) is not int or number <= 0:
        raise ValueError("Invalid Issue number")
    body = issue.get("body")
    if not isinstance(body, str) or len(body) > 8000:
        raise ValueError("Invalid approval body")
    match = re.match(r"\A<!-- acprof-tui-golden-review-v1\r?\n([^\r\n]+)\r?\n-->", body)
    if not match:
        raise ValueError("Missing structured approval marker")
    try:
        request = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid approval JSON") from exc
    if not isinstance(request, dict) or set(request) != {"version", "run_id", "head_sha", "scenes"}:
        raise ValueError("Invalid approval fields")
    if request["version"] != 1 or type(request["version"]) is not int:
        raise ValueError("Unsupported approval version")
    run_id = request["run_id"]
    head_sha = request["head_sha"]
    scenes = request["scenes"]
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("Invalid source run ID")
    if not isinstance(head_sha, str) or SHA_PATTERN.fullmatch(head_sha) is None:
        raise ValueError("Invalid source commit SHA")
    if (
        not isinstance(scenes, list)
        or not 1 <= len(scenes) <= 20
        or any(
            not isinstance(scene, str) or SCENE_PATTERN.fullmatch(scene) is None for scene in scenes
        )
        or len(set(scenes)) != len(scenes)
    ):
        raise ValueError("Invalid or repeated scene selection")
    return ApprovalRequest(
        issue_number=number, run_id=run_id, head_sha=head_sha, scenes=tuple(scenes)
    )


def validate_run(run: dict[str, Any], request: ApprovalRequest, repository: str) -> None:
    """Prevent PR/fork artifacts, stale baselines, or another workflow's output."""
    checks = {
        "id": run.get("id") == request.run_id,
        "source workflow": run.get("path", "").split("@", 1)[0] == SOURCE_WORKFLOW,
        "source repository": run.get("head_repository", {}).get("full_name") == repository,
        "main branch": run.get("head_branch") == "main",
        "main event": run.get("event") in {"push", "workflow_dispatch"},
        "source SHA": run.get("head_sha") == request.head_sha,
        "run completed": run.get("status") == "completed",
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("Untrusted source CI run: " + ", ".join(failures))


def _checked_png(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PNG_BYTES:
        raise ValueError(f"Missing, linked, or oversized screenshot: {path.name}")
    data = path.read_bytes()
    if (
        len(data) < 45
        or not data.startswith(b"\x89PNG\r\n\x1a\n")
        or data[12:16] != b"IHDR"
        or struct.unpack(">I", data[8:12])[0] != 13
    ):
        raise ValueError(f"Invalid PNG screenshot: {path.name}")
    width, height = struct.unpack(">II", data[16:24])
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError(f"Unexpected PNG dimensions: {path.name}")
    return data


def stage_goldens(request: ApprovalRequest, *, artifact: Path, checkout: Path) -> list[Path]:
    """Copy only changed, selected, valid screenshots from that exact CI artifact."""
    actual = artifact / "internal-testing" / "shared" / "tui-png"
    report = actual / "report.json"
    if report.is_symlink() or not report.is_file() or report.stat().st_size > 100_000:
        raise ValueError("The source run has no trustworthy PNG comparison report")
    rows = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Malformed comparison report")
    statuses: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("scene"), str):
            raise ValueError("Malformed comparison result")
        if row["scene"] in statuses:
            raise ValueError("Duplicated comparison scene")
        statuses[row["scene"]] = row.get("status", "")
    target_dir = checkout / "tests" / "visual_png" / "__snapshots__"
    replacements: list[tuple[Path, bytes]] = []
    for scene in request.scenes:
        if statuses.get(scene) not in VALID_STATUSES:
            raise ValueError(f"Scene is not a completed visual regression: {scene}")
        target = target_dir / f"{scene}.png"
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"Golden baseline missing: {scene}")
        snapshot = _checked_png(actual / f"{scene}.png")
        if target.read_bytes() != snapshot:
            replacements.append((target, snapshot))
    if not replacements:
        raise ValueError("No selected Golden baseline would change")
    for target, snapshot in replacements:
        target.write_bytes(snapshot)
    return [path for path, _ in replacements]


def command(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(list(args), cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def run_approval() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    checkout = Path.cwd()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    request = parse_issue(event, repository)
    # Fetch remote main explicitly; no stale report may silently replace newer goldens.
    command("git", "fetch", "--no-tags", "origin", "main", cwd=checkout)
    latest_main = command("git", "rev-parse", "FETCH_HEAD", cwd=checkout, capture=True)
    local_head = command("git", "rev-parse", "HEAD", cwd=checkout, capture=True)
    if latest_main != request.head_sha or local_head != request.head_sha:
        raise ValueError(
            "The published report is stale; re-run TUI PNG Visual on current main and retry."
        )
    run = json.loads(
        command("gh", "api", f"repos/{repository}/actions/runs/{request.run_id}", capture=True)
    )
    validate_run(run, request, repository)
    branch = f"chore/tui-golden-review-issue-{request.issue_number}"
    with tempfile.TemporaryDirectory(
        prefix="acprof-golden-", dir=os.environ.get("RUNNER_TEMP")
    ) as tmp:
        command(
            "gh",
            "run",
            "download",
            str(request.run_id),
            "--repo",
            repository,
            "--name",
            ARTIFACT_NAME,
            "--dir",
            tmp,
        )
        # Attach the checked-out source commit to a named branch before editing.
        command("git", "switch", "-c", branch, cwd=checkout)
        changed = stage_goldens(request, artifact=Path(tmp), checkout=checkout)
    paths = [path.relative_to(checkout).as_posix() for path in changed]
    command("git", "add", "--", *paths, cwd=checkout)
    command("git", "config", "user.name", "github-actions[bot]", cwd=checkout)
    command(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
        cwd=checkout,
    )
    command(
        "git",
        "commit",
        "-m",
        "test(tui): approve reviewed Golden PNG snapshots",
        "-m",
        "- Update owner-selected PNG baselines from a verified main-branch CI run\n"
        "- Preserve independent visual review through a pull request\n"
        "- Leave all other snapshots unchanged",
        cwd=checkout,
    )
    command("git", "push", "origin", f"HEAD:refs/heads/{branch}", cwd=checkout)
    scene_list = ", ".join(f"`{scene}`" for scene in request.scenes)
    pr_url = command(
        "gh",
        "pr",
        "create",
        "--repo",
        repository,
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        "test(tui): approve reviewed Golden PNG snapshots",
        "--body",
        f"Owner-approved in #{request.issue_number}.\n\n"
        f"Source: https://github.com/{repository}/actions/runs/{request.run_id}\n\n"
        f"Selected scenes: {scene_list}\n\n"
        "Review the PNG changes and run TUI PNG Visual before merging. "
        "GitHub may require approval to run checks on bot-created PRs.",
        capture=True,
    )
    print(f"Created Golden update pull request: {pr_url}")
    command(
        "gh",
        "issue",
        "comment",
        str(request.issue_number),
        "--repo",
        repository,
        "--body",
        f"Golden update PR created: {pr_url}",
    )
    command(
        "gh",
        "issue",
        "close",
        str(request.issue_number),
        "--repo",
        repository,
        "--reason",
        "completed",
    )


if __name__ == "__main__":
    run_approval()
