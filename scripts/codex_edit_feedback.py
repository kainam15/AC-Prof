"""Codex command hook. Read one event from stdin and report actionable edit failures."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

# The direct script entry point must also work from a checkout subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.codex_edit_checks import is_checkable, run_checks

PATCH_PATH = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$", re.MULTILINE)


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], stdout=subprocess.PIPE, check=True,
    ).stdout


def git_root(directory: Path) -> Path:
    return Path(os.fsdecode(git(directory, "rev-parse", "--show-toplevel").removesuffix(b"\n")))


def patch_paths(root: Path, cwd: Path, patch: str) -> dict[Path, set[str]]:
    """Route absolute patch headers to their linked worktrees, including move sources."""
    entries = git(root, "worktree", "list", "--porcelain", "-z").split(b"\0")
    roots = {Path(os.fsdecode(e[9:])).resolve() for e in entries if e.startswith(b"worktree ")}
    groups: dict[Path, set[str]] = {}
    for header in PATCH_PATH.findall(patch):
        path = cwd / header
        # Resolve directory symlinks, but keep the name of an edited/deleted symlink.
        path = path.parent.resolve() / path.name
        if not any(path.is_relative_to(worktree) for worktree in roots):
            continue
        directory = next(parent for parent in path.parents if parent.is_dir())
        worktree = git_root(directory)
        if worktree in roots:
            relative = path.relative_to(worktree).as_posix()
            if is_checkable(relative):
                groups.setdefault(worktree, set()).add(relative)
    return groups


def changed_paths(root: Path, revision: str) -> set[str]:
    # --no-renames retains both ends of a move. NUL delimiters preserve Git filenames.
    tracked = git(root, "diff", "--name-only", "--no-renames", "--no-ext-diff", "-z", revision, "--")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return {name for raw in (tracked + untracked).split(b"\0")
            if raw and is_checkable(name := os.fsdecode(raw))}


def fingerprint(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    if stat.S_ISLNK(mode):
        return "symlink:" + os.readlink(path)
    if not stat.S_ISREG(mode):
        return f"mode:{mode}"
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_baseline(root: Path, state: Path) -> None:
    head = git(root, "rev-parse", "--verify", "HEAD").decode().strip()
    baseline = {p: fingerprint(root / p) for p in changed_paths(root, head)}
    state.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=state.parent) as directory:
        temporary = Path(directory) / "baseline.json"
        temporary.write_text(json.dumps({"head": head, "dirty": baseline}), encoding="utf-8")
        temporary.replace(state)


def changes_since_prompt(root: Path, state: Path) -> set[str]:
    try:
        baseline = json.loads(state.read_text())
    except FileNotFoundError:
        # A resumed conversation may have missed UserPromptSubmit.
        return changed_paths(root, "HEAD")
    dirty = baseline["dirty"]
    changed = changed_paths(root, baseline["head"]) | dirty.keys()
    return {p for p in changed if p not in dirty or fingerprint(root / p) != dirty[p]}


def handle(event: dict) -> str:
    cwd = Path(event["cwd"]).resolve()
    root = git_root(cwd)
    kind = event["hook_event_name"]
    if kind == "PostToolUse":
        groups = patch_paths(root, cwd, event["tool_input"]["command"])
        errors = []
        for worktree, paths in groups.items():
            if message := run_checks(worktree, paths):
                errors.append(f"{worktree}\n{message}")
        return "\n".join(errors)

    # Concurrent Codex sessions share a checkout but never share their prompt baseline.
    key = hashlib.sha256(event["session_id"].encode()).hexdigest()
    state = root / "internal-testing/cache/codex-edit-feedback" / f"{key}.json"
    if kind == "UserPromptSubmit":
        capture_baseline(root, state)
    elif kind == "Stop":
        error = run_checks(root, changes_since_prompt(root, state))
        if error:
            return f"{root}\n{error}"
        state.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unsupported Codex hook event: {kind}")
    return ""


def main() -> int:
    event = {}
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("expected a Codex hook event object")
        event = payload
        error = handle(event)
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        error = f"AC-Prof edit feedback could not run: {exc}"
    if not error:
        return 0
    if event.get("hook_event_name") == "UserPromptSubmit" or event.get("stop_hook_active"):
        # Surface an unavailable baseline or an unresolved failure without an endless Stop loop.
        print(json.dumps({"systemMessage": error}))
        return 0
    print(error, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
