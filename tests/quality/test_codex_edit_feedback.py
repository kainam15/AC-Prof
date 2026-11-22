"""Exercise edit feedback against real Git paths and Codex event boundaries."""
import io
import json
import subprocess

import pytest

from scripts import codex_edit_feedback as feedback


def commit(root):
    subprocess.run(["git", "-C", root, "add", "-A"], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "fixture"], check=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "project"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, value in (
        ("user.email", "test@example.invalid"), ("user.name", "Test"), ("commit.gpgSign", "false"),
        ("core.hooksPath", "/dev/null"),
    ):
        subprocess.run(["git", "-C", root, "config", name, value], check=True)
    (root / ".gitignore").write_text("/internal-testing/\n")
    (root / "old file.py").write_text("v = 1\n")
    commit(root)
    return root


@pytest.fixture
def checked(monkeypatch):
    calls = []

    def run(root, paths):
        calls.append((root, paths))
        return ""

    monkeypatch.setattr(feedback, "run_checks", run)
    return calls


def event(root, name, **fields):
    return {"cwd": str(root), "session_id": "one", "hook_event_name": name, **fields}


def test_stop_detects_shell_edits_including_commits_and_move_sources(repo, checked):
    (repo / "untouched.py").write_text("existing user edit")
    (repo / "overwritten.py").write_text("existing user edit")
    (repo / "old file.py").write_text("v = 2\n")
    feedback.handle(event(repo, "UserPromptSubmit"))
    (repo / "old file.py").rename(repo / "moved file.py")
    (repo / "overwritten.py").write_text("new content")
    (repo / "readme.md").write_text("description")
    (repo / "internal-testing" / "example.py").write_text("ignored")
    commit(repo)
    assert feedback.handle(event(repo, "Stop")) == ""
    assert checked == [(repo, {"old file.py", "moved file.py", "overwritten.py"})]


def test_sessions_keep_independent_baselines_and_include_untracked_edits(repo, checked):
    feedback.handle(event(repo, "UserPromptSubmit"))
    (repo / "new.py").write_text("first edit")
    feedback.handle(event(repo, "UserPromptSubmit", session_id="two"))
    assert feedback.handle(event(repo, "Stop", session_id="two")) == ""
    assert feedback.handle(event(repo, "Stop")) == ""
    assert checked == [(repo, set()), (repo, {"new.py"})]


def test_resumed_session_with_no_baseline_uses_dirty_paths(repo, checked):
    (repo / "old file.py").unlink()
    (repo / "new module.py").write_text("work")
    assert feedback.handle(event(repo, "Stop")) == ""
    assert checked == [(repo, {"old file.py", "new module.py"})]


def test_patch_routes_relative_and_absolute_headers_to_their_own_linked_worktree(repo, checked):
    sibling = repo.parent / "linked"
    subprocess.run(["git", "-C", repo, "worktree", "add", "-qb", "edit", sibling], check=True)
    subdir = repo / "nested"
    subdir.mkdir()
    foreign = repo.parent / "unrelated.py"
    patch = "\n".join((
        "*** Begin Patch", "*** Update File: ../old file.py", "*** Move to: renamed.py",
        f"*** Delete File: {sibling}/deleted/old.py", f"*** Add File: {foreign}",
        "*** End Patch",
    ))
    assert feedback.handle(event(subdir, "PostToolUse", tool_input={"command": patch})) == ""
    assert dict(checked) == {repo: {"old file.py", "nested/renamed.py"}, sibling: {"deleted/old.py"}}


def test_stop_returns_failure_to_model_retries_once_and_clears_state_after_repair(
    repo, monkeypatch, capsys,
):
    diagnosis = "tests/test_example.py:7: AssertionError: expected hidden"
    failures = [diagnosis, diagnosis, ""]
    monkeypatch.setattr(feedback, "run_checks", lambda *_: failures.pop(0))

    def invoke(name, **fields):
        monkeypatch.setattr(feedback.sys, "stdin", io.StringIO(json.dumps(event(repo, name, **fields))))
        return feedback.main(), capsys.readouterr()

    code, output = invoke("UserPromptSubmit")
    assert code == 0
    assert output.out == output.err == ""
    (repo / "new.py").write_text("example")
    code, output = invoke("Stop")
    assert code == 2
    assert diagnosis in output.err
    assert output.out == ""
    code, output = invoke("Stop", stop_hook_active=True)
    assert code == 0
    assert diagnosis in json.loads(output.out)["systemMessage"]
    assert output.err == ""
    code, output = invoke("Stop")
    assert code == 0
    assert output.out == output.err == ""
    assert list((repo / "internal-testing/cache/codex-edit-feedback").glob("*.json")) == []


def test_invalid_event_has_visible_diagnostic(monkeypatch, capsys):
    monkeypatch.setattr(feedback.sys, "stdin", io.StringIO("[]"))
    assert feedback.main() == 2
    assert "expected a Codex hook event object" in capsys.readouterr().err
