"""Bounded local runner must preserve scope, evidence and failure behavior."""

import json
import sys
from pathlib import Path

import pytest

from scripts.run_local_tests import build_command, collect_shards, main

ROOT = Path(__file__).resolve().parents[2]


def test_parallel_command_keeps_pytest_scope_and_isolated_evidence(tmp_path):
    command = build_command(
        ROOT / "tests" / "analysis",
        scope="fast",
        shard=1,
        jobs=2,
        output=tmp_path,
        keyword="stats",
    )
    assert command[0] == sys.executable
    assert command[command.index("-m") + 1] == "unit"
    assert command[command.index("-k") + 1] == "stats"
    assert command[command.index("--shard-index") + 1] == "1"
    assert command[command.index("--shard-count") + 1] == "2"
    assert str(tmp_path / "reports" / "shard-1.json") in command
    assert str(tmp_path / "artifacts" / "tmp-1") in command
    assert "no:cacheprovider" in command


def test_serial_scopes_do_not_accidentally_select_unit(tmp_path):
    for scope, expected_marker in (("tui", "tui"), ("full", None)):
        command = build_command(ROOT / "tests", scope=scope, shard=0, jobs=1, output=tmp_path)
        assert (command[command.index("-m") + 1] if "-m" in command else None) == expected_marker


@pytest.mark.parametrize(
    "options",
    [
        ("--scope", "full", "--jobs", "2"),
        ("--scope", "tui", "--jobs", "2"),
        ("--jobs", "0"),
        ("--jobs", "3"),
        ("--jobs", "2"),
        ("--jobs", "2", "--path", "tests/tui"),
        ("--path", "../"),
        ("--output", "results/pytest"),
        ("--output", "../outside"),
    ],
)
def test_reject_unsafe_selection_before_starting_shards(options):
    with pytest.raises(SystemExit) as error:
        main(list(options))
    assert error.value.code == 2


def test_missing_shard_is_a_failure_even_with_an_existing_report(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "shard-0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": "3.12.15",
                "shard": {"index": 0, "count": 2},
                "counts": {},
                "tests": [],
            }
        )
    )
    summary = collect_shards(tmp_path, jobs=2)
    assert not summary["successful"]
    assert any("missing shards" in error for error in summary["errors"])
    assert any("shard-1.json" in error for error in summary["errors"])


def test_nonzero_worker_exit_overrides_optimistic_aggregate(tmp_path, monkeypatch):
    from scripts import run_local_tests as runner

    class FakeProcess:
        def wait(self):
            return 3

        def poll(self):
            return 3

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        runner,
        "collect_shards",
        lambda output, *, jobs: {
            "successful": True,
            "errors": [],
            "versions": {"3.12": {"tests": 2}},
        },
    )
    output = tmp_path / "run"
    assert not runner.run_local(
        directory=ROOT / "tests" / "quality",
        scope="fast",
        jobs=1,
        output=output,
    )
    report = json.loads((output / "reports" / "summary.json").read_text())
    assert not report["successful"]
    assert report["exit_codes"] == [3]
    assert "nonzero shard exit codes" in report["errors"][0]


def test_interrupted_run_terminates_owned_process_group(tmp_path, monkeypatch):
    from scripts import run_local_tests as runner

    terminated = []

    class FakeProcess:
        pid = 456123
        waits = 0

        def wait(self, **kwargs):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

        def poll(self):
            return None

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: terminated.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        runner.run_local(
            directory=ROOT / "tests" / "quality",
            scope="fast",
            jobs=1,
            output=tmp_path / "interrupted",
        )
    assert terminated == [(456123, runner.signal.SIGTERM)]
