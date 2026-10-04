"""Exercise the public pytest CLI and schema v1 consumer in isolated processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.aggregate_test_reports import aggregate_reports

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def suite(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    def run(source, *args, wrapper=False, existing_report=False):
        for name, body in (source if isinstance(source, dict) else {"test_sample.py": source}).items():
            (tmp_path / name).write_text(body)
        report = tmp_path / "evidence.json"
        report.unlink(missing_ok=True)
        if existing_report:
            report.write_text("{}")
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.pop("PYTEST_ADDOPTS", None)
        command = ([sys.executable, str(ROOT / "scripts/run_tests.py"), "--directory", str(tmp_path)]
                   if wrapper else [sys.executable, "-m", "pytest", "-p", "acprof.testing.plugin", str(tmp_path)])
        result = subprocess.run([*command, "--report", str(report), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=60)
        return result, json.loads(report.read_text()) if report.exists() else None

    return run


def test_wrapper_runs_native_pytest_functions(suite):
    result, report = suite("import asyncio\ndef test_native():\n    assert 2 + 2 == 4\n"
                           "async def test_async():\n    await asyncio.sleep(0)\n", wrapper=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["schema_version"] == 1
    assert report["counts"]["passed"] == 2
    assert {row["id"].split("::")[-1] for row in report["tests"]} == {"test_native", "test_async"}
    assert "PytestAssertRewriteWarning" not in result.stdout + result.stderr
    # A reused report outside the checkout must not change config discovery or
    # remove tests/ from the import path (as with a /evidence container mount).
    result, report = suite("", "--directory", str(ROOT / "tests"), "--pattern", "test_cv_runtime.py",
                           "--collect-only", wrapper=True, existing_report=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not report["collection_errors"]
    assert report["shard"]["discovered"] > 0


def test_phase_outcomes_keep_errors_and_failures_distinct(suite):
    result, report = suite('''
import pytest
@pytest.fixture
def broken_setup():
    raise RuntimeError("setup reason")
@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("teardown reason")
def test_pass():
    assert True
def test_assertion():
    assert False, "assertion reason"
def test_exception():
    raise RuntimeError("call reason")
def test_setup(broken_setup):
    pass
def test_teardown(broken_teardown):
    pass
@pytest.mark.skip(reason="skip reason")
def test_skip():
    pass
@pytest.mark.xfail(reason="known defect", strict=True)
def test_xfail():
    assert False
@pytest.mark.xfail(reason="unexpected fix", strict=True)
def test_xpass():
    pass
@pytest.mark.xfail
def test_xfail_without_reason():
    assert False
@pytest.mark.xfail
def test_xpass_without_reason():
    pass
''')
    assert result.returncode != 0
    assert report["successful"] is False
    assert report["counts"] == {"run": 10, "passed": 1, "failed": 1, "errors": 3,
                                "skipped": 1, "expected_failures": 2, "unexpected_successes": 2}
    assert len({row["id"] for row in report["tests"]}) == 10
    assert all(row["duration_s"] >= 0 for row in report["tests"])
    assert all(row["reason"] for row in report["tests"] if row["outcome"] != "passed")
    assert all(report[key] for key in ("python", "platform", "packages", "timestamp"))
    result, report = suite("import pytest\n@pytest.mark.xfail\ndef test_fixed():\n    pass\n")
    assert result.returncode == 1
    assert report["counts"]["unexpected_successes"] == 1
    assert report["successful"] is False


@pytest.mark.parametrize("mark", [
    "skip(reason='intentional')", "xfail(reason='known', strict=True)",
    "xfail(reason='known', run=False)",
])
def test_no_skips_rejects_skips_and_expected_failures(suite, mark):
    result, report = suite(f"import pytest\n@pytest.mark.{mark}\ndef test_case():\n    assert False\n",
                           "--require-no-skips")
    assert result.returncode != 0
    assert report["successful"] is False
    assert report["require_no_skips"] is True


def test_shards_are_sorted_reproducible_and_aggregate_without_schema_changes(suite):
    source = "\n".join(f"def test_{i}():\n    assert True\n" for i in reversed(range(9)))
    reports = []
    for index in range(4):
        result, report = suite(source, "--shard-index", str(index), "--shard-count", "4")
        assert result.returncode == 0, result.stdout + result.stderr
        reports.append(report)
    result, repeated = suite(source, "--shard-index", "2", "--shard-count", "4")
    assert result.returncode == 0
    assert repeated["shard"] == reports[2]["shard"]
    assert [r["id"] for r in repeated["tests"]] == [r["id"] for r in reports[2]["tests"]]
    version = ".".join(repeated["python"].split(".")[:2])
    aggregate = aggregate_reports(reports, python_versions=[version], shard_count=4)
    assert aggregate["successful"], aggregate["errors"]
    assert aggregate["versions"][version]["tests"] == 9


def test_evidence_identifies_test_content_even_when_node_ids_do_not_change(suite):
    result, first = suite("def test_case():\n    assert 1 == 1\n")
    assert result.returncode == 0
    result, second = suite("def test_case():\n    assert 2 == 2\n")
    assert result.returncode == 0
    assert first['shard']['suite_sha256'] == second['shard']['suite_sha256']
    assert first['provenance']['tests_sha256'] != second['provenance']['tests_sha256']
    assert first['provenance']['source_sha256'] == second['provenance']['source_sha256']
    assert first['provenance']['locks_sha256'] == second['provenance']['locks_sha256']


def test_source_changed_during_execution_invalidates_evidence(suite):
    result, report = suite("from pathlib import Path\ndef test_case():\n"
                           "    path = Path(__file__)\n"
                           "    path.write_text(path.read_text() + '# changed\\n')\n")
    assert result.returncode != 0
    assert report['counts']['passed'] == 1
    assert report['successful'] is False
    assert 'changed during' in report['provenance_error']


@pytest.mark.parametrize("source, args", [
    ("def test_ok():\n    pass\n", ["--pattern", "test_missing*.py"]),
    ("def test_ok():\n    pass\n", ["--shard-count", "4", "--shard-index", "3"]),
    ("raise RuntimeError('collection failure')\n", []),
])
def test_empty_or_broken_collection_cannot_report_success(suite, source, args):
    result, report = suite(source, *args)
    assert result.returncode != 0
    assert report["successful"] is False


def test_overlapping_patterns_do_not_duplicate_cases(suite):
    result, report = suite("def test_one():\n    pass\n", "--pattern", "test_*.py",
                           "--pattern", "test_sample.py")
    assert result.returncode == 0
    assert report["counts"]["run"] == 1
    assert report["discovery_counts"] == {"test_*.py": 1, "test_sample.py": 1}
    result, report = suite({"check_custom.py": "def test_custom():\n    pass\n",
                            "unselected_test.py": "raise RuntimeError('unselected dependency')\n"},
                           "--pattern", "check_custom.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["discovery_counts"] == {"check_custom.py": 1}
    assert report["counts"]["passed"] == 1


def test_deselection_precedes_sharding_and_collection_only_is_not_execution(suite):
    source = "def test_selected():\n    pass\ndef test_other():\n    pass\n"
    result, report = suite(source, "-k", "selected", "--collect-only")
    assert result.returncode == 0
    assert report["successful"] is False
    assert report["counts"]["run"] == 0
    assert report["shard"]["discovered"] == 1
    assert len(report["collected_ids"]) == 1


@pytest.mark.parametrize("option", ["--setup-only", "--setup-plan"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_setup_diagnostics_are_not_execution_evidence(suite, option, wrapper):
    result, report = suite("def test_body():\n    assert False, 'body must not run'\n",
                           option, wrapper=wrapper)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["successful"] is False
    assert report["counts"]["run"] == report["counts"]["passed"] == 0
    assert report["tests"] == []
    assert report["shard"]["selected"] == report["shard"]["discovered"] == 1
    assert len(report["collected_ids"]) == 1


def test_xfail_without_call_keeps_its_expected_outcome(suite):
    result, report = suite("import pytest\n@pytest.mark.xfail(run=False, reason='known defect')\n"
                           "def test_case():\n    assert False, 'body must not run'\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["successful"] is True
    assert report["counts"]["run"] == report["counts"]["expected_failures"] == 1
    assert report["counts"]["passed"] == 0
    assert "known defect" in report["tests"][0]["reason"]


@pytest.mark.parametrize("phase", ["call", "teardown"])
def test_interruption_preserves_only_completed_passes(suite, phase):
    result, report = suite(f'''
import pytest
@pytest.fixture
def lifecycle():
    yield
    if {phase!r} == "teardown":
        raise KeyboardInterrupt
def test_a_completed():
    assert True
def test_b_interrupted(lifecycle):
    if {phase!r} == "call":
        raise KeyboardInterrupt
def test_c_unreached():
    assert False, "must not run after interruption"
''')
    assert result.returncode == 2, result.stdout + result.stderr
    assert report["successful"] is False
    assert report["shard"]["selected"] == report["shard"]["discovered"] == 3
    assert report["counts"]["run"] == report["counts"]["passed"] == 1
    assert [row["id"].split("::")[-1] for row in report["tests"]] == ["test_a_completed"]


def test_measurement_root_is_isolated_for_direct_pytest_and_all_fixture_phases(suite):
    result, report = suite('''
from pathlib import Path
import pytest
from acprof.host import run_state
@pytest.fixture
def lifecycle():
    root = run_state.MEASUREMENT_LOCK_ROOT
    assert root != Path("/tmp")
    assert root.is_dir()
    yield root
    assert root.is_dir()
    assert run_state.MEASUREMENT_LOCK_ROOT == root
def test_lock(lifecycle):
    assert run_state.MEASUREMENT_LOCK_ROOT == lifecycle
''')
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["successful"] is True


def test_no_skips_also_rejects_a_module_skipped_during_collection(suite):
    result, report = suite({
        "test_good.py": "def test_ok(): pass\n",
        "test_missing.py": "import pytest\npytest.skip('missing runtime', allow_module_level=True)\n",
    }, "--require-no-skips")
    assert result.returncode != 0
    assert report["counts"]["passed"] == 1
    assert "missing runtime" in report["collection_skips"][0]["reason"]
    assert report["successful"] is False
