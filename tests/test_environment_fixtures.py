"""Test environment isolation must preserve pytest/plugin cancellation evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from environment_fixtures import isolated_environment

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    ("test_image_management", "TestImageManagement"),
    ("test_tui_images", "TestTuiImages"),
    ("test_tui_all_tables", "TestAllTables"),
    ("test_tui_environment", "TestTuiEnvironment"),
    ("test_tui_preflight", "TestTuiPreflight"),
    ("test_tui_storage", "TestTuiStorage"),
)


def test_isolated_environment_preserves_harness_not_application_settings():
    parent = {
        "PYTEST_CURRENT_TEST": "test_isolation (call)",
        "PYTEST_XDIST_WORKER": "gw0",
        "TEXTUAL_SNAPSHOT_TEMPDIR": "/snapshot-fixture",
        "ACPROF_MODEL_SOURCE": "modelscope",
        "HTTP_PROXY": "http://proxy.invalid",
        "HF_TOKEN": "fixture-only-not-a-credential",
        "PATH": "/host-only",
    }
    overrides = {"PATH": "/fixture-bin"}
    with patch.dict(os.environ, parent, clear=True):
        actual = isolated_environment(overrides)
        assert actual == {
            "PYTEST_CURRENT_TEST": "test_isolation (call)",
            "PYTEST_XDIST_WORKER": "gw0",
            "TEXTUAL_SNAPSHOT_TEMPDIR": "/snapshot-fixture",
            "PATH": "/fixture-bin",
        }
        assert dict(os.environ) == parent
    assert overrides == {"PATH": "/fixture-bin"}


def test_isolated_environment_does_not_create_absent_plugin_state():
    with patch.dict(os.environ, {"ACPROF_MODEL_SOURCE": "modelscope"}, clear=True):
        assert isolated_environment() == {}


@pytest.mark.parametrize("module_name,class_name", FIXTURES)
@pytest.mark.parametrize("interrupted", [False, True], ids=["success", "interrupt"])
def test_real_fixture_keeps_snapshot_session_teardown_usable(tmp_path, module_name, class_name, interrupted):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    test_file = tmp_path / "test_fixture.py"
    ending = ('        raise KeyboardInterrupt("intentional fixture interruption")\n'
              if interrupted else '        assert os.environ["TEXTUAL_SNAPSHOT_TEMPDIR"]\n')
    test_file.write_text(
        f"import os\nfrom {module_name} import {class_name} as _Harness\n"
        "class TestFixture:\n"
        "    _setup = _Harness._setup\n"
        "    def test_body(self):\n"
        "        assert 'ACPROF_TEST_PARENT_SENTINEL' not in os.environ\n"
        + ending,
        encoding="utf-8",
    )
    report = tmp_path / "evidence.json"
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "tests"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        ACPROF_TEST_PARENT_SENTINEL="must-not-leak",
    )
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_XDIST_WORKER", None)
    environment.pop("TEXTUAL_SNAPSHOT_TEMPDIR", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "acprof.testing.plugin",
         "-p", "pytest_textual_snapshot", "-p", "no:cacheprovider",
         str(test_file), "--report", str(report)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == (2 if interrupted else 0), output
    assert "KeyError" not in output
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["successful"] is (not interrupted)
    if interrupted:
        assert "intentional fixture interruption" in output
    else:
        assert evidence["counts"]["passed"] == 1
