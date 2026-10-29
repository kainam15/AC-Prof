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
                "PATH": "/fixture-bin",
        }
        assert dict(os.environ) == parent
    assert overrides == {"PATH": "/fixture-bin"}


def test_isolated_environment_does_not_create_absent_plugin_state():
    with patch.dict(os.environ, {"ACPROF_MODEL_SOURCE": "modelscope"}, clear=True):
        assert isolated_environment() == {}


@pytest.mark.parametrize(("module_name", "class_name"), FIXTURES)
@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
@pytest.mark.parametrize("parent_no_color", [False, True], ids=["color-absent", "color-present"])
def test_real_fixture_restores_environment_after_nested_patches(
    tmp_path, module_name, class_name, outcome, parent_no_color,
):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    test_file = tmp_path / "test_fixture.py"
    ending = {
        "success": '        assert os.environ["PYTEST_CURRENT_TEST"]\n',
        "failure": '        assert False, "intentional fixture failure"\n',
        "interrupt": '        raise KeyboardInterrupt("intentional fixture interruption")\n',
    }[outcome]
    test_file.write_text(
        f"import os\nfrom {module_name} import {class_name} as _Harness\n"
        "class TestFixture:\n"
        "    _setup = _Harness._setup\n"
        "    def test_body(self, monkeypatch):\n"
        "        assert 'ACPROF_TEST_PARENT_SENTINEL' not in os.environ\n"
        "        assert 'NO_COLOR' not in os.environ\n"
        "        assert 'HF_TOKEN' not in os.environ\n"
        "        assert 'HTTP_PROXY' not in os.environ\n"
        "        monkeypatch.setenv('NO_COLOR', 'nested-value')\n"
        "        monkeypatch.setenv('HTTP_PROXY', 'http://nested.invalid')\n"
        "        monkeypatch.delenv('PYTEST_TEST_SENTINEL')\n"
        "        os.environ['HF_TOKEN'] = 'fixture-only-child-token'\n"
        "        os.environ['ACPROF_TEST_DIRECT_ONLY'] = 'must-be-removed'\n"
        + ending,
        encoding="utf-8",
    )
    report = tmp_path / "evidence.json"
    restored = tmp_path / "environment.json"
    runner = tmp_path / "run_fixture.py"
    runner.write_text(
        "import json\nimport os\nimport sys\nfrom pathlib import Path\nimport pytest\n"
        "keys = ('NO_COLOR', 'HF_TOKEN', 'HTTP_PROXY', 'ACPROF_TEST_PARENT_SENTINEL', "
        "'ACPROF_TEST_DIRECT_ONLY', 'PYTEST_TEST_SENTINEL')\n"
        "before = {key: os.environ.get(key) for key in keys}\n"
        "code = pytest.main(['-q', '-p', 'acprof.testing.plugin', '-p', 'no:cacheprovider', "
        "'test_fixture.py', '--report', 'evidence.json'])\n"
        "after = {key: os.environ.get(key) for key in keys}\n"
        "Path('environment.json').write_text(json.dumps({'before': before, 'after': after}))\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "tests"))),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_TEST_SENTINEL="parent-plugin-state",
        ACPROF_TEST_PARENT_SENTINEL="must-not-leak",
        HF_TOKEN="fixture-only-parent-token",
        HTTP_PROXY="http://parent.invalid",
    )
    if parent_no_color:
        environment["NO_COLOR"] = "parent-value"
    else:
        environment.pop("NO_COLOR", None)
    environment.pop("ACPROF_TEST_DIRECT_ONLY", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == {"success": 0, "failure": 1, "interrupt": 2}[outcome], output
    assert "KeyError" not in output
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["successful"] is (outcome == "success")
    if outcome == "interrupt":
        assert "intentional fixture interruption" in output
    elif outcome == "failure":
        assert evidence["counts"]["failed"] == 1
        assert "intentional fixture failure" in output
    else:
        assert evidence["counts"]["passed"] == 1
    state = json.loads(restored.read_text(encoding="utf-8"))
    assert state["after"] == state["before"]
