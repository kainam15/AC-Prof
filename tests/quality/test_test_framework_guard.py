"""Aliases cannot bypass the runner policy; mock remains available."""

import pytest

from scripts.check_test_framework import main, violations


@pytest.mark.parametrize("source", [
    "import unittest as framework", "from unittest import TestCase as Base",
    "from unittest.case import TestCase", "from unittest import *",
])
def test_rejects_runner_imports_including_aliases(source):
    assert violations(source)


@pytest.mark.parametrize("source", [
    "from unittest.mock import Mock, patch", "import unittest.mock as mock",
    "from unittest import mock", "import pytest", "# import unittest",
])
def test_allows_mock_and_does_not_scan_comments(source):
    assert violations(source) == []


def test_cli_checks_new_untracked_python_files(tmp_path):
    source = tmp_path / "test_new.py"
    source.write_text("import unittest as framework\n")
    assert main([str(tmp_path)]) == 1
    source.write_text("import unittest.mock\nclass Example(unittest." + "TestCase): pass\n")
    assert main([str(tmp_path)]) == 1
    source.write_text("import unittest.mock\nvalue = unittest.mock.Mock()\n")
    assert main([str(tmp_path)]) == 0
