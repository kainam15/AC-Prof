"""Behavioural fixtures for the production private-dependency gate."""
from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path

import pytest

from scripts.check_private_api import Dependency, main, read_baseline, scan_sources, write_baseline


class TestPrivateDependency:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))
        self.write("acprof/__init__.py", "")
        self.write("acprof/pkg/__init__.py", "")
        self.write("acprof/pkg/provider.py", "raise RuntimeError('must not import')\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @pytest.mark.parametrize('spelling_case', range(6))
    def test_import_spellings_resolve_to_the_same_dependency(self, spelling_case):
        spellings = (
            "from acprof.pkg.provider import _secret as public\npublic()",
            "import acprof.pkg.provider\nacprof.pkg.provider._secret()",
            "import acprof.pkg.provider as p\np._secret()",
            "from acprof.pkg import provider\nprovider._secret()",
            "from .pkg import provider as p\np._secret()",
            "from .pkg.provider import _secret",
        )
        expected = Dependency("acprof.consumer", "acprof.pkg.provider", "_secret")
        spelling = tuple(spellings)[spelling_case]
        self.write("acprof/consumer.py", spelling)
        assert (set(scan_sources(self.root))) == ({expected})

    def test_occurrences_are_recorded_but_dependencies_are_unique(self):
        self.write("acprof/consumer.py", "from acprof.pkg import provider as p\np._secret()\np._secret()\n")
        refs = scan_sources(self.root)
        assert (len(refs)) == (1)
        assert (next(iter(refs.values()))) == (["acprof/consumer.py:2", "acprof/consumer.py:3"])

    def test_self_public_dunder_external_and_instance_attributes_are_excluded(self):
        self.write("acprof/pkg/provider.py", "from .provider import _own\n")
        self.write("acprof/consumer.py", """
import os
from acprof.pkg import provider as p
from acprof.pkg.provider import Public
p.public()
p.__name__
os._exit(0)
Public._method()
def f(p):
    p._instance()
def g():
    p = Public()
    p._instance()
""")
        assert (scan_sources(self.root)) == ({})

    def test_local_imports_do_not_leak_into_sibling_scopes(self):
        self.write("acprof/consumer.py", """
def first():
    from acprof.pkg import provider as p
    p._secret()
def second(p):
    p._instance()
""")
        assert (set(scan_sources(self.root))) == ({
            Dependency("acprof.consumer", "acprof.pkg.provider", "_secret"),
        })

    def test_relative_imports_in_packages_and_nested_scopes(self):
        self.write("acprof/pkg/__init__.py", "from .provider import _secret\n")
        self.write("acprof/pkg/consumer.py", """
from ..pkg import provider
def outer():
    def inner():
        return provider._nested()
    return inner()
""")
        assert (set(scan_sources(self.root))) == ({
            Dependency("acprof.pkg", "acprof.pkg.provider", "_secret"),
            Dependency("acprof.pkg.consumer", "acprof.pkg.provider", "_nested"),
        })

    def test_tests_are_counted_separately_from_production(self):
        self.write("tests/test_example.py", "from acprof.pkg.provider import _fixture\n")
        assert (scan_sources(self.root)) == ({})
        assert (set(scan_sources(self.root, ("tests",)))) == ({
            Dependency("tests.test_example", "acprof.pkg.provider", "_fixture"),
        })

    def run_guard(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(["--root", str(self.root), "--baseline", "baseline.json", *args])
        return result, output.getvalue()

    def test_new_dependencies_are_rejected_even_if_another_is_removed(self):
        write_baseline(self.root / "baseline.json", {Dependency("acprof.consumer", "acprof.pkg.provider", "_old")})
        self.write("acprof/consumer.py", "from acprof.pkg.provider import _new\n")
        code, output = self.run_guard()
        assert (code) == (1)
        assert ("NEW") in (output)
        assert ("STALE") in (output)

    def test_pruning_only_removes_debt_and_readding_is_rejected(self):
        baseline = self.root / "baseline.json"
        write_baseline(baseline, {Dependency("acprof.consumer", "acprof.pkg.provider", "_old")})
        assert (self.run_guard()[0]) == (1)
        assert (self.run_guard("--prune")[0]) == (0)
        assert (read_baseline(baseline)) == (set())
        self.write("acprof/consumer.py", "from acprof.pkg.provider import _old\n")
        assert (self.run_guard("--prune")[0]) == (1)
        assert (read_baseline(baseline)) == (set())

    def test_baseline_cannot_silently_hide_duplicate_entries(self):
        self.write("baseline.json", '{"schema_version": 1, "dependencies": [{"source":"a","target":"b","symbol":"_x"},{"source":"a","target":"b","symbol":"_x"}]}')
        with pytest.raises(ValueError, match="duplicates"):
            read_baseline(self.root / "baseline.json")

    def test_wrong_root_cannot_prune_the_entire_baseline(self):
        baseline = self.root / "baseline.json"
        expected = {Dependency("acprof.consumer", "acprof.pkg.provider", "_old")}
        write_baseline(baseline, expected)
        shutil.rmtree(self.root / "acprof")
        assert (self.run_guard("--prune")[0]) == (1)
        assert (read_baseline(baseline)) == (expected)

    def test_comprehension_and_lambda_parameters_shadow_module_aliases(self):
        self.write("acprof/consumer.py", """
from acprof.pkg import provider as p
[p._instance() for p in p._items()]
f = lambda p: p._instance()
""")
        assert (set(scan_sources(self.root))) == ({
            Dependency("acprof.consumer", "acprof.pkg.provider", "_items"),
        })
