"""Capability decisions are exact, reviewed, and independent of version ordering."""
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from acprof.container import local_pipeline


class TestLocalPipelineRouting:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        directory = tmp_path
        self.root = Path(str(directory))
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "config.json").write_text(json.dumps({"custom_pipelines": {
            "fixture": {"impl": "entry.Pipeline"}}}))
        self.transformers = ModuleType("transformers")
        self.transformers.__version__ = "4.57.6"
        self.dynamic = ModuleType("transformers.dynamic_module_utils")
        self.native_class = type("NativePipeline", (), {})
        self.compat_class = type("CompatPipeline", (), {})
        self.dynamic.HF_MODULES_CACHE = str(self.root / "cache")
        self.dynamic.get_relative_import_files = Mock(return_value=[])
        self.dynamic.get_cached_module_file = Mock(return_value="snapshot/entry.py")
        self.dynamic.get_class_in_module = Mock(return_value=self.compat_class)
        self.dynamic.get_class_from_dynamic_module = Mock(return_value=self.native_class)
        modules = {"transformers": self.transformers, "transformers.dynamic_module_utils": self.dynamic}
        patcher = patch.dict("sys.modules", modules)
        patcher.start()
        self._request.addfinalizer(partial(patcher.stop))

    def catalog(self, installed_version, capabilities, **overrides):
        self.transformers.__version__ = installed_version
        path = self.root / "extensions" / "transformers" / f"{installed_version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "version": installed_version, "mappings": {}, "capabilities": capabilities}
        data.update(overrides)
        path.write_text(json.dumps(data))
        patcher = patch("acprof.model_resolution.__file__", str(self.root / "model_resolution.py"))
        patcher.start()
        self._request.addfinalizer(partial(patcher.stop))

    def load(self):
        return local_pipeline.load_local_pipeline_class(str(self.snapshot), "fixture")

    def assert_no_loader_calls(self):
        self.dynamic.get_class_from_dynamic_module.assert_not_called()
        self.dynamic.get_cached_module_file.assert_not_called()
        self.dynamic.get_relative_import_files.assert_not_called()
        self.dynamic.get_class_in_module.assert_not_called()

    @pytest.mark.parametrize('version', ('4.57.6', '5.6.0'))
    def test_supported_versions_keep_compat(self, version):
        self.transformers.__version__ = version
        assert (self.load()) is (self.compat_class)
        assert (self.dynamic.get_relative_import_files.call_count) == (1)
        self.dynamic.get_class_from_dynamic_module.assert_not_called()

    def test_reviewed_native_uses_only_upstream_loader(self):
        self.catalog("99.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        with patch.dict("sys.modules", {"acprof.container.compat.transformers_dynamic": None}):
            assert (self.load()) is (self.native_class)
        self.dynamic.get_class_from_dynamic_module.assert_called_once_with(
            "entry.Pipeline", str(self.snapshot), local_files_only=True)
        self.dynamic.get_relative_import_files.assert_not_called()
        self.dynamic.get_cached_module_file.assert_not_called()
        self.dynamic.get_class_in_module.assert_not_called()

    @pytest.mark.parametrize('transitive,symlink', ((False, True), (True, False)))
    def test_partial_capabilities_still_use_compat(self, transitive, symlink):
        self.catalog(f"88.{int(transitive)}.0", {
            "local_dynamic_transitive_imports": transitive, "local_dynamic_symlink_safe": symlink})
        assert (self.load()) is (self.compat_class)
        self.dynamic.get_class_from_dynamic_module.assert_not_called()

    def test_native_capability_does_not_depend_on_version_order_or_flat_module_names(self):
        self.catalog("3.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        (self.snapshot / "config.json").write_text(json.dumps({"custom_pipelines": {
            "fixture": {"impl": "package.entry.Pipeline"}}}))
        assert (self.load()) is (self.native_class)
        self.dynamic.get_class_from_dynamic_module.assert_called_once_with(
            "package/entry.Pipeline", str(self.snapshot), local_files_only=True)

    @pytest.mark.parametrize('version', ('99.0.1', '5.6.0+unreviewed'))
    def test_unknown_version_fails_before_any_loader(self, version):
        self.transformers.__version__ = version
        with pytest.raises(ValueError, match="[Tt]ransformers.*" + version.split("+")[0]):
            self.load()
        self.assert_no_loader_calls()

    @pytest.mark.parametrize("capabilities", [
        pytest.param(None, id="missing-capabilities"),
        pytest.param({}, id="empty-capabilities"),
        pytest.param({"local_dynamic_transitive_imports": True}, id="missing-symlink-true"),
        pytest.param({"local_dynamic_transitive_imports": False}, id="missing-symlink-false"),
        *(pytest.param({"local_dynamic_transitive_imports": True,
                        "local_dynamic_symlink_safe": value}, id=f"unreviewed-symlink-{value}")
          for value in (None, "true", 1, "unknown")),
    ])
    def test_missing_or_unreviewed_capability_fails_closed(self, capabilities):
        self.catalog("77.0.0", capabilities)
        with pytest.raises(ValueError, match="capabilit"):
            self.load()
        self.assert_no_loader_calls()

    def test_catalog_identity_mismatch_fails_before_import(self):
        self.catalog("66.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True},
                     version="65.0.0")
        with pytest.raises(ValueError, match="catalog"):
            self.load()
        self.assert_no_loader_calls()

    def test_native_exception_propagates_without_compat_retry(self):
        self.catalog("55.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        error = ImportError("native dependency is unavailable")
        self.dynamic.get_class_from_dynamic_module.side_effect = error
        with pytest.raises(ImportError) as raised:
            self.load()
        assert (raised.value) is (error)
        self.dynamic.get_relative_import_files.assert_not_called()


def test_public_entry_is_lazy_and_retired_module_is_rejected():
    script = '''
import importlib
import sys
from acprof.container.local_pipeline import load_local_pipeline_class
assert callable(load_local_pipeline_class)
assert "transformers" not in sys.modules
assert "acprof.container.compat.transformers_dynamic" not in sys.modules
try:
    importlib.import_module("acprof.container.dynamic_modules")
except ModuleNotFoundError as error:
    assert error.name == "acprof.container.dynamic_modules"
else:
    raise AssertionError("retired loader entry must not remain available")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert (result.returncode) == (0), result.stderr
