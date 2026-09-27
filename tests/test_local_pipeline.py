"""Capability decisions are exact, reviewed, and independent of version ordering."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from acprof.container import local_pipeline


class LocalPipelineRoutingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
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
        self.addCleanup(patcher.stop)

    def catalog(self, installed_version, capabilities, **overrides):
        self.transformers.__version__ = installed_version
        path = self.root / "extensions" / "transformers" / f"{installed_version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "version": installed_version, "mappings": {}, "capabilities": capabilities}
        data.update(overrides)
        path.write_text(json.dumps(data))
        patcher = patch("acprof.model_resolution.__file__", str(self.root / "model_resolution.py"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def load(self):
        return local_pipeline.load_local_pipeline_class(str(self.snapshot), "fixture")

    def assert_no_loader_calls(self):
        self.dynamic.get_class_from_dynamic_module.assert_not_called()
        self.dynamic.get_cached_module_file.assert_not_called()
        self.dynamic.get_relative_import_files.assert_not_called()
        self.dynamic.get_class_in_module.assert_not_called()

    def test_supported_versions_keep_compat(self):
        for version in ("4.57.6", "5.6.0"):
            with self.subTest(version=version):
                self.transformers.__version__ = version
                self.assertIs(self.load(), self.compat_class)
        self.assertEqual(self.dynamic.get_relative_import_files.call_count, 2)
        self.dynamic.get_class_from_dynamic_module.assert_not_called()

    def test_reviewed_native_uses_only_upstream_loader(self):
        self.catalog("99.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        with patch.dict("sys.modules", {"acprof.container.compat.transformers_dynamic": None}):
            self.assertIs(self.load(), self.native_class)
        self.dynamic.get_class_from_dynamic_module.assert_called_once_with(
            "entry.Pipeline", str(self.snapshot), local_files_only=True)
        self.dynamic.get_relative_import_files.assert_not_called()
        self.dynamic.get_cached_module_file.assert_not_called()
        self.dynamic.get_class_in_module.assert_not_called()

    def test_partial_capabilities_still_use_compat(self):
        for transitive, symlink in ((False, True), (True, False)):
            with self.subTest(transitive=transitive, symlink=symlink):
                self.catalog(f"88.{int(transitive)}.0", {
                    "local_dynamic_transitive_imports": transitive, "local_dynamic_symlink_safe": symlink})
                self.assertIs(self.load(), self.compat_class)
        self.dynamic.get_class_from_dynamic_module.assert_not_called()

    def test_native_capability_does_not_depend_on_version_order_or_flat_module_names(self):
        self.catalog("3.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        (self.snapshot / "config.json").write_text(json.dumps({"custom_pipelines": {
            "fixture": {"impl": "package.entry.Pipeline"}}}))
        self.assertIs(self.load(), self.native_class)
        self.dynamic.get_class_from_dynamic_module.assert_called_once_with(
            "package/entry.Pipeline", str(self.snapshot), local_files_only=True)

    def test_unknown_version_fails_before_any_loader(self):
        for version in ("99.0.1", "5.6.0+unreviewed"):
            with self.subTest(version=version):
                self.transformers.__version__ = version
                with self.assertRaisesRegex(ValueError, "[Tt]ransformers.*" + version.split("+")[0]):
                    self.load()
                self.assert_no_loader_calls()

    def test_missing_or_unreviewed_capability_fails_closed(self):
        cases = [None, {}, {"local_dynamic_transitive_imports": True},
                 {"local_dynamic_transitive_imports": False}]
        cases.extend({"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": value}
                     for value in (None, "true", 1, "unknown"))
        for index, capabilities in enumerate(cases):
            with self.subTest(capabilities=capabilities):
                self.catalog(f"77.{index}.0", capabilities)
                with self.assertRaisesRegex(ValueError, "capabilit"):
                    self.load()
                self.assert_no_loader_calls()

    def test_catalog_identity_mismatch_fails_before_import(self):
        self.catalog("66.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True},
                     version="65.0.0")
        with self.assertRaisesRegex(ValueError, "catalog"):
            self.load()
        self.assert_no_loader_calls()

    def test_native_exception_propagates_without_compat_retry(self):
        self.catalog("55.0.0", {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True})
        error = ImportError("native dependency is unavailable")
        self.dynamic.get_class_from_dynamic_module.side_effect = error
        with self.assertRaises(ImportError) as raised:
            self.load()
        self.assertIs(raised.exception, error)
        self.dynamic.get_relative_import_files.assert_not_called()


class LocalPipelineImportTests(unittest.TestCase):
    def test_public_entry_is_lazy_and_retired_module_is_rejected(self):
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
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
