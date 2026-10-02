"""Editable installs must not ship a second copy of the checkout's sources."""
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class BuildHookTests(unittest.TestCase):
    def make_hook(self, root):
        # Hatchling is an isolated build dependency, not a host test dependency.
        interface = ModuleType("hatchling.builders.hooks.plugin.interface")
        interface.BuildHookInterface = object
        with patch.dict(sys.modules, {interface.__name__: interface}):
            hook_type = runpy.run_path(str(ROOT / "hatch_build.py"))["CustomBuildHook"]
        hook = hook_type()
        hook.root = str(root)
        return hook

    def test_editable_build_does_not_bundle_checkout(self):
        hook = self.make_hook(ROOT)
        build_data = {"force_include": {}}
        hook.initialize("editable", build_data)
        try:
            self.assertEqual(build_data["force_include"], {})
        finally:
            hook.finalize("editable", build_data, "unused.whl")

    def test_standard_build_bundles_resources_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = (
                "acprof/host/env_utils.py", "dockerfiles/platform.Dockerfile",
                "assets/sample.wav", "examples/workload.json", ".dockerignore",
                "LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt",
            )
            for relative in (*resources, "acprof/AGENTS.md", "acprof/.env.local",
                             "acprof/_bundle/stale.py", "acprof/__pycache__/stale.pyc"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            hook = self.make_hook(root)
            build_data = {"force_include": {}}
            hook.initialize("standard", build_data)
            try:
                self.assertEqual(list(build_data["force_include"].values()), ["acprof/_bundle"])
                bundle = Path(next(iter(build_data["force_include"])))
                files = {path.relative_to(bundle).as_posix(): path.read_text(encoding="utf-8")
                         for path in bundle.rglob("*") if path.is_file()}
                self.assertEqual(files, {relative: relative for relative in resources})
            finally:
                hook.finalize("standard", build_data, "unused.whl")
            self.assertFalse(bundle.exists())


if __name__ == "__main__":
    unittest.main()
