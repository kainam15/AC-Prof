"""Ship a self-contained Docker context without workspace files or credentials."""
import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Editable installs resolve Docker resources from the live checkout.
        if version == "editable":
            return
        root = Path(self.root)
        self.bundle = tempfile.TemporaryDirectory(prefix="acprof-wheel-resources-")
        destination = Path(self.bundle.name)
        # Docker needs real Python source even when the host runs a frozen binary.
        suffixes = {".py", ".json", ".tcss", ".wav", ".md", ".txt", ".in", ".Dockerfile"}
        excluded = {"AGENTS.md", "__pycache__", "_bundle", "tests", "docs", ".git", ".github", ".codex"}
        for directory in ("acprof", "dockerfiles", "assets", "examples"):
            for path in sorted((root / directory).rglob("*")):
                if (not path.is_file() or excluded.intersection(path.relative_to(root).parts)
                        or path.suffix not in suffixes):
                    continue
                relative = path.relative_to(root).as_posix()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        shutil.copyfile(root / ".dockerignore", destination / ".dockerignore")
        for relative in ("LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt"):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        build_data["force_include"][str(destination)] = "acprof/_bundle"

    def finalize(self, version, build_data, artifact_path):
        if version != "editable":
            self.bundle.cleanup()
