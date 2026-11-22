"""Identify the running distribution before offering any mutating command."""
from __future__ import annotations

import importlib.metadata
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from acprof.system.command import run_command
from acprof.system.uv import locate_uv
from acprof.updates.releases import Channel
from acprof.updates.worker import read_receipt, validate_uv_layout


@dataclass(frozen=True)
class Installation:
    kind: str
    prefix: Path
    python: Path
    entrypoint: Path
    manager: Path | None = None
    reason: str = ""
    channel: Channel = "stable"

    @property
    def supported(self) -> bool:
        return self.kind == "uv tool" and not self.reason

    def manual_command(self, channel: Channel) -> str:
        preview = ["--pre"] if channel == "preview" else []
        if self.kind == "uv tool":
            return shlex.join([
                "env", f"UV_TOOL_DIR={self.prefix.parent}", f"UV_TOOL_BIN_DIR={self.entrypoint.parent}",
                str(self.manager or "uv"), "tool", "install", "--force", "acprof", "--prerelease",
                "allow" if channel == "preview" else "disallow",
            ])
        if self.kind == "pipx":
            return shlex.join(["env", f"PIPX_HOME={self.prefix.parent.parent}",
                               str(self.manager or "pipx"), "upgrade", "acprof", *(["--pip-args=--pre"] if preview else [])])
        if self.kind == "venv":
            if self.manager:
                return shlex.join([str(self.manager), "pip", "install", "--python", str(self.python),
                                   "--upgrade", "acprof", "--prerelease",
                                   "allow" if channel == "preview" else "disallow"])
            return shlex.join([str(self.python), "-m", "pip", "install", "--upgrade", *preview, "acprof"])
        if self.kind in {"source", "system"}:
            return "uv tool install acprof" + (" --prerelease allow" if preview else "")
        return "https://github.com/kainam15/AC-Prof/releases"


def detect_installation() -> Installation:
    prefix = Path(sys.prefix).resolve()
    python = Path(sys.executable).absolute()
    entry = prefix / "bin/acprof"
    if getattr(sys, "frozen", False):
        return Installation("standalone", prefix, python, python, reason="standalone")
    try:
        dist = importlib.metadata.distribution("acprof")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if not isinstance(direct, dict):
            raise ValueError("Invalid direct_url.json")
        import acprof

        if direct.get("dir_info") is not None or not Path(acprof.__file__).resolve().is_relative_to(prefix):
            return Installation("source", prefix, python, entry, reason="source")
    except (importlib.metadata.PackageNotFoundError, ValueError, OSError) as exc:
        return Installation("unknown", prefix, python, entry, reason=str(exc))
    if (prefix / "uv-receipt.toml").exists():
        uv = locate_uv().evidence.get("path")
        try:
            tool = read_receipt(prefix)
            entries = tool.get("entrypoints", [])
            if (isinstance(entries, list) and entries and isinstance(entries[0], dict)
                    and isinstance(entries[0].get("install-path"), str)):
                entry = Path(entries[0].get("install-path", ""))
            tool = validate_uv_layout(prefix, entry)
            if sys.platform != "linux":
                raise ValueError("Automatic replacement currently requires Linux")
            if not uv:
                raise ValueError("uv executable not found")
            # An environment override must not redirect the running tool's update.
            for args, expected in (([], prefix.parent), (["--bin"], entry.parent)):
                result = run_command([uv, "tool", "dir", *args], check=True, timeout=5)
                if Path(result.stdout.strip()).resolve() != expected.resolve():
                    raise ValueError("uv tool dir does not match the running installation")
            if direct:
                raise ValueError("A direct URL installation needs manual source review")
            channel: Channel = "preview" if tool.get("options", {}).get("prerelease") == "allow" else "stable"
            return Installation("uv tool", prefix, python, entry, Path(uv).resolve(), channel=channel)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            return Installation("uv tool", prefix, python, entry, Path(uv) if uv else None, str(exc))
    if (prefix / "pipx_metadata.json").exists():
        return Installation("pipx", prefix, python, entry,
                            Path(shutil.which("pipx") or "pipx"), "pipx")
    if sys.prefix != sys.base_prefix:
        uv = locate_uv().evidence.get("path")
        return Installation("venv", prefix, python, entry, Path(uv) if uv else None, reason="venv")
    return Installation("system", prefix, python, entry, reason="system")
