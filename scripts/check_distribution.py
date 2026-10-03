"""Smoke-test an installed wheel or standalone binary from an empty directory."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="默认使用当前 Python 的已安装 acprof")
    parser.add_argument("--wheel", type=Path, help="检查 wheel 的资源、许可及排除规则")
    parser.add_argument("--sdist", type=Path, help="检查 sdist 的构建 hook 与根目录约束")
    parser.add_argument("--expected-version", help="核对 wheel、sdist、安装 metadata 与 CLI 版本")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    executable = args.binary.resolve() if args.binary else Path(sysconfig.get_path("scripts")) / (
        "acprof.exe" if os.name == "nt" else "acprof")
    prefix = [str(executable)]
    evidence = []
    if args.wheel:
        with zipfile.ZipFile(args.wheel) as archive:
            names = set(archive.namelist())
            metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
            assert len(metadata_names) == 1, metadata_names
            metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        assert metadata['Name'] == 'acprof', metadata['Name']
        if args.expected_version:
            assert metadata['Version'] == args.expected_version, metadata['Version']
        bundle = "acprof/_bundle/"
        for directory in ("acprof", "dockerfiles", "assets", "examples"):
            assert any(name.startswith(f"{bundle}{directory}/") for name in names), directory
        assert {bundle + name for name in ("LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt", ".dockerignore")} <= names
        excluded = {"tests", "docs", ".git", ".github", ".codex", "AGENTS.md", "__pycache__"}
        assert not any(excluded.intersection(Path(name).parts)
                       or Path(name).parts.count("_bundle") > 1 for name in names)
        evidence.append({"check": "wheel_contents", "path": str(args.wheel), "files": len(names)})
    if args.sdist:
        with tarfile.open(args.sdist) as archive:
            names = {Path(*Path(member.name).parts[1:]) for member in archive.getmembers() if member.isfile()}
            metadata_members = [member for member in archive.getmembers()
                                if len(Path(member.name).parts) == 2 and member.name.endswith('/PKG-INFO')]
            assert len(metadata_members) == 1, metadata_members
            with archive.extractfile(metadata_members[0]) as stream:
                metadata = BytesParser().parsebytes(stream.read())
        assert metadata['Name'] == 'acprof', metadata['Name']
        if args.expected_version:
            assert metadata['Version'] == args.expected_version, metadata['Version']
        assert Path("packaging/hatch_build.py") in names
        assert not any(len(name.parts) == 1 and (name.suffix == ".py" or name.name == "acprof-tui") for name in names)
        evidence.append({"check": "sdist_contents", "path": str(args.sdist), "files": len(names)})
    with tempfile.TemporaryDirectory(prefix="acprof-install-check-") as temporary:
        workspace = Path(temporary)
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment.update({"XDG_CONFIG_HOME": str(workspace / "config"), "MPLBACKEND": "Agg"})

        def run(arguments, *, accepted=(0,), env=None):
            result = subprocess.run([*prefix, *arguments], cwd=workspace,
                                    env=env or environment, text=True, capture_output=True, timeout=90)
            evidence.append({"arguments": arguments, "returncode": result.returncode})
            if result.returncode not in accepted:
                raise RuntimeError(f"{arguments}: exit={result.returncode}\n{result.stdout}\n{result.stderr}")
            return result

        if not args.binary:
            # Reject source/editable imports even if a .pth file bypasses PYTHONPATH isolation.
            origin = subprocess.run([sys.executable, "-I", "-c",
                "import acprof,json,sys; from importlib.metadata import version; from acprof.cli.main import COMMANDS; "
                "print(json.dumps({'path':acprof.__file__,'prefix':sys.prefix,'commands':list(COMMANDS),"
                "'version':acprof.__version__,'metadata_version':version('acprof')}))"],
                cwd=workspace, env=environment, text=True, capture_output=True, timeout=30, check=True)
            installed = json.loads(origin.stdout)
            assert Path(installed["path"]).resolve().is_relative_to(Path(installed["prefix"]).resolve()), installed
            assert installed['version'] == installed['metadata_version'], installed
            if args.expected_version:
                assert installed['version'] == args.expected_version, installed
            evidence.append({"check": "installed_package_origin", "path": installed["path"]})
            commands = installed["commands"]
            # Exercise the same staging used by runtime_images, from installed resources.
            context = subprocess.run([sys.executable, "-I", "-c", """
import json
from pathlib import Path
from acprof.installation import resource_root
from acprof.source_identity import service_context_files, source_fingerprint, stage_service_context
root = resource_root()
assert root.name == '_bundle', root
destination = Path('docker-context')
stage_service_context(root, destination)
files = service_context_files(destination)
assert (destination / 'acprof/container/server.py').is_file()
assert (destination / 'dockerfiles/runtime-final.Dockerfile').is_file()
assert (destination / 'LICENSE').is_file() and (destination / 'NOTICE').is_file()
expected = source_fingerprint(root, service_context_files(root), scope='service-context-v1')
assert source_fingerprint(destination, files, scope='service-context-v1') == expected
print(json.dumps({'root': str(root), 'files': len(files), 'fingerprint': expected}))
"""], cwd=workspace, env=environment, text=True, capture_output=True, timeout=30, check=True)
            evidence.append({"check": "installed_docker_context", **json.loads(context.stdout)})
        else:
            commands = ("run", "probe", "plot", "tui", "doctor", "profile", "audit", "stats", "inspect", "auto",
                        "coverage", "report", "compare", "load", "model-store")
        version_result = run(["--version"])
        if args.expected_version:
            assert version_result.stdout.strip() == f'AC-Prof {args.expected_version}', version_result.stdout
        top_help = run(["--help"])
        assert "usage: acprof" in top_help.stdout
        for command in commands:
            help_result = run([command, "--help"])
            assert f"usage: acprof {command}" in help_result.stdout, help_result.stdout
            assert not re.search(r"\b(?:run|probe|profile|plot|audit|stats|tui)\.py\b", help_result.stdout)
        run(["invalid-command"], accepted=(2,))
        # Simulate a machine without Docker; JSON must still include valid bundled resources.
        doctor = run(["doctor", "--profiling-mode", "basic", "--json"], accepted=(1,),
                     env={**environment, "PATH": ""})
        report = json.loads(doctor.stdout)
        assert not report["ready"] and report["scope"] == "prerequisites_only", report
        assert next(item for item in report["checks"] if item["name"] == "resources")["status"] == "available", report
        assert all(path.name in {"config", "docker-context"} for path in workspace.iterdir()), "Help/doctor created result files"

        visualization_source = workspace / "visualization.csv"
        visualization_source.write_text(
            "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n"
            "2,4,off,32,0,0,ok,0.04\n", encoding="utf-8")
        visualization = workspace / "report.html"
        run(["report", str(visualization_source), "--output", str(visualization)])
        html = visualization.read_text(encoding="utf-8")
        assert "Plotly" in html and "ACProfViews" in html and "report-data" in html
        assert "__APP__" not in html and "__PLOTLY__" not in html
        run(["report", str(visualization_source), "--output", str(visualization)], accepted=(1,))

        # Exercise the actual child-process dispatcher, not just its command construction.
        source = workspace / "case.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sniff_group_id", "latency_s", "batch_size"])
            writer.writeheader()
            writer.writerow({"sniff_group_id": "req", "latency_s": "nan", "batch_size": "1"})
        (workspace / "static_meta.json").write_text(json.dumps({"schema_version": 7, "batch_size": 1}))
        metrics = workspace / "packet.json"
        metrics.write_text(json.dumps({"schema_version": 2, "requests": {"req:1": {"latency_s": 0.25}}}))
        merged = workspace / "merged.csv"
        run(["_worker", "acprof.packet.merge_packet_latency", str(source), str(metrics), str(merged)])
        with merged.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and float(rows[0]["latency_s"]) == 0.25, rows
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"successful": True, "checks": evidence}, indent=2) + "\n")
    print(f"Distribution smoke passed: {len(evidence)} checks from an empty workspace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
