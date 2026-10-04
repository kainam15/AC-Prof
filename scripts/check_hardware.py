"""在专用 Linux/Docker 主机执行小型真实采集，保存命令、日志和审计结果。"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.analysis.audit import audit_result, number  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.artifacts import (  # noqa: E402 -- 脚本先设置仓库导入路径。
    atomic_write_json,
    read_json_object,
    require_schema_version,
)
from acprof.experiment import RunConfig, build_run_command  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.host.run_state import host_identity  # noqa: E402 -- 脚本先设置仓库导入路径。
from acprof.result_csv import read_result_csv  # noqa: E402 -- 脚本先设置仓库导入路径。


def run_config(config: RunConfig, output: Path, *, ui: str = "cli") -> dict:
    if ui == "terminal" and not sys.stdout.isatty():
        raise ValueError("terminal 对照须在终端中运行；自动化可用 script 分配 PTY 并保存终端输出")
    command = build_run_command(config, project_dir=ROOT, python_executable=sys.executable)
    environment = os.environ.copy()
    environment.update(ACPROF_WECOM_WEBHOOK_URL="", PYTHONUNBUFFERED="1")
    environment.pop("TMUX", None)
    environment.pop("TMUX_PANE", None)
    atomic_write_json(output / "command.json", {"command": command, "config": asdict(config), "ui": ui})
    launch = command if ui == "cli" else [sys.executable, str(ROOT / "scripts/run_tui_validation.py"),
                                          str(output / "command.json"), "--ui", ui]
    if ui == "terminal":
        (output / "run.log").write_text("TUI 渲染输出到当前终端；可用 script 保存会话。\n")
        code = subprocess.run(launch, cwd=ROOT, env=environment).returncode
    else:
        with (output / "run.log").open("w") as log:
            code = subprocess.run(launch, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    audit = audit_result(config.result_dir(ROOT))
    audit["exit_code"] = code
    audit["successful"] = code == 0 and audit["valid"] and audit["completion"] == "complete"
    counts = audit["counts"]
    audit["successful"] &= counts["formal_ok"] > 0 and counts["formal_ok"] == counts["rows"] - counts["warmup"]
    unavailable = []
    if audit["valid"]:
        _, rows = read_result_csv(config.result_csv(ROOT))
        for index, row in enumerate(rows, 2):
            if row["warmup"] == "0":
                required = ["latency_s", "latency_app_s", "cpu_energy_total_j", "cpu_instructions_per_request"]
                if row["gpu_mode"] == "on":
                    required.append("gpu_energy_total_j")
                unavailable.extend({"row": index, "field": name} for name in required if number(row.get(name)) is None)
    audit["required_metrics_missing"] = unavailable
    audit["successful"] &= not unavailable
    atomic_write_json(output / "audit.json", audit)
    return audit


def load_hardware_matrix(path: Path, defaults: RunConfig, output: Path) -> list[tuple[str, RunConfig]]:
    """Validate every pinned case before creating outputs or launching a workload."""
    payload = read_json_object(path, label="hardware matrix")
    require_schema_version(payload, 1, "hardware matrix")
    cases = payload.get("cases")
    if set(payload) != {"schema_version", "cases"} or not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError("hardware matrix requires 1..16 cases and no unknown fields")
    selected, names = [], set()
    required = {"name", "model", "revision", "task", "input_scales"}
    for case in cases:
        if not isinstance(case, dict) or set(case) != required or any(
            not isinstance(value, str) or not value.strip() for value in case.values()
        ):
            raise ValueError("hardware matrix case requires name, model, revision, task and input_scales")
        name = case["name"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("hardware matrix case names must be unique safe directory names")
        if not re.fullmatch(r"[0-9a-f]{40}", case["revision"]):
            raise ValueError("hardware matrix revision must be an immutable 40-character commit SHA")
        names.add(name)
        config = replace(defaults, **{key: case[key] for key in required - {"name"}},
                         output_dir=str(output / name / "results")).validate(project_dir=ROOT)
        selected.append((name, config))
    return selected


def run_matrix(cases: list[tuple[str, RunConfig]], output: Path, *, ui: str) -> dict:
    """Run cases serially and preserve failed/cancelled/not-run evidence separately."""
    report = {"schema_version": 1, "kind": "hardware_matrix", "successful": False,
              "source": host_identity(ROOT), "ci_commit": os.environ.get("GITHUB_SHA"),
              "cases": [{"name": name, "config": asdict(config), "output_dir": str(output / name),
                         "status": "not_run"} for name, config in cases]}
    destination = output / "hardware_matrix.json"
    atomic_write_json(destination, report)
    try:
        for row, (name, config) in zip(report["cases"], cases):
            row["status"] = "running"
            atomic_write_json(destination, report)
            try:
                (output / name).mkdir()
                audit = run_config(config, output / name, ui=ui)
                row["status"] = "passed" if audit["successful"] is True else "failed"
                row["audit"] = audit
            except BaseException as exc:
                row.update(status="cancelled" if not isinstance(exc, Exception) else "failed",
                           error=f"{type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception):
                    raise
            finally:
                atomic_write_json(destination, report)
            if row["status"] != "passed":
                break  # Cleanup or measurement failure must not start another case.
        report["successful"] = all(row["status"] == "passed" for row in report["cases"])
    finally:
        atomic_write_json(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model")
    selection.add_argument("--matrix", type=Path, help="固定模型 revision 的硬件验收矩阵 JSON")
    parser.add_argument("--revision", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--gpus", default="off,on")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--mems", default="8")
    parser.add_argument("--input-scales")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--repeat-window-seconds", type=float, default=2.0)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--compute-profile-tool", choices=("none", "torch", "ncu", "both"), default="none")
    parser.add_argument("--execution-profile-tool", choices=("none", "massif", "nsys", "both"), default="none")
    parser.add_argument("--ui", choices=("cli", "headless", "terminal"), default="cli")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("硬件验证输出目录必须为空")
    config = RunConfig(model=args.model or "", revision=args.revision, task=args.task,
                       cpus=args.cpus, mems=args.mems, gpus=args.gpus,
                       input_scales=args.input_scales or "64", output_dir=str(output / "results"),
                       warmup=1, repeat=args.repeat, repeat_window_seconds=args.repeat_window_seconds,
                       idle_seconds=args.idle_seconds, idle_cooldown_seconds=1.0, sample_hz=args.sample_hz,
                       compute_profile_tool=args.compute_profile_tool, execution_profile_tool=args.execution_profile_tool,
                       notify="none", skip_build=True)
    try:
        if args.matrix:
            if args.revision or args.task or args.input_scales is not None:
                parser.error("--matrix 中定义 revision、task 和 input_scales，不接受对应命令行覆盖")
            cases = load_hardware_matrix(args.matrix, config, output)
        else:
            config = config.validate(project_dir=ROOT)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    report = run_matrix(cases, output, ui=args.ui) if args.matrix else run_config(config, output, ui=args.ui)
    print(f"硬件验证{'通过' if report['successful'] else '未通过'}，证据：{output}")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
