"""主实验状态、目录互斥与 case 恢复；所有操作均在测量窗口之外。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json
from acprof.host.execution_conditions import measurement_environment
from acprof.messages import Message, join_messages, message
from acprof.platform import detect_environment
from acprof.result_csv import expected_measurements, read_result_csv
from acprof.run_args import flatten_run_options
from acprof.source_identity import measurement_sources, source_fingerprint

RUN_STATE_NAME = "run_state.json"
RESULT_LOCK_NAME = ".acprof-result.lock"
MAX_RUN_STATE_BYTES = 4 * 1024 * 1024
# Native Linux only. Never derive this machine-wide per-user namespace from TMPDIR.
# Test runners inject an isolated directory in-process, not via a production env option.
MEASUREMENT_LOCK_ROOT = Path("/tmp")


class RunStateError(RuntimeError):
    pass


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(message("run_state.json 包含非有限数值"))
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_identity(project_dir: str | Path) -> dict:
    source_hash = source_fingerprint(project_dir, measurement_sources(project_dir),
                                     scope="measurement-source-v2")
    packages = sorted((dist.metadata["Name"] or "", dist.version)
                      for dist in importlib.metadata.distributions())
    machine_id = Path("/etc/machine-id")
    return {
        **detect_environment().metadata(),
        "machine": platform.machine(), "kernel": platform.release(), "hostname": platform.node(),
        "machine_id_sha256": file_sha256(machine_id) if machine_id.is_file() else None,
        "python": platform.python_version(), "source_sha256": source_hash,
        "packages_sha256": hashlib.sha256(json.dumps(packages).encode()).hexdigest(),
    }


def run_options(args) -> dict:
    from acprof.host.gpu_device import selected_gpu_device
    ignored = {"resume", "skip_build", "output_dir", "notify"}
    options = {name: value for name, value in flatten_run_options(vars(args)).items() if name not in ignored}
    if options.get("revision") is None:
        options.pop("revision", None)
    if options.get("input_scale_policy", "auto") == "auto":
        options.pop("input_scale_policy", None)
    device = selected_gpu_device()
    if device:
        options["gpu_device"] = device["uuid"]
    elif options.get("gpu_device") is None:
        options.pop("gpu_device", None)
    from acprof.cpu_affinity import normalize_cpu_set
    cpuset = normalize_cpu_set(options.get("cpuset_cpus", ""))
    if cpuset:
        options["cpuset_cpus"] = cpuset
    else:
        options.pop("cpuset_cpus", None)
    for name in ("cpus", "mems", "gpus"):
        options[name] = ",".join(part.strip().lower() for part in options[name].split(",") if part.strip())
    # New options only enter pre-execution identity when explicitly requested.
    # Absent defaults retain the old serialized options; no observed values enter here.
    options["measurement_environment"] = measurement_environment()
    if options.get("workload_spec"):
        path = Path(options["workload_spec"]).expanduser().resolve()
        options["workload_spec"] = str(path)
        options["workload_spec_sha256"] = file_sha256(path)
    if options.get("model_spec"):
        path = Path(options["model_spec"]).expanduser().resolve()
        options["model_spec"] = str(path)
        options["model_spec_sha256"] = file_sha256(path)
    else:
        options.pop("model_spec", None)
    return options


def load_run_state(directory: str | Path) -> dict:
    try:
        layout = ArtifactLayout.discover(directory)
        path = layout.path(RUN_STATE_NAME)
        with path.open("rb") as stream:
            content = stream.read(MAX_RUN_STATE_BYTES + 1)
        if len(content) > MAX_RUN_STATE_BYTES:
            raise ValueError(message("run_state.json 超过 4 MiB 读取上限"))
        try:
            payload = json.loads(content.decode("utf-8"), parse_float=_finite_json_number,
                                 parse_constant=_finite_json_number)
        except json.JSONDecodeError as exc:
            raise ValueError(message("run_state.json 的 JSON 格式无效")) from exc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        reason = exc.args[0] if exc.args and isinstance(exc.args[0], Message) else str(exc)
        raise RunStateError(message(
            "无法读取恢复状态 {0}；历史实验请使用新输出目录：{1}", directory, reason,
        )) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RunStateError(message("不支持的主实验状态格式：{0}", path))
    if payload.get("layout_version", 1) != layout.layout_version:
        raise RunStateError(message("恢复状态与产物布局不一致：{0}", path))
    return payload


def _expected(data: dict, cpu=None, mem=None, gpu=None):
    options = data["options"]
    return expected_measurements(
        [cpu] if cpu is not None else [int(x) for x in options["cpus"].split(",")],
        [mem] if mem is not None else [int(x) for x in options["mems"].split(",")],
        [gpu] if gpu is not None else options["gpus"].split(","),
        data["runtime"]["planned"]["scales"], options["warmup"], options["repeat"],
    )


def recovery_phase(directory: str | Path, data: dict) -> str:
    layout = ArtifactLayout.discover(directory)
    if data.get("status") == "complete":
        return "complete"
    if (data.get("runtime") or data.get("cases") or layout.result_csv.exists()
            or any(layout.root.glob("result_case_*.csv"))
            or any((layout.root / ".acprof/work/cases").glob("*/result.csv"))):
        return "measurement"
    return "preparation"


def validate_resume(directory: str | Path, data: dict, *, project_dir: str | Path,
                    options: dict | None = None) -> str:
    """Read-only checks shared by recovery review and the locked writer.

    Image availability and live hardware preflight remain the runner's responsibility.
    Returning a phase never grants permission to mix changed measurement identities.
    """
    layout = ArtifactLayout.discover(directory)
    issues = []
    if not isinstance(data.get("options"), dict) or not isinstance(data.get("host"), dict):
        raise RunStateError(message("原实验缺少完整的参数或主机身份，无法续跑。"))
    if options is not None:
        previous = {"profiling_mode": "full", **flatten_run_options(data["options"])}
        current = {"profiling_mode": "full", **flatten_run_options(options)}
        changed = sorted(name for name in previous.keys() | current.keys()
                         if (name in previous) != (name in current) or previous.get(name) != current.get(name))
        if changed:
            issues.append(message("实验参数已变化：{0}", ", ".join(changed)))
    previous_host, current_host = data["host"], host_identity(project_dir)
    labels = {"source_sha256": "AC-Prof 采集源码", "packages_sha256": "Python 依赖",
              "python": "Python 版本"}
    for name in sorted(previous_host.keys() | current_host.keys()):
        if previous_host.get(name) != current_host.get(name):
            issues.append(message("{0}已变化", message(labels.get(name, "主机身份（{0}）"), name)))
    if issues:
        raise RunStateError(message("无法续跑；请使用新实验保留原结果。\n{0}", join_messages("\n", issues)))
    if not isinstance(data.get("cases", {}), dict) or not isinstance(data.get("artifacts", {}), dict):
        raise RunStateError(message("原实验的 case 或产物记录无效。"))
    phase = recovery_phase(directory, data)
    complete = phase == "complete"
    if not data.get("runtime") and phase != "preparation":
        raise RunStateError(message("已有测量证据但缺少冻结运行环境，不能按准备失败重试。"))
    # Completed CSVs may have documented posthoc changes; preserve that contract.
    if complete:
        read_result_csv(layout.result_csv, expected=_expected(data))
    for name, digest in data.get("artifacts", {}).items():
        if complete and Path(name).name not in {"matrix_plan.json", "startup_oom_pruning.json"}:
            continue
        path = layout.contained(name)
        if not path.is_file() or file_sha256(path) != digest:
            raise RunStateError(message("恢复产物缺失或已改变：{0}；请恢复原文件或创建新实验。", path))
    if not complete:
        for name, record in data.get("cases", {}).items():
            if not isinstance(record, dict):
                raise RunStateError(message("原实验的 case 或产物记录无效。"))
            if record.get("status") == "complete":
                path = layout.contained(name)
                if not path.is_file() or file_sha256(path) != record.get("sha256"):
                    raise RunStateError(message("已完成 case 的 CSV 缺失或改变：{0}", path))
    return phase


class ResultDirectoryLock:
    """Linux advisory lock: process exit releases ownership without stale PID recovery."""
    def __init__(self, directory: str | Path, *, new: bool = False):
        layout = ArtifactLayout.for_new_run(directory) if new else ArtifactLayout.discover(directory)
        self.path = layout.path(RESULT_LOCK_NAME)
        self.stream = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RunStateError(f"另一个采集或补采进程正在使用结果目录：{self.path.parent}") from exc
        return self

    def __exit__(self, *_args):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class MeasurementLock(ResultDirectoryLock):
    """同机同用户的实验串行化，防止跨结果目录共享端口及采样资源。"""
    def __init__(self):
        self.path = MEASUREMENT_LOCK_ROOT / f"acprof-measurement-{os.getuid()}.lock"
        self.stream = None

    def __enter__(self):
        import fcntl

        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            self.stream = os.fdopen(descriptor, "a+")
            info = os.fstat(self.stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise OSError("measurement lock is not a regular file owned by this user")
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.__exit__()
            raise RunStateError(
                "无法取得本机测量锁；同一用户可能已有 AC-Prof 采集或补采运行，"
                f"或锁文件不可用：{self.path} ({error})"
            ) from error
        return self


class RunState:
    def __init__(self, directory: str | Path, options: dict, *, resume: bool, project_dir: str,
                 preparation_artifacts: dict[str, str] | None = None):
        self.directory = Path(directory).resolve()
        self.layout = ArtifactLayout.discover(self.directory) if resume else ArtifactLayout.for_new_run(self.directory)
        self.path = self.layout.path(RUN_STATE_NAME)
        self.lock = ResultDirectoryLock(self.directory, new=not resume)
        self.measurement_lock = MeasurementLock()
        self.measurement_lock.__enter__()
        try:
            self.lock.__enter__()
            attempt = {"started_at": utc_now(), "pid": os.getpid(), "resume": resume}
            if resume:
                self.data = load_run_state(self.directory)
                phase = validate_resume(self.directory, self.data, project_dir=project_dir, options=options)
                if phase == "preparation":
                    attempt["preparation_backup"] = self._archive_preparation()
                    self.data["status"] = "preparing"
            else:
                prepared = preparation_artifacts or {}
                allowed_prepared = {str(self.layout.path(name).relative_to(self.directory))
                                    for name in ("auto_report.json", "model_resolution.json")}
                if set(prepared) - allowed_prepared:
                    raise RunStateError("自动准备产物包含不允许的文件")
                for name, digest in prepared.items():
                    path = self.layout.contained(name)
                    if not path.is_file() or file_sha256(path) != digest:
                        raise RunStateError(f"自动准备产物缺失或已改变：{path}")
                try:
                    self.layout.initialize(allowed_files=prepared)
                except ValueError as exc:
                    raise RunStateError(str(exc)) from exc
                self.data = {"schema_version": 1, "layout_version": 2, "run_id": uuid4().hex, "created_at": utc_now(),
                             "status": "preparing", "options": options, "host": host_identity(project_dir),
                             "cases": {}, "artifacts": {}, "attempts": []}
            self.data["attempts"].append(attempt)
            self.save()
        except BaseException:
            self.lock.__exit__(None, None, None)
            self.measurement_lock.__exit__(None, None, None)
            raise

    @property
    def ready(self) -> bool:
        return "runtime" in self.data

    @property
    def complete(self) -> bool:
        return self.data.get("status") == "complete"

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def artifact_path(self, relative: str) -> Path:
        try:
            return self.layout.contained(relative)
        except ValueError as exc:
            raise RunStateError(str(exc)) from exc

    def _archive_preparation(self) -> str:
        """Copy preparation evidence under the writer lock before any retry overwrites it."""
        names = ("run_state.json", "auto_report.json", "model_resolution.json",
                 "interface_validation.json", "runtime_validation.json", "static_meta.json",
                 "capability_report.json", "collection_history.json", "input_scale_plan.json",
                 "compute_profile_plan.json", "execution_profile_plan.json")
        sources = [self.layout.path(name) for name in names]
        for name in ("logs", "compute_profiles", "execution_profiles"):
            sources.extend(self.layout.path(name).rglob("*"))
        backup = self.layout.path("interrupted_cases").parent / "preparation_attempts" / uuid4().hex
        for source in sources:
            source = self.artifact_path(str(source.relative_to(self.directory)))
            if not source.exists():
                continue
            if source.is_dir():
                continue
            if not source.is_file():
                raise RunStateError(message("不支持的准备产物类型：{0}", source))
            target = backup / source.relative_to(self.directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        return str(backup.relative_to(self.directory))

    def bind_matrix_plan(self, plan: dict) -> None:
        from acprof.host.matrix_plan import MATRIX_PLAN_NAME
        from acprof.host.startup_probe import PROBE_NAME
        previous = self.data.get("matrix_plan_sha256")
        if previous is not None and previous != plan["plan_sha256"]:
            raise RunStateError("冻结 matrix plan 身份已改变")
        names = [MATRIX_PLAN_NAME]
        if plan["identity"]["prune_startup_oom"]:
            names.append(PROBE_NAME)
        for name in names:
            name = str(self.layout.path(name).relative_to(self.directory))
            checksum = file_sha256(self.artifact_path(name))
            recorded = self.data["artifacts"].get(name)
            if recorded is not None and recorded != checksum:
                raise RunStateError(f"冻结计划或 probe 证据已改变：{name}")
            self.data["artifacts"][name] = checksum
        self.data["matrix_plan_sha256"] = plan["plan_sha256"]
        self.save()

    def bind_runtime(self, task, image, planned, compute_plan: str, execution_plan: str) -> None:
        paths = [str(self.layout.path(name).relative_to(self.directory))
                 for name in ("static_meta.json", "collection_history.json", "input_scale_plan.json")]
        for value in (compute_plan, execution_plan):
            if value:
                paths.append(str(Path(value).resolve().relative_to(self.directory)))
        for name in paths:
            path = self.artifact_path(name)
            if path.is_file():
                self.data["artifacts"][name] = file_sha256(path)
        plan = asdict(planned)
        if plan["plan_file"]:
            plan["plan_file"] = str(Path(plan["plan_file"]).resolve().relative_to(self.directory))
        self.data["runtime"] = {
            "task": {key: value for key, value in asdict(task).items() if key != "repository_sources"},
            "image": asdict(image), "planned": plan,
            "compute_plan": str(Path(compute_plan).resolve().relative_to(self.directory)) if compute_plan else "",
            "execution_plan": str(Path(execution_plan).resolve().relative_to(self.directory)) if execution_plan else "",
        }
        self.data["status"] = "running"
        self.save()

    def restore_runtime(self):
        from acprof.host.detect import TaskInfo
        from acprof.host.input_plan import PlannedInputScales
        from acprof.host.runtime_images import ImageInfo, require_image_identity
        snapshot = self.data["runtime"]
        image = ImageInfo(**snapshot["image"])
        if not image.tag.startswith("sha256:"):
            raise RunStateError("恢复要求原实验的不可变 image ID；历史标签镜像请使用新输出目录")
        require_image_identity(image.tag, image.runtime_environment)
        plan = dict(snapshot["planned"])
        if plan.get("plan_file"):
            plan["plan_file"] = str(self.artifact_path(plan["plan_file"]))
        return (TaskInfo(**snapshot["task"]), image, PlannedInputScales(**plan),
                str(self.artifact_path(snapshot["compute_plan"])) if snapshot["compute_plan"] else "",
                str(self.artifact_path(snapshot["execution_plan"])) if snapshot["execution_plan"] else "")

    def expected(self, cpu=None, mem=None, gpu=None):
        return _expected(self.data, cpu, mem, gpu)

    def prepare_case(self, filename: str, cpu: int, mem: int, gpu: str) -> str | None:
        path = (self.layout.case("", cpu, mem, gpu).csv if self.layout.layout_version == 2
                else self.artifact_path(filename))
        filename = str(path.relative_to(self.directory))
        case = self.layout.case_from_csv(path)
        record = self.data["cases"].get(filename, {})
        if record.get("status") == "complete":
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                raise RunStateError(f"已完成 case 的 CSV 缺失或改变：{path}")
            read_result_csv(path, expected=self.expected(cpu, mem, gpu))
            return str(path)
        candidates = [*case.temporary_files(), case.retained_requests, case.idle]
        existing = list(dict.fromkeys(candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()))
        if existing:
            backup = self.layout.path("interrupted_cases") / case.case_id / uuid4().hex
            backup.mkdir(parents=True)
            # Preserve every source before removing any file, including partial PCAPs.
            for source in existing:
                self.artifact_path(str(source.relative_to(self.directory)))
                if source.is_symlink() or not source.is_file():
                    raise RunStateError(f"不支持的 case 产物类型：{source}")
                # Raw samples and idle diagnostics share case IDs; retain provenance.
                target = backup / source.relative_to(self.directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
            self.data["cases"][filename] = {"status": "archived", "backup": str(backup.relative_to(self.directory))}
            self.save()
            for source in existing:
                source.unlink()
        self.data["cases"][filename] = {"status": "running", "started_at": utc_now()}
        self.save()
        path.parent.mkdir(parents=True, exist_ok=True)
        return None

    def finish_case(self, path: str, cpu: int, mem: int, gpu: str) -> None:
        _, rows = read_result_csv(path, expected=self.expected(cpu, mem, gpu))
        case = self.layout.case_from_csv(path)
        case.retain_requests()
        self.data["cases"][str(Path(path).relative_to(self.directory))] = {
            "status": "complete", "completed_at": utc_now(), "sha256": file_sha256(path),
            "row_count": len(rows), "error_rows": sum(row.get("status") == "error" for row in rows),
        }
        self.save()

    def finish(self, final_csv: str) -> None:
        _, rows = read_result_csv(final_csv, expected=self.expected())
        self.data.update(status="complete", completed_at=utc_now(), result_sha256=file_sha256(final_csv),
                         row_count=len(rows), outcome="partial" if any(row.get("status") == "error" for row in rows) else "ok")
        self.save()

    def close(self, outcome: str = "interrupted") -> None:
        try:
            if not self.complete:
                self.data["status"] = outcome
            self.data["attempts"][-1]["ended_at"] = utc_now()
            self.data["attempts"][-1]["outcome"] = self.data["status"]
            self.save()
        finally:
            self.lock.__exit__(None, None, None)
            self.measurement_lock.__exit__(None, None, None)
