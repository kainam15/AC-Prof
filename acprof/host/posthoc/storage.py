"""Post-hoc locking, backups, and transactional file publication."""
from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from acprof.artifacts import atomic_write
from acprof.host.collection_history import COLLECTION_HISTORY_NAME, normalize_collection_history
from acprof.host.posthoc.context import (
    BACKUP_DIRNAME,
    RESULT_CSV_NAME,
    STATIC_META_NAME,
    PosthocError,
    ResultContext,
    _load_json_object,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def write(stream) -> None:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    atomic_write(path, write)


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_parent_directories(*paths: Path) -> None:
    seen: set[Path] = set()
    for path in paths:
        directory = path.parent
        if directory in seen:
            continue
        seen.add(directory)
        _fsync_directory(directory)


def _timestamp_token() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def create_backup(context: ResultContext) -> Path:
    from acprof.artifact_layout import ArtifactLayout
    root = ArtifactLayout.discover(context.result_dir).path(BACKUP_DIRNAME)
    root.mkdir(parents=True, exist_ok=True)
    base = _timestamp_token()
    backup = root / base
    suffix = 1
    while backup.exists():
        backup = root / f"{base}-{suffix}"
        suffix += 1
    backup.mkdir()
    try:
        sources = [
            (context.result_csv, RESULT_CSV_NAME),
            (context.static_meta_path, STATIC_META_NAME),
        ]
        if context.collection_history_existed:
            sources.append(
                (context.collection_history_path, COLLECTION_HISTORY_NAME)
            )
        for source, name in sources:
            destination = backup / name
            shutil.copy2(source, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(backup)
        _fsync_directory(root)
    except Exception:
        try:
            shutil.rmtree(backup)
        except FileNotFoundError:
            pass
        else:
            _fsync_directory(root)
        raise
    return backup


def _write_csv_temporary(
    destination: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    encoding: str,
) -> Path:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(fieldnames),
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if destination.exists()
            else 0o644
        )
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_temporary(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        mode = (
            stat.S_IMODE(destination.stat().st_mode)
            if destination.exists()
            else 0o644
        )
        temporary_path.chmod(mode)
        return temporary_path
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _restore_from_backup(destination: Path, backup_file: Path) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.restore.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(backup_file, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def commit_result_files(
    context: ResultContext,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    static_meta: Mapping[str, Any],
    collection_history: Mapping[str, Any],
    backup_dir: Path,
) -> None:
    csv_temporary = _write_csv_temporary(
        context.result_csv,
        fieldnames=fieldnames,
        rows=rows,
        encoding=context.csv_encoding,
    )
    meta_temporary = _write_json_temporary(context.static_meta_path, static_meta)
    history_temporary = _write_json_temporary(
        context.collection_history_path,
        collection_history,
    )
    csv_replaced = False
    meta_replaced = False
    history_replaced = False
    try:
        # Validate all complete temporary documents before publishing any of them.
        with csv_temporary.open(
            "r", encoding=context.csv_encoding, newline=""
        ) as f:
            if sum(1 for _row in csv.DictReader(f)) != len(rows):
                raise PosthocError("temporary result CSV row-count validation failed")
        _load_json_object(meta_temporary, "temporary static metadata")
        temporary_history = _load_json_object(
            history_temporary,
            "temporary collection history",
        )
        normalize_collection_history(temporary_history)

        os.replace(csv_temporary, context.result_csv)
        csv_replaced = True
        os.replace(meta_temporary, context.static_meta_path)
        meta_replaced = True
        os.replace(history_temporary, context.collection_history_path)
        history_replaced = True
        _fsync_parent_directories(
            context.result_csv,
            context.static_meta_path,
            context.collection_history_path,
        )
    except Exception:
        if csv_replaced:
            _restore_from_backup(
                context.result_csv, backup_dir / RESULT_CSV_NAME
            )
        if meta_replaced:
            _restore_from_backup(
                context.static_meta_path, backup_dir / STATIC_META_NAME
            )
        if history_replaced:
            history_backup = backup_dir / COLLECTION_HISTORY_NAME
            if history_backup.is_file():
                _restore_from_backup(
                    context.collection_history_path,
                    history_backup,
                )
            else:
                try:
                    context.collection_history_path.unlink()
                except FileNotFoundError:
                    pass
                else:
                    _fsync_directory(context.collection_history_path.parent)
        raise
    finally:
        for temporary in (csv_temporary, meta_temporary, history_temporary):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_process_cmdline(pid_dir: Path) -> List[str]:
    try:
        data = (pid_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return []
    return [
        part.decode("utf-8", errors="replace")
        for part in data.split(b"\0")
        if part
    ]


def _option_value(args: Sequence[str], name: str) -> Optional[str]:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def find_active_processes(
    result_dir: Path,
    *,
    model_id: str = "",
) -> List[Tuple[int, str]]:
    expected = result_dir.resolve()
    matches: List[Tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        args = _read_process_cmdline(pid_dir)
        if not args:
            continue
        command = " ".join(args)
        executable = Path(args[0]).name.lower()
        cli_subcommand = next((subcommand for program, subcommand in zip(args[:4], args[1:5])
                               if Path(program).name.startswith("acprof")
                               and subcommand in {"run", "profile"}), "")
        profiler_process = executable not in {"bash", "sh", "dash", "zsh"} and (bool(cli_subcommand) or any(
            marker in command
            for marker in (
                "acprof.cli.run",
                "compute_profile_runner",
                " ncu ",
                "/ncu ",
                "nsys",
                "massif",
                "acprof.cli.posthoc",
            )
        ))
        direct_match = str(expected) in command and profiler_process
        run_match = False
        if model_id and _option_value(args, "--model") == model_id:
            is_run_command = cli_subcommand == "run" or (
                "-m" in args and "acprof.cli.run" in args
            )
            if is_run_command:
                output_root = _option_value(args, "--output-dir") or "results"
                output_root_path = Path(output_root)
                if not output_root_path.is_absolute():
                    try:
                        output_root_path = (pid_dir / "cwd").resolve(strict=True) / output_root_path
                    except OSError:
                        continue
                candidate = output_root_path / model_id.replace("/", "--")
                run_match = candidate.resolve() == expected
        if direct_match or run_match:
            matches.append((int(pid_dir.name), command[:500]))
    return sorted(matches)


class PosthocLock:
    def __init__(self, result_dir: Path):
        from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
        self._result_lock = ResultDirectoryLock(result_dir)
        self._measurement_lock = MeasurementLock()

    def __enter__(self) -> "PosthocLock":
        self._measurement_lock.__enter__()
        try:
            self._result_lock.__enter__()
        except BaseException:
            self._measurement_lock.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self._result_lock.__exit__(exc_type, exc, traceback)
        finally:
            self._measurement_lock.__exit__(exc_type, exc, traceback)
