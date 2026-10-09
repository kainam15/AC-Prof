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
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
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
        if context.result_csv.name == "result_layers.json":
            from acprof.artifacts import read_json_object
            from acprof.result_layers import LAYER_FILES, read_result_layers
            read_result_layers(context.result_dir)  # Reject broken snapshots before backup.
            manifest = read_json_object(context.result_csv, label="result layers")
            sources.extend((context.result_dir / LAYER_FILES[layer], LAYER_FILES[layer])
                           for layer in manifest["layers"])
        if context.collection_history_existed:
            sources.append(
                (context.collection_history_path, COLLECTION_HISTORY_NAME)
            )
        for source, name in sources:
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(backup)
        _fsync_directory(root)
    except BaseException:
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
    except BaseException:
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
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
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
    except BaseException:
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


def _commit_layer_result_files(context: ResultContext, *, fieldnames, rows, static_meta,
                               collection_history, backup_dir: Path) -> None:
    """Publish profiler deltas with a recoverable multi-file transaction.

    All candidate layers and metadata are validated off to the side; the
    result manifest becomes visible only after the corresponding files exist.
    """
    from acprof.artifacts import atomic_write_json, read_json_object
    from acprof.result_layers import LAYER_FILES, publish_result_rows, read_result_layers

    original_manifest = read_json_object(context.result_csv, label="result layers")
    existing_layers = {LAYER_FILES[layer] for layer in original_manifest["layers"]}
    replaced: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".posthoc-layers-", dir=context.result_dir) as temporary:
        stage = Path(temporary)
        publish_result_rows(fieldnames, rows, stage)
        check_fields, check_rows = read_result_layers(stage)
        if len(check_rows) != len(rows) or set(check_fields) != set(fieldnames):
            raise PosthocError("staged result layers have mismatched schema or measurements")
        meta_stage, history_stage = stage / STATIC_META_NAME, stage / COLLECTION_HISTORY_NAME
        atomic_write_json(meta_stage, static_meta)
        atomic_write_json(history_stage, collection_history)
        _load_json_object(meta_stage, "staged static metadata")
        normalize_collection_history(_load_json_object(history_stage, "staged collection history"))
        staged_manifest = read_json_object(stage / RESULT_CSV_NAME, label="staged result layers")
        try:
            for layer in staged_manifest["layers"]:
                relative = LAYER_FILES[layer]
                destination = context.result_dir / relative
                candidate = stage / relative
                if destination.is_symlink():
                    raise PosthocError(f"result layer is a symlink: {destination}")
                if destination.exists() and destination.read_bytes() == candidate.read_bytes():
                    continue
                replaced.append(destination)
                atomic_write(destination, lambda stream, data=candidate.read_bytes().decode("utf-8"): stream.write(data))
            # Publish metadata before the result manifest; rollback restores both.
            for candidate, destination in (
                (meta_stage, context.static_meta_path),
                (history_stage, context.collection_history_path),
                (stage / RESULT_CSV_NAME, context.result_csv),
            ):
                replaced.append(destination)
                atomic_write(destination, lambda stream, data=candidate.read_bytes().decode("utf-8"): stream.write(data))
            _fsync_parent_directories(*replaced)
            read_result_layers(context.result_dir)
        except BaseException as primary_error:
            errors = []
            cancellation = primary_error if not isinstance(primary_error, Exception) else None
            for destination in reversed(replaced):
                relative = destination.relative_to(context.result_dir).as_posix()
                backup = backup_dir / relative
                try:
                    if backup.is_file():
                        _restore_from_backup(destination, backup)
                    elif (relative in existing_layers or destination == context.result_csv
                          or destination == context.static_meta_path
                          or (destination == context.collection_history_path and context.collection_history_existed)):
                        raise PosthocError(f"missing required recovery backup: {backup}")
                    else:
                        destination.unlink(missing_ok=True)
                        _fsync_directory(destination.parent)
                except BaseException as error:
                    errors.append(f"{destination}: {type(error).__name__}: {error}")
                    if cancellation is None and not isinstance(error, Exception):
                        cancellation = error
            if errors:
                failure = PosthocError(
                    f"layer publication failed ({primary_error}); recovery incomplete: "
                    + "; ".join(errors) + f"; backups at {backup_dir}"
                )
                if cancellation is not None:
                    raise cancellation from failure
                raise failure from primary_error
            raise


def commit_result_files(
    context: ResultContext,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    static_meta: Mapping[str, Any],
    collection_history: Mapping[str, Any],
    backup_dir: Path,
) -> None:
    temporaries: list[Path] = []
    if context.result_csv.name == "result_layers.json":
        return _commit_layer_result_files(context, fieldnames=fieldnames, rows=rows,
                                          static_meta=static_meta,
                                          collection_history=collection_history, backup_dir=backup_dir)
    attempted_publications: list[tuple[Path, str]] = []
    try:
        csv_temporary = _write_csv_temporary(
            context.result_csv,
            fieldnames=fieldnames,
            rows=rows,
            encoding=context.csv_encoding,
        )
        temporaries.append(csv_temporary)
        meta_temporary = _write_json_temporary(context.static_meta_path, static_meta)
        temporaries.append(meta_temporary)
        history_temporary = _write_json_temporary(
            context.collection_history_path,
            collection_history,
        )
        temporaries.append(history_temporary)
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

        for temporary, destination, backup_name in (
            (csv_temporary, context.result_csv, RESULT_CSV_NAME),
            (meta_temporary, context.static_meta_path, STATIC_META_NAME),
            (history_temporary, context.collection_history_path, COLLECTION_HISTORY_NAME),
        ):
            # A signal can arrive after rename succeeds but before the next statement.
            attempted_publications.append((destination, backup_name))
            os.replace(temporary, destination)
        _fsync_parent_directories(
            context.result_csv,
            context.static_meta_path,
            context.collection_history_path,
        )
        # Keep the published layer view consistent with a successful post-hoc
        # transaction. Unchanged tool CSVs are not rewritten.
        if (context.result_dir / "result_layers.json").is_file():
            from acprof.result_layers import publish_result_layers
            publish_result_layers(context.result_csv)
    except BaseException as primary_error:
        recovery_errors: list[str] = []
        cancellation = primary_error if not isinstance(primary_error, Exception) else None
        # Attempt every potentially published file, including when cleanup is cancelled.
        for destination, backup_name in attempted_publications:
            try:
                if backup_name == COLLECTION_HISTORY_NAME and not context.collection_history_existed:
                    destination.unlink(missing_ok=True)
                    _fsync_directory(destination.parent)
                else:
                    _restore_from_backup(destination, backup_dir / backup_name)
            except BaseException as recovery_error:
                recovery_errors.append(
                    f"{destination}: {type(recovery_error).__name__}: {recovery_error}"
                )
                if cancellation is None and not isinstance(recovery_error, Exception):
                    cancellation = recovery_error
        # Rebuild the prior view from the restored wide result when possible.
        # Report an incomplete recovery if storage failures also prevent repair.
        if attempted_publications and (context.result_dir / "result_layers.json").is_file():
            try:
                from acprof.result_layers import publish_result_layers
                publish_result_layers(context.result_csv)
            except BaseException as layer_error:
                recovery_errors.append(
                    f"result layers: {type(layer_error).__name__}: {layer_error}"
                )
                if cancellation is None and not isinstance(layer_error, Exception):
                    cancellation = layer_error
        if recovery_errors:
            failure = PosthocError(
                f"post-hoc publication failed ({type(primary_error).__name__}: {primary_error}); "
                f"recovery incomplete: {'; '.join(recovery_errors)}; "
                f"backups retained at {backup_dir}"
            )
            if cancellation is not None:
                raise cancellation from failure
            raise failure from primary_error
        raise
    finally:
        for temporary in temporaries:
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


def _argument_references_directory(argument: str, directory: Path) -> bool:
    """Match an argv path token, not a textual prefix of a sibling directory."""
    expected = str(directory)
    offset = 0
    while True:
        index = argument.find(expected, offset)
        if index < 0:
            return False
        end = index + len(expected)
        before_ok = index == 0 or argument[index - 1] in "=:,"
        after_ok = end == len(argument) or argument[end] == os.sep
        if before_ok and after_ok:
            return True
        offset = index + 1


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
        direct_match = profiler_process and any(
            _argument_references_directory(argument, expected) for argument in args
        )
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
