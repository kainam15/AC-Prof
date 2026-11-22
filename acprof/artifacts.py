"""基于同目录临时文件发布产物；调用方负责在测量窗口之外写入。"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from acprof.cancellation import cancellable, check_cancelled
from acprof.results.measurement_plan import decimal_coordinate, decimal_key_text
from acprof.system.cleanup import cleanup_on_exit

MAX_JSON_ARTIFACT_BYTES = 4 * 1024 * 1024
_HASH_CHUNK_BYTES = 2**18
_JSON_READ_CHUNK_BYTES = 64 * 1024


def file_sha256(path: str | Path, *, cancelled: Callable[[], bool] | None = None) -> str:
    """Hash a file with bounded memory for large experiment artifacts."""
    check_cancelled(cancelled)
    digest = hashlib.sha256()
    buffer = bytearray(_HASH_CHUNK_BYTES)
    view = memoryview(buffer)
    with Path(path).open("rb", buffering=0) as stream:
        while True:
            check_cancelled(cancelled)
            size = stream.readinto(buffer)
            if size is None:
                raise BlockingIOError("artifact hash read would block")
            if size == 0:
                break
            digest.update(view[:size])
    check_cancelled(cancelled)
    return digest.hexdigest()


def _finite_json_number(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def finite_json_decimal(raw: str) -> Decimal:
    """Read an exact decimal token in the finite JSON artifact range."""
    _finite_json_number(raw)  # Keep the artifact format's original finite range.
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("number outside the decimal range") from exc
    if not value.is_finite():  # InvalidOperation may be untrapped in the caller's context.
        raise ValueError("number outside the decimal range")
    return value


def loads_finite_json(content: str, *, exact_decimals: bool = False) -> Any:
    """Decode JSON while rejecting non-standard or overflowed finite numbers."""
    return json.loads(
        content,
        parse_float=finite_json_decimal if exact_decimals else _finite_json_number,
        parse_constant=_finite_json_number,
    )


def reject_isolated_surrogates(payload: Any) -> None:
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("JSON contains an isolated surrogate") from error
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())


@dataclass(frozen=True, slots=True)
class JsonObjectSnapshot:
    """The original bytes and finite object decoded from one open file."""

    content: bytes
    payload: dict
    exact_input_scales: dict[int, str] = field(default_factory=dict)


class _PlanNumberToken(str):
    """A bare JSON number whose spelling differs from its native float."""


def _restore_plan_numbers(container, cancelled) -> None:
    def children(value):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return iter(cancellable(items, cancelled))

    pending = [(container, children(container))]
    while pending:
        parent, items = pending[-1]
        for key, value in items:
            if isinstance(value, _PlanNumberToken):
                parent[key] = float(value)
            elif isinstance(value, (dict, list)):
                pending.append((value, children(value)))
                break
        else:
            pending.pop()


def decode_plan_json(text: str, *, inspect: Callable[[dict], None], cancelled=None) -> Any:
    """Inspect scale tokens before returning native finite JSON from the same parse."""
    marked = False

    def number(token: str) -> float | _PlanNumberToken:
        nonlocal marked
        value = _finite_json_number(token)
        if token == str(value):
            return value
        marked = True
        return _PlanNumberToken(token)

    payload = json.loads(text, parse_float=number, parse_constant=_finite_json_number)
    if isinstance(payload, dict):
        inspect(payload)
        if marked:
            _restore_plan_numbers(payload, cancelled)
    return payload


def _lossy_scale_text(token: _PlanNumberToken, field: str) -> str | None:
    try:
        coordinate = decimal_coordinate(token)
    except InvalidOperation as error:
        raise ValueError(f"{field} is outside the decimal range") from error
    if coordinate != decimal_coordinate(str(float(token))):
        return decimal_key_text(coordinate)
    return None


def _decode_input_plan(text: str, exact: dict[int, str], cancelled) -> Any:
    def inspect(plan: dict) -> None:
        entries = plan.get("entries")
        if isinstance(entries, list):
            for index, entry in enumerate(cancellable(entries, cancelled)):
                token = entry.get("input_scale") if isinstance(entry, dict) else None
                if isinstance(token, _PlanNumberToken):
                    loss = _lossy_scale_text(token, f"input_scale at index {index}")
                    if loss is not None:
                        exact[index] = loss

    return decode_plan_json(text, inspect=inspect, cancelled=cancelled)


def _read_json_bytes(
    source: Path, label: str, *, expected_sha256: str | None, bounded: bool,
    cancelled: Callable[[], bool] | None, byte_budget: list[int] | None,
) -> bytes:
    check_cancelled(cancelled)
    if byte_budget is not None and byte_budget[0] <= 0:
        raise ValueError(f"{source}: {label} read byte budget exceeded")
    # Raw reads keep the overflow probe from prefetching beyond its byte budget.
    with source.open("rb", buffering=0) as stream, BytesIO() as received:
        while True:
            check_cancelled(cancelled)
            size = _JSON_READ_CHUNK_BYTES
            if bounded:
                size = min(size, MAX_JSON_ARTIFACT_BYTES + 1 - received.tell())
            if byte_budget is not None:
                size = min(size, byte_budget[0] + 1)
            chunk = stream.read(size)
            if chunk is None:
                raise BlockingIOError("artifact JSON read would block")
            if byte_budget is not None:
                byte_budget[0] -= len(chunk)
                if byte_budget[0] < 0:
                    raise ValueError(f"{source}: {label} read byte budget exceeded")
            check_cancelled(cancelled)
            if not chunk:
                break
            received.write(chunk)
            if bounded and received.tell() > MAX_JSON_ARTIFACT_BYTES:
                raise ValueError(f"{source}: {label} exceeds the 4 MiB read limit")
        content = received.getvalue()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{source}: {label} hash does not match expected SHA256")
    return content


def _json_text(source: Path, label: str, content: bytes) -> str:
    # Historical metadata may carry one UTF-8 BOM; the byte limit includes it.
    try:
        return content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"{source}: invalid {label} JSON: {exc}") from exc


def _json_object(source: Path, label: str, text: str, cancelled: Callable[[], bool] | None,
                 decode: Callable[[str], Any]) -> dict:
    try:
        payload = decode(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{source}: invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: top-level JSON value must be an object")
    check_cancelled(cancelled)
    return payload


def read_json_object_snapshot(
    path: str | Path, *, label: str = "artifact", expected_sha256: str | None = None,
    bounded: bool = True, cancelled: Callable[[], bool] | None = None,
    byte_budget: list[int] | None = None, decode: Callable[[str], Any] | None = None,
    preserve_input_scales: bool = False,
) -> JsonObjectSnapshot:
    """Read and validate JSON without reopening it to obtain its byte identity.

    An optional shared budget counts received bytes, including at most one
    overflow probe. Growing records may opt out of the metadata limit while
    retaining cancellation, any shared budget and content validation.
    """
    if preserve_input_scales and decode is not None:
        raise ValueError("preserve_input_scales cannot be combined with a custom decode function")
    source = Path(path)
    content = _read_json_bytes(source, label, expected_sha256=expected_sha256, bounded=bounded,
                               cancelled=cancelled, byte_budget=byte_budget)
    exact: dict[int, str] = {}
    if preserve_input_scales:
        decode = partial(_decode_input_plan, exact=exact, cancelled=cancelled)
    payload = _json_object(source, label, _json_text(source, label, content), cancelled, decode or loads_finite_json)
    return JsonObjectSnapshot(content, payload, exact)


def read_json_object(
    path: str | Path, *, label: str = "artifact", expected_sha256: str | None = None,
    bounded: bool = True, cancelled: Callable[[], bool] | None = None,
    byte_budget: list[int] | None = None, exact_decimals: bool = False,
) -> dict:
    """Read finite JSON, optionally preserving decimal tokens for exact coordinate matching."""
    source = Path(path)
    content = _read_json_bytes(source, label, expected_sha256=expected_sha256, bounded=bounded,
                               cancelled=cancelled, byte_budget=byte_budget)
    text = _json_text(source, label, content)
    del content
    decode = partial(loads_finite_json, exact_decimals=True) if exact_decimals else loads_finite_json
    return _json_object(source, label, text, cancelled, decode)


def require_schema_version(payload: Any, expected: int, artifact: str) -> None:
    """在消费产物前拒绝缺失、过期或未知的协议版本。"""
    version = payload.get("schema_version") if isinstance(payload, dict) else None
    if type(version) is not int or version != expected:
        raise ValueError(
            f"{artifact}: unsupported schema_version={version!r}; expected {expected}. "
            "Use artifacts generated by the current AC-Prof version."
        )


def executable_input_scale(raw: object, field: str) -> float:
    """Validate one recorded scale against the decimal coordinate used by the runtime."""
    if isinstance(raw, _PlanNumberToken):
        loss = _lossy_scale_text(raw, field)
        if loss is not None:
            raise ValueError(f"{field} cannot round-trip through the runtime: {loss}")
        raw = float(raw)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"invalid {field}: expected a positive JSON number")
    try:
        scale = float(raw)
    except OverflowError as exc:
        raise ValueError(f"{field} exceeds the finite float range") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid {field}: expected a finite positive value")
    if isinstance(raw, int) and raw > 2**53 and raw != decimal_coordinate(str(scale)):
        raise ValueError(f"{field} cannot round-trip through the runtime: {raw!r}")
    return scale


def input_plan_scales(plan: Mapping[str, Any], *, cancelled: Callable[[], bool] | None = None) -> set[float]:
    """Validate the one-to-one mapping from effective scale to materialized payload."""
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("input_scale_plan.json has no entries")
    positions: dict[float, int] = {}
    for index, entry in enumerate(cancellable(entries, cancelled)):
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise ValueError(f"invalid input scale plan entry at index {index}: payload missing")
        scale = executable_input_scale(entry.get("input_scale"), f"input_scale at index {index}")
        if scale in positions:
            raise ValueError(f"duplicate input_scale {scale} in plan entries {positions[scale]} and {index}")
        positions[scale] = index
    return set(positions)


def read_input_scale_plan_snapshot(
    path: str | Path, *, expected_sha256: str | None = None, expected_scales: set[float] | None = None,
) -> JsonObjectSnapshot:
    """Validate the scale-to-payload mapping and retain its original bytes."""
    snapshot = read_json_object_snapshot(path, label="input scale plan", expected_sha256=expected_sha256,
                                         preserve_input_scales=True)
    require_schema_version(snapshot.payload, 2, "input_scale_plan.json")
    if snapshot.exact_input_scales:
        index, value = next(iter(snapshot.exact_input_scales.items()))
        raise ValueError(f"input_scale at index {index} cannot round-trip through the runtime: {value}")
    scales = input_plan_scales(snapshot.payload)
    if expected_scales is not None:
        expected = {executable_input_scale(scale, "expected input_scale") for scale in expected_scales}
        if len(expected) != len(expected_scales) or scales != expected:
            raise ValueError(f"{path}: input plan scales do not match the matrix scales")
    return snapshot


def read_input_scale_plan(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    """Read the current bounded input plan before any runtime or measurement starts."""
    return read_input_scale_plan_snapshot(path, expected_sha256=expected_sha256).payload


def read_static_metadata(result_dir: str | Path, *, required: bool = False) -> dict:
    """读取当前静态元数据；旧 CSV 和内嵌采集历史均不能自动转换。"""
    from acprof.config import STATIC_META_SCHEMA_VERSION
    directory = Path(result_dir)
    from acprof.results.layout import ArtifactLayout
    path = ArtifactLayout.discover(directory).path("static_meta.json")
    if not path.is_file():
        if (directory / "static_meta.csv").exists():
            raise ValueError("static_meta.csv is no longer supported; regenerate current result artifacts")
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = read_json_object(path, label="static metadata")
    version = payload.get("schema_version")
    if type(version) is not int or version not in {7, STATIC_META_SCHEMA_VERSION}:
        require_schema_version(payload, STATIC_META_SCHEMA_VERSION, str(path))
    configuration = payload.get("inference_configuration", {})
    if type(configuration) is not dict:
        raise ValueError(f"{path}: inference_configuration must be an object")
    if configuration:
        from acprof.models.inference_configuration import validate_frozen_configuration

        validate_frozen_configuration(configuration)
    history = [key for key in payload if key.endswith(("_history", "_last_run"))]
    if history:
        raise ValueError(f"{path}: collection history must be in collection_history.json: {history}")
    from acprof.platform import recorded_identity
    return {**payload, **recorded_identity(payload)}


def sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    with cleanup_on_exit(lambda: os.close(directory_fd), description=f"directory descriptor {directory}"):
        os.fsync(directory_fd)


def sync_directory_chain(directory: Path, root: Path) -> None:
    directory = directory.absolute()
    root = root.absolute()
    if not directory.is_relative_to(root):
        raise ValueError(f"Recovery directory escapes experiment root: {directory}")
    while True:
        sync_directory(directory)
        if directory == root:
            return
        directory = directory.parent


def create_durable_directory(directory: Path) -> None:
    """Persist newly created directory entries before publishing files below them."""
    missing = []
    ancestor = directory.absolute()
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in missing:
        sync_directory(created.parent)


def replace_file_durably(source: str | Path, destination: str | Path) -> None:
    """Publish an already-written file and make the rename durable on POSIX."""
    source_path = Path(source)
    destination_path = Path(destination)
    with source_path.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(source_path, destination_path)
    sync_directory(destination_path.parent)
    if source_path.parent != destination_path.parent:
        sync_directory(source_path.parent)


def atomic_write(
    path: str | Path,
    write: Callable[[TextIO], None],
    *,
    default_mode: int = 0o644,
    overwrite: bool = True,
) -> None:
    """Publish UTF-8 text; exclusive writes reject existing final entries."""
    _atomic_write(path, write, binary=False, default_mode=default_mode, overwrite=overwrite)


def atomic_write_binary(
    path: str | Path,
    write: Callable[[BinaryIO], None],
    *,
    default_mode: int = 0o644,
    overwrite: bool = True,
) -> None:
    """Publish bytes without text conversion, using the same durability contract."""
    _atomic_write(path, write, binary=True, default_mode=default_mode, overwrite=overwrite)


def staged_text_write(
    path: str | Path, write: Callable[[TextIO], None], *, default_mode: int = 0o644,
) -> AbstractContextManager[None]:
    """Prepare and fsync text on entry. Replace its name only on successful exit."""
    return _staged_write(path, write, binary=False, default_mode=default_mode, overwrite=True)


@dataclass
class CapturedBinaryArtifact:
    """An anonymous, private capture that can be published once."""
    stream: BinaryIO
    _linkable: bool

    def _link(self, destination: Path, verify_existing: Callable[[], None]) -> bool:
        fd = self.stream.fileno()
        self.stream.flush()
        os.fchmod(fd, 0o644)
        os.fsync(fd)
        try:
            # Force linkat(AT_SYMLINK_FOLLOW) for the procfs descriptor. With an
            # absolute source the kernel ignores src_dir_fd, even for a regular fd.
            os.link(f"/proc/self/fd/{fd}", destination, src_dir_fd=fd, follow_symlinks=True)
        except FileExistsError:
            verify_existing()
        except NotImplementedError:
            return False  # The interpreter cannot request linkat with these flags.
        except OSError as error:
            # A different mount or restricted/unavailable procfs cannot link
            # this descriptor. The destination-local exclusive claim still works.
            if error.errno not in {errno.EXDEV, errno.ENOENT, errno.ENOTDIR, errno.EACCES,
                                   errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
            return False
        return True

    def publish(self, destination: Path, *, verify_existing: Callable[[], None]) -> None:
        """Claim a durable name or verify a competing claim before returning."""
        if destination.exists():
            verify_existing()
            sync_directory(destination.parent)
        elif self._linkable and self._link(destination, verify_existing):
            sync_directory(destination.parent)
        else:
            self.stream.seek(0)

            def copy(stream: BinaryIO) -> None:
                shutil.copyfileobj(self.stream, stream, length=_HASH_CHUNK_BYTES)

            _atomic_write(destination, copy, binary=True, default_mode=0o644, overwrite=False,
                          on_existing=verify_existing)
        self.stream.close()


@contextmanager
def capture_binary_artifact(directory: Path) -> Iterator[CapturedBinaryArtifact]:
    """Keep unpublished bytes anonymous even if the publishing process dies."""
    flags = getattr(os, "O_TMPFILE", 0)
    stream: BinaryIO | None = None
    if flags:
        # tempfile.TemporaryFile also sets O_EXCL, which prevents later linking.
        def open_linkable(path: str, _: int) -> int:
            return os.open(path, flags | os.O_RDWR, 0o600)

        try:
            stream = open(directory, "w+b", buffering=0, opener=open_linkable)
        except OSError as error:
            if error.errno == errno.ENOENT and not directory.is_dir():
                raise
            if error.errno not in {errno.EOPNOTSUPP, errno.EISDIR, errno.EINVAL, errno.ENOENT,
                                   errno.EPERM, errno.EACCES, errno.ENOSYS}:
                raise
    linkable = stream is not None
    if stream is None:
        stream = tempfile.TemporaryFile(dir=directory, buffering=0)
    with cleanup_on_exit(stream.close, description="anonymous artifact capture"):
        yield CapturedBinaryArtifact(stream, linkable)


def publish_immutable_bytes(destination: Path, payload: bytes, *, conflict: str,
                            before_publish: Callable[[], None] | None = None) -> None:
    """Run any publication prerequisite after checking a visible file for conflicts.

    Concurrent claims install at most one file. Existing content must match byte for byte.
    """
    def require_match() -> None:
        if destination.read_bytes() != payload:
            raise ValueError(f"{conflict}: {destination}")

    exists = destination.exists()
    if exists:
        require_match()
    if before_publish is not None:
        before_publish()
    if exists:
        sync_directory(destination.parent)
        return

    def write(stream: BinaryIO) -> None:
        stream.write(payload)

    _atomic_write(destination, write, binary=True, default_mode=0o644, overwrite=False, on_existing=require_match)


def _atomic_write(
    path: str | Path,
    write: Callable[[Any], None],
    *,
    binary: bool,
    default_mode: int,
    overwrite: bool,
    on_existing: Callable[[], None] | None = None,
) -> None:
    with _staged_write(path, write, binary=binary, default_mode=default_mode,
                       overwrite=overwrite, on_existing=on_existing):
        pass


@contextmanager
def _staged_write(
    path: str | Path, write: Callable[[Any], None], *, binary: bool, default_mode: int,
    overwrite: bool, on_existing: Callable[[], None] | None = None,
) -> Iterator[None]:
    destination = Path(path)
    create_durable_directory(destination.parent)
    mode = stat.S_IMODE(destination.stat().st_mode) if overwrite and destination.exists() else default_mode
    stream = tempfile.NamedTemporaryFile(mode="wb" if binary else "w",
                                        encoding=None if binary else "utf-8", newline=None if binary else "",
                                        dir=destination.parent, prefix=f".{destination.name}.",
                                        suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    cleanup_required = True

    def remove_temporary() -> None:
        if cleanup_required:
            temporary.unlink(missing_ok=True)

    with cleanup_on_exit(remove_temporary, description=f"temporary artifact {temporary}"):
        with cleanup_on_exit(stream.close, description=f"temporary stream {temporary}"):
            write(stream)
            stream.flush()
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), mode)
            else:
                os.chmod(temporary, mode)
            os.fsync(stream.fileno())
        yield
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Same-directory hard linking checks existence at publication time.
            # Unsupported filesystems must fail, never fall back to replacement.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if on_existing is None:
                    raise
                on_existing()
            temporary.unlink()
        cleanup_required = False
        sync_directory(destination.parent)


def atomic_write_json(path: str | Path, payload: Any, *, overwrite: bool = True) -> None:
    """Publish JSON; exclusive writes require a new destination entry."""
    def write(stream: TextIO) -> None:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    atomic_write(path, write, overwrite=overwrite)
