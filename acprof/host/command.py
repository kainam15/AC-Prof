"""Synchronous host commands; long-running children keep their existing owners."""
from __future__ import annotations

import logging
import math
import os
import re
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

_LOG = logging.getLogger(__name__)
_SENSITIVE = re.compile(r"password|passwd|token|secret|credential|authorization|api[-_]?key|webhook|^key$", re.I)
_URL_USER = re.compile(r"(https?://)[^/@\s]+@", re.I)
_URL_SECRET = re.compile(r"([?&](?:token|key|api_key|access_token|password|secret)=)[^&#\s]*", re.I)


@dataclass(frozen=True)
class CommandMetadata:
    """Opt-in evidence: no output, inherited environment, or automatic file writes."""

    command: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str]
    duration_s: float
    returncode: int | None
    error_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommandResult(subprocess.CompletedProcess[str]):
    def __init__(self, result: subprocess.CompletedProcess[str], metadata: CommandMetadata):
        super().__init__(result.args, result.returncode, result.stdout, result.stderr)
        self.metadata = metadata
        self.duration_s = metadata.duration_s


def _redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = _URL_USER.sub(r"\1<redacted>@", text)
    return _URL_SECRET.sub(r"\1<redacted>", text)


def _command_metadata(command: list[str], cwd: str, overrides: Mapping[str, str],
                      secrets: Sequence[str], duration_s: float,
                      returncode: int | None, error_type: str) -> CommandMetadata:
    secrets = (*secrets, *(value for key, value in overrides.items() if _SENSITIVE.search(key)))
    redacted = []
    hide_next = False
    for part in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
        elif "=" in part and _SENSITIVE.search(part.split("=", 1)[0]):
            redacted.append(_redact(part.split("=", 1)[0], secrets) + "=<redacted>")
        else:
            redacted.append(_redact(part, secrets))
            hide_next = part.startswith("-") and bool(_SENSITIVE.search(part.lstrip("-")))
    return CommandMetadata(tuple(redacted), _redact(cwd, secrets),
                           {key: "<redacted>" if _SENSITIVE.search(key) else _redact(value, secrets)
                            for key, value in overrides.items()}, duration_s, returncode, error_type)


DOCKER_INSPECT_TIMEOUT_S = 15
DOCKER_RUN_TIMEOUT_S = 120


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass  # The whole group already exited.


def _run_process_group(argv: list[str], *, check: bool, capture_output: bool,
                       timeout: float | None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Own a POSIX command tree with bounded termination, including cancellation."""
    if os.name != "posix":
        raise ValueError("process-group commands require POSIX")
    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("process-group commands require a finite positive timeout")
    if "start_new_session" in kwargs or "process_group" in kwargs:
        raise ValueError("process-group commands own their session")
    input_data = kwargs.pop("input", None)
    if input_data is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input cannot both be supplied")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("capture_output cannot be combined with stdout/stderr")
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    try:
        try:
            stdout, stderr = process.communicate(input_data, timeout=timeout)
        except BaseException:
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # Escalate even when descendants keep inherited pipes open.
            finally:
                _signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv, stdout, stderr)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    finally:
        # A successful wrapper must not leave detached work in its owned group.
        _signal_group(process.pid, signal.SIGKILL)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def run_command(
    command: Sequence[str], *, check: bool = False, capture_output: bool = True,
    timeout: float | None = None, cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None, env_overrides: Mapping[str, str] | None = None,
    redact_values: Sequence[str] = (), kill_process_group: bool = False, **kwargs: Any,
) -> CommandResult:
    """Run literal argv with UTF-8 replacement and native subprocess exceptions.

    ``env`` retains subprocess's replacement semantics; ``env_overrides`` merges
    over it (or the parent). Timeout kills/waits for the direct child, exactly as
    subprocess.run does. Callers can opt into ``kill_process_group`` for bounded
    POSIX trees: TERM, then KILL after a 2-second grace and at most 5 seconds to
    reap the direct child. Retries and artifact publication belong to callers.
    Exceptions retain raw stdout/stderr and gain command_metadata;
    only the latter is safe to publish. Never call from a sampling window.
    """
    if isinstance(command, (str, bytes)) or kwargs.get("shell"):
        raise ValueError("host commands require literal argv with shell=False")
    for key, expected in (("text", True), ("encoding", "utf-8"), ("errors", "replace")):
        if kwargs.pop(key, expected) != expected:
            raise ValueError(f"host commands require {key}={expected!r}")
    argv = list(command)
    overrides = dict(env_overrides or {})
    command_env = dict(env) if env is not None else None
    if overrides:
        command_env = {**(os.environ if command_env is None else command_env), **overrides}
    # Record explicit changes only, never a copy of the inherited environment.
    recorded_env = {key: value for key, value in (env or {}).items() if os.environ.get(key) != value}
    recorded_env.update(overrides)
    resolved_cwd = os.path.abspath(cwd if cwd is not None else os.getcwd())
    started = perf_counter()
    result = None
    failure: BaseException | None = None
    try:
        runner = _run_process_group if kill_process_group else subprocess.run
        result = runner(argv, check=check, capture_output=capture_output,
                                timeout=timeout, cwd=cwd, env=command_env,
                                text=True, encoding="utf-8", errors="replace", **kwargs)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        returncode = result.returncode if result is not None else getattr(failure, "returncode", None)
        metadata = _command_metadata(argv, resolved_cwd, recorded_env, redact_values,
                                     perf_counter() - started, returncode,
                                     type(failure).__name__ if failure is not None else "")
        if failure is not None:
            setattr(failure, "command_metadata", metadata)
        _LOG.debug("host command %s", metadata.as_dict())
    return CommandResult(result, metadata)
