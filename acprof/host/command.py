"""Synchronous host commands; long-running children keep their existing owners."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

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


def run_command(
    command: Sequence[str], *, check: bool = False, capture_output: bool = True,
    timeout: float | None = None, cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None, env_overrides: Mapping[str, str] | None = None,
    redact_values: Sequence[str] = (), **kwargs: Any,
) -> CommandResult:
    """Run literal argv with UTF-8 replacement and native subprocess exceptions.

    ``env`` retains subprocess's replacement semantics; ``env_overrides`` merges
    over it (or the parent). Timeout kills/waits for the direct child, exactly as
    subprocess.run does. Process groups, retries and artifact publication belong
    to callers. Exceptions retain raw stdout/stderr and gain command_metadata;
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
        result = subprocess.run(argv, check=check, capture_output=capture_output,
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
