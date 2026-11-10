"""Client configuration is read and validated only at the execution boundary."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from acprof.capabilities import require_profiling_mode
from acprof.config import (
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from acprof.experiments.constraints import MEASUREMENT_NUMBERS, ConstraintViolation
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.measurement.numbers import resource_text
from acprof.monitors.sampling import DEFAULT_RAW_SAMPLE_LIMIT, SamplingPolicy, sampling_policy


def _resource_from_env(value: str) -> int | None:
    text = value.strip()
    if text.lower() in {"", "nan", "none", "inherited"}:
        return None
    return int(text)


def _resource_plan_from_env(value: str) -> dict[str, Any]:
    plan = json.loads(value)
    if not isinstance(plan, dict):
        raise ValueError("ACPROF_RESOURCE_PLAN must be a JSON object")
    return plan


def _input_scale_order_from_env(value: str) -> list[float] | None:
    if not value.strip():
        return None
    order = json.loads(value)
    if not isinstance(order, list):
        raise ValueError("INPUT_SCALE_ORDER must be a JSON list")
    return [float(item) for item in order]


def _text(value: str) -> str:
    return value


def _boolean_from_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _boolean_to_env(value: bool) -> str:
    return "1" if value else "0"


def _input_scale_order_to_env(value: list[float] | None) -> str:
    return json.dumps(value) if value is not None else ""


@dataclass(frozen=True)
class ClientEnvField:
    """One local wire binding, not a second configuration/default schema."""

    variable: str
    decode: Callable[[str], Any] = _text
    encode: Callable[[Any], str] = _text
    inherit_default: bool = False


# Preserve the existing decode order, including its first-error behavior.
# Only the three declared conditional fields inherit ambient default values.
CLIENT_ENV_FIELDS: Mapping[str, ClientEnvField] = MappingProxyType({
    "model_revision": ClientEnvField("MODEL_REVISION"),
    "task_family": ClientEnvField("TASK_FAMILY"),
    "pipeline_tag": ClientEnvField("PIPELINE_TAG"),
    "runtime_backend": ClientEnvField("RUNTIME_BACKEND"),
    "image_tag": ClientEnvField("IMAGE_TAG"),
    "cpu_cores": ClientEnvField("CPU_CORES", _resource_from_env, resource_text),
    "mem_cap_gb": ClientEnvField("MEM_CAP_GB", _resource_from_env, resource_text),
    "gpu_mode": ClientEnvField("GPU_MODE", normalize_gpu_mode),
    "base_url": ClientEnvField("BASE_URL", lambda value: value.rstrip("/")),
    "endpoint": ClientEnvField("ENDPOINT"),
    "batch_size": ClientEnvField("BATCH_SIZE", int, str),
    "warmup": ClientEnvField("WARMUP", int, str),
    "repeat": ClientEnvField("REPEAT", int, str),
    "repeat_in_window": ClientEnvField("REPEAT_IN_WINDOW", int, str),
    "repeat_window_seconds": ClientEnvField("REPEAT_WINDOW_SECONDS", float, str),
    "auto_warmup_requests": ClientEnvField("AUTO_WARMUP_REQUESTS", int, str, inherit_default=True),
    "request_timeout_seconds": ClientEnvField("REQUEST_TIMEOUT_SECONDS", float, str),
    "cold_start_s": ClientEnvField("COLD_START_S"),
    "cold_start_started_at": ClientEnvField("COLD_START_STARTED_AT"),
    "cold_start_ready_at": ClientEnvField("COLD_START_READY_AT"),
    "cold_start_container_launch_s": ClientEnvField("COLD_START_CONTAINER_LAUNCH_S"),
    "cold_start_server_setup_s": ClientEnvField("COLD_START_SERVER_SETUP_S"),
    "cold_start_cuda_init_s": ClientEnvField("COLD_START_CUDA_INIT_S"),
    "cold_start_model_load_s": ClientEnvField("COLD_START_MODEL_LOAD_S"),
    "cold_start_ready_wait_s": ClientEnvField("COLD_START_READY_WAIT_S"),
    "out_csv": ClientEnvField("OUT_CSV"),
    "case_name": ClientEnvField("CASE_NAME", str.strip),
    "container_name": ClientEnvField("CONTAINER_NAME", str.strip),
    "sniff_groups_path": ClientEnvField("SNIFF_GROUPS_PATH", str.strip, inherit_default=True),
    "idle_debug": ClientEnvField("IDLE_DEBUG", _boolean_from_env, _boolean_to_env),
    "idle_diag_path": ClientEnvField("IDLE_DIAG_PATH", str.strip),
    "client_error_path": ClientEnvField("CLIENT_ERROR_PATH", str.strip),
    "idle_debug_trace_interval_s": ClientEnvField("IDLE_DEBUG_TRACE_INTERVAL_S", float, str, inherit_default=True),
    "use_mips": ClientEnvField("USE_MIPS", _boolean_from_env, _boolean_to_env),
    "profiling_mode": ClientEnvField("PROFILING_MODE", require_profiling_mode),
    "dram_energy": ClientEnvField("DRAM_ENERGY"),
    "sample_hz": ClientEnvField("SAMPLE_HZ", float, str),
    "idle_seconds": ClientEnvField("IDLE_SECONDS", float, str),
    "device_index": ClientEnvField("DEVICE_INDEX", int, str),
    "gpu_device_uuid": ClientEnvField("GPU_DEVICE_UUID"),
    "idle_cooldown_seconds": ClientEnvField("IDLE_COOLDOWN_SECONDS", float, str),
    "input_scales_str": ClientEnvField("INPUT_SCALES"),
    "input_scale_plan_file": ClientEnvField("INPUT_SCALE_PLAN_FILE", str.strip),
    "compute_profile_plan_file": ClientEnvField("COMPUTE_PROFILE_PLAN_FILE", str.strip),
    "execution_profile_plan_file": ClientEnvField("EXECUTION_PROFILE_PLAN_FILE", str.strip),
    "runtime_kind": ClientEnvField("ACPROF_EXECUTION_RUNTIME"),
    "process_pid": ClientEnvField("ACPROF_PROCESS_PID", int, str),
    "process_start_ticks": ClientEnvField("ACPROF_PROCESS_START_TICKS", int, str),
    "resource_plan": ClientEnvField("ACPROF_RESOURCE_PLAN", _resource_plan_from_env, json.dumps),
    "process_startup_s": ClientEnvField("ACPROF_PROCESS_STARTUP_S"),
    "sampling_policy": ClientEnvField("ACPROF_SAMPLING_POLICY", sampling_policy),
    "raw_sample_limit": ClientEnvField("ACPROF_RAW_SAMPLE_LIMIT", int, str),
    "run_id": ClientEnvField("ACPROF_RUN_ID"),
    "case_id": ClientEnvField("ACPROF_CASE_ID"),
    "case_attempt_id": ClientEnvField("ACPROF_CASE_ATTEMPT_ID"),
    "window_records_dir": ClientEnvField("ACPROF_WINDOW_RECORDS_DIR"),
    "telemetry_dir": ClientEnvField("ACPROF_TELEMETRY_DIR"),
    "result_root": ClientEnvField("ACPROF_RESULT_ROOT"),
    "input_scale_order": ClientEnvField("INPUT_SCALE_ORDER", _input_scale_order_from_env, _input_scale_order_to_env),
})


@dataclass
class ClientConfig:
    runtime_kind: str = 'docker'
    process_pid: int = 0
    process_start_ticks: int = 0
    resource_plan: dict[str, Any] = field(default_factory=dict)
    process_startup_s: str = 'nan'
    model_revision: str = 'main'
    task_family: str = 'nlp'
    pipeline_tag: str = 'text-generation'
    runtime_backend: str = 'transformers_pipeline'
    image_tag: str = ''
    cpu_cores: int | None = None
    mem_cap_gb: int | None = None
    gpu_mode: str = 'off'
    base_url: str = 'http://127.0.0.1:8002'
    endpoint: str = '/predict'
    batch_size: int = 1
    warmup: int = 2
    repeat: int = 5
    repeat_in_window: int = DEFAULT_REPEAT_IN_WINDOW
    repeat_window_seconds: float = DEFAULT_REPEAT_WINDOW_SECONDS
    auto_warmup_requests: int = 5
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    cold_start_s: str = 'nan'
    cold_start_started_at: str = 'nan'
    cold_start_ready_at: str = 'nan'
    cold_start_container_launch_s: str = 'nan'
    cold_start_server_setup_s: str = 'nan'
    cold_start_cuda_init_s: str = 'nan'
    cold_start_model_load_s: str = 'nan'
    cold_start_ready_wait_s: str = 'nan'
    out_csv: str = 'result.csv'
    case_name: str = ''
    container_name: str = ''
    sniff_groups_path: str = ''
    idle_debug: bool = False
    idle_diag_path: str = ''
    client_error_path: str = ''
    idle_debug_trace_interval_s: float = 0.1
    use_mips: bool = False
    profiling_mode: str = 'full'
    dram_energy: str = 'auto'
    sample_hz: float = 20.0
    sampling_policy: SamplingPolicy = "burst"
    raw_sample_limit: int = DEFAULT_RAW_SAMPLE_LIMIT
    run_id: str = ""
    case_id: str = ""
    case_attempt_id: str = ""
    window_records_dir: str = ""
    telemetry_dir: str = ""
    result_root: str = ""
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    device_index: int = 0
    gpu_device_uuid: str = ''
    idle_cooldown_seconds: float = DEFAULT_IDLE_COOLDOWN_SECONDS
    input_scales_str: str = ''
    input_scale_plan_file: str = ''
    compute_profile_plan_file: str = ''
    execution_profile_plan_file: str = ''
    input_scale_order: list[float] | None = None

    @property
    def request_namespace(self) -> str:
        """Include the runtime attempt in every captured request, including preparation."""
        return f"{self.case_name}_{self.case_attempt_id}" if self.result_root else self.case_name

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClientConfig:
        env = os.environ if env is None else env
        defaults = cls()
        values = {
            name: spec.decode(env.get(spec.variable, spec.encode(getattr(defaults, name))))
            for name, spec in CLIENT_ENV_FIELDS.items()
        }
        config = cls(**values)
        config.validate()
        return config

    def to_env(self) -> dict[str, str]:
        defaults = ClientConfig()
        return {
            spec.variable: spec.encode(getattr(self, name))
            for name, spec in CLIENT_ENV_FIELDS.items()
            if not spec.inherit_default or getattr(self, name) != getattr(defaults, name)
        }

    def validate(self) -> None:
        if self.runtime_kind not in {'process', 'docker'}:
            raise ValueError('runtime_kind must be process or docker')
        if self.runtime_kind == 'process' and (self.process_pid <= 0 or self.process_start_ticks <= 0 or self.profiling_mode != 'basic'):
            raise ValueError('Process client requires a verified process identity and basic profiling')
        require_profiling_mode(self.profiling_mode)
        sampling_policy(self.sampling_policy)
        evidence_config = (self.run_id, self.case_id, self.case_attempt_id, self.window_records_dir, self.telemetry_dir)
        if any(evidence_config) and not all(evidence_config):
            raise ValueError("window evidence requires run_id, case_id, case_attempt_id and both evidence directories")
        for name, rule in MEASUREMENT_NUMBERS.items():
            try:
                value = rule.parse(getattr(self, name), allow_integral_float=name != "raw_sample_limit")
            except ConstraintViolation as exc:
                raise ValueError(f"{name} {exc}") from exc
            setattr(self, name, value)
        if self.dram_energy not in {"auto", "off", "required"}:
            raise ValueError("dram_energy must be auto, off or required")
