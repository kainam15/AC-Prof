"""Client configuration is read and validated only at the execution boundary."""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from acprof.capabilities import require_profiling_mode
from acprof.config import (
    DEFAULT_IDLE_COOLDOWN_SECONDS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_REPEAT_IN_WINDOW,
    DEFAULT_REPEAT_WINDOW_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from acprof.gpu_mode import normalize_gpu_mode
from acprof.monitors.sampling import DEFAULT_RAW_SAMPLE_LIMIT, SamplingPolicy, sampling_policy


@dataclass
class ClientConfig:
    runtime_kind: str = 'docker'
    process_pid: int = 0
    process_start_ticks: int = 0
    resource_plan: str = '{}'
    process_startup_s: str = 'nan'
    model_revision: str = 'main'
    task_family: str = 'nlp'
    pipeline_tag: str = 'text-generation'
    runtime_backend: str = 'transformers_pipeline'
    image_tag: str = ''
    cpu_cores: str = ''
    mem_cap_gb: str = ''
    gpu_mode: str = 'off'
    base_url: str = 'http://127.0.0.1:8002'
    endpoint: str = '/predict'
    batch_size: int = 1
    warmup: int = 2
    repeat: int = 5
    repeat_in_window: int = 0
    repeat_window_seconds: float = 10.0
    auto_warmup_requests: int = 5
    request_timeout_seconds: float = 300.0
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
    idle_seconds: float = 20.0
    device_index: int = 0
    gpu_device_uuid: str = ''
    idle_cooldown_seconds: float = 5.0
    input_scales_str: str = ''
    input_scale_plan_file: str = ''
    compute_profile_plan_file: str = ''
    execution_profile_plan_file: str = ''
    input_scale_order: str = ""

    @property
    def request_namespace(self) -> str:
        """Include the runtime attempt in every captured request, including preparation."""
        return f"{self.case_name}_{self.case_attempt_id}" if self.result_root else self.case_name

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClientConfig:
        env = os.environ if env is None else env
        model_revision = env.get("MODEL_REVISION", "main")
        task_family = env.get("TASK_FAMILY", "nlp")
        pipeline_tag = env.get("PIPELINE_TAG", "text-generation")
        runtime_backend = env.get("RUNTIME_BACKEND", "transformers_pipeline")
        image_tag = env.get("IMAGE_TAG", "")
        cpu_cores = env.get("CPU_CORES", "")
        mem_cap_gb = env.get("MEM_CAP_GB", "")
        gpu_mode = normalize_gpu_mode(env.get("GPU_MODE", "off"))
        base_url = env.get("BASE_URL", "http://127.0.0.1:8002").rstrip("/")
        endpoint = env.get("ENDPOINT", "/predict")
        batch_size = int(env.get("BATCH_SIZE", "1"))
        warmup = int(env.get("WARMUP", "2"))
        repeat = int(env.get("REPEAT", "5"))
        repeat_in_window = int(env.get("REPEAT_IN_WINDOW", str(DEFAULT_REPEAT_IN_WINDOW)))
        repeat_window_seconds = float(env.get("REPEAT_WINDOW_SECONDS", str(DEFAULT_REPEAT_WINDOW_SECONDS)))
        auto_warmup_requests = int(env.get("AUTO_WARMUP_REQUESTS", "5"))
        request_timeout_seconds = float(
            env.get("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        )
        cold_start_s = env.get("COLD_START_S", "nan")
        cold_start_started_at = env.get("COLD_START_STARTED_AT", "nan")
        cold_start_ready_at = env.get("COLD_START_READY_AT", "nan")
        cold_start_container_launch_s = env.get(
            "COLD_START_CONTAINER_LAUNCH_S",
            "nan",
        )
        cold_start_server_setup_s = env.get("COLD_START_SERVER_SETUP_S", "nan")
        cold_start_cuda_init_s = env.get("COLD_START_CUDA_INIT_S", "nan")
        cold_start_model_load_s = env.get("COLD_START_MODEL_LOAD_S", "nan")
        cold_start_ready_wait_s = env.get("COLD_START_READY_WAIT_S", "nan")
        out_csv = env.get("OUT_CSV", "result.csv")
        case_name = env.get("CASE_NAME", "").strip()
        container_name = env.get("CONTAINER_NAME", "").strip()
        sniff_groups_path = env.get("SNIFF_GROUPS_PATH", "").strip()
        idle_debug = env.get("IDLE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        idle_diag_path = env.get("IDLE_DIAG_PATH", "").strip()
        client_error_path = env.get("CLIENT_ERROR_PATH", "").strip()
        idle_debug_trace_interval_s = float(env.get("IDLE_DEBUG_TRACE_INTERVAL_S", "0.1"))
        use_mips = env.get("USE_MIPS", "").strip().lower() in {"1", "true", "yes", "on"}
        profiling_mode = require_profiling_mode(env.get("PROFILING_MODE", "full"))
        dram_energy = env.get("DRAM_ENERGY", "auto")
        sample_hz = float(env.get("SAMPLE_HZ", "20"))
        idle_seconds = float(env.get("IDLE_SECONDS", str(DEFAULT_IDLE_SECONDS)))
        device_index = int(env.get("DEVICE_INDEX", "0"))
        gpu_device_uuid = env.get("GPU_DEVICE_UUID", "")
        idle_cooldown_seconds = float(
            env.get("IDLE_COOLDOWN_SECONDS", str(DEFAULT_IDLE_COOLDOWN_SECONDS))
        )
        input_scales_str = env.get("INPUT_SCALES", "")
        input_scale_plan_file = env.get("INPUT_SCALE_PLAN_FILE", "").strip()
        compute_profile_plan_file = env.get("COMPUTE_PROFILE_PLAN_FILE", "").strip()
        execution_profile_plan_file = env.get(
            "EXECUTION_PROFILE_PLAN_FILE",
            "",
        ).strip()
        config = cls(
            runtime_kind=env.get('ACPROF_EXECUTION_RUNTIME', 'docker'),
            process_pid=int(env.get('ACPROF_PROCESS_PID', '0')),
            process_start_ticks=int(env.get('ACPROF_PROCESS_START_TICKS', '0')),
            resource_plan=env.get('ACPROF_RESOURCE_PLAN', '{}'),
            process_startup_s=env.get('ACPROF_PROCESS_STARTUP_S', 'nan'),
            model_revision=model_revision,
            task_family=task_family,
            pipeline_tag=pipeline_tag,
            runtime_backend=runtime_backend,
            image_tag=image_tag,
            cpu_cores=cpu_cores,
            mem_cap_gb=mem_cap_gb,
            gpu_mode=gpu_mode,
            base_url=base_url,
            endpoint=endpoint,
            batch_size=batch_size,
            warmup=warmup,
            repeat=repeat,
            repeat_in_window=repeat_in_window,
            repeat_window_seconds=repeat_window_seconds,
            auto_warmup_requests=auto_warmup_requests,
            request_timeout_seconds=request_timeout_seconds,
            cold_start_s=cold_start_s,
            cold_start_started_at=cold_start_started_at,
            cold_start_ready_at=cold_start_ready_at,
            cold_start_container_launch_s=cold_start_container_launch_s,
            cold_start_server_setup_s=cold_start_server_setup_s,
            cold_start_cuda_init_s=cold_start_cuda_init_s,
            cold_start_model_load_s=cold_start_model_load_s,
            cold_start_ready_wait_s=cold_start_ready_wait_s,
            out_csv=out_csv,
            case_name=case_name,
            container_name=container_name,
            sniff_groups_path=sniff_groups_path,
            idle_debug=idle_debug,
            idle_diag_path=idle_diag_path,
            client_error_path=client_error_path,
            idle_debug_trace_interval_s=idle_debug_trace_interval_s,
            use_mips=use_mips,
            profiling_mode=profiling_mode,
            dram_energy=dram_energy,
            sample_hz=sample_hz,
            sampling_policy=sampling_policy(env.get("ACPROF_SAMPLING_POLICY", "burst")),
            raw_sample_limit=int(env.get("ACPROF_RAW_SAMPLE_LIMIT", str(DEFAULT_RAW_SAMPLE_LIMIT))),
            run_id=env.get("ACPROF_RUN_ID", ""),
            case_id=env.get("ACPROF_CASE_ID", ""),
            case_attempt_id=env.get("ACPROF_CASE_ATTEMPT_ID", ""),
            window_records_dir=env.get("ACPROF_WINDOW_RECORDS_DIR", ""),
            telemetry_dir=env.get("ACPROF_TELEMETRY_DIR", ""),
            result_root=env.get("ACPROF_RESULT_ROOT", ""),
            idle_seconds=idle_seconds,
            device_index=device_index,
            gpu_device_uuid=gpu_device_uuid,
            idle_cooldown_seconds=idle_cooldown_seconds,
            input_scales_str=input_scales_str,
            input_scale_plan_file=input_scale_plan_file,
            compute_profile_plan_file=compute_profile_plan_file,
            execution_profile_plan_file=execution_profile_plan_file,
            input_scale_order=env.get("INPUT_SCALE_ORDER", ""),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.runtime_kind not in {'process', 'docker'}:
            raise ValueError('runtime_kind must be process or docker')
        if self.runtime_kind == 'process' and (self.process_pid <= 0 or self.process_start_ticks <= 0 or self.profiling_mode != 'basic'):
            raise ValueError('Process client requires a verified process identity and basic profiling')
        require_profiling_mode(self.profiling_mode)
        sampling_policy(self.sampling_policy)
        if isinstance(self.raw_sample_limit, bool) or not isinstance(self.raw_sample_limit, int) or self.raw_sample_limit < 2:
            raise ValueError("raw_sample_limit must be an integer of at least 2")
        evidence_config = (self.run_id, self.case_id, self.case_attempt_id, self.window_records_dir, self.telemetry_dir)
        if any(evidence_config) and not all(evidence_config):
            raise ValueError("window evidence requires run_id, case_id, case_attempt_id and both evidence directories")
        for name in ("request_timeout_seconds", "sample_hz", "repeat_window_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("idle_seconds", "idle_cooldown_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.batch_size < 1 or min(self.warmup, self.repeat, self.repeat_in_window, self.auto_warmup_requests) < 0:
            raise ValueError("invalid batch size or repeat counts")
        if self.dram_energy not in {"auto", "off", "required"}:
            raise ValueError("dram_energy must be auto, off or required")
