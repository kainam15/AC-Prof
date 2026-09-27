"""Client configuration is read and validated only at the execution boundary."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping

from acprof.capabilities import require_profiling_mode
from acprof.config import (
    DEFAULT_REPEAT_IN_WINDOW, DEFAULT_REPEAT_WINDOW_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_IDLE_SECONDS, DEFAULT_IDLE_COOLDOWN_SECONDS,
)


@dataclass
class ClientConfig:
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
    idle_seconds: float = 20.0
    device_index: int = 0
    gpu_device_uuid: str = ''
    idle_cooldown_seconds: float = 5.0
    input_scales_str: str = ''
    input_scale_plan_file: str = ''
    compute_profile_plan_file: str = ''
    execution_profile_plan_file: str = ''
    input_scale_order: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ClientConfig":
        env = os.environ if env is None else env
        model_revision = env.get("MODEL_REVISION", "main")
        task_family = env.get("TASK_FAMILY", "nlp")
        pipeline_tag = env.get("PIPELINE_TAG", "text-generation")
        runtime_backend = env.get("RUNTIME_BACKEND", "transformers_pipeline")
        image_tag = env.get("IMAGE_TAG", "")
        cpu_cores = env.get("CPU_CORES", "")
        mem_cap_gb = env.get("MEM_CAP_GB", "")
        gpu_mode = env.get("GPU_MODE", "off").lower()
        gpu_mode = "on" if gpu_mode == "on" else "off"
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
        require_profiling_mode(self.profiling_mode)
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
