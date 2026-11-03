"""Dispatch inference service ownership to the selected execution adapter."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from acprof.host.runtimes.contracts import runtime_kind

if TYPE_CHECKING:
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.measurement.config import ClientConfig
    from acprof.host.runtimes.contracts import ServiceSession
    from acprof.host.runtimes.process import RunningProcess


def execution_runtime(kind):
    if kind == "docker":
        from acprof.host.runtimes.docker import DockerRuntime
        return DockerRuntime()
    if kind == "process":
        from acprof.host.runtimes.process import ProcessRuntime
        return ProcessRuntime()
    raise ValueError(f"unknown execution runtime: {kind}")


def start_service_session(**kwargs):
    return execution_runtime(runtime_kind(kwargs["image_info"])).start(**kwargs)


def stop_service_session(session, log_prefix=None):
    execution_runtime(getattr(session, "runtime_kind", "docker")).stop(session, log_prefix=log_prefix)


def service_failure(session, memory, returncode):
    return execution_runtime(getattr(session, "runtime_kind", "docker")).failure(session, memory, returncode)


def configure_client(config: ClientConfig, session: ServiceSession) -> ClientConfig:
    if getattr(session, "runtime_kind", "docker") == "process":
        process_session = cast("RunningProcess", session)
        return replace(
            config,
            base_url=session.base_url,
            device_index=int(session.gpu_device.get("index", 0)),
            gpu_device_uuid=str(session.gpu_device.get("uuid", "")),
            runtime_kind="process",
            process_pid=process_session.process.identity["pid"],
            process_start_ticks=process_session.process.identity["start_ticks"],
            resource_plan=dict(process_session.resource_plan),
            process_startup_s=str(process_session.startup.get("process_startup_s", "nan")),
        )
    docker_session = cast("RunningContainer", session)
    return replace(
        config,
        base_url=session.base_url,
        device_index=int(session.gpu_device.get("index", 0)),
        gpu_device_uuid=str(session.gpu_device.get("uuid", "")),
        runtime_kind="docker",
        resource_plan={"method": "docker_cgroup"},
        process_startup_s="nan",
        cold_start_started_at=docker_session.cold_start_started_at,
        cold_start_ready_at=docker_session.cold_start_ready_at,
        cold_start_container_launch_s=str(docker_session.cold_start_container_launch_s),
        cold_start_server_setup_s=str(docker_session.cold_start_server_setup_s),
        cold_start_cuda_init_s=str(docker_session.cold_start_cuda_init_s),
        cold_start_model_load_s=str(docker_session.cold_start_model_load_s),
        cold_start_ready_wait_s=str(docker_session.cold_start_ready_wait_s),
        cold_start_s=f"{docker_session.cold_start_s:.6f}",
    )


def bind_session(state, path, session):
    if getattr(session, "runtime_kind", "docker") == "process":
        state.bind_case_process(path, session.process.evidence())
    else:
        state.bind_case_container(path, session.container_id)
