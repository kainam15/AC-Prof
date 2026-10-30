"""Dispatch inference service ownership to the selected execution adapter."""
from __future__ import annotations

from acprof.host.runtimes.contracts import runtime_kind


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


def client_environment(session):
    if getattr(session, "runtime_kind", "docker") == "process":
        import json

        from acprof.results.metrics import METRICS
        return {**{name.upper(): "nan" for name in METRICS if name.startswith("cold_start_")},
                "ACPROF_EXECUTION_RUNTIME": "process",
                "ACPROF_PROCESS_PID": str(session.process.identity["pid"]),
                "ACPROF_PROCESS_START_TICKS": str(session.process.identity["start_ticks"]),
                "ACPROF_RESOURCE_PLAN": json.dumps(session.resource_plan),
                "ACPROF_PROCESS_STARTUP_S": str(session.startup.get("process_startup_s", "nan"))}
    from acprof.host.docker_runtime import cold_start_client_env
    return {"ACPROF_EXECUTION_RUNTIME": "docker", "ACPROF_PROCESS_STARTUP_S": "nan",
            "ACPROF_RESOURCE_PLAN": '{"method":"docker_cgroup"}', **cold_start_client_env(session)}


def bind_session(state, path, session):
    if getattr(session, "runtime_kind", "docker") == "process":
        state.bind_case_process(path, session.process.evidence())
    else:
        state.bind_case_container(path, session.container_id)
