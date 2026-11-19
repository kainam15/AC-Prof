"""Explicit native policy fixture for tests that simulate native collectors."""
from functools import partial
from unittest.mock import patch

from acprof.platform import Environment


def docker_collection_plan(runtime_kind, **_kwargs):
    """Return a fresh Docker preflight result without accessing host collectors."""
    assert runtime_kind == "docker"
    return {
        "schema_version": 2, "runtime_kind": runtime_kind, "rapl_topology": {},
        "probes": {
            name: {"status": "unavailable", "detail": "fixture", "source": "fixture"}
            for name in ("cpu_energy", "cpu_instructions", "gpu_power", "packet_latency", "dram_energy")
        },
    }


def native_policy(request):
    for target in ("acprof.platform.detect_environment", "acprof.capabilities.detect_environment",
                   "acprof.host.preflight.detect_environment", "acprof.host.metadata.static.detect_environment",
                   "acprof.host.run_state.detect_environment"):
        fixture = patch(target, return_value=Environment("native_linux", "Linux", "fixture-kernel"))
        fixture.start()
        request.addfinalizer(partial(fixture.stop))
    provenance = patch("acprof.host.metadata.platform.collect_platform_metadata", return_value={"git_commit": "fixture"})
    provenance.start()
    request.addfinalizer(partial(provenance.stop))
