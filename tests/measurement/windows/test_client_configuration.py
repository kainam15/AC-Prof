import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.host.measurement.config import ClientConfig
from acprof.host.runtimes.service import configure_client


def test_import_does_not_read_runtime_environment_or_initialize_workload():
    script = '''
import os
from unittest.mock import patch
before = dict(os.environ)
with patch("acprof.workloads.get_generator", side_effect=AssertionError("workload during import")):
    import acprof.host.client
assert dict(os.environ) == before
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "WARMUP": "invalid-until-entrypoint", "NO_PROXY": "", "no_proxy": ""},
        capture_output=True, text=True, timeout=20)
    assert (result.returncode) == (0), result.stderr
    assert (result.stdout) == ("")


@pytest.mark.parametrize("timeout", [
    300.0,
    123.5,
    1.2345678901234567,
    300.123456789,
    math.nextafter(300.0, math.inf),
    1.2345678901234567e-5,
])
def test_client_config_environment_codec_preserves_request_timeout(timeout):
    config = ClientConfig(request_timeout_seconds=timeout)
    config.validate()

    restored = ClientConfig.from_env(config.to_env())

    assert restored.request_timeout_seconds == timeout


def test_process_session_populates_typed_client_configuration():
    session = SimpleNamespace(
        runtime_kind="process",
        base_url="http://127.0.0.1:9000",
        gpu_device={"index": 2, "uuid": "GPU-fixture"},
        process=SimpleNamespace(identity={"pid": 123, "start_ticks": 456}),
        resource_plan={"method": "inherited", "verified": True},
        startup={"process_startup_s": 1.25},
    )

    config = configure_client(ClientConfig(profiling_mode="basic"), session)
    config.validate()

    assert config.base_url == session.base_url
    assert config.runtime_kind == "process"
    assert config.process_pid == 123
    assert config.process_start_ticks == 456
    assert config.resource_plan == session.resource_plan
    assert config.resource_plan is not session.resource_plan
    assert config.process_startup_s == "1.25"
    assert config.device_index == 2
    assert config.gpu_device_uuid == "GPU-fixture"
