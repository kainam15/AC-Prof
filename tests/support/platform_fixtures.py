"""Explicit native policy fixture for tests that simulate native collectors."""
from functools import partial
from unittest.mock import patch

from acprof.platform import Environment


def native_policy(request):
    for target in ("acprof.platform.detect_environment", "acprof.capabilities.detect_environment",
                   "acprof.host.preflight.detect_environment", "acprof.host.static_metadata.detect_environment",
                   "acprof.host.run_state.detect_environment"):
        fixture = patch(target, return_value=Environment("native_linux", "Linux", "fixture-kernel"))
        fixture.start()
        request.addfinalizer(partial(fixture.stop))
    provenance = patch("acprof.host.platform_metadata.collect_platform_metadata", return_value={"git_commit": "fixture"})
    provenance.start()
    request.addfinalizer(partial(provenance.stop))
