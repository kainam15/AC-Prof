import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.profilers import execution_environment as environment

IMAGE_ID = "sha256:" + "a" * 64
MODEL_TAG = "acprof-test:latest"


def image_result(labels=None):
    return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
        "Id": IMAGE_ID, "Config": {"Labels": labels}, "RepoTags": [MODEL_TAG],
    }))


def test_both_tools_use_the_same_immutable_model_image():
    labels = {environment.EXECUTION_RUNTIME_LABEL_PREFIX + tool: "1"
              for tool in ("massif", "nsys")}
    with patch.object(environment, "run_command", return_value=image_result(labels)) as run:
        for tool in ("massif", "nsys"):
            assert (environment.require_execution_image(MODEL_TAG, tool)) == (IMAGE_ID)
    assert (run.call_count) == (2)
    for call in run.call_args_list:
        assert (call.args[0][:4]) == (["docker", "image", "inspect", MODEL_TAG])

@pytest.mark.parametrize('labels_case', range(3), ids=['None', '{}', "{environment.EXECUTION_RUNTIME_LABEL_PREFIX + tool: '0'}"])
@pytest.mark.parametrize('tool', ('massif', 'nsys'))
def test_missing_or_outdated_capability_fails_without_build_or_tag(labels_case, tool):
    labels = tuple((None, {}, {environment.EXECUTION_RUNTIME_LABEL_PREFIX + tool: '0'}))[labels_case]
    with patch.object(
        environment, "run_command", return_value=image_result(labels),
    ) as run, pytest.raises(RuntimeError, match=tool + "_runtime_unavailable.*rebuild"):
        environment.require_execution_image(MODEL_TAG, tool)
    assert (run.call_count) == (1)

def test_capability_is_checked_for_the_requested_tool():
    labels = {environment.EXECUTION_RUNTIME_LABEL_PREFIX + "nsys": "1"}
    with patch.object(environment, "run_command", return_value=image_result(labels)) as run:
        with pytest.raises(RuntimeError, match="massif_runtime_unavailable"):
            environment.require_execution_image(MODEL_TAG, "massif")
    assert (run.call_count) == (1)

def test_missing_image_does_not_trigger_a_pull_or_build():
    missing = SimpleNamespace(returncode=1, stdout="", stderr="Error: No such image: test")
    with patch.object(environment, "run_command", return_value=missing) as run:
        with pytest.raises(RuntimeError, match="nsys_runtime_unavailable.*image not found"):
            environment.require_execution_image(MODEL_TAG, "nsys")
    assert (run.call_count) == (1)

def test_daemon_failure_is_reported_without_rebuilding():
    failed = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to Docker daemon")
    with patch.object(environment, "run_command", return_value=failed) as run:
        with pytest.raises(RuntimeError, match="execution_image_inspect_failed.*Docker daemon"):
            environment.require_execution_image(MODEL_TAG, "nsys")
    assert (run.call_count) == (1)

@pytest.mark.parametrize('payload_case', range(4), ids=["'not json'", "'[]'", '\'{"Id":"latest"}\'', "json.dumps({'Id': IMAGE_ID, 'Config': {'Labels': ['invalid']}})"])
def test_invalid_image_metadata_is_rejected(payload_case):
    payload = tuple(('not json', '[]', '{"Id":"latest"}', json.dumps({'Id': IMAGE_ID, 'Config': {'Labels': ['invalid']}})))[payload_case]
    result = SimpleNamespace(returncode=0, stdout=payload, stderr="")
    with patch.object(environment, "run_command", return_value=result) as run:
        with pytest.raises(RuntimeError, match="invalid_metadata"):
            environment.require_execution_image(MODEL_TAG, "nsys")
    assert (run.call_count) == (1)
