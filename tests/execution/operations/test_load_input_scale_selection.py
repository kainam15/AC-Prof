"""Replay selectors must name the frozen payload's exact scale."""
import hashlib
import json
from unittest.mock import Mock

import pytest

from acprof.cli import load as load_cli
from tests.execution.operations.test_process_load_replay import replay_environment


def test_replay_rejects_a_decimal_neighbor_before_restoring_the_service(tmp_path, monkeypatch, capsys):
    replay = replay_environment(tmp_path, monkeypatch)
    run = Mock(return_value={"successful": True, "success_rate": 1})
    monkeypatch.setattr(load_cli, "run_load", run)

    with pytest.raises(SystemExit) as refused:
        load_cli.main([*replay.argv, "--input-scale", "2.0000000000000001"])

    assert refused.value.code == 2
    assert "input scale must be present in the source experiment" in capsys.readouterr().err
    assert not replay.output.exists()
    replay.restore.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize(("selector", "scale"), [([], 2), (["--input-scale", "1.00e-1"], 0.1)])
def test_replay_uses_the_selected_plan_payload_and_recorded_scale(tmp_path, monkeypatch, selector, scale):
    replay = replay_environment(tmp_path, monkeypatch)
    plan = replay.source / "input_scale_plan.json"
    contents = json.loads(plan.read_bytes())
    alternate = {"text": "second frozen prompt"}
    contents["entries"].append({"input_scale": 0.1, "payload": alternate})
    plan.write_text(json.dumps(contents))
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    replay.state["artifacts"][plan.name] = digest
    replay.state["runtime"]["planned"]["scales"].append(0.1)
    run = Mock(return_value={"successful": True, "success_rate": 1})
    monkeypatch.setattr(load_cli, "run_load", run)

    assert load_cli.main([*replay.argv, "--warmup", "0", *selector]) == 0

    identity = json.loads((replay.output / "load.json").read_bytes())["identity"]
    assert identity["input_scale_plan_sha256"] == digest
    assert identity["input_scale"] == scale
    assert run.call_args.args[1] == (replay.payload if scale == 2 else alternate)
    assert load_cli._read_source_input_entry(plan, digest, float(scale))["payload"] == run.call_args.args[1]
