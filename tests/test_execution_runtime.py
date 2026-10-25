"""Execution selection is independent of model loading and measurement intensity."""
from dataclasses import replace

import pytest

from acprof.experiment import RunConfig
from acprof.platform import Environment, collection_policy_error
from acprof.run_args import build_parser


def test_process_runtime_is_a_shared_cli_and_config_option():
    args = build_parser().parse_args([
        "--model", "example/model", "--runtime", "process", "--profiling-mode", "basic",
        "--cpus", "auto", "--mems", "auto", "--gpus", "off",
    ])
    config = RunConfig.from_namespace(args).validate()
    assert config.runtime == "process"
    assert config.cpus == config.mems == "auto"


def test_restricted_linux_allows_basic_but_not_full():
    for kind in ("container_host", "vm", "cloud"):
        env = Environment(kind, "Linux")
        assert collection_policy_error(env, profiling_mode="basic") == ""
        assert collection_policy_error(env, profiling_mode="full")


def test_explicit_full_is_not_silently_changed_for_process():
    config = replace(RunConfig.smoke("example/model"), runtime="process", profiling_mode="full")
    with pytest.raises(ValueError, match="Process.*basic"):
        config.validate()
