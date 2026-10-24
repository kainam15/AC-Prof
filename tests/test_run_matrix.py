"""Resource matrix validation stays independent of the collection workflow."""
from argparse import Namespace

import pytest

from acprof.cli.run_matrix import _format_elapsed, _resource_matrix


class Parser:
    def error(self, message):
        raise ValueError(message)


def options(**kwargs):
    fields = dict(cpus="1,2", mems="2,4", gpus="off,on", warmup=0, repeat=2,
                  execution_profile_tool="none", massif_sampling="full",
                  massif_reference_cpu=None, massif_reference_mem=None,
                  nsys_sampling="full", nsys_reference_cpu=None, nsys_reference_mem=None)
    fields.update(kwargs)
    return Namespace(**fields)


def test_matrix_canonical_values_and_elapsed():
    assert _resource_matrix(options(gpus=" OFF , ON "), Parser()) == ([1, 2], [2, 4], ["off", "on"])
    assert _format_elapsed(1) == "1s"
    assert _format_elapsed(61) == "1m 1s"
    assert _format_elapsed(3661) == "1h 1m 1s"


@pytest.mark.parametrize("override", [
    {"cpus": ""}, {"mems": "0"}, {"gpus": "sometimes"}, {"gpus": "on,on"},
    {"cpus": "1,1"}, {"warmup": -1}, {"repeat": 0},
    {"execution_profile_tool": "massif", "massif_sampling": "per-scale", "massif_reference_cpu": 8},
    {"execution_profile_tool": "nsys", "nsys_sampling": "per-scale", "nsys_reference_mem": 16},
])
def test_invalid_matrix_aborts_before_runtime_preparation(override):
    with pytest.raises(ValueError):
        _resource_matrix(options(**override), Parser())
