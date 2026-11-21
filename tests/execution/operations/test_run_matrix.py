"""Resource matrix validation stays independent of the collection workflow."""
from argparse import Namespace

import pytest

from acprof.application.resources import format_elapsed, resource_matrix


def options(**kwargs):
    fields = dict(cpus="1,2", mems="2,4", gpus="off,on", warmup=0, repeat=2,
                  execution_profile_tool="none", massif_sampling="full",
                  massif_reference_cpu=None, massif_reference_mem=None,
                  nsys_sampling="full", nsys_reference_cpu=None, nsys_reference_mem=None)
    fields.update(kwargs)
    return Namespace(**fields)


def test_matrix_canonical_values_and_elapsed():
    assert resource_matrix(options(gpus=" OFF , ON ")) == ([1, 2], [2, 4], ["off", "on"])
    assert format_elapsed(1) == "1s"
    assert format_elapsed(61) == "1m 1s"
    assert format_elapsed(3661) == "1h 1m 1s"


@pytest.mark.parametrize(("override", "error"), [
    ({"cpus": ""}, "resource lists must be non-empty"),
    ({"mems": "0"}, "resource lists must be non-empty"),
    ({"gpus": "sometimes"}, "unsupported gpu_mode"),
    ({"gpus": "on,on"}, "resource lists must not contain duplicate cases"),
    ({"cpus": "1,1"}, "resource lists must not contain duplicate cases"),
    ({"warmup": -1}, "--warmup must be >= 0 and --repeat must be > 0"),
    ({"repeat": 0}, "--warmup must be >= 0 and --repeat must be > 0"),
    ({"execution_profile_tool": "massif", "massif_sampling": "per-scale", "massif_reference_cpu": 8},
     "--massif-reference-cpu=8 must be present"),
    ({"execution_profile_tool": "nsys", "nsys_sampling": "per-scale", "nsys_reference_mem": 16},
     "--nsys-reference-mem=16 must be present"),
])
def test_invalid_matrix_aborts_before_runtime_preparation(override, error):
    with pytest.raises(ValueError, match=error):
        resource_matrix(options(**override))


@pytest.mark.parametrize(("gpu_text", "gpu_modes", "references"), [
    pytest.param("CPU", ["off"], {"massif_sampling": "per-scale", "massif_reference_cpu": 1,
                                 "nsys_sampling": "per-scale", "nsys_reference_mem": 99},
                 id="cpu-only-ignores-nsys"),
    pytest.param("TRUE", ["on"], {"massif_sampling": "per-scale", "massif_reference_cpu": 99,
                                 "nsys_sampling": "per-cpu-scale", "nsys_reference_cpu": 99, "nsys_reference_mem": 2},
                 id="gpu-only-ignores-massif-and-nsys-cpu"),
    pytest.param("0,GPU", ["off", "on"], {"massif_sampling": "full", "massif_reference_cpu": 99,
                                         "nsys_sampling": "full", "nsys_reference_mem": 99},
                 id="full-ignores-references"),
])
def test_run_matrix_ignores_unused_profiler_references(gpu_text, gpu_modes, references):
    raw = options(gpus=gpu_text, execution_profile_tool="both", **references)
    assert resource_matrix(raw) == ([1, 2], [2, 4], gpu_modes)


def test_per_cpu_scale_rejects_out_of_matrix_memory_with_the_public_option_name():
    raw = options(gpus="gpu", execution_profile_tool="nsys", nsys_sampling="per-cpu-scale",
                  nsys_reference_cpu=99, nsys_reference_mem=8)
    with pytest.raises(ValueError, match="nsys-reference-mem") as error:
        resource_matrix(raw)
    assert str(error.value) == "--nsys-reference-mem=8 must be present in the selected resource matrix [2, 4]"


def test_raw_namespace_only_needs_reference_fields_used_by_its_profiler():
    raw = Namespace(cpus="2", mems="4", gpus="gpu", warmup=0, repeat=1, execution_profile_tool="nsys",
                    nsys_sampling="per-cpu-scale", nsys_reference_mem=4)
    assert resource_matrix(raw) == ([2], [4], ["on"])
