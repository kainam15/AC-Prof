"""旧参数和旧产物必须在执行或写入前明确失败。"""
import contextlib
import importlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.handlers import HandlerRegistry, resolve_model_source
from acprof.host.dependency_images import runtime_fingerprint
from acprof.host.profiler_support import load_input_scale_plan_entries
from acprof.host.profilers.ncu import _resolve_ncu_metrics, _select_ncu_flop_metrics
from acprof.host.runtime_validation import validate_runtime
from acprof.host.static_metadata import enrich_static_meta_from_input_plan
from acprof.packet.merge_packet_latency import _request_records
from acprof.plotting.data import prepare_df, read_static_meta
from acprof.run_args import build_parser
from acprof.runtime_profiles import RuntimeProfile
from acprof.tui.settings import load_settings


@pytest.mark.parametrize('suffix', ('core', 'settings', 'i18n', 'input', 'log', 'scrollbar', 'themes'))
def test_retired_tui_imports_fail(suffix):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("acprof.cli.tui_" + suffix)

def test_current_version_with_retired_settings_field_fails():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tui.json"
        path.write_text(json.dumps({"version": 4, "run_defaults": {"allow_cgroup_v1": False}}))
        original = path.read_bytes()
        with pytest.raises(ValueError, match="allow_cgroup_v1"):
            load_settings(path, Path(temporary))
        assert (path.read_bytes()) == (original)

def test_missing_baked_snapshot_does_not_fall_back_to_hub():
    with pytest.raises(FileNotFoundError, match="snapshot"):
        resolve_model_source("example/model", "/missing/acprof/model-snapshot")

@pytest.mark.parametrize('module,name', [('acprof.config', 'DEFAULT_BACKEND'), ('acprof.config', 'LIBRARY_TO_BACKEND'), ('acprof.config', 'PIPELINE_TAG_TO_FAMILY'), ('acprof.config', 'ARCHITECTURE_TO_TASK'), ('acprof.runtime_profiles', 'MOSS_MODEL_ID'), ('acprof.runtime_profiles', 'MOSS_ADAPTER'), ('acprof.runtime_profiles', 'MOSS_PROMPT'), ('acprof.runtime_profiles', 'ARCHITECTURE_PROFILES'), ('acprof.runtime_profiles', 'MODEL_PROFILES')])
def test_retired_extension_routing_exports_fail_explicitly(module, name):
    import importlib
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(module), name)
    from acprof.extensions import CATALOG
    with pytest.raises(AttributeError):
        getattr(CATALOG, "default_backend")

def test_unknown_backend_does_not_choose_another_family_handler():
    with patch.dict(HandlerRegistry._handlers, {"test:current": object()}, clear=True):
        with pytest.raises(ValueError, match="backend"):
            HandlerRegistry.get("test", "retired")

def test_old_ncu_counters_are_not_selected():
    assert (_select_ncu_flop_metrics(["flop_count_sp", "flop_count_dp"])) == ([])

def test_failed_ncu_query_does_not_guess_a_metric_list():
    failed = SimpleNamespace(returncode=1, stdout="", stderr="query unavailable")
    with patch("acprof.host.profilers.ncu.run_command", return_value=failed):
        metrics, error = _resolve_ncu_metrics("ncu")
    assert (metrics) == ([])
    assert ("query unavailable") in (error)

@pytest.mark.parametrize('arguments', (['--no-compute-profile'], ['--compute-profile-tool', 'auto'], ['--allow-cgroup-v1']))
def test_removed_cli_options_fail_during_argument_parsing(arguments):
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit) as caught:
            build_parser().parse_args(["--model", "example/model", *arguments])
        assert (caught.value.code) == (2)

@pytest.mark.parametrize('payload', ({}, {'version': 1}, {'version': 2}, {'version': 3}))
def test_old_settings_fail_without_overwriting_the_file(payload):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "tui.json"
        path.write_text(json.dumps(payload))
        original = path.read_bytes()
        with pytest.raises(ValueError, match="version|版本"):
            load_settings(path, root)
        assert (path.read_bytes()) == (original)

@pytest.mark.parametrize('version', (None, 1, True, 99))
def test_old_input_plan_is_rejected_and_current_payload_is_preserved(version):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "input_scale_plan.json"
        entry = {"input_scale": 1, "payload": {"text": "exact input"}, "input_metadata": {}}
        payload = {"entries": [entry]}
        if version is not None:
            payload["schema_version"] = version
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="schema_version"):
            load_input_scale_plan_entries(str(path))
        path.write_text(json.dumps({"schema_version": 2, "entries": [entry]}))
        assert (load_input_scale_plan_entries(str(path))[0]["payload"]) == (entry["payload"])

def test_unlocked_runtime_is_rejected_before_building():
    with pytest.raises(ValueError, match="锁|lock"):
        runtime_fingerprint(RuntimeProfile("unlocked", "nlp"))

def test_unmanaged_image_cannot_skip_runtime_validation():
    with patch("subprocess.run") as run, pytest.raises(ValueError, match="runtime_environment|运行环境"):
        validate_runtime(task_info=None, image_info=SimpleNamespace(runtime_environment={}),
                         planned=None, cpu_list=[1], mem_list=[4], gpu_list=["off"], output_dir="unused")
    run.assert_not_called()

def test_static_metadata_stand_in_is_rejected():
    with pytest.raises(TypeError):
        enrich_static_meta_from_input_plan(object(), SimpleNamespace(workload={}, plan_sha256=""))

def test_packet_flat_map_is_rejected():
    with pytest.raises(ValueError, match="schema_version|schema v2"):
        _request_records({"request-1": 0.25})
    assert (_request_records({"schema_version": 2, "requests": {
        "request-1": {"latency_s": 0.25}}})) == ({"request-1": {"latency_s": 0.25}})

def test_legacy_csv_fields_and_static_csv_fail_without_rewriting():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "result_all.csv"
        original = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,status,warmup,energy_eff_j\n1,4,on,1,ok,0,2\n"
        path.write_text(original)
        with pytest.raises(ValueError, match="energy_eff_j"):
            prepare_df(str(path))
        assert (path.read_text()) == (original)
        (path.parent / "static_meta.csv").write_text("batch_size\n1\n")
        with pytest.raises(ValueError, match="static_meta.csv"):
            read_static_meta(str(path))
