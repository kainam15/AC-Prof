import json
import os

import pytest

from acprof.artifact_layout import ArtifactLayout
from acprof.host.profiling import compute, execution
from acprof.host.profiling.compute_plan import load_compute_profile_plan
from acprof.host.profiling.execution_plan import load_execution_profile_plan
from tests.test_execution_profile import _task_info, _write_input_scale_plan


@pytest.mark.parametrize("kind", ["compute", "execution"])
@pytest.mark.parametrize("partial", [False, True])
def test_failed_raw_deletion_cannot_publish_or_reuse_a_plan(tmp_path, monkeypatch, kind, partial):
    layout = ArtifactLayout.discover(tmp_path)
    root = layout.path(f"{kind}_profiles")
    root.mkdir(parents=True)
    (root / "first.raw").write_bytes(b"first")
    (root / "second.raw").write_bytes(b"second")
    plan_path = layout.path(f"{kind}_profile_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"profiles": {} if kind == "compute" else [],
                           "static_metadata": {f"{kind}_profiles_retained": True}})
    plan_path.write_text(original)
    input_plan = _write_input_scale_plan(str(tmp_path))
    unlink = os.unlink
    removed = []

    def fail_unlink(path, *args, **kwargs):
        if str(path).endswith(".raw"):
            if not partial or removed:
                raise PermissionError("injected raw file deletion failure")
            removed.append(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_unlink)
    monkeypatch.setattr(execution, "require_execution_image", lambda *_: "fixture")
    monkeypatch.setattr(execution, "get_massif_version", lambda *_: "fixture")
    monkeypatch.setattr(execution, "_collect_massif", lambda *a, **k: {("massif", 1, 4): {"entries": []}})
    def collect():
        if kind == "compute":
            compute.collect_compute_profile_plan(
                task_info=_task_info(), image_tag="fixture", cpu_list=[1], mem_list=[4],
                gpu_list=["off"], output_dir=str(tmp_path), input_scale_plan_file=input_plan,
                advisor_root=None, ncu_root=None, advisor_repeat=1, ncu_repeat=1,
                keep_profiles=False, compute_profile_tool="none",
            )
        else:
            execution.collect_execution_profile_plan(
                execution.ExecutionProfileRequest(_task_info(), "fixture", [1], [4], ["off"],
                                                   str(tmp_path), input_plan),
                execution.ExecutionProfileOptions(tool_mode="massif", keep_profiles=False),
            )

    with pytest.raises(OSError, match="cleanup"):
        collect()
    assert len(list(root.glob("*.raw"))) == (1 if partial else 2)
    assert not plan_path.exists()
    loader = load_compute_profile_plan if kind == "compute" else load_execution_profile_plan
    assert "not_found" in loader(str(plan_path))["_load_error"]
    invalidated = list(plan_path.parent.glob(plan_path.name + ".invalidated-*"))
    assert len(invalidated) == 1
    assert invalidated[0].read_text() == original
