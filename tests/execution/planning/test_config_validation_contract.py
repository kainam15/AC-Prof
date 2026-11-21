"""Public configuration behavior shared by CLI, forms, recovery and preflight."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from acprof.experiments.configuration import RunConfig, RunConfigError
from acprof.messages import Message


def test_identity_source_and_resource_issues_keep_their_order_and_messages():
    config = replace(
        RunConfig.smoke(" "), task_run_id="bad/id", tasks="bad task,also/bad",
        resume=True, rebuild_image=True, extra_options={"ncu_repeat": 0},
        revision="bad ref", model_source="invalid", download_mode="invalid",
        cpuset_cpus="invalid", cpus="0", mems="invalid", gpus="",
    )
    before = deepcopy(asdict(config))
    with pytest.raises(RunConfigError) as caught:
        config.validate()

    expected = [
        ("task_run_id", "参数 task_run_id 必须是安全标识符"),
        ("tasks", "参数 tasks 必须是安全标识符"),
        ("tasks", "参数 tasks 必须是安全标识符"),
        ("rebuild_image", "恢复实验必须使用原镜像；如需重建，请创建新实验。"),
        ("ncu_repeat", "参数 ncu_repeat 必须是正整数"),
        ("revision", "模型 revision 必须是无空白的 branch、tag 或 commit SHA"),
        ("model_source", "模型来源必须是 huggingface 或 modelscope"),
        ("download_mode", "下载源模式必须是 auto、mirror-only、mirror-preferred 或 official"),
        ("cpuset_cpus", "CPU 集合格式无效；示例：0-3,8"),
        ("model", "模型 ID 不能为空"),
        ("cpus", "CPU 列表必须全部大于 0"),
        ("mems", "内存列表必须是逗号分隔的整数"),
        ("gpus", "GPU 模式不能为空"),
    ]
    assert [(issue.field, str(issue.reason)) for issue in caught.value.issues] == expected
    assert str(caught.value) == "；".join(f"[{field}] {reason}" for field, reason in expected)
    assert all(isinstance(issue.reason, Message) for issue in caught.value.issues)
    first_reason = caught.value.issues[0].reason
    assert isinstance(first_reason, Message)
    assert first_reason.template == "参数 {0} 必须是安全标识符"
    assert first_reason.values == ("task_run_id",)
    assert asdict(config) == before


def test_runtime_resource_aliases_and_later_issues_are_reported_together():
    config = replace(
        RunConfig.smoke("fixture/model"), runtime="process", compute_profile_tool="torch",
        cpus="1,1", mems="2,2", gpus="0,off", input_scales="nan", repeat=0,
        task_family="invalid", notify="invalid", extra_options={"dram_energy": "required"},
    )
    with pytest.raises(RunConfigError) as caught:
        config.validate()

    assert [(issue.field, str(issue.reason)) for issue in caught.value.issues] == [
        ("runtime", "Process 不支持所选的独立分析器或旧采集策略"),
        ("cpus", "启用启动 OOM 剪枝时 CPU 列表不能重复"),
        ("mems", "启用启动 OOM 剪枝时内存列表不能重复"),
        ("gpus", "启用启动 OOM 剪枝时 GPU 模式不能重复"),
        ("input_scales", "输入规模必须全部大于 0"),
        ("repeat", "Repeat 必须大于 0"),
        ("task_family", "任务族必须是 nlp/cv/audio/timeseries/diffusion/multimodal/structured"),
        ("notify", "无效的通知模式"),
        ("idle_seconds", "必需 DRAM 有效能耗需要正的 Idle 窗口"),
    ]


def test_normalization_keeps_resource_order_paths_and_original_request(tmp_path):
    (tmp_path / "workload.json").write_text("{}")
    (tmp_path / "inference.json").write_text("{}")
    (tmp_path / "model.json").write_text(
        '{"schema_version":1,"format":"transformers-pipeline",'
        '"task":"text-generation","pipeline_task":"text-generation"}'
    )
    config = replace(
        RunConfig.smoke(" fixture/model "), runtime="docker", task=" text-generation ",
        task_family=" NLP ", backend=" transformers_pipeline ", tasks=" first_task, second_task ",
        cpus=" 02,1,2 ", mems=" 04,8 ", cpuset_cpus="3,1-2", gpus=" CPU,1,off ",
        prune_startup_oom=False, input_scales=" 1.0,2e0 ",
        workload_spec=" workload.json ", model_spec=" model.json ",
        inference_config=" inference.json ", confirm_inference=" " + "a" * 64 + " ",
        output_dir=" results/example ", sniff_iface=" lo ",
        batch_size="2", warmup="0", repeat=3.0, repeat_in_window="0", raw_sample_limit="2",
        repeat_window_seconds="1.5", request_timeout_seconds="2", sample_hz="3.5",
        idle_seconds="0", idle_cooldown_seconds="0.25",
        extra_options={"ncu_repeat": "2", "massif_reference_cpu": 3.0},
    )
    before = deepcopy(asdict(config))
    normalized = config.validate(project_dir=tmp_path)
    expected = replace(
        config, model="fixture/model", task="text-generation", task_family="nlp", tasks="first_task,second_task",
        backend="transformers_pipeline", cpus="2,1,2", mems="4,8", cpuset_cpus="1-3",
        gpus="off,on,off", input_scales="1.0,2e0", workload_spec="workload.json",
        model_spec="model.json", inference_config="inference.json", confirm_inference="a" * 64,
        output_dir="results/example", sniff_iface="lo", batch_size=2, warmup=0,
        repeat=3, repeat_in_window=0, raw_sample_limit=2, repeat_window_seconds=1.5,
        request_timeout_seconds=2.0, sample_hz=3.5, idle_seconds=0.0, idle_cooldown_seconds=0.25,
        extra_options={"ncu_repeat": 2, "massif_reference_cpu": 3},
    )
    assert normalized == expected
    for name in ("batch_size", "warmup", "repeat", "repeat_in_window", "raw_sample_limit"):
        assert type(getattr(normalized, name)) is int
    for name in ("repeat_window_seconds", "request_timeout_seconds", "sample_hz",
                 "idle_seconds", "idle_cooldown_seconds"):
        assert type(getattr(normalized, name)) is float
    assert normalized.validate(project_dir=tmp_path) == normalized
    assert normalized.extra_options is not config.extra_options
    assert asdict(config) == before


@pytest.mark.parametrize(("runtime", "allow_auto", "mode", "expected_fields"), [
    ("auto", True, "auto", []),
    ("docker", True, "auto", []),
    ("process", True, "auto", []),
    ("auto", False, "auto", ["profiling_mode"]),
    ("process", False, "auto", ["runtime", "profiling_mode"]),
    ("process", True, "full", []),
    ("process", True, "invalid", ["runtime", "profiling_mode"]),
])
def test_automatic_preflight_only_defers_the_pending_profile(runtime, allow_auto, mode, expected_fields):
    config = replace(RunConfig.smoke("fixture/model"), runtime=runtime, profiling_mode=mode,
                     cpus="auto", mems="auto")
    if expected_fields:
        with pytest.raises(RunConfigError) as caught:
            config.validate(allow_auto_profiling=allow_auto)
        assert [issue.field for issue in caught.value.issues] == expected_fields
    else:
        assert config.validate(allow_auto_profiling=allow_auto) == config


@pytest.mark.parametrize("options", [
    {"compute_profile_tool": "torch"}, {"execution_profile_tool": "massif"},
])
def test_pending_profile_still_rejects_process_profiler_requirements(options):
    config = replace(RunConfig.smoke("fixture/model"), runtime="process", profiling_mode="auto", **options)
    with pytest.raises(RunConfigError) as caught:
        config.validate(allow_auto_profiling=True)
    assert [issue.field for issue in caught.value.issues] == ["runtime"]


@pytest.mark.parametrize("absolute", [False, True])
def test_path_errors_follow_earlier_issues_and_keep_confirmation_last(tmp_path, absolute):
    (tmp_path / "model.json").write_text("{}")
    workload = str(tmp_path / "workload.json") if absolute else "workload.json"
    model = str(tmp_path / "model.json") if absolute else "model.json"
    inference = str(tmp_path / "inference.json") if absolute else "inference.json"
    config = replace(RunConfig.smoke("fixture/model"), repeat=0, workload_spec=workload,
                     model_spec=model, inference_config=inference, confirm_inference="A" * 64)
    with pytest.raises(RunConfigError) as caught:
        config.validate(project_dir=tmp_path)
    assert [(issue.field, str(issue.reason)) for issue in caught.value.issues] == [
        ("repeat", "Repeat 必须大于 0"),
        ("workload_spec", f"Workload manifest 不存在：{workload}"),
        ("model_spec", f"模型接口声明无效：{model}；model spec must declare schema_version=1"),
        ("inference_config", f"推理请求 JSON 不存在：{inference}"),
        ("confirm_inference", "推理确认必须为审阅时显示的完整小写 SHA-256"),
    ]
    with pytest.raises(RunConfigError) as without_root:
        config.validate()
    assert [issue.field for issue in without_root.value.issues] == ["repeat", "confirm_inference"]


@pytest.mark.parametrize("error_type", [PermissionError, ValueError, RuntimeError])
def test_model_spec_errors_keep_their_classification(tmp_path, monkeypatch, error_type):
    failure = error_type("fixture model declaration failure")

    def fail_read(path):
        assert path == tmp_path / "model.json"
        raise failure

    monkeypatch.setattr("acprof.models.spec.read_model_spec", fail_read)
    config = replace(RunConfig.smoke("fixture/model"), model_spec="model.json")
    if error_type is RuntimeError:
        with pytest.raises(RuntimeError) as caught:
            config.validate(project_dir=tmp_path)
        assert caught.value is failure
    else:
        with pytest.raises(RunConfigError) as caught:
            config.validate(project_dir=tmp_path)
        assert [(issue.field, str(issue.reason)) for issue in caught.value.issues] == [
            ("model_spec", "模型接口声明无效：model.json；fixture model declaration failure"),
        ]
