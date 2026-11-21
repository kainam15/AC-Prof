import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.artifacts import read_input_scale_plan
from acprof.container.handlers.nlp import NLPHandler
from acprof.host import input_plan
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import families
from acprof.tasks.workloads import parse_task, workload_manifest
from tests.workloads.nlp.test_nlp_handler import FakeTokenizer


@pytest.mark.parametrize('flags', [dict(limit_exceeded=True, truncated_by_limit=False),
                                   dict(limit_exceeded=False, truncated_by_limit=True)])
def test_manual_nlp_limit_error_keeps_requested_scale_reason_and_cleanup(tmp_path, flags):
    task = TaskInfo('org/model', 'fill-mask', 'nlp', 'transformers_pipeline', 'transformers', 'a' * 40, 'test')
    session = Mock()
    result = {'effective_input_scale': 32, 'reason': 'model token limit 32', **flags}
    with patch.object(families, 'start_probe_session', return_value=session), patch.object(
        families, 'post_probe_payload', return_value=result
    ), patch.object(families, 'stop_service_session') as stop:
        with pytest.raises(RuntimeError, match=r'manual NLP input scales exceed.*64 \(model token limit 32; effective_input_scale=32\)'):
            input_plan.plan_input_scales(task_info=task, image_info=ImageInfo(tag='fixture'), cpu_list=[1],
                                        mem_list=[4], gpu_list=['off'], batch_size=1,
                                        output_dir=str(tmp_path), input_scales='64')
    stop.assert_called_once_with(session, log_prefix='[probe]')
    assert not (tmp_path / 'input_scale_plan.json').exists()


@pytest.mark.parametrize(("total", "prefix", "expected"), [
    (4, "", [1]), (6, "", [1, 2, 3]), (6, "query: ", [2, 3]), (5, "fixed prefix ", []),
    (15, "", [2, 4, 6, 8, 10, 12]),
    (15, "one two three four five six ", [7, 8, 9, 10, 11, 12]),
    (15, "word " * 10, [11, 12]), (9, "query: ", [2, 3, 4, 5, 6]),
    (15, "word " * 12, []),
])
def test_auto_plan_materializes_the_usable_decoder_prompt_lengths(tmp_path, monkeypatch, total, prefix, expected):
    task = TaskInfo("org/model", "text-generation", "nlp", "transformers_pipeline", "transformers", "a" * 40, "test")
    handler = NLPHandler()
    ctx = {"task_type": task.pipeline_tag, "pipeline": SimpleNamespace(tokenizer=FakeTokenizer("[MASK]"))}
    source = tmp_path / "generation.json"
    selected = parse_task(task.pipeline_tag, {"prefix": prefix, "input_text": "word", "max_length": total},
                          encoder_decoder=False, source="fixture")
    source.write_text(json.dumps(workload_manifest(selected)))
    session = Mock()
    stop = Mock()

    probed = {}

    def probe(_session, payload, _description):
        prepared = handler.preprocess(ctx, payload)
        assert payload["text"] not in probed
        probed[payload["text"]] = (prepared["_effective_input_scale"], prepared["_truncated_by_limit"])
        return {"payload": payload, "effective_input_scale": prepared["_effective_input_scale"],
                "truncated_by_limit": prepared["_truncated_by_limit"]}

    monkeypatch.setattr(families, "start_probe_session", lambda *args: session)
    monkeypatch.setattr(families, "stop_service_session", stop)
    monkeypatch.setattr(families, "request_nlp_scale_meta", lambda _, payload: handler.get_scale_metadata(ctx, payload))
    monkeypatch.setattr(families, "post_probe_payload", probe)
    args = (task, ImageInfo(tag="fixture"), [1], [4], ["off"], 1, str(tmp_path))
    if expected:
        plan = input_plan.plan_input_scales(*args, workload_spec_path=str(source))
        assert plan.scales == expected
        saved = read_input_scale_plan(plan.plan_file, expected_sha256=plan.plan_sha256)
        assert [entry["input_scale"] for entry in saved["entries"]] == expected
        for entry in saved["entries"]:
            assert probed[entry["payload"]["text"]] == (entry["input_scale"], False)
        assert all(entry["payload"]["params"] == {"max_length": total, "do_sample": False}
                   for entry in saved["entries"])
    else:
        with pytest.raises(RuntimeError, match=f"no non-truncated NLP payload fits the input limit of {total - 3}"):
            input_plan.plan_input_scales(*args, workload_spec_path=str(source))
        assert not (tmp_path / "input_scale_plan.json").exists()
    stop.assert_called_once_with(session, log_prefix="[probe]")
