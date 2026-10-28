from unittest.mock import Mock, patch

import pytest

from acprof.host import input_plan
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import families


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
