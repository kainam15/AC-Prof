from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from acprof.notifications import (
    WECOM_WEBHOOK_ENV,
    NotificationConfigError,
    NotificationDeliveryError,
    NotificationEvent,
    WeComWebhookNotifier,
    redact_notification_secrets,
    render_notification_text,
    validate_wecom_webhook_url,
)

WEBHOOK_KEY = "00000000-1111-2222-3333-444444444444"
WEBHOOK_URL = (
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=" + WEBHOOK_KEY
)


class TestWeComNotification:
    def _event(self, **overrides) -> NotificationEvent:
        values = {
            "status": "success",
            "model_id": "org/model",
            "output_dir": "/tmp/results/org--model",
            "elapsed_seconds": 65.0,
            "total_cases": 2,
            "completed_cases": 2,
            "result_rows": 12,
            "error_rows": 0,
            "final_csv": "/tmp/results/org--model/result_all.csv",
            "host": "test-host",
        }
        values.update(overrides)
        return NotificationEvent(**values)

    @pytest.mark.parametrize('value_case', range(6))
    def test_validate_accepts_only_wecom_group_robot_endpoint(self, value_case) -> None:
        assert (validate_wecom_webhook_url(WEBHOOK_URL)) == (WEBHOOK_URL)

        invalid_urls = (
            "",
            "http://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x",
            "https://example.com/cgi-bin/webhook/send?key=x",
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=x",
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/send",
            "https://qyapi.weixin.qq.com:invalid/cgi-bin/webhook/send?key=x",
        )
        value = tuple(invalid_urls)[value_case]
        with pytest.raises(NotificationConfigError):
            validate_wecom_webhook_url(value)

    def test_from_env_requires_configuration_without_echoing_secret(self) -> None:
        with pytest.raises(NotificationConfigError) as raised:
            WeComWebhookNotifier.from_env({})
        assert (WECOM_WEBHOOK_ENV) in (str(raised.value))

        notifier = WeComWebhookNotifier.from_env({WECOM_WEBHOOK_ENV: WEBHOOK_URL})
        assert (WEBHOOK_KEY) not in (repr(notifier))
        assert ("<redacted>") in (repr(notifier))

    def test_render_includes_partial_counts_and_redacts_webhook(self) -> None:
        text = render_notification_text(
            self._event(
                status="partial",
                error_rows=2,
                detail=f"request failed for {WEBHOOK_URL}",
            )
        )

        assert ("采集部分完成") in (text)
        assert ("资源组合：2/2") in (text)
        assert ("异常行：2") in (text)
        assert (WEBHOOK_KEY) not in (text)
        assert ("key=<redacted>") in (text)
        assert (WEBHOOK_KEY) not in (redact_notification_secrets(WEBHOOK_URL))

    def test_render_progress_reports_elapsed_case_and_percentage(self) -> None:
        text = render_notification_text(
            self._event(
                status="progress",
                elapsed_seconds=3661.0,
                completed_cases=1,
                total_cases=4,
                result_rows=7,
                error_rows=1,
                final_csv=None,
                detail="刚完成：CPU=2, MEM=4GB, GPU=off",
            )
        )

        assert ("AC-Prof 采集进度") in (text)
        assert ("耗时：1小时1分1秒") in (text)
        assert ("资源组合：1/4（25.0%）") in (text)
        assert ("当前 case 结果行：7") in (text)
        assert ("当前 case 异常行：1") in (text)
        assert ("CPU=2, MEM=4GB, GPU=off") in (text)

    def test_render_started_includes_command_and_redacts_webhook(self) -> None:
        text = render_notification_text(
            self._event(
                status="started",
                elapsed_seconds=0.4,
                run_command=(
                    "acprof run --model 'org/model with space' "
                    f"--callback {WEBHOOK_URL}"
                ),
                total_cases=None,
                completed_cases=None,
                result_rows=None,
                error_rows=None,
                final_csv=None,
                detail="命令已启动，正在执行环境预检",
            )
        )

        assert ("AC-Prof 实验开始") in (text)
        assert ("状态：已启动") in (text)
        assert ("指令：acprof run --model 'org/model with space'") in (text)
        assert ("正在执行环境预检") in (text)
        assert (WEBHOOK_KEY) not in (text)
        assert ("key=<redacted>") in (text)

    @pytest.mark.parametrize('status,label', (('success', '成功'), ('partial', '部分失败'), ('failed', '失败'), ('no_results', '无结果')))
    def test_render_profiler_outcomes_and_sample_counts(self, status, label) -> None:
        text = render_notification_text(
            NotificationEvent(
                status=f"profiler_{status}",
                model_id="org/model",
                output_dir="/tmp/results/org--model",
                elapsed_seconds=3661.0,
                profiler="GPU Torch",
                profile_elapsed_seconds=65.0,
                profile_samples=4,
                profile_error_samples=1,
                detail=f"failed request: {WEBHOOK_URL}",
            )
        )
        assert (f"状态：{label}") in (text)
        assert ("Profiler：GPU Torch") in (text)
        assert ("阶段耗时：1分5秒") in (text)
        assert ("耗时：1小时1分1秒") in (text)
        assert ("采样项：4") in (text)
        assert ("失败采样项：1") in (text)
        assert ("结果行") not in (text)
        assert ("资源组合") not in (text)
        assert (WEBHOOK_KEY) not in (text)
        assert ("key=<redacted>") in (text)

    def test_send_posts_text_payload_with_short_timeout(self) -> None:
        response = SimpleNamespace(status_code=200, json=lambda: {"errcode": 0})
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=1,
            timeout_seconds=3.0,
        )

        with patch(
            "acprof.notifications.requests.post",
            return_value=response,
        ) as post:
            notifier.send(self._event())

        post.assert_called_once()
        args, kwargs = post.call_args
        assert (args) == ((WEBHOOK_URL,))
        assert (kwargs["timeout"]) == (3.0)
        assert (kwargs["json"]["msgtype"]) == ("text")
        assert ("AC-Prof 采集完成") in (kwargs["json"]["text"]["content"])

    def test_send_retries_then_succeeds(self) -> None:
        response = SimpleNamespace(status_code=200, json=lambda: {"errcode": 0})
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=2,
            retry_delay_seconds=0.5,
        )

        with patch(
            "acprof.notifications.requests.post",
            side_effect=[requests.ConnectionError("offline"), response],
        ) as post, patch("acprof.notifications.time.sleep") as sleep:
            notifier.send(self._event())

        assert (post.call_count) == (2)
        sleep.assert_called_once_with(0.5)

    def test_request_failure_never_exposes_webhook(self) -> None:
        notifier = WeComWebhookNotifier(
            WEBHOOK_URL,
            attempts=1,
        )
        unsafe_error = requests.ConnectionError(f"failed to reach {WEBHOOK_URL}")

        with patch(
            "acprof.notifications.requests.post",
            side_effect=unsafe_error,
        ), pytest.raises(NotificationDeliveryError) as raised:
            notifier.send(self._event())

        assert (WEBHOOK_KEY) not in (str(raised.value))
        assert ("ConnectionError") in (str(raised.value))

    def test_api_error_is_reported_without_request_url(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"errcode": 93000, "errmsg": "invalid webhook"},
        )
        notifier = WeComWebhookNotifier(WEBHOOK_URL, attempts=1)

        with patch(
            "acprof.notifications.requests.post",
            return_value=response,
        ), pytest.raises(NotificationDeliveryError) as raised:
            notifier.send(self._event())

        assert ("errcode=93000") in (str(raised.value))
        assert (WEBHOOK_KEY) not in (str(raised.value))
