import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.config import (
    CONTAINER_HF_HOME,
    CONTAINER_MODEL_LOCAL_PATH,
)
from acprof.hf_endpoints import HF_DEFAULT_ENDPOINT, hf_endpoints
from acprof.host import env_utils


def test_save_config_preserves_unrelated_lines_and_secures_backup():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / '.env.local'
        original = '# keep this comment\nUNRELATED=value\nHF_TOKEN=previous\n'
        target.write_text(original)
        env = {'HF_TOKEN': 'previous', 'HUGGING_FACE_HUB_TOKEN': 'previous'}
        env_utils.save_project_env(root, {'HF_TOKEN': 'hf_testonly'}, environ=env)
        loaded = {}
        env_utils.load_project_env(root, environ=loaded)
        assert (loaded['HF_TOKEN']) == ('hf_testonly')
        assert (loaded['UNRELATED']) == ('value')
        assert ('# keep this comment') in (target.read_text())
        assert (target.stat().st_mode & 0o777) == (0o600)
        backup = root / '.env.local.bak'
        assert (backup.read_text()) == (original)
        assert (backup.stat().st_mode & 0o777) == (0o600)
        assert (env['HUGGING_FACE_HUB_TOKEN']) == ('hf_testonly')


def test_new_private_config_syncs_file_and_parent_directory():
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.host.env_utils.os.fsync", wraps=os.fsync,
    ) as fsync:
        env_utils.save_project_env(Path(directory), {"HF_TOKEN": "hf_testonly"}, environ={})

    assert fsync.call_count >= 2

@pytest.mark.parametrize('values', ({'HF_TOKEN': 'secret\nEVIL=1'}, {'ACPROF_SUDO_PASSWORD': 'secret'}, {'HF_ENDPOINT': 'https://user:secret@host.example'}, {'ACPROF_WECOM_WEBHOOK_URL': 'https://invalid.example?key=secret'}))
def test_save_config_rejects_secret_injection_without_changing_file_or_environment(values):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / '.env.local'
        path.write_text('HF_TOKEN=original\n')
        env = {}
        with pytest.raises(ValueError) as error:
            env_utils.save_project_env(root, values, environ=env)
        assert ('secret') not in (str(error.value))
        assert (path.read_text()) == ('HF_TOKEN=original\n')
        assert (env) == ({})

def test_save_config_roundtrips_literal_quotes_and_shell_text_without_evaluation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        value = 'literal "quoted" \\ ${UNCHANGED} $(never-execute) # fragment'
        env_utils.save_project_env(root, {'HF_TOKEN': value}, environ={})
        loaded = {}
        env_utils.load_project_env(root, environ=loaded)
        assert (loaded['HF_TOKEN']) == (value)

def test_save_config_refuses_symlinks_and_external_edits():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'external'
        source.write_text('HF_TOKEN=external\n')
        target = root / '.env.local'
        target.symlink_to(source)
        with pytest.raises(ValueError):
            env_utils.save_project_env(root, {'HF_TOKEN': 'test-only'}, environ={})
        assert (source.read_text()) == ('HF_TOKEN=external\n')
        target.unlink()
        target.write_text('HF_TOKEN=changed\n')
        with pytest.raises(ValueError):
            env_utils.save_project_env(root, {'HF_TOKEN': 'test-only'}, environ={}, expected_text='old')
        assert (target.read_text()) == ('HF_TOKEN=changed\n')

def test_local_settings_override_env_file_but_not_explicit_process_values():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / '.env').write_text('HF_TOKEN=old\nHF_ENDPOINT=https://old.example\n')
        (root / '.env.local').write_text('HF_TOKEN=new\nHF_ENDPOINT=https://local.example\n')
        env = {'HF_ENDPOINT': 'https://process.example'}
        env_utils.load_project_env(root, environ=env)
        assert (env['HF_TOKEN']) == ('new')
        assert (env['HF_ENDPOINT']) == ('https://process.example')

def test_local_env_can_be_loaded_without_changing_process_environment() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
        os.environ, {"EXPLICIT_SETTING": "process value"}, clear=True,
    ):
        root = Path(tmp_dir)
        (root / ".env").write_text("EXPLICIT_SETTING=file value\n", encoding="utf-8")
        (root / ".env.local").write_text(
            "LOCAL_SETTING='test-only-value'\n", encoding="utf-8",
        )
        probe_environ = os.environ.copy()

        env_utils.load_project_env(root, environ=probe_environ)

        assert (probe_environ["EXPLICIT_SETTING"]) == ("process value")
        assert (probe_environ["LOCAL_SETTING"]) == ("test-only-value")
        assert ("LOCAL_SETTING") not in (os.environ)

def test_bootstrap_defaults_to_mirror_and_preserves_proxy_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
        "acprof.host.env_utils.os.environ",
        {
            "HTTP_PROXY": "http://127.0.0.1:7890",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
            "ALL_PROXY": "socks5h://127.0.0.1:7891",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        },
        clear=True,
    ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
        env_utils.bootstrap_project_env(tmp_dir)

        assert (env_utils.os.environ["HF_ENDPOINT"]) == ("https://hf-mirror.com")
        assert (env_utils.os.environ["HF_HUB_ENDPOINT"]) == ("https://hf-mirror.com")
        assert (env_utils.os.environ["NO_PROXY"]) == ("localhost,127.0.0.1")
        assert (env_utils.os.environ["no_proxy"]) == ("localhost,127.0.0.1")

def test_bootstrap_preserves_explicit_endpoint_from_env_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
        "acprof.host.env_utils.os.environ",
        {},
        clear=True,
    ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
        Path(tmp_dir, ".env").write_text("HF_ENDPOINT=https://example.invalid\n", encoding="utf-8")

        env_utils.bootstrap_project_env(tmp_dir)

        assert (env_utils.os.environ["HF_ENDPOINT"]) == ("https://example.invalid")
        assert (env_utils.os.environ["HF_HUB_ENDPOINT"]) == ("https://example.invalid")
        assert ("NO_PROXY") not in (env_utils.os.environ)

def test_bootstrap_replaces_blank_endpoint_env_vars() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
        "acprof.host.env_utils.os.environ",
        {"HF_ENDPOINT": "", "HF_HUB_ENDPOINT": "   "},
        clear=True,
    ), patch("acprof.host.env_utils.resolve_hf_token", return_value=None):
        env_utils.bootstrap_project_env(tmp_dir)

        assert (env_utils.os.environ["HF_ENDPOINT"]) == (HF_DEFAULT_ENDPOINT)
        assert (env_utils.os.environ["HF_HUB_ENDPOINT"]) == (HF_DEFAULT_ENDPOINT)

@pytest.mark.parametrize('blank', ('', ' \t '))
def test_blank_primary_endpoint_uses_configured_fallback(blank) -> None:
    with patch.dict(
        os.environ,
        {"HF_ENDPOINT": blank, "HF_HUB_ENDPOINT": "https://example.invalid"},
        clear=True,
    ):
        assert (env_utils.configure_hf_network()) == ("https://example.invalid")
        assert (os.environ["HF_ENDPOINT"]) == ("https://example.invalid")
        assert (os.environ["HF_HUB_ENDPOINT"]) == ("https://example.invalid")
        assert ("NO_PROXY") not in (os.environ)

@pytest.mark.parametrize('url', ('https://user:test-secret@host.example', 'https://host.example?token=test-secret'))
def test_endpoint_rejects_credentials_without_echoing_them(url):
    with pytest.raises(ValueError) as raised:
        hf_endpoints({"HF_ENDPOINT": url})
    assert ("test-secret") not in (str(raised.value))

def test_explicit_endpoint_retains_priority_and_preserves_nonblank_alias() -> None:
    with patch.dict(
        os.environ,
        {"HF_ENDPOINT": "https://primary.invalid", "HF_HUB_ENDPOINT": "https://secondary.invalid"},
        clear=True,
    ):
        assert (env_utils.configure_hf_network()) == ("https://primary.invalid")
        assert (os.environ["HF_HUB_ENDPOINT"]) == ("https://secondary.invalid")

@pytest.mark.parametrize('blank', ('', ' \t '))
def test_token_fallback_fills_blank_primary_without_reading_login(blank) -> None:
    with patch.dict(
        os.environ, {"HF_TOKEN": blank, "HUGGING_FACE_HUB_TOKEN": "test-only-legacy"},
        clear=True,
    ), patch("huggingface_hub.utils.get_token", return_value=None) as get_token:
        assert (env_utils.resolve_hf_token()) == ("test-only-legacy")
        assert (os.environ["HF_TOKEN"]) == ("test-only-legacy")
        assert (os.environ["HUGGING_FACE_HUB_TOKEN"]) == ("test-only-legacy")
        get_token.assert_not_called()

@pytest.mark.parametrize('blank', ('', ' \t '))
def test_primary_token_fills_blank_legacy_alias(blank) -> None:
    with patch.dict(
        os.environ, {"HF_TOKEN": "test-only-primary", "HUGGING_FACE_HUB_TOKEN": blank},
        clear=True,
    ), patch("huggingface_hub.utils.get_token") as get_token:
        assert (env_utils.resolve_hf_token()) == ("test-only-primary")
        assert (os.environ["HUGGING_FACE_HUB_TOKEN"]) == ("test-only-primary")
        get_token.assert_not_called()

@pytest.mark.parametrize('values', ({}, {'HF_TOKEN': '', 'HUGGING_FACE_HUB_TOKEN': ' \t '}))
def test_local_login_fills_missing_or_blank_token_aliases(values) -> None:
    with patch.dict(
        os.environ, values, clear=True,
    ), patch("huggingface_hub.utils.get_token", return_value="test-only-local") as get_token:
        assert (env_utils.resolve_hf_token()) == ("test-only-local")
        assert (os.environ["HF_TOKEN"]) == ("test-only-local")
        assert (os.environ["HUGGING_FACE_HUB_TOKEN"]) == ("test-only-local")
        get_token.assert_called_once_with()

def test_primary_token_retains_priority_without_overwriting_legacy_alias() -> None:
    with patch.dict(
        os.environ,
        {"HF_TOKEN": "test-only-primary", "HUGGING_FACE_HUB_TOKEN": "test-only-legacy"},
        clear=True,
    ), patch("huggingface_hub.utils.get_token") as get_token:
        assert (env_utils.resolve_hf_token()) == ("test-only-primary")
        assert (os.environ["HF_TOKEN"]) == ("test-only-primary")
        assert (os.environ["HUGGING_FACE_HUB_TOKEN"]) == ("test-only-legacy")
        get_token.assert_not_called()

@pytest.mark.parametrize('outcome_case', range(2), ids=['None', "OSError('test-only unavailable login')"])
def test_missing_or_unavailable_login_keeps_anonymous_environment(outcome_case) -> None:
    outcome = tuple((None, OSError('test-only unavailable login')))[outcome_case]
    with patch.dict(os.environ, {}, clear=True), patch(
        "huggingface_hub.utils.get_token",
        **({"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}),
    ):
        assert (env_utils.resolve_hf_token()) is None
        assert ("HF_TOKEN") not in (os.environ)
        assert ("HUGGING_FACE_HUB_TOKEN") not in (os.environ)

def test_offline_docker_env_disables_hub_and_exposes_local_snapshot() -> None:
    args = env_utils.hf_offline_docker_env_args()
    env_values = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "-e"
    }

    assert ("HF_HUB_OFFLINE=1") in (env_values)
    assert ("TRANSFORMERS_OFFLINE=1") in (env_values)
    assert (f"HF_HOME={CONTAINER_HF_HOME}") in (env_values)
    assert (f"HF_HUB_CACHE={CONTAINER_HF_HOME}") in (env_values)
    assert (f"TRANSFORMERS_CACHE={CONTAINER_HF_HOME}") in (env_values)
    assert (f"MODEL_LOCAL_PATH={CONTAINER_MODEL_LOCAL_PATH}") in (env_values)
