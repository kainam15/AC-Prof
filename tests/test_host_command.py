"""Real child processes exercise the short-lived host command boundary."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from acprof.host.command import run_command


def test_streams_encoding_returncode_and_duration():
    result = run_command([sys.executable, "-c", "import os; os.write(1,b'out\\xff'); os.write(2,b'err\\xff'); raise SystemExit(7)"], capture_output=True)
    assert ((result.stdout, result.stderr, result.returncode)) == (("out�", "err�", 7))
    assert (result.duration_s) > (0)
    assert (result.metadata.returncode) == (7)

def test_cwd_and_environment_override_do_not_mutate_parent():
    with tempfile.TemporaryDirectory() as directory:
        result = run_command([sys.executable, "-c", "import os,json; print(json.dumps([os.getcwd(),os.environ['ACPROF_COMMAND_TEST'],bool(os.environ.get('PATH'))]))"],
                             capture_output=True, cwd=Path(directory), env_overrides={"ACPROF_COMMAND_TEST": "child"})
        assert (json.loads(result.stdout)) == ([directory, "child", True])
    assert ("ACPROF_COMMAND_TEST") not in (os.environ)
    assert (result.metadata.cwd) == (directory)

def test_check_preserves_subprocess_exception_and_output():
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_command([sys.executable, "-c", "print('failure'); raise SystemExit(9)"], capture_output=True, check=True)
    assert (caught.value.returncode) == (9)
    assert (caught.value.stdout) == ("failure\n")
    assert (caught.value.command_metadata.returncode) == (9)

def test_timeout_preserves_partial_output_and_metadata():
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        # Avoid IDE Python subprocess instrumentation delaying the first byte.
        # exec keeps sleep as the direct child, so timeout reaps it as usual.
        run_command(["/bin/sh", "-c", "printf ready; exec sleep 30"], capture_output=True, timeout=0.2)
    assert (b"ready") in (caught.value.output)
    assert (caught.value.command_metadata.error_type) == ("TimeoutExpired")

def test_missing_executable_preserves_oserror():
    with pytest.raises(FileNotFoundError) as caught:
        run_command(["/acprof-command-test/nonexistent"])
    assert (caught.value.command_metadata.error_type) == ("FileNotFoundError")

def test_evidence_and_debug_are_redacted_without_changing_execution(caplog):
    secret = "unique-test-credential"
    args = [sys.executable, "-c", "import sys; print(sys.argv[1])", secret, "--token", secret,
            "API_KEY=" + secret, "https://user:" + secret + "@example.test/path?token=" + secret]
    with caplog.at_level("DEBUG", logger="acprof.host.command"):
        result = run_command(args, capture_output=True, env_overrides={"API_KEY": secret, "OMP_NUM_THREADS": "2"}, redact_values=[secret])
    assert (result.stdout.strip()) == (secret)
    serialized = json.dumps(result.metadata.as_dict())
    assert (secret) not in (serialized + caplog.text)
    assert ("stdout") not in (serialized)
    assert ("OMP_NUM_THREADS") in (serialized)
    assert ("PATH") not in (serialized)

def test_default_redaction_handles_flags_environment_and_urls(caplog):
    with caplog.at_level("DEBUG", logger="acprof.host.command"):
        result = run_command([sys.executable, "-c", "pass", "--password", "flag-secret", "HF_TOKEN=env-secret", "https://a:url-secret@host.test/?key=query-secret"],
                             capture_output=True, env_overrides={"CUSTOM_PASSWORD": "override-secret"})
    evidence = json.dumps(result.metadata.as_dict()) + caplog.text
    for secret in ("flag-secret", "env-secret", "url-secret", "query-secret", "override-secret"):
        assert (secret) not in (evidence)

def test_no_implicit_publication_or_presentation():
    import io
    from contextlib import redirect_stderr, redirect_stdout
    with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        run_command([sys.executable, "-c", "print('captured')"], capture_output=True, cwd=directory)
        assert (list(Path(directory).iterdir())) == ([])
    assert ((out.getvalue(), err.getvalue())) == (("", ""))
