"""The validated state API rejects incompatible updates during static checking."""
import subprocess
import sys


def test_loaded_state_fields_are_not_dynamic(tmp_path):
    source = tmp_path / "state_contract.py"
    source.write_text(
        "from acprof.host.run_state import load_run_state\n"
        "state = load_run_state('.')\n"
        "wrong_identity: int = state['run_id']\n"
        "wrong_pid: str = state['attempts'][0]['pid']\n"
        "state['cases']['case.csv']['processing_attempts'] = 'invalid'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(source)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("error:") == 3, result.stdout
    assert "state_contract.py:3:" in result.stdout
    assert "state_contract.py:4:" in result.stdout
    assert "state_contract.py:5:" in result.stdout
