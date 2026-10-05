"""Release publication must be downstream of the reusable regression suite."""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('publisher', ('publish', 'pypi'))
def test_publish_requires_same_revision_ci_and_distribution(publisher):
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    jobs = release['jobs']
    assert (publisher) in (jobs)
    needs = jobs[publisher]['needs']
    needs = [needs] if isinstance(needs, str) else needs
    assert ({'build', 'verify-dist', 'standalone'} <= set(needs))
    verification = [jobs[name] for name in needs if jobs[name].get('uses') == './.github/workflows/ci.yml']
    assert (len(verification)) == (1), 'publish must wait for same-commit reusable CI'
    assert ('always()') not in (jobs[publisher].get('if', ''))

def test_regression_workflow_exposes_reusable_entrypoint():
    ci = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    assert ('workflow_call') in (ci['on'])
    assert ({'lint', 'host', 'onnx-cpu', 'runtime'} <= ci['jobs'].keys())


def test_wheel_ci_resolves_declared_runtime_dependencies_without_host_lock():
    ci = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    steps = ci['jobs']['wheel']['steps']
    step = next(item for item in steps if item.get('name') == '验证公开依赖范围可由安装器解析')
    script = step['run']
    assert ('acprof-resolved') in (script)
    assert ('wheel-dist/*.whl') in (script)
    assert ('uv pip check') in (script)
    assert ('env -u PYTHONPATH') in (script)
    assert (' -I -c ') in (script)
    assert ("'pytest==8.4.2'") in (script)
    assert ("'pytest-asyncio==1.2.0'") in (script)
    assert ('tests/test_hf_auto_download.py') in (script)
    assert ('-p acprof.testing.plugin') in (script)
    assert ('--no-deps') not in (script)
    assert ('requirements/host.lock') not in (script)

@pytest.mark.parametrize('commit,conclusion,expired,present,expected', (('abc', 'success', False, True, 'workflow_passed'), ('other', 'success', False, True, 'not_verified'), ('abc', 'failure', False, True, 'not_verified'), ('abc', 'success', True, True, 'not_verified'), ('abc', 'success', False, False, 'not_verified')))
def test_hardware_evidence_requires_matching_success_and_retained_artifact(commit, conclusion, expired, present, expected):
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    script = next(step['run'] for step in release['jobs']['publish']['steps']
                  if 'hardware-runs.json' in step.get('run', ''))
    code = script.split("python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'dist').mkdir()
        (root / 'hardware-runs.json').write_text(json.dumps({'workflow_runs': [
            {'head_sha': commit, 'conclusion': conclusion, 'html_url': 'https://github.com/fixture/run'}
        ] if present else []}))
        (root / 'hardware-artifacts.json').write_text(json.dumps({'artifacts': [
            {'name': 'hardware', 'expired': expired, 'id': 42}
        ] if present else []}))
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True,
                                env={**os.environ, 'GITHUB_SHA': 'abc', 'GITHUB_REPOSITORY': 'fixture/project',
                                     'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_RUN_ID': '1'})
        assert (result.returncode) == (0), result.stderr
        evidence = json.loads((root / 'dist/verification.json').read_text())
        assert (evidence['commit']) == ('abc')
        assert (evidence['hardware']['status']) == (expected)

def test_only_version_tag_push_can_publish():
    release = yaml.load((ROOT / '.github/workflows/release.yml').read_text(), Loader=yaml.BaseLoader)
    assert (release['on']['push']) == ({'tags': ['v*']})
    for publisher in ('publish', 'pypi'):
        assert (publisher) in (release['jobs'])
        condition = release['jobs'][publisher]['if']
        assert ("github.event_name == 'push'") in (condition)
        assert ("startsWith(github.ref, 'refs/tags/v')") in (condition)

def test_pypi_uses_oidc_and_the_same_built_artifact_as_github():
    jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
    assert ('pypi') in (jobs)
    pypi = jobs['pypi']
    assert (pypi['environment']['name']) == ('pypi')
    assert (pypi['permissions']) == ({'id-token': 'write'})
    assert not (any('checkout@' in step.get('uses', '') for step in pypi['steps']))
    action = next(step for step in pypi['steps'] if step.get('uses', '').startswith('pypa/gh-action-pypi-publish@'))
    assert re.search(r'@[0-9a-f]{40}$', action['uses'])
    assert not (action.get('with', {}).get('skip-existing', False))
    assert not ({'password', 'user'} & action.get('with', {}).keys())
    for consumer in ('verify-dist', 'standalone', 'publish', 'pypi'):
        downloads = [step['with']['name'] for step in jobs[consumer]['steps']
                     if step.get('uses', '').startswith('actions/download-artifact@')]
        assert ('python-dist') in (downloads)
    assert ('uv build') not in ('\n'.join(step.get('run', '') for step in pypi['steps']))

@pytest.mark.parametrize('tag_case', range(3), ids=["('', 0)", "('v' + __version__, 0)", "('v999.0.0', 1)"])
def test_version_mismatch_is_rejected_before_building(tag_case):
    jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
    steps = jobs['build']['steps']
    version_steps = [step for step in steps if step.get('id') == 'version']
    assert (len(version_steps)) == (1), 'version gate must run before uv build'
    gate = version_steps[0]
    assert (steps.index(gate)) < (next(i for i, step in enumerate(steps) if 'uv build' in step.get('run', '')))
    code = gate['run'].split("python3 - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
    from acprof import __version__
    (tag, expected) = tuple((('', 0), ('v' + __version__, 0), ('v999.0.0', 1)))[tag_case]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'output'
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                env={**os.environ, 'RELEASE_TAG': tag, 'GITHUB_OUTPUT': str(output)},
                                text=True, capture_output=True)
        assert (result.returncode) == (expected), result.stderr
        if expected:
            assert ('tag and package version differ') in (result.stderr)
            assert not (output.exists())
        else:
            assert (output.read_text().strip()) == (f'version={__version__}')

def test_pypi_install_check_cannot_republish():
    jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
    assert ('pypi-smoke') in (jobs)
    smoke = jobs['pypi-smoke']
    assert ('pypi') in (smoke['needs'])
    assert (smoke['continue-on-error'])
    assert ('id-token') not in (smoke.get('permissions', {}))
    assert not (any(step.get('uses', '').startswith('pypa/gh-action-pypi-publish@')
                         for step in smoke['steps']))

@pytest.mark.parametrize('changed,missing', ((False, False), (True, False), (False, True)))
def test_published_hash_check_rejects_changed_or_missing_artifacts(changed, missing):
    jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
    script = next(step['run'] for step in jobs['pypi-smoke']['steps'] if 'check_pypi_hashes' in step.get('run', ''))
    code = script.split("python3 - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
    with tempfile.TemporaryDirectory() as temporary:
        dist = Path(temporary) / 'dist'
        dist.mkdir()
        files = ('acprof-0.2.0-py3-none-any.whl', 'acprof-0.2.0.tar.gz')
        remote = {'urls': []}
        for name in files:
            content = name.encode()
            remote['urls'].append({'filename': name, 'digests': {'sha256': hashlib.sha256(content).hexdigest()}})
            if not missing:
                (dist / name).write_bytes(b'changed' if changed else content)
        with patch.dict(os.environ, {'RELEASE_VERSION': '0.2.0', 'GITHUB_WORKSPACE': temporary}), \
                        patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(remote).encode())):
            if changed or missing:
                with pytest.raises(AssertionError):
                    exec(code, {})
            else:
                exec(code, {})

@pytest.mark.skipif(not (shutil.which('bash') and os.name != 'nt'), reason='Linux release job uses Bash')
@pytest.mark.parametrize('success_after,expected_attempts,expected_exit', ((99, 5, 1), (3, 3, 0)))
def test_pypi_install_retry_is_bounded_and_stops_after_success(success_after, expected_attempts, expected_exit):
    bash = shutil.which('bash')
    assert bash is not None
    jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
    script = next(step['run'] for step in jobs['pypi-smoke']['steps'] if 'check_pypi_hashes' in step.get('run', ''))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        # Replace only external tools. Execute the real workflow's retry/exit logic.
        programs = {
            'python3': 'exit 0\n',
            'uv': 'echo attempt >> "$ACPROF_TEST_ATTEMPTS"\n'
                  'count=$(/usr/bin/wc -l < "$ACPROF_TEST_ATTEMPTS")\n'
                  'test "$count" -ge "$ACPROF_TEST_SUCCESS_AFTER"\n',
            'sleep': 'echo sleep >> "$ACPROF_TEST_SLEEPS"\n',
            'acprof': 'echo "AC-Prof $RELEASE_VERSION"\n',
        }
        for name, body in programs.items():
            path = root / name
            path.write_text('#!/bin/sh\n' + body)
            path.chmod(0o755)
        attempts, sleeps = root / 'attempts', root / 'sleeps'
        result = subprocess.run([bash, '-e', '-o', 'pipefail', '-c', script],
                                cwd=root, capture_output=True, text=True, timeout=5,
                                env={**os.environ, 'PATH': f'{root}{os.pathsep}{os.defpath}',
                                     'UV_TOOL_BIN_DIR': str(root), 'RELEASE_VERSION': '0.2.0',
                                     'ACPROF_TEST_ATTEMPTS': str(attempts),
                                     'ACPROF_TEST_SLEEPS': str(sleeps),
                                     'ACPROF_TEST_SUCCESS_AFTER': str(success_after)})
        assert (result.returncode) == (expected_exit), result.stderr
        assert (len(attempts.read_text().splitlines())) == (expected_attempts)
        assert (len(sleeps.read_text().splitlines())) == (expected_attempts - 1)
        if expected_exit:
            assert ('::warning::') in (result.stdout)
