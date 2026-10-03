"""Release publication must be downstream of the reusable regression suite."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateTests(unittest.TestCase):
    def test_publish_requires_same_revision_ci_and_distribution(self):
        release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
        jobs = release['jobs']
        for publisher in ('publish', 'pypi'):
            with self.subTest(publisher=publisher):
                self.assertIn(publisher, jobs)
                needs = jobs[publisher]['needs']
                needs = [needs] if isinstance(needs, str) else needs
                self.assertTrue({'build', 'verify-dist', 'standalone'} <= set(needs))
                verification = [jobs[name] for name in needs if jobs[name].get('uses') == './.github/workflows/ci.yml']
                self.assertEqual(len(verification), 1, 'publish must wait for same-commit reusable CI')
                self.assertNotIn('always()', jobs[publisher].get('if', ''))

    def test_only_version_tag_push_can_publish(self):
        release = yaml.load((ROOT / '.github/workflows/release.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(release['on']['push'], {'tags': ['v*']})
        for publisher in ('publish', 'pypi'):
            self.assertIn(publisher, release['jobs'])
            condition = release['jobs'][publisher]['if']
            self.assertIn("github.event_name == 'push'", condition)
            self.assertIn("startsWith(github.ref, 'refs/tags/v')", condition)

    def test_pypi_uses_oidc_and_the_same_built_artifact_as_github(self):
        jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
        self.assertIn('pypi', jobs)
        pypi = jobs['pypi']
        self.assertEqual(pypi['environment']['name'], 'pypi')
        self.assertEqual(pypi['permissions'], {'id-token': 'write'})
        self.assertFalse(any('checkout@' in step.get('uses', '') for step in pypi['steps']))
        action = next(step for step in pypi['steps'] if step.get('uses', '').startswith('pypa/gh-action-pypi-publish@'))
        self.assertRegex(action['uses'], r'@[0-9a-f]{40}$')
        self.assertFalse(action.get('with', {}).get('skip-existing', False))
        self.assertFalse({'password', 'user'} & action.get('with', {}).keys())
        for consumer in ('verify-dist', 'standalone', 'publish', 'pypi'):
            downloads = [step['with']['name'] for step in jobs[consumer]['steps']
                         if step.get('uses', '').startswith('actions/download-artifact@')]
            self.assertIn('python-dist', downloads)
        self.assertNotIn('uv build', '\n'.join(step.get('run', '') for step in pypi['steps']))

    def test_version_mismatch_is_rejected_before_building(self):
        jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
        steps = jobs['build']['steps']
        version_steps = [step for step in steps if step.get('id') == 'version']
        self.assertEqual(len(version_steps), 1, 'version gate must run before uv build')
        gate = version_steps[0]
        self.assertLess(steps.index(gate), next(i for i, step in enumerate(steps) if 'uv build' in step.get('run', '')))
        code = gate['run'].split("python3 - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
        from acprof import __version__
        for tag, expected in (('', 0), ('v' + __version__, 0), ('v999.0.0', 1)):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'output'
                result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                        env={**os.environ, 'RELEASE_TAG': tag, 'GITHUB_OUTPUT': str(output)},
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected:
                    self.assertIn('tag and package version differ', result.stderr)
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(output.read_text().strip(), f'version={__version__}')

    def test_pypi_install_check_cannot_republish(self):
        jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
        self.assertIn('pypi-smoke', jobs)
        smoke = jobs['pypi-smoke']
        self.assertIn('pypi', smoke['needs'])
        self.assertTrue(smoke['continue-on-error'])
        self.assertNotIn('id-token', smoke.get('permissions', {}))
        self.assertFalse(any(step.get('uses', '').startswith('pypa/gh-action-pypi-publish@')
                             for step in smoke['steps']))

    def test_published_hash_check_rejects_changed_or_missing_artifacts(self):
        jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
        script = next(step['run'] for step in jobs['pypi-smoke']['steps'] if 'check_pypi_hashes' in step.get('run', ''))
        code = script.split("python3 - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
        for changed, missing in ((False, False), (True, False), (False, True)):
            with self.subTest(changed=changed, missing=missing), tempfile.TemporaryDirectory() as temporary:
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
                        with self.assertRaises(AssertionError):
                            exec(code, {})
                    else:
                        exec(code, {})

    @unittest.skipUnless(shutil.which('bash') and os.name != 'nt', 'Linux release job uses Bash')
    def test_pypi_install_retry_is_bounded_and_stops_after_success(self):
        bash = shutil.which('bash')
        assert bash is not None
        jobs = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())['jobs']
        script = next(step['run'] for step in jobs['pypi-smoke']['steps'] if 'check_pypi_hashes' in step.get('run', ''))
        for success_after, expected_attempts, expected_exit in ((99, 5, 1), (3, 3, 0)):
            with self.subTest(success_after=success_after), tempfile.TemporaryDirectory() as temporary:
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
                self.assertEqual(result.returncode, expected_exit, result.stderr)
                self.assertEqual(len(attempts.read_text().splitlines()), expected_attempts)
                self.assertEqual(len(sleeps.read_text().splitlines()), expected_attempts - 1)
                if expected_exit:
                    self.assertIn('::warning::', result.stdout)

    def test_regression_workflow_exposes_reusable_entrypoint(self):
        ci = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertIn('workflow_call', ci['on'])
        self.assertTrue({'lint', 'host', 'onnx-cpu', 'runtime'} <= ci['jobs'].keys())

    def test_hardware_evidence_requires_matching_success_and_retained_artifact(self):
        release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
        script = next(step['run'] for step in release['jobs']['publish']['steps']
                      if 'hardware-runs.json' in step.get('run', ''))
        code = script.split("python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        for commit, conclusion, expired, present, expected in (
            ('abc', 'success', False, True, 'workflow_passed'),
            ('other', 'success', False, True, 'not_verified'),
            ('abc', 'failure', False, True, 'not_verified'),
            ('abc', 'success', True, True, 'not_verified'),
            ('abc', 'success', False, False, 'not_verified'),
        ):
            with self.subTest(commit=commit, conclusion=conclusion, expired=expired, present=present), tempfile.TemporaryDirectory() as temporary:
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
                self.assertEqual(result.returncode, 0, result.stderr)
                evidence = json.loads((root / 'dist/verification.json').read_text())
                self.assertEqual(evidence['commit'], 'abc')
                self.assertEqual(evidence['hardware']['status'], expected)
