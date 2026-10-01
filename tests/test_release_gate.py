"""Release publication must be downstream of the reusable regression suite."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateTests(unittest.TestCase):
    def test_publish_requires_same_revision_ci_and_distribution(self):
        release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
        jobs = release['jobs']
        needs = jobs['publish']['needs']
        needs = [needs] if isinstance(needs, str) else needs
        self.assertIn('build', needs)
        verification = [jobs[name] for name in needs if jobs[name].get('uses') == './.github/workflows/ci.yml']
        self.assertEqual(len(verification), 1, 'publish must wait for same-commit reusable CI')
        self.assertNotIn('always()', jobs['publish'].get('if', ''))

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
