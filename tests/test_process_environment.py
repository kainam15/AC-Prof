"""A reused Python environment must retain the installed code, not just versions."""
import hashlib

import pytest

from acprof.dependency_locks import content_digest
from acprof.host.runtimes import environment


def test_reuse_rejects_dependency_edits_with_unchanged_package_versions(tmp_path, monkeypatch):
    files = {'bin/python': b'fixture interpreter', 'lib/site-packages/fixture.py': b'original code'}
    for name, data in files.items():
        path = tmp_path / 'venv' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    identity = {'host_abi': environment.host_abi(), 'service_sha256': 'fixture-code'}
    manifest = {'identity': identity, 'environment_id': content_digest(identity), 'packages': {'fixture': '1'},
                'interpreter_sha256': hashlib.sha256(files['bin/python']).hexdigest(),
                'installed_tree_sha256': content_digest({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})}
    monkeypatch.setattr(environment, 'installed_packages', lambda python: {'fixture': '1'})
    monkeypatch.setattr(environment, 'service_sources', lambda root: [])
    monkeypatch.setattr(environment, 'source_fingerprint', lambda *a, **kw: 'fixture-code')
    environment.verify_environment(tmp_path, manifest)
    (tmp_path / 'venv/lib/site-packages/fixture.py').write_text('modified code; same version')
    with pytest.raises(ValueError, match='installed.*changed'):
        environment.verify_environment(tmp_path, manifest)
