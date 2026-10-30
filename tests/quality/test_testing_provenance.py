"""Test identities follow input bytes across checkouts and exclude generated state."""
import shutil
from unittest.mock import patch

import pytest

from acprof.testing.provenance import capture_provenance


def test_unreadable_source_directory_cannot_produce_complete_identity(tmp_path):
    (tmp_path / 'acprof').mkdir()
    with patch('acprof.testing.provenance.os.scandir', side_effect=PermissionError('source denied')):
        with pytest.raises(PermissionError, match='source denied'):
            capture_provenance([], root=tmp_path)


def test_collected_fixture_under_checkout_is_hashed_and_relocatable(tmp_path):
    root = tmp_path / 'first'
    suite = root / 'internal-testing' / 'fixture'
    suite.mkdir(parents=True)
    test = suite / 'test_sample.py'
    test.write_text('def test_ok(): pass\n')
    original = capture_provenance([test], root=root)
    moved = tmp_path / 'second'
    shutil.copytree(root, moved)
    moved_test = moved / test.relative_to(root)
    assert capture_provenance([moved_test], root=moved) == original
    moved_test.write_text('def test_ok(): assert True\n')
    changed = capture_provenance([moved_test], root=moved)
    assert {key for key in original if changed[key] != original[key]} == {'tests_sha256'}


def test_input_hashes_are_relocatable_and_separate_sources_tests_and_locks(tmp_path):
    root = tmp_path / 'first'
    for name, content in {'acprof/example.py': 'VALUE = 1\n', 'tests/test_example.py': 'def test_ok(): pass\n',
                          'uv.lock': 'version = 1\n', '.python-version': '3.12\n'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    original = capture_provenance([root / 'tests/test_example.py'], root=root)
    moved = tmp_path / 'second'
    shutil.copytree(root, moved)
    assert capture_provenance([moved / 'tests/test_example.py'], root=moved) == original
    for name, field in [('acprof/example.py', 'source_sha256'), ('tests/test_example.py', 'tests_sha256'),
                        ('uv.lock', 'locks_sha256'), ('.python-version', 'source_sha256')]:
        path = moved / name
        old = path.read_text()
        path.write_text(old + '# changed\n')
        changed = capture_provenance([], root=moved)
        assert {key for key in original if changed[key] != original[key]} == {field}
        path.write_text(old)
    for name in ('acprof/.env.local', 'acprof/__pycache__/module.pyc', 'internal-testing/host.json'):
        path = moved / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('generated state')
    assert capture_provenance([], root=moved) == original
