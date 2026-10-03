"""依赖内容身份、平台边界和完整锁的行为回归。"""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from acprof.dependency_locks import read_python_lock, require_exact_packages
from acprof.host.dependency_images import runtime_fingerprint
from acprof.runtime_profiles import ENVIRONMENTS, PROFILES, environment_id

ROOT = Path(__file__).resolve().parents[1]


class TestEnvironmentIdentity:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))
        shutil.copytree(ROOT / 'dockerfiles', self.root / 'dockerfiles')
        (self.root / 'acprof').mkdir()
        shutil.copyfile(ROOT / 'acprof/dependency_locks.py', self.root / 'acprof/dependency_locks.py')
        shutil.copyfile(ROOT / 'acprof/network_policy.py', self.root / 'acprof/network_policy.py')
        self.env = ENVIRONMENTS['audio-cpu']

    def test_profiles_share_exact_environments_without_merging_near_matches(self):
        assert (len(PROFILES)) == (40)
        assert (len({environment_id(p.environment, ROOT) for p in PROFILES.values()})) == (27)
        for variant in ('cpu', 'cu124', 'cu128'):
            assert (PROFILES[f'custom-multimodal-{variant}'].environment) is (ENVIRONMENTS[f'custom-multimodal-{variant}'])
            assert (environment_id(PROFILES[f'custom-multimodal-{variant}'].environment, ROOT)) != (environment_id(PROFILES[f'multimodal-transformers560-{variant}'].environment, ROOT))
            for family in ('nlp', 'cv', 'audio', 'multimodal'):
                assert (PROFILES[f'{family}-transformers560-{variant}'].environment) is (ENVIRONMENTS[f'transformers560-{variant}'])
        for profile in ('onnxruntime-cpu', 'onnxruntime-cv-cpu', 'onnxruntime-nlp-cpu'):
            assert (PROFILES[profile].environment) is (ENVIRONMENTS['onnxruntime-cpu'])
        for variant in ('cpu', 'cu124'):
            assert (PROFILES['audio-' + variant].environment) is (PROFILES['multimodal-transformers4576-' + variant].environment)
        assert (environment_id(ENVIRONMENTS['audio-cu128'], ROOT)) != (environment_id(ENVIRONMENTS['multimodal-transformers4576'], ROOT))

    def test_comments_order_and_filename_do_not_change_identity(self):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.requirements_lock
        lines = [line for line in path.read_text().splitlines() if line and not line.startswith('#')]
        replacement = path.with_name('renamed.txt')
        replacement.write_text('# review only\n\n' + '\n'.join(reversed(lines)) + '\n')
        changed = dataclasses.replace(self.env, environment_key='renamed',
                                      requirements_lock=str(replacement.relative_to(self.root)))
        assert (original) == (environment_id(changed, self.root))
        assert (runtime_fingerprint(self.env, self.root)) == (runtime_fingerprint(changed, self.root))

    @pytest.mark.parametrize('changed_case', range(3), ids=["entry.replace('librosa-0.11.0-', 'librosa-0.11.1-')", "entry.replace('files.pythonhosted.org', 'example.org')", "entry.rsplit(':', 1)[0] + ':' + '1' * 64"])
    def test_artifact_source_hash_or_version_changes_environment_identity(self, changed_case):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.requirements_lock
        source = path.read_text()
        entry = next(line for line in source.splitlines() if line.startswith('librosa @'))
        changed = tuple((entry.replace('librosa-0.11.0-', 'librosa-0.11.1-'), entry.replace('files.pythonhosted.org', 'example.org'), entry.rsplit(':', 1)[0] + ':' + '1' * 64))[changed_case]
        path.write_text(source.replace(entry, changed))
        assert (original) != (environment_id(self.env, self.root))
        path.write_text(source)

    def test_system_package_change_changes_environment_identity(self):
        original = environment_id(self.env, self.root)
        path = self.root / self.env.platform.system_lock
        lock = json.loads(path.read_text())
        # 基础镜像继承包也属于平台内容，不能只比较显式 apt 安装列表。
        inherited = next(key for key in lock['packages']
                         if key not in {f"{p['name']}:{p['architecture']}" for p in lock['artifacts']})
        lock['packages'][inherited] += '.changed'
        path.write_text(json.dumps(lock))
        assert (original) != (environment_id(self.env, self.root))

    def test_build_recipe_changes_cache_but_not_environment_identity(self):
        original = environment_id(self.env, self.root)
        build = runtime_fingerprint(self.env, self.root)
        path = self.root / 'dockerfiles/runtime.Dockerfile'
        path.write_text(path.read_text() + '\n# build change\n')
        assert (original) == (environment_id(self.env, self.root))
        assert (build) != (runtime_fingerprint(self.env, self.root))

    def test_business_code_changes_do_not_invalidate_dependency_cache(self):
        original = runtime_fingerprint(self.env, self.root)
        for relative in ("acprof/container/server.py", "acprof/tui/app.py", "acprof/host/model_store.py"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# ordinary code update\n")
        assert (original) == (runtime_fingerprint(self.env, self.root))

    def test_parent_dependency_cannot_be_removed_or_replaced(self):
        path = self.root / self.env.requirements_lock
        source = path.read_text()
        entry = next(line for line in source.splitlines() if line.startswith('torch @'))
        for replacement in ('', entry.replace('torch-2.11.0', 'torch-2.11.1')):
            path.write_text(source.replace(entry, replacement))
            with pytest.raises(ValueError, match='父层'):
                environment_id(self.env, self.root)

    @pytest.mark.parametrize('content', ('torch==2.11.0\n', '-r another.txt\n', '', 'torch>=2\n'))
    def test_unhashed_or_unpinned_input_is_rejected(self, content):
        path = self.root / 'bad-lock.txt'
        path.write_text(content)
        with pytest.raises(ValueError):
            read_python_lock(path)

    @pytest.mark.parametrize('actual', ({}, {'torch': '2.11', 'hidden': '1'}, {'torch': '2.12'}))
    def test_missing_extra_or_changed_packages_are_rejected(self, actual):
        with pytest.raises(ValueError, match='package set'):
            require_exact_packages({'torch': '2.11'}, actual)
