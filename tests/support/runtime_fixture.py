"""隔离依赖构建输入，避免身份测试修改工作区。"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_dependency_tree(root):
    shutil.copytree(ROOT / 'dockerfiles', root / 'dockerfiles')
    for name in ('LICENSE', 'NOTICE'):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(ROOT / 'licenses', root / 'licenses')
    (root / 'acprof').mkdir(exist_ok=True)
    (root / 'acprof/runtime').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'acprof/runtime/dependency_locks.py', root / 'acprof/runtime/dependency_locks.py')
    (root / 'acprof/hub/huggingface').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'acprof/hub/huggingface/endpoints.py', root / 'acprof/hub/huggingface/endpoints.py')
    shutil.copyfile(ROOT / 'acprof/network_policy.py', root / 'acprof/network_policy.py')
