"""隔离依赖构建输入，避免身份测试修改工作区。"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_dependency_tree(root):
    shutil.copytree(ROOT / 'dockerfiles', root / 'dockerfiles')
    for name in ('LICENSE', 'NOTICE'):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(ROOT / 'licenses', root / 'licenses')
    (root / 'acprof').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'acprof/dependency_locks.py', root / 'acprof/dependency_locks.py')
    shutil.copyfile(ROOT / 'acprof/hf_endpoints.py', root / 'acprof/hf_endpoints.py')
    shutil.copyfile(ROOT / 'acprof/network_policy.py', root / 'acprof/network_policy.py')
