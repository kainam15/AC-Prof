"""Small on-disk Model Store fixtures with real plans, blobs and locks."""
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from acprof.container.model_files import PLAN_FILENAME, plan_download, seal_plan


class ModelStoreFixture:
    def __init__(self, root: Path):
        self.root = root

    def blob(self, name: str, size: int) -> Path:
        path = self.root / 'hf/models--example--test/blobs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * size)
        return path

    def entry(self, index: int, blobs: list[Path]) -> str:
        key = f'{index:064x}'
        path = self.root / 'entries' / key
        snapshot = path / 'hf/models--example--test/snapshots' / ('a' * 40)
        snapshot.mkdir(parents=True)
        files = {}
        for index, blob in enumerate(blobs):
            name = f'file-{index}.bin'
            (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
            files[name] = {'size': blob.stat().st_size, 'sha256': hashlib.sha256(blob.read_bytes()).hexdigest()}
        plan = plan_download(model_id='example/test', revision='a' * 40, family='nlp',
            backend='transformers_pipeline', policy='full', files=files, read_json=lambda _: {})
        plan['verification'] = 'sha256'
        (path / PLAN_FILENAME).write_text(json.dumps(seal_plan(plan)))
        stamp = path / 'last-used'
        stamp.touch()
        os.utime(stamp, (int(key, 16), int(key, 16)))
        return key


@contextmanager
def locked_store(root):
    process = subprocess.Popen([sys.executable, '-I', '-c',
        "import fcntl,sys; stream=open(sys.argv[1], 'a'); fcntl.flock(stream, fcntl.LOCK_EX); "
        "print('locked', flush=True); sys.stdin.read()", str(root / '.lock')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if process.stdout.readline().strip() != 'locked':
            raise AssertionError('lock holder did not start')
        yield process
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
