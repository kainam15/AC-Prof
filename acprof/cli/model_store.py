"""Inspect or explicitly prune the host Model Store; default is read-only."""
import argparse
import json
from pathlib import Path

from acprof.host.env_utils import load_project_env
from acprof.host.model_store import disk_report, prune_store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "prune"), nargs="?", default="status")
    parser.add_argument("--apply", action="store_true", help="Actually remove unused entries and unreferenced weights")
    parser.add_argument("--keep", action="append", default=[], help="Entry ID to retain; can be repeated")
    parser.add_argument("--target-size", help="Prune least recently used entries toward this capacity, e.g. 100GB")
    parser.add_argument("--model-store", type=Path, help="Store to inspect or prune; defaults to ACPROF_MODEL_STORE")
    args = parser.parse_args(argv)
    load_project_env(Path.cwd())
    root = args.model_store.expanduser().resolve() if args.model_store else None
    result = disk_report(root) if args.action == "status" else prune_store(root=root, apply=args.apply, keep=set(args.keep), target_bytes=args.target_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
