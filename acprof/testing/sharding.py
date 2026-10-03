"""Stable, complete-suite identities shared by pytest and the evidence consumer."""

import hashlib


def suite_digest(nodeids):
    return hashlib.sha256("\n".join(sorted(nodeids)).encode()).hexdigest()


def select_shard(items, index, count):
    if not 0 <= index < count:
        raise ValueError("需要 0 <= --shard-index < --shard-count")
    ordered = sorted(items, key=lambda item: item.nodeid)
    return ordered, ordered[index::count]
