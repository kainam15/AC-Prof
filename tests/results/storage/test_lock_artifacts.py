"""Runtime audit inventories must match the exact install locks and target ABI."""
import json

import pytest
from tests.support.runtime_fixture import copy_dependency_tree

from scripts.compile_locks import check_catalog


@pytest.mark.parametrize("lock", ["platform-cpu", "nlp-cpu"])
@pytest.mark.parametrize("change", [
    "missing", "schema", "python", "target", "hash", "url", "version",
    "duplicate", "omitted", "size",
])
def test_check_rejects_stale_artifact_inventory(tmp_path, lock, change):
    copy_dependency_tree(tmp_path)
    path = tmp_path / "dockerfiles" / "locks" / f"{lock}.artifacts.json"
    data = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "schema":
            data["schema_version"] = 2
        elif change == "python":
            data["python_version"] = "3.10.21"
        elif change == "target":
            data["python_target"] = "aarch64-manylinux_2_28"
        elif change == "hash":
            data["artifacts"][0]["sha256"] = "0" * 64
        elif change == "url":
            data["artifacts"][0]["url"] = data["artifacts"][0]["url"].replace(
                "files.pythonhosted.org", "mirror.example")
        elif change == "version":
            data["artifacts"][0]["version"] = "0.0"
        elif change == "duplicate":
            data["artifacts"].append(data["artifacts"][0])
        elif change == "omitted":
            data["artifacts"].pop()
        elif change == "size":
            data["artifacts"][0]["size"] = -1
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="artifact"):
        check_catalog(tmp_path)


def test_artifact_order_is_not_part_of_lock_identity(tmp_path):
    copy_dependency_tree(tmp_path)
    path = tmp_path / "dockerfiles/locks/nlp-cpu.artifacts.json"
    data = json.loads(path.read_text())
    data["artifacts"].reverse()
    path.write_text(json.dumps(data))
    assert len(check_catalog(tmp_path)["environments"]) == 27
