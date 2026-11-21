"""Check capacity arguments before allowing a Model Store prune."""
import json

import pytest

from acprof.cli.model_store import main
from tests.support.model_store_fixtures import ModelStoreFixture


@pytest.mark.parametrize(("target", "apply"), [
    pytest.param("", [], id="empty-preview"),
    pytest.param("", ["--apply"], id="empty-apply"),
    pytest.param(" \t", ["--apply"], id="blank"),
    pytest.param("10XB", ["--apply"], id="invalid-unit"),
    pytest.param("0.5B", ["--apply"], id="fractional-byte"),
    pytest.param("1e999", ["--apply"], id="exponent"),
])
def test_invalid_target_is_a_usage_error_before_accessing_store(tmp_path, capsys, target, apply):
    store = ModelStoreFixture(tmp_path)
    blob = store.blob("weights", 16)
    key = store.entry(1, [blob])

    with pytest.raises(SystemExit) as error:
        main(["prune", "--model-store", str(tmp_path), "--target-size", target, *apply])

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "--target-size" in output.err
    assert not (tmp_path / ".lock").exists()
    assert (tmp_path / "entries" / key).is_dir()
    assert blob.read_bytes() == b"x" * 16


@pytest.mark.parametrize(("target", "retain"), [
    pytest.param(None, False, id="omitted"),
    pytest.param("0", False, id="zero"),
    pytest.param("1048576", True, id="bytes"),
    pytest.param(" 1.5 KiB ", True, id="fractional-unit"),
    pytest.param("1" + "0" * 4301, True, id="large-byte-count"),
])
def test_valid_target_preserves_cleanup_scope(tmp_path, capsys, target, retain):
    store = ModelStoreFixture(tmp_path)
    blob = store.blob("weights", 16)
    key = store.entry(1, [blob])
    args = ["prune", "--model-store", str(tmp_path), "--apply"]
    if target is not None:
        args.extend(["--target-size", target])

    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == ([] if retain else [key])
    assert blob.exists() is retain
