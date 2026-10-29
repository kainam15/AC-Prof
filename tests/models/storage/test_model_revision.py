import tempfile

import pytest

from acprof.container.handlers import model_revision_kwargs, resolve_model_source


def test_local_snapshot_is_preferred_and_does_not_pass_hub_revision() -> None:
    with tempfile.TemporaryDirectory() as local_path:
        source = resolve_model_source(
            "google-bert/bert-base-uncased",
            local_path,
        )

        assert (source) == (local_path)
        assert (model_revision_kwargs(source, "deadbeef")) == ({})

def test_missing_local_snapshot_is_rejected():
    with pytest.raises(FileNotFoundError, match="snapshot"):
        resolve_model_source("google-bert/bert-base-uncased", "/missing/model-snapshot")

def test_manual_hub_source_keeps_explicit_revision():
    source = resolve_model_source("google-bert/bert-base-uncased", "")
    assert (source) == ("google-bert/bert-base-uncased")
    assert (model_revision_kwargs(source, "deadbeef")) == ({"revision": "deadbeef"})
