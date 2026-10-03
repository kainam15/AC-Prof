"""Loading evidence is collected without scraping logs or changing loader returns."""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.loading_quality import capture_loading_quality


def test_loading_info_preserves_return_contract_and_restores_on_failure():
    class Model:
        @classmethod
        def from_pretrained(cls, *, output_loading_info=False):
            instance = cls()
            info = {"missing_keys": ["head.weight"], "unexpected_keys": ["unused.weight"]}
            return (instance, info) if output_loading_info else instance

    descriptor = Model.__dict__["from_pretrained"]
    with patch.dict(sys.modules, {"transformers": SimpleNamespace(PreTrainedModel=Model)}):
        with pytest.raises(ValueError, match="downstream"):
            with capture_loading_quality("transformers_pipeline") as checks:
                assert isinstance(Model.from_pretrained(), Model)
                assert isinstance(Model.from_pretrained(output_loading_info=True), tuple)
                raise ValueError("downstream")
    assert (Model.__dict__["from_pretrained"]) is (descriptor)
    assert ({item["code"] for item in checks}) == ({"weights_reinitialized", "unused_checkpoint_weights"})
    assert (all(item["threshold"] == 0 for item in checks))
