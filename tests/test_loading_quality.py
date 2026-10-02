"""Loading evidence is collected without scraping logs or changing loader returns."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acprof.container.loading_quality import capture_loading_quality


class LoadingQualityTests(unittest.TestCase):
    def test_loading_info_preserves_return_contract_and_restores_on_failure(self):
        class Model:
            @classmethod
            def from_pretrained(cls, *, output_loading_info=False):
                instance = cls()
                info = {"missing_keys": ["head.weight"], "unexpected_keys": ["unused.weight"]}
                return (instance, info) if output_loading_info else instance

        descriptor = Model.__dict__["from_pretrained"]
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(PreTrainedModel=Model)}):
            with self.assertRaisesRegex(ValueError, "downstream"):
                with capture_loading_quality("transformers_pipeline") as checks:
                    self.assertIsInstance(Model.from_pretrained(), Model)
                    self.assertIsInstance(Model.from_pretrained(output_loading_info=True), tuple)
                    raise ValueError("downstream")
        self.assertIs(Model.__dict__["from_pretrained"], descriptor)
        self.assertEqual({item["code"] for item in checks}, {"weights_reinitialized", "unused_checkpoint_weights"})
        self.assertTrue(all(item["threshold"] == 0 for item in checks))


if __name__ == "__main__":
    unittest.main()
