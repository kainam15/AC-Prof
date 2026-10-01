import tempfile
import unittest

from acprof.container.handlers import model_revision_kwargs, resolve_model_source


class ModelRevisionTests(unittest.TestCase):
    def test_local_snapshot_is_preferred_and_does_not_pass_hub_revision(self) -> None:
        with tempfile.TemporaryDirectory() as local_path:
            source = resolve_model_source(
                "google-bert/bert-base-uncased",
                local_path,
            )

            self.assertEqual(source, local_path)
            self.assertEqual(model_revision_kwargs(source, "deadbeef"), {})

    def test_missing_local_snapshot_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "snapshot"):
            resolve_model_source("google-bert/bert-base-uncased", "/missing/model-snapshot")

    def test_manual_hub_source_keeps_explicit_revision(self):
        source = resolve_model_source("google-bert/bert-base-uncased", "")
        self.assertEqual(source, "google-bert/bert-base-uncased")
        self.assertEqual(model_revision_kwargs(source, "deadbeef"), {"revision": "deadbeef"})

if __name__ == "__main__":
    unittest.main()
