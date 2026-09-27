"""The host can inspect upstream registries without executing model code."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.export_transformers_support import export_support


class TransformersCatalogTests(unittest.TestCase):
    SOURCE = b'MODEL_MAPPING_NAMES = OrderedDict([("a", "A")])\nMODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([("a", "B")])'

    def test_new_export_keeps_dynamic_capabilities_unknown_until_reviewed(self):
        catalog = export_support(self.SOURCE, "99.0.0")
        self.assertEqual(catalog.get("capabilities"), {
            "local_dynamic_transitive_imports": None, "local_dynamic_symlink_safe": None})

    def test_reexport_preserves_review_only_for_identical_version_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "modeling_auto.py", root / "catalog.json"
            source.write_bytes(self.SOURCE)
            reviewed = export_support(self.SOURCE, "99.0.0")
            reviewed["capabilities"] = {"local_dynamic_transitive_imports": True, "local_dynamic_symlink_safe": True}
            script = Path(__file__).resolve().parents[1] / "scripts" / "export_transformers_support.py"
            for version, contents, keep in (("99.0.0", self.SOURCE, True), ("99.0.1", self.SOURCE, False),
                                             ("99.0.0", self.SOURCE + b"\n# changed", False)):
                with self.subTest(version=version, keep=keep):
                    output.write_text(json.dumps(reviewed))
                    source.write_bytes(contents)
                    result = subprocess.run([sys.executable, str(script), "--source", str(source),
                                             "--version", version, "--output", str(output)],
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = reviewed["capabilities"] if keep else {
                        "local_dynamic_transitive_imports": None, "local_dynamic_symlink_safe": None}
                    self.assertEqual(json.loads(output.read_text()).get("capabilities"), expected)

    def test_static_composition_preserves_native_heads_without_executing_source(self):
        source = b'''
raise RuntimeError("upstream module must not execute")
MODEL_MAPPING_NAMES = OrderedDict([("encoder", "EncoderModel")])
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([("decoder", "DecoderForCausalLM")])
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict([
    *list(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()),
    ("vision", ("VisionModel", "VisionVariant")),
])
'''
        catalog = export_support(source, "test")
        mapping = catalog["mappings"]["MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"]
        self.assertEqual(mapping["decoder"], "DecoderForCausalLM")
        self.assertEqual(mapping["vision"], ("VisionModel", "VisionVariant"))
        self.assertEqual(len(catalog["source_sha256"]), 64)

    def test_changed_registry_shape_is_rejected_instead_of_publishing_partial_data(self):
        source = '''
MODEL_MAPPING_NAMES = OrderedDict([("encoder", "EncoderModel")])
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict([("decoder", "DecoderForCausalLM")])
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = REPLACEMENT
'''
        for expression in ("OrderedDict(build_dynamic_mapping())", "build_dynamic_mapping()", '{"vision": "Model"}'):
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "unsupported Auto registry"):
                export_support(source.replace("REPLACEMENT", expression).encode(), "test")

    def test_calls_inside_entries_are_rejected_without_execution(self):
        with self.assertRaises(ValueError):
            export_support(b'MODEL_MAPPING_NAMES = OrderedDict([("a", load_remote_code())])', "test")


if __name__ == "__main__":
    unittest.main()
