"""Real upstream loader acceptance; never simulate copying or module imports.

Run against a candidate with ACPROF_TEST_NATIVE_TRANSFORMERS=<exact version>.
Otherwise this suite runs only for runtimes already declaring both capabilities.
The override exists only in test subprocesses, never in the production loader.
"""
import os

from acprof.model_resolution import transformers_capabilities
import test_custom_multimodal_runtime as fixtures


class NativeLocalPipelineRuntimeTests(fixtures.LocalPipelineDependencyRuntimeTests):
    def setUp(self):
        import transformers

        candidate = os.environ.get("ACPROF_TEST_NATIVE_TRANSFORMERS")
        if candidate:
            self.assertEqual(transformers.__version__, candidate, "install the exact candidate before acceptance")
        else:
            capabilities = transformers_capabilities(transformers.__version__)
            if not all(capabilities.values()):
                self.skipTest("runtime requires compat; select an exact native candidate for acceptance")

    @staticmethod
    def probe(root, cache, *, native: bool = True):
        return fixtures.LocalPipelineDependencyRuntimeTests.probe(root, cache, native=native)
