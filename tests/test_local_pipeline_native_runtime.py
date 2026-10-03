"""Real upstream loader acceptance; never simulate copying or module imports.

Run against a candidate with ACPROF_TEST_NATIVE_TRANSFORMERS=<exact version>.
Otherwise this suite runs only for runtimes already declaring both capabilities.
The override exists only in test subprocesses, never in the production loader.
"""
import os

import pytest
import test_custom_multimodal_runtime as fixtures

from acprof.model_resolution import transformers_capabilities

pytestmark = pytest.mark.runtime


class TestNativeLocalPipelineRuntime(fixtures.TestLocalPipelineDependencyRuntime):
    @pytest.fixture(autouse=True)
    def _native_capabilities(self):
        import transformers

        candidate = os.environ.get("ACPROF_TEST_NATIVE_TRANSFORMERS")
        if candidate:
            assert (transformers.__version__) == (candidate), "install the exact candidate before acceptance"
        else:
            capabilities = transformers_capabilities(transformers.__version__)
            if not all(capabilities.values()):
                pytest.skip("runtime requires compat; select an exact native candidate for acceptance")

    @staticmethod
    def probe(root, cache, *, native: bool = True):
        return fixtures.TestLocalPipelineDependencyRuntime.probe(root, cache, native=native)
