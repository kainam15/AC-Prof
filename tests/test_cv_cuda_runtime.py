"""Offline CUDA checks; run explicitly in a GPU-enabled CV environment."""
import importlib.util

import pytest
from cv_runtime_fixtures import CVRuntimeFixture

_RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
pytestmark = [pytest.mark.runtime, pytest.mark.hardware]


@pytest.mark.skipif(not (_RUNTIME_AVAILABLE), reason="requires the CV image runtime")
class TestCVCudaRuntime(CVRuntimeFixture):
    def test_sam_mask_pipeline_defaults_to_fp32_on_cuda(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        self._sam_mask_pipeline("cuda")
