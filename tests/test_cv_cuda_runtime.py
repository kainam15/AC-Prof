"""Offline CUDA checks; run explicitly in a GPU-enabled CV environment."""

import importlib.util
import unittest

from cv_runtime_fixtures import CVRuntimeFixture

_RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(_RUNTIME_AVAILABLE, "requires the CV image runtime")
class CVCudaRuntimeTests(CVRuntimeFixture):
    def test_sam_mask_pipeline_defaults_to_fp32_on_cuda(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self._sam_mask_pipeline("cuda")


if __name__ == "__main__":
    unittest.main()
