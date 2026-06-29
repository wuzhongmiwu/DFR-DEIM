import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.extre_module.custom_nn.stem.ODALStem import ODALStem


def test_odal_stem_preserves_expected_downsample_shape():
    x = torch.randn(2, 3, 64, 64)
    module = ODALStem(3, 32)

    out = module(x)

    assert out.shape == (2, 32, 16, 16)


def test_odal_stem_exposes_normalized_orientation_weights():
    x = torch.randn(2, 3, 64, 64)
    module = ODALStem(3, 32)

    _ = module(x)

    assert hasattr(module, "last_orientation_weights")
    weights = module.last_orientation_weights
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)


def test_odal_stem_handles_odd_spatial_sizes_with_ceil_downsampling():
    x = torch.randn(1, 3, 65, 67)
    module = ODALStem(3, 32)

    out = module(x)

    assert out.shape == (1, 32, 17, 17)
