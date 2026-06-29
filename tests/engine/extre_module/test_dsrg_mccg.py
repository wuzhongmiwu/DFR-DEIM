import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_class(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    return getattr(module, class_name, None)


def test_mccg_preserves_output_shape():
    mccg_cls = _load_class("engine.extre_module.custom_nn.mlp.MCCG", "MCCG")
    assert mccg_cls is not None, "MCCG should be implemented under custom_nn/mlp"

    module = mccg_cls(in_features=32, hidden_features=64, out_features=32)
    x = torch.randn(2, 32, 24, 24)

    out = module(x)

    assert out.shape == x.shape


def test_dsrg_preserves_output_shape():
    dsrg_cls = _load_class("engine.extre_module.custom_nn.mlp.DSRG", "DSRG")
    assert dsrg_cls is not None, "DSRG should be implemented under custom_nn/mlp"

    module = dsrg_cls(in_features=32, hidden_features=64, out_features=32)
    x = torch.randn(2, 32, 24, 24)

    out = module(x)

    assert out.shape == x.shape


def test_mccg_supports_channel_change():
    mccg_cls = _load_class("engine.extre_module.custom_nn.mlp.MCCG", "MCCG")
    assert mccg_cls is not None, "MCCG should be implemented under custom_nn/mlp"

    x = torch.randn(1, 16, 20, 20)
    mccg = mccg_cls(in_features=16, hidden_features=48, out_features=24)

    assert mccg(x).shape == (1, 24, 20, 20)


def test_dsrg_supports_channel_change():
    dsrg_cls = _load_class("engine.extre_module.custom_nn.mlp.DSRG", "DSRG")
    assert dsrg_cls is not None, "DSRG should be implemented under custom_nn/mlp"

    x = torch.randn(1, 16, 20, 20)
    dsrg = dsrg_cls(in_features=16, hidden_features=48, out_features=24)

    assert dsrg(x).shape == (1, 24, 20, 20)
