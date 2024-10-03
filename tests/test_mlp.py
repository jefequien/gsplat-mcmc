"""Tests for MLPs.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import pytest
import torch

from examples.mlp import _create_mlp_tcnn

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_create_mlp():
    in_dim = 16
    num_layers = 3
    layer_width = 64
    out_dim = 3
    mlp_tcnn = _create_mlp_tcnn(
        in_dim=in_dim,
        num_layers=num_layers,
        layer_width=layer_width,
        out_dim=out_dim,
        initialize_last_layer_zeros=True,
    ).to(device)

    x = torch.randn(10, 16).to(device)
    out = mlp_tcnn(x).float()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-2, atol=1e-2)

    num_nonzeros = torch.count_nonzero(mlp_tcnn.params)
    expected_nonzeros = (
        in_dim * layer_width + (num_layers - 2) * layer_width * layer_width
    )
    assert num_nonzeros == expected_nonzeros
