import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from gsplat.utils import log_transform
import random
import tinycudann as tcnn

from mlp import create_mlp, get_encoder, HexPlaneField

kplanes_config = {
    'grid_dimensions': 2,
    'input_coordinate_dim': 4,
    'output_coordinate_dim': 16,
    'resolution': [64, 64, 64, 150]
}
multires = [1,2,4]
bounds = 1.0

class DeformationOptModule(nn.Module):
    """Deformation optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # self.means_encoder = get_encoder(num_freqs=9, input_dims=3)
        # self.means_encoder = get_encoder(num_freqs=13, input_dims=3)
        # self.means_encoder = HashGridEncoder(input_dims=3)
        # self.times_encoder = get_encoder(num_freqs=9, input_dims=1)
        self.grid = HexPlaneField(bounds, kplanes_config, multires)

        self.mlp = create_mlp(
            # in_dim=self.means_encoder.out_dim + self.times_encoder.out_dim,
            # in_dim=self.times_encoder.out_dim + 8,
            in_dim=self.grid.feat_dim,
            num_layers=5,
            layer_width=64,
            # num_layers=7,
            # layer_width=128,
            out_dim=7,
            initialize_last_layer_zeros=True,
        )

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        sh0: Tensor,
        features: Tensor,
        render_times: Tensor,
        image_times: Tensor,
    ):
        quats = F.normalize(quats, dim=-1)
        # features_avg = average_with_k_neighbors(features, k=0)
        means_log = log_transform(means)
        # means_norm = ((means_log / 5.0) + 0.5).clip(0.0, 1.0)
        means_aabb = (means_log / 5.0).clip(-1.0, 1.0)

        # means_enc = self.means_encoder.encode(means_log)
        # times_enc = self.times_encoder.encode(image_times).repeat(means.shape[0], 1)
        grid_features = self.grid(means_aabb, image_times.repeat(means.shape[0], 1))
        mlp_in = torch.cat([
            # means_enc, 
            # times_enc,
            # features,
            grid_features,
        ], dim=-1)
        mlp_out = self.mlp(mlp_in).float()
        means = means + mlp_out[:, :3]
        quats = quats + mlp_out[:, 3:]

        # self.last_mlp_out = mlp_out
        return means, quats
    
    # def alap_loss(self):
    #     means_delta = self.last_mlp_out[:,:3]
    #     quats_delta = self.last_mlp_out[:,3:]
    #     return means_delta.abs().mean() + 0.1 * quats_delta.abs().mean()


def average_with_k_neighbors(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Differentiable 1D smoothing over the first dimension (window size = 2k+1).
    Applies the same kernel to each feature channel independently (depthwise conv).
    
    Args:
        x: Tensor of shape (n, d)
        k: Number of neighbors on each side

    Returns:
        Tensor of shape (n, d)
    """
    if k == 0:
        return x

    n, d = x.shape

    # Prepare input for conv1d: (B=1, C=d, L=n)
    x_3d = x.T.unsqueeze(0)  # (1, d, n)

    # Pad along the "sequence" dimension
    x_padded = F.pad(x_3d, (k, k), mode="replicate")  # (1, d, n + 2k)

    # Create (d, 1, 2k+1) depthwise kernel: one kernel per channel
    kernel = torch.ones(d, 1, 2 * k + 1, device=x.device, dtype=x.dtype)
    kernel /= (2 * k + 1)

    # Perform depthwise 1D convolution
    out = F.conv1d(x_padded, kernel, groups=d)  # (1, d, n)

    return out.squeeze(0).T  # (n, d)


class HashGridEncoder(nn.Module):
    """
    Multiresolution hash‑grid wrapper that mimics the API of
    get_encoder(...): it exposes .out_dim and .encode().
    """
    def __init__(
        self,
        input_dims: int = 3,
        n_levels: int = 32,
        n_feats_per_level: int = 2,
        log2_hashmap_size: int = 22,
        base_resolution: int = 16,
        per_level_scale: float = 1.3819,
    ):
        super().__init__()

        enc_cfg = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_feats_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }

        self._encoder = tcnn.Encoding(input_dims, enc_cfg)
        self.out_dim = n_levels * n_feats_per_level  # keep same contract

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, input_dims)
        return self._encoder(x)                                  # (B, out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
