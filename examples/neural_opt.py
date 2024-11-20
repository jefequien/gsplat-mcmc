import os
import torch
from torch import Tensor
import torch.nn.functional as F

from examples.mlp import create_mlp, get_encoder
from gsplat.utils import log_transform


class NeuralOptModule(torch.nn.Module):
    """Neural optimization module."""

    def __init__(self, feature_dim: int, sh_degree: int = 3):
        super().__init__()
        self.means_encoder = get_encoder(num_freqs=3, input_dims=3)
        self.cov_mlp = create_mlp(
            in_dim=self.means_encoder.out_dim + feature_dim + 4,
            num_layers=5,
            layer_width=64,
            out_dim=7,
            initialize_last_layer_zeros=False,
        )
        self.shN_mlp = create_mlp(
            in_dim=self.means_encoder.out_dim + feature_dim + 4,
            num_layers=5,
            layer_width=64,
            out_dim=((sh_degree + 1) ** 2 - 1) * 3,
            initialize_last_layer_zeros=True,
        )

    def forward(self, means: Tensor, opacities: Tensor, sh0: Tensor, features: Tensor):
        means_emb = self.means_encoder.encode(log_transform(means))
        opacities_emb = opacities[:, None]
        sh0_emb = sh0[:, 0, :]
        mlp_in = torch.cat([means_emb, opacities_emb, sh0_emb, features], dim=-1)
        mlp_out = self.cov_mlp(mlp_in).float()
        shN_out = self.shN_mlp(mlp_in).float()

        quats = mlp_out[:, :4]
        scales = mlp_out[:, 4:7] - 5.0
        shN = shN_out.reshape(means.shape[0], -1, 3)
        return quats, scales, shN

    def compress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = self.half().state_dict()
        torch.save(state_dict, compress_path)

    def decompress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = torch.load(compress_path, weights_only=True)
        self.load_state_dict(state_dict)
