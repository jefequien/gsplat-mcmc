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
        self.mlp = create_mlp(
            in_dim=self.means_encoder.out_dim + feature_dim + 1,
            num_layers=3,
            layer_width=64,
            out_dim=64,
            initialize_last_layer_zeros=False,
        )
        self.head = create_mlp(
            in_dim=64,
            num_layers=2,
            layer_width=64,
            out_dim=10,
            initialize_last_layer_zeros=False,
        )
        self.shN_head = create_mlp(
            in_dim=64,
            num_layers=2,
            layer_width=64,
            out_dim=((sh_degree + 1) ** 2 - 1) * 3,
            initialize_last_layer_zeros=True,
        )

    def forward(self, means: Tensor, opacities: Tensor, features: Tensor):
        means_emb = self.means_encoder.encode(log_transform(means))
        opacities_emb = opacities[:, None]
        mlp_in = torch.cat([means_emb, opacities_emb, features], dim=-1)
        mlp_out = self.mlp(mlp_in)
        head_out = self.head(mlp_out).float()
        shN_head_out = self.shN_head(mlp_out).float()

        quats = head_out[:, :4]
        scales = head_out[:, 4:7] - 5.0
        sh0 = head_out[:, 7:10].reshape(means.shape[0], -1, 3)
        shN = shN_head_out.reshape(means.shape[0], -1, 3)
        return quats, scales, sh0, shN

    def compress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = self.half().state_dict()
        torch.save(state_dict, compress_path)

    def decompress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = torch.load(compress_path, weights_only=True)
        self.load_state_dict(state_dict)
