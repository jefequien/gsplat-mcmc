import torch
from torch import Tensor

from examples.mlp import create_mlp, get_encoder
from gsplat.utils import log_transform


class MlpOptModule(torch.nn.Module):
    """MLP optimization module."""

    def __init__(
        self,
        sh_degree: int = 3,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.K = (sh_degree + 1) ** 2 - 1

        self.means_encoder = get_encoder(num_freqs=3, input_dims=3)
        self.shN_mlp = create_mlp(
            in_dim=self.means_encoder.out_dim + 3,
            num_layers=5,
            layer_width=64,
            out_dim=self.K * 3,
        )

    def predict_shN(self, means: Tensor, colors: Tensor):
        means_emb = self.means_encoder.encode(log_transform(means))
        colors_emb = colors[:, 0, :]
        mlp_in = torch.cat([means_emb, colors_emb], dim=-1)
        shN = self.shN_mlp(mlp_in).reshape(-1, self.K, 3)
        return shN
