import os
import torch
from torch import Tensor
import torch.nn.functional as F

from examples.mlp import create_mlp, get_encoder
from gsplat.utils import log_transform


class MlpOptModule(torch.nn.Module):
    """MLP optimization module."""

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.means_encoder = get_encoder(num_freqs=3, input_dims=3)
        self.shN_mlp = create_mlp(
            in_dim=self.means_encoder.out_dim + 3 + 8,
            num_layers=5,
            layer_width=64,
            out_dim=((sh_degree + 1) ** 2 - 1) * 3,
        )

    def predict_shN(
        self,
        means: Tensor,
        sh0: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
    ):
        N, _ = means.shape
        means_emb = self.means_encoder.encode(log_transform(means))
        sh0_emb = sh0[:, 0, :]
        quats_emb = F.normalize(quats, dim=-1)
        opacities_emb = opacities[:, None]
        mlp_in = torch.cat(
            [means_emb, sh0_emb, quats_emb, scales, opacities_emb], dim=-1
        )
        mlp_out = self.shN_mlp(mlp_in)
        shN = mlp_out.reshape(N, -1, 3)
        return shN

    def compress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = self.half().state_dict()
        torch.save(state_dict, compress_path)

    def decompress(self, compress_dir: str) -> None:
        compress_path = os.path.join(compress_dir, "mlp_module.pt")
        state_dict = torch.load(compress_path, weights_only=True)
        self.load_state_dict(state_dict)
