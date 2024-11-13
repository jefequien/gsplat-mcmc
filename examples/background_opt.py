import torch
import torch.nn as nn
from torch import Tensor

from gsplat.utils import log_transform


class BackgroundOptModule(nn.Module):
    """Background optimization module."""

    def __init__(self, n: int, embed_dim: int = 4):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, embed_dim)
        # self.depths_encoder = get_encoder(num_freqs=3, input_dims=1)
        # self.mlp = create_mlp(
        #     in_dim=embed_dim + self.depths_encoder.out_dim + self.grid_encoder.out_dim,
        #     num_layers=5,
        #     layer_width=64,
        #     out_dim=1,
        # )
        self.bkgd_mlp = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(
        self,
        image_ids: Tensor,
        depths: Tensor,
    ):
        # depths_emb = self.depths_encoder.encode(log_transform(depths))
        depths_emb = log_transform(depths)
        images_emb = self.embeds(image_ids).repeat(*depths_emb.shape[:-1], 1)

        mlp_in = torch.cat([images_emb, depths_emb], dim=-1)
        mlp_out = self.bkgd_mlp(mlp_in.reshape(-1, mlp_in.shape[-1])).reshape(
            *mlp_in.shape[:-1], -1
        )
        bkgd = torch.sigmoid(mlp_out)
        return bkgd
