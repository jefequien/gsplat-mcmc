import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from examples.mlp import create_mlp, get_encoder
from gsplat.utils import log_transform, get_projection_matrix


class BackgroundOptModule(nn.Module):
    """Background optimization module."""

    def __init__(self, n: int, embed_dim: int = 4):
        super().__init__()
        # self.embeds = torch.nn.Embedding(n, embed_dim)
        self.depths_encoder = get_encoder(num_freqs=3, input_dims=1)
        # self.rays_encoder = get_encoder(num_freqs=3, input_dims=3)
        self.bkgd_mlp = create_mlp(
            in_dim=3,
            num_layers=5,
            layer_width=64,
            out_dim=3,
        )
        self.mask_mlp = create_mlp(
            in_dim=self.depths_encoder.out_dim + 4,
            num_layers=5,
            layer_width=64,
            out_dim=1,
        )
        # self.bounded_l1_loss = bounded_l1_loss(-10.0, 0.5)

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(
        self,
        depths: Tensor,
        camtoworlds: Tensor,
        Ks: Tensor,
        near_plane: float,
        far_plane: float,
    ):
        # depths_emb = self.depths_encoder.encode(log_transform(depths))
        # images_emb = self.embeds(image_ids).repeat(*depths_emb.shape[:-1], 1)
        height, width = depths.shape[1:3]
        rays = depth_to_rays(
            depths,
            camtoworlds,
            Ks,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        rays = F.normalize(rays, dim=-1)
        # rays_emb = self.rays_encoder.encode(rays)
        mlp_in = torch.cat([rays], dim=-1)
        mlp_out = self.bkgd_mlp(mlp_in.reshape(-1, mlp_in.shape[-1])).reshape(
            1, height, width, -1
        )
        bkgd = torch.sigmoid(mlp_out)
        return bkgd

    def predict_mask(self, colors: Tensor, alphas: Tensor, depths: Tensor):
        height, width = depths.shape[1:3]

        depths_emb = self.depths_encoder.encode(log_transform(depths))
        mlp_in = torch.cat([colors, alphas, depths_emb], dim=-1)
        mlp_out = self.mask_mlp(mlp_in.reshape(-1, mlp_in.shape[-1])).reshape(
            1, height, width, -1
        )
        mask = torch.sigmoid(mlp_out)
        return mask

    # def mask_loss(self, blur_mask: Tensor):
    #     """Loss function for regularizing the blur mask by controlling its mean.
    #     Uses bounded l1 loss which diverges to +infinity at 0 and 1 to prevents the mask
    #     from collapsing all 0s or 1s.
    #     """
    #     x = blur_mask.mean()
    #     print(x)
    #     return self.bounded_l1_loss(x)


def bounded_l1_loss(lambda_a: float, lambda_b: float, eps: float = 1e-2):
    """L1 loss function with discontinuities at 0 and 1.
    Args:
        lambda_a (float): Coefficient of L1 loss.
        lambda_b (float): Coefficient of bounded loss.
        eps (float, optional): Epsilon to prevent divide by zero. Defaults to 1e-2.
    """

    def loss_fn(x: Tensor):
        return lambda_a * x + lambda_b * (1 / (1 - x + eps) + 1 / (x + eps))

    # Compute constant that sets min to zero
    xs = torch.linspace(0, 1, 1000)
    ys = loss_fn(xs)
    c = ys.min()
    return lambda x: loss_fn(x) - c


def depth_to_rays(depths, camtoworlds, Ks, near_plane, far_plane):
    height, width = depths.shape[1:3]
    viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

    FoVx = 2 * math.atan(width / (2 * Ks[0, 0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * Ks[0, 1, 1].item()))
    world_view_transform = viewmats[0].transpose(0, 1)
    projection_matrix = get_projection_matrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    ray_o = _depths_to_points(
        depths[0], world_view_transform, full_proj_transform, ray=True
    )
    return ray_o


def _depths_to_points(depthmap, world_view_transform, full_proj_transform, ray=False):
    c2w = (world_view_transform.T).inverse()
    H, W = depthmap.shape[:2]
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    if ray:
        return rays_d
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points
