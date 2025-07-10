import torch
import torch.nn.functional as F

def backproject_reproject_zbuffer_scatter(source_img, depth_map, K, T_src, T_tgt):
    """
    Back-project target pixels to 3D, reproject into source view,
    apply Z-buffering using scatter_reduce, and resample source image.

    Args:
        source_img: (1, 3, H, W) tensor, source RGB image
        depth_map: (1, 1, H, W) tensor, target depth map
        K: (1, 3, 3) tensor, intrinsics
        T_src: (1, 4, 4) tensor, world-to-source
        T_tgt: (1, 4, 4) tensor, world-to-target

    Returns:
        warped: (1, 3, H, W) tensor, warped source image
        valid_mask: (1, 1, H, W) bool tensor, valid (unoccluded) mask
    """
    B, _, H, W = source_img.shape
    device = source_img.device
    N = H * W

    # 1. Make meshgrid of pixel coords
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=0).float().reshape(1, 3, -1)  # (1, 3, HW)

    # 2. Backproject to 3D
    depth = depth_map.view(B, 1, -1)
    cam_points = torch.inverse(K) @ pix_coords * depth  # (B, 3, HW)
    cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)  # (B, 4, HW)

    # 3. Transform to source camera frame
    world_points = torch.inverse(T_tgt) @ cam_points_h  # (B, 4, HW)
    cam_src = T_src @ world_points  # (B, 4, HW)
    z_src = cam_src[:, 2, :]  # (B, HW)
    cam_src = cam_src[:, :3, :] / (z_src.unsqueeze(1) + 1e-6)

    # 4. Project into source image
    pix_src = K @ cam_src  # (B, 3, HW)
    x_src = pix_src[:, 0, :]
    y_src = pix_src[:, 1, :]

    # 5. Filter valid projections
    valid = (
        (z_src > 0) &
        (x_src >= 0) & (x_src <= W - 1) &
        (y_src >= 0) & (y_src <= H - 1)
    )

    # 6. Discretize coordinates for z-buffer
    x_ind = x_src.floor().long().clamp(0, W - 1)
    y_ind = y_src.floor().long().clamp(0, H - 1)
    linear_idx = y_ind * W + x_ind  # (B, HW)

    # 7. Z-buffering via scatter_reduce
    z_vals = z_src.clone()
    z_vals[~valid] = float('inf')
    min_z = torch.full((B, H * W), float('inf'), device=device)
    min_z = min_z.scatter_reduce(1, linear_idx, z_vals, reduce='amin', include_self=True)

    # 8. Create valid mask: points that match the min depth
    matched_z = torch.gather(min_z, 1, linear_idx)
    occlusion_mask = torch.isclose(z_src, matched_z, rtol=1e-3, atol=1e-5)
    final_valid = valid & occlusion_mask

    # 9. Normalize coordinates for grid_sample
    x_norm = 2 * (x_src / (W - 1)) - 1
    y_norm = 2 * (y_src / (H - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).view(B, H, W, 2)

    # 10. Warp image
    warped = F.grid_sample(source_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 11. Valid mask
    valid_mask = final_valid.view(B, 1, H, W)
    return warped, valid_mask

def backproject_reproject(source_img, depth_map, K, T_src, T_tgt):
    """
    Back-project target view pixels using depth map, transform to source view,
    reproject to 2D, and sample from source image using bilinear interpolation.

    Args:
        source_img: (1, 3, H, W) tensor, source RGB image
        depth_map: (1, 1, H, W) tensor, depth map for target view
        K: (1, 3, 3) tensor, intrinsics
        T_src: (1, 4, 4) tensor, world-to-source
        T_tgt: (1, 4, 4) tensor, world-to-target

    Returns:
        (1, 3, H, W) tensor: Reprojected image
    """
    B, _, H, W = source_img.shape

    # Step 1: create a meshgrid of pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=depth_map.device),
        torch.arange(W, device=depth_map.device),
        indexing='ij'
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=0).float()  # (3, H, W)
    pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 3, H, W)

    # Step 2: backproject to 3D in target camera frame
    K_inv = torch.inverse(K)  # (B, 3, 3)
    cam_points = K_inv @ pix_coords.view(B, 3, -1)  # (B, 3, HW)
    cam_points = cam_points * depth_map.view(B, 1, -1)  # scale by depth
    cam_points_hom = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)  # (B, 4, HW)

    # Step 3: transform to source camera frame
    world_points = torch.inverse(T_tgt) @ cam_points_hom  # (B, 4, HW)
    cam_src = T_src @ world_points  # (B, 4, HW)
    cam_src = cam_src[:, :3] / cam_src[:, 2:3]  # (B, 3, HW)

    # Step 4: project to 2D source image plane
    pix_src = K @ cam_src  # (B, 3, HW)
    x_src = pix_src[:, 0].view(B, H, W)
    y_src = pix_src[:, 1].view(B, H, W)

    # Step 5: normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2 * (x_src / (W - 1)) - 1
    y_norm = 2 * (y_src / (H - 1)) - 1
    grid = torch.stack((x_norm, y_norm), dim=-1)  # (B, H, W, 2)

    # Step 6: sample the source image using bilinear interpolation
    warped = F.grid_sample(source_img, grid, mode='bilinear', align_corners=True, padding_mode='zeros')

    return warped