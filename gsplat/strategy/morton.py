import torch

# -------- bit helpers --------------------------------------------------------
def _expand_bits(v: torch.Tensor) -> torch.Tensor:
    """
    Interleave 10 zero bits between the lower 10 bits of v.
    Works on uint32 tensors (CPU or GPU).
    """
    v = (v & 0x3ff)                     # keep only 10 LSBs
    v = (v | (v << 16)) & 0x30000ff
    v = (v | (v << 8))  & 0x300f00f
    v = (v | (v << 4))  & 0x30c30c3
    v = (v | (v << 2))  & 0x9249249     # final 0b100100100... mask
    return v

def _morton3D(x, y, z):
    return (_expand_bits(x) << 2) | (_expand_bits(y) << 1) | _expand_bits(z)

# -------- public API ---------------------------------------------------------
def morton_sort(points: torch.Tensor, voxel_size: float = 1.0) -> torch.Tensor:
    """
    Args
    ----
    points      : (N, 3) float32/float64 tensor, any device
    voxel_size  : scalar size that maps world units -> grid units (default 1.0)

    Returns
    -------
    sorted_idx  : (N,) int64 tensor of indices that sorts points in Morton order
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("input must be (N,3)")

    device = points.device
    pts = points / voxel_size            # scale to voxel grid
    mins = pts.min(dim=0).values.floor() # shift so all coords are >= 0
    grid = (pts - mins).floor().to(torch.int64)

    # ---- compute 30‑bit Morton code ----------------------------------------
    if not torch.cuda.is_available() or device.type == "cpu":
        grid = grid.to(torch.int32)      # uint32 ops cheaper on CPU
    x, y, z = [grid[:, i].to(torch.int32) for i in range(3)]
    codes = _morton3D(x, y, z)

    # ---- argsort gives the permutation -------------------------------------
    return torch.argsort(codes)

# ---------------- example ----------------------------------------------------
if __name__ == "__main__":
    import time
    N = 100_000_000
    pts = torch.randn(N, 3, device="cuda") * 5.0  # random points on GPU
    print(torch.linalg.norm(pts[0:-1] - pts[1:], dim=-1).mean())
    t0 = time.time()
    idx = morton_sort(pts, voxel_size=0.05)
    pts = pts[idx]                # ready for cache‑friendly traversal
    t1 = time.time()
    print(torch.linalg.norm(pts[0:-1] - pts[1:], dim=-1).mean())
    print(t1 - t0, pts.shape)
