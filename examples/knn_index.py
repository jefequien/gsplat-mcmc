import torch
from torch_cluster import knn_graph

@torch.no_grad()
def knn_index_matrix(means: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k nearest neighbors (including self) using torch_cluster.knn_graph.

    Args:
        means: (n, 3) tensor of point positions
        k: number of neighbors (including self)

    Returns:
        neighbors: (n, k) tensor of neighbor indices
    """
    n = means.size(0)
    
    # returns edge_index: shape (2, n * k)
    edge_index = knn_graph(means, k=k, loop=True)  # loop=True includes self

    # edge_index[0] are the query indices (rows), edge_index[1] are their neighbors
    row, col = edge_index

    # reshape neighbor indices into (n, k)
    neighbors = col.view(n, k)

    return neighbors

from pytorch3d.ops import knn_points

def approx_knn_pytorch3d(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Approximate k-NN using PyTorch3D's fast GPU method.

    Args:
        x: (N, 3) float tensor
        k: number of neighbors

    Returns:
        neighbor_idx: (N, k) long tensor
    """
    x_ = x.unsqueeze(0)  # (1, N, 3)
    _, idx, _ = knn_points(x_, x_, K=k, return_sorted=True)
    return idx.squeeze(0)  # (N, k)

# ---------------- example ----------------------------------------------------
if __name__ == "__main__":
    import time
    N = 4_000_000
    means = torch.randn(N, 3, device="cuda")
    t0 = time.time()
    nbrs = approx_knn_pytorch3d(means, k=4)  # (1000, 4) includes self
    t1 = time.time()
    print(t1 - t0, nbrs.shape)
