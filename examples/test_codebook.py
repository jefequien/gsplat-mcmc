import torch

ckpt_path = "examples/results/360_v2/3dgs_1m_codebook2/garden/ckpts/ckpt_6999.pt"
splats = torch.load(ckpt_path)["splats"]
n_gs = len(splats["means"])

indices = splats["shN_indices"].cpu()
codebook = splats["shN_codebook"].cpu()

codebook = codebook.reshape(2**16, -1)
norm = codebook.abs().max(dim=-1)[0]
# norm = torch.linalg.norm(codebook, dim=-1)
norm_mask = norm < 0.05
print(norm_mask.sum())
print(norm.shape)
print(norm.min(), norm.max())

hist = torch.histogram(norm, bins=22)
print(hist[1])
print(hist[0].int())

_, counts = torch.unique(indices, return_counts=True)
# print(counts)
x = torch.unique(counts, return_counts=True)
# print(x)
hist = torch.histogram(counts.float(), bins=128)
