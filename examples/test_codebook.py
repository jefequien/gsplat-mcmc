import torch

# ckpt_path = "examples/results/360_v2/3dgs_1m_codebook2/garden/ckpts/ckpt_6999.pt"
ckpt_path = "examples/results/360_v2/3dgs_1m_codebook_rewrite/garden/ckpts/ckpt_6999.pt"
splats = torch.load(ckpt_path)["splats"]
n_gs = len(splats["means"])

indices = splats["shN_indices"].cpu()
codebook = splats["shN_codebook"].cpu()
codebook_size = len(splats["shN_codebook"])


# codebook = codebook.reshape(2**16, -1)
# norm = codebook.abs().max(dim=-1)[0]
# # norm = torch.linalg.norm(codebook, dim=-1)
# norm_mask = norm < 0.05
# print(norm_mask.sum())
# print(norm.shape)
# print(norm.min(), norm.max())

# hist = torch.histogram(norm, bins=22)
# print(hist[1])
# print(hist[0].int())

codebook_counts = torch.bincount(indices.int(), minlength=codebook_size)
sampled_codebook_indices = torch.argsort(codebook_counts, descending=True)

sorted_counts = codebook_counts[sampled_codebook_indices]

for i in range(10):
    print(sorted_counts[i], sampled_codebook_indices[i])

# x = 0
# for idx in indices:
#     if idx == 1612:
#         x += 1
# print(x)

# _, counts = torch.unique(indices, return_counts=True)
# # print(counts)
# x = torch.unique(counts, return_counts=True)
# # print(x)
# hist = torch.histogram(counts.float(), bins=128)
