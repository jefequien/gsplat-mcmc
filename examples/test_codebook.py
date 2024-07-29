import torch
import numpy as np

# ckpt_path = "examples/results/360_v2/3dgs_1m_finetune/garden_record/ckpts/ckpt_29999.pt"
ckpt_path = "examples/results/360_v2/3dgs_1m_codebook_mcmc/garden/ckpts/ckpt_14999.pt"
splats = torch.load(ckpt_path)["splats"]
n_gs = len(splats["means"])

indices = splats["shN_indices"].cpu()
codebook = splats["shN_codebook"].cpu()

# shN0 = splats["shN"].cpu()
# codebook = splats["centroids"].reshape(-1, 15, 3).cpu()
# npz = np.load("examples/results/360_v2/3dgs_1m_finetune/garden_record/compress/shN.npz")
# indices = torch.tensor(npz["labels"]).int()
# shN = codebook[indices]
# print((shN - shN0).abs().max())

codebook = codebook.reshape(2**16, -1)
norm = codebook.abs().max(dim=-1)[0]
# print(norm.shape)
# norm = torch.linalg.norm(codebook, dim=-1)
# norm_mask = (norm < 0.01) & (norm != 0)
# print(norm_mask.sum())
# print(norm.shape)
# print(norm.min(), norm.max())

hist = torch.histogram(norm, bins=22)
print(hist[0].shape, hist[1].shape)
print(hist[0].int())
print(hist[1])

codebook_size = len(codebook)
codebook_counts = torch.bincount(indices.int(), minlength=codebook_size)
sampled_codebook_indices = torch.argsort(codebook_counts, descending=False)
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
