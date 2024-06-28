import os
import imageio
import json
import pandas as pd
from tabulate import tabulate
from collections import defaultdict
import numpy as np

experiments = ["3dgs_random", "3dgs_sfm", "mcmc_random", "mcmc_sfm"]
# scenes = ["counter", "stump", "kitchen", "bicycle", "bonsai", "room", "garden"]
scenes = ["counter", "kitchen", "bonsai", "room"]

psnrs = defaultdict(dict)
num_gss = defaultdict(dict)
for experiment in experiments:
    for scene in scenes:
        stats_path = f"results/{experiment}/{scene}/stats/val_step29999.json"
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
        else:
            stats = {"psnr": 0.0, "num_GS": 0.0}

        psnrs[experiment][scene] = f"{stats['psnr']:.2f}"
        num_gss[experiment][scene] = f"{stats['num_GS'] / 1e6:.2f}M"

print(tabulate(pd.DataFrame(psnrs), headers="keys", tablefmt="github"))
print(tabulate(pd.DataFrame(num_gss), headers="keys", tablefmt="github"))


# for scene in scenes:
#     steps = [0] + [(i + 1) * 100 - 1 for i in range(300)]
#     image_dir = f"./results/benchmark/{scene}/renders"

#     # save to video
#     video_dir = f"./results/videos"
#     os.makedirs(video_dir, exist_ok=True)
#     writer = imageio.get_writer(f"{video_dir}/{scene}.mp4", fps=30)
#     for step in steps:
#         canvas = imageio.imread(f"{image_dir}/{step:06d}/val_0011.png")
#         writer.append_data(canvas)
#     writer.close()
#     print(f"Video saved to {video_dir}/{scene}.mp4")
