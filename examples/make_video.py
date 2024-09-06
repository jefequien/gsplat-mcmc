from PIL import Image
import imageio
import numpy as np
from tqdm import tqdm

scene_dir = "results/codebook/benchmark_mcmc_1M_png_compression/garden"
writer = imageio.get_writer(f"{scene_dir}/debug.mp4", fps=30)

for step in tqdm(range(0, 29000, 100)):
    image = Image.open(f"{scene_dir}/renders/debug_step{step}_0000.png")
    writer.append_data(np.array(image))
writer.close()
