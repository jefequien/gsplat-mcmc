from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import torch
from einops import rearrange

# Add rae to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "VideoRAE"))
from src.utils.datasets import DL3DVDataset


@dataclass
class RAEDataset:
    """A simple dataset class for RAE data."""

    data_dir: str
    """The path to the RAE scene, consisting of renders and transforms.json"""
    split: Literal["train", "test", "val"] = "train"
    """Which split to use."""

    def __post_init__(self):
        root_dir = Path(self.data_dir).parent
        index = int(Path(self.data_dir).name)
        self.dataset = DL3DVDataset(
            root_dir=root_dir,
            split="train",
            skips=[10],
            num_frames=30,
            image_size=256,
            use_random_start=False,
        )
        video, y_dict = self.dataset[index]

        self.images = video
        self.Ks = y_dict["Ks"]
        self.cam_to_worlds = torch.inverse(y_dict["viewmats"])
        # Blender scenes are small. Scale them by 10.0 to make them similar to COLMAP scenes.
        self.cam_to_worlds[:,:3,3] *= 10.0
        self.image_height, self.image_width = self.images.shape[-2:]
        self.intrinsics = y_dict["Ks"][0].numpy()
        self.intrinsics[0, :] *= self.image_width
        self.intrinsics[1, :] *= self.image_height

        # compute scene scale (as is done in the colmap parser)
        camera_locations = np.stack(self.cam_to_worlds, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = dict(
            K=torch.from_numpy(self.intrinsics).float(),
            camtoworld=self.cam_to_worlds[item],
            image=rearrange(self.images[item], "c h w -> h w c") * 255.0,
            image_id=item,
        )
        return data
