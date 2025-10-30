from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class DL3DVDataset:
    """A simple dataset class for DL3DVDataset data."""

    data_dir: str
    """The path to the scene"""
    split: Literal["train", "val"] = "train"
    """Which split to use."""
    test_every: int = 8
    """Every N images there is a test image."""

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.root_dir = self.data_dir.parent
        self.scene_folder, self.scene_name = self.data_dir.name.split("-")

        self.rgb_dir = (
            self.root_dir
            / "processed_dl3dv_ours"
            / self.scene_folder
            / self.scene_name
            / "dense"
            / "rgb"
        )
        image_paths = sorted(
            [
                p
                for p in self.rgb_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        frames = [imageio.imread(str(p))[..., :3] for p in image_paths]
        self.images = np.stack(frames, axis=0)
        self.image_height, self.image_width = self.images.shape[1:3]

        self.cam_dir = (
            self.root_dir
            / "processed_dl3dv_ours"
            / self.scene_folder
            / self.scene_name
            / "dense"
            / "cam"
        )
        cam_paths = sorted(
            [p for p in self.cam_dir.rglob("*") if p.suffix.lower() in {".npz"}]
        )
        cam_params = [np.load(str(p)) for p in cam_paths]
        self.intrinsics = cam_params[0]["intrinsic"]
        self.cam_to_worlds = np.array([p["pose"] for p in cam_params])

        # compute scene scale (as is done in the colmap parser)
        camera_locations = np.stack(self.cam_to_worlds, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Split indices
        self.indices = np.arange(len(self.images))
        if self.split == "train":
            self.indices = self.indices[self.indices % self.test_every != 0]
        else:
            self.indices = self.indices[self.indices % self.test_every == 0]
        
        # start_indices = {
        #     "2aadacbd3623dff25834e1f13cb6c1d6f91996e2957e8fd7de1ca7883e424393": 21,
        #     "50208cdb39510fdf8dedbd57536a6869dc93027d77608983925c9d7955578882": 260,
        #     "de3b622853799be8b8b5f2acd72c32f58e45b0b12f482c4b1f555a4602302424": 333,
        #     "2799116f2a663fee45296028841c2f838e525246a5dcb67a2e6699b7f3deabed": 149,
        #     "5dac8fa15625e54b1bd487b36701fb99c8ed909563b86ce3728caebfefde8dda": 276,
        # }
        # self.indices = np.arange(start_indices[self.scene_name], start_indices[self.scene_name] + 36, 5)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = dict(
            K=torch.from_numpy(self.intrinsics).float(),
            camtoworld=torch.from_numpy(self.cam_to_worlds[self.indices[item]]).float(),
            image=torch.from_numpy(self.images[self.indices[item]]).float(),
            image_id=item,
        )
        return data
