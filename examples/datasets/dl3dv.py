from dataclasses import dataclass
import json
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
    split: Literal["train", "test", "val"] = "train"
    """Which split to use."""

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.root_dir = self.data_dir.parent
        self.scene_name = self.data_dir.name
        
        self.rgb_dir = self.root_dir / "processed_dl3dv_ours" / "1K" / self.scene_name / "dense" / "rgb"
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

        self.cam_dir = self.root_dir / "processed_dl3dv_ours" / "1K" / self.scene_name / "dense" / "cam"
        cam_paths = sorted(
            [
                p
                for p in self.cam_dir.rglob("*")
                if p.suffix.lower() in {".npz"}
            ]
        )
        cam_params = [np.load(str(p)) for p in cam_paths]
        self.intrinsics = cam_params[0]["intrinsic"]
        self.cam_to_worlds = np.array([p["pose"] for p in cam_params])

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
            camtoworld=torch.from_numpy(self.cam_to_worlds[item]).float(),
            image=torch.from_numpy(self.images[item]).float(),
            image_id=item,
        )
        return data
