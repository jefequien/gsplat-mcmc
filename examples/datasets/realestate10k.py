from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class Realestate10kDataset:
    """A simple dataset class for Real Estate 10k data."""

    data_dir: str
    """The path to the scene, consisting of renders and transforms.json"""
    split: Literal["train", "test", "val"] = "train"
    """Which split to use."""

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.scene_name = self.data_dir.name
        
        self.video_path = self.data_dir.parent / "training_256" / f"{self.scene_name}.mp4"
        reader = imageio.get_reader(self.video_path)
        frames = []
        for frame in reader:
            frames.append(frame[..., :3])
        reader.close()
        self.images = np.stack(frames, axis=0)
        self.image_height, self.image_width = self.images.shape[1:3]

        training_poses = torch.load(self.data_dir.parent / "training_poses" / f"{self.scene_name}.pt").numpy()
        # fx = training_poses[0, 0] * self.image_width # Bad due to cropping
        fy = training_poses[0, 1] * self.image_height
        cx = training_poses[0, 2] * self.image_width
        cy = training_poses[0, 3] * self.image_height
        self.intrinsics = np.array([[fy, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.cam_to_worlds = training_poses[:, -12:].reshape(-1, 3, 4)
        self.cam_to_worlds = np.concatenate(
            [
                self.cam_to_worlds,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(self.cam_to_worlds), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]
        self.cam_to_worlds = np.array([np.linalg.inv(p) for p in self.cam_to_worlds])

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
