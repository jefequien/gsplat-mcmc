from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class BlenderDataset:
    """A simple dataset class for synthetic blender data."""

    data_dir: str
    """The path to the blender scene, consisting of renders and transforms.json"""
    split: Literal["train", "test", "val"] = "train"
    """Which split to use."""

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        transforms_path = self.data_dir / f"transforms_{self.split}.json"
        if not transforms_path.exists():
            transforms_path = self.data_dir / f"transforms.json"
        with transforms_path.open("r") as transforms_handle:
            transforms = json.load(transforms_handle)
        image_ids = []
        cam_to_worlds = []
        images = []
        masks = []
        for frame in transforms["frames"]:
            image_id = frame["file_path"].replace("./", "")
            image_ids.append(image_id)
            file_path = self.data_dir / f"{image_id}.png"
            image = imageio.imread(file_path)
            images.append(image[..., :3])
            # mask = image[..., 3] if image.shape[2] == 4 else None
            mask = None
            masks.append(mask)

            c2w = np.array(frame["transform_matrix"])
            # Convert from OpenGL to OpenCV coordinate system
            c2w[0:3, 1:3] *= -1
            cam_to_worlds.append(c2w)

        self.image_ids = image_ids
        self.cam_to_worlds = np.array(cam_to_worlds)
        # Blender scenes are small. Scale them by 10.0 to make them similar to COLMAP scenes.
        self.cam_to_worlds[:,:3,3] *= 10.0
        self.images = images
        self.masks = masks

        # all renders have the same intrinsics
        # see also
        # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/blender_dataparser.py
        image_height, image_width = self.images[0].shape[:2]
        cx = image_width / 2.0
        cy = image_height / 2.0
        if "camera_angle_x" in transforms:
            fx = 0.5 * image_width / np.tan(0.5 * transforms["camera_angle_x"])
            fy = fx
        else:
            fy = transforms["fl_y"]
            fx = fy
        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )
        self.image_height = image_height
        self.image_width = image_width

        # compute scene scale (as is done in the colmap parser)
        camera_locations = np.stack(self.cam_to_worlds, axis=0)[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = dict(
            K=torch.from_numpy(self.intrinsics).float(),
            camtoworld=torch.from_numpy(self.cam_to_worlds[item]).float(),
            image=torch.from_numpy(self.images[item]).float(),
            image_id=item,
        )
        if self.masks[item] is not None:
            data["mask"] = torch.from_numpy(self.masks[item]).bool()
        return data
