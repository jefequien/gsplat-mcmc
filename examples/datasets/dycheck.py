import os
import json
import glob
import numpy as np
import torch
from PIL import Image


class DycheckParser:
    """
    Parses the Dycheck dataset structure and preloads all metadata.

    Args:
        root_dir (str): Root path to the Dycheck dataset.
        factor (float): Downscaling factor for images and intrinsics (e.g., 1.0, 4.0).
    """
    def __init__(self, root_dir, factor=1.0):
        self.root = root_dir
        self.factor = factor

        # Set image and camera paths
        self.image_dir = os.path.join(root_dir, f"rgb/{int(factor)}x")
        self.camera_dir = os.path.join(root_dir, "camera")

        # Collect and sort file paths
        self.all_image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.all_camera_paths = sorted(glob.glob(os.path.join(self.camera_dir, "*.json")))

        if len(self.all_image_paths) != len(self.all_camera_paths):
            raise ValueError(f"Mismatch: {len(self.all_image_paths)} images vs {len(self.all_camera_paths)} cameras.")

        self.num_all_frames = len(self.all_image_paths)

        # Map frame basename (no extension) to index
        self.name_to_index = {
            os.path.splitext(os.path.basename(p))[0]: i
            for i, p in enumerate(self.all_image_paths)
        }

        # Load all camera poses (OpenCV convention, scale positions by 30.0)
        self.camtoworlds = self._load_all_camtoworlds()

        # Compute scene scale (max distance from center)
        camera_positions = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_positions, axis=0)
        dists = np.linalg.norm(camera_positions - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Estimate horizontal field of view (camera_angle_x)
        cam = self._load_camera(0)
        fx = cam["focal_length"] / self.factor
        width = cam["image_size"][0] / self.factor
        self.camera_angle_x = 2 * np.arctan(width / (2 * fx))

        # Load all time_ids from split files and normalize to [0, 1]
        self.time_ids = self._load_all_time_ids()
        self.image_times = self.time_ids / self.time_ids.max()

        # Store intrinsics and image sizes per frame
        self.Ks_dict = {}
        self.imsize_dict = {}
        for i in range(self.num_all_frames):
            cam = self._load_camera(i)
            frame_name = os.path.splitext(os.path.basename(self.all_image_paths[i]))[0]
            K = self.load_intrinsics(cam)
            self.Ks_dict[frame_name] = K
            width = int(cam["image_size"][0] / self.factor)
            height = int(cam["image_size"][1] / self.factor)
            self.imsize_dict[frame_name] = (width, height)

    def _load_camera(self, index):
        with open(self.all_camera_paths[index], "r") as f:
            return json.load(f)

    def _load_all_camtoworlds(self):
        camtoworlds = []
        for cam_path in self.all_camera_paths:
            with open(cam_path, "r") as f:
                cam = json.load(f)
            R = np.array(cam["orientation"]).T  # OpenCV camera-to-world
            t = np.array(cam["position"]) * 30.0
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            camtoworlds.append(c2w)
        return np.stack(camtoworlds, axis=0)

    def _load_all_time_ids(self):
        """
        Loads and aggregates time_ids from all split files (train/val).
        Assumes each frame in the dataset appears in at least one split.
        """
        time_ids = np.zeros(self.num_all_frames, dtype=np.float32)

        for split_name in ["train", "val"]:
            split_path = os.path.join(self.root, "splits", f"{split_name}.json")
            if not os.path.exists(split_path):
                continue
            with open(split_path, "r") as f:
                split_data = json.load(f)
            for name, tid in zip(split_data["frame_names"], split_data["time_ids"]):
                idx = self.name_to_index[name]
                time_ids[idx] = tid
        return time_ids

    def load_image(self, index):
        image_path = self.all_image_paths[index]
        img = Image.open(image_path).convert("RGB")
        return np.asarray(img).astype(np.uint8)

    def load_intrinsics(self, cam):
        fx = cam["focal_length"] / self.factor
        fy = fx * cam["pixel_aspect_ratio"]
        cx, cy = cam["principal_point"]
        cx /= self.factor
        cy /= self.factor
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K

    def load_pose(self, cam):
        R = np.array(cam["orientation"]).T
        t = np.array(cam["position"]) * 30.0
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        return c2w


class DycheckDataset:
    """
    Provides access to a split of the Dycheck dataset using a shared DycheckParser.

    Args:
        parser (DycheckParser): Parser containing dataset metadata and data access methods.
        split (str): One of "train" or "val".
    """
    def __init__(self, parser: DycheckParser, split: str):
        self.parser = parser

        # Load split data
        split_path = os.path.join(parser.root, "splits", f"{split}.json")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            split_data = json.load(f)

        frame_names = split_data["frame_names"]
        self.indices = [parser.name_to_index[name] for name in frame_names]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.indices[idx]
        cam = self.parser._load_camera(item)
        K = self.parser.load_intrinsics(cam)
        pose = torch.from_numpy(self.parser.load_pose(cam)).float()
        image = torch.from_numpy(self.parser.load_image(item)).float()
        image_time = torch.tensor(self.parser.image_times[item]).float()

        return {
            "K": torch.from_numpy(K).float(),
            "camtoworld": pose,
            "image": image,  # shape: (H, W, 3)
            "image_id": item,
            "image_time": image_time,
        }




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/iphone/apple")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    parser = DycheckParser(args.data_dir, factor=args.factor)
    dataset = DycheckDataset(parser, split="train")
    print(f"Dataset: {len(dataset)} images.")
    for item in dataset:
        print(item["image_id"], item["image_time"], item["camtoworld"][:3, 3])
