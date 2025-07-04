# data/loader.py

from torch.utils.data import Dataset
import torch
from utils.io import load_dataset
import os
import PIL.Image as Image
import numpy as np

class NeRFDataset(Dataset):
    """
    Loads rendered images + camera poses from disk.

    Args:
        root_dir: path to your dataset folder (must contain images/, poses.json, intrinsics.json)
        img_transform: optional torchvision/PIL transform to apply to each image
    """
    poses: dict[str, torch.Tensor]  # Maps image filenames to world→cam matrices
    intrinsics: dict[str, torch.Tensor]  # Camera intrinsics (fx, fy, cx, cy, width, height)

    def __init__(self, root_dir: str, img_transform=None):
        # 1) load poses.json into self.poses (world→cam matrices)
        # 2) load intrinsics.json into self.intrinsics
        # 3) build and sort self.keys from the pose dictionary
        # 4) store transform
        
        self.root_dir = root_dir
        if not os.path.exists(f'{root_dir}/poses.json'):
            raise FileNotFoundError(f"Cannot find poses.json in {root_dir}. Please check the dataset path.")
        if not os.path.exists(f'{root_dir}/intrinsics.json'):
            raise FileNotFoundError(f"Cannot find intrinsics.json in {root_dir}. Please check the dataset path.")

        dataset = load_dataset(root_dir)
        assert 'poses' in dataset and 'intrinsics' in dataset, "Invalid dataset, possibly missing files."
        self.poses = dataset['poses']
        self.intrinsics = dataset['intrinsics']
        self.keys = dataset['image_indexes']

        self.img_transform = img_transform

        for i in range(len(self.keys)):
            w2c = self.poses[self.keys[i]]
            self.poses[self.keys[i]] = torch.linalg.inv(w2c)

    def __len__(self) -> int:
        """Return number of views (len(self.keys))."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor [3,H,W], float in [0,1]
            c2w:   Tensor [4,4], camera-to-world matrix
        """
        # 1) open image file for self.keys[idx]
        # 2) apply self.img_transform
        # 3) read world→cam from self.poses, invert to c2w
        # 4) return (image, c2w)
        
        image_path = os.path.join(self.root_dir, 'images', f'{self.keys[idx]}.png')
        image = Image.open(image_path).convert('RGB')

        if self.img_transform:
            image_tensor = self.img_transform(image)
            assert image_tensor.shape[0] == 3, "Image must be a tensor with 3 channels (RGB)"
        else:
            np_image = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            # Convert to dtype float32
            image_tensor = image_tensor.float()

        c2w = self.poses[self.keys[idx]]

        return {
            'image': image_tensor,
            'c2w': c2w
        }
        
if __name__ == "__main__":
    # Example usage
    dataset = NeRFDataset('dataset/')
    print(f"Number of views: {len(dataset)}")
    d = dataset[0]
    image, c2w = d['image'], d['c2w']
    print(f"Image shape: {image.shape}, Camera-to-world matrix shape: {c2w.shape}")
    print(f"Intrinsics: {dataset.intrinsics}")