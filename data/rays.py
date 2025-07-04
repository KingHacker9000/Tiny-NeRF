# data/rays.py
import numpy as np
import torch

def _compute_camera_dirs(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> torch.Tensor:
    """
    Precompute camera-space ray directions.

    Args:
        width, height: image resolution
        fx, fy: focal lengths in pixels
        cx, cy: principal point offsets in pixels

    Returns:
        dirs_cam: Tensor of shape [H, W, 3] where
                  dirs_cam[j,i] = ((i-cx)/fx, (j-cy)/fy, 1)
    """

    vec_X = (torch.arange(width, dtype=torch.float32) - cx) / fx
    vec_Y = (torch.arange(height, dtype=torch.float32) - cy) / fy

    dirs_x, dirs_y = torch.meshgrid(vec_X, vec_Y, indexing='xy')
    dirs_z = torch.ones_like(dirs_x)

    dirs_cam = torch.stack((dirs_x, dirs_y, dirs_z), dim=-1)

    return dirs_cam


def _normalize_camera_dirs(
    camera_dirs: torch.Tensor
) -> torch.Tensor:
    """
    Normalize camera-space ray directions.

    Args:
        camera_dirs: Tensor of shape [H, W, 3] with unnormalized directions

    Returns:
        dirs_cam: Tensor of shape [H, W, 3] with normalized directions
    """
    # Normalize the directions
    norms = camera_dirs.norm(dim=-1, keepdim=True)
    dirs_cam = camera_dirs / norms

    return dirs_cam

def _get_world_rays(
    dirs_cam: torch.Tensor,
    c2w: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn camera-space directions into world-space rays.

    Args:
        dirs_cam: [H, W, 3] camera-space vectors
        c2w: (4x4) camera-to-world extrinsic matrix

    Returns:
        rays_o: [H*W, 3] ray origins in world space
        rays_d: [H*W, 3] ray directions in world space (normalized)
    """

    # Assertions
    assert isinstance(dirs_cam, torch.Tensor), "dirs_cam must be a torch.Tensor."
    assert dirs_cam.shape[-1] == 3, "dirs_cam must have shape [H, W, 3]."
    assert isinstance(c2w, np.ndarray) and c2w.shape == (4, 4), "c2w must be a (4x4) numpy array."

    # Normalize camera directions
    dirs_cam = _normalize_camera_dirs(dirs_cam)

    # Reshape dirs_cam to [H*W, 3]
    dirs_cam_flat = dirs_cam.reshape(-1, 3)

    # Convert c2w to a torch tensor
    c2w_tensor = torch.from_numpy(c2w).to(dirs_cam.dtype)#.to(dirs_cam.device)

    # Extract rotation and translation
    rotation = c2w_tensor[:3, :3]
    translation = c2w_tensor[:3, 3]

    # Compute ray directions in world space
    rays_d = dirs_cam_flat @ rotation.T

    # Normalize the ray directions
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Compute ray origins (camera position in world space)
    N = rays_d.shape[0]  # Number of rays
    rays_o = translation.unsqueeze(0).expand(N,3)  # Expand does NOT actually copy data

    return rays_o, rays_d

class RayGenerator:
    """
    Utility to manage ray creation for an entire dataset.

    Usage:
        rg = RayGenerator(intrinsics)
        # Precompute once:
        dirs_cam = rg.compute_camera_dirs()
        # For each sample:
        rays_o, rays_d = rg.get_world_rays(dirs_cam, c2w_matrix)
    """

    def __init__(self, intrinsics: dict):
        """
        Parse intrinsics JSON into fx, fy, cx, cy, width, height.
        """

        # Assertions
        assert isinstance(intrinsics, dict), "Intrinsics must be a dictionary."
        for key in ['resolution_x', 'resolution_y', 'f_mm', 'sensor_width_mm']:
            assert key in intrinsics, f"Missing required key '{key}' in intrinsics."
            
        # Get Height and Width
        self.width = intrinsics['resolution_x']
        self.height = intrinsics['resolution_y']

        # Get Focal Lengths in pixels
        self.fx = intrinsics['f_mm'] * self.width / intrinsics['sensor_width_mm']
        if 'sensor_height_mm' not in intrinsics:
            print("Warning: 'sensor_height_mm' not found in intrinsics, assuming square sensor.")
            sensor_height_mm = intrinsics['sensor_width_mm']
        else:
            sensor_height_mm = intrinsics['sensor_height_mm']
        self.fy = intrinsics['f_mm'] * self.height / sensor_height_mm

        # Get Principal Point Offsets in pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def compute_camera_dirs(self) -> torch.Tensor:
        """
        Calls compute_camera_dirs(...) with stored intrinsics.
        """
        return _compute_camera_dirs(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

    def get_world_rays(
        self,
        dirs_cam: torch.Tensor,
        c2w: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calls get_world_rays(...) with the given dirs_cam and c2w.
        """
        assert isinstance(dirs_cam, torch.Tensor), "dirs_cam must be a torch.Tensor."
        assert dirs_cam.shape == (self.height, self.width, 3), \
            f"dirs_cam must have shape ({self.height}, {self.width}, 3), but got {dirs_cam.shape}."
        assert isinstance(c2w, np.ndarray) and c2w.shape == (4, 4), f"c2w must be a (4x4) numpy array. {c2w.shape}"

        return _get_world_rays(dirs_cam, c2w)

      
    
    def __repr__(self) -> str:
        return (f"RayGenerator(width={self.width}, height={self.height}, "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})")
