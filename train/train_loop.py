# train/train_loop.py

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os

# your project imports:
# from data.loader import NeRFDataset
from data.rays   import RayGenerator
# from models.nerf import TinyNeRF
# from rendering.volume_render import volume_render
# from train.eval  import evaluate_novel_views
# from utils.io    import save_checkpoint, load_checkpoint
# from utils.visualization import plot_loss_curve


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options.
    Returns:
        args.config: YAML config file
        args.resume: optional checkpoint path
        args.device: 'cuda' or 'cpu'
    """
    parser = argparse.ArgumentParser(description="Train a TinyNeRF model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run training on (e.g., "cuda" or "cpu").')
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        parser.error(f"Config file {args.config!r} does not exist.")
    return args


def load_config(path: str) -> dict:
    """
    Load and return the configuration dictionary from a YAML file.
    """
    
    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file {path!r}: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path!r} must contain a dictionary.")
    
    # Ensure required
    missing = []
    for sec, subkeys in {
        "data": ["root_dir", "intrinsics"],
        "training": ["lr", "num_epochs"],
        "output": ["checkpoint_dir", "log_dir"]
    }.items():
        if sec not in config:
            missing.append(sec)
        else:
            for sub in subkeys:
                if sub not in config[sec]:
                    missing.append(f"{sec}.{sub}")
    if missing:
        raise ValueError(f"Config missing required entries: {missing}")

    # Ensure defaults
    for sec, subkeys in {
        "training": ["device"],
        "model_params": ["body_depth", "color_head_depth", "width", "pos_freqs", "dir_freqs", "skip_layer"],
        "render_params": ["num_samples", "near", "far", "chunk_size", "background_color", "normalize_weights"]
    }.items():
        for sub in subkeys:
            if sub not in config[sec]:
                config[sec][sub] = None  # Set default to None

    return config


def build_dataloader(cfg: dict, device: torch.device) -> tuple[DataLoader, RayGenerator, torch.Tensor]:
    """
    - Instantiate NeRFDataset(cfg['data']['root_dir'])
    - Instantiate RayGenerator(cfg['data']['intrinsics'])
    - Precompute camera-space dirs: dirs_cam = rg.compute_camera_dirs()
    - Wrap dataset in DataLoader (batch_size=1, shuffle=True)
    - Return (dataloader, rg, dirs_cam)
    """
    ...


def build_model_and_opt(cfg: dict, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
    """
    - Instantiate TinyNeRF(**cfg['model_params']) and move to device
    - Create optimizer (e.g. Adam) from cfg['training']['lr'] etc.
    - (Optionally) create LR scheduler from cfg['training']['scheduler']
    - Return (model, optimizer, scheduler_or_None)
    """
    ...


def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    rg: 'RayGenerator',
    dirs_cam: torch.Tensor,
    device: torch.device
) -> float:
    """
    - Loop over dataloader:
        * Load one view: image RGBs + its pose (c2w)
        * Use rg.get_world_rays(dirs_cam, c2w) → rays_o, rays_d (move to device)
        * Call volume_render(rays_o, rays_d, model, **cfg['render_params'])
        * Compute loss = MSE(predicted_rgb, true_rgb)
        * Backprop + optimizer.step()
        * Accumulate loss
    - Return average loss over all images
    """
    ...


def main():
    # 1) args = parse_args()
    # 2) cfg  = load_config(args.config)
    # 3) device = torch.device(cfg['training']['device'])
    #
    # 4) dataloader, rg, dirs_cam = build_dataloader(cfg, device)
    # 5) model, optimizer, scheduler = build_model_and_opt(cfg, device)
    #
    # 6) optionally resume from checkpoint
    #
    # 7) for epoch in range(...):
    #       avg_loss = train_one_epoch(...)
    #       log/print loss
    #       scheduler.step() if exists
    #       save_checkpoint(...) at intervals
    #       evaluate_novel_views(...) at intervals
    #
    # 8) final save + plot_loss_curve(...)
    ...
    

if __name__ == "__main__":
    main()
