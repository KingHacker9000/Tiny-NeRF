# train/train_loop.py

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os

# project imports:
from data.loader import NeRFDataset
from data.rays   import RayGenerator
from models.nerf import TinyNeRF
from rendering.volume_render import volume_render, render_chunks
from train.eval  import evaluate_novel_views
from utils.visualization import plot_loss_curve
from utils.io import save_checkpoint, load_checkpoint


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
        "output": ["checkpoint_dir", "log_dir"],
        "eval": ["output_dir", "root_dir"],
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
    
    dataset = NeRFDataset(cfg['data']['root_dir'])
    
    rg = RayGenerator(dataset.intrinsics)  # Move to device
    dirs_cam = rg.compute_camera_dirs()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    return dataloader, rg, dirs_cam


def build_model_and_opt(cfg: dict, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
    """
    - Instantiate TinyNeRF(**cfg['model_params']) and move to device
    - Create optimizer (e.g. Adam) from cfg['training']['lr'] etc.
    - (Optionally) create LR scheduler from cfg['training']['scheduler']
    - Return (model, optimizer, scheduler_or_None)
    """
    
    model = TinyNeRF(**cfg['model_params']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    
    scheduler = None
    if 'scheduler' in cfg['training']:
        if cfg['training']['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['step_size'], gamma=cfg['training']['gamma'])
        elif cfg['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['T_max'])
    
    return model, optimizer, scheduler


def train_one_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    rg: RayGenerator,
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
    
    model.train()
    epoch_loss = 0.0
    num_rays = 0

    for i, batch in enumerate(dataloader):
        image_rgb = batch["image"].squeeze(0).permute(1,2,0).to(device)  # [H,W,3]
        gt_flat   = image_rgb.reshape(-1, 3)                              # [R,3]
        c2w       = batch["c2w"].squeeze(0).cpu().numpy()

        rays_o, rays_d = rg.get_world_rays(dirs_cam, c2w)

        optimizer.zero_grad()           # zero once per image
        running_loss = 0.0
        torch.cuda.reset_peak_memory_stats()
        print(f"batch {i}")

        for pred_chunk, idx in render_chunks(
            rays_o, rays_d, model,
            num_samples   = cfg["render_params"]["num_samples"],
            near          = cfg["render_params"]["near"],
            far           = cfg["render_params"]["far"],
            chunk_size    = cfg["render_params"]["chunk_size"],
            background_color = cfg["render_params"]["background_color"],
            normalize_weights = cfg["render_params"]["normalize_weights"],
        ):
            #print(f"--chunk {idx}")
            target = gt_flat[idx].to(device)            # slice GT to match chunk

            loss   = torch.nn.functional.mse_loss(pred_chunk, target, reduction="mean")
            loss.backward()

            running_loss += loss.item()

        print(torch.cuda.max_memory_allocated()/1e9, "GB peak")
        optimizer.step()

        avg_img_loss = running_loss              # already per-ray mean
        epoch_loss  += avg_img_loss

    return epoch_loss / len(dataloader)    # average per image


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
    
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg['training']['device'])
    dataloader, rg, dirs_cam = build_dataloader(cfg, device)
    model, optimizer, scheduler = build_model_and_opt(cfg, device)

    if args.resume:
        load_checkpoint(cfg['training']['checkpoint_dir'], model, optimizer, scheduler, map_location=device)
        pass
    loss_history = []
    for epoch in range(cfg['training']['num_epochs']):
        avg_loss = train_one_epoch(dataloader, model, optimizer,cfg, rg, dirs_cam, device)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{cfg['training']['num_epochs']}, Loss: {avg_loss:.4f}")
        
        if scheduler:
            scheduler.step()
        
        if (epoch + 1) % cfg['training'].get('checkpoint_interval', 1) == 0:
            save_checkpoint(model, optimizer, epoch + 1, cfg['output']['checkpoint_dir'], scheduler)
            pass
        
        if (epoch + 1) % cfg['training'].get('eval_interval', 5) == 0:
            model.eval()
            with torch.no_grad():
                evaluate_novel_views(model, rg, dirs_cam, cfg['eval'], device, )
            model.train()
        if (epoch + 1) % cfg['training'].get('plot_interval', 1) == 0:
            plot_loss_curve(loss_history, cfg['output']['log_dir'], epoch + 1)
    # Final save
    save_checkpoint(model, optimizer, cfg['training']['num_epochs'], cfg['output']['checkpoint_dir'], scheduler)
    plot_loss_curve(loss_history, cfg['output']['log_dir'])
    print("Training complete!")

if __name__ == "__main__":
    main()
