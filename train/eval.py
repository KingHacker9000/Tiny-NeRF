# train/eval.py

import torch
from PIL import Image
import os
from data.rays import RayGenerator
from rendering.volume_render import volume_render
from utils.io import save_predicted_image
from data.loader import NeRFDataset

def evaluate_novel_views(
    model: torch.nn.Module,
    ray_gen: RayGenerator,          # your RayGenerator instance
    dirs_cam: torch.Tensor,
    eval_cfg: dict,
    device: torch.device,
    limit: int = 3
) -> None:
    """
    Renders a handful of held-out poses and reports PSNR and SSIM.

    Args:
        model: trained TinyNeRF
        ray_gen: RayGenerator to produce rays_o, rays_d
        dirs_cam: precomputed camera-space dirs [H,W,3]
        eval_cfg: dict with keys like
                  - 'novel_poses': list of 4x4 matrices
                  - 'near', 'far', 'num_samples'
                  - 'output_dir' where to save renderings
        device: torch device
    """
    # for each c2w in eval_cfg['novel_poses']:
    #   rays_o, rays_d = ray_gen.get_world_rays(dirs_cam, c2w)
    #   rgb_pred = volume_render(rays_o.to(device), rays_d.to(device), model, **eval_cfg)
    #   compare rgb_pred vs. ground-truth if available
    #   compute PSNR = -10 * log10(MSE)
    #   save predicted images to disk
    # at end, print average PSNR and/or SSIM
    
    os.makedirs(eval_cfg['output_dir'], exist_ok=True)

    model.eval()
    with torch.no_grad():
        dataset = NeRFDataset(eval_cfg['root_dir'])
        for idx, data in enumerate(dataset):
            if limit > 0 and idx >= limit:
                break
            c2w = data['c2w']
            if isinstance(c2w, torch.Tensor):
                c2w = c2w.detach().cpu().numpy()
            rays_o, rays_d = ray_gen.get_world_rays(dirs_cam, c2w)
            rays_o, rays_d = rays_o.to(device), rays_d.to(device)
            
            # Render the image using the model
            rgb_pred = volume_render(rays_o, rays_d, model, 
                                near=eval_cfg['near'], 
                                far=eval_cfg['far'], 
                                num_samples=eval_cfg['num_samples'])
            
            H, W = dirs_cam.shape[:2]
            rgb_image = rgb_pred.view(H, W, 3)
            rgb_image = rgb_image.permute(2, 0, 1)
            
            ### TODO: COMPARE TO GROUND-TRUTH LATER ###
            
            print(f"Rendered image {idx+1}/{limit if limit > 0 else len(dataset)}: shape {rgb_image.shape}")
            
            # Save the rendered image to disk
            output_path = f"{eval_cfg['output_dir']}/rendered_{idx:03d}.png"
            save_predicted_image(rgb_image, output_path)


if __name__ == "__main__":
    # Example usage
    # Assuming you have a trained model, ray_gen, and dirs_cam ready
    from models.nerf import TinyNeRF  # Replace with your actual model import
    from utils.model_utils import load_model_for_inference
    model = TinyNeRF()  # Load your trained model here
    load_model_for_inference(
        checkpoint_path='checkpoints/checkpoint_epoch_0002.pth',
        model_class=TinyNeRF,
        model_kwargs= {'body_depth': 4, "color_head_depth": 2, "width": 64, "pos_freqs": 10, "dir_freqs": 4, "skip_layer": 3},
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    from utils.io import load_dataset  # Replace with your actual dataset loading function
    intrinsics = load_dataset()['intrinsics']  # Replace with your actual dataset loading function
    ray_gen = RayGenerator(intrinsics=intrinsics)  # Replace with your actual RayGenerator instance
    dirs_cam = ray_gen.compute_camera_dirs().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    eval_cfg = {
        'novel_poses': [torch.eye(4) for _ in range(5)],  # Example poses
        'near': 0.1,
        'far': 100.0,
        'num_samples': 64,
        'output_dir': 'renders',
        'root_dir': 'C:/Dev/Random_Projects/NeRF_Demo/dataset'  # Path to your dataset
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move model to the appropriate device
    
    evaluate_novel_views(model, ray_gen, dirs_cam, eval_cfg, device)