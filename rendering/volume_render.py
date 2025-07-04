"""
rendering/volume_render.py

Volume rendering utilities: stratified sampling along rays and alpha compositing.
"""
import torch
from typing import Optional, Union


def stratified_sampling(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    num_samples: int,
    near: float,
    far: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points uniformly (stratified) along each ray between near and far.

    Args:
        rays_o: [R, 3] ray origins
        rays_d: [R, 3] ray directions (assumed normalized)
        num_samples: number of samples per ray
        near, far: near and far plane distances

    Returns:
        pts: [R, num_samples, 3] sampled 3D points along each ray
        deltas: [R, num_samples, 1] distance intervals between samples
    """
    # implement stratified sampling of t values and compute pts and deltas
    
    R = rays_o.shape[0]  # number of rays
    edges = torch.linspace(near, far, num_samples+1, device=rays_o.device)  # shape (N+1,)
    bins  = edges[:-1]
    # Add jitter to t_vals for stratified sampling
    t_vals = bins.unsqueeze(0).repeat(R, 1)  # shape (R, N)
    t_vals += torch.rand_like(t_vals) * (far - near) / num_samples

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)  # shape (R, N, 3)
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # shape (R, N-1)
    deltas = deltas.unsqueeze(-1)  # shape (R, N-1, 1)
    deltas = torch.cat([deltas, torch.full((R, 1, 1), 1e10, device=rays_o.device)], dim=1)  # shape (R, N, 1)

    return pts, deltas


def alpha_composite(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    deltas: torch.Tensor,
    background_color: Optional[Union[torch.Tensor, tuple[float, float, float]]] = None,
    normalize_weights: bool = False
) -> torch.Tensor:
    """
    Perform alpha compositing (volume rendering) given densities and colors.

    Args:
        sigma: [R, N, 1] density predictions at sample points
        rgb:   [R, N, 3] color predictions at sample points
        deltas: [R, N, 1] distance between adjacent samples
        background_color: optional RGB tuple or tensor for the background
        normalize_weights: whether to normalize weights to sum to 1

    Returns:
        rgb_map: [R, 3] rendered pixel colors
    """
    # Assertions
    assert sigma.dim() == 3 and sigma.shape[2] == 1, "sigma should be [R, N, 1]"
    assert rgb.dim() == 3 and rgb.shape[2] == 3, "rgb should be [R, N, 3]"
    assert deltas.dim() == 3 and deltas.shape[2] == 1, "deltas should be [R, N, 1]"
    assert sigma.shape[0] == rgb.shape[0] == deltas.shape[0], "sigma, rgb, and deltas must have the same number of rays"

    R, N, _ = sigma.shape  # N samples, R rays

    alpha = 1.0 - torch.exp(-sigma * deltas)  # shape (R, N, 1)
    p = (1.0 - alpha).squeeze(-1) # shape (R, N)
    first_col = torch.ones(R, 1, device=p.device)
    survival_before = torch.cat([first_col, p[:, :-1]], dim=1)

    T = torch.cumprod(survival_before, dim=1)  # shape (R, N)
    T = T.unsqueeze(-1)  # shape (R, N, 1)
    weights = alpha * T  # shape (R, N, 1)

    if normalize_weights:
        weight_sum = weights.sum(dim=1, keepdim=True) + 1e-10
        weights = weights / weight_sum
    rgb_map = torch.sum(weights * rgb, dim=1)  # shape (R, 3)

    # add background if needed
    if background_color is not None:
        # prepare bg color
        if not torch.is_tensor(background_color):
            bg = torch.tensor(background_color, device=rgb_map.device).view(1, 3)
        else:
            bg = background_color.to(rgb_map.device).view(1, 3)
        # compute residual opacity per ray (shape [R, 1])
        residual = 1.0 - weights.sum(dim=1).squeeze(-1)   # now [R,]
        residual = residual.view(R, 1)                   # now [R,1]
        # blend in background
        rgb_map = rgb_map + residual * bg                # both [R,3] after broadcasting
    
    return rgb_map

# put this near the bottom, after alpha_composite()
def render_chunks(
    rays_o, rays_d, model,
    num_samples, near, far,
    chunk_size,
    background_color=None,
    normalize_weights=False,
):
    """
    Generator that yields (rgb_chunk, gt_slice_indices) for each chunk.
    It keeps only the current chunk’s graph in memory.
    """
    pts, deltas = stratified_sampling(rays_o, rays_d, num_samples, near, far)
    R, N, _ = pts.shape

    pts_flat  = pts.reshape(-1, 3)
    dirs_flat = rays_d.unsqueeze(1).expand(R, N, 3).reshape(-1, 3)

    for start in range(0, pts_flat.shape[0], chunk_size):
        end = min(start + chunk_size, pts_flat.shape[0])
        pts_chunk  = pts_flat[start:end].to(model.device)
        dirs_chunk = dirs_flat[start:end].to(model.device)

        sigma, rgb = model(pts_chunk, dirs_chunk)

        # reshape and composite this chunk only
        num_pts = sigma.shape[0]                 # actual points in this chunk
        rays    = num_pts // num_samples         # rays in this chunk
     
        sigma = sigma.view(rays, num_samples, 1)
        rgb   = rgb.view(rays, num_samples, 3)
     
        start_ray = start // num_samples
        end_ray   = start_ray + rays
        delt = deltas[start_ray:end_ray].to(model.device)

        rgb_chunk = alpha_composite(
            sigma, rgb, delt,
            background_color=background_color,
            normalize_weights=normalize_weights,
        )  # [rays, 3]

        yield rgb_chunk, slice(start // num_samples, end // num_samples)


def volume_render(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    model: torch.nn.Module,
    num_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    chunk_size: int = 1<<18,
    background_color: Optional[Union[torch.Tensor, tuple[float, float, float]]] = None,
    normalize_weights: bool = False
) -> torch.Tensor:
    """
    High-level volume rendering.

    Args:
        rays_o: [R, 3] ray origins
        rays_d: [R, 3] ray directions
        model: TinyNeRF model that returns (sigma, rgb) for inputs
        num_samples: samples per ray
        near, far: near/far plane distances
        background_color: optional RGB tuple or tensor for the background
        normalize_weights: whether to normalize weights to sum to 1

    Returns:
        rgb_map: [R, 3] rendered pixel colors for each ray
    """
    # 1) Stratified sampling
    pts, deltas = stratified_sampling(rays_o, rays_d, num_samples, near, far)
    R, N, _ = pts.shape

    # 2) Flatten pts to [R*num_samples, 3], dirs to same shape
    points_flat = pts.reshape(-1, 3)  # [R*num_samples, 3]
    directions_flat = rays_d.unsqueeze(1).expand(R, num_samples, 3).contiguous().view(-1, 3)  # [R*num_samples, 3]

    sigma_chunks = []
    rgb_chunks = []

    for i in range(0, points_flat.shape[0], chunk_size):
        # 3) Model prediction: sigma, rgb
        chunk_pts = points_flat[i:i+chunk_size].to(model.device)
        chunk_dirs = directions_flat[i:i+chunk_size].to(model.device)
        sigma_chunk, rgb_chunk = model(chunk_pts, chunk_dirs)

        sigma_chunks.append(sigma_chunk)
        rgb_chunks.append(rgb_chunk)
        del chunk_pts, chunk_dirs, sigma_chunk, rgb_chunk
        torch.cuda.empty_cache()          # frees cached blocks

    sigma_flat = torch.cat(sigma_chunks, dim=0)  # [R*num_samples, 1]
    rgb_flat = torch.cat(rgb_chunks, dim=0)      # [R*num_samples, 3]

    # 4) Reshape predictions and call alpha_composite
    sigma = sigma_flat.reshape(R, num_samples, 1)  # [R, num_samples, 1]
    rgb = rgb_flat.reshape(R, num_samples, 3)      # [R, num_samples, 3]
    
    rgb_map = alpha_composite(sigma, rgb, deltas, background_color=background_color, normalize_weights=normalize_weights)  # [R, 3]

    return rgb_map



if __name__ == "__main__":
    rays_o = torch.zeros(32, 3)
    rays_d = torch.tensor([[0, 0, 1.]]).repeat(32, 1)
    from models.nerf import TinyNeRF
    model = TinyNeRF()  # on CPU or GPU
    rgb_map = volume_render(rays_o, rays_d, model, num_samples=16, chunk_size=128)
    assert rgb_map.shape == (32, 3)
    print("Volume rendering test passed. RGB map shape:", rgb_map.shape)
    print("RGB map:", rgb_map)