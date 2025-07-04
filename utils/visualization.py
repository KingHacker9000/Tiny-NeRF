# utils/visualization.py

import matplotlib.pyplot as plt
from typing import Sequence
import torch

def plot_loss_curve(
    losses: Sequence[float],
    save_path: str = None,
    loss_type: str = 'MSE'
) -> None:
    """
    Plots training loss over epochs.

    Args:
        losses: list of per-epoch (or per-iteration) loss values
        save_path: if provided, plt.savefig(save_path)
    """
    # plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE Loss')
    # plt.title('Training Loss Curve')
    # if save_path: plt.savefig(...)
    # else: plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type} Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def show_render_comparison(
    rgbs_gt: torch.Tensor,
    rgbs_pred: torch.Tensor,
    save_path: str = None
) -> None:
    """
    Displays ground-truth vs. predicted images side by side.

    Args:
        rgbs_gt: [N,H,W,3] or [N,3,H,W] tensor
        rgbs_pred: same shape as rgbs_gt
    """
    # for each index i:
    #   plt.subplot(n_rows, 2, 2*i+1); plt.imshow(gt); ...
    #   plt.subplot(n_rows, 2, 2*i+2); plt.imshow(pred); ...
    # save or show
    
    assert rgbs_gt.shape == rgbs_pred.shape, "Ground truth and predicted RGBs must have the same shape."
    assert rgbs_gt.ndim in (3, 4), "Input tensors must be 3D or 4D."
    if rgbs_gt.ndim == 3:
        rgbs_gt = rgbs_gt.unsqueeze(0)  # [1, H, W, 3]
        rgbs_pred = rgbs_pred.unsqueeze(0)  # [1, H, W, 3]
    assert rgbs_gt.shape[-1] == 3, "RGB tensors must have 3 channels."

    if rgbs_gt.ndim == 4 and rgbs_gt.shape[1] == 3:
        rgbs_gt = rgbs_gt.permute(0,2,3,1)
        rgbs_pred = rgbs_pred.permute(0,2,3,1)

    n_images = rgbs_gt.shape[0]
    n_cols = 2
    n_rows = n_images

    plt.figure(figsize=(12, 6 * n_rows))
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, 2*i + 1)
        plt.imshow(rgbs_gt[i].cpu().numpy())
        plt.title(f'Ground Truth {i+1}')
        plt.axis('off')

        plt.subplot(n_rows, n_cols, 2*i + 2)
        plt.imshow(rgbs_pred[i].cpu().numpy())
        plt.title(f'Predicted {i+1}')
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":
    # Example usage
    import numpy as np
    # Generate dummy data
    losses = np.random.rand(100).tolist()
    plot_loss_curve(losses)

    rgbs_gt = torch.rand(4, 64, 64, 3)  # 4 images of size 64x64
    rgbs_pred = torch.rand(4, 64, 64, 3)
    show_render_comparison(rgbs_gt, rgbs_pred)