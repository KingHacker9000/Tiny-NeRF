# models/nerf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .posenc import PositionalEncoding

class TinyNeRF(nn.Module):
    """
    TinyNeRF model with positional encoding and MLP architecture.

    Args:
        body_depth: Number of MLP layers in body.
        color_head_depth: Number of MLP layers in color head.
        width: Width of each MLP layer.
        pos_freqs: Number of frequency bands for positional encoding.
        dir_freqs: Number of frequency bands for direction encoding.

    Attributes:
        pos_enc: Positional encoding for 3D points.
        dir_enc: Positional encoding for view directions.
        body: MLP body that outputs density and feature vector.
        color_head: MLP head that outputs RGB color from feature vector and direction encoding.
    """
    pos_enc: PositionalEncoding
    dir_enc: PositionalEncoding
    body: nn.ModuleList
    color_head: nn.Sequential

    def __init__( self, body_depth: int = 6, color_head_depth: int = 4, width: int = 128, pos_freqs: int = 10, dir_freqs: int = 4, skip_layer: int = 3) -> None:

        # Assertions
        assert body_depth >= 2, f"body_depth {body_depth} must be >= 2"
        assert color_head_depth >= 2, f"color_head_depth {color_head_depth} must be >= 2"
        assert width > 0, f"width {width} must be > 0"
        assert skip_layer < body_depth, f"skip_layer {skip_layer} must be < body_depth {body_depth}"

        super().__init__()

        # Store Parameters
        self.width = width
        self.skip_layer = skip_layer
        self.body_depth = body_depth
        self.color_head_depth = color_head_depth

        # 1) Positional encoders for positions & directions
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)

        # 2) MLP body: depth layers, each of size width
        #    – Input size = pos_enc.out_dim (+ optionally skip connections)
        #    – Output of last layer splits into sigma and feature vector
        #    – Color head consumes feature vector + dir_enc.out_dim
        self.body = nn.ModuleList([
            nn.Linear(self.pos_enc.out_dim, width),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(width + (self.pos_enc.out_dim if i == skip_layer else 0), width), nn.ReLU()) for i in range(body_depth - 2)],
            nn.Linear(width, width + 1)
        ])

        self.color_head = nn.Sequential(
            nn.Linear(width + self.dir_enc.out_dim, width),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(color_head_depth - 2)],
            nn.Linear(width, 3),
            nn.Sigmoid()  # Output RGB color
        )



    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [R, 3] 3D sample points
            d: [R, 3] view directions
        Returns:
            sigma: [R, 1] densities
            rgb:   [R, 3] colors in [0,1]
        """
        assert x.ndim == 2 and x.shape[-1] == 3, "Input x must have shape [R, 3]"
        assert d.ndim == 2 and d.shape[-1] == 3, "Input d must have shape [R, 3]"

        # Encode positions
        x_enc = self.pos_enc(x)  # [R, pos_enc.out_dim]
        # Encode directions
        d_enc = self.dir_enc(d)  # [R, dir_enc.out_dim]

        #################### MLP body ####################
        h = x_enc
        h = self.body[0](h)  # First layer: [R, width]
        h = self.body[1](h)  # Apply ReLU activation
        for i, block in enumerate(self.body[2:-1]):
            if i == self.skip_layer:
                h = torch.cat([h, x_enc], dim=-1)
            h = block(h)

        out = self.body[-1](h)
        # Split output into sigma and feature vector
        sigma = out[..., :1]  # [R, 1]
        sigma = F.softplus(sigma) # Apply Softplus to sigma for positive density

        ################### Color Head ####################
        # Feature vector is the rest of the output
        feature_vector = out[..., 1:]

        # Concatenate feature vector with direction encoding
        color_input = torch.cat([feature_vector, d_enc], dim=-1)  # [R, feature_vector_dim + dir_enc.out_dim]

        # Pass through color head to get RGB color
        rgb = self.color_head(color_input)  # [R, 3]

        return sigma, rgb  # Return densities and colors


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(body_depth={self.body_depth}, "
            f"color_head_depth={self.color_head_depth}, width={self.width}, "
            f"skip_layer={self.skip_layer})"
            f"pos_enc_out_dim={self.pos_enc.out_dim}, dir_enc_out_dim={self.dir_enc.out_dim}), "
        )
    
if __name__ == "__main__":
    # Example usage
    model = TinyNeRF(body_depth=6, color_head_depth=4, width=128, pos_freqs=10, dir_freqs=4, skip_layer=3)
    print(model)

    # Dummy inputs
    x = torch.randn(100, 3)  # 100 random 3D points
    d = torch.randn(100, 3)  # 100 random view directions

    sigma, rgb = model(x, d)
    print("Sigma shape:", sigma.shape)
    print("RGB shape:", rgb.shape)