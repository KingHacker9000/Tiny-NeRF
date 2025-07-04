"""
models/posenc.py

Positional encoding for 3D points and view directions, following NeRF's sinusoidal embedding.
"""
import torch
import torch.nn as nn
from typing import Callable, Union


def _get_positional_encoder(num_freqs: int, include_input: bool = True, log_sampling: bool = True) -> tuple[Union[nn.Module , callable], int]:
    """
    Returns a function that maps (N, 3) -> (N, 3 * (include_input + num_freqs*2)).
    where 3 is the input dimension.
    
    Args:
        num_freqs: Number of frequency bands (L).
        include_input: If True, include the original input in the embedding.
        log_sampling: If True, frequencies are 2^0, 2^1, ..., 2^(L-1); else linear spacing.

    Returns:
        encode_fn: A function or nn.Module that applies positional encoding.
        out_dim: Output dimension of the encoding.
    """
    
    # Assertions
    assert isinstance(num_freqs, int) and num_freqs > 0, "num_freqs must be a positive integer."
    assert isinstance(include_input, bool), "include_input must be a boolean."
    assert isinstance(log_sampling, bool), "log_sampling must be a boolean."

    # Calculate output dimension
    input_dim = 3  # Input is expected to be of shape [..., 3]
    out_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * (2 * num_freqs)

    # Define the positional encoding function
    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 3] input tensor (e.g., points or directions)
        Returns:
            encoded: [..., D * (include_input + num_freqs*2)] positional encoding
        """
        # Assert input shape
        assert x.ndim >= 2 and x.shape[-1] == 3, "Input must have shape [..., 3]."

        # Compute frequencies
        if log_sampling:
            freqs = 2 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device)    # Frequencies: 2^0, 2^1, ..., 2^(num_freqs-1)
        else:
            freqs = torch.arange(1, num_freqs+1, dtype=x.dtype, device=x.device)    # Frequencies: 1, 2, ..., num_freqs

        # Compute sin and cos for each frequency
        encoded = [x] if include_input else []
        for freq in freqs:
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))

        return torch.cat(encoded, dim=-1)
    
    # Return the encoding function and output dimension
    return encode_fn, out_dim


class PositionalEncoding(nn.Module):
    """
    PyTorch module wrapping positional encoding for ease of use in models.

    Args:
        num_freqs: Number of frequency bands (L).
        include_input: Include original input in encoding.
        log_sampling: Use logarithmic frequency spacing.
    """
    def __init__(self, num_freqs: int, include_input: bool = True, log_sampling: bool = True) -> None:
        super().__init__()
        # store parameters
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        self.encode_fn, self.out_dim = _get_positional_encoder(num_freqs, include_input, log_sampling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D] input tensor (e.g., [..., 3] for points or directions)
        Returns:
            encoded: [..., D * (include_input + num_freqs*2)] positional encoding
        """
        # Assert input shape
        assert x.ndim >= 2 and x.shape[-1] == 3, "Input must have shape [..., 3]."
        
        # Apply positional encoding
        return self.encode_fn(x)
    
    def __repr__(self) -> str:
        """
        String representation of the PositionalEncoding module.
        """
        return (f"PositionalEncoding(num_freqs={self.num_freqs}, "
                f"include_input={self.include_input}, log_sampling={self.log_sampling})")


if __name__ == "__main__":
    # Example usage:
    # Create a positional encoding module
    pe = PositionalEncoding(num_freqs=10, include_input=True, log_sampling=True)
    
    # Example input tensor of shape (N, 3)
    x = torch.randn(5, 3)  # 5 random points in 3D space
    
    # Apply positional encoding
    x_encoded = pe(x)  # x: (5,3) -> x_encoded: (5,3*(1+2*10))
    
    print("Input shape:", x.shape)
    print("Encoded shape:", x_encoded.shape)
