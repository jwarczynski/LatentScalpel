"""Top-K Sparse Autoencoder model definition."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TopKSAE(nn.Module):
    """Sparse Autoencoder with Top-K activation function.

    The encoder projects activations to a higher-dimensional dictionary space,
    selects the top K activations (setting the rest to zero), and the decoder
    reconstructs back to the original dimension.

    Args:
        activation_dim: Dimensionality of the input activations.
        dictionary_size: Size of the sparse dictionary (typically > activation_dim).
        k: Number of top activations to keep in the sparse code.
    """

    def __init__(self, activation_dim: int, dictionary_size: int, k: int) -> None:
        super().__init__()
        self.activation_dim = activation_dim
        self.dictionary_size = dictionary_size
        self.k = k
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)

    def encode(self, x: Tensor) -> Tensor:
        """Encode input and apply top-k sparsity.

        Args:
            x: Input tensor of shape (..., activation_dim).

        Returns:
            Sparse tensor of shape (..., dictionary_size) with exactly k
            non-zero entries along the last dimension.
        """
        z = self.encoder(x)
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk_indices, topk_values)
        return sparse_z

    def decode(self, z: Tensor) -> Tensor:
        """Decode sparse representation back to activation space.

        Args:
            z: Sparse tensor of shape (..., dictionary_size).

        Returns:
            Reconstructed tensor of shape (..., activation_dim).
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass: encode with top-k sparsity, then decode.

        Args:
            x: Input tensor of shape (..., activation_dim).

        Returns:
            Tuple of (reconstruction, sparse_code) where reconstruction has
            shape (..., activation_dim) and sparse_code has shape
            (..., dictionary_size).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
