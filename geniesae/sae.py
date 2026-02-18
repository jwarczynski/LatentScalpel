"""Top-K Sparse Autoencoder model definition.

Implements the Top-K SAE with:
- Explicit W_enc, W_dec, b_enc, b_dec parameters
- Decoder weight normalization (unit L2 ball projection)
- K-annealing support via mutable _current_k
- Data-driven initialization (b_dec = mean, W_enc = W_dec^T)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TopKSAE(nn.Module):
    """Sparse Autoencoder with Top-K activation function.

    Forward pass:
        f = ReLU(W_enc @ (x - b_dec) + b_enc)
        z = TopK(f, current_k)
        x_hat = W_dec^T @ z + b_dec

    Args:
        activation_dim: Dimensionality of the input activations.
        dictionary_size: Size of the sparse dictionary (typically activation_dim * expansion_factor).
        k: Number of top activations to keep in the sparse code.
    """

    def __init__(self, activation_dim: int, dictionary_size: int, k: int) -> None:
        super().__init__()
        self.activation_dim = activation_dim
        self.dictionary_size = dictionary_size
        self.k = k
        self._current_k = k  # Mutable, updated by k-annealing

        # Encoder: W_enc @ (x - b_dec) + b_enc
        self.W_enc = nn.Parameter(torch.empty(dictionary_size, activation_dim))
        self.b_enc = nn.Parameter(torch.zeros(dictionary_size))

        # Decoder: W_dec @ z + b_dec  (W_dec shape: activation_dim x dictionary_size)
        self.W_dec = nn.Parameter(torch.empty(activation_dim, dictionary_size))
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        # Default initialization
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def initialize_from_data(self, data_mean: Tensor) -> None:
        """Initialize b_dec to data mean, W_enc = W_dec^T.

        Args:
            data_mean: Mean of the dataset activations, shape (activation_dim,).
        """
        with torch.no_grad():
            self.b_dec.copy_(data_mean)
            nn.init.kaiming_uniform_(self.W_dec)
            self.normalize_decoder_()
            self.W_enc.copy_(self.W_dec.T)

    def normalize_decoder_(self) -> None:
        """Project W_dec columns onto unit L2 ball (in-place).

        After this call, ||W_dec[:, i]||_2 <= 1.0 for all i.
        """
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.W_dec.div_(norms)

    def set_k(self, k: int) -> None:
        """Update current k (used by k-annealing)."""
        self._current_k = k

    def encode(self, x: Tensor) -> Tensor:
        """Encode input and apply top-k sparsity.

        f = ReLU(W_enc @ (x - b_dec) + b_enc), then TopK.

        Args:
            x: Input tensor of shape (..., activation_dim).

        Returns:
            Sparse tensor of shape (..., dictionary_size) with exactly
            _current_k non-zero entries along the last dimension.
        """
        pre_act = F.relu(
            F.linear(x - self.b_dec, self.W_enc, self.b_enc)
        )
        topk_values, topk_indices = torch.topk(pre_act, self._current_k, dim=-1)
        sparse_z = torch.zeros_like(pre_act)
        sparse_z.scatter_(-1, topk_indices, topk_values)
        return sparse_z

    def decode(self, z: Tensor) -> Tensor:
        """Decode sparse representation back to activation space.

        x_hat = W_dec @ z + b_dec  (W_dec shape: activation_dim x dictionary_size).

        Args:
            z: Sparse tensor of shape (..., dictionary_size).

        Returns:
            Reconstructed tensor of shape (..., activation_dim).
        """
        return F.linear(z, self.W_dec) + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass: encode with top-k sparsity, then decode.

        Args:
            x: Input tensor of shape (..., activation_dim).

        Returns:
            Tuple of (reconstruction, sparse_code).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
