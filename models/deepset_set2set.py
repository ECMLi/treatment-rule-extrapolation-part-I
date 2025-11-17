# models/deepset_set2set.py

import torch
import torch.nn as nn


class DeepSetSet2Set(nn.Module):
    """
    Set-to-set DeepSet model that predicts tau(x) at each element
    in a cluster, using both local and global information.

    Input:
        x:    [B, N, in_dim]   (features; here in_dim=2)
        mask: [B, N]           (1 for valid, 0 for padded)

    Output:
        tau_hat: [B, N]        (predictions, padded positions zeroed out)
    """

    def __init__(self, in_dim: int = 2, hid: int = 64):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, N, in_dim]
        mask: [B, N] (0/1 floats or bools)

        Returns:
            [B, N] predictions, with padded positions masked to 0.
        """
        B, N, _ = x.shape

        # Element-wise embedding
        h = self.phi(x)  # [B, N, H]

        # Zero-out padded positions
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        h = h * mask_expanded

        # Global summary per cluster
        s = h.sum(dim=1)             # [B, H]
        S = s.unsqueeze(1).expand(B, N, -1)  # [B, N, H]

        # Concatenate local + global
        Hcat = torch.cat([h, S], dim=-1)     # [B, N, 2H]

        tau_hat = self.rho(Hcat).squeeze(-1)  # [B, N]

        # Mask out padding in the output
        return tau_hat * mask
