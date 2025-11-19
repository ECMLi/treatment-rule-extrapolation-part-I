import torch
import torch.nn as nn


class DeepSetSiteConstant(nn.Module):
    """
    DeepSet that takes per-unit inputs (x, mu0) and
    outputs ONE constant per site (cluster).

    Inputs:
        x:    [B, N, 2]     (size, exp) or similar 2D features
        mu0:  [B, N]        (scalar per unit)
        mask: [B, N]        (1 for valid units, 0 for padded)

    Output:
        y_hat: [B]          (site-level scalar prediction, e.g. site ATE)
    """

    def __init__(self, hid: int = 64):
        super().__init__()

        in_dim_total = 2 + 1  # 2 dims from x + 1 dim from mu0

        # φ: element-wise network
        self.phi = nn.Sequential(
            nn.Linear(in_dim_total, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )

        # ρ: site-level network (maps aggregated embedding to scalar)
        self.rho = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, 1),
        )

    def forward(
        self,
        x: torch.Tensor,    # [B, N, 2]
        mu0: torch.Tensor,  # [B, N]
        mask: torch.Tensor, # [B, N]
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Concatenate x and mu0 along feature dimension: [B, N, 3]
        mu0_expanded = mu0.unsqueeze(-1)        # [B, N, 1]
        feat = torch.cat([x, mu0_expanded], dim=-1)  # [B, N, 3]

        # Elementwise embedding via φ: [B, N, H]
        h = self.phi(feat)

        # Apply mask (zero out padded positions)
        mask_expanded = mask.unsqueeze(-1)      # [B, N, 1]
        h = h * mask_expanded

        # Aggregate over set dimension → site embedding [B, H]
        # (sum; you could also use mean)
        h_sum = h.sum(dim=1)                    # [B, H]

        # Optionally normalise by number of valid points (mean aggregation)
        counts = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
        h_mean = h_sum / counts                 # [B, H]

        # Site-level prediction via ρ: [B, 1] → [B]
        y_hat = self.rho(h_mean).squeeze(-1)    # [B]

        return y_hat
