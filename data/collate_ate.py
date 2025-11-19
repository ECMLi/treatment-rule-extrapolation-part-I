# data/collate.py

from typing import List, Tuple
import torch

from typing import List, Tuple
import torch


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Collate function for variable-length clusters for site-level prediction.

    batch: list of (x, mu0, y_site)
        x:      [n_i, 2]
        mu0:    [n_i]
        y_site: scalar ([]) or [1]

    Returns:
        X:    [B, Nmax, 2]   padded features (size, exp)
        MU0:  [B, Nmax]      padded mu0
        MASK: [B, Nmax]      mask (1 for valid, 0 for padded)
        Y:    [B]            site-level targets
    """
    # Unzip the batch
    X_list, MU0_list, Y_list = zip(*batch)

    B = len(X_list)
    lengths = [x.shape[0] for x in X_list]
    Nmax = max(lengths)

    # Allocate padded tensors
    dtype_x = X_list[0].dtype
    dtype_mu0 = MU0_list[0].dtype

    X = torch.zeros((B, Nmax, 2), dtype=dtype_x)
    MU0 = torch.zeros((B, Nmax), dtype=dtype_mu0)
    MASK = torch.zeros((B, Nmax), dtype=torch.float32)

    # Pad each site
    for i, (x, mu0) in enumerate(zip(X_list, MU0_list)):
        n = x.shape[0]
        X[i, :n, :] = x
        MU0[i, :n] = mu0
        MASK[i, :n] = 1.0

    # Stack site-level targets into a vector [B]
    Y = torch.stack(Y_list, dim=0).float()  # [B]

    return X, MU0, MASK, Y
