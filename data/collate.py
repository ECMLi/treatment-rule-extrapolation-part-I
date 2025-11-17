# data/collate.py

from typing import List, Tuple
import torch


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    Collate function for variable-length clusters.

    batch: list of (feat, tau)
        feat: [n_i, 2]
        tau:  [n_i]

    Returns:
        X: [B, Nmax, 2]    padded features
        M: [B, Nmax]       mask (1 for valid, 0 for padded)
        Y: [B, Nmax]       padded targets
    """
    feats, taus = zip(*batch)
    B = len(feats)

    lengths = [f.size(0) for f in feats]
    Nmax = max(lengths)

    dtype_feat = feats[0].dtype
    dtype_tau = taus[0].dtype

    X = torch.zeros((B, Nmax, 2), dtype=dtype_feat)
    M = torch.zeros((B, Nmax),     dtype=dtype_feat)  # float mask (0/1)
    Y = torch.zeros((B, Nmax),     dtype=dtype_tau)

    for i, (f, t) in enumerate(zip(feats, taus)):
        n = f.size(0)
        X[i, :n] = f
        M[i, :n] = 1.0
        Y[i, :n] = t

    return X, M, Y
