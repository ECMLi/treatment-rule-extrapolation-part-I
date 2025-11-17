# data/cluster_cate_dataset.py

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class ClusterCATEDataset(Dataset):
    """
    Dataset where each item is a cluster with keys:
      - "x":   1D array-like, shape [n_i]
      - "mu0": 1D array-like, shape [n_i]
      - "tau": 1D array-like, shape [n_i]

    We build a feature matrix [n_i, 2] by concatenating x and mu0.
    """

    def __init__(self, clusters: List[Dict[str, Any]]):
        """
        clusters: list of dicts, each with keys "x", "mu0", "tau"
        """
        self.clusters = clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, i: int):
        c = self.clusters[i]

        # [n_i] -> [n_i, 1]
        x = torch.tensor(c["x"], dtype=torch.float32).unsqueeze(1)
        mu0 = torch.tensor(c["mu0"], dtype=torch.float32).unsqueeze(1)

        # Features [n_i, 2]
        feat = torch.cat([x, mu0], dim=1)

        # Targets [n_i]
        tau = torch.tensor(c["tau"], dtype=torch.float32)

        return feat, tau
