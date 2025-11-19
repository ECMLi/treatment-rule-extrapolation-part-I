# data/cluster_cate_dataset.py
# data/cluster_ate_dataset.py

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class ClusterATEDataset(Dataset):
    """
    Dataset where each item corresponds to one site/cluster, with keys:
      - "x":   array [n_i, 2]
      - "mu0": array [n_i]
      - "ate": scalar
    """

    def __init__(self, clusters: List[Dict[str, Any]]):
        """
        clusters: list of dicts produced by feature engineering.
        Each dict must contain keys: "x", "mu0", and "ate".
        """
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, i: int):
        c = self.clusters[i]

        # x: [n_i, 2]
        x = torch.tensor(c["x"], dtype=torch.float32)

        # mu0: [n_i]
        mu0 = torch.tensor(c["mu0"], dtype=torch.float32)

        # site-level target scalar
        y_site = torch.tensor(float(c["ate"]), dtype=torch.float32)

        return x, mu0, y_site
