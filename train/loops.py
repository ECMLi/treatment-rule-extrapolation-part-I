# train/loops.py

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    One training epoch over all clusters in loader.
    Returns MSE over all non-padded points.
    """
    model.train()
    total_se, total_pts = 0.0, 0.0

    for X, M, Y in loader:
        X = X.to(device)
        M = M.to(device)
        Y = Y.to(device)

        tau_hat = model(X, M)

        # Squared error only on valid locations
        se = ((tau_hat - Y) ** 2 * M).sum()

        optimizer.zero_grad()
        se.backward()
        optimizer.step()

        total_se += se.item()
        total_pts += M.sum().item()

    return total_se / total_pts


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate model over loader.
    Returns:
        mse, mae, corr   computed over all non-padded points.
    """
    model.eval()
    total_se, total_abs, total_pts = 0.0, 0.0, 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X, M, Y in loader:
            X = X.to(device)
            M = M.to(device)
            Y = Y.to(device)

            tau_hat = model(X, M)

            total_se  += ((tau_hat - Y) ** 2 * M).sum().item()
            total_abs += ((tau_hat - Y).abs() * M).sum().item()
            total_pts += M.sum().item()

            # Collect valid entries only
            mask_flat = (M == 1).cpu().numpy().astype(bool)
            pred_flat = tau_hat.cpu().numpy().flatten()
            true_flat = Y.cpu().numpy().flatten()

            all_preds.append(pred_flat[mask_flat.flatten()])
            all_trues.append(true_flat[mask_flat.flatten()])

    mse = total_se / total_pts
    mae = total_abs / total_pts

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    if all_preds.size > 1:
        corr = np.corrcoef(all_preds, all_trues)[0, 1]
    else:
        corr = np.nan

    return mse, mae, corr


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Run the model on loader and return two 1-D numpy arrays:
      preds = all predicted tau_hat (flattened over all clusters & positions)
      trues = the corresponding ground-truth tau (same order)
    """
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X, M, Y in loader:
            X = X.to(device)
            M = M.to(device)
            Y = Y.to(device)

            tau_hat = model(X, M)

            mask_flat = (M == 1).cpu().numpy().astype(bool)
            pred_flat = tau_hat.cpu().numpy().flatten()
            true_flat = Y.cpu().numpy().flatten()

            all_preds.append(pred_flat[mask_flat.flatten()])
            all_trues.append(true_flat[mask_flat.flatten()])

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues
