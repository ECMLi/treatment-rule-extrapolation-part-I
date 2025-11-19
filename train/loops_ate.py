# train/loops_ate.py

import torch
import numpy as np


def train_one_epoch(model, loader, optimizer, device):
    """
    One training epoch for site-level DeepSet.
    Inputs:
        model: DeepSetSiteConstant
        loader: DataLoader yielding (X, MU0, MASK, Y)
        optimizer: torch optimizer
        device: torch.device
    Output:
        average MSE over the epoch
    """
    model.train()
    total_loss = 0.0
    total_sites = 0

    for X, MU0, MASK, Y in loader:
        X = X.to(device)
        MU0 = MU0.to(device)
        MASK = MASK.to(device)
        Y = Y.to(device)  # [B]

        optimizer.zero_grad()
        preds = model(X, MU0, MASK)  # [B]
        loss = ((preds - Y) ** 2).mean()
        loss.backward()
        optimizer.step()

        batch_size = Y.shape[0]
        total_loss += loss.item() * batch_size
        total_sites += batch_size

    return total_loss / total_sites



@torch.no_grad()
def eval_one_epoch(model, loader, device):
    """
    Evaluate on validation set.
    Returns:
        mse, corr
    """
    model.eval()
    total_loss = 0.0
    total_sites = 0
    all_preds, all_trues = [], []

    for X, MU0, MASK, Y in loader:
        X = X.to(device)
        MU0 = MU0.to(device)
        MASK = MASK.to(device)
        Y = Y.to(device)

        preds = model(X, MU0, MASK)

        batch_size = Y.shape[0]
        loss = ((preds - Y) ** 2).mean()

        total_loss += loss.item() * batch_size
        total_sites += batch_size

        all_preds.append(preds.cpu().numpy())
        all_trues.append(Y.cpu().numpy())

    avg_loss = total_loss / total_sites

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    if all_preds.size > 1:
        corr = np.corrcoef(all_preds, all_trues)[0, 1]
    else:
        corr = np.nan

    return avg_loss, corr
