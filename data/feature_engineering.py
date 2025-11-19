# data/feature_engineering.py

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg


def compute_mu0_ate_features(
    df: pd.DataFrame,
    ate_weighted_df: pd.DataFrame,
    ate_y0_weighted: pd.DataFrame,
    ate_y1_weighted: pd.DataFrame,
    best_bw: float,
) -> List[Dict[str, Any]]:
    """
    For each (target t, site s), fit a kernel regression for mu0,
    evaluate it on a grid of (size, exp) x values, and pair with the ATE.

    Returns a list of dictionaries, one per (target, site):
        {
          'target': t,
          'site':   s,
          'x':      x_grid   [n_i, 2] array of (size, exp)
          'mu0':    mu0_hat  [n_i]    array of predicted mu0
          'ate':    ate_val  scalar   ATE for that (t, s)
        }
    """

    mu0_ate = []

    for t in ate_weighted_df["target"].unique():
        # all sites that appear for this target
        sites_t = ate_weighted_df.loc[
            ate_weighted_df["target"] == t, "site"
        ].unique()

        for s in sites_t:
            # outcome / treatment-specific subsets
            y0 = ate_y0_weighted.loc[
                (ate_y0_weighted["site"] == s)
                & (ate_y0_weighted["target"] == t),
                "y0",
            ].to_numpy(dtype=float)

            y1 = ate_y1_weighted.loc[  # currently unused, but kept for symmetry
                (ate_y1_weighted["site"] == s)
                & (ate_y1_weighted["target"] == t),
                "y1",
            ].to_numpy(dtype=float)

            x0 = df.loc[
                (df["site"] == s) & (df["T"] == 0),
                ["size", "exp"],
            ].to_numpy(dtype=float)

            x1 = df.loc[
                (df["site"] == s) & (df["T"] == 1),
                ["size", "exp"],
            ].to_numpy(dtype=float)

            # kernel regression for mu0
            kr0 = KernelReg(
                endog=y0,
                exog=x0,
                var_type="cc",    # both covariates continuous
                reg_type="ll",    # local linear
                bw=[best_bw, best_bw],
            )

            # evaluation grid: all (size, exp) observed at site s
            x_grid = df.loc[
                df["site"] == s,
                ["size", "exp"],
            ].to_numpy(dtype=float)

            # IMPORTANT: you probably want a 2D grid; if x_grid comes out unsorted
            # and you really want sorting, define explicitly what "sorted" means.
            # For now, leave as-is or sort lexicographically:
            # x_grid = x_grid[np.lexsort((x_grid[:, 1], x_grid[:, 0]))]

            mu0_hat, _ = kr0.fit(x_grid)

            # ATE for this (t, s): assume scalar
            ate_series = ate_weighted_df.loc[
                (ate_weighted_df["site"] == s)
                & (ate_weighted_df["target"] == t),
                "ate",
            ]
            # get scalar value
            ate_val = float(ate_series.iloc[0])

            mu0_ate.append(
                {
                    "target": t,
                    "site": s,
                    "x": x_grid,       # [n_i, 2]
                    "mu0": mu0_hat,    # [n_i]
                    "ate": ate_val,    # scalar
                }
            )

    return mu0_ate
