from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

best_bw = 0.2782559402207124  # unused here, just keeping for reference


def group_stats(g: pd.DataFrame) -> pd.Series:
    """
    Compute weighted mean and its SE for one (site, T) group.

    Uses normalised weights w_norm = r / sum(r) and:
        mu = sum w_norm * y
        Var(mu) ≈ sum w_norm^2 (y - mu)^2
    """
    y = g['y'].to_numpy()
    r = g['r'].to_numpy()
    n = len(y)
    if n == 0:
        return pd.Series({'mu': np.nan, 'se': np.nan, 'n': 0})

    w_sum = r.sum()
    if w_sum <= 0:
        return pd.Series({'mu': np.nan, 'se': np.nan, 'n': n})

    w_norm = r / w_sum
    mu = np.sum(w_norm * y)
    var_hat = np.sum(w_norm**2 * (y - mu)**2)
    se = np.sqrt(var_hat)
    return pd.Series({'mu': mu, 'se': se, 'n': n})
