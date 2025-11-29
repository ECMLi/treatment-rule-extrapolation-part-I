import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from preprocess.group_statistics import group_stats

def density_rebalance(df: pd.DataFrame):
    
    # twostudy_df must have: 'site', 'T', 'y', 'exp', 'size'
    sites = df['site'].unique()
    site_to_idx = {s: i for i, s in enumerate(sites)}
    idx_to_site = {i: s for s, i in site_to_idx.items()}

    X = df[['exp', 'size']].to_numpy(dtype=float)
    y_site_idx = df['site'].map(site_to_idx).to_numpy()

    # 1 multinomial logistic: P(site | X)
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000
    ).fit(X, y_site_idx)

    # P[i, k] = P(site = sites[k] | X_i)
    P = clf.predict_proba(X)      # shape (N, S)
    N, S = P.shape

    # class priors π_s
    pi = np.bincount(y_site_idx, minlength=S) / N      # shape (S,)
    pi_not = 1.0 - pi
    P_not = 1.0 - P
    ratio_pi = pi_not / pi                             # shape (S,)

    # density ratio r_t(x) = f_t(x) / f_{-t}(x) for ALL targets t
    r_matrix = (P / P_not) * ratio_pi                  # shape (N, S)

    # optional clipping to avoid crazy weights
    r_matrix = np.clip(r_matrix, a_min=0, 
                   a_max=np.quantile(r_matrix, 0.9999))
    
    return r_matrix



