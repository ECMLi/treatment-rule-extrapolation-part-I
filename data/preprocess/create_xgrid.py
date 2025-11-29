import pandas as pd
import numpy as np
# from preprocess.group_statistics import group_stats

def create_grid(df: pd.DataFrame, n_exp=50, n_size=50):
    # 1. Get global min/max for both covariates
    exp_min = df['exp'].min()
    exp_max = df['exp'].max()
    size_min = df['size'].min()
    size_max = df['size'].max()

    # 2. Choose resolution of the grid
    n_exp = 50   # number of grid points along exp
    n_size = 50  # number of grid points along size

    # 3. Build 1D grids
    exp_grid_1d = np.linspace(exp_min, exp_max, n_exp)
    size_grid_1d = np.linspace(size_min, size_max, n_size)

    # 4. Build 2D meshgrid
    E, S = np.meshgrid(exp_grid_1d, size_grid_1d, indexing='ij')
    # E.shape = (n_exp, n_size), S.shape = (n_exp, n_size)

    # 5. Flatten into (n_exp * n_size, 2) array for KernelReg
    x_grid_common = np.column_stack([E.ravel(), S.ravel()])
    # x_grid.shape == (n_exp * n_size, 2)
    
    return x_grid_common



































