import cvxpy as cp
import numpy as np
from scipy.stats import chi2

def solve_robust(mu_hat, s_squared, c, confidence_level=0.95, d=1):
    """
    Solves the robust maximization problem for Action d={0,1} (Don't Treat).
    
    Objective: max t
    Subject to: 
       1. t >= 0
       2. t <= mu_k + c_k (for all k) for d=0, t <= c_k - mu_k for d=1
       3. mu is within the 95% confidence ellipsoid defined by mu_hat and s_squared
    """
    
    # 1. Setup Data Dimensions
    K = len(mu_hat)
    
    # 2. Calculate the Ellipsoid Radius (Chi-Squared Critical Value)
    # The constraint is (mu - mu_hat)^T Sigma^-1 (mu - mu_hat) <= chi^2_val
    chi2_val = chi2.ppf(confidence_level, df=K)
    
    # Construct the Inverse Covariance Matrix (assuming independence between studies)
    # If studies are independent, Sigma is diagonal with entries s_squared.
    # Sigma_inv is diagonal with entries 1/s_squared.
    Sigma_inv = np.diag(1.0 / s_squared)
    
    # 3. Define Decision Variables
    # mu: The vector of "true" parameters inside the uncertainty set
    mu = cp.Variable(K) 
    # t: The auxiliary scalar variable we want to maximize
    t = cp.Variable(1)
    
    # 4. Define Constraints
    constraints = []
    
    # Constraint A: The Hinge/Epigraph constraints (Linear)
    # t >= 0
    constraints.append(t >= 0)
    
    # t <= mu_k + c_k for all k
    # CVXPY handles vector addition element-wise automatically
    if d == 0:
        constraints.append(t <= mu + c)
    elif d == 1:
        constraints.append(t <= c - mu)
    else:
        print("'d' must take value 0 or 1.")
    
    # Constraint B: The Uncertainty Set (Quadratic/Conic)
    # This defines the ellipsoid: (mu - mu_hat)^T * Sigma_inv * (mu - mu_hat) <= rho^2
    constraints.append(cp.quad_form(mu - mu_hat, Sigma_inv) <= chi2_val)
    
    # 5. Define Objective: Maximize t
    objective = cp.Maximize(t)
    
    # 6. Solve
    problem = cp.Problem(objective, constraints)
    
    # Using a robust solver like ECOS or SCS (included with CVXPY)
    try:
        result = problem.solve()
    except cp.SolverError:
        result = problem.solve(solver=cp.SCS)

    return {
        "robust_risk_value": t.value[0],
        "worst_case_mu": mu.value,
        "status": problem.status
    }
