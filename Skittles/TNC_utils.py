# define a class to implement the TNC-cost analysis from Muller & Sternad, 2009, EBR
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import time

class TNCCost:
    def __init__(self, env, n_grid=60, n_collapse=100, n_samples=60):
        """
        env: Skittles environment with method `compute_error(action)`
        n_grid: resolution of grid search for T-Cost
        n_collapse: steps for collapsing dataset toward mean for N-Cost
        n_samples: trials per block (default = 60, like in paper)
        """
        self.env = env

    def mean_error(self, actions):
        errors = []
        for action in actions:
            error, _ = self.env.compute_error(action)  # keep only the error
            errors.append(error)
        return np.mean(errors)

    # -------------------------------
    # T-Cost: shift distribution to most tolerant region
    # -------------------------------
    def t_cost(self, actions):
        actual_err = self.mean_error(actions)

        # Objective: shift dataset by delta, then compute mean error
        def objective(delta):
            shifted = actions + delta
            return self.mean_error(shifted)

        # Start at no shift
        x0 = np.array([0.0, 0.0])

        # Optimize (you can set bounds if you want to limit angle/vel range)
        res = minimize(objective, x0, method="Nelder-Mead")

        best_err = res.fun
        return actual_err - best_err

    # -------------------------------
    # N-Cost: collapse variability radially toward mean
    # -------------------------------
    def n_cost(self, actions):
        mu = np.mean(actions, axis=0)
        actual_err = self.mean_error(actions)

        # Objective: collapse factor alpha ∈ [0, 1]
        # alpha = 1 → original dataset
        # alpha = 0 → all points at mean
        def objective(alpha):
            shrunk_actions = mu + alpha * (actions - mu)
            return self.mean_error(shrunk_actions)

        res = minimize_scalar(objective, bounds=(0,5), method="bounded")

        best_err = res.fun
        return actual_err - best_err

    # -------------------------------
    # C-Cost: optimize covariation via greedy pair-swapping
    # -------------------------------
    def c_cost(self, actions):
        n = len(actions)
        angles = actions[:,0]
        vels   = actions[:,1]

        # Precompute all possible errors
        error_matrix = np.zeros((n,n))
        for i, a in enumerate(angles):
            for j, v in enumerate(vels):
                error_matrix[i,j] = self.env.compute_error([a,v])[0]

        # Initial pairing = identity (angle_i with vel_i)
        pairing = np.arange(n)
        current_err = np.mean([error_matrix[i, pairing[i]] for i in range(n)])
    
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    # Try swapping velocities (pairing indices)
                    new_pairing = pairing.copy()
                    new_pairing[i], new_pairing[j] = pairing[j], pairing[i]

                    new_err = np.mean([error_matrix[k, new_pairing[k]] for k in range(n)])
                    if new_err < current_err:
                        pairing = new_pairing
                        current_err = new_err
                        improved = True

        best_err = current_err
        actual_err = np.mean([error_matrix[i,i] for i in range(n)])  # original pairs
        return actual_err - best_err

    # -------------------------------
    # Wrapper: compute all three costs
    # -------------------------------
    def compute_all(self, actions):
        t_cost_val = self.t_cost(actions)
        n_cost_val = self.n_cost(actions)
        c_cost_val = self.c_cost(actions)
        return {
            "T-Cost": t_cost_val,
            "N-Cost": n_cost_val,
            "C-Cost": c_cost_val,
            "MeanError": self.mean_error(actions)
        }
