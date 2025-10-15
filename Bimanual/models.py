#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:21:20 2025

@author: adrianhaith
"""

# define models of the environment and the learner

import numpy as np
from utils import wrap_to_pi

# define the environment class for the cursor control task
class CursorControlEnv:
    def __init__(self, radius=0.12, discrete_targs=True, motor_noise_std=.1, seed=1):
        self.radius = radius # distance of targets from start pos
        self.target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) # possible target angles
        u_range = (-2*radius, 2*radius)
        self.action_space = np.array([u_range, u_range, u_range, u_range])  # define range of allowable actions (not needed here, but consistent with gym-style environments)
        self.observation_space = np.array([0.0, 2*np.pi])  # Target angle index
        self.state = None
        self.target_pos = None
        self.discrete = discrete_targs
        if(motor_noise_std is None):
            self.motor_noise_std = radius*.1 # default at 10% of distance to target
        else:
            self.motor_noise_std = motor_noise_std
        self.rng = np.random.default_rng(seed) # random number generator for motor noise

    # method to reset task environment (selects a new target position)
    def reset(self):
        if(self.discrete):
            idx = np.random.randint(len(self.target_angles))    # select from discrete set of possible targets
            angle = self.target_angles[idx]                 
        else:
            angle = np.random.uniform(0, 2 * np.pi)             # select from a continuous range of targets

        self.state = angle
        self.target_pos = self.radius * np.array([np.cos(angle), np.sin(angle)])
        return angle  # NB - target 'angle' is the state for this environment

    # simulate outcome of executing an action in the environment
    def step(self, action):
        # action = [Lx, Ly, Rx, Ry]
        Lx, Ly, Rx, Ry = action
        cursor_x = Ly  # controlled by L_y
        cursor_y = Rx  # controlled by R_x
        cursor_pos = np.array([cursor_x, cursor_y])
        motor_noise = self.rng.normal([0, 0], [self.motor_noise_std, self.motor_noise_std])
        error = np.linalg.norm(cursor_pos + motor_noise - self.target_pos)
        reward = -np.linalg.norm(error)

        # Compute directional error
        targ_ang = np.atan2(self.target_pos[1],self.target_pos[0])
        reach_ang = np.atan2(cursor_pos[1],cursor_pos[0])
        angular_error = reach_ang - targ_ang
        abs_dir_error = np.abs(angular_error)
        directional_error = wrap_to_pi(angular_error)

        vec_target = self.target_pos
        vec_actual = cursor_pos
        dot = np.dot(vec_target, vec_actual)
        norm_prod = np.linalg.norm(vec_target) * np.linalg.norm(vec_actual) + 1e-8  # avoid 0/0
        cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)
        angular_error = wrap_to_pi(np.arccos(cos_theta))  # radians
        abs_dir_error = np.abs(angular_error)

        info = {
            'abs_directional_error': abs_dir_error,
            'cursor_pos': cursor_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'directional_error': directional_error,
            'reach_ang': reach_ang
        }

        return self.state, reward, True, info
    
from scipy.special import i0  # Bessel function of the first kind, order 0

# learner class
class CursorControlLearner:
    def __init__(self, 
                 n_basis=16,            # number of von-mises basis functions for constructing policy and value function
                 kappa=5.0,             # width parameter for von-mises basis functions
                 alpha=0.01,            # learning rate for mean
                 alpha_nu=0.01,         # learning rate for variance (log-eigenvalues)
                 sigma=0.15,            # initial standard deviation of policy
                 seed=None,             # random seed to use
                 baseline_decay=0.99,   # decay rate for updating value function
                 radius=.12,            # distance of targets from start position (in m)
                 epsilon=0.2            # parameter to limit update size (to maintain stable learning)
                ):
        self.n_basis = n_basis
        self.kappa = kappa
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.rng = np.random.default_rng(seed)
        self.radius = radius
        self.init_std = sigma

        self.basis_centers = np.linspace(0, 2 * np.pi, n_basis, endpoint=False) # center of basis functions
        # randomly initialize weigths for policy
        self.W = self.rng.normal(scale=self.radius/4, size=(4, n_basis)) / self.init_std # divide by sigma to normalize

        self.nu = np.array([0.0, 0.0, 0.0, 0.0]) # initial normalized log-eigenvalues

        self.V = np.zeros(self.n_basis)         # value function weights
        self.baseline_decay = baseline_decay

        self.epsilon = epsilon

    # helper function to evaluate basis functions
    def compute_basis(self, s):
        diffs = s - self.basis_centers
        unnormalized = np.exp(self.kappa * np.cos(diffs))
        normalization = 2 * np.pi * i0(self.kappa)
        return unnormalized / normalization  # shape (n_basis,)

    # sample an action from the policy
    def sample_action(self, s):
        phi = self.compute_basis(s)            # (n_basis,)
        mu_norm = self.W @ phi                      # (4,)
        sigma_norm = np.exp(self.nu)                # (4,)
        action_norm = self.rng.normal(mu_norm, sigma_norm)    # (4,)
        return self.init_std*action_norm, self.init_std*mu_norm, self.init_std*sigma_norm, phi

    # compute log-probability of a selected action under the current policy
    def log_prob(self, a, mu, sigma):
        # convert to normalized coordinates
        a_norm = a/self.init_std
        mu_norm = mu/self.init_std
        sigma_norm = sigma/self.init_std
        return -0.5 * np.sum(((a_norm - mu_norm) / sigma_norm) ** 2 + 2 * self.nu + np.log(2 * np.pi))

    # update policy parameters
    def update(self, a, mu, sigma, phi, reward):
        # convert to normalized coordinates
        a_norm = a/self.init_std
        mu_norm = mu/self.init_std
        sigma_norm = sigma/self.init_std

        # Subtract baseline value to get advantage function
        b = self.V @ phi
        adv = reward - b
        
        # Value-function update: weight the 
        self.V += (1 - self.baseline_decay) * adv * phi  # update value function by gradient descent

        # W-update
        delta = a_norm - mu_norm
        grad_W = np.outer(delta / (sigma_norm**2), phi) * adv

        # clip updates according to PPO-based upper bound on change in mean
        for i in range(4):
            # Compute proposed update for this row of W
            delta_w_i = self.alpha * grad_W[i]  # shape (n_basis,)
            
            # Compute the corresponding change in mean action
            delta_mu_i = delta_w_i @ phi        # scalar
            
            # Cap the size of the update to the mean to avoid instability
            max_step = self.epsilon / (np.linalg.norm(a_norm[i]-mu_norm[i])/(sigma_norm[i]**2)+1e-8)
            #max_step = self.epsilon * (sigma_norm[i]) / abs(a_norm[i] - mu_norm[i] + 1e-8)  # add epsilon for stability

            if abs(delta_mu_i) > max_step:
                scale = max_step / abs(delta_mu_i)
                delta_w_i *= scale  # scale the update to satisfy constraint

            self.W[i] += delta_w_i

        grad_nu = ((delta**2) / (sigma_norm**2) - 1) 
        self.nu += self.alpha_nu * grad_nu * adv

    # initialize value function by sampling actions from current policy, obtaining associated rewards, and fitting weights V
    def initialize_value_function(self, env, n_trials=100):
        activations = [] # basis activations across trials
        rewards = []
        actions = []
        states = []
        for t in range(n_trials):
            s = env.reset()
            a, mu, sigma, phi = self.sample_action(s)
            _, r, _, _ = env.step(a)
            rewards.append(r)
            actions.append(a)
            activations.append(phi)
            states.append(s)

        Phi = np.stack(activations)
        R = np.array(rewards)
        self.V = np.linalg.lstsq(Phi, R, rcond=None)[0]
        return states, rewards, actions
        
    # evaluate mean policy over a range of angles - useful for visualization
    def evaluate_policy_over_angles(self, n_points=100):
        """
        Returns:
            angles: array of target angles (radians)
            means: array of shape (n_points, 4) with policy means at each angle
        """
        angles = np.linspace(0, 2 * np.pi, n_points)
        means = np.zeros((n_points, 4))

        for i, angle in enumerate(angles):
            phi = self.compute_basis(angle)
            mu_norm = self.W @ phi  # (4,)
            means[i] = mu_norm*self.init_std
            
        return angles, means
