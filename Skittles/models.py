#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:08 2025

@author: adrianhaith
"""

# script to define environment and agent models for the skittles task
import numpy as np

# environment model - defines the skittles task, action space, reward etc.
class SkittlesEnv:
    def __init__(self, dt=0.01, T=2.0, target=[0.75, 0.75]):
        # Physical constants
        self.k = 1.0     # spring stiffness
        self.m = 0.1     # mass
        self.l = 0.4     # paddle length
        self.xp = 0.0    # paddle pivot x
        self.yp = -1.3   # paddle pivot y

        self.target = target # target position

        # Action: [angle (rad), velocity (m/s)]
        self.angle_range = (0.0, np.pi)
        self.vel_range = (0.0, 10.0)
        self.action_space = np.array([self.angle_range, self.vel_range])  # for convenience
        self.observation_space = None  # no state

        # Time vector for simulation
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T + dt, dt)
        self.omega = np.sqrt(self.k / self.m)

    def reset(self, *, seed=None, options=None):
        # function to reset environment
        # (trivial for this environment, but in-keeping with standard gym-style environment classes)
        return np.array([0.0], dtype=np.float32), {}

    def compute_error(self, action):
        # compute error (i.e. min distance to skittle) for a given action
        angle, v = action

        # Initial release position
        xr = self.xp + self.l * np.cos(angle)
        yr = self.yp + self.l * np.sin(angle)

        # Initial velocity
        vx = v * np.sin(angle)
        vy = v * np.cos(angle)

        # Trajectory under central spring dynamics
        x_t = xr * np.cos(self.omega * self.t) + (vx / self.omega) * np.sin(self.omega * self.t)
        y_t = yr * np.cos(self.omega * self.t) + (vy / self.omega) * np.sin(self.omega * self.t)

        # Distance to target at each time step
        dx = x_t - self.target[0]
        dy = y_t - self.target[1]
        distances = np.sqrt(dx**2 + dy**2)

        # Store full kinematic information in info
        info = {
            "trajectory": np.stack([x_t, y_t], axis=1),
            "min_distance": np.min(distances),
            "release_point": (xr, yr),
            "target": self.target
        }
        return np.min(distances), info
        
    def step(self, action):
        min_dist, info = self.compute_error(action)

        # Reward: negative of the minimum distance
        reward = -min_dist

        return np.array([0.0], dtype=np.float32), reward, True, False, info
    
    def compute_reward_grid(self, angle_range=None, velocity_range=None, resolution=300, return_degrees=False):
        """
        Computes a reward heatmap over the environment's action space.
    
        Parameters:
            angle_range (tuple): (min_angle, max_angle) in radians. Defaults to action_space angle range.
            velocity_range (tuple): (min_velocity, max_velocity). Defaults to action_space velocity range.
            resolution (int): number of points per axis.
            return_degrees (bool): if True, converts angle axis to degrees.
    
        Returns:
            A, V: meshgrid of angles and velocities
            R: reward matrix
        """
        # Use environment's action space bounds by default
        if angle_range is None:
            angle_range = self.angle_range
        if velocity_range is None:
            velocity_range = self.vel_range
    
        angles = np.linspace(angle_range[0], angle_range[1], resolution)
        velocities = np.linspace(velocity_range[0], velocity_range[1], resolution)
        A, V = np.meshgrid(angles, velocities)
        R = np.zeros_like(A)
    
        for i in range(resolution):
            for j in range(resolution):
                action = np.array([A[i, j], V[i, j]])
                _, reward, _, _, _ = self.step(action)
                R[i, j] = reward
    
        if return_degrees:
            A = np.rad2deg(A)
    
        return A, V, R
    
    def plot_sample_trajectories(self, actions=None, n_samples=10, seed=None, ax=None,
                                show_paddle=True, show_goal=True, cmap="viridis"):
        """
        Plot top-down view of the Skittles task with paddle, target, and sample trajectories.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        rng = np.random.default_rng(seed)

        if actions is None:
            # sample a random action
            actions = np.column_stack([
                rng.uniform(self.angle_range[0], self.angle_range[1], size=n_samples),
                rng.uniform(self.vel_range[0], self.vel_range[1], size=n_samples)
            ])
        else:
            actions = np.asarray(actions)
            if actions.ndim == 1:
                actions = actions[None, :]

        speeds = actions[:, 1]
        cmap_fn = get_cmap(cmap)
        speed_min, speed_max = speeds.min(), speeds.max()
        colors = [cmap_fn((v - speed_min) / (speed_max - speed_min + 1e-9)) for v in speeds]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        release_points = []
        trajectories = []

        if show_goal:
            ax.plot(self.target[0], self.target[1], marker="o", ms=20,
                    color="red", mec="black", mew=1.0, zorder=4, label="goal")
            ax.plot(0, 0, marker='o', color="black", ms=30)

        for (angle, v), color in zip(actions, colors):
            
            # Release point
            xr = self.xp + self.l * np.cos(angle)
            yr = self.yp + self.l * np.sin(angle)
            release_points.append((xr, yr))

            # Velocity at release
            vx = v * np.sin(angle)
            vy = - v * np.cos(angle)

            # Ball trajectory
            x_t = xr * np.cos(self.omega * self.t) + (vx / self.omega) * np.sin(self.omega * self.t)
            y_t = yr * np.cos(self.omega * self.t) + (vy / self.omega) * np.sin(self.omega * self.t)
            traj = np.column_stack([x_t, y_t])
            trajectories.append(traj)

            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2, alpha=0.9)
            ax.plot(x_t[25], y_t[25], "o", ms=30, mfc="white", mec=color, mew=1.5)

            if show_paddle:
                ax.plot([self.xp, xr], [self.yp, yr], "-", color="black", lw=2)
                ax.plot(self.xp, self.yp, "o", ms=10, color="black")

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_xlim(-1.5, 1.5)
        ax.grid(False)

        return ax, {"actions": actions,
                    "release_points": np.array(release_points),
                    "trajectories": trajectories}
    
    # function to compute angular velocity based on release velocity
    def vel2angvel(self, vel):
        return np.rad2deg(vel / self.l)
    
    def angvel2vel(self, angvel):
        return np.deg2rad(angvel * self.l)

    
# agent model class - defines the policy and learning rules
# note that this class internall represents the policy in normalized units. 
# This helps make learning rate parameters more meaningful, and makes changes of covariance between action dimensions more meaningful
# Without normalization, 90° rotation of the initial covariance matrix leads to drastic changes of the policy due to different scaling of the different action variables
class SkittlesLearner:
    def __init__(self, init_mean=None, init_std=None, alpha=0.01, alpha_nu=0.01, alpha_phi=0.01, rwd_baseline_decay=0.99):
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.alpha_phi = alpha_phi

        # Defaults
        if init_mean is None:
            init_mean = np.array([200.0, 2.0])
        if init_std is None:
            init_std = np.array([20.0, 0.5])  # 20° and 0.5 m/s

        self.init_mean = np.array(init_mean)
        self.init_std = np.array(init_std)

        # Learnable parameters (in normalized space)
        self.mu_norm = np.zeros(2)                      # normalized mean = 0
        self.nu = np.zeros(2)                           # log-eigenvalues in normalized units
        self.phi = 0.0
        self.Q = self._rotation_matrix(self.phi)

        self.rwd_baseline = 0.0
        self.rwd_baseline_decay = rwd_baseline_decay

    def _rotation_matrix(self, phi):
        return np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

    def _covariance_norm(self):
        Lambda = np.diag(np.exp(self.nu))
        return self.Q @ Lambda @ self.Q.T

    def _to_normalized(self, action_real):
        return (action_real - self.init_mean) / self.init_std

    def _from_normalized(self, action_norm):
        return self.init_mean + action_norm * self.init_std

    def initialize_rwd_baseline(self, env, n_samples=100):
        rewards = []
        for _ in range(n_samples):
            env.reset()
            action = self.select_action()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        self.rwd_baseline = np.mean(rewards)
    
    def select_action(self):
        # key function - samples an action given the current mean and covariance matrix
        cov = self._covariance_norm()
        a_norm = np.random.multivariate_normal(self.mu_norm, cov)
        self._last_action_norm = a_norm
        a_real = self._from_normalized(a_norm)
        return np.array([np.deg2rad(a_real[0]), a_real[1]])  # convert angle to radians for env

    def update(self, action_real_rad, reward):
        # Convert back to real degrees for consistency with init_mean/init_std
        action_deg = np.array([np.rad2deg(action_real_rad[0]), action_real_rad[1]])
        delta_real = action_deg - self.init_mean
        delta_norm = delta_real / self.init_std

        # --- Mean update (in normalized space) ---
        cov = self._covariance_norm()
        grad_logp_mu = np.linalg.inv(cov) @ (delta_norm - self.mu_norm)
        self.mu_norm += self.alpha * (reward - self.rwd_baseline) * grad_logp_mu

        # --- ν update ---
        z = self.Q.T @ (delta_norm - self.mu_norm)
        lambdas = np.exp(self.nu)
        grad_logp_nu = 0.5 * (-1 + (z ** 2) / lambdas)
        self.nu += self.alpha_nu * (reward - self.rwd_baseline) * grad_logp_nu

        # --- φ update ---
        Lambda_inv = np.diag(1.0 / lambdas)
        grad_logp_dQ = - self.Q.T @ np.outer(z, z) @ Lambda_inv
        dQ_dphi = np.array([
            [-np.sin(self.phi), -np.cos(self.phi)],
            [ np.cos(self.phi), -np.sin(self.phi)]
        ])
        grad_logp_phi = np.trace(grad_logp_dQ.T @ dQ_dphi)
        self.phi += self.alpha_phi * (reward - self.rwd_baseline) * grad_logp_phi
        self.Q = self._rotation_matrix(self.phi)

        # --- Update the reward baseline
        self.rwd_baseline = self.rwd_baseline_decay * self.rwd_baseline + (1 - self.rwd_baseline_decay) * reward

    # attribute to access the un-normalized mean
    @property
    def mu(self):
        return self._from_normalized(self.mu_norm)
    
    # attribute to access the un-normalized covariance matrix
    @property
    def covariance(self):
        cov_norm = self._covariance_norm()
        scaling_mat = np.diag(self.init_std)
        return scaling_mat @ cov_norm @ scaling_mat