#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:08 2025

@author: adrianhaith
"""

# script to define environment and agent models for the skittles task
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# environment model - defines the skittles task, action space, reward etc.
class SkittlesEnv(gym.Env):
    def __init__(self, dt=0.01, T=2.0):
        super().__init__()

        # Physical constants
        self.k = 1.0     # spring stiffness
        self.m = 0.1     # mass
        self.l = 0.2     # paddle length
        self.xp = 0.0    # paddle pivot x
        self.yp = -1.5   # paddle pivot y

        self.target = np.array([0.8, 0.8], dtype=np.float32)

        # Action: [angle (rad), velocity (m/s)]
        self.action_space = spaces.Box(
        low=np.array([np.pi, 0.0], dtype=np.float32),
        high=np.array([2 * np.pi, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        # Dummy observation
        self.observation_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32
        )

        # Time vector for simulation
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T + dt, dt)
        self.omega = np.sqrt(self.k / self.m)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        angle, v = action

        # Initial release position
        xr = self.xp - self.l * np.cos(angle)
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

        # Reward: negative of the minimum distance
        reward = -np.min(distances)

        # Store full kinematic information in info
        info = {
            "trajectory": np.stack([x_t, y_t], axis=1),
            "min_distance": np.min(distances),
            "release_point": (xr, yr),
            "target": self.target
        }

        return np.array([0.0], dtype=np.float32), reward, True, False, info
    
    def compute_reward_grid(self, angle_range=None, velocity_range=None, resolution=100, return_degrees=False):
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
            angle_range = (self.action_space.low[0], self.action_space.high[0])
        if velocity_range is None:
            velocity_range = (self.action_space.low[1], self.action_space.high[1])
    
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
    
    
