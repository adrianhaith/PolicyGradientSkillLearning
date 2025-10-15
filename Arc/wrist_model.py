#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:36:54 2025

@author: adrianhaith
"""

import numpy as np
from scipy.linalg import expm

class WristLDS:
    def __init__(self, dt, sigma_u = 1000):
        self.T = 0.78            # total movement duration (s)
        self.horizon = 0.28     # time horizon for optimal control (s)
        self.dt = dt
        self.goal_dt = 0.13     # update interval for subgoals
        self.x_init = np.zeros(10)
        self.wQ = 1_000_000
        self.sigma_u = sigma_u

        # System parameters
        m = 5     # mass
        l = 100    # viscosity
        k = 0     # stiffness
        tau1 = 0.05
        tau2 = 0.05

        # Define continuous-time system
        A = np.array([
            [0, 1, 0, 0, 0],
            [-k/m, -l/m, 1, 0, 0],
            [0, 0, -1/tau1, 1/tau1, 0],
            [0, 0, 0, -1/tau2, 0],
            [0, 0, 0, 0, 0]
        ])
        B = np.array([[0], [0], [0], [1/tau2], [0]])

        # Discretize system using matrix exponential
        M = np.block([
            [A, B],
            [np.zeros((B.shape[1], A.shape[0] + B.shape[1]))]
        ])
        Md = expm(M * self.dt)
        A1d = Md[:A.shape[0], :A.shape[1]]
        B1d = Md[:A.shape[0], A.shape[1]:]

        # Expand to 2D system (x and y dimensions)
        self.Ad = np.block([
            [A1d, np.zeros_like(A1d)],
            [np.zeros_like(A1d), A1d]
        ])
        self.Bd = np.block([
            [B1d, np.zeros_like(B1d)],
            [np.zeros_like(B1d), B1d]
        ])

        self.Nstates = self.Ad.shape[0]
        self.Nu = self.Bd.shape[1]

        # Cost matrices
        Q1d = np.array([
            [1, 0, 0, 0, -1],
            [0, 0.01, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1]
        ])
        self.Q = self.wQ * np.block([
            [Q1d, np.zeros_like(Q1d)],
            [np.zeros_like(Q1d), Q1d]
        ])
        self.R = self.dt * np.eye(self.Nu)

        # Solve for optimal feedback gains
        self.NT = int(self.horizon / self.dt)
        self.V = np.zeros((self.Nstates, self.Nstates, self.NT))
        self.L = np.zeros((self.Nu, self.Nstates, self.NT))

        self.V[:, :, -1] = self.Q
        for i in reversed(range(self.NT - 1)):
            BtV = self.Bd.T @ self.V[:, :, i+1]
            inv_term = np.linalg.inv(self.R + BtV @ self.Bd)
            self.L[:, :, i] = -inv_term @ BtV @ self.Ad
            self.V[:, :, i] = (
                self.Ad.T @ self.V[:, :, i+1] @ (self.Ad + self.Bd @ self.L[:, :, i])
                + self.L[:, :, i].T @ self.R @ self.L[:, :, i]
            )

    def simulate(self, subgoals):
        """
        Simulate a full wrist movement given a sequence of subgoals.
    
        Parameters:
            subgoals: (2, Ng) array — sequence of (x, y) goal positions,
                      one per goal update interval (e.g., every 130 ms)
    
        Returns:
            x_traj: (NT+1, Nstates) array — full state trajectory
            u_traj: (NT, Nu) array — control inputs at each timestep
        """
        subgoals = np.asarray(subgoals)
        Ng = subgoals.shape[1]  # number of submovements
        NT_sub = int(self.goal_dt / self.dt)  # number of simulation timesteps in a submovement
        NT = NT_sub * Ng  # total number of simulation timesteps in the whole movement
    
        x_traj = np.zeros((NT + 1, self.Nstates))
        u_traj = np.zeros((NT, self.Nu))
    
        ig = self.Nstates // 2  # index of x/y position goals
        x = self.x_init.copy()
    
        sim_idx = 0
        for j in range(Ng):  # loop over submovements
            # set the goal for future timesteps to the current goals
            gx, gy = subgoals[:, j]
            x[ig-1]=gx 
            x[2*ig-1]=gy
    
            for i in range(NT_sub):
                L_t = self.L[:, :, i]
                u = L_t @ x + self.sigma_u * np.random.normal(size=(1,2))
                x_traj[sim_idx] = x
                u_traj[sim_idx] = u
    
                x = self.Ad @ x + self.Bd @ u[0]
                sim_idx += 1
    
        x_traj[sim_idx] = x  # add final state
        return x_traj, u_traj
    
    
