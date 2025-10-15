#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  14 14:05:08 2025

@author: adrianhaith
"""

# script to define agent models for the skittles task
# this version includes a version that uses pytorch to compute the gradient, rather than depending on an analytical formula


import numpy as np
import gymnasium as gym
from gymnasium import spaces

# agent model - defines the policy and learning rules
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

        


# ------------------------------- pytorch version below ----------------

import torch
import torch.nn as nn
import torch.distributions as D

# agent model - defines the policy and learning rules
class SkittlesLearner_pytorch:
    def __init__(self, init_mean=None, init_std=None, alpha=0.01, alpha_nu=0.01, alpha_phi=0.01, rwd_baseline_decay=0.99):
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.alpha_phi = alpha_phi

        # Defaults
        if init_mean is None:
            init_mean = np.array([200.0, 2.0])
        if init_std is None:
            init_std = np.array([20.0, 0.5])  # 20° and 0.5 m/s

        self.init_mean = torch.tensor(init_mean, dtype=torch.float32)
        self.init_std = torch.tensor(init_std, dtype=torch.float32)

        # Learnable parameters (in normalized space)
        self.mu_norm = nn.Parameter(torch.zeros(2))                     # mean in normalized space
        self.nu = nn.Parameter(torch.zeros(2))                          # log-eigenvalues in normalized units
        self.phi = nn.Parameter(torch.tensor(0.0))

        self.rwd_baseline = 0.0
        self.rwd_baseline_decay = rwd_baseline_decay

    def _rotation_matrix(self):
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)
        return torch.stack([
            torch.stack([cos_phi, -sin_phi]),
            torch.stack([sin_phi, cos_phi])
        ])
    
        return np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

    def _covariance_norm(self):
        Q = self._rotation_matrix()
        Lambda = torch.diag(torch.exp(self.nu))
        return Q @ Lambda @ Q.T

    def _to_real(self, action_norm):
        action_real = self.init_mean + action_norm * self.init_std
        return torch.stack([
            torch.deg2rad(action_real[0]),
            action_real[1]
        ])

    def _to_normalized(self, action_real):
        action_norm =  (action_real - self.init_mean) / self.init_std
        return torch.stack([
            torch.deg2rad(action_norm[0]),
            action_real[1]
        ])

    #def _from_normalized(self, action_norm):
    #    return self.init_mean + action_norm * self.init_std

    def initialize_rwd_baseline(self, env, n_samples=100):
        rewards = []
        for _ in range(n_samples):
            env.reset()
            action = self.select_action()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        self.rwd_baseline = np.mean(rewards)
    
    def select_action(self):
        cov = self._covariance_norm()
        dist = D.MultivariateNormal(self.mu, covariance_matrix=cov) # define distribution
        action_norm = dist.rsample()
        action_real = self._to_real(action_norm)
        return action_real.numpy()

    def update(self, action_real_rad, reward):
        reward = torch.tensor(reward, dtype=torch.float32)

        a_norm = self._to_normalized(torch.tensor(action_real_rad, dtype=torch.float32))

        # Recreate the distribution for log_prob computation
        cov = self._covariance_norm()
        dist = D.MultivariateNormal(self.mu, covariance_matrix=cov)
        log_prob = dist.log_prob(a_norm)

        advantage = reward - self.rwd_baseline
        #loss_mu = -advantage * log_prob

        # Backpropagate
        log_prob.backward() # do back-propagation to get the gradient of the log-probability wrt to the parameters
 
        # Manual gradient scaling per parameter
        with torch.no_grad():
            self.mu -= self.alpha * self.mu.grad * advantage
            self.nu -= self.alpha_nu * self.nu.grad * advantage
            self.phi -= self.alpha_phi * self.phi.grad * advantage

            # reset the gradients (which would otherwise accumulate)
            self.mu.grad.zero_()
            self.nu.grad.zero_()
            self.phi.grad.zero_()              

        # Update reward baseline
        self.rwd_baseline = self.rwd_baseline_decay * self.rwd_baseline + (1 - self.rwd_baseline_decay) * reward.item()

        return self.mu.grad, self.nu.grad, self.phi.grad

    @property
    def mu_numpy(self):
        return self.mu.detach().numpy()

    @property
    def nu_numpy(self):
        return self.nu.detach().numpy()

    @property
    def phi_numpy(self):
        return self.phi.detach().numpy()




        # # Convert back to real degrees for consistency with init_mean/init_std
        # action_deg = np.array([np.rad2deg(action_real_rad[0]), action_real_rad[1]])
        # delta_real = action_deg - self.init_mean
        # delta_norm = delta_real / self.init_std

        # # --- Mean update (in normalized space) ---
        # cov = self._covariance_norm()
        # grad_logp_mu = np.linalg.inv(cov) @ (delta_norm - self.mu_norm)
        # self.mu_norm += self.alpha * (reward - self.rwd_baseline) * grad_logp_mu

        # # --- ν update ---
        # z = self.Q.T @ (delta_norm - self.mu_norm)
        # lambdas = np.exp(self.nu)
        # grad_logp_nu = 0.5 * (-1 + (z ** 2) / lambdas)
        # self.nu += self.alpha_nu * (reward - self.rwd_baseline) * grad_logp_nu

        # # --- φ update ---
        # Lambda_inv = np.diag(1.0 / lambdas)
        # grad_logp_dQ = - self.Q.T @ np.outer(z, z) @ Lambda_inv
        # dQ_dphi = np.array([
        #     [-np.sin(self.phi), -np.cos(self.phi)],
        #     [ np.cos(self.phi), -np.sin(self.phi)]
        # ])
        # grad_logp_phi = np.trace(grad_logp_dQ.T @ dQ_dphi)
        # self.phi += self.alpha_phi * (reward - self.rwd_baseline) * grad_logp_phi
        # self.Q = self._rotation_matrix(self.phi)

        # # --- Update the reward baseline
        # self.rwd_baseline = self.rwd_baseline_decay * self.rwd_baseline + (1 - self.rwd_baseline_decay) * reward