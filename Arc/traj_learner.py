#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 08:48:40 2025

@author: adrianhaith
"""

import numpy as np
class TrajLearner:
    def __init__(self, Ng, alpha=0.01, alpha_nu=0.01, init_goals=None, init_std=0.05, baseline_decay=0.95, epsilon=0.3):
        """
        Parameters:
            Ng: number of subgoals
            alpha: learning rate
            init_arc: optional array of shape (2, Ng) giving initial mean subgoal locations
            init_std: initial standard deviation (shared or scalar)
        """
        self.Ng = Ng
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        
        self.action_dim = 2 * Ng

        self.rwd_baseline = 0
        self.baseline_decay = baseline_decay

        self.init_std = init_std

        if init_goals is not None:
            self.mean_norm = init_goals.flatten() / self.init_std
        else:
            self.mean_norm = np.zeros(self.action_dim) / self.init_std  # should be overridden

        self.nu = np.ones(self.action_dim) * 0 # initial normalized log-variance (normalized variance = 1)

        self.epsilon = epsilon # PPO parameter (closer to 0 = more conservative updating)

    def sample_action(self):
        std_norm = np.sqrt(np.exp(self.nu))
        action_norm = self.mean_norm + std_norm * np.random.randn(self.action_dim) # sample normalized action
        action_flat = self.init_std * action_norm
        return action_flat.reshape((2, self.Ng))

    def initialize_baseline(self, env, n_trials=100):
        rewards = []
        actions = []
        for _ in range(n_trials):
            env.reset()
            a  = self.sample_action()
            _, r, _, _ = env.step(a)
            rewards.append(r)
            actions.append(a)

        self.rwd_baseline = np.mean(rewards)
        return np.mean(rewards)

    def update(self, action, reward):
        """
        Policy gradient update

        Parameters:
            actions
            rewards: list of scalar rewards
        """
        # update reward baseline
        self.rwd_baseline = self.baseline_decay * self.rwd_baseline + (1 - self.baseline_decay) * reward
        
        # convert to normalized action
        action_norm = action / self.init_std
        action_flat = action_norm.flatten()
        delta = action_flat - self.mean_norm

        std_norm = np.sqrt(np.exp(self.nu))
        std2 = np.exp(self.nu)
        #cov_norm = np.diag(std_norm **2) # covariance matrix
        
        grad_logp_mu = delta / std2

        step = self.alpha * (reward - self.rwd_baseline) * grad_logp_mu

        # PPO-style clipping to avoid excessively large updates to the mean
        step_size = np.linalg.norm(step)
        max_step_size = self.epsilon / (np.linalg.norm(delta/(std_norm **2)) + 1e-8) #np.linalg.norm(std_norm **2 / (delta + 1e-8))
        
        if max_step_size > 80:
            breakpoint()
            aaa=1

        if step_size > max_step_size:
            step = step * max_step_size / (step_size + 1e-8) # scale down step magnitude if needed

        self.mean_norm += step

        grad_logp_nu = -.5 +.5*(delta **2) * np.exp(-self.nu) 
        self.nu += self.alpha_nu * grad_logp_nu.ravel() * (reward - self.rwd_baseline)
        
        # clamp variance to make sure it doesn't get too small
        min_variance = 1e-6
        self.nu = np.maximum(self.nu, np.log(min_variance))

        return step, max_step_size
        
    def get_policy_mean(self):
        return self.mean_norm.reshape((2, self.Ng)) * self.init_std