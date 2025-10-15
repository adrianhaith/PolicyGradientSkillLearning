# ------------------------------- pytorch version of SkittlesLearner agent for learning skittles task by policy gradient
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
        self.mu_torch = nn.Parameter(torch.zeros(2))                     # mean in normalized space
        self.nu_torch = nn.Parameter(torch.zeros(2))                     # log-eigenvalues in normalized units
        self.phi_torch = nn.Parameter(torch.tensor(0.0))

        # Initialize internal variables to keep track of gradients
        self._mu_grad = None
        self._nu_grad = None
        self._phi_grad = None

        self.rwd_baseline = 0.0
        self.rwd_baseline_decay = rwd_baseline_decay

        self._log_prob = None

    def _rotation_matrix(self):
        cos_phi = torch.cos(self.phi_torch)
        sin_phi = torch.sin(self.phi_torch)
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
        Lambda = torch.diag(torch.exp(self.nu_torch))
        return Q @ Lambda @ Q.T

    def _to_real(self, a_norm):
        action_real = self.init_mean + a_norm * self.init_std
        return action_real

    def _to_normalized(self, a_real_deg):
        a_norm =  (a_real_deg - self.init_mean) / self.init_std
        return a_norm

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
        dist = D.MultivariateNormal(self.mu_torch, covariance_matrix=cov) # define distribution
        action_norm = dist.rsample()
        action_real = self._to_real(action_norm)
        action_real_rad = np.array([np.deg2rad(action_real[0].detach().numpy()), action_real[1].detach().numpy()])
        return action_real_rad

    def update(self, a_real_rad, reward):
        reward = torch.tensor(reward, dtype=torch.float32)

        # convert action to degrees, and then normalize
        a_real_deg = [np.rad2deg(a_real_rad[0]), a_real_rad[1]]
        a_norm = self._to_normalized(torch.tensor(a_real_deg, dtype=torch.float32))

        #print("pytorch a_norm in update():", a_norm)

        # Recreate the distribution for log_prob computation
        cov = self._covariance_norm()
        dist = D.MultivariateNormal(self.mu_torch, covariance_matrix=cov)
        log_prob = dist.log_prob(a_norm)
        self._log_prog = log_prob

        # store the log-probability
        self._log_prob = log_prob.detach()
        #print("log_prob in pytorch update():",self._log_prob)

        advantage = reward - self.rwd_baseline
        #loss_mu = -advantage * log_prob

        # Backpropagate
        log_prob.backward() # do back-propagation to get the gradient of the log-probability wrt to the parameters
 
        # Manual gradient scaling per parameter
        with torch.no_grad():
            self.mu_torch += self.alpha * self.mu_torch.grad * advantage
            self.nu_torch += self.alpha_nu * self.nu_torch.grad * advantage
            self.phi_torch += self.alpha_phi * self.phi_torch.grad * advantage

            self._mu_grad  = self.mu_torch.grad.detach().clone()
            self._nu_grad  = self.nu_torch.grad.detach().clone()
            self._phi_grad = self.phi_torch.grad.detach().clone()


            # reset the gradients (which would otherwise accumulate)
            self.mu_torch.grad.zero_()
            self.nu_torch.grad.zero_()
            self.phi_torch.grad.zero_()              

        # Update reward baseline
        self.rwd_baseline = self.rwd_baseline_decay * self.rwd_baseline + (1 - self.rwd_baseline_decay) * reward.item()

    def log_prob(self, a_real_rad):
        # convert action to degrees, and then normalize
        a_real_deg = [np.rad2deg(a_real_rad[0]), a_real_rad[1]]
        a_norm = self._to_normalized(torch.tensor(a_real_deg, dtype=torch.float32))
        #print("pytorch a_norm in log_prob():", a_norm)
        # Recreate the distribution for log_prob computation
        cov = self._covariance_norm()
        dist = D.MultivariateNormal(self.mu_torch, covariance_matrix=cov)
        log_prob = dist.log_prob(a_norm)
        #print("log_prob in pytorch log_prob():",log_prob)
        return log_prob.detach().numpy()

    # recast mu, nu, and phi from torch to numpy, and make them available as attributes
    @property
    def mu(self):
        return self.mu_torch.detach().numpy()

    @property
    def nu(self):
        return self.nu_torch.detach().numpy()

    @property
    def phi(self):
        return self.phi_torch.detach().numpy()
    

    # make gradients available as 
    @property
    def mu_grad(self):
        return None if self.mu_torch.grad is None else self._mu_grad.numpy()

    @property
    def nu_grad(self):
        return None if self.nu_torch.grad is None else self._nu_grad.numpy()

    @property
    def phi_grad(self):
        return None if self.phi_torch.grad is None else self._phi_grad.numpy()
    
    # define setters so that parameters can be adjusted from outside of this class (useful for e.g. calculating gradient based on finite differences)
    @mu.setter
    def mu(self, value):
        self.mu_torch.data = torch.tensor(value, dtype=torch.float32)

    @nu.setter
    def nu(self, value):
        self.nu_torch.data = torch.tensor(value, dtype=torch.float32)

    @phi.setter
    def phi(self, value):
        self.phi_torch.data = torch.tensor(value, dtype=torch.float32)