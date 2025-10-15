# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:16:51 2025

Script to simulate reinforcement learning model of the bimanual cursor control task

@author: adrianhaith
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import CursorControlEnv, CursorControlLearner
from plotting import plot_value_function, plot_policy
from visualization import CursorLearningVisualizer

from utils import compute_von_mises_basis, bin_data

np.random.seed(1)

# % Simulate learning
# Create cursor control environment
env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False)

# Create learner
participant = CursorControlLearner(
    alpha=.1,               # learning rate for mean
    alpha_nu=0.1,           # learning rate for variance (log-eigenvalues of cov. matrix)
    sigma=.05,              # initial standard dev. of policy
    seed=11,                # random number seed for this agent
    baseline_decay=0.95,    # decay rate for updating reward baseline
    kappa=5,                # von-Mises basis function width
    epsilon=0.5             # parameter related to max step size during updates (to avoid very large updates, which destabilize learning). Smaller epsilong = smaller max. step size
    )

# Initialize the value function (target-dependent baseline reward) 
bsl_states, bsl_rewards, actions = participant.initialize_value_function(env, n_trials=100)

# visualize the value-function
ax = plot_value_function(participant.V, participant)
ax.plot(bsl_states, bsl_rewards, 'o', label='Sampled rewards')

# visualize initial policy
mu_init = plot_policy(participant)

# %%
# Simulate learning
n_trials = 2600  # number of trials to simulate

# set up history to store variables (for later plotting/analysis)
history = {
    'target_angles': np.zeros(n_trials),
    'actions': np.zeros((n_trials,4)),
    'rewards': np.zeros(n_trials),
    'Ws': np.zeros((n_trials, 4, participant.n_basis)),     # basis function weights for policy
    'nus': np.zeros((n_trials, 4)),                         # log-eigenvalues of cov. matrix
    'Vs': np.zeros((n_trials, participant.n_basis)),        # basis function weights for value function
    'dir_errors': np.zeros(n_trials)                        # directional errors
}

# simulate learning
for trial in range(n_trials):
    # store variables at start of trial
    history['Vs'][trial] = participant.V.copy() # NB use .copy() so this doesn't change when V/W/nu get updated
    history['Ws'][trial] = participant.W.copy()
    history['nus'][trial] = participant.nu.copy()

    # Get new target angle
    s = env.reset()
    history['target_angles'][trial]=s
    
    # Sample action and get reward
    a, mu, sigma, phi = participant.sample_action(s)
    _, r, _, info = env.step(a)
    
    # Update learner
    participant.update(a, mu, sigma, phi, r)
    
    # Store data for this trial
    history['actions'][trial] = a
    history['rewards'][trial] = r
    history['dir_errors'][trial] = info['directional_error']



# %% ------plot learning time course------

time = np.arange(n_trials)
action_labels = ['Lx', 'Ly', 'Rx', 'Ry']

# Compute standard deviations from nu
stds = np.exp(history['nus'])  # shape (n_trials, 4)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(4, 8), sharex=True)

bin_size = 60
rwd_binned, n_bins, _ = bin_data(history['rewards'], bin_size=bin_size)
bin_centers = bin_size*(np.arange(n_bins)+.5)

# Top panel: Rewards
axs[0].plot(bin_centers, rwd_binned, marker='o', label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_title("Learning performance")

# Middle panel: Actions
for i in range(4):
    action_binned, _, _ = bin_data(history['actions'][:, i], bin_size=bin_size)
    axs[1].plot(bin_centers, action_binned, marker='o', label=action_labels[i])
axs[1].set_ylabel("Action values")
axs[1].legend()

# Standard deviations (sqrt eigenvalues)

for i in range(4):
    std_binned, _, _ = bin_data(np.exp(history['nus'][:, i]), bin_size=bin_size)
    axs[2].plot(bin_centers, std_binned, marker='o', label=action_labels[i])
axs[2].set_ylabel("Std Dev")
axs[2].set_xlabel("Trial")
axs[2].legend()

# absolute direction error - for comparison with human data
dir_errors_binned, _, _ = bin_data(np.rad2deg(np.abs(history['dir_errors'])), bin_size=bin_size)
axs[3].plot(bin_centers, dir_errors_binned, marker='o', label='|directional_error|',markersize=3)
axs[3].set_yticks([0, 30, 60, 90])
axs[3].set_xlabel("Trial")
axs[3].set_ylabel("Absolute Directional Error")

plt.savefig("learning_timecourse.svg", format="svg", bbox_inches='tight')

plt.tight_layout()
plt.show()

#%%
# create viz object which facilitates plotting and analysis of run history
viz = CursorLearningVisualizer(participant, env, history)

# inspect learning updates for a given trials
tt=99 # trial of interest to inspect learning updates
viz.plot_policy_update(participant, tt) 

plt.savefig("endpoint_convergence.svg", format="svg", bbox_inches='tight')

# %% plot snapshots of current policy at different points during learning

target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
viz.plot_snapshot_with_samples(trial_idx=0, n_samples=10)
plt.savefig("endpoints_early.svg", format="svg", bbox_inches='tight')

viz.plot_snapshot_with_samples(trial_idx=1000, n_samples=10)
plt.savefig("endpoints_mid.svg", format="svg", bbox_inches='tight')

viz.plot_snapshot_with_samples(trial_idx=2599, n_samples=10)
plt.savefig("endpoints_late.svg", format="svg", bbox_inches='tight')