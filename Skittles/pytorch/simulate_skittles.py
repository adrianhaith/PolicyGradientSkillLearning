#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:38 2025

@author: adrianhaith
"""

# script to simulate skittles

# first, import needed libraries
# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from environment import SkittlesEnv
#from learner_ana import SkittlesLearner
from learner_pytorch import SkittlesLearner_pytorch as SkittlesLearner

# %% Simulate learning

# Set random number seed
np.random.seed(4)

participant = SkittlesLearner(
    init_mean=[190, 2],
    init_std=[10, .5],
    alpha=0.05,
    alpha_nu=0.05,
    alpha_phi=0.05,
    rwd_baseline_decay=0.99
)
env = SkittlesEnv()

#print(np.exp(participant.nu.numpy()))

n_trials = 6000
actions = np.zeros((n_trials, 2))
rewards = np.zeros(n_trials)
nus     = np.zeros((n_trials, 2))   # <-- store log‐eigs
phis    = np.zeros(n_trials) 
rwd_baselines = np.zeros(n_trials)

#participant.initialize_rwd_baseline(env)
participant.rwd_baseline = -0.7
# %%
for t in range(n_trials):

    print(t)

    rwd_baselines[t] = participant.rwd_baseline
    
    _, _ = env.reset()
    a = participant.select_action()
    _, r, _, _, _ = env.step(a)
    participant.update(a, r)
    
    actions[t] = a
    rewards[t] = r
    nus[t]     = participant.nu   # <-- copy current ν
    phis[t]    = participant.phi
    

# %% plot outcome of simulations
bin_size = 10
n_bins   = n_trials // bin_size

def bin_mean(x):
    x = x[:n_bins * bin_size]
    return x.reshape(n_bins, bin_size).mean(axis=1)

angles_deg     = np.rad2deg(actions[:,0])
velocities     = actions[:,1]
rewards_binned = bin_mean(rewards)

lams = np.exp(nus)  # eigenvalues = exp(ν)
lam1_binned = bin_mean(lams[:,0])
lam2_binned = bin_mean(lams[:,1])

phis_deg       = np.rad2deg(phis)

# ——————————————————————
# 3) Plot all five panels
# ——————————————————————
fig, axs = plt.subplots(6,1, figsize=(10,10), sharex=True)

axs[0].plot( bin_mean(angles_deg) )
axs[0].set_ylabel("Angle (°)")
axs[0].set_title("Mean Launch Angle")

axs[1].plot( bin_mean(velocities), color='orange' )
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_title("Mean Launch Velocity")

axs[2].plot( rewards_binned, color='black' )
axs[2].plot( bin_mean(rwd_baselines), color='red')
axs[2].set_ylabel("Reward")
axs[2].set_title("Mean Reward")

axs[3].plot( lam1_binned, label="λ₁ (angle var)", color='blue' )
axs[3].plot( lam2_binned, label="λ₂ (vel var)", color='red' )
axs[3].set_ylabel("Eigenvalues")
axs[3].set_title("Covariance Eigenvalues")
axs[3].legend()

axs[4].plot( bin_mean(phis_deg), color='purple' )
axs[4].set_ylabel("Covariance Angle")
axs[4].set_xlabel("Trial Bin (10 trials)")
print("made plots")

plt.tight_layout()

# Convert angles to degrees
actions_deg = np.copy(actions)
actions_deg[:, 0] = np.rad2deg(actions[:, 0])

# Split actions
first_100 = actions_deg[:100]
last_100 = actions_deg[-100:]

# %%---Plot heatmap with two sets of actions
A_deg, V, R = env.compute_reward_grid(return_degrees = True)
plt.figure(figsize=(6, 6))
plt.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)
plt.scatter(first_100[:, 0], first_100[:, 1], color='blue', s=10, label='First 100 Trials')
plt.scatter(last_100[:, 0], last_100[:, 1], color='red', s=10, label='Last 100 Trials')

plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Velocity (m/s)")
plt.title("Skittles Task: Action Samples Over Learning")
#plt.xlim(A_deg.min(), A_deg.max())
plt.ylim(V.min(), V.max())
plt.legend()
plt.grid(True)
plt.tight_layout()





# # %% ---------Create animation showing learning evolving over time ----------
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML, display

# # --- Animation setup ---
# frame_size = 100
# step_size = 20
# n_frames = (n_trials - frame_size) // step_size + 1

# fig, ax = plt.subplots(figsize=(6, 6))
# c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

# # Set fixed limits
# ax.set_xlim(A_deg.min(), A_deg.max())
# ax.set_ylim(V.min(),V.max())
# ax.set_xlabel("Launch Angle (degrees)")
# ax.set_ylabel("Velocity (m/s)")
# ax.set_title("Learning Trajectory")

# # Initialize scatter plot
# sc = ax.scatter([], [], color='red', s=10)

# # --- Animation update function ---
# def update(frame):
#     start = frame * step_size
#     end = start + frame_size
#     batch = actions_deg[start:end]
#     sc.set_offsets(batch)
#     ax.set_title(f"Trials {start+1}–{end}")
    
#     return sc,

# print("making animation...")
# ani = FuncAnimation(fig, update, frames=n_frames, interval=50 , blit=False)

# plt.tight_layout()
# #display(HTML(ani.to_jshtml()))

# # Save to MP4
# ani.save("learning_animation.mp4", fps=24)
# #from IPython.display import Video
# #Video("learning_animation.mp4")
# print("done")

# # # %%
# # from matplotlib.animation import FuncAnimation
# # from IPython.display import HTML, display
# # import numpy as np

# # # --- Animation setup ---
# # frame_size = 100
# # step_size = 20
# # n_frames = (n_trials - frame_size) // step_size + 1

# # fig, ax = plt.subplots(figsize=(6, 6))
# # c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

# # ax.set_xlim(A_deg.min(), A_deg.max())
# # ax.set_ylim(V.min(), V.max())
# # ax.set_xlabel("Launch Angle (degrees)")
# # ax.set_ylabel("Velocity (m/s)")
# # ax.set_title("Learning Trajectory")

# # sc = ax.scatter([], [], color='red', s=10, zorder=3)

# # def update(frame):
# #     start = frame * step_size
# #     end = start + frame_size
# #     batch = actions_deg[start:end]
# #     sc.set_offsets(np.atleast_2d(batch))
# #     ax.set_title(f"Trials {start+1}–{end}")
# #     return sc,

# # print("making animation...")
# # ani = FuncAnimation(fig, update, frames=n_frames, interval=25, blit=False)

# # plt.tight_layout()
# # plt.close(fig)
# # display(HTML(ani.to_jshtml()))
# # print("done")
# # # %%

# %%
