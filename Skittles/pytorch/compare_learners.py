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
from learner_ana import SkittlesLearner, make_history_dict
from learner_pytorch import SkittlesLearner_pytorch

#---- Simulate learning

# Set random number seed
np.random.seed(4)

participant = SkittlesLearner(
    init_mean=[190, 2],
    init_std=[10, .5],
    alpha=0.0,#0.05,
    alpha_nu=0.0,#0.05,
    alpha_phi=0.0,#0.05,
    rwd_baseline_decay=0.99
)

participant_torch = SkittlesLearner_pytorch(
    init_mean=[190, 2],
    init_std=[10, .5],
    alpha=0.0,#0.05,
    alpha_nu=0.0,#0.05,
    alpha_phi=0.0,#0.05,
    rwd_baseline_decay=0.99
)
env = SkittlesEnv()

#print(np.exp(participant.nu.numpy()))
history = make_history_dict()
history_torch = make_history_dict()

n_trials = 6000
participant.initialize_rwd_baseline(env)
#participant_torch.initialize_rwd_baseline(env)

# set torch participant baseline to match that of the non-torch version:
participant_torch.rwd_baseline = participant.rwd_baseline 

# %% --- Key debug cell: inspect a single update for the same actions in each model

_, _ = env.reset()
# sample action from participant
#a = participant.select_action()
#_, r, _, _, _= env.step(a)

# define arbitrary action and reward
a = [np.deg2rad(200), 3]
r = -0.4

# # %% -- Now, update BOTH agents with the same action and reward
participant.update(a,r)
participant_torch.update(a,r)

# # now can look at the gradients and how they differ
# print("updated both agents")

# %% -- numerically compute the gradients wrt mu, nu, and phi ------

# define arbitrary action and reward
a = [np.deg2rad(200), 3]
r = -0.4

# reset parameters
participant.mu = [0, 0]
participant.nu = [0, 0]

logp = participant.log_prob(a)
dd = .000001
participant.mu=[dd, 0]
logp2 = participant.log_prob(a)
d_logp_dmu0 = (logp2-logp)/dd

participant.mu=[0, dd]
logp2 = participant.log_prob(a)
d_logp_dmu1 = (logp2-logp)/dd

participant.mu = [0,0]

print("Analytical Model------")
print("log prob: ",logp)
print("Finite Differences:")
print(".  d logp d mu[0]:", d_logp_dmu0)
print(".  d logp d mu[1]:", d_logp_dmu1)

# now do the nu's
logp = participant.log_prob(a)

nu_init = participant.nu
participant.nu=[dd, 0]
logp2 = participant.log_prob(a)
d_logp_dnu0 = (logp2-logp)/dd

participant.nu=[0, dd]
logp2 = participant.log_prob(a)
d_logp_dnu1 = (logp2-logp)/dd

participant.nu = [0,0]

print(".  d logp d nu[0]:", d_logp_dnu0)
print(".  d logp d nu[1]:", d_logp_dnu1)

# compare to analytical
# first, update the model (to ensure gradients are calculated and stored)
participant.update(a,r)
print("Analytical:")
print(".  d logp d mu[0]:", participant.mu_grad[0])
print(".  d logp d mu[1]:", participant.mu_grad[1])
print(".  d logp d nu[0]:", participant.nu_grad[0])
print(".  d logp d nu[1]:", participant.nu_grad[1])



# %% ------Now, do same for torch model------

# define arbitrary action and reward
a = [np.deg2rad(200), 3]
r = -0.4

participant_torch.mu = [0, 0]
participant_torch.nu = [0, 0]
logp = participant_torch.log_prob(a)
dd = .001
participant_torch.mu=[dd, 0]
logp1_1 = participant_torch.log_prob(a)
d_logp_dmu0 = (logp1_1-logp)/dd

participant_torch.mu=[0, dd]
logp1_2 = participant_torch.log_prob(a)
d_logp_dmu1 = (logp1_2-logp)/dd

participant_torch.mu = [0, 0]

print("Pytorch Model------")
print("log prob: ",logp)
print("Finite Differences:")
print(".  d logp d mu[0]:", d_logp_dmu0)
print(".  d logp d mu[1]:", d_logp_dmu1)


# % nu for torch model------
#logp = participant_torch.log_prob(a)
dd = .001

participant_torch.nu = [dd, 0]
logp2_1 = participant_torch.log_prob(a)
d_logp_dnu0 = (logp2_1-logp)/dd

participant_torch.nu=[0, dd]
logp2_2 = participant_torch.log_prob(a)
d_logp_dnu1 = (logp2_2-logp)/dd

participant_torch.nu = [0, 0]

print(".  d logp d nu[0]:", d_logp_dnu0)
print(".  d logp d nu[1]:", d_logp_dnu1)

# comare to analytical
# first, update the model to ensure gradients are computed
participant_torch.update(a,r)
print("Analytical:")
print(".  d logp d mu[0]:", participant_torch.mu_grad[0])
print(".  d logp d mu[1]:", participant_torch.mu_grad[1])
print(".  d logp d nu[0]:", participant_torch.nu_grad[0])
print(".  d logp d nu[1]:", participant_torch.nu_grad[1])

# %%

# history["mu_grad"].append(participant.mu_grad)
# history["nu_grad"].append(participant.nu_grad)
# history["phi_grad"].append(participant.phi_grad)

# history_torch["mu_grad"].append(participant_torch.mu_grad)
# history_torch["nu_grad"].append(participant_torch.nu_grad)
# history_torch["phi_grad"].append(participant_torch.phi_grad)






# for t in range(n_trials):

#     history["rwd_baseline"].append(participant.rwd_baseline)
    
#     _, _ = env.reset()
#     a = participant.select_action()
#     _, r, _, _, _ = env.step(a)
#     participant.update(a, r)
    
#     history["action"].append(a)
#     history["reward"].append(r)
#     history["mu"].append(participant.mu)
#     history["nu"].append(participant.nu)
#     history["phi"].append(participant.phi)
#     history["mu_grad"].append(participant.mu_grad)
#     history["nu_grad"].append(participant.nu_grad)
#     history["phi_grad"].append(participant.phi_grad)

#     # ---torch versions of these updates
#     history_torch["rwd_baseline"].append(participant_torch.rwd_baseline)
    
#     _, _ = env.reset()
#     a_torch = participant_torch.select_action()
#     _, r_torch, _, _, _ = env.step(a_torch)
#     participant_torch.update(a_torch, r_torch)
    
#     history_torch["action"].append(a_torch)
#     history_torch["reward"].append(r_torch)
#     history_torch["mu"].append(participant_torch.mu)
#     history_torch["nu"].append(participant_torch.nu)
#     history_torch["phi"].append(participant_torch.phi)
#     history_torch["mu_grad"].append(participant_torch.mu_grad)
#     history_torch["nu_grad"].append(participant_torch.nu_grad)
#     history_torch["phi_grad"].append(participant_torch.phi_grad)

# # %% plot outcome of simulations
# bin_size = 10
# n_bins   = n_trials // bin_size

# def bin_mean(x):
#     x = x[:n_bins * bin_size]
#     return x.reshape(n_bins, bin_size).mean(axis=1)

# angles_deg     = np.rad2deg(actions[:,0])
# velocities     = actions[:,1]
# rewards_binned = bin_mean(rewards)

# lams = np.exp(nus)  # eigenvalues = exp(ν)
# lam1_binned = bin_mean(lams[:,0])
# lam2_binned = bin_mean(lams[:,1])

# phis_deg       = np.rad2deg(phis)

# # ——————————————————————
# # 3) Plot all five panels
# # ——————————————————————
# fig, axs = plt.subplots(6,1, figsize=(10,10), sharex=True)

# axs[0].plot( bin_mean(angles_deg) )
# axs[0].set_ylabel("Angle (°)")
# axs[0].set_title("Mean Launch Angle")

# axs[1].plot( bin_mean(velocities), color='orange' )
# axs[1].set_ylabel("Velocity (m/s)")
# axs[1].set_title("Mean Launch Velocity")

# axs[2].plot( rewards_binned, color='black' )
# axs[2].plot( bin_mean(rwd_baselines), color='red')
# axs[2].set_ylabel("Reward")
# axs[2].set_title("Mean Reward")

# axs[3].plot( lam1_binned, label="λ₁ (angle var)", color='blue' )
# axs[3].plot( lam2_binned, label="λ₂ (vel var)", color='red' )
# axs[3].set_ylabel("Eigenvalues")
# axs[3].set_title("Covariance Eigenvalues")
# axs[3].legend()

# axs[4].plot( bin_mean(phis_deg), color='purple' )
# axs[4].set_ylabel("Covariance Angle")
# axs[4].set_xlabel("Trial Bin (10 trials)")
# print("made plots")

# plt.tight_layout()

# # Convert angles to degrees
# actions_deg = np.copy(actions)
# actions_deg[:, 0] = np.rad2deg(actions[:, 0])

# # Split actions
# first_100 = actions_deg[:100]
# last_100 = actions_deg[-100:]

# # %%---Plot heatmap with two sets of actions
# A_deg, V, R = env.compute_reward_grid(return_degrees = True)
# plt.figure(figsize=(6, 6))
# plt.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)
# plt.scatter(first_100[:, 0], first_100[:, 1], color='blue', s=10, label='First 100 Trials')
# plt.scatter(last_100[:, 0], last_100[:, 1], color='red', s=10, label='Last 100 Trials')

# plt.xlabel("Launch Angle (degrees)")
# plt.ylabel("Velocity (m/s)")
# plt.title("Skittles Task: Action Samples Over Learning")
# plt.xlim(A_deg.min(), A_deg.max())
# plt.ylim(V.min(), V.max())
# plt.legend()
# plt.grid(True)
# plt.tight_layout()





# # # %% ---------Create animation showing learning evolving over time ----------
# # from matplotlib.animation import FuncAnimation
# # from IPython.display import HTML, display

# # # --- Animation setup ---
# # frame_size = 100
# # step_size = 20
# # n_frames = (n_trials - frame_size) // step_size + 1

# # fig, ax = plt.subplots(figsize=(6, 6))
# # c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

# # # Set fixed limits
# # ax.set_xlim(A_deg.min(), A_deg.max())
# # ax.set_ylim(V.min(),V.max())
# # ax.set_xlabel("Launch Angle (degrees)")
# # ax.set_ylabel("Velocity (m/s)")
# # ax.set_title("Learning Trajectory")

# # # Initialize scatter plot
# # sc = ax.scatter([], [], color='red', s=10)

# # # --- Animation update function ---
# # def update(frame):
# #     start = frame * step_size
# #     end = start + frame_size
# #     batch = actions_deg[start:end]
# #     sc.set_offsets(batch)
# #     ax.set_title(f"Trials {start+1}–{end}")
    
# #     return sc,

# # print("making animation...")
# # ani = FuncAnimation(fig, update, frames=n_frames, interval=50 , blit=False)

# # plt.tight_layout()
# # #display(HTML(ani.to_jshtml()))

# # # Save to MP4
# # ani.save("learning_animation.mp4", fps=24)
# # #from IPython.display import Video
# # #Video("learning_animation.mp4")
# # print("done")

# # # # %%
# # # from matplotlib.animation import FuncAnimation
# # # from IPython.display import HTML, display
# # # import numpy as np

# # # # --- Animation setup ---
# # # frame_size = 100
# # # step_size = 20
# # # n_frames = (n_trials - frame_size) // step_size + 1

# # # fig, ax = plt.subplots(figsize=(6, 6))
# # # c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

# # # ax.set_xlim(A_deg.min(), A_deg.max())
# # # ax.set_ylim(V.min(), V.max())
# # # ax.set_xlabel("Launch Angle (degrees)")
# # # ax.set_ylabel("Velocity (m/s)")
# # # ax.set_title("Learning Trajectory")

# # # sc = ax.scatter([], [], color='red', s=10, zorder=3)

# # # def update(frame):
# # #     start = frame * step_size
# # #     end = start + frame_size
# # #     batch = actions_deg[start:end]
# # #     sc.set_offsets(np.atleast_2d(batch))
# # #     ax.set_title(f"Trials {start+1}–{end}")
# # #     return sc,

# # # print("making animation...")
# # # ani = FuncAnimation(fig, update, frames=n_frames, interval=25, blit=False)

# # # plt.tight_layout()
# # # plt.close(fig)
# # # display(HTML(ani.to_jshtml()))
# # # print("done")
# # # # %%

# %%
