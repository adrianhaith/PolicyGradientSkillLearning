# %%
"""
Created on Sun Jul  6 16:54:38 2025

@author: adrianhaith
"""
# script to simulate learning of skittles task through policy-gradient RL update

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotting import plot_policy_snapshot
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import SkittlesEnv, SkittlesLearner

make_animation = False      # flag to determine whether to create an animation for the evolution of learning over trials

# %% Simulate learning

# Set random number seed
np.random.seed(10)

env = SkittlesEnv(target=[.25, 1.2])

participant = SkittlesLearner(
    init_mean=[85, 3.5],    # initial mean action
    init_std=[8, .6],       # standard deviation of actions
    alpha=0.1,              # learning rate for mean
    alpha_nu=0.1,           # learning rate for variance (log-eigenvalues of cov. matrix)
    alpha_phi=0.1,          # learning rate for phi (covariance angle)
    rwd_baseline_decay=0.99 # decay rate for updating baseline reward
)

# plot sample trajectories with different, random actions
ax, out = env.plot_sample_trajectories(n_samples=5)

# initialize the baseline reward (take average reward over 100 sampled actions from initial policy)
participant.initialize_rwd_baseline(env)

# set number of trials to simulate
n_trials = 3000

# set up variables to store results during simulation
actions = np.zeros((n_trials, 2))
rewards = np.zeros(n_trials)
mus = np.zeros((n_trials, 2))
nus     = np.zeros((n_trials, 2))
phis    = np.zeros(n_trials) 
cov_mats = np.zeros((n_trials, 2, 2))
rwd_baselines = np.zeros(n_trials)

#%% simulate learning
for t in range(n_trials):

    rwd_baselines[t] = participant.rwd_baseline # record rwd baseline before trial states
    
    _, _ = env.reset()                  # reset task environment
    a = participant.select_action()     # select an action by the participant (sample from the Gaussian policy)
    _, r, _, _, _ = env.step(a)         # execute this action in the environment and get rewards
    participant.update(a, r)            # update the participants' policy based on the action and the reward
    
    # store variables
    actions[t] = a
    rewards[t] = r
    mus[t]     = participant.mu
    nus[t]     = participant.nu   # <-- copy current ν
    phis[t]    = participant.phi
    cov_mats[t] = participant.covariance

# %% plot outcome of simulations
bin_size = 10   # define bin size for smoothed plotting
n_bins   = n_trials // bin_size
def bin_mean(x):
    x = x[:n_bins * bin_size]
    return x.reshape(n_bins, bin_size).mean(axis=1)

angles_deg     = np.rad2deg(actions[:,0]) # convert angles to degrees
velocities     = actions[:,1]
rewards_binned = bin_mean(rewards)

lams = np.exp(nus)  # eigenvalues = exp(ν)
lam1_binned = bin_mean(lams[:,0])
lam2_binned = bin_mean(lams[:,1])

phis_deg       = np.rad2deg(phis)

# plot evolution of key values during learning
fig, axs = plt.subplots(5,1, figsize=(10,10), sharex=True)

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

plt.tight_layout()

# Convert angles to degrees
actions_deg = np.copy(actions)
actions_deg[:, 0] = np.rad2deg(actions[:, 0])

# %% optionally, create an animation showing evolution of learning across trials
if(make_animation):

    A_deg, V, R = env.compute_reward_grid(return_degrees = True)    # create grid of angles, velocities and associated reward to plot heatmap

    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    import numpy as np

    # --- Animation setup ---
    frame_size = 100
    step_size = 20
    n_frames = (n_trials - frame_size) // step_size + 1

    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

    ax.set_xlim(A_deg.min(), A_deg.max())
    ax.set_ylim(V.min(), V.max())
    ax.set_xlabel("Launch Angle (degrees)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Learning Trajectory")

    sc = ax.scatter([], [], color='red', s=10, zorder=3)

    def update(frame):
        start = frame * step_size
        end = start + frame_size
        batch = actions_deg[start:end]
        sc.set_offsets(np.atleast_2d(batch))
        ax.set_title(f"Trials {start+1}–{end}")
        return sc,

    print("making animation...")
    ani = FuncAnimation(fig, update, frames=n_frames, interval=25, blit=False)

    plt.tight_layout()
    plt.close(fig)
    display(HTML(ani.to_jshtml()))
    print("done")


# %% ---- Plot evolution of training along with covariance ellipses

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm

A_deg, V, R = env.compute_reward_grid(return_degrees = True)    # create grid of angles, velocities and associated reward to plot heatmap

# Color map setup
cmap = plt.cm.summer_r  # or 'plasma', 'inferno', etc.
norm = PowerNorm(gamma=.5, vmin=0, vmax=n_trials)  # normalize trial index for colormap
fig, ax = plt.subplots(figsize=(5, 4))

rwd_norm = PowerNorm(gamma=5, vmin=-1, vmax=0)
rwd_map = ax.pcolormesh(A_deg, V, R, cmap='Greys_r', alpha=0.9, norm=rwd_norm, rasterized=True)
cbar = plt.colorbar(rwd_map)

# list of trials at which to plot a 'snapshot' of current learning progress
snapshot_trials = np.array([0, 999, 1999, 2999])

colors = [cmap(norm(t)) for t in snapshot_trials] # color to use for each snapshot

for trial, color in zip(snapshot_trials, colors):
    mu = mus[trial]
    nu = nus[trial]
    phi = phis[trial]
    cov = cov_mats[trial]
    plot_policy_snapshot(ax, mu, cov, color, n_samples=20) # plot a snapshot of current policy (sampled trials + cov. ellipse)

# Formatting
ax.set_xlim(A_deg.min(), 120)
ax.set_ylim(0, env.angvel2vel(1000))
ax.set_xlabel("Launch Angle (degrees)")
ax.set_ylabel("Velocity (m/s)")

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


plt.tight_layout()
plt.savefig("policy_evolution.svg", format="svg", bbox_inches='tight')
plt.show()