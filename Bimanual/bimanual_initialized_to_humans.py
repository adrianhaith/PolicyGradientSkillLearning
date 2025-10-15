# %% Script to compare simulation of a single learning run of policy-gradient RL model for the
# bimanual cursor-control task, with policy initialized to early behavior of human learners

import numpy as np
from scipy.io import loadmat
from models import CursorControlLearner
from models import CursorControlEnv
from utils import wrap_to_2pi, bin_data, analyze_bias_variability_sliding, fit_human_policy
import matplotlib.pyplot as plt
from visualization import CursorLearningVisualizer

from plotting import plot_policy, plot_value_function

# Load human data from .mat file
# NB - code here needs to be slightly different depending on whether or not you are using Jupyter
from pathlib import Path
#DATA_PATH = Path('human_data.mat').parent / 'human_data.mat'.  # use this line instead if not using Jupyter
DATA_PATH = Path(__file__).parent / 'human_data.mat'
mat = loadmat(DATA_PATH)
subj_id = 7 # subject id to use for this simulation run (can be between 0 and 12)
subj_data = mat['human_data'][0][subj_id]  

target_angles = wrap_to_2pi(np.pi/2 - subj_data['target_angles'].squeeze())  # (n_trials,)
hand_movements = subj_data['hand_movements'].T  # (n_trials, 2)

# Fit initial policy to initial human performance (first 120 trials)
W_init, std_init, nu_init = fit_human_policy(target_angles, hand_movements, n_basis=16)

# Initialize environment and agent
env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=1)
participant = CursorControlLearner(
    alpha=0.1,
    alpha_nu=0.02,
    sigma=std_init,
    seed=1,
    baseline_decay=0.95,
    kappa=5,
    epsilon=0.4
)

# set initial policy of the agent based on the policy fitted to human behavior
participant.W = W_init.T.copy() / std_init
participant.nu = nu_init.copy()
participant.initialize_value_function(env, n_trials=1000)

# visualize intial policy
mu_init = plot_policy(participant)

# %% Run learning
n_trials = 2600 # number of trials to simulate

# set up dictionary to store simulation data
history = {
    'rewards': np.zeros(n_trials),
    'target_angles': np.zeros(n_trials),
    'actions': np.zeros((n_trials, 4)),
    'nus': np.zeros((n_trials, 4)),
    'Ws': np.zeros((n_trials, 4, participant.n_basis)),
    'Vs': np.zeros((n_trials, participant.n_basis)),
    'dir_errors': np.zeros(n_trials)
}

for t in range(n_trials):
    history['Vs'][t] = participant.V.copy()
    history['Ws'][t] = participant.W.copy()
    history['nus'][t]     = participant.nu.copy() 

    s = env.reset()
    a, mu, sigma, phi = participant.sample_action(s)
    _, r, _, info = env.step(a)
    participant.update(a, mu, sigma, phi, r)

    history['rewards'][t] = r
    history['target_angles'][t] = s
    history['actions'][t] = a
    history['dir_errors'][t] = info['directional_error']


# %% Visualize results

fig, axs = plt.subplots(3, 1, figsize=(4, 8), sharex=True)

bin_size = 60
rwd_binned, n_bins, _ = bin_data(history['rewards'], bin_size=bin_size)
bin_centers = bin_size*(np.arange(n_bins)+.5)

# Top panel: Rewards
axs[0].plot(bin_centers, rwd_binned, marker='o', label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_title("Learning performance")

action_labels = ['Lx', 'Ly', 'Rx', 'Ry']
for i in range(4):
    std_binned, _, _ = bin_data(np.exp(history['nus'][:, i]), bin_size=bin_size)
    axs[1].plot(bin_centers, std_binned, marker='o', label=action_labels[i])
axs[1].set_ylabel("Std Dev")
axs[1].set_xlabel("Trial")
axs[1].legend()

dir_errors_binned, _, _ = bin_data(np.rad2deg(np.abs(history['dir_errors'])), bin_size=bin_size)
axs[2].plot(bin_centers, dir_errors_binned, marker='o', label='|directional_error|',markersize=3)
axs[2].set_yticks([0, 30, 60, 90])
axs[2].set_xlabel("Trial")
axs[2].set_ylabel("Absolute Directional Error")

# %% bias/variance decomposition
bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(history['target_angles'], theta_errors=history['dir_errors'], window=60)

# %% compare to full learning curve for same participant

mat_full = loadmat('human_de_novo_errors.mat')
subjects = mat_full['subject_data'][0]


bias_vars_human_all = []
resid_vars_human_all = []
total_vars_human_all = []


bias_vars_human, resid_vars_human, total_vars_human = analyze_bias_variability_sliding(theta_targets=wrap_to_2pi(subjects[subj_id]['target_dir'].squeeze()*180/np.pi), theta_errors=subjects[subj_id]['directional_error'].squeeze(), window=60)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

axs[0].plot(total_vars, color='black', label='total squared error')
axs[0].plot(bias_vars, color='blue', label='squared error due to bias')
axs[0].plot(resid_vars, color='red', label = 'squared error due to variance')
axs[0].set_title("Model")
axs[0].legend()
axs[0].set_xlabel('Trial bin (60 trials/bin)')
axs[0].set_ylabel('Variance (°^2)')

axs[1].plot(total_vars_human, color="black", label='total squared error')
axs[1].plot(bias_vars_human, color="blue", label='squared error due to bias')
axs[1].plot(resid_vars_human, color="red", label = 'squared error due to variance')
axs[1].set_title("Human Learner")
axs[0].legend()
axs[0].set_xlabel('Trial bin (60 trials/bin)')

plt.tight_layout()
plt.show()

#%% plot convergence of policy in task space
viz = CursorLearningVisualizer(participant, env, history)

target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
viz.plot_snapshot_with_samples(trial_idx=0, n_samples=10)
plt.savefig("endpoints_early.svg", format="svg", bbox_inches='tight')

viz.plot_snapshot_with_samples(trial_idx=1000, n_samples=10)
plt.savefig("endpoints_mid.svg", format="svg", bbox_inches='tight')

viz.plot_snapshot_with_samples(trial_idx=2199, n_samples=10)
plt.savefig("endpoints_late.svg", format="svg", bbox_inches='tight')