# %%
# script to inspect human initial policies and compare with fitted policies used to initialize RL simulations
import numpy as np
from scipy.io import loadmat
from numpy.linalg import lstsq
from models import CursorControlLearner
from models import CursorControlEnv
from utils import compute_von_mises_basis, wrap_to_pi, wrap_to_2pi, fit_human_policy
import matplotlib.pyplot as plt
from visualization import CursorLearningVisualizer

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

def visualize_policy(W, target_angles, actions):
    # compare actions and fitted policies
    n_trials = np.size(target_angles[0])

    fig, axs = plt.subplots(1,2,figsize=(8,4),gridspec_kw={'width_ratios': [2, 1]})
    theta_grid = np.linspace(0, 2*np.pi, 500)
    Phi_grid = compute_von_mises_basis(theta_grid, n_basis=16, kappa=5)
    predicted_action = Phi_grid @ W  # shape (500,)
    angles_grid= np.linspace(0, 2 * np.pi, 500)
    axs[0].plot(np.rad2deg(angles_grid), predicted_action[:,1],'-',color='blue')
    axs[0].plot(np.rad2deg(angles_grid), predicted_action[:,2],'-',color='orange')
    axs[0].plot(np.rad2deg(target_angles), actions[:,1],'o',color='blue')
    axs[0].plot(np.rad2deg(target_angles), actions[:,2],'o',color='orange')   


    # plot ideal policy
    r = 0.12
    ideal_Ly = r * np.cos(angles_grid)
    ideal_Rx = r * np.sin(angles_grid)
    axs[0].plot(np.rad2deg(angles_grid),ideal_Ly,':',color='blue')
    axs[0].plot(np.rad2deg(angles_grid),ideal_Rx,':',color='orange')

    axs[0].grid(True)
    axs[0].set_ylim(-.2, .2)
    axs[0].set_xticks((0,45,90,135,180,225, 270, 315, 360))
    # plot actions in 2d, color-coded by target direction
    # create colormap and function to convert target angle into a color
    cmap=plt.cm.hsv
    def angle_to_color(angle):
        return cmap((angle % (2 * np.pi)) / (2 * np.pi))

    # plot targets
    r=0.12
    for angle in angles_grid:
        axs[1].plot(r*np.cos(angle), r*np.sin(angle), 'o', color=angle_to_color(angle), markersize=2)

    for i in range(n_trials):
        axs[1].plot(actions[i,1],actions[i,2], 'o', color=angle_to_color(target_angles[i]))

    axs[1].plot(0,0,'k.')
    #axs[1].axis('off')
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(-.3, .3)
    axs[1].set_ylim(-.3, .3)

    # calculate predicted actions at target angles for data
    Phi_data = compute_von_mises_basis(target_angles, n_basis=16, kappa=5)
    predicted_actions_data = Phi_data @ W_init
    residuals = actions - predicted_actions_data

    return fig


# Load human data from .mat file
# NB - code needed here depends on whether or not this is being run in Jupyter
from pathlib import Path
DATA_PATH = Path('human_data_120.mat').parent / 'human_data.mat'
#DATA_PATH = Path(__file__).parent / 'human_data.mat'     # <---use this line if not in Jupyter
mat = loadmat(DATA_PATH)
subj_id = 6 # id number of subject (0-12)
subj_data = mat['human_data'][0][subj_id]  # adjust indexing as needed

target_angles = wrap_to_2pi(np.pi/2 - subj_data['target_angles'].squeeze())  # (n_trials,)
hand_movements = subj_data['hand_movements'].T  # (n_trials, 2)
cursor_angle = np.atan2(hand_movements[:,2],hand_movements[:,1])

# Fit initial policy
W_init, std_init, nu_init = fit_human_policy(target_angles, hand_movements, n_basis=16)

# Initialize environment and agent
env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=1)
participant = CursorControlLearner(
    alpha=0.1,
    alpha_nu=0.1,
    sigma=std_init,
    seed=1,
    baseline_decay=0.95,
    kappa=5,
    epsilon=0.4
)

# initial policy
participant.W = W_init.T.copy() / std_init
participant.nu = nu_init.copy()

participant.initialize_value_function(env, n_trials=1000)

print(" ")
# %% compare initial human behavior to behavior of initialized policy

n_trials_sim = np.size(target_angles)

actions = np.zeros((n_trials_sim,4))
all_actions = []
all_target_angles = []
for k in range(1):
    for i, s in enumerate(target_angles):
        a, _, _, _ = participant.sample_action(s)
        _, _, _, info = env.step(a)
        actions[i] = a
        e = info['directional_error'].copy()
    all_actions.extend(actions.copy())
    all_target_angles.extend(target_angles.copy())    

all_actions = np.array(all_actions)
all_target_angles = np.array(all_target_angles)

fig = visualize_policy(W_init, target_angles, hand_movements)
fig.savefig('initial_policy.svg', format='svg', bbox_inches='tight')
plt.title("Human")

#%%
fig = visualize_policy(W_init, all_target_angles, all_actions)
plt.title("Model")

# %% Train model and re-plot policy
n_trials = 2200
for t in range(n_trials):
        ss = env.reset()
        aa, mu, sigma, phi = participant.sample_action(ss)
        _, reward, _, info = env.step(aa)
        participant.update(aa, mu, sigma, phi, reward)

WW = participant.W.copy()
W_final = WW.T

fig = visualize_policy(W_final * std_init, target_angles, hand_movements)
fig.savefig('late_policy.svg', format='svg', bbox_inches='tight')

