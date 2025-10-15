# script to compare policy-gradient RL model with human behavior after being initialized to the same initial policy

import numpy as np
from scipy.io import loadmat
from numpy.linalg import lstsq
from models import CursorControlLearner
from models import CursorControlEnv
from utils import compute_von_mises_basis, wrap_to_pi, wrap_to_2pi, ridge_fit, bin_data, analyze_bias_variability_sliding, fit_human_policy, plot_kde
import matplotlib.pyplot as plt
from visualization import CursorLearningVisualizer
from pathlib import Path

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

# function to simulate learning that is initialized to mimic a human suject's initial policy
def simulate_learning_for_subject(subj_id, seed, n_trials=2600, window=60, n_basis=16):
    # Load subject data
    DATA_PATH = Path('human_data.mat').parent / 'human_data.mat'
    #DATA_PATH = Path(__file__).parent / 'human_data.mat'
    mat = loadmat(DATA_PATH)
    subj_data = mat['human_data'][0][subj_id]
    target_angles = wrap_to_2pi(np.pi/2 - subj_data['target_angles'].squeeze())
    hand_movements = subj_data['hand_movements'].T

    # Fit initial policy
    W_init, std_init, nu_init = fit_human_policy(target_angles, hand_movements, n_basis=n_basis)

    # Set up environment and agent with seed
    env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=seed)
    alpha = 0.1 # learning rate
    learner = CursorControlLearner(alpha=alpha, alpha_nu=alpha/5, sigma=std_init, seed=seed,
                                   baseline_decay=0.95, kappa=5, epsilon=0.3)
    learner.W = W_init.T.copy() / std_init
    learner.nu = nu_init.copy()
    learner.initialize_value_function(env, n_trials=1000)

    # Run learning simulation of initialized model
    dir_errors = np.zeros(n_trials)
    target_dirs = np.zeros(n_trials)
    reach_dirs = np.zeros(n_trials)
    reach_errors = np.zeros(n_trials)
    for t in range(n_trials):
        s = env.reset()
        a, mu, sigma, phi = learner.sample_action(s)
        _, reward, _, info = env.step(a)
        learner.update(a, mu, sigma, phi, reward)

        dir_errors[t] = info['directional_error'].copy()
        target_dirs[t] = s

    # Analyze bias/variance decomposition
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(
        target_dirs, theta_errors=dir_errors, window=window
    )
    abs_errors_binned, _, _ = bin_data(np.abs(dir_errors), window)

    return dir_errors, bias_vars, resid_vars, total_vars


# script to average multiple runs for a single subject
def analyze_subject_group_model(subj_id, n_repeats=10, seed_offset=0):
    all_bias = []
    all_resid = []
    all_total = []
    all_dir_errs = []

    for i in range(n_repeats):
        seed = seed_offset + i
        dir_errors, bias, resid, total = simulate_learning_for_subject(subj_id, seed)
        all_bias.append(bias.copy())
        all_resid.append(resid.copy())
        all_total.append(total.copy())
        all_dir_errs.append(dir_errors.copy())

    

    return {
        'bias': np.mean(all_bias, axis=0),
        'resid': np.mean(all_resid, axis=0),
        'total': np.mean(all_total, axis=0),
        'all_abs_dir_errors': np.mean(np.abs(all_dir_errs), axis=0)
    }



#---- main part of the script: iterate over subjects and run a batch of simulations per subject. Average learning curves
#------------------

n_repeats = 100 # number of repeat simulations for each participant-initialized model

model_group_bias = []
model_group_resid = []
model_group_total = []
model_abs_dir_errors = []
model_dir_errors = []

human_group_bias = []
human_group_resid = []
human_group_total = []
human_group_abs_errors =[]

# list of subject ids to include in this analysis
# this list omits participants for whom fitted initial policy did not match their behavior well
included_subjects = [0,4,5,7,8,9,10,11,12] 

for subj_id in included_subjects:
    # Simulated model learner
    sim_results = analyze_subject_group_model(subj_id, n_repeats=n_repeats)

    # Human data
    mat = loadmat('human_de_novo_errors.mat')
    subj = mat['subject_data'][0][subj_id]
    bias_human, resid_human, total_human = analyze_bias_variability_sliding(
        wrap_to_2pi(subj['target_dir'].squeeze()*180/np.pi),
        theta_errors=subj['directional_error'].squeeze(),
        window=60
    )

    # Save for group average
    model_group_bias.append(sim_results['bias'])
    model_group_resid.append(sim_results['resid'])
    model_group_total.append(sim_results['total'])
    model_abs_dir_errors.append(sim_results['all_abs_dir_errors'])

    human_group_bias.append(bias_human)
    human_group_resid.append(resid_human)
    human_group_total.append(total_human)
    abs_dir_errors, _, _ = bin_data(np.abs(subj['directional_error']),bin_size=60)
    human_group_abs_errors.append(abs_dir_errors)

#%% Average across subjects
model_mean_bias = np.mean(model_group_bias, axis=0)
model_mean_resid = np.mean(model_group_resid, axis=0)
model_mean_total = np.mean(model_group_total, axis=0)
human_mean_bias = np.mean(human_group_bias, axis=0)
human_mean_resid = np.mean(human_group_resid, axis=0)
human_mean_total = np.mean(human_group_total, axis=0)

n_subjects = np.size(included_subjects)
human_bias_sem = np.std(human_group_bias, axis=0) / np.sqrt(n_subjects)
human_resid_sem = np.std(human_group_resid, axis=0) / np.sqrt(n_subjects)
human_total_sem = np.std(human_group_total, axis=0) / np.sqrt(n_subjects)
model_bias_sem = np.std(model_group_bias, axis=0) / np.sqrt(n_subjects)
model_resid_sem = np.std(model_group_resid, axis=0) / np.sqrt(n_subjects)
model_total_sem = np.std(model_group_total, axis=0) / np.sqrt(n_subjects)

human_mean_abs_errors = np.mean(human_group_abs_errors, axis=0)
human_sem_abs_errors = np.mean(human_group_abs_errors, axis=0)

model_mean_abs_dir_errors = np.mean(model_abs_dir_errors, axis=0)
model_sem_abs_dir_errors = np.std(model_abs_dir_errors, axis=0) / np.sqrt(n_subjects)

# %% series 
for subj in range(np.size(included_subjects)):
    fig, axs = plt.subplots(1, 2, figsize=(8, 2), sharey=True)

    axs[0].plot(human_group_bias[subj], color='blue', label='bias')
    axs[0].plot(human_group_resid[subj], color='red', label = 'variance')
    axs[0].plot(human_group_total[subj], color='black', label='total')
    axs[0].set_title(f"Human Learners - Subj {subj+1}")
    axs[0].set_xlabel('Trial Blocks (60 trials/block)')
    axs[0].set_ylabel('Variance (°^2)')
    axs[0].legend()

    axs[1].plot(model_group_bias[subj], color='blue',ls='-', label='bias')
    axs[1].plot(model_group_resid[subj], color='red',ls='-', label = 'variance')
    axs[1].plot(model_group_total[subj], color='black',ls='-', label='total')
    axs[1].set_xlabel('Trial Blocks (60 trials/block)')
    axs[1].set_title(f"Policy-Gradient RL - Subj {subj+1}")

# %% Plot All subjects
fig, axs = plt.subplots(1, 2, figsize=(9, 2), sharey=True)

# x-axis points
window_size = 60
n_windows = np.shape(human_mean_bias)[0]
window_centers = window_size * (np.arange(0,n_windows)+.5)

axs[0].fill_between(window_centers, human_mean_bias - human_bias_sem, human_mean_bias + human_bias_sem, color='blue', alpha=.3)
axs[0].plot(window_centers,human_mean_bias, color='blue', label='bias')
axs[0].fill_between(window_centers, human_mean_resid - human_resid_sem, human_mean_resid + human_resid_sem, color='red', alpha=.3)
axs[0].plot(window_centers,human_mean_resid, color='red', label='variance')
axs[0].set_ylim(0,3500)
axs[0].set_xlabel('Trial Blocks (60 trials/block)')
axs[0].set_ylabel('Variance (°^2)')
axs[0].legend()

n_windows = np.shape(model_mean_bias)[0]
window_centers = window_size * (np.arange(0,n_windows)+.5)
axs[1].fill_between(window_centers, model_mean_bias - model_bias_sem, model_mean_bias + model_bias_sem, color='blue', alpha=.3)
axs[1].plot(window_centers,model_mean_bias, color='blue',ls='-', label='bias')
axs[1].fill_between(window_centers, model_mean_resid - model_resid_sem, model_mean_resid + model_resid_sem, color='red', alpha=.3)
axs[1].plot(window_centers,model_mean_resid, color='red',ls='-', label='variance')
axs[1].set_xlabel('Trial Blocks (60 trials/block)')

plt.savefig("learning_curves.svg", format="svg", bbox_inches='tight')

plt.show()


# %% plot absolute directional error

plt.figure(figsize=(3,2))
abs_err_all = []
for subj in range(np.size(included_subjects)):
    abs_err, _, _ = bin_data(np.rad2deg(np.abs(model_abs_dir_errors[subj])))
    abs_err_all.append(abs_err)
    #plt.plot(window_centers, abs_err)

abs_err_sem = np.std(abs_err_all, axis=0) / np.sqrt(n_subjects)
abs_err_mean = np.mean(abs_err_all, axis=0)
plt.fill_between(window_centers, abs_err_mean - abs_err_sem, abs_err_mean + abs_err_sem, color='blue', alpha=.3)
plt.plot(window_centers, abs_err_mean, linewidth=3)
plt.ylim(0, 90)
plt.xlabel('Trials')
plt.ylabel('Absolute Dir. Error (°)')

# save figure
plt.savefig("learning_curve_abs_err_initialized_opt.svg", format="svg", bbox_inches='tight')


