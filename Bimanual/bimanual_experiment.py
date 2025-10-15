
#%% Script to run multiple simulations and plot averaged learning curves etc.
# All simulations initialized to a different random policy

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import CursorControlEnv, CursorControlLearner
#from plotting import plot_value_function, plot_policy
from visualization import CursorLearningVisualizer
from utils import wrap_to_pi, wrap_to_2pi, analyze_bias_variability_sliding


def run_simulation(n_trials=2500, seed=0):
# simulate a single learning run

    np.random.seed(2*seed)

    env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=seed)
    learner = CursorControlLearner(
        alpha=0.1,
        alpha_nu=0.1,
        sigma=.05,
        seed=seed,
        baseline_decay=0.95,
        kappa=5,
        epsilon=0.4
    )

    learner.initialize_value_function(env, n_trials=1000)

    history = {
        'rewards': np.zeros(n_trials),
        'abs_dir_errors': np.zeros(n_trials),
        'nus': np.zeros((n_trials, 4)),
        'actions': np.zeros((n_trials,4)),
        'targ_dir': np.zeros(n_trials),
        'directional_error': np.zeros(n_trials)
    }

    for t in range(n_trials):
        s = env.reset()
        a, mu, sigma, phi = learner.sample_action(s)
        _, r, _, info = env.step(a)
        learner.update(a, mu, sigma, phi, r)

        history['rewards'][t] = r
        history['abs_dir_errors'][t] = info['abs_directional_error']
        history['nus'][t] = learner.nu.copy()
        history['actions'][t] = a
        history['targ_dir'][t] = s
        history['directional_error'][t] = info['directional_error']

    return history

def bin_data(array, bin_size=60):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size



# %% run multiple simulations
n_runs = 100        # number of runs to simulate
n_trials = 3000     # number of trials in each run
bin_size = 60       # bin size for smoothed plotting, bin size of 60 matches plots for empirical data

all_binned_rewards = []
all_binned_dir_errors = []

all_targ_dirs = np.zeros((n_runs,n_trials))
all_errors = np.zeros((n_runs,n_trials))

for run in range(n_runs):
    np.random.seed(run)
    hist = run_simulation(n_trials=n_trials, seed=run)
    rwd_binned, n_bins, _ = bin_data(hist['rewards'], bin_size=bin_size)
    dir_binned, _, _ = bin_data(np.rad2deg(wrap_to_pi(hist['abs_dir_errors'])), bin_size=bin_size)
    all_binned_rewards.append(rwd_binned)
    all_binned_dir_errors.append(dir_binned)

    all_errors[run]=hist['directional_error']
    all_targ_dirs[run] = hist['targ_dir']
    
    

    print("run ",run )

all_binned_rewards = np.stack(all_binned_rewards)         # shape (n_runs, n_bins)
all_binned_dir_errors = np.stack(all_binned_dir_errors)

# Compute mean and SEM
mean_rwd = np.mean(all_binned_rewards, axis=0)
sem_rwd = np.std(all_binned_rewards, axis=0) / np.sqrt(n_runs)
std_rwd = np.std(all_binned_rewards, axis=0)

mean_dir = np.mean(all_binned_dir_errors, axis=0)
sem_dir = np.std(all_binned_dir_errors, axis=0) / np.sqrt(n_runs)
std_dir = np.std(all_binned_dir_errors, axis=0)

bin_centers = bin_size * (np.arange(all_binned_rewards.shape[1]) + 0.5)

# %% plot outcomes

plt.figure(figsize=(4, 4))

plt.subplot(2,1,2)
plt.plot(bin_centers, mean_dir, label='|directional error|')
plt.fill_between(bin_centers, mean_dir - std_dir, mean_dir + std_dir, alpha=0.3)
plt.ylabel("Directional Error (deg)")
plt.xlabel("Trial")
plt.yticks([0, 30, 60, 90])
plt.tight_layout()
plt.savefig("averaged_learning_curve.svg", format="svg", bbox_inches='tight')
plt.grid(True)
plt.show()


# %% Now examine bias versus variance decomposition
from numpy.linalg import lstsq
from scipy.special import i0


bias_vars_all = []
resid_vars_all = []
total_vars_all = []
for run in range(n_runs):
    # estimate bias across angles and calculate residual error due to variance
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(theta_targets=all_targ_dirs[run], theta_errors=all_errors[run], window=60)
    bias_vars_all.append(bias_vars)
    resid_vars_all.append(resid_vars)
    total_vars_all.append(total_vars)
# %
scaling = 1#180*180/(np.pi**2)
plt.plot(scaling*np.mean(total_vars_all, axis=0), color="black", label='total squared error')
plt.plot(scaling*np.mean(bias_vars_all, axis=0), color="blue", label='squared error due to bias')
plt.plot(scaling*np.mean(resid_vars_all, axis=0), color="red", label = 'squared error due to variance')
plt.grid(True)
plt.xlabel('Trial bin (60 trials/bin')
plt.ylabel('Contributed variance')
plt.title('Bias-Variance Decomposition for policy-gradient RL model')
plt.legend()

# %% load human data for comparison
from scipy.io import loadmat

mat = loadmat('human_de_novo_errors.mat')
subjects = mat['subject_data'][0]


bias_vars_human_all = []
resid_vars_human_all = []
total_vars_human_all = []

for subj in range(13):
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(theta_targets=wrap_to_2pi(subjects[subj]['target_dir'].squeeze()*180/np.pi), theta_errors=subjects[subj]['directional_error'].squeeze(), window=60)

    bias_vars_human_all.append(bias_vars)
    resid_vars_human_all.append(resid_vars)
    total_vars_human_all.append(total_vars)

plt.plot(np.mean(total_vars_human_all, axis=0), color="black", label='total squared error')
plt.plot(np.mean(bias_vars_human_all, axis=0), color="blue", label='squared error due to bias')
plt.plot(np.mean(resid_vars_human_all, axis=0), color="red", label = 'squared error due to variance')
plt.grid(True)
plt.xlabel('Trial bin (60 trials/bin')
plt.ylabel('Contributed variance')
plt.title('Bias-Variance Decomposition for Human Learners')
plt.legend()
