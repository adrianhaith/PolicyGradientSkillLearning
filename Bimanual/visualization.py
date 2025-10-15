import matplotlib.pyplot as plt
import numpy as np

class CursorLearningVisualizer:
    def __init__(self, learner, env, history, cmap=plt.cm.hsv):
        self.learner = learner
        self.env = env
        self.history = history
        self.n_basis = learner.n_basis
        self.n_trials = np.shape(history['actions'])[0]
        self.radius = env.radius
        self.target_angles = env.target_angles
        self.cmap = cmap

    def angle_to_color(self, angle):
        return self.cmap((angle % (2 * np.pi)) / (2 * np.pi))

    def plot_snapshot(self, trial_idx):
        fig, ax = plt.subplots(figsize=(4, 4))

        # Extract policy + value weights for this trial
        W = self.history['Ws'][trial_idx]
        V = self.history['Vs'][trial_idx]

        # Trial-specific info
        trial_angle = self.history['target_angles'][trial_idx]
        trial_action = self.history['actions'][trial_idx]
        cursor_endpoint = [trial_action[1], trial_action[2]]  # Ly, Rx

        # Plot central start position
        ax.plot(0, 0, 'ko', markersize=4)

        for angle in self.target_angles:
            phi = self.learner.compute_basis(angle)
            mu = W @ phi         # mean action
            val = V @ phi        # estimated value

            # Task-space coordinates
            cursor_xy = [mu[1], mu[2]]
            target_xy = self.radius * np.array([np.cos(angle), np.sin(angle)])
            color = self.angle_to_color(angle)

            # Target location
            ax.plot(*target_xy, 'o', color=color, markersize=6)

            # Value function visualized as circle around target
            circle = plt.Circle(target_xy, radius=abs(val), color=color, alpha=0.2)
            ax.add_patch(circle)

            # Mean action (in task space)
            ax.plot(*cursor_xy, 'x', color=color, markersize=4)

        # Actual sampled action on this trial (as green dot)
        ax.plot(cursor_endpoint[0], cursor_endpoint[1], 'o', color=self.angle_to_color(self.history['target_angles'][trial_idx]), markersize=4, label='Sampled')
        #ax.plot(cursor_endpoint[0], cursor_endpoint[1], 'o', color='black', markersize=4, label='Sampled')

        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_title(f"Trial {trial_idx}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_snapshot_with_samples(self, trial_idx, target_angles=None, n_samples=1):
        """
        Plot policy snapshot at a given trial, and simulate sampled actions
        for a given set of target angles.

        Parameters:
            trial_idx: int, trial index to pull weights from
            target_angles: list of angles to probe (default: canonical 8)
            n_samples: number of actions to sample per target
        """
        if target_angles is None:
            target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        W = self.history['Ws'][trial_idx]
        nu = self.history['nus'][trial_idx]
        sigma = self.learner.init_std * np.exp(nu)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(0, 0, 'ko', markersize=4)  # start position

        for angle in target_angles:
            phi = self.learner.compute_basis(angle)
            mu_norm = W @ phi
            mu = self.learner.init_std * mu_norm

            # Plot mean action in task space (Ly → x, Rx → y)
            cursor_xy = [mu[1], mu[2]]
            target_xy = self.env.radius * np.array([np.cos(angle), np.sin(angle)])
            color = self.angle_to_color(angle)

            ax.plot(*target_xy, 'o', color=color, markersize=15)
            #ax.plot(*cursor_xy, 'x', color=color, markersize=5, label='Mean')

            # Sample and plot actions
            for _ in range(n_samples):
                a_sample = self.learner.rng.normal(mu, sigma)
                a_cursor = [a_sample[1], a_sample[2]]
                ax.plot(*a_cursor, 'o', color=color, alpha=1, markersize=4)

        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.axis('off')
        ax.set_title(f"Policy Snapshot @ Trial {trial_idx}")
        plt.tight_layout()


    def plot_value_function(self, trial_idx=None, n_points=200):
        """
        Plot the baseline value function b(s) = Vᵀ * phi(s)
        for a given trial (or current learner state if trial_idx is None).
        """
        angles = np.linspace(0, 2 * np.pi, n_points)

        current_angle = self.history['target_angles'][trial_idx]
        current_reward = self.history['rewards'][trial_idx]

        # Get weights
        if trial_idx is None:
            V = self.learner.V
        else:
            V = self.history['Vs'][trial_idx]

        values = np.array([V @ self.learner.compute_basis(s) for s in angles])

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(angles, values, label='Baseline b(s)')
        ax.plot(current_angle, current_reward, 'o', color='orange')

        ax.set_xlabel("Target angle (radians)")
        ax.set_ylabel("Estimated reward baseline")
        ax.set_title(f"Value Function at trial {trial_idx}" if trial_idx is not None else "Current Value Function")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


    def plot_policy_update(self, learner, trial_idx, r=0.12, n_points=200):
        """
        Plot the pre- vs post-update policy around a given trial.
        """
        W_pre = self.history['Ws'][trial_idx]
        W_post = self.history['Ws'][trial_idx+1]
        target_angle = self.history['target_angles'][trial_idx]
        action = self.history['actions'][trial_idx]

        angles = np.linspace(0, 2 * np.pi, n_points)
        phi_matrix = np.array([self.learner.compute_basis(s) for s in angles])

        mu_pre = learner.init_std * phi_matrix @ W_pre.T
        mu_post = learner.init_std * phi_matrix @ W_post.T

        ideal_Ly = r * np.cos(angles)
        ideal_Rx = r * np.sin(angles)

        target_xy = r * np.array([np.cos(target_angle), np.sin(target_angle)])
        cursor_xy = np.array([action[1], action[2]])

        fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

        # Left hand y (cursor x)
        axs[0].plot(angles, mu_pre[:, 1], label='Pre-update', color='gray')
        axs[0].plot(angles, mu_post[:, 1], label='Post-update', color='blue')
        axs[0].plot(angles, ideal_Ly, '--', label='Ideal', color='black')
        axs[0].axvline(target_angle, color='red', linestyle=':')
        axs[0].scatter([target_angle], [target_xy[0]], color='red', label='Target X', zorder=5)
        axs[0].scatter([target_angle], [cursor_xy[0]], color='green', label='Cursor X', zorder=5)
        axs[0].set_title("Left Hand Y (Ly)")
        axs[0].set_ylabel("Ly displacement (cursor x)")
        axs[0].grid(True)

        # Right hand x (cursor y)
        axs[1].plot(angles, mu_pre[:, 2], label='Pre-update', color='gray')
        axs[1].plot(angles, mu_post[:, 2], label='Post-update', color='blue')
        axs[1].plot(angles, ideal_Rx, '--', label='Ideal', color='black')
        axs[1].axvline(target_angle, color='red', linestyle=':')
        axs[1].scatter([target_angle], [target_xy[1]], color='red', label='Target Y', zorder=5)
        axs[1].scatter([target_angle], [cursor_xy[1]], color='green', label='Cursor Y', zorder=5)
        axs[1].set_title("Right Hand X (Rx)")
        axs[1].set_ylabel("Rx displacement (cursor y)")
        axs[1].grid(True)

        for ax in axs:
            ax.set_xlabel("Target angle (radians)")
            ax.set_xlim(0, 2 * np.pi)

        plt.suptitle(f"Policy Before and After Update (Trial {trial_idx})")
        plt.tight_layout()
        plt.show()

    

    def plot_learning_progress(self, starts=[0, 500, 1000], window=200, cmap=plt.cm.hsv):
        """
        Plots snapshots of learning progress over fixed trial windows.
        
        Parameters:
            starts: list of starting trial indices
            window: number of trials to include in each snapshot
            cmap: colormap for angle-to-color mapping
        """
        n_panels = len(starts)
        target_angles = self.history['target_angles']
        actions = self.history['actions']

        # Canonical targets for visual context
        canonical_angles = np.linspace(0, 2 * np.pi, len(np.unique(target_angles)), endpoint=False)
        target_locs = np.stack([np.cos(canonical_angles), np.sin(canonical_angles)], axis=1) * self.env.radius

        def angle_to_color(angle):
            return cmap((angle % (2 * np.pi)) / (2 * np.pi))

        fig, axs = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
        
        if n_panels == 1:
            axs = [axs]  # make iterable even for single plot

        for i, start in enumerate(starts):
            ax = axs[i]
            trial_indices = np.arange(start, min(start + window, self.n_trials))
            ax.set_title(f"Trials {start}–{start + window}")

            # Plot center
            ax.plot(0, 0, 'ko', markersize=3)

            # Plot canonical target positions
            for angle, (x, y) in zip(canonical_angles, target_locs):
                ax.plot(x, y, 'o', color=angle_to_color(angle), markersize=6)

            # Plot cursor endpoints for this window
            for t in trial_indices:
                angle = target_angles[t]
                action = actions[t]
                endpoint = [action[1], action[2]]  # Ly, Rx
                ax.plot(*endpoint, 'o', color=angle_to_color(angle), markersize=2, alpha=0.6)

            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout()
        plt.show()