import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_policy_snapshot(ax, mu, cov, color, n_samples=25, show_ellipse=True, label=None):
    """
    Plot a policy snapshot (samples + optional ellipse) on the given axes.

    Parameters:
    - ax: matplotlib Axes
    - mu: mean of policy (2,)
    - nu: log-eigenvalues (2,)
    - phi: rotation angle (scalar)
    - color: color to use for this snapshot
    - n_samples: number of action samples to draw
    - show_ellipse: whether to plot the 1 SD ellipse
    - label: optional label for the scatter points
    """
    # Sample actions
    samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    ax.scatter(samples[:, 0], samples[:, 1], s=25, color=color, label=label)

    if show_ellipse:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        chisq_val = 5.991  # from chi^2 with 2 degrees of freedom - appropriate for plotting a 95% confidence interval for the covariance ellipse
        width, height = 2 * np.sqrt(chisq_val * eigvals)
        ellipse = Ellipse(mu, width, height, angle=angle, edgecolor=color,
                          facecolor='none', lw=2)
        ax.add_patch(ellipse)