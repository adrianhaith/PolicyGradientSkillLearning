#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:10:55 2025

@author: adrianhaith
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_arc_trials(env, learner, n_trials=5, color='C0', title="cursor trajectories"):
    """
    Simulate and plot several cursor trajectories using the current policy.

    Parameters:
        env: ArcTaskEnv instance
        learner: ArcLearner instance
        n_trials: number of trials to simulate
        color: matplotlib color for trajectories
    """
    # Arc parameters
    r = env.radius
    w = env.width
    center = env.center
    theta = np.linspace(-np.pi/2, np.pi/2, 300)


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')

    # Plot arc boundaries
    arc_outer = center[:, None] + (r + w/2) * np.vstack((np.sin(theta), np.cos(theta)))
    arc_inner = center[:, None] + (r - w/2) * np.vstack((np.sin(theta), np.cos(theta)))
    ax.plot(arc_outer[0], arc_outer[1], 'k-', linewidth=1)
    ax.plot(arc_inner[0], arc_inner[1], 'k-', linewidth=1)

    # Plot start and end circles
    for point in [np.array([0, 0]), np.array([0.5, 0])]:
        circle = plt.Circle(point, w/2, color='gray', fill=False, linestyle='-')
        ax.add_patch(circle)


    # get and plot mean goal locations
    mean_subgoals = learner.get_policy_mean()
    ax.plot(np.hstack(([0], mean_subgoals[0, :])) , np.hstack(([0], mean_subgoals[1,:])), linestyle='-', color='red', marker='o', linewidth=2, markersize=5, alpha=1)

    
    # Simulate and plot trials
    for _ in range(n_trials):
        subgoals = learner.sample_action()
        _, _, _, info = env.step(subgoals)
        traj = info["trajectory"]
        ax.plot(traj[:, 0], traj[:, 5], color=color, alpha=1)
        
    # now plot ellipses around the subgoal points to illustate policy variance
    n_goals = learner.Ng
    stds = learner.init_std * np.sqrt(np.exp(learner.nu))
    std_x = stds[:n_goals]
    std_y = stds[n_goals:]

    for i in range(n_goals):
        mu_x = mean_subgoals[0, i]
        mu_y = mean_subgoals[1, i]
        width = 2 * std_x[i]
        height = 2 * std_y[i]
        ellipse = Ellipse((mu_x, mu_y), width=width, height=height,
                          edgecolor='red', facecolor='none', linewidth=1, linestyle='-', alpha=1)
        ax.add_patch(ellipse)


    ax.set_aspect('equal')
    ax.set_xlim([-0.1, 0.65])
    ax.set_ylim([-.05, .35])
    ax.set_axis_off()
    plt.tight_layout()

