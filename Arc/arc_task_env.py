#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 08:43:14 2025

@author: adrianhaith
"""

import numpy as np
from wrist_model import WristLDS  # your LDS class with simulate()
import matplotlib as plt

class ArcTaskEnv:
    def __init__(self, dt=0.01, radius=0.25, width=.08):
        self.dt = dt
        self.radius = radius
        self.width = width
        self.center = np.array([0.25, 0.0])  # center of semi-circle arc
        self.arc_theta = np.linspace(-np.pi/2, np.pi/2, 100)  # for visualization or interpolation

        self.arc_points = self.center.reshape(1, 2) + self.radius * np.stack([
            np.sin(self.arc_theta),
            -np.cos(self.arc_theta)
        ], axis=-1)  # shape (100, 2)

        self.lds = WristLDS(dt=self.dt)

        # Derived attributes
        self.Ng = int(self.lds.T / self.lds.goal_dt)  # number of subgoals
        self.reset()

    def reset(self):
        return None  # stateless environment

    def step(self, subgoals):
        """
        Simulates movement given a (2, Ng) array of subgoal positions.

        Returns:
            obs: None
            reward: negative cost
            done: True
            info: dict with full trajectory
        """
        #subgoals_mat = 
        x_traj, u_traj = self.lds.simulate(subgoals)
        #cursor_pos = x_traj[:, [self.lds.Nstates//2 - 1, self.lds.Nstates - 1]]  # extract x, y
        #cursor_pos = x_traj[:,[0, self.lds.Nstates//2-1]] # pull x and y position out of the full state

        cost, left_channel = self.compute_arc_cost(x_traj, u_traj)

        return None, -cost, True, {
            "trajectory": x_traj,
            "actions": u_traj,
            "subgoals": subgoals,
            "left_channel": left_channel
        }

    def compute_arc_cost(self, x_traj, u_traj):
        """
        Compute cost given cursor trajectory (pos) and full state trajectory (x_traj).

        Cost is sum of:
        1. Distance outside arc channel
        2. Final position error
        3. Final velocity magnitude
        4. Overall acceleration magnitude
        """
        # 1. Path deviation from arc centerline (outside channel)
        pos = x_traj[:,[0,5]]
        #ypos = x_traj[5,:]
        vel = x_traj[:,[1,6]]
        #yvel = x_traj[6,:]
        
        tangential_velocity = np.sqrt(np.sum(vel**2, axis=1))
        
        radial_distance_from_arc_origin = np.sqrt((self.radius-pos[:,0])**2 + pos[:,1]**2)
        radial_distance_from_arc_center = radial_distance_from_arc_origin-self.radius
        radial_pos_cost = np.sum(tangential_velocity * (radial_distance_from_arc_center**2)) * self.dt
        
        # variable to flag whether this trial left the channel or not
        left_channel = np.max(radial_distance_from_arc_center) > self.width / 2
        #radial_distance_from_channel_center = np.linalg.norm(pos - arc_path, axis=1)
        #outside_channel = np.maximum(0, radial_deviation - self.width / 2)
        #cost1 = np.sum(outside_channel) * self.dt

        # 2. Squared endpoint error
        channel_endpoint = np.array([0.5, 0.0])  # end of the arc
        final_pos = pos[-1,:]
        end_pos_cost = np.sum((final_pos - channel_endpoint)**2)

        # 3. Final velocity magnitude
        v_final = vel[-1,:]
        end_vel_cost = np.sum(v_final**2)

        # 4. Acceleration cost (derivative of velocity)
        accel = np.diff(vel, axis=0) / self.dt
        accel_cost = np.sum(accel**2) * self.dt

        # cost weights
        w_radial_pos = 500
        w_end_pos = 100
        w_end_vel = .5
        w_accel = .001
        total_cost = w_radial_pos*radial_pos_cost + w_end_pos*end_pos_cost + w_end_vel*end_vel_cost + w_accel*accel_cost
        return total_cost, left_channel

    def _nearest_point_on_arc(self, pos):
        """
        Given pos (N, 2), return closest points on the ideal arc path.
        """
        # Vector from center to each point
        r = pos - self.center
        norms = np.linalg.norm(r, axis=1, keepdims=True)
        unit_r = r / np.maximum(norms, 1e-8)
        arc_proj = self.center + self.radius * unit_r
        return arc_proj

def make_arc_subgoals(Ng, radius=0.25, center=(0.25, 0)):
    """
    Generates Ng subgoals evenly spaced along a semicircular arc.

    Returns:
        (2, Ng) numpy array of (x, y) coordinates
    """
    theta = np.linspace(-np.pi/2, np.pi/2, Ng)
    theta = np.append(theta,np.pi/2)
    
    cx, cy = center
    x = cx + radius * np.sin(theta[1:])
    y = cy + radius * np.cos(theta[1:])
    return np.stack([x, y])