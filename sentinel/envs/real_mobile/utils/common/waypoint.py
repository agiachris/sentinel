"""
Basic trajectory utilities.

Authors: @yjy0625, @contactrika

"""

import numpy as np

from scipy.interpolate import interp1d


def interpolate_waypoints(waypts, num_steps, curr_pos=None, interp_mode="linear"):
    if curr_pos is not None:
        waypts = np.concatenate((curr_pos.reshape(1, -1), waypts), axis=0)
    ids = np.arange(waypts.shape[0])
    interp_i = []
    for i, num_step in enumerate(num_steps):
        interp_i.append(np.linspace(i, i + 1, num_step, endpoint=False))
    interp_i = np.concatenate(interp_i)
    interp_args = {
        "x": ids,
        "kind": interp_mode,  # linear, quadratic, cubic
        "fill_value": "extrapolate",
    }
    parts = []
    for i in range(waypts.shape[1]):
        parts.append(interp1d(y=waypts[:, i], **interp_args)(interp_i))
    traj = np.array(parts).T
    return traj


def make_linear_waypoints(
    n_steps, start_pos=(0.58, 0.0, 0.43), end_pos=(0.6, 0.0, 0.4)
):
    start_pos, end_pos = np.array(start_pos), np.array(end_pos)
    all_waypoints = np.zeros((n_steps, 3))
    for i in range(n_steps):
        all_waypoints[i] = (end_pos - start_pos) * (i / n_steps) + start_pos
    return all_waypoints


def make_circular_waypoints(n_steps, center=(0.7, 0.0, 0.4), radius=0.1, n_loops=1):
    all_waypoints = np.zeros((n_loops * n_steps, 3))
    for l in range(n_loops):
        for i in range(n_steps):
            theta = -2 * np.pi * i / n_steps
            x = radius * np.cos(theta + np.pi) + center[0]
            y = radius * np.sin(theta + np.pi) + center[1]
            all_waypoints[l * n_steps + i] = np.array((x, y, center[2]))
    return all_waypoints


def make_flip_waypoints():
    # position: stay for T/2 steps, move back 0.1, then move forward 0.1
    # ang: tilt down, then up, then down again slowly, motion in sin curve
    # reference poses:
    #   upright ang [90 0 90]
    #   tilt down ang [135 - -]
    #   tilt up ang [45 - -]
    #   end effector rotate 90 deg [- 90 -]
    #   near point pos [0.6 0 0.3]
    #   far point pos [0.95 0 0.3]
    T = 200

    traj_center = np.array([0.45, 0, 0.4])
    traj_r = 0.2

    init_pt, high_pt, return_pt = -20.0, 20.0, -5.0
    tilt_angles = np.concatenate(
        [np.linspace(init_pt, high_pt, 100), np.linspace(high_pt, return_pt, 100)]
    )

    pos_x = traj_center[0] + traj_r * np.cos(np.deg2rad(tilt_angles))
    pos_y = np.array([0.0] * T)
    pos_z = traj_center[2] + traj_r * np.sin(np.deg2rad(tilt_angles))
    positions = np.c_[pos_x, pos_y, pos_z]
    positions[T // 4 * 3 :, 0] += 0.05 * np.linspace(0, 1, T - T // 4 * 3)
    waypoints = np.c_[positions, 90.0 - tilt_angles, np.array([[0.0, 90.0]] * T)]
    return waypoints[::5]
