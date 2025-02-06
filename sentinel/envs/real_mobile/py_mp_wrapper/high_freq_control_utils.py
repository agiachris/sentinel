import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp

import numpy as np
import numpy as np

import numpy as np
from scipy.interpolate import PchipInterpolator


def generate_arm_high_freq_traj(
    target_eef_traj,
    gripper_status_traj,
    step_interval_traj,
    arm_control_freq=125,
    re_assign_time=True,
):
    """
    target_eef_traj: np.array, shape=(n_waypoints, 6), dtype=float
    gripper_status_traj: np.array, shape=(n_waypoints, 1), dtype=float
    step_interval_traj: np.array, shape=(n_waypoints-1, 1), dtype=float
    arm_control_freq: int
    """

    # print("target_eef_traj: ", target_eef_traj)
    # print("gripper_status_traj: ", gripper_status_traj)
    # print("step_interval_traj: ", step_interval_traj)

    print("target_eef_traj.shape: ", target_eef_traj.shape)
    print("gripper_status_traj.shape: ", gripper_status_traj.shape)
    print("step_interval_traj.shape: ", step_interval_traj.shape)
    print("arm_control_freq: ", arm_control_freq)

    # [0] Extract the spatial and orientation components
    # (ac_horizon + 1, 3)
    posi_traj = target_eef_traj[:, :3]
    ori_traj = target_eef_traj[:, 3:]  # axis-angle in degrees

    # [1] Initialize the trajectory
    t_total = np.sum(step_interval_traj)
    num_pts = int(t_total * arm_control_freq)
    interpolated_times = np.linspace(0, t_total, num_pts)

    # [2] Interpolate the positions using PCHIP
    # (ac_horizon + 1, )
    time_traj = np.zeros(len(target_eef_traj))
    # (ac_horizon,)
    dist_traj = np.linalg.norm(np.diff(posi_traj, axis=0), axis=1)
    dist_traj = dist_traj + 1e-2  # Avoid zero division
    if re_assign_time:
        average_speed = (np.sum(dist_traj)) / t_total
        time_traj[1:] = np.cumsum(
            dist_traj.reshape(
                -1,
            )
            / average_speed
        )
    else:
        time_traj[1:] = np.cumsum(
            step_interval_traj.reshape(
                -1,
            )
        )

    print(
        f"time_traj: {time_traj}, posi_traj: {posi_traj}, num_pts: {num_pts}, t_total: {t_total}, arm_control_freq: {arm_control_freq}, interpolated_times: {interpolated_times}"
    )
    posi_interpolator = PchipInterpolator(time_traj, posi_traj, axis=0)
    high_freq_posi_traj = posi_interpolator(interpolated_times)

    # [3] Interpolate the orientations via simple linear interpolation
    ori_interpolator = PchipInterpolator(time_traj, ori_traj, axis=0)
    high_freq_ori_traj = ori_interpolator(interpolated_times)

    # [4] Combine the interpolated positions and orientations
    high_freq_pose_traj = np.hstack((high_freq_posi_traj, high_freq_ori_traj))

    # [5] Track which interpolated positions correspond to the waypoints
    # e.g. kpt_high_freq_indices = array([  0,  30,  50,  71, 100, 129, 150, 170, 199])
    kpt_high_freq_indices = np.searchsorted(interpolated_times, time_traj)

    # Ensure indices are within bounds
    kpt_high_freq_indices = np.clip(
        kpt_high_freq_indices, 0, len(high_freq_posi_traj) - 1
    )

    kpt_high_freq_posi_traj = high_freq_posi_traj[
        kpt_high_freq_indices
    ]  # (ac_horizon+1, 3)

    # Find the time points where gripper status changes
    # e.g. gripper_switch_ac_idxs = [1, 4, 6]
    gripper_switch_ac_idxs = [
        i
        for i in range(1, len(gripper_status_traj))
        if gripper_status_traj[i] != gripper_status_traj[i - 1]
    ]
    gripper_switch_kpt_high_freq_indices = [
        kpt_high_freq_indices[i] for i in gripper_switch_ac_idxs
    ]
    gripper_switch_kpt_posi_list = [
        kpt_high_freq_posi_traj[i] for i in gripper_switch_ac_idxs
    ]

    info = dict(
        # posi_traj=posi_traj,
        # ori_traj=ori_traj,
        # interpolated_times=interpolated_times,
        high_freq_pose_traj=high_freq_pose_traj,
        kpt_high_freq_indices=kpt_high_freq_indices,
        gripper_switch_ac_idxs=gripper_switch_ac_idxs,
        gripper_switch_kpt_high_freq_indices=gripper_switch_kpt_high_freq_indices,
        gripper_switch_kpt_posi_list=gripper_switch_kpt_posi_list,
    )
    return info


# Interpolate the orientations using slerp with PD control
def slerp_interpolation(times, orientations, num_points):
    rotations = R.from_quat(orientations)
    slerp = Slerp(times, rotations)
    interpolated_times = np.linspace(times[0], times[-1], num_points)
    interpolated_rotations = slerp(interpolated_times)

    # PD control parameters
    Kp = 1.0  # Proportional gain
    Kd = 0.1  # Derivative gain
    prev_error = np.zeros(3)

    for i in range(1, len(interpolated_rotations)):
        delta_time = interpolated_times[i] - interpolated_times[i - 1]
        error = (
            interpolated_rotations[i].as_rotvec()
            - interpolated_rotations[i - 1].as_rotvec()
        )
        derivative = (error - prev_error) / delta_time
        correction = Kp * error + Kd * derivative
        interpolated_rotations[i] = R.from_rotvec(
            interpolated_rotations[i - 1].as_rotvec() + correction
        )
        prev_error = error

    return interpolated_rotations


def generate_smooth_traj(waypoints, ac_intervals, gripper_status_seq=None, freq=100):
    """
    waypoints: np.array, shape=(n_waypoints, 7), dtype=float
    ac_intervals: np.array, shape=(n_waypoints-1, 1), dtype=float
    """
    # Extract the spatial and orientation components
    positions = waypoints[:, :3]
    orientations = waypoints[:, 3:]

    # Compute the distances between adjacent waypoints
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    average_speed = np.sum(distances) / np.sum(ac_intervals)

    # Compute the cumulative time for each waypoint based on the distances and velocities
    times = np.zeros(len(waypoints))
    # times[1:] = np.cumsum(distances / step_interval_list)
    times[1:] = np.cumsum(distances / average_speed)

    # Interpolate the positions using PCHIP
    position_interpolator = PchipInterpolator(times, positions, axis=0)

    num_points = int(times[-1]) * freq  # Number of points in the smooth trajectory
    interpolated_times = np.linspace(times[0], times[-1], num_points)
    interpolated_positions = position_interpolator(interpolated_times)
    interpolated_rotations = slerp_interpolation(times, orientations, num_points)

    # Combine the interpolated positions and orientations
    interpolated_orientations = interpolated_rotations.as_quat()
    smooth_trajectory = np.hstack((interpolated_positions, interpolated_orientations))

    # Track which interpolated positions correspond to the waypoints
    waypoint_indices = np.searchsorted(interpolated_times, times)
    interpolated_waypoints_positions = interpolated_positions[waypoint_indices]

    if gripper_status_seq is None:
        gripper_status_seq = np.zeros((len(waypoints), 1))

    # Find the time points where gripper status changes
    change_indices = [
        i
        for i in range(1, len(gripper_status_seq))
        if gripper_status_seq[i] != gripper_status_seq[i - 1]
    ]
    change_waypoints = [waypoints[i] for i in change_indices]
    change_waypoints_positions = [
        interpolated_waypoints_positions[i] for i in change_indices
    ]

    info = dict(
        positions=positions,
        interpolated_times=interpolated_times,
        interpolated_positions=interpolated_positions,
        interpolated_waypoints_positions=interpolated_waypoints_positions,
        change_waypoints=change_waypoints,
        change_waypoints_positions=change_waypoints_positions,
    )
    return info
