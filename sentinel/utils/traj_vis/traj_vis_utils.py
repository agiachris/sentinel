import numpy as np

from collections import deque


import pybullet
import numpy as np
from collections import deque


from sentinel.utils.transformations import quat2mat
from sentinel.utils.transformations import (
    quat2mat,
    quat_multiply,
    axisangle2quat,
    mat2euler,
)


def convert_state_to_poses(state):
    """
    Input: state (n_robot, 13)
    Output: poses (n_robot, 6)
    """
    posis, x_axis, z_axis, gravity_direction, gripper_poses = (
        state[:, :3],
        state[:, 3:6],
        state[:, 6:9],
        state[:, 9:12],
        state[:, -1:],
    )
    # posis.shape = (n_robot, 3)
    # x_axis.shape = (n_robot, 3)
    # z_axis.shape = (n_robot, 3)
    # gravity_direction.shape = (n_robot, 3)
    # gripper_poses.shape = (n_robot, 1)

    # Find y_axis
    y_axis = np.cross(z_axis, x_axis)
    # y_axis.shape = (n_robot, 3)

    # Find rotation matrix
    rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=2)
    # rot_matrix.shape = (n_robot, 3, 3)

    # Find euler radians
    euler_radians = np.array([mat2euler((quat)) for quat in rot_matrix])
    # euler_radians.shape = (n_robot, 3)

    # # change the order of euler angles to be (z, x, y) instead of (x, y, z)
    # euler_radians = np.array([euler_radians[i][[2, 0, 1]] for i in range(euler_radians.shape[0])])

    # Find poses
    poses = np.concatenate([posis, euler_radians], axis=1)
    # poses.shape = (n_robot, 6)

    return poses, gripper_poses


def convert_poses_to_state(poses, gripper_poses=None):
    """
    Input: poses (n_robot, 6)
    gripper_poses (n_robot, 1)
    Output: state (n_robot, 13)
    """
    posis, euler_radians = poses[:, :3], poses[:, 3:6]
    gripper_status_shape = (poses.shape[0], 1)
    # posis.shape = (n_robot, 3)
    # euler_radians.shape = (n_robot, 3)

    # Convert Euler angles to rotation matrix

    # NOTE: euler2mat from transformations.py is different from the one in pybullet!!!!!
    # rot_matrices = np.array([euler2mat(euler) for euler in euler_radians])

    rot_matrices = np.array(
        [quat2mat(pybullet.getQuaternionFromEuler(euler)) for euler in euler_radians]
    )
    # rot_matrices.shape = (n_robot, 3, 3)

    # Extract X, Y, Z axes from the rotation matrices
    x_axis = rot_matrices[:, :, 0]
    # y_axis = rot_matrices[:, :, 1]
    z_axis = rot_matrices[:, :, 2]

    # Assume gravity_direction is constant or known, set as a downward direction if not provided
    gravity_direction = np.tile(np.array([[0, 0, -1]]), gripper_status_shape)
    # Assume gripper_poses is zero if not provided
    if gripper_poses is None:
        gripper_poses = np.zeros(gripper_status_shape)
    else:
        gripper_poses = gripper_poses.reshape(gripper_status_shape)

    # Reconstruct the state
    state = np.concatenate(
        [posis, x_axis, z_axis, gravity_direction, gripper_poses], axis=-1
    )
    # state.shape = (n_robot, 13)

    return state


def apply_axisangle_rotation(curr_euler, axisangle_ac, n_robot=1):
    """
    Apply axisangle_ac rotation to curr_euler, return the next orientation in euler format
    """

    ac_next_oris = []

    for idx in range(n_robot):
        curr_quat = pybullet.getQuaternionFromEuler(curr_euler[idx])
        ac_next_eef_quat = quat_multiply(axisangle2quat(axisangle_ac[idx]), curr_quat)
        ac_next_eef_euler = pybullet.getEulerFromQuaternion(ac_next_eef_quat)
        ac_next_oris.append(ac_next_eef_euler)

    return np.array(ac_next_oris)


def extract_ac_traj(ac_ref_state, non_scaled_acs, num_skip_steps=0, ac_horizon=16):
    """
    ac_ref_state: (n_robot, 13) np.array
    ac_without_scale: (16, n_robot, 7) np.array
    """

    ac_ref_poses, ac_ref_gripper_status = convert_state_to_poses(
        ac_ref_state
    )  # (n_robot, 6)

    n_robot = ac_ref_poses.shape[0]  # 1

    # Apply scaling to the raw_agent_ac
    expected_states_dq = deque(maxlen=ac_horizon)

    for ac_ix in range(num_skip_steps, num_skip_steps + ac_horizon):
        non_scaled_ac = non_scaled_acs[ac_ix]

        if ac_ix == num_skip_steps:
            ac_next_posis = ac_ref_poses[:, :3]  # (n_robot, 3)
            ac_next_oris = ac_ref_poses[:, 3:]  # (n_robot, 3) in euler radians format

        # Expected next positions (n_robot, 3)
        ac_next_posis = non_scaled_ac[:, 1:4] + ac_next_posis
        # Expected next gripper states (n_robot, 1)
        ac_next_grps = non_scaled_ac[:, 0:1]

        # Expected next orientations (n_robot, 3) in euler radians format
        # Apply axis-angle rotation to the reference orientation
        ac_next_oris = apply_axisangle_rotation(
            ac_next_oris, non_scaled_ac[:, 4:], n_robot
        )

        # Compose the expected next poses (n_robot, 6)
        ac_next_poses = np.concatenate([ac_next_posis, ac_next_oris], axis=1)

        expected_states = convert_poses_to_state(ac_next_poses, ac_next_grps)

        expected_states_dq.append(expected_states)

    expected_states = np.array(expected_states_dq)
    return expected_states
