import numpy as np
from kortex_api.autogen.messages import Base_pb2


def compute_ik(base, pos=None, ang=None, guess=None):
    input_joint_angles = base.GetMeasuredJointAngles()
    pose = base.ComputeForwardKinematics(input_joint_angles)
    if guess is None:
        guess = [x.value for x in input_joint_angles.joint_angles]

    if pos is not None:
        pose.x, pose.y, pose.z = pos[0], pos[1], pos[2]
    if ang is not None:
        pose.theta_x, pose.theta_y, pose.theta_z = ang[0], ang[1], ang[2]

    # set ik input data
    input_ik_data = Base_pb2.IKData()
    input_ik_data.cartesian_pose.x = pose.x
    input_ik_data.cartesian_pose.y = pose.y
    input_ik_data.cartesian_pose.z = pose.z
    input_ik_data.cartesian_pose.theta_x = pose.theta_x
    input_ik_data.cartesian_pose.theta_y = pose.theta_y
    input_ik_data.cartesian_pose.theta_z = pose.theta_z

    # add guessed joint angles
    for guess_value in guess:
        guessed_joint_angle = input_ik_data.guess.joint_angles.add()
        guessed_joint_angle.value = guess_value

    result = base.ComputeInverseKinematics(input_ik_data)
    result = np.array([x.value for x in result.joint_angles])
    computed_joint_angles = (result - guess + 180) % 360 - 180 + guess

    return computed_joint_angles
