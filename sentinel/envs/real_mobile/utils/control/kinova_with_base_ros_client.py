"""
Centralized client for Kinova with Base ROS2 communication.
Should be instantiated on bohg-ws-14 and send python tcp request to iprl-botX.

Author: Zi-ang Cao
"""

# Network communication
from multiprocessing.connection import Client

# Basic Python Dependencies
import numpy as np


class KinovaWithBaseROS2Client(object):
    def __init__(self, name, ip, port, rest_base_pose=np.array([0.0, 0.0, 0.0])):
        self.name = name
        self.ip = ip
        self.port = port
        print(
            f"[kinova with base ROS 2 Client] Trying to connect to kinova with base at IP {self.ip} and port {self.port}"
        )
        self.conn = Client((ip, port), authkey=b"123456")

        # update rest base pose
        req = dict(rest_base_pose=rest_base_pose)
        assert self.send_request(
            "base_set_rest_pose", req
        ), "Failed to set rest base pose."

    def send_request(self, rtype, args=dict()):
        req = {"type": rtype}
        req.update(args)
        self.conn.send(req)
        return self.conn.recv()

    def resume(self):
        pass
        # if self.conn.closed: self.conn = Client((self.ip, self.port), authkey=b'123456')
        # Bring other node back to alive?

    #################### Command Arm only ####################
    def arm_home(self):
        try:
            return self.send_request("arm_home")
        except:
            return False

    def arm_open_gripper(self):
        try:
            return self.send_request("arm_open_gripper")
        except:
            return False

    def arm_close_gripper(self):
        try:
            return self.send_request("arm_close_gripper")
        except:
            return False

    def arm_move_precise(self, pos, ang):
        # NOTE: Leave the wait and reach_time to default!!
        data = np.concatenate([pos, ang])  # [x, y, z, roll, pitch, yaw]
        req = dict(data=data)
        try:
            return self.send_request("arm_move_precise", req)
        except:
            return False

    def arm_move_vel(self, target_arm_pos_vel, target_arm_ori_vel):
        req = dict(
            target_arm_pos_vel=target_arm_pos_vel,
            target_arm_ori_vel=target_arm_ori_vel,
        )
        try:
            return self.send_request(
                "arm_move_vel", req
            )  # will return self.send_request('get_whole_robot_pose')
        except:
            return None

    #################### Command Base only ####################
    def base_home(self):
        try:
            return self.send_request("base_home")
        except:
            return False

    def base_refresh_ref_angle(self):
        try:
            return self.send_request("base_refresh_ref_angle")
        except:
            return False

    def base_move_precise(self, target_base_pose):
        data = np.array(target_base_pose)
        print(f"[kinova_with_base_ros_client] base_move_precise: {data}")
        req = dict(data=data)
        try:
            return self.send_request("base_move_precise", req)
        except:
            return False

    #################### Command Whole Robot ####################
    def get_whole_robot_pose(self):
        try:
            base_pose, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori = (
                self.send_request("get_whole_robot_pose")
            )
            # self.base_pose = base_pose
            # self.local_ee_pos = local_ee_pos
            # self.local_ee_ori = local_ee_ori
            # self.global_ee_pos = global_ee_pos
            # self.global_ee_ori = global_ee_ori  # [roll, pitch, yaw]
            return base_pose, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori
        except:
            return None

    def stop(self):
        try:
            return self.send_request("stop")
        except:
            return False

    def resume(self):
        try:
            return self.send_request("resume")
        except:
            return False

    def home(self):
        try:
            return self.send_request("home")
        except:
            return False

    def move_base_and_arm(
        self, target_base_pose, target_arm_pos_vel, target_arm_ori_vel, duration=0.1
    ):
        req = dict(
            target_base_pose=target_base_pose,
            target_arm_pos_vel=target_arm_pos_vel,
            target_arm_ori_vel=target_arm_ori_vel,
            duration=duration,
        )
        try:

            # will return self.send_request('get_whole_robot_pose')
            base_pose, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori = (
                self.send_request("move_base_and_arm", req)
            )
            # self.base_pose = base_pose
            # self.local_ee_pos = local_ee_pos
            # self.local_ee_ori = local_ee_ori    # [roll, pitch, yaw]
            # self.global_ee_pos = global_ee_pos
            # self.global_ee_ori = global_ee_ori
            return base_pose, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori
        except:
            return None
