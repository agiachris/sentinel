"""
Centralized client for Kinova with Base ROS2/Py communication.
Should be instantiated on bohg-ws-14 and send python tcp request to iprl-botX.

Author: Zi-ang Cao
"""

# Network communication
from multiprocessing.connection import Client

# Basic Python Dependencies
import numpy as np


class KinovaWithBasePyClient(object):
    def __init__(self, name, ip, port, rest_base_pose=np.array([0.0, 0.0, 0.0])):
        self.name = name
        self.ip = ip
        self.port = port
        print(
            f"[General Py Client] For Kinova (& Base) at IP {self.ip} and port {self.port}"
        )
        self.conn = Client((ip, port), authkey=b"123456")

        # print(f"remove set_rest_base_pose() in __init__!!!")

        # # update rest base pose
        req = dict(base_rest_pose=rest_base_pose)
        assert self.send_request(
            "base_set_rest_pose", req
        ), "Failed to set rest base pose."

        self._arm_gripper_is_closed = False

    def send_request(self, rtype, args=dict()):
        req = {"type": rtype}
        req.update(args)
        self.conn.send(req)
        return self.conn.recv()

    ########### Enforce to set before controling robot to run #############
    def update_env_name(self, env_name):
        # Will be stored in the RobotPyServerMP class
        req = dict(
            env_name=env_name,
        )
        try:
            return self.send_request("update_env_name", req)
        except:
            return False

    def arm_update_workspace(self, arm_limit):
        # Will be stored in the RobotPyServerMP class
        req = dict(
            arm_limit=arm_limit,  # (3, 2)
        )
        try:
            return self.send_request("arm_update_workspace", req)
        except:
            return False

    def base_update_workspace(self, corner_xys):
        # Will be stored in the RobotPyServerMP class
        req = dict(
            corner_xys=corner_xys,
        )
        try:
            return self.send_request("base_update_workspace", req)
        except:
            return False

    def arm_update_fin_gripper_offset(self, fin_gripper_offsets):
        req = dict(fin_gripper_offsets=fin_gripper_offsets)
        try:
            return self.send_request("arm_update_fin_gripper_offset", req)
        except:
            return False

    def base_refresh_ref_angle(self):
        try:
            return self.send_request("base_refresh_ref_angle")
        except:
            return False

    #################### TELEOP CMD ####################
    def teleop_arm_vel(
        self,
        target_arm_xyz_vel,
        target_arm_ori_rotvec_rad_vel,
        vel_cmd_duration,
        gripper_cmd=None,
    ):
        req = dict(
            target_arm_xyz_vel=target_arm_xyz_vel,
            target_arm_ori_rotvec_rad_vel=target_arm_ori_rotvec_rad_vel,
            vel_cmd_duration=vel_cmd_duration,
            gripper_cmd=gripper_cmd,
        )
        try:
            return self.send_request("teleop_arm_vel", req)
        except:
            return None

    #################### High Priority CMD ####################
    def track_eef_traj(
        self,
        target_eef_traj,
        gripper_status_traj,
        step_interval_traj,
        arm_control_freq=125,
        fix_base=False,
        env_name=None,
    ):
        req = dict(
            target_eef_traj=target_eef_traj.reshape(-1, 6),
            gripper_status_traj=gripper_status_traj.reshape(-1, 1),
            step_interval_traj=step_interval_traj.reshape(-1, 1),
            arm_control_freq=int(arm_control_freq),
            fix_base=fix_base,
            env_name=env_name,
        )
        # Print the target_eef_traj, gripper_status_traj, step_interval_traj, kp_traj
        print(f"[track_eef_traj] target_eef_traj.shape: {target_eef_traj.shape}")
        print(
            f"[track_eef_traj] gripper_status_traj.shape: {gripper_status_traj.shape}"
        )
        print(f"[track_eef_traj] step_interval_traj.shape: {step_interval_traj.shape}")
        print(f"[track_eef_traj] arm_control_freq: {arm_control_freq}")
        print(f"[track_eef_traj] fix_base: {fix_base}")
        print(f"[track_eef_traj] env_name: {env_name}")
        ok = self.send_request("track_eef_traj", req)
        assert ok, "Failed to track eef traj."
        return ok

    def get_global_eef_pose_with_gripper_status(self):
        try:
            # dict: base_pose, global_ee_posi, global_ee_ori, gripper_status
            return self.send_request("get_global_eef_pose_with_gripper_status")
        except:
            return None

    def base_move_precise(self, target_base_pose):
        req = dict(
            target_base_pose=np.array(target_base_pose),
        )
        try:
            return self.send_request("base_move_precise", req)
        except:
            return False

    #################### Middle Priority CMD -- Reset only ####################
    def arm_open_gripper(self):
        try:
            ok = self.send_request("arm_open_gripper")
            self._arm_gripper_is_closed = False
            return ok
        except:
            return False

    def arm_close_gripper(self):
        try:
            ok = self.send_request("arm_close_gripper")
            self._arm_gripper_is_closed = True
            return ok
        except:
            return False

    def arm_move_precise_in_joint_space(self, joint_rad):
        req = dict(
            joint_rad=joint_rad,
            joint_deg=np.rad2deg(joint_rad),
        )
        try:
            return self.send_request("arm_move_precise_in_joint_space", req)
        except:
            return False

    def arm_move_precise(self, target_arm_pos, target_arm_ori_deg):
        # NOTE: Leave the wait and reach_time to default!!
        data = np.concatenate(
            [target_arm_pos, target_arm_ori_deg]
        )  # [x, y, z, roll, pitch, yaw]
        req = dict(
            data=data,
            target_arm_pos=target_arm_pos,
            target_arm_ori_deg=target_arm_ori_deg,
        )
        try:
            return self.send_request("arm_move_precise", req)
        except:
            return False

    def arm_home(self):
        try:
            return self.send_request("arm_home")
        except:
            return False

    #################### Low Priority CMD -- debug only ####################
    def arm_move_vel(self, target_arm_pos_vel, target_arm_ori_vel, ac_interval=0.5):
        req = dict(
            target_arm_pos_vel=target_arm_pos_vel,
            target_arm_ori_vel=target_arm_ori_vel,
            ac_interval=ac_interval,
        )
        try:
            return self.send_request("arm_move_vel", req)
        except:
            return None

    #################### Stop and Resume ####################
    def stop(self):
        try:
            return self.send_request("stop")
        except:
            return False

    def arm_stop(self):
        try:
            return self.send_request("arm_stop")
        except:
            return False

    def base_stop(self):
        try:
            return self.send_request("base_stop")
        except:
            return False

    def base_resume(self):
        try:
            return self.send_request("base_resume")
        except:
            return False
