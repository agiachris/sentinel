# Author: Jimmy Wu
# Date: July 2022

import os
import time
import click
import threading
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import (
    Base_pb2,
)

from sentinel.envs.real_mobile.utils.control.ik import compute_ik
from sentinel.envs.real_mobile.utils.control.connection import DeviceConnection
from sentinel.envs.real_mobile.utils.common.waypoint import make_flip_waypoints

from sentinel.utils.media import save_video

import threading
import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

# from kortex_api.autogen.client_stubs.RouterClient import RouterClientSendOptions
from kortex_api.autogen.messages import Base_pb2


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color_print(msg, color=bcolors.OKGREEN):
    print(f"{color}{msg}{bcolors.ENDC}")


class KinovaArm(object):
    ACTION_TIMEOUT_DURATION = 20

    def __init__(self, router, router_real_time=None):
        self.base = BaseClient(router)
        if router_real_time is not None:
            print(f"Using real-time router. {router_real_time}")
        self.base_cyclic = None
        self.num_joints = self.base.GetActuatorCount().count

        self.vel_cmd_template = self._preint_vel_cmd()
        self.zero_vel_cmd_template = self._preint_zero_vel_cmd()

        self.last_vel_cmd_time = time.time()

    def _preint_vel_cmd(self):
        vel_cmd_template = Base_pb2.TwistCommand()
        vel_cmd_template.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        vel_twist = vel_cmd_template.twist
        vel_twist.linear_x = 0.0
        vel_twist.linear_y = 0.0
        vel_twist.linear_z = 0.0
        vel_twist.angular_x = 0.0
        vel_twist.angular_y = 0.0
        vel_twist.angular_z = 0.0

        return vel_cmd_template

    def _preint_zero_vel_cmd(self):
        zero_vel_cmd_template = Base_pb2.TwistCommand()
        zero_vel_cmd_template.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        zero_vel_twist = zero_vel_cmd_template.twist
        zero_vel_twist.linear_x = 0.0
        zero_vel_twist.linear_y = 0.0
        zero_vel_twist.linear_z = 0.0
        zero_vel_twist.angular_x = 0.0
        zero_vel_twist.angular_y = 0.0
        zero_vel_twist.angular_z = 0.0

        return zero_vel_cmd_template

    def _get_robot_pose(self):
        kpos = self.base.GetMeasuredCartesianPose()
        pos = np.array([kpos.x, kpos.y, kpos.z])
        ang = np.array([kpos.theta_x, kpos.theta_y, kpos.theta_z])  # in degrees
        return pos, ang

    def get_joint_pose(self):
        joint_angles = self.base.GetMeasuredJointAngles().joint_angles
        joint_angles = [a.value for a in joint_angles]
        joint_angles = np.array(joint_angles)
        return joint_angles

    def command_zero_velocity(self):
        cmd = self.zero_vel_cmd_template
        print(f"{bcolors.WARNING}Commanding ZERO velocity: {cmd.twist}{bcolors.ENDC}")
        self.base.SendTwistCommand(cmd)
        return True

    def command_velocity(
        self, vel_cmd_xyz, vel_cmd_ori_deg, vel_cmd_duration=0.1, verbose=False
    ):
        # https://github.com/stanford-iprl-lab/kinova-basics/blob/main/kinova_basics/utils/kinova_utils.py
        # https://docs.kinovarobotics.com/ref/autogen/Messages/Base.html#kortex_api.autogen.TwistCommand
        cmd = self.vel_cmd_template
        twist = cmd.twist
        twist.linear_x = vel_cmd_xyz[0]
        twist.linear_y = vel_cmd_xyz[1]
        twist.linear_z = vel_cmd_xyz[2]
        twist.angular_x = vel_cmd_ori_deg[0]
        twist.angular_y = vel_cmd_ori_deg[1]
        twist.angular_z = vel_cmd_ori_deg[2]
        self.base.SendTwistCommand(cmd)
        if verbose:
            elapsed_time = time.time() - self.last_vel_cmd_time
            color_print(
                f"Sent new non-zero vel_cmd after {elapsed_time} seconds with duration {vel_cmd_duration} seconds. And the command is {twist}."
            )
            self.last_vel_cmd_time = time.time()
        return True

    def move_angular(self, joint_positions, wait=True, reach_time=None):
        assert len(joint_positions) == self.num_joints
        assert (
            type(joint_positions) is list
        ), f"type(joint_positions) should be list but is [{type(joint_positions)}]"
        self.high_level_servoing()

        if reach_time is None:
            action = Base_pb2.Action()
            for i in range(self.num_joints):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = i
                joint_angle.value = joint_positions[i]

        else:
            waypoint_list = Base_pb2.WaypointList()
            waypoint_list.use_optimal_blending = True
            waypoint_list.duration = reach_time  # must be int type -- but won't stop the arm from moving [BAD API DOCUMENTATION]
            waypoint = waypoint_list.waypoints.add()
            waypoint.angular_waypoint.angles.extend(joint_positions)
            waypoint.angular_waypoint.duration = reach_time
            result = self.base.ValidateWaypointList(waypoint_list)
            if len(result.trajectory_error_report.trajectory_error_elements) > 0:
                print("Error found in trajectory")
                print(result.trajectory_error_report)
                return False

        if wait:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
            )
            if reach_time is None:
                self.base.ExecuteAction(action)
            else:
                self.base.ExecuteWaypointTrajectory(waypoint_list)
            finished = e.wait(KinovaArm.ACTION_TIMEOUT_DURATION)
            self.base.Unsubscribe(notification_handle)
        else:
            if reach_time is None:
                self.base.ExecuteAction(action)
            else:
                self.base.ExecuteWaypointTrajectory(waypoint_list)
            finished = True

        return finished

    def high_level_servoing(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def _reference_action(self, action_name):
        # self.high_level_servoing()
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == action_name:
                action_handle = action.handle
        if action_handle is None:
            return False
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )
        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(KinovaArm.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        return finished

    def home(self):
        return self._reference_action("Home")

    def _gripper_command(self, value):
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        # self.cmd_gripper_is_closed = (value == 0.0)
        self.base.SendGripperCommand(gripper_command)
        return True

    def open_gripper(self):
        return self._gripper_command(0.0)

    def close_gripper(self):
        return self._gripper_command(1.0)

    def stop(self):
        self.base.Stop()

    def compute_ik(self, pos, ang, num_attempts=10):
        joint_pos = None
        while num_attempts > 0:
            num_attempts -= 1
            print(f"Left {num_attempts} attempts to compute IK.")
            try:
                joint_pos = compute_ik(self.base, pos=pos, ang=ang)
                break
            except:
                continue
        assert (
            joint_pos is not None
        ), f"Failed to compute IK. for pos: {pos}, ang: {ang}"
        return joint_pos

    @staticmethod
    def check_for_end_or_abort(e):
        def check(notification, e=e):
            if (
                notification.action_event == Base_pb2.ACTION_END
                or notification.action_event == Base_pb2.ACTION_ABORT
            ):
                e.set()

        return check


@click.command()
@click.option("--log_dir", type=str, default="logs", help="Log directory.")
@click.option("--recompute_traj", is_flag=True, help="Whether to recompute trajectory.")
def main(log_dir, recompute_traj):
    os.makedirs(log_dir, exist_ok=True)
    ok = True
    cams = None
    with DeviceConnection.createTcpConnection() as router:
        with DeviceConnection.createUdpConnection() as router_real_time:
            arm = KinovaArm(router, router_real_time)
            ok = arm.home()
            # waypoints = np.concatenate([make_linear_waypoints(30),
            #                             make_circular_waypoints(100)])
            waypoints = make_flip_waypoints()
            traj = arm.waypoint2traj(
                waypoints,
                log_dir,
                recompute_traj,
                max_vel=200,
                max_accel=200,
                max_decel=200,
            )
            while True:
                run_ok, info = arm.run_traj(traj, cams=cams)
                if cams is not None:
                    gray_video = np.array([fr[1] for fr in info["frames"]])
                    rgb_video = np.repeat(gray_video[..., None], 3, -1)
                    print(f'Recorded {len(rgb_video)} frames in {info["t"]:.3f}s.')
                    save_video(
                        rgb_video,
                        os.path.join(log_dir, "vid.mp4"),
                        fps=int(len(rgb_video) / info["t"] / 5),
                    )
                ok = ok and run_ok
                break


if __name__ == "__main__":
    main()
