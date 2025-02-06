# Author: Jimmy Wu
# Date: July 2022

import os
import copy
import time
import click
import threading
import numpy as np
from tqdm import tqdm

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import (
    ActuatorCyclic_pb2,
    Base_pb2,
    BaseCyclic_pb2,
)
from kortex_api.autogen.messages import ControlConfig_pb2
from kortex_api.RouterClient import RouterClientSendOptions

from sentinel.envs.real_mobile.utils.control.ik import compute_ik
from sentinel.envs.real_mobile.utils.control.connection import DeviceConnection
from sentinel.envs.real_mobile.utils.common.retime import retime
from sentinel.envs.real_mobile.utils.common.waypoint import make_flip_waypoints

from sentinel.utils.media import save_video


class KinovaLowLevelController(object):
    def __init__(self, base, base_cyclic):
        self.step_size = 1e-3
        self.base = base
        self.base_cyclic = base_cyclic  # low-level control
        self.num_joints = self.base.GetActuatorCount().count
        self.base_command = BaseCyclic_pb2.Command()
        for _ in range(self.num_joints):
            self.base_command.actuators.add()
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        self.base_feedback = None
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3
        self.cyclic_running = False
        self.trajectory = None
        self.control_config = ControlConfigClient(self.base.router)
        # self._set_joint_limits([60] * 7, [80] * 7)
        self._set_joint_limits([60] * 7, [200] * 7)

    def _to_high_level(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def _to_low_level(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def _wait_until(self, t):
        while True:
            now = time.time()
            if now >= t:
                break
        return now

    def _set_joint_limits(
        self, speed_limits=(7 * [60]), acceleration_limits=(7 * [80])
    ):
        joint_speed_soft_limits = ControlConfig_pb2.JointSpeedSoftLimits()
        joint_speed_soft_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
        joint_speed_soft_limits.joint_speed_soft_limits.extend(speed_limits)
        self.control_config.SetJointSpeedSoftLimits(joint_speed_soft_limits)
        joint_acceleration_soft_limits = ControlConfig_pb2.JointAccelerationSoftLimits()
        joint_acceleration_soft_limits.control_mode = (
            ControlConfig_pb2.ANGULAR_TRAJECTORY
        )
        joint_acceleration_soft_limits.joint_acceleration_soft_limits.extend(
            acceleration_limits
        )
        self.control_config.SetJointAccelerationSoftLimits(
            joint_acceleration_soft_limits
        )

    def run_trajectory(self, traj, cams=None, render_interval=5):
        # Start cyclic thread
        success = True
        self.trajectory = traj
        self.cams = cams
        self.render_interval = render_interval
        self._init_cyclic()
        while self.cyclic_running:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                success = False
                break
        self._stop_cyclic()
        return success, {"frames": self.frames, "t": self.exe_time}

    def _init_cyclic(self):
        # change to low-level mode
        self._to_low_level()

        self.base_feedback = self.base_cyclic.RefreshFeedback()
        for i in range(self.num_joints):
            self.base_command.actuators[i].flags = ActuatorCyclic_pb2.SERVO_ENABLE
            self.base_command.actuators[i].position = self.base_feedback.actuators[
                i
            ].position
        self.base_feedback = self.base_cyclic.Refresh(
            self.base_command, 0, self.send_options
        )

        self.cyclic_thread = threading.Thread(target=self._run_cyclic)
        self.cyclic_thread.daemon = True
        self.cyclic_thread.start()

    def _run_cyclic(self):
        self.cyclic_running = True
        actuators = self.base_command.actuators
        step_size = self.step_size

        ok = True

        t_now = time.time()
        t_start = t_now
        t_prev = t_now
        step_times = []
        self.frames = []
        for t in range(len(self.trajectory) * 2):
            t_now = self._wait_until(t_prev + step_size)
            step_times.append(t_now - t_prev)
            t_prev = t_now

            for i in range(self.num_joints):
                actual_t = min(t, len(self.trajectory) - 1)
                actuators[i].position = self.trajectory[actual_t][i]

            self.base_command.frame_id += 1
            if self.base_command.frame_id > 65535:
                self.base_command.frame_id = 0
            for i in range(self.num_joints):
                actuators[i].command_id = self.base_command.frame_id

            try:
                self.base_feedback = self.base_cyclic.Refresh(
                    self.base_command, 0, self.send_options
                )
            except:
                ok = False
                break

            if (
                self.cams is not None
                and len(self.cams) > 0
                and t % self.render_interval == 0
            ):
                frame = self.cams.get_frame()
                frame = [copy.deepcopy(frame[0][0]), copy.deepcopy(frame[0][1])]
                self.frames.append(frame)
        self.exe_time = time.time() - t_start

        self.cyclic_running = False
        if not ok:
            return False

        bins = np.r_[np.arange(step_size, 11 * step_size, step_size), np.inf]
        print(np.histogram(step_times, bins=bins))

        return True

    def _stop_cyclic(self):
        if self.cyclic_running:
            self.cyclic_thread.join()

        # change back to high-level mode
        self._to_high_level()


class KinovaArm(object):
    ACTION_TIMEOUT_DURATION = 20

    def __init__(self, router, router_real_time=None):
        self.base = BaseClient(router)
        if router_real_time is not None:
            self.base_cyclic = BaseCyclicClient(router_real_time)
        else:
            self.base_cyclic = None
        self.num_joints = self.base.GetActuatorCount().count
        self.controller = KinovaLowLevelController(self.base, self.base_cyclic)

        self.cmd_vel_reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

    def _get_robot_pose(self):
        kpos = self.base.GetMeasuredCartesianPose()
        pos = np.array([kpos.x, kpos.y, kpos.z])
        ang = np.array([kpos.theta_x, kpos.theta_y, kpos.theta_z])
        return pos, ang

    def get_joint_pose(self):
        joint_angles = self.base.GetMeasuredJointAngles().joint_angles
        joint_angles = [a.value / 180 * np.pi for a in joint_angles]
        joint_angles = np.array(joint_angles)
        return joint_angles

    def waypoint2traj(
        self,
        waypoints,
        log_dir=None,
        traj_name="traj",
        recompute_traj=False,
        max_vel=200,
        max_accel=200,
        max_decel=200,
    ):
        assert waypoints.shape[-1] in [3, 6], "Unknown waypoint format."
        curr_pos, curr_ang = self._get_robot_pose()
        print(f"Current robot pose: pos {curr_pos}, ang {curr_ang}")
        if log_dir is not None:
            traj_path = os.path.join(log_dir, f"{traj_name}.npz")
        if os.path.isfile(traj_path) and not recompute_traj:
            traj = np.load(traj_path)["traj"]
        else:
            joint_poses = [None]
            for i, waypoint in enumerate(tqdm(waypoints, desc="Compute Traj")):
                while True:
                    try:
                        convert_kwargs = dict(guess=joint_poses[-1])
                        if len(waypoint) == 3:
                            convert_kwargs["pos"] = waypoint
                        elif len(waypoint) == 6:
                            convert_kwargs["pos"] = waypoint[:3]
                            convert_kwargs["ang"] = waypoint[3:]
                        converted_pose = compute_ik(self.base, **convert_kwargs)
                        joint_poses.append(converted_pose)
                        break
                    except:
                        continue
            traj = np.array(joint_poses[1:])
            np.savez(traj_path, traj=traj)
        retime_kwargs = dict(max_vel=max_vel, max_accel=max_accel, max_decel=max_decel)
        retimed_x, retimed_v, retimed_a = retime(traj, **retime_kwargs)
        max_v = np.max(np.abs(retimed_v)).round(2)
        max_a = np.max(np.abs(retimed_a)).round(2)
        print(f"Trajectory (size {len(traj)}) max v {max_v} and max a {max_a}")
        return retimed_x

    def run_traj(self, traj, cams=None, render_interval=5):
        assert self.router_real_time is not None
        input(f"[Press enter to move robot to start pose {traj[0].round(2)}]")
        self.move_angular(traj[0])
        input("[Press enter to run trajectory]")
        success = self.controller.run_trajectory(traj, cams, render_interval)
        return success

    def command_velocity(
        self, tgt_velocity, orin_velocity, duration=0.1, stop_at_time=False
    ):
        # https://github.com/stanford-iprl-lab/kinova-basics/blob/main/kinova_basics/utils/kinova_utils.py
        command = Base_pb2.TwistCommand()
        command.reference_frame = (
            self.cmd_vel_reference_frame
        )  # CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0
        twist = command.twist
        twist.linear_x = tgt_velocity[0]
        twist.linear_y = tgt_velocity[1]
        twist.linear_z = tgt_velocity[2]
        twist.angular_x = orin_velocity[0]
        twist.angular_y = orin_velocity[1]
        twist.angular_z = orin_velocity[2]
        self.base.SendTwistCommand(command)
        if stop_at_time:
            finish_time = time.time() + duration
            time.sleep(finish_time - time.time())
            twist.linear_x = 0
            twist.linear_y = 0
            twist.linear_z = 0
            twist.angular_x = 0
            twist.angular_y = 0
            twist.angular_z = 0
            self.base.SendTwistCommand(command)
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
            waypoint_list.duration = reach_time
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

    def _reference_action(self, action_name):
        self.high_level_servoing()
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

    def high_level_servoing(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def _gripper_command(self, value):
        # self.high_level_servoing()
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)
        # time.sleep(0.8)
        return True

    def open_gripper(self):
        return self._gripper_command(0)

    def close_gripper(self):
        return self._gripper_command(1)

    def stop(self):
        self.base.Stop()

    def compute_ik(self, pos, ang, num_attempts=10):
        joint_pos = None
        while num_attempts > 0:
            num_attempts -= 1
            print("IK num_attempts", num_attempts)
            try:
                joint_pos = compute_ik(self.base, pos=pos, ang=ang)
                break
            except:
                continue
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
