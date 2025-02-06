# Author: Jimmy Wu
# Date: June 2024

# References:
# - https://github.com/Kinovarobotics/kortex/blob/master/api_python/examples/108-Gen3_torque_control/01-torque_control_cyclic.py
# - https://github.com/empriselab/kortex_hardware/blob/main/src/Gen3Robot.cpp

import math
import os
import subprocess
import threading
import time
import numpy as np
import pinocchio as pin
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import (
    ActuatorCyclic_pb2,
    ActuatorConfig_pb2,
    Base_pb2,
    BaseCyclic_pb2,
    Common_pb2,
    ControlConfig_pb2,
    Session_pb2,
)
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from scipy.spatial.transform import Rotation as R


class DeviceConnection:
    IP_ADDRESS = "192.168.1.10"
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection():
        return DeviceConnection(port=DeviceConnection.TCP_PORT)

    @staticmethod
    def createUdpConnection():
        return DeviceConnection(port=DeviceConnection.UDP_PORT)

    def __init__(
        self, ip_address=IP_ADDRESS, port=TCP_PORT, credentials=("admin", "admin")
    ):
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = (
            TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        )
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ip_address)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()


class TorqueControlledArm:
    def __init__(self):
        # Check whether arm is connected
        try:
            subprocess.run(
                ["ping", "-c", "1", "192.168.1.10"],
                check=True,
                timeout=1,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            raise Exception("Could not communicate with arm") from e

        # Lock file to enforce single instance
        self.lock_file = "/tmp/kinova.lock"
        if os.path.exists(self.lock_file):
            with open(self.lock_file, "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale lock file (PID {pid})")
                os.remove(self.lock_file)
            else:
                raise Exception(
                    f"Another instance of the arm is already running (PID {pid})"
                )
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        # General Kortex setup
        self.tcp_connection = DeviceConnection.createTcpConnection()
        self.udp_connection = DeviceConnection.createUdpConnection()
        self.base = BaseClient(self.tcp_connection.__enter__())
        self.base_cyclic = BaseCyclicClient(self.udp_connection.__enter__())
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count
        self.control_config = ControlConfigClient(self.base.router)
        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier
            for handle in device_handles.device_handle
            if handle.device_type
            in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3

        # Command and feedback setup
        self.base_command = BaseCyclic_pb2.Command()
        for _ in range(self.actuator_count):
            self.base_command.actuators.add()
        self.motor_cmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        self.base_feedback = BaseCyclic_pb2.Feedback()

        # Make sure actuators are in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "POSITION"
        )
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Torque control setup
        # Note: Torque commands are converted to current commands since
        # Kinova's torque controller is unable to achieve commanded torques.
        # See relevant GitHub issue: https://github.com/Kinovarobotics/kortex/issues/38
        self.torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
        self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
        self.current_limit_min = -self.current_limit_max

        # Cyclic thread setup
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.cyclic_running = False

        # Robot state setup (only used in low-level servoing mode)
        self.q = np.zeros(self.actuator_count)
        self.dq = np.zeros(self.actuator_count)
        self.tau = np.zeros(self.actuator_count)
        self.gripper_pos = 0

        # Pinocchio setup (only used in low-level servoing mode)
        self.model = pin.buildModelFromUrdf("models/gen3_robotiq_2f_85.urdf")
        self.data = self.model.createData()
        self.q_pin = np.zeros(self.model.nq)
        self.tool_frame_id = self.model.getFrameId("tool_frame")

    def disconnect(self):
        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()
        os.remove(self.lock_file)

    def _execute_reference_action(self, action_name):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Retrieve reference action
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == action_name:
                action_handle = action.handle
        if action_handle is None:
            return

        # Execute action
        end_or_abort_event = threading.Event()

        def check_for_end_or_abort(e):
            def check(notification, e=e):
                if notification.action_event in [
                    Base_pb2.ACTION_END,
                    Base_pb2.ACTION_ABORT,
                ]:
                    e.set()

            return check

        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(end_or_abort_event), Base_pb2.NotificationOptions()
        )
        self.base.ExecuteActionFromReference(action_handle)
        end_or_abort_event.wait(20)
        self.base.Unsubscribe(notification_handle)

    def home(self):
        self._execute_reference_action("Home")

    def retract(self):
        self._execute_reference_action("Retract")

    def zero(self):
        self._execute_reference_action("Zero")

    def _gripper_position_command(self, value):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        # Send gripper command
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported position to match value
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if abs(value - gripper_measure.finger[0].value) < 0.01:
                break
            time.sleep(0.01)

    def open_gripper(self):
        self._gripper_position_command(0)

    def close_gripper(self):
        self._gripper_position_command(1)

    def set_joint_limits(
        self,
        speed_limits=(60, 60, 60, 60, 60, 60, 60),
        acceleration_limits=(80, 80, 80, 80, 80, 80, 80),
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

    def clear_faults(self):
        if self.base.GetArmState().active_state == Base_pb2.ARMSTATE_IN_FAULT:
            self.base.ClearFaults()
            while (
                self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY
            ):
                time.sleep(0.1)

    def zero_torque_offsets(self):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        # Move arm to zero configuration
        print("Arm will be moved to the candlestick configuration")
        input(
            "Please make sure arm is clear of obstacles and then press <Enter> to continue..."
        )
        self.zero()

        # Wait for arm to become fully still
        input(
            "Please wait until the the arm is fully still and then press <Enter> to continue..."
        )

        # Set zero torque offsets
        for device_id in self.actuator_device_ids:
            torque_offset = ActuatorConfig_pb2.TorqueOffset()
            self.actuator_config.SetTorqueOffset(torque_offset, device_id)
        print("Torque offsets have been set to zero")

        # Move arm to home configuration
        input(
            "Arm will now be moved to the home configuration, press <Enter> to continue..."
        )
        self.home()

    def init_cyclic(self, control_callback):
        assert not self.cyclic_running, "Cyclic thread is already running"

        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(
                0,
                os.SCHED_FIFO,
                os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)),
            )
        except PermissionError:
            print(
                "Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf"
            )

        # Initialize command frame
        self.base_feedback = self.base_cyclic.RefreshFeedback()
        for i in range(self.actuator_count):
            self.base_command.actuators[i].flags = ActuatorCyclic_pb2.SERVO_ENABLE
            self.base_command.actuators[i].position = self.base_feedback.actuators[
                i
            ].position
            self.base_command.actuators[i].current_motor = self.base_feedback.actuators[
                i
            ].current_motor
        self.motor_cmd.position = (
            self.base_feedback.interconnect.gripper_feedback.motor[0].position
        )
        self.motor_cmd.velocity = 0
        self.motor_cmd.force = 100

        # Set arm to low-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        print("Arm is in low-level servoing mode")

        # Send first frame and update robot state
        self.base_feedback = self.base_cyclic.Refresh(
            self.base_command, 0, self.send_options
        )

        # Set actuators to current control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "CURRENT"
        )
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Start cyclic thread
        self.kill_the_thread = False
        self.cyclic_thread = threading.Thread(
            target=self.run_cyclic, args=(control_callback,), daemon=True
        )
        self.cyclic_thread.start()

    def run_cyclic(self, control_callback):
        self.cyclic_running = True
        failed_cyclic_count = 0
        # cyclic_count = 0
        # data = []

        # Update state before entering loop
        self.update_state()

        t_now = time.time()
        t_cyclic = t_now
        while not self.kill_the_thread:
            t_now = time.time()
            step_time = t_now - t_cyclic
            if step_time >= 0.001:  # 1 kHz
                t_cyclic = t_now
                if step_time > 0.004:  # 4 ms
                    print(
                        f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} run_cyclic"
                    )

                # Get torque command
                torque_command, gripper_command = control_callback(self)

                # Convert to current command (Note: Current refers to electrical current)
                current_command = np.divide(torque_command, self.torque_constant)

                # Clamp using current limits
                np.clip(
                    current_command,
                    self.current_limit_min,
                    self.current_limit_max,
                    out=current_command,
                )

                # Increment frame ID to ensure actuators can reject out-of-order frames
                self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536

                # Update arm command
                for i in range(self.actuator_count):
                    # Update current command
                    self.base_command.actuators[i].current_motor = current_command[i]

                    # Update position command to avoid triggering following error
                    self.base_command.actuators[i].position = (
                        self.base_feedback.actuators[i].position
                    )

                    # Update command ID
                    self.base_command.actuators[i].command_id = (
                        self.base_command.frame_id
                    )

                # Update gripper command
                self.motor_cmd.position = 100 * gripper_command
                self.motor_cmd.velocity = np.clip(
                    abs(400 * (gripper_command - self.gripper_pos)), 0, 100
                )

                # Send command frame
                try:
                    # Note: This call takes up most of the 1000 us cyclic step time
                    self.base_feedback = self.base_cyclic.Refresh(
                        self.base_command, 0, self.send_options
                    )
                except:
                    failed_cyclic_count += 1

                # Update robot state
                self.update_state()

                # data.append({
                #     'timestamp': t_now,
                #     'position': [actuator.position for actuator in self.base_feedback.actuators],
                #     'velocity': [actuator.velocity for actuator in self.base_feedback.actuators],
                #     'torque': [actuator.torque for actuator in self.base_feedback.actuators],
                #     'current_motor': [actuator.current_motor for actuator in self.base_feedback.actuators],
                #     'gripper_pos': self.base_feedback.interconnect.gripper_feedback.motor[0].position,
                #     'torque_command': torque_command.tolist(),
                #     'gripper_command': gripper_command,
                #     'current_command': current_command.tolist(),
                # })
                # cyclic_count += 1
                # if cyclic_count >= 5000:
                #     break

        # import pickle
        # output_path = 'actuator-states.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(data, f)
        # print(f'Data saved to {output_path}')

        self.cyclic_running = False

    def stop_cyclic(self):
        # Kill cyclic thread
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()

        # Set actuators back to position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "POSITION"
        )
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Set arm back to high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        print("Arm is back in high-level servoing mode")

    def update_state(self):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"

        # Robot state
        for i in range(self.actuator_count):
            self.q[i] = self.base_feedback.actuators[i].position
            self.dq[i] = self.base_feedback.actuators[i].velocity
            self.tau[i] = self.base_feedback.actuators[i].torque
        np.deg2rad(self.q, out=self.q)
        np.deg2rad(self.dq, out=self.dq)
        np.negative(
            self.tau, out=self.tau
        )  # Raw torque readings are negative relative to actuator direction
        self.gripper_pos = (
            self.base_feedback.interconnect.gripper_feedback.motor[0].position / 100.0
        )

        # Pinocchio joint configuration
        self.q_pin = np.array(
            [
                math.cos(self.q[0]),
                math.sin(self.q[0]),
                self.q[1],
                math.cos(self.q[2]),
                math.sin(self.q[2]),
                self.q[3],
                math.cos(self.q[4]),
                math.sin(self.q[4]),
                self.q[5],
                math.cos(self.q[6]),
                math.sin(self.q[6]),
            ]
        )

    def gravity(self):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"
        return pin.computeGeneralizedGravity(self.model, self.data, self.q_pin)

    def get_tool_pose(self):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"
        pin.framesForwardKinematics(self.model, self.data, self.q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]
        pos = tool_pose.translation.copy()  # This copy() is important!
        # quat = pin.Quaternion(tool_pose.rotation).coeffs()  # This causes a segfault
        quat = R.from_matrix(tool_pose.rotation).as_quat()
        return pos, quat


def grav_comp_control_callback(arm):
    torque_command = arm.gravity()
    gripper_command = 0
    return torque_command, gripper_command


def main():
    arm = TorqueControlledArm()
    try:
        arm.zero_torque_offsets()
        arm.home()
        arm.init_cyclic(grav_comp_control_callback)
        while arm.cyclic_running:
            time.sleep(0.01)
    except KeyboardInterrupt:
        arm.stop_cyclic()
        arm.disconnect()


if __name__ == "__main__":
    main()
