# Author: Jimmy Wu
# Date: June 2024

# References:
# - https://github.com/empriselab/gen3_compliant_controllers/blob/main/src/JointSpaceCompliantController.cpp
# - https://github.com/empriselab/gen3_compliant_controllers/blob/main/media/controller_formulation.pdf

import math
import queue
import threading
import time
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from constants import POLICY_CONTROL_PERIOD
from kinova_lowlevel import TorqueControlledArm

POLICY_CONTROL_PERIOD = 0.1

ALPHA = 0.01
K_r = np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18])
K_l = np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0])
K_lp = np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
# K_p = np.diag([20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0])
K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
# K_d = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
K_r_inv = np.linalg.inv(K_r)
K_r_K_l = K_r @ K_l
DT = 0.001


class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.y = initial_value

    def filter(self, x):
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class JointCompliantController:
    def __init__(self, command_queue):
        self.q_s = None
        self.q_d = None
        self.dq_d = None
        self.q_n = None
        self.dq_n = None
        self.tau_filter = None
        self.gripper_pos = None
        self.command_queue = command_queue

        # OTG
        self.last_command_time = None
        self.otg = None
        self.otg_inp = None
        self.otg_out = None
        self.otg_res = None

        # self.data = []

    def control_callback(self, arm):
        # Initialize variables on first call
        if self.q_s is None:
            self.q_s = arm.q.copy()
            self.q_d = arm.q.copy()
            self.dq_d = np.zeros_like(arm.q)
            self.q_n = arm.q.copy()
            self.dq_n = arm.dq.copy()
            self.tau_filter = LowPassFilter(ALPHA, arm.tau.copy())
            self.gripper_pos = arm.gripper_pos

            # Initialize OTG
            self.last_command_time = time.time()
            self.otg = Ruckig(arm.actuator_count, DT)
            self.otg_inp = InputParameter(arm.actuator_count)
            self.otg_out = OutputParameter(arm.actuator_count)
            self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
            self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [
                math.radians(450)
            ]
            self.otg_inp.current_position = arm.q.copy()
            self.otg_inp.current_velocity = arm.dq.copy()
            self.otg_inp.target_position = arm.q.copy()
            self.otg_inp.target_velocity = np.zeros(arm.actuator_count)
            self.otg_res = Result.Finished

        # Sensor readings
        self.q_s = (
            self.q_s + np.mod(arm.q - self.q_s + np.pi, 2 * np.pi) - np.pi
        )  # Unwrapped joint angle
        dq_s = arm.dq.copy()
        tau_s = arm.tau.copy()
        tau_s_f = self.tau_filter.filter(tau_s)

        # Check for new command
        if not self.command_queue.empty():
            qpos, self.gripper_pos = self.command_queue.get()
            self.last_command_time = time.time()
            qpos = (
                self.q_s + np.mod(qpos - self.q_s + np.pi, 2 * np.pi) - np.pi
            )  # Unwrapped joint angle
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.q_d[:] = self.otg_out.new_position
            self.dq_d[:] = self.otg_out.new_velocity

        # self.data.append({
        #     'timestamp': time.time(),
        #     'q_s': self.q_s.tolist(),
        #     'dq_s': dq_s.tolist(),
        #     'q_d': self.q_d.tolist(),
        #     'dq_d': self.dq_d.tolist(),
        #     'target_position': self.otg_inp.target_position,
        #     'target_velocity': self.otg_inp.target_velocity,
        #     'new_position': self.otg_out.new_position,
        #     'new_velocity': self.otg_out.new_velocity,
        # })

        # Compute joint torque for task
        g = arm.gravity()
        tau_task = -K_p @ (self.q_n - self.q_d) - K_d @ (self.dq_n - self.dq_d) + g

        # Nominal motor plant
        ddq_n = K_r_inv @ (tau_task - tau_s_f)
        self.dq_n += ddq_n * DT
        self.q_n += self.dq_n * DT

        # Nominal friction
        tau_f = K_r_K_l @ ((self.dq_n - dq_s) + K_lp @ (self.q_n - self.q_s))

        # Torque command
        tau_c = tau_task + tau_f

        return tau_c, self.gripper_pos


def command_loop_retract(command_queue, stop_event):
    # qpos = np.array([0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633])  # Home
    qpos = np.array(
        [0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633]
    )
    gripper_pos = 0
    while not stop_event.is_set():
        command_queue.put((qpos, gripper_pos))
        time.sleep(POLICY_CONTROL_PERIOD)


def command_loop_circle(arm, command_queue, stop_event):
    from ik_solver import IKSolver

    ik_solver = IKSolver(ee_offset=0.12)
    quat = np.array([0.707, 0.707, 0.0, 0.0])  # (x, y, z, w)
    radius = 0.1
    num_points = 30
    center = np.array([0.45, 0.0, 0.2])
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(num_points)
    points = np.column_stack((x, y, z))
    points += center
    gripper_pos = 0
    while not stop_event.is_set():
        for pos in points:
            qpos = ik_solver.solve(pos, quat, arm.q)
            command_queue.put((qpos, gripper_pos))
            time.sleep(POLICY_CONTROL_PERIOD)


if __name__ == "__main__":
    arm = TorqueControlledArm()
    command_queue = queue.Queue(1)
    controller = JointCompliantController(command_queue)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=command_loop_retract, args=(command_queue, stop_event), daemon=True
    )
    # thread = threading.Thread(target=command_loop_circle, args=(arm, command_queue, stop_event), daemon=True)
    thread.start()
    arm.init_cyclic(controller.control_callback)
    try:
        while arm.cyclic_running:
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()
        time.sleep(0.5)  # Wait for arm to stop moving
        arm.stop_cyclic()
        arm.disconnect()
        # import pickle
        # output_path = 'controller-states.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(controller.data, f)
        # print(f'Data saved to {output_path}')
