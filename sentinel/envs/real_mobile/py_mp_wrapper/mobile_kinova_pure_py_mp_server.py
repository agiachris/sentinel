# Python
import time
import click
import socket
import numpy as np

from multiprocessing import Process, Lock
from multiprocessing.connection import Listener

# Onboard Arm
from sentinel.envs.real_mobile.utils.control.connection import DeviceConnection

from sentinel.envs.real_mobile.utils.control.kinova_new import KinovaArm
from sentinel.envs.real_mobile.utils.common.convert_coords import (
    local_to_global_pose,
)

# Network communication
from sentinel.envs.real_mobile.utils.common.constant import hardware

from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory as SM

from copy import deepcopy as dcp

# Onboard Base
from sentinel.envs.real_mobile.utils.control.mobile_base import MobileBase
from sentinel.envs.real_mobile.utils.perception.frame_converter import (
    CoordFrameConverter,
)

# Safety + Exceptions
from sentinel.envs.real_mobile.utils.common.constant import hardware

# MultiThreadedExecutor

from sentinel.utils.mocap_client.mocap_agent import MocapAgent

from sentinel.envs.sim_mobile.utils.init_utils import rotate_around_z

import pybullet
from sentinel.utils.transformations import (
    quat2mat,
)

from sentinel.envs.real_mobile.py_mp_wrapper.high_freq_control_utils import (
    generate_arm_high_freq_traj,
    # generate_conservative_polygon,
    # plan_robot_base_movement,
    # wrap_angle,
)

from shapely.geometry import Polygon, Point
from sentinel.envs.real_mobile.py_mp_wrapper.utils.base_movement import (
    move_robot_base_in_xy,
)

from scipy.spatial.transform import Rotation


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


def init_SHM(topic_name="global_ee_pos", shape=(3,), create=False, just_close=False):
    if just_close:
        try:
            array_shm = SM(name=topic_name)
            array_shm.close()
            print(f"closed shared memory for {topic_name}.")
        except FileNotFoundError:
            print(f"shared memory for {topic_name} is not found.")
        return

    if "time" in topic_name or "interval" in topic_name:
        shape = (1,)
        dtype = np.float64
    elif ("needs" in topic_name) or ("is" in topic_name) or ("enable" in topic_name):
        shape = (1,)
        dtype = np.float32
    elif "joint" in topic_name:
        shape = (7,)
        dtype = np.float32
    else:
        shape = (3,)
        dtype = np.float32

    if create:
        try:
            array_shm = SM(name=topic_name)
            array_shm.close()
            print(
                f"first close the previous leaked shared memory with same topic name = {topic_name}"
            )
        except FileNotFoundError:
            pass

        data = np.zeros(shape, dtype)
        print(f"created shared memory for {topic_name}.")
        SM(name=topic_name, create=True, size=data.nbytes)
        return
    else:
        data_shm = SM(name=topic_name)
        data = np.ndarray(shape, dtype, buffer=data_shm.buf)
        return data_shm, data


def create_all_sm_topics(just_close=False):
    topic_list = [
        # Kinova Local EEF Pose
        "local_ee_pos",
        "local_ee_ori",
        # Kinova Global EEF Pose
        "global_ee_pos",
        "global_ee_ori",
        # Kinova Global EEF Pose + Fin Gripper Pose
        "global_ee_fin_gripper_pos",
        "global_ee_fin_gripper_ori",
        "target_arm_joint_rad",
        "is_joint_cmd_arrived",
        "enable_arm_vel_cmd_stop_callback",
        "last_arm_vel_cmd_time",
        "arm_cmd_vel_interval",
        "is_ready_for_next_arm_vel_cmd",
        "is_all_zero_vel_cmd_arrived",
        # Others
        "is_gripper_close",
        "arm_fin_gripper_offset",
        "target_arm_xyz_vel",
        "target_arm_ori_deg_vel",
        "target_arm_pos",
        "target_arm_ori_deg",
        "is_target_gripper_close",
        "is_pos_cmd_arrived",
        "is_vel_cmd_arrived",
        "is_gripper_cmd_arrived",
        "arm_needs_home",
        "arm_needs_stop",
        "base_mocap_pose",
        "target_base_pose",
        "is_base_pos_cmd_arrived",
        "base_needs_stop",
        "base_needs_wait",
        "base_needs_refresh_ref_angle",
        "base_needs_set_rest_pose",
        "base_rest_pose",
        "base_needs_resume",
        "base_target_pose_is_updated",
    ]

    for topic_name in topic_list:
        if just_close:
            init_SHM(topic_name, just_close=True)
        else:
            init_SHM(topic_name, create=True)


class MoCapMPAgent(object):
    def __init__(self):
        # Get base name
        machine_name = socket.gethostname()
        base_name = machine_name[-4:]  # 'bot1' or 'bot2' or 'bot3'
        assert base_name[:3] == "bot", "Machine name should start with bot"

        self.bot_name = base_name
        print("init MocapAgent")

        self._init_SM_to_local()
        print("init SHM locally to MocapAgent.")

        self.mocap_agent = MocapAgent(
            focus_bot_name=self.bot_name,
            shared_memory=self.base_mocap_pose,
            shared_memory_shm=self.base_mocap_pose_shm,
        )

    def _init_SM_to_local(self):
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM(
            "base_mocap_pose"
        )  # x, y, theta(radians from -pi to pi)


class BaseControlLoop(object):
    def __init__(self):
        # Get base name
        machine_name = socket.gethostname()
        base_name = machine_name[-4:]  # 'bot1' or 'bot2' or 'bot3'
        assert base_name[:3] == "bot", "Machine name should start with bot"

        self.base = MobileBase(name=base_name)

        self.coord_frame_converter = CoordFrameConverter()
        # self.local_base_rest_pose = np.zeros(3)   # x, y, theta(radians from -pi to pi)
        self.prev_pose_from_odom_global = np.zeros(
            3
        )  # x, y, theta(radians from -pi to pi)

        print("init BaseControlLoop")

        self._init_SM_to_local()
        print(f"Current base rest pose is {self.base_rest_pose}")

        print("init SHM locally to BaseControlLoop.")

        self.base_odom_pose_lock = Lock()

        # Base init requirements
        self.base.set_ref_ang()

        self.base_odom_pose = np.zeros(3)  # x, y, theta(radians from -pi to pi)
        self._update_base_odom_pose()
        self._update_converter()

        self.run()

    def run(self):
        while True:
            if self.is_base_pos_cmd_arrived[0]:
                self.base_target_pose_is_updated[0] = 0
                print(
                    f"[Base] Trying to reach target_pose_in_mocap: {self.target_base_pose}"
                )

                if self.base_needs_wait[0] == 1:
                    self.base_needs_wait[0] = 0
                    for i in range(5):
                        self._posi_cmd(self.target_base_pose, wait=True)
                else:
                    print(
                        f"self._posi_cmd(self.target_base_pose={self.target_base_pose}, wait=False)"
                    )
                    self._posi_cmd(self.target_base_pose, wait=False)

                self.is_base_pos_cmd_arrived[0] = 0

            elif self.base_needs_resume[0] == 1:
                if self.base._stopped:
                    self.base.go()
                self.base_needs_resume[0] = 0
            elif self.base_needs_stop[0] == 1:
                self.base.stop()
                self.base_needs_stop[0] = 0
            elif self.base_needs_refresh_ref_angle[0] == 1:
                self.base.set_ref_ang()
                self.base_needs_refresh_ref_angle[0] = 0
            elif self.base_needs_set_rest_pose[0] == 1:
                print(f"Set base rest pose to {self.base_rest_pose}")
                self.base_needs_set_rest_pose[0] = 0
            else:
                self._update_converter()

    def _init_SM_to_local(self):

        self.base_target_pose_is_updated_shm, self.base_target_pose_is_updated = (
            init_SHM("base_target_pose_is_updated")
        )
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM(
            "base_mocap_pose"
        )  # x, y, theta(radians from -pi to pi)

        #### BASE [Start] ####
        self.target_base_pose_shm, self.target_base_pose = init_SHM("target_base_pose")
        self.base_rest_pose_shm, self.base_rest_pose = init_SHM("base_rest_pose")

        # response to base_set_rest_pose or base_refresh_ref_angle
        # response to base_move_precise or move_base_and_arm
        self.is_base_pos_cmd_arrived_shm, self.is_base_pos_cmd_arrived = init_SHM(
            "is_base_pos_cmd_arrived"
        )

        self.base_needs_stop_shm, self.base_needs_stop = init_SHM("base_needs_stop")
        self.base_needs_resume_shm, self.base_needs_resume = init_SHM(
            "base_needs_resume"
        )

        self.base_needs_wait_shm, self.base_needs_wait = init_SHM("base_needs_wait")
        self.base_needs_refresh_ref_angle_shm, self.base_needs_refresh_ref_angle = (
            init_SHM("base_needs_refresh_ref_angle")
        )
        self.base_needs_set_rest_pose_shm, self.base_needs_set_rest_pose = init_SHM(
            "base_needs_set_rest_pose"
        )

        #### BASE [End] ####

    def _posi_cmd(self, target_pose_in_mocap, wait=False):
        if self.base._stopped:
            self.base.go()  # resume

        self._update_converter()
        target_pose_in_base_odom = self._convert_pose_in_mocap_2_base_cmd_pose(
            target_pose_in_mocap
        )
        self.base.goto_pose(
            self._convert_pose_in_mocap_2_base_cmd_pose(target_pose_in_mocap)
        )
        """
        time.sleep(0.2)

        if wait:
            max_block_time = 3.0
        else:
            max_block_time = 1.0

        st = time.time()
        while (time.time() - st < max_block_time) and (self.base_target_pose_is_updated[0] == 0):
        """
        time.sleep(1.0)
        # print(" >> " * 10)
        # print(f"[Base] Trying to reach target_pose_in_mocap: {target_pose_in_mocap}")
        # print(" >> " * 10)

        if self.base.close_to_pose(target_pose_in_base_odom):
            self._update_converter()
            target_pose_in_base_odom = self._convert_pose_in_mocap_2_base_cmd_pose(
                target_pose_in_mocap
            )
            self.base.goto_pose(
                self._convert_pose_in_mocap_2_base_cmd_pose(target_pose_in_mocap)
            )
            """
            if self.base.close_to_pose(target_pose_in_base_odom):
                break
            else:
                print(f"[Base] Trying to reach target_pose_in_mocap: {target_pose_in_mocap}")
                time.sleep(0.1)
            """

    def _update_converter(self):
        self._update_base_odom_pose()

        self.coord_frame_converter.update(self.base_mocap_pose, self.base_odom_pose)

    def _update_base_odom_pose(self):
        assert self.base_rest_pose is not None, "Base rest pose is not initialized"

        # Base local2global pose!!!!!
        ix, iy, ia = self.base_rest_pose
        lx, ly, la = (
            self.base.get_pose()
        )  # base local ref pose -- where the base is while driver starting
        pos_x = ix + lx * np.cos(ia) - ly * np.sin(ia)
        pos_y = iy + lx * np.sin(ia) + ly * np.cos(ia)
        ang = ia + la  # in radians

        self.base_odom_pose_lock.acquire()
        self.base_odom_pose = np.array([pos_x, pos_y, ang])
        self.base_odom_pose_lock.release()

    def _convert_pose_in_mocap_2_base_cmd_pose(self, target_pose_in_mocap):
        target_pose_in_odom_global = self.coord_frame_converter.convert_pose(
            target_pose_in_mocap
        )

        assert self.base_rest_pose is not None, "Base rest pose is not initialized"
        ix, iy, ia = self.base_rest_pose
        gx, gy, ga = target_pose_in_odom_global
        pos_x = (gx - ix) * np.cos(ia) + (gy - iy) * np.sin(ia)
        pos_y = -(gx - ix) * np.sin(ia) + (gy - iy) * np.cos(ia)
        ang = ga - ia
        return np.array([pos_x, pos_y, ang])


class KinovaControlLoop(object):
    def __init__(
        self,
        debug=False,
        kinova_ip="192.168.1.10",
        tcp_port=10000,
        freq=30.0,
    ):
        self.debug = debug
        kinova_router = DeviceConnection.createTcpConnection(kinova_ip, tcp_port)
        self.arm = KinovaArm(kinova_router.__enter__(), router_real_time=None)
        print("init Kinova Arm.")

        self._init_SM_to_local()
        print("init SHM  initializing KinovaControlLoop.")

        self.arm_pose_update_lock = Lock()

        if self.debug:
            self.update_info()
            print(f"local_ee_pos: {self.local_ee_pos}")
            print(f"local_ee_ori: {self.local_ee_ori}")

        self.run()

    def _reset_arm(self):
        self._arm_open_gripper(force=True)
        time.sleep(0.8)
        self.is_gripper_close[0] = 0
        self.arm.home()
        self._arm_open_gripper(force=True)
        time.sleep(0.8)

    def _init_SM_to_local(self):
        # Stop Event

        self.is_all_zero_vel_cmd_arrived_shm, self.is_all_zero_vel_cmd_arrived = (
            init_SHM("is_all_zero_vel_cmd_arrived")
        )

        (
            self.enable_arm_vel_cmd_stop_callback_shm,
            self.enable_arm_vel_cmd_stop_callback,
        ) = init_SHM("enable_arm_vel_cmd_stop_callback")
        self.last_arm_vel_cmd_time_shm, self.last_arm_vel_cmd_time = init_SHM(
            "last_arm_vel_cmd_time"
        )
        self.arm_cmd_vel_interval_shm, self.arm_cmd_vel_interval = init_SHM(
            "arm_cmd_vel_interval"
        )

        # Info
        self.arm_fin_gripper_offset_shm, self.arm_fin_gripper_offset = init_SHM(
            "arm_fin_gripper_offset"
        )

        # _, self.base_pose = init_SHM("base_pose")
        self.local_ee_pos_shm, self.local_ee_pos = init_SHM("local_ee_pos")
        self.local_ee_ori_shm, self.local_ee_ori = init_SHM("local_ee_ori")

        self.global_ee_pos_shm, self.global_ee_pos = init_SHM("global_ee_pos")
        self.global_ee_ori_shm, self.global_ee_ori = init_SHM("global_ee_ori")
        self.is_gripper_close_shm, self.is_gripper_close = init_SHM("is_gripper_close")

        # Command
        self.target_arm_xyz_vel_shm, self.target_arm_xyz_vel = init_SHM(
            "target_arm_xyz_vel"
        )
        self.target_arm_ori_deg_vel_shm, self.target_arm_ori_deg_vel = init_SHM(
            "target_arm_ori_deg_vel"
        )
        self.target_arm_pos_shm, self.target_arm_pos = init_SHM("target_arm_pos")
        self.target_arm_ori_deg_shm, self.target_arm_ori_deg = init_SHM(
            "target_arm_ori_deg"
        )
        self.target_gripper_close_shm, self.is_target_gripper_close = init_SHM(
            "is_target_gripper_close"
        )
        self.target_arm_joint_rad_shm, self.target_arm_joint_rad = init_SHM(
            "target_arm_joint_rad"
        )  # (7,)

        # Command status
        self.is_joint_cmd_arrived_shm, self.is_joint_cmd_arrived = init_SHM(
            "is_joint_cmd_arrived"
        )
        self.is_pos_cmd_arrived_shm, self.is_pos_cmd_arrived = init_SHM(
            "is_pos_cmd_arrived"
        )
        self.is_vel_cmd_arrived_shm, self.is_vel_cmd_arrived = init_SHM(
            "is_vel_cmd_arrived"
        )
        self.is_gripper_cmd_arrived_shm, self.is_gripper_cmd_arrived = init_SHM(
            "is_gripper_cmd_arrived"
        )

        self.arm_needs_home_shm, self.arm_needs_home = init_SHM("arm_needs_home")
        self.arm_needs_stop_shm, self.arm_needs_stop = init_SHM("arm_needs_stop")

        self.arm_cmd_vel_interval_shm, self.arm_cmd_vel_interval = init_SHM(
            "arm_cmd_vel_interval"
        )

        #### BASE--MOCAP [START] ####
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM("base_mocap_pose")
        #### BASE--MOCAP [END] ####

    ############### Main busy loop ################
    """
    Please run different control loops in different threads
    """

    # def fire_cmd(self):
    #     while True:
    #         if self.high_freq_cmd_arrived():
    #             self.fire_high_freq_cmd()
    #         elif self.low_freq_cmd_arrived():
    #             self.fire_low_freq_cmd()

    # def refresh_pose(self):
    #     while True:
    #         self.update_info()

    def run(self):
        step = 0
        while True:
            if self.high_freq_cmd_arrived():
                self.fire_high_freq_cmd()
            elif self.low_freq_cmd_arrived():
                self.fire_low_freq_cmd()
            else:
                self.update_info()

            # if self.high_freq_cmd_arrived():
            #     self.fire_high_freq_cmd()
            #     step += 30
            # elif self.low_freq_cmd_arrived():
            #     self.fire_low_freq_cmd()
            #     step += 30
            # else:
            #     if step > 30:
            #         self.update_info()
            #         step = 0
            #     else:
            #         step += 15
            #         # step += 1

    def update_info(self):
        # st = time.time()
        local_ee_pos, local_ee_ori_deg = self.arm._get_robot_pose()

        if self.debug:
            print(
                f"{bcolors.OKGREEN}pos from kinova_new has local_ee_pos.z from KinovaControlLoop =: {local_ee_pos[2]}{bcolors.ENDC}"
            )
        local_ee_ori_rad = local_ee_ori_deg / 180 * np.pi

        self.arm_pose_update_lock.acquire()
        self.local_ee_pos[:] = local_ee_pos
        self.local_ee_ori[:] = local_ee_ori_rad
        self.arm_pose_update_lock.release()

    def high_freq_cmd_arrived(self):
        # Open/close gripper belongs to pos_cmd
        return (
            self.is_pos_cmd_arrived[0]
            or self.is_vel_cmd_arrived[0]
            or self.is_gripper_cmd_arrived[0]
            or self.is_joint_cmd_arrived[0]
            or self.is_all_zero_vel_cmd_arrived[0]
        )

    def low_freq_cmd_arrived(self):
        return self.arm_needs_home[0] or self.arm_needs_stop[0]

    def fire_low_freq_cmd(self):
        if self.arm_needs_home[0] == 1:
            self._arm_open_gripper(force=True)
            self.arm.home()
            self.arm_needs_home[0] = 0
            self.update_info()
        elif self.arm_needs_stop[0]:
            self._arm_stop()
        else:
            raise ValueError("Invalid command arrived.")

    def fire_high_freq_cmd(self):
        if self.is_all_zero_vel_cmd_arrived[0]:
            print(
                f"{bcolors.OKGREEN} recived is_all_zero_vel_cmd_arrived after: {time.time() - self.last_arm_vel_cmd_time[0]} s {bcolors.ENDC}"
            )
            self.arm.command_zero_velocity()
            self.is_all_zero_vel_cmd_arrived[0] = 0
            return

        if self.is_vel_cmd_arrived[0]:
            self._arm_cmd_vel(self.target_arm_xyz_vel, self.target_arm_ori_deg_vel)
            print(
                f"recived new vel cmd after: {time.time() - self.last_arm_vel_cmd_time[0]} seconds"
            )
            self.is_vel_cmd_arrived[0] = 0
            self.enable_arm_vel_cmd_stop_callback[0] = 1
            self.last_arm_vel_cmd_time[0] = time.time()
        elif self.is_pos_cmd_arrived[0]:
            self._arm_move_precise(self.target_arm_pos, self.target_arm_ori_deg)
            self.is_pos_cmd_arrived[0] = 0
            self.update_info()
        elif self.is_joint_cmd_arrived[0]:
            self._arm_move_precise_in_joint_space(dcp(self.target_arm_joint_rad))
            self.is_joint_cmd_arrived[0] = 0
            self.update_info()
        if self.is_gripper_cmd_arrived[0]:
            if self.is_target_gripper_close[0] == 1:
                # Close gripper
                self._arm_close_gripper()
                self.is_gripper_close[0] = 1
            elif self.is_target_gripper_close[0] == 0:
                # Open gripper
                self._arm_open_gripper()
                self.is_gripper_close[0] = 0
            else:
                print(f"self.is_target_gripper_close={self.is_target_gripper_close}")
                raise ValueError("Invalid command arrived.")
            self.is_gripper_cmd_arrived[0] = 0  # False

    ############### Private methods ################
    def _arm_cmd_vel(self, target_arm_pos_vel, target_arm_ori_deg_vel):
        self.arm.command_velocity(target_arm_pos_vel, target_arm_ori_deg_vel)

    def _arm_stop(self):
        self.arm.stop()
        self.arm_needs_stop[0] = 0
        self.update_info()

    def _arm_open_gripper(self, force=False):
        # ON PURPOSE BLOCK THE KINOVA ARM CONTROL -- The Velocity Control should be stopped
        # if (self.is_gripper_close[0]) or force:
        # threading.Thread(target=self.arm.open_gripper).start()
        self.arm.open_gripper()
        # self.update_info()

    def _arm_close_gripper(self, force=False):
        # ON PURPOSE BLOCK THE KINOVA ARM CONTROL -- The Velocity Control should be stopped
        # if (not self.is_gripper_close[0]) or force:
        # threading.Thread(target=self.arm.close_gripper).start()
        self.arm.close_gripper()
        # self.update_info()

    def _arm_move_precise_in_joint_space(self, joint_rad):
        print(f"recived joint_rad: {joint_rad}")
        wait = False
        joints_deg = joint_rad * 180 / np.pi
        run_ok = self.arm.move_angular(joints_deg.tolist(), wait)
        assert run_ok, f"Failed to execute arm_cmd_precise: with joint_rad={joint_rad}"
        self.update_info()

    def _arm_move_precise(self, pos, ang):
        wait = False
        """
        if pos[-1] < -0.19:
            # # hard code for the luggage close_task
            joints_deg = np.array([0.12, 1.99, 3.12, 5.  , 0.24, 1.58, 1.61]) * 180 / np.pi

            # packing
            # joints_deg = np.array([5.45, 1.81, 3.97, 4.32, 2.95, 5.15, 2.  ]) * 180 / np.pi
        else:
        """
        joints_deg = self.arm.compute_ik(pos, ang)  # pos in meter, ang in degree
        assert (
            joints_deg is not None
        ), f"Failed to compute_ik: with pos={pos}, ang={ang}, wait={wait}"

        print(f"joints_deg: {joints_deg}, with pos={pos}, ang={ang}")
        run_ok = self.arm.move_angular(joints_deg.tolist(), wait)
        assert (
            run_ok
        ), f"Failed to execute arm_cmd_precise: with pos={pos}, ang={ang}, wait={wait}"
        self.update_info()


class StopKinovaMovement(object):
    def __init__(self):
        self._init_SM_to_local()

        # set the flag to the default value
        self.arm_needs_stop[0] = 0
        self.enable_arm_vel_cmd_stop_callback[0] = 0
        self.arm_cmd_vel_interval[0] = 0.1
        self.is_ready_for_next_arm_vel_cmd[0] = 1

        # run the main loop
        self.run()

    def _init_SM_to_local(self):
        (
            self.enable_arm_vel_cmd_stop_callback_shm,
            self.enable_arm_vel_cmd_stop_callback,
        ) = init_SHM("enable_arm_vel_cmd_stop_callback")
        self.last_arm_vel_cmd_time_shm, self.last_arm_vel_cmd_time = init_SHM(
            "last_arm_vel_cmd_time"
        )
        self.arm_cmd_vel_interval_shm, self.arm_cmd_vel_interval = init_SHM(
            "arm_cmd_vel_interval"
        )
        self.is_ready_for_next_arm_vel_cmd_shm, self.is_ready_for_next_arm_vel_cmd = (
            init_SHM("is_ready_for_next_arm_vel_cmd")
        )

        self.is_all_zero_vel_cmd_arrived_shm, self.is_all_zero_vel_cmd_arrived = (
            init_SHM("is_all_zero_vel_cmd_arrived")
        )

        self.arm_needs_stop_shm, self.arm_needs_stop = init_SHM("arm_needs_stop")

    def run(self):
        while True:
            if self.enable_arm_vel_cmd_stop_callback[0] == 1:
                elapsed_time = time.time() - self.last_arm_vel_cmd_time[0]

                # update `arm_needs_stop` flag --  when no new command is ready
                if elapsed_time > 4.0 * self.arm_cmd_vel_interval:
                    print("Stopping velocity control due to timeout.")
                    # self.is_vel_cmd_arrived[0] = 1
                    self.is_all_zero_vel_cmd_arrived[0] = 1
                    # disable the callback when the arm is stopped
                    self.enable_arm_vel_cmd_stop_callback[0] = 0

            time.sleep(0.05)


class GlobalEEFUpdateLoop(object):
    """
    ONLY Ready local_ee_pos, local_ee_ori, base_mocap_pose
    Compute (ONLY WRITE TO) global_ee_pos, global_ee_ori
    """

    def __init__(
        self,
        debug=False,
    ):
        self.debug = debug

        self._init_SM_to_local()

        if debug:
            self._init_fin_gripper_offset()

        print("init SHM locally to GlobalEEFUpdateLoop.")

        self.arm_pose_write_lock = Lock()

        if self.debug:
            print(
                f"<GlobalEEFUpdateLoop> fake arm_fin_gripper_offset: {self.arm_fin_gripper_offset}"
            )
            print(f"<GlobalEEFUpdateLoop> local_ee_pos: {self.local_ee_pos}")
            print(f"<GlobalEEFUpdateLoop>  local_ee_ori: {self.local_ee_ori}")
            print(f"<GlobalEEFUpdateLoop>  base_mocap_pose: {self.base_mocap_pose}")
        else:
            print("init GlobalEEFUpdateLoop with debug=False.")
        self.run()

    def _init_fin_gripper_offset(self):
        # self.arm_fin_gripper_offset[0] = 0.0
        # self.arm_fin_gripper_offset[1] = -0.015
        # self.arm_fin_gripper_offset[2] = 0.09

        # AT LEAST, FOR SRC LAUNDRY SETUP
        self.arm_fin_gripper_offset[0] = 0.0
        self.arm_fin_gripper_offset[1] = -0.02
        self.arm_fin_gripper_offset[2] = 0.15

    def _init_SM_to_local(self):
        # Info
        self.arm_fin_gripper_offset_shm, self.arm_fin_gripper_offset = init_SHM(
            "arm_fin_gripper_offset"
        )

        # Functionality
        self.local_ee_pos_shm, self.local_ee_pos = init_SHM("local_ee_pos")
        self.local_ee_ori_shm, self.local_ee_ori = init_SHM("local_ee_ori")

        self.global_ee_pos_shm, self.global_ee_pos = init_SHM("global_ee_pos")
        self.global_ee_ori_shm, self.global_ee_ori = init_SHM("global_ee_ori")

        self.global_ee_fin_gripper_pos_shm, self.global_ee_fin_gripper_pos = init_SHM(
            "global_ee_fin_gripper_pos"
        )
        self.global_ee_fin_gripper_ori_shm, self.global_ee_fin_gripper_ori = init_SHM(
            "global_ee_fin_gripper_ori"
        )

        self.is_gripper_close_shm, self.is_gripper_close = init_SHM("is_gripper_close")

        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM("base_mocap_pose")

        # SAFETY Concern
        self.arm_needs_stop_shm, self.arm_needs_stop = init_SHM("arm_needs_stop")

        self.base_needs_stop_shm, self.base_needs_stop = init_SHM("base_needs_stop")

    def run(self):
        listener = Listener(("0.0.0.0", 6050), authkey=b"123456")
        while True:
            print("[0.0.0.0:6050] waiting for connection...")
            conn = listener.accept()
            print("[0.0.0.0:6050] connected.")
            try:
                while True:
                    if conn.poll():
                        req = conn.recv()
                        if req["type"] == "get_global_eef_pose_with_gripper_status":
                            conn.send(self._global_eef_pose_with_gripper_status_dict)
                    else:
                        self._compute_global_ee_pose()
            except (ConnectionResetError, EOFError):
                print("[0.0.0.0:6050] disconnected.")
                self.arm_needs_stop[0] = 1
                self.base_needs_stop[0] = 1

    def _compute_global_ee_pose(self):
        # Change deg to rad
        local_ee_pos = dcp(self.local_ee_pos)
        local_ee_ori = dcp(self.local_ee_ori)
        base_mocap_pose = dcp(self.base_mocap_pose)

        global_ee_pos, global_ee_ori, _ = local_to_global_pose(
            local_ee_pos[:],
            local_ee_ori[:],
            base_xy=base_mocap_pose[0:2],
            base_rot=base_mocap_pose[2],
            height_offset=hardware.ARM_MOUNTING_HEIGHT,
        )

        if self.debug:
            print(
                f"{bcolors.WARNING}pos from kinovaControlLoop has local_ee_pos.z=: {local_ee_pos[2]}{bcolors.ENDC}"
            )

        ##############################
        # Global Kinova EEF Pose
        self.global_ee_pos[:] = global_ee_pos
        self.global_ee_ori[:] = global_ee_ori
        ##############################

        if self.debug:
            print(
                f"{bcolors.WARNING}pos from kinovaControlLoop has global_ee_pos.z=: {global_ee_pos[2]}{bcolors.ENDC}"
            )

        ##############################
        # Global Kinova EEF + Fin Gripper Pose
        global_ee_fin_gripper_pos = dcp(global_ee_pos)
        global_ee_fin_gripper_ori = dcp(global_ee_ori)

        R = quat2mat(pybullet.getQuaternionFromEuler(global_ee_ori))
        gripper_x_dir = R[:, 0]
        gripper_y_dir = R[:, 1]
        gripper_z_dir = R[:, 2]

        # Add the offset
        global_ee_fin_gripper_pos += (
            self.arm_fin_gripper_offset[2] * gripper_z_dir
        )  # 0.09
        global_ee_fin_gripper_pos += (
            self.arm_fin_gripper_offset[1] * gripper_y_dir
        )  # -0.015
        global_ee_fin_gripper_pos += (
            self.arm_fin_gripper_offset[0] * gripper_x_dir
        )  # 0.0
        ##############################

        self.arm_pose_write_lock.acquire()
        self.global_ee_fin_gripper_pos[:] = global_ee_fin_gripper_pos
        self.global_ee_fin_gripper_ori[:] = global_ee_fin_gripper_ori
        self.arm_pose_write_lock.release()

        # if self.debug:
        #     print(f"{bcolors.FAIL}pos from kinovaControlLoop has global_ee_fin_gripper_pos.z=: {global_ee_fin_gripper_pos[2]} @ gripper_z_dir={gripper_z_dir}. {bcolors.ENDC}")

        if self.debug:
            print(f"base_mocap_pose: {base_mocap_pose}")
            print(f"global_ee_pos: {global_ee_pos}")
            print(f"global_ee_ori: {global_ee_ori}")
            print(f"global_ee_fin_gripper_pos: {global_ee_fin_gripper_pos}")
            print(f"global_ee_fin_gripper_ori: {global_ee_fin_gripper_ori}")
            print(f"local_ee_pos: {local_ee_pos}")
            print(f"local_ee_ori: {local_ee_ori}")
            print(f"arm_fin_gripper_offset: {self.arm_fin_gripper_offset}")

    @property
    def _global_eef_pose_with_gripper_status_dict(self):
        return dict(
            base_pose=dcp(self.base_mocap_pose),
            global_ee_posi=dcp(self.global_ee_fin_gripper_pos),
            local_ee_posi=dcp(self.local_ee_pos),
            local_ee_ori=dcp(self.local_ee_ori),
            global_ee_ori=dcp(self.global_ee_fin_gripper_ori),
            gripper_status=dcp(self.is_gripper_close),  # 1: closed, 0: open
        )


class RobotPyServerMP(object):
    """
    response client request and save the command to a shared memory.
    """

    def __init__(
        self,
        arm_only=False,
    ):
        machine_name = socket.gethostname()
        base_name = machine_name[-4:]  # 'bot1' or 'bot2' or 'bot3'
        assert base_name[:3] == "bot", "Machine name should start with bot"
        self.base_name = base_name

        self.arm_only = arm_only

        self._init_SM_to_local()

        print(
            f"Inited pose_dict from SHM: {self._global_eef_pose_with_gripper_status_dict}"
        )

        self.arm_pose_get_lock = Lock()

        self.arm_cmd_send_lock = Lock()

        self.base_cmd_send_lock = Lock()

        self.env_name = None

        self.reset()

        try:
            self.run()
        finally:
            create_all_sm_topics(just_close=True)

    def _init_SM_to_local(self):
        self.arm_fin_gripper_offset_shm, self.arm_fin_gripper_offset = init_SHM(
            "arm_fin_gripper_offset"
        )
        # Info
        self.local_ee_pos_shm, self.local_ee_pos = init_SHM("local_ee_pos")
        self.local_ee_ori_shm, self.local_ee_ori = init_SHM("local_ee_ori")

        self.global_ee_pos_shm, self.global_ee_pos = init_SHM("global_ee_pos")
        self.global_ee_ori_shm, self.global_ee_ori = init_SHM("global_ee_ori")

        self.global_ee_fin_gripper_pos_shm, self.global_ee_fin_gripper_pos = init_SHM(
            "global_ee_fin_gripper_pos"
        )
        self.global_ee_fin_gripper_ori_shm, self.global_ee_fin_gripper_ori = init_SHM(
            "global_ee_fin_gripper_ori"
        )

        self.is_gripper_close_shm, self.is_gripper_close = init_SHM("is_gripper_close")

        # Command
        self.target_arm_xyz_vel_shm, self.target_arm_xyz_vel = init_SHM(
            "target_arm_xyz_vel"
        )
        self.target_arm_ori_deg_vel_shm, self.target_arm_ori_deg_vel = init_SHM(
            "target_arm_ori_deg_vel"
        )
        self.target_arm_pos_shm, self.target_arm_pos = init_SHM("target_arm_pos")
        self.target_arm_ori_deg_shm, self.target_arm_ori_deg = init_SHM(
            "target_arm_ori_deg"
        )
        self.target_gripper_close_shm, self.is_target_gripper_close = init_SHM(
            "is_target_gripper_close"
        )
        self.target_arm_joint_rad_shm, self.target_arm_joint_rad = init_SHM(
            "target_arm_joint_rad"
        )  # (7,)

        # Command status
        self.is_joint_cmd_arrived_shm, self.is_joint_cmd_arrived = init_SHM(
            "is_joint_cmd_arrived"
        )
        self.is_pos_cmd_arrived_shm, self.is_pos_cmd_arrived = init_SHM(
            "is_pos_cmd_arrived"
        )
        self.is_vel_cmd_arrived_shm, self.is_vel_cmd_arrived = init_SHM(
            "is_vel_cmd_arrived"
        )
        self.is_gripper_cmd_arrived_shm, self.is_gripper_cmd_arrived = init_SHM(
            "is_gripper_cmd_arrived"
        )

        self.arm_needs_home_shm, self.arm_needs_home = init_SHM("arm_needs_home")
        self.arm_needs_stop_shm, self.arm_needs_stop = init_SHM("arm_needs_stop")

        self.arm_cmd_vel_interval_shm, self.arm_cmd_vel_interval = init_SHM(
            "arm_cmd_vel_interval"
        )

        #### BASE--MOCAP [START] ####
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM("base_mocap_pose")
        #### BASE--MOCAP [END] ####

        #### BASE [Start] ####
        self.base_target_pose_is_updated_shm, self.base_target_pose_is_updated = (
            init_SHM("base_target_pose_is_updated")
        )

        self.target_base_pose_shm, self.target_base_pose = init_SHM("target_base_pose")
        self.base_rest_pose_shm, self.base_rest_pose = init_SHM("base_rest_pose")

        # response to base_set_rest_pose or base_refresh_ref_angle
        # response to base_move_precise or move_base_and_arm
        self.is_base_pos_cmd_arrived_shm, self.is_base_pos_cmd_arrived = init_SHM(
            "is_base_pos_cmd_arrived"
        )

        self.base_needs_stop_shm, self.base_needs_stop = init_SHM("base_needs_stop")
        self.base_needs_resume_shm, self.base_needs_resume = init_SHM(
            "base_needs_resume"
        )

        self.base_needs_wait_shm, self.base_needs_wait = init_SHM("base_needs_wait")
        self.base_needs_refresh_ref_angle_shm, self.base_needs_refresh_ref_angle = (
            init_SHM("base_needs_refresh_ref_angle")
        )
        self.base_needs_set_rest_pose_shm, self.base_needs_set_rest_pose = init_SHM(
            "base_needs_set_rest_pose"
        )

        #### BASE [End] ####

    @property
    def _global_eef_pose_with_gripper_status_dict(self):
        return dict(
            base_pose=dcp(self.base_mocap_pose),
            local_ee_posi=dcp(self.local_ee_pos),
            local_ee_ori=dcp(self.local_ee_ori),
            global_ee_posi=dcp(self.global_ee_fin_gripper_pos),
            global_ee_ori=dcp(self.global_ee_fin_gripper_ori),
            gripper_status=dcp(self.is_gripper_close),  # 1: closed, 0: open
        )

    def reset(self):
        # Task level primitive
        self.env_name = None

        self._t = 0
        self._clip_num = 0

        # Define Safety Workspace
        if self.env_name == "laundry_load":
            # laundry_center = np.array([3.78, -2.83, 0.55])   # in y-z plane
            self.arm_workspace = dict(
                arm_limit=None,
            )
        else:
            self.arm_workspace = dict(
                arm_limit=None,
            )

        self.base_workspace = dict(
            corner_xys=None,
            workspace_poly=None,
        )

    def _arm_update_workspace(self, arm_limit):
        self.arm_workspace["arm_limit"] = arm_limit.reshape(3, 2)

    def _base_update_workspace(self, corner_xys):
        self.base_workspace["corner_xys"] = corner_xys
        self.base_workspace["workspace_poly"] = Polygon(corner_xys)

    def _apply_clip_to_target_eef_pose(self, raw_eef_pose):
        target_eef_pose = dcp(raw_eef_pose)

        target_eef_xyz = target_eef_pose[:3]
        target_eef_euler = target_eef_pose[3:]

        if self.env_name == "laundry_load":
            if target_eef_xyz[0] > 3.7:
                target_eef_euler[0] = np.clip(
                    target_eef_euler[-1], np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6
                )
            # enforce no rotation about Y axis
            target_eef_euler[1] = 0.0
            # clip the rotation about Z axis
            target_eef_euler[-1] = np.clip(
                target_eef_euler[-1], np.pi / 2, np.pi / 2 + np.pi / 3
            )

            # project yz in a workspace

        ##############################
        # Find target FinGripper Rotation Matrix
        R = quat2mat(pybullet.getQuaternionFromEuler(target_eef_euler))
        gripper_x_dir = R[:, 0]
        gripper_y_dir = R[:, 1]
        gripper_z_dir = R[:, 2]

        ##############################
        # Update arm_fin_gripper_offset for complex tasks
        # make a local copy of the `arm_fin_gripper_offset`
        arm_fin_gripper_offset = dcp(self.arm_fin_gripper_offset)

        if self.env_name == "laundry_load":
            laundry_center = np.array([3.75, -2.83, 0.55])  # in y-z plane

            if target_eef_xyz[1] < laundry_center[1]:
                arm_fin_gripper_offset[0] = 0.1
            else:
                arm_fin_gripper_offset[0] = -0.1

            if target_eef_xyz[2] < laundry_center[2]:
                arm_fin_gripper_offset[1] = -0.035
            else:
                arm_fin_gripper_offset[1] = 0.0

            arm_fin_gripper_offset[2] = 0.18
        else:
            if gripper_z_dir[-1] > 0:
                arm_fin_gripper_offset[1] = -0.035
            else:
                arm_fin_gripper_offset[1] = 0.0

        ##############################
        # Convert Kinova EEF Posi to Kinova+FinGripper EEF Posi
        target_eef_fin_gripper_posi = dcp(target_eef_xyz)

        target_eef_fin_gripper_posi += arm_fin_gripper_offset[2] * gripper_z_dir  # 0.09
        target_eef_fin_gripper_posi += (
            arm_fin_gripper_offset[1] * gripper_y_dir
        )  # -0.015
        target_eef_fin_gripper_posi += arm_fin_gripper_offset[0] * gripper_x_dir  # 0.0
        ##############################

        ##############################
        # xyz in m, axis-angle in rad
        arm_limit = self.arm_workspace["arm_limit"]

        """
        if self.env_name == "load_shoes":
            if target_eef_fin_gripper_posi[0] > 3.4:
                # Prevent the robot from hitting the top of the shelf
                arm_limit[2, 0] = 0.49
        """

        additional_clip = False
        if self.env_name == "laundry_load":
            if target_eef_fin_gripper_posi[0] > 3.9:
                additional_clip = True
                # arm_limit[1, 0] = -2.94 # -2.83 - 0.11
                # arm_limit[1, 1] = -2.72 # -2.83 + 0.11

                arm_limit[0, 0] = 3.7  # -2.83 - 0.11
                arm_limit[0, 1] = 4.25  # -2.83 + 0.11

                arm_limit[1, 0] = -2.92  # -2.83 - 0.11
                arm_limit[1, 1] = -2.60  # -2.83 + 0.11
                arm_limit[2, 0] = 0.41  #
                arm_limit[2, 1] = 0.7  #

        # Clip Kinova+FinGripper EEF Posi with Arm Limit
        for i, key_str in enumerate(["x", "y", "z"]):
            raw_value = target_eef_fin_gripper_posi[i]
            new_value = np.clip(raw_value, arm_limit[i, 0], arm_limit[i, 1])
            target_eef_fin_gripper_posi[i] = new_value

            # if raw_value not close to new_value, print the info
            if np.abs(raw_value - new_value) > 1e-3:
                print(f"for target_eef_fin_gripper_posi={target_eef_fin_gripper_posi}")
                print(f"Clip {key_str} direction: {raw_value} -> {new_value}")
                if self.env_name != "laundry_load":
                    print(f"Consider to adjust arm limits to fin gripper offset!!!")

        if additional_clip:
            # check whether target_eef_fin_gripper_posi is in the self.arm_workspace.yz_plane_poly
            target_yz = np.array(
                [target_eef_fin_gripper_posi[1], target_eef_fin_gripper_posi[2]]
            )
            y_center = laundry_center[1]
            z_center = laundry_center[2]
            R = 0.05
            yz_plane_poly = Polygon(
                [
                    [y_center - R, z_center],
                    [y_center, z_center - R],
                    [y_center + R, z_center],
                    [y_center, z_center + R],
                ]
            )
            # if not, project it to the nearest point on the plane
            if not yz_plane_poly.contains(Point(target_yz)):
                new_target_yz = np.array(
                    yz_plane_poly.exterior.interpolate(
                        yz_plane_poly.exterior.project(Point(target_yz))
                    ).coords[0]
                )
                # print(f"for target_yz={target_yz}, we have new_target_yz={new_target_yz} at x={target_eef_fin_gripper_posi[0]} and global self.arm_fin_gripper_offset = {self.arm_fin_gripper_offset}, local arm_fin_gripper_offset = {arm_fin_gripper_offset}")
                target_eef_fin_gripper_posi[1:3] = new_target_yz

                self._clip_num += 1
                if self._clip_num > 0:
                    print(
                        f"[{self.env_name}] additional self._clip_num/self._t: {self._clip_num} / {self._t}!!!"
                    )

        # # # Adjust arm limit for specific tasks -- avoid collision
        # if self.env_name == "laundry_load":
        #     if target_eef_fin_gripper_posi[0] > 3.7:
        #         arm_limit[1, 0] = -2.94 # -2.83 - 0.11
        #         arm_limit[1, 1] = -2.72 # -2.83 + 0.11
        #         arm_limit[2, 0] = 0.41  #
        #         arm_limit[2, 1] = 0.7  #

        ##############################

        ##############################
        # Convert back to Kinova EEF Posi to control the robot
        target_eef_posi = dcp(target_eef_fin_gripper_posi)
        target_eef_posi -= arm_fin_gripper_offset[2] * gripper_z_dir  # 0.09
        target_eef_posi -= arm_fin_gripper_offset[1] * gripper_y_dir  # -0.015
        target_eef_posi -= arm_fin_gripper_offset[0] * gripper_x_dir  # 0.0
        ##############################

        target_eef_pose[:3] = target_eef_posi
        target_eef_pose[-3:] = target_eef_euler

        return target_eef_pose

    def _find_next_base_pose(self, target_eef_posi, close_to_eef=False):
        current_base_pose = dcp(self.base_mocap_pose)

        if self.env_name in ["make_bed", "unfold"]:
            workspace = self.base_workspace["workspace_poly"]
            new_base_xy = np.array(
                workspace.exterior.interpolate(
                    workspace.exterior.project(Point(target_eef_posi[:2]))
                ).coords[0]
            )

            new_base_pose = current_base_pose
            new_base_pose[:2] = new_base_xy
        else:
            if self.env_name in ["laundry_door", "load_shoes"]:
                arm_reaching_recommend = 0.6
                arm_reaching_min = 0.5
                arm_reaching_max = 0.7
            elif self.env_name == "laundry_load":
                # arm_reaching_recommend = 0.63
                # arm_reaching_min = 0.5
                # arm_reaching_max = 0.7

                arm_reaching_recommend = 0.73
                arm_reaching_min = 0.55
                arm_reaching_max = 0.8
            elif self.env_name == "close_luggage":
                arm_reaching_recommend = 0.6
                arm_reaching_min = 0.45
                arm_reaching_max = 0.75
                if close_to_eef:
                    arm_reaching_recommend = 0.5
                    arm_reaching_min = 0.45
                    arm_reaching_max = 0.55
            else:
                arm_reaching_recommend = 0.75
                arm_reaching_min = 0.6
                arm_reaching_max = 0.9

            # Currently, Only plan the base movement in the x-y plane
            base_motion_info = move_robot_base_in_xy(
                workspace=self.base_workspace["workspace_poly"],
                target_eef_xy=target_eef_posi[:2],
                cur_base_xy=current_base_pose[:2],
                arm_reaching_recommend=arm_reaching_recommend,
                arm_reaching_min=arm_reaching_min,
                arm_reaching_max=arm_reaching_max,
            )
            new_base_pose = dcp(current_base_pose)
            new_base_pose[:2] = base_motion_info["new_base_xy"]
            new_dist = base_motion_info["new_dist"]
            print(
                f"new_base_pose: {new_base_pose} vs the current_base_pose: {current_base_pose}, --> new_dist: {new_dist} away from the target_eef_posi: {target_eef_posi}"
            )

        # Note: In SRC, when base facing the <negative> x-axis, the base_pose[2] is 0.0
        return new_base_pose

    @property
    def _allow_to_control(self):
        if all(
            [self.arm_workspace[key] is not None for key in self.arm_workspace.keys()]
        ):
            if self.arm_only:
                return True
            else:
                if all(
                    [
                        self.base_workspace[key] is not None
                        for key in self.base_workspace.keys()
                    ]
                ):
                    return True
        return False

    def run(self):
        hostname = "0.0.0.0"
        port = 6040
        listener = Listener((hostname, port), authkey=b"123456")
        while True:
            print(f"[{hostname}:{port}] waiting for connection...")
            conn = listener.accept()
            print(f"[{hostname}:{port}] connected.")
            # resume base and arm
            self.base_needs_resume[0] = 1
            time.sleep(0.1)
            while True:
                try:
                    if conn.poll():
                        req = conn.recv()
                        data = True
                        rtype = req["type"]
                        if rtype == "get_global_eef_pose_with_gripper_status":
                            data = self._global_eef_pose_with_gripper_status_dict
                        elif rtype == "track_eef_traj":
                            assert (
                                self._allow_to_control
                            ), "[NO WORKSPACED HAS FOUND]: does not allow to control the robot."

                            target_eef_traj = req[
                                "target_eef_traj"
                            ]  # (ac_horizon+1, 6)
                            gripper_status_traj = req[
                                "gripper_status_traj"
                            ]  # (ac_horizon+1, 1)
                            step_interval_traj = req[
                                "step_interval_traj"
                            ]  # (ac_horizon, 1)
                            arm_control_freq = req["arm_control_freq"]  # default 125
                            fix_base = req["fix_base"]  # default False
                            # kp_traj = req["kp_traj"]    # (ac_horizon, 3)
                            # There is a inner loop in self._track_eef_traj !!!
                            self._track_eef_traj(
                                target_eef_traj,
                                gripper_status_traj,
                                step_interval_traj,
                                arm_control_freq,
                                fix_base,
                            )
                        elif rtype == "base_update_workspace":
                            self._base_update_workspace(req["corner_xys"])
                        elif rtype == "arm_update_workspace":
                            self._arm_update_workspace(req["arm_limit"])
                        elif rtype == "update_env_name":
                            self.env_name = req["env_name"]
                            assert self.env_name in [
                                "laundry_door",
                                "load_shoes",
                                "close_luggage",
                                "packing",
                                "laundry_load",
                                "lift",
                                "make_bed",
                                "push_chair",
                                "fold",
                                "unfold",
                                "teleop",
                            ], f"Invalid env_name: {self.env_name}"

                        elif rtype == "base_move_precise":
                            self.target_base_pose[:] = req["target_base_pose"]
                            self.base_needs_wait[0] = 1
                            self.base_target_pose_is_updated[0] = 1
                            self.is_base_pos_cmd_arrived[0] = 1
                        elif rtype == "arm_open_gripper":
                            if self.is_gripper_close[0] == 1:
                                self.is_target_gripper_close[0] = 0  # False
                                self.is_gripper_cmd_arrived[0] = 1
                        elif rtype == "arm_close_gripper":
                            if self.is_gripper_close[0] == 0:
                                self.is_target_gripper_close[0] = 1  # True
                                self.is_gripper_cmd_arrived[0] = 1
                        elif rtype == "arm_move_precise_in_joint_space":
                            self.target_arm_joint_rad[:] = req["joint_rad"]
                            self.is_joint_cmd_arrived[0] = 1
                        elif rtype == "arm_move_precise":
                            self.target_arm_pos[:] = req["data"][:3]
                            self.target_arm_ori_deg[:] = req["data"][3:6]
                            self.is_pos_cmd_arrived[0] = 1
                        else:
                            data = self.other_req(req)
                        conn.send(data)
                except (ConnectionResetError, EOFError):
                    self.arm_needs_stop[0] = 1
                    self.base_needs_stop[0] = 1
                    self.reset()
                    break
            print(f"[{hostname}:{port}] disconnected.")

    def _track_eef_traj(
        self,
        target_eef_traj,
        gripper_status_traj,
        step_interval_traj,
        arm_control_freq=125,
        fix_base=False,
    ):
        """
        target_eef_traj: (ac_horizon+1, 6)
        gripper_status_traj: (ac_horizon+1, 1)
        step_interval_traj: (ac_horizon, 1)
        kp_traj: (ac_horizon, 3)
        """
        st = time.time()

        if self.env_name in ["teleop"]:
            fix_base = True

        num_ac_horizon = target_eef_traj.shape[0] - 1
        # Insert the current eef_pose and current_gripper_status to the beginning of the traj
        target_eef_traj[0] = np.concatenate(
            [
                dcp(self.global_ee_fin_gripper_pos[:]),
                dcp(self.global_ee_fin_gripper_ori[:]),
            ],
            axis=-1,
        )
        gripper_status_traj[0] = dcp(self.is_gripper_close[0])

        print(f"[In _track_eef_traj()] target_eef_traj: {target_eef_traj}")
        print(f"[In _track_eef_traj()] gripper_status_traj: {gripper_status_traj}")

        if self.env_name in ["close_luggage"]:  # load_shoes
            gripper_status_traj = np.zeros_like(gripper_status_traj)

        ##############################
        # Change the traj for Kinova+FinGripper EEF to Kinova self EEF
        target_eef_traj_without_gripper_offset = dcp(target_eef_traj)

        for i in range(num_ac_horizon + 1):
            R = quat2mat(pybullet.getQuaternionFromEuler(target_eef_traj[i, 3:]))
            gripper_x_dir = R[:, 0]
            gripper_y_dir = R[:, 1]
            gripper_z_dir = R[:, 2]

            # Minus the offset!!!
            target_eef_traj_without_gripper_offset[i, :3] -= (
                self.arm_fin_gripper_offset[2] * gripper_z_dir
            )  # -1 * 0.09
            target_eef_traj_without_gripper_offset[i, :3] -= (
                self.arm_fin_gripper_offset[1] * gripper_y_dir
            )  # -1 * -0.015
            target_eef_traj_without_gripper_offset[i, :3] -= (
                self.arm_fin_gripper_offset[0] * gripper_x_dir
            )  # -1 * 0.0
        ##############################

        info = generate_arm_high_freq_traj(
            target_eef_traj_without_gripper_offset,
            gripper_status_traj,
            step_interval_traj,
            arm_control_freq,
            re_assign_time=False,
            #    re_assign_time=False if self.env_name in ["teleop"] else True,
        )

        high_freq_pose_traj = info["high_freq_pose_traj"]  # (num_pts, 6)

        # array([  0,  30,  50,  71, 100, 129, 150, 170, 199])
        kpt_high_freq_indices = info["kpt_high_freq_indices"]  # (ac_horizon+1, )

        gripper_switch_ac_idxs = info[
            "gripper_switch_ac_idxs"
        ]  # (num_gripper_switch_kpts, )

        gripper_switch_kpt_posi_list = info[
            "gripper_switch_kpt_posi_list"
        ]  # (num_gripper_switch_kpts, 1)

        if not fix_base:
            # Store the desired_base_pose into a list
            base_pose_movement_plan = dict(
                finish_by_ac_idxs=[],  # ( np.min(1, num_gripper_switch_kpts), )
                planed_base_poses=[],
            )

            close_to_eef = False
            if self.env_name in ["close_luggage"]:
                if high_freq_pose_traj[-1, 2] > 0.5:
                    close_to_eef = True

            target_base_pose = self._find_next_base_pose(
                target_eef_posi=high_freq_pose_traj[-1, :3], close_to_eef=close_to_eef
            )
            # If target_base_pose equals to the current base_pose, then do not append it to the base_pose_traj
            if not np.allclose(target_base_pose, self.base_mocap_pose[:]):
                base_pose_movement_plan["finish_by_ac_idxs"].append(num_ac_horizon - 1)
                base_pose_movement_plan["planed_base_poses"].append(target_base_pose)

            print(f"base_pose_movement_plan: {base_pose_movement_plan}")

        delta_t = 1.0 / arm_control_freq
        prev_kpt_idx = 0

        prev_error = np.zeros(6)

        # if self.env_name in ["teleop"]:
        #     kpt_high_freq_indices = kpt_high_freq_indices[1:]
        #     print(f"skip the first keypoint for teleop!! Now, kpt_high_freq_indices: {kpt_high_freq_indices}")

        for i, kpt_idx in enumerate(kpt_high_freq_indices):
            # Move The Base
            # if not fix_base:
            #     if self.env_name in ["close_luggage"]:
            #         if high_freq_pose_traj[kpt_idx][2] < 0.135:
            #             fix_base = True

            if not fix_base:
                if self.env_name in ["lift"]:
                    if i == 0:
                        planed_base_pose = dcp(self.base_mocap_pose)
                        planed_base_pose[2] = -1.57
                        # planed_base_pose[0] = target_eef_traj[-1, 0]
                        if target_eef_traj[-1, 1] - planed_base_pose[1] > 0.1:
                            planed_base_pose[1] = target_eef_traj[-1, 1] - 0.1
                            # clip the y direction to avoid collision
                            # < -0.556
                            if planed_base_pose[1] > -0.45:
                                planed_base_pose[1] = -0.45

                        self.base_cmd_send_lock.acquire()
                        self.target_base_pose[:] = planed_base_pose
                        self.base_target_pose_is_updated[0] = 1
                        self.base_needs_wait[0] = 0
                        self.is_base_pos_cmd_arrived[0] = 1
                        self.base_cmd_send_lock.release()
                else:
                    if len(base_pose_movement_plan["planed_base_poses"]) > 0:
                        if i < base_pose_movement_plan["finish_by_ac_idxs"][0]:
                            base_pose_movement_plan["finish_by_ac_idxs"].pop(0)
                            planed_base_pose = base_pose_movement_plan[
                                "planed_base_poses"
                            ].pop(0)
                            self.base_cmd_send_lock.acquire()

                            if self.env_name in ["laundry_load"]:
                                planed_base_pose[2] = -3.14

                            self.target_base_pose[:] = planed_base_pose
                            self.base_target_pose_is_updated[0] = 1
                            self.base_needs_wait[0] = 0
                            self.is_base_pos_cmd_arrived[0] = 1
                            self.base_cmd_send_lock.release()

            # Move The Arm
            if i == 0:
                prev_kpt_idx = kpt_idx + int((time.time() - st) * arm_control_freq)
                print(
                    f"[At i ==0 before clip] prev_kpt_idx, last_idx: { prev_kpt_idx}, {(kpt_high_freq_indices[1]-1)}"
                )
                prev_kpt_idx = np.clip(prev_kpt_idx, 0, kpt_high_freq_indices[1] - 1)
                print(
                    f"[At i ==0] prev_kpt_idx/last_idx: { prev_kpt_idx / (kpt_high_freq_indices[1]-1)}"
                )
                st = time.time()

                if self.env_name in ["laundry_load"]:
                    if i in gripper_switch_ac_idxs:
                        # if self.env_name in ["close_luggage"]
                        # Stop the arm and base for gripper switch
                        self.base_needs_stop[0] = 1
                        self.arm_needs_stop[0] = 1
                        time.sleep(0.1)  # Wait for the arm and base to respond
                        # print(f"self.is_target_gripper_close={self.is_target_gripper_close}")
                        # print(f"gripper_status_traj={gripper_status_traj}")
                        self.is_target_gripper_close[0] = gripper_status_traj[i, 0]
                        self.is_gripper_cmd_arrived[0] = 1
                        time.sleep(0.8)
                        self.base_needs_resume[0] = 1
                        time.sleep(0.1)  # Wait for the arm and base to respond
                continue

            # Move the arm from prev_kpt_idx to kpt_idx
            for ii, idx in enumerate(range(prev_kpt_idx, kpt_idx)):
                st = time.time()
                # if self.env_name in ["laundry_load", "laundry_door", "make_bed", "load_shoes", "fold", "lift", "unfold", "teleop"]:
                if self.env_name in [
                    "close_luggage",
                    "laundry_load",
                    "laundry_door",
                    "make_bed",
                    "load_shoes",
                    "fold",
                    "lift",
                    "unfold",
                    "teleop",
                ]:
                    # target_eef_pose = high_freq_pose_traj[idx]
                    target_eef_pose = high_freq_pose_traj[kpt_idx]
                else:
                    # target_eef_pose = high_freq_pose_traj[kpt_idx]
                    target_eef_pose = high_freq_pose_traj[idx]

                while time.time() - st < 0.95 * delta_t:
                    time.sleep(0.0001)

                # Use target_eef_pose and current global_ee_pos + global_ee_ori to compute the target_arm_pos and target_arm_ori_deg
                cur_eef_pose = np.concatenate(
                    [dcp(self.global_ee_pos), dcp(self.global_ee_ori)], axis=-1
                )

                # Apply clip onto the target_eef_pose, based on self.arm_workspace["z_limit"]
                target_eef_pose = self._apply_clip_to_target_eef_pose(target_eef_pose)

                alpha = self.base_mocap_pose[-1]

                PD_control = True

                error = self._find_vel_cmd_delta_in_kinova_local(
                    target_eef_pose, cur_eef_pose, alpha
                )

                if self.env_name in ["push_chair"]:
                    error[1] = 0.0
                    error[3] = 0.0

                if self.env_name in ["close_luggage"]:
                    error[2] = 1.5 * error[2]

                derivative_error = (error - prev_error) / delta_t
                # NOTE: debug print
                # print(f"error={error}, with target_ee_pose_ori={target_eef_pose[-3:]} vs cur_ee_pose_ori={cur_eef_pose[-3:]}")

                if self.env_name == "teleop":
                    Kp = 1.5  # 0.9
                    Kd = 0.3  # 0.1
                    arm_vel_ctl_signal = Kp * error + Kd * derivative_error
                    prev_error = error
                elif PD_control:
                    Kp = 1.5  # 0.9
                    Kd = 0.3  # 0.1
                    arm_vel_ctl_signal = Kp * error + Kd * derivative_error
                    prev_error = error
                else:
                    Kp = 0.9
                    arm_vel_ctl_signal = Kp * derivative_error

                self._cmd_arm_vel_callback(arm_vel_ctl_signal, ac_interval=delta_t)

            if i in gripper_switch_ac_idxs:
                if self.env_name in ["teleop"]:
                    print("Won't stop arm or base before throwing the ball")
                    self.is_target_gripper_close[0] = gripper_status_traj[i, 0]
                    self.is_gripper_cmd_arrived[0] = 1
                else:
                    # if self.env_name in ["close_luggage"]
                    # Stop the arm and base for gripper switch
                    self.base_needs_stop[0] = 1
                    self.arm_needs_stop[0] = 1
                    time.sleep(0.1)  # Wait for the arm and base to respond
                    # print(f"self.is_target_gripper_close={self.is_target_gripper_close}")
                    # print(f"gripper_status_traj={gripper_status_traj}")
                    self.is_target_gripper_close[0] = gripper_status_traj[i, 0]
                    self.is_gripper_cmd_arrived[0] = 1
                    time.sleep(0.8)
                    self.base_needs_resume[0] = 1
                    time.sleep(0.1)  # Wait for the arm and base to respond

            # Update the prev_kpt_idx
            prev_kpt_idx = kpt_idx

        print("Current Trajectory tracking is done.")

    def _find_vel_cmd_delta_in_kinova_local(self, target_eef_pose, cur_eef_pose, alpha):
        target_xyz_in_kinova_local = self._global_to_kinova_local(
            target_eef_pose[:3], alpha
        )
        cur_xyz_in_kinova_local = self._global_to_kinova_local(cur_eef_pose[:3], alpha)

        error_xyz_in_kinova_local = target_xyz_in_kinova_local - cur_xyz_in_kinova_local

        if self.env_name in ["lift"]:
            error_xyz_in_kinova_local[0] = 0

        if self.env_name in ["teleop"]:
            error_xyz_in_kinova_local = np.zeros_like(error_xyz_in_kinova_local)

        target_euler_angle_in_global = target_eef_pose[-3:]
        current_euler_angle_in_global = cur_eef_pose[-3:]

        # Convert Euler angles to rotation matrices -- EXTRINSIC FORMAT
        R_cur_global = Rotation.from_euler(
            "xyz", current_euler_angle_in_global
        ).as_matrix()
        R_target_global = Rotation.from_euler(
            "xyz", target_euler_angle_in_global
        ).as_matrix()

        beta = np.pi - alpha
        R_global_in_local = Rotation.from_euler(
            "z", beta
        ).as_matrix()  # EXTRINSIC FORMAT

        R_cur_local = R_global_in_local @ R_cur_global
        R_target_local = R_global_in_local @ R_target_global

        # EXTRINSIC FORMAT
        delta_euler_angle_in_local = Rotation.from_matrix(
            R_target_local @ R_cur_local.T
        ).as_euler("xyz")

        delta_euler_angle_in_local = np.rad2deg(delta_euler_angle_in_local)

        error = np.concatenate(
            [error_xyz_in_kinova_local, delta_euler_angle_in_local], axis=-1
        )
        return error

    def _global_to_kinova_local(self, global_xyz, alpha):
        local_xyz = rotate_around_z(global_xyz, np.pi - alpha)
        return local_xyz

    def _cmd_arm_vel_callback(self, arm_vel_ctl_signal, ac_interval):
        target_arm_xyz_vel = arm_vel_ctl_signal[:3]
        target_arm_ori_vel = arm_vel_ctl_signal[3:]

        self.arm_cmd_send_lock.acquire()
        self.arm_cmd_vel_interval[0] = ac_interval
        self.target_arm_xyz_vel[:] = target_arm_xyz_vel
        self.target_arm_ori_deg_vel[:] = target_arm_ori_vel  # convert rad to deg
        self.is_vel_cmd_arrived[0] = 1
        self.arm_cmd_send_lock.release()

    def other_req(self, req):
        data = True
        rtype = req["type"]
        if rtype == "arm_home":
            self.arm_needs_home[0] = 1

        elif rtype == "arm_move_vel":
            target_arm_pos_vel = req["target_arm_xyz_vel"]
            target_arm_ori_deg_vel = req["target_arm_ori_deg_vel"]
            ac_interval = req["ac_interval"]

            self.arm_cmd_send_lock.acquire()
            self.target_arm_xyz_vel[:] = target_arm_pos_vel
            self.target_arm_ori_deg_vel[:] = target_arm_ori_deg_vel
            self.arm_cmd_vel_interval[0] = ac_interval
            self.is_vel_cmd_arrived[0] = 1
            self.arm_cmd_send_lock.release()
        else:
            data = self.low_priority_req(req)
        return data

    def low_priority_req(self, req):
        data = True
        rtype = req["type"]
        if rtype == "stop":
            self.arm_needs_stop[0] = 1
            self.base_needs_stop[0] = 1
        elif rtype == "arm_stop":
            self.arm_needs_stop[0] = 1
        elif rtype == "base_stop":
            self.base_needs_stop[0] = 1
        elif rtype == "base_resume":
            self.base_needs_resume[0] = 1
        else:
            data = self.update_info_req(req)
        return data

    def update_info_req(self, req):
        data = True
        rtype = req["type"]
        if rtype == "arm_update_fin_gripper_offset":
            self.arm_fin_gripper_offset[:] = req["fin_gripper_offsets"]
        elif rtype == "base_refresh_ref_angle":
            self.base_needs_refresh_ref_angle[0] = 1
        elif rtype == "base_set_rest_pose":
            self.base_rest_pose[:] = req["base_rest_pose"]
            self.base_needs_set_rest_pose[0] = 1
        else:
            raise ValueError(f"Request type {rtype} not recognized.")
        return data


@click.command()
@click.option("--arm_only", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--base_pose_override", type=str, default=None)
def main(arm_only, debug, base_pose_override):
    create_all_sm_topics()

    if base_pose_override is not None:
        # Must return the shm and data to keep the shared memory alive
        base_mocap_pose_shm, base_mocap_pose = init_SHM("base_mocap_pose")
        base_mocap_pose[:] = np.array(
            [float(s.strip()) for s in base_pose_override[1:-1].split(",")]
        )
    else:
        mocap_process = Process(target=MoCapMPAgent)
    kinova_control_process = Process(target=KinovaControlLoop, args=(debug,))
    kinova_timeout_process = Process(target=StopKinovaMovement)
    global_eef_update_process = Process(target=GlobalEEFUpdateLoop, args=(debug,))
    response_process = Process(target=RobotPyServerMP, args=(arm_only,))
    if not arm_only:
        base_control_process = Process(target=BaseControlLoop)

    if base_pose_override is None:
        mocap_process.start()
    kinova_control_process.start()
    kinova_timeout_process.start()
    global_eef_update_process.start()
    if not arm_only:
        base_control_process.start()
    response_process.start()

    # All processes are daemon, so no need to join


if __name__ == "__main__":
    main()
