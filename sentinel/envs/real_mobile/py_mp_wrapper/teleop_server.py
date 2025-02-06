# Python
import numpy as np
import socket

# Python
import click
import socket
import numpy as np
import time

from multiprocessing import Process
from multiprocessing.connection import Listener

# Onboard Arm
from sentinel.envs.real_mobile.utils.control.connection import DeviceConnection
from sentinel.envs.real_mobile.utils.control.kinova_new_for_teleop import KinovaArm
from sentinel.envs.real_mobile.utils.common.convert_coords import (
    local_to_global_pose,
)

# Network communication
from sentinel.envs.real_mobile.utils.common.constant import hardware

from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory as SM

from copy import deepcopy as dcp

# Onboard Base

# Safety + Exceptions
from sentinel.envs.real_mobile.utils.common.constant import hardware

# MultiThreadedExecutor

from sentinel.utils.mocap_client.mocap_agent import MocapAgent

from scipy.spatial.transform import Rotation as R


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
        "is_gripper_close",
        # Kinova Global EEF Pose
        "global_ee_pos",
        "global_ee_ori",
        # Base Mocap Pose
        "base_mocap_pose",
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
        print("init SHM locally to GlobalEEFUpdateLoop.")

        self._deviation_t = 0

        self.run()

    def _init_SM_to_local(self):
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM("base_mocap_pose")

        # Functionality
        self.local_ee_pos_shm, self.local_ee_pos = init_SHM("local_ee_pos")
        self.local_ee_ori_shm, self.local_ee_ori = init_SHM("local_ee_ori")

        self.is_gripper_close_shm, self.is_gripper_close = init_SHM("is_gripper_close")

        # Additional
        self.global_ee_pos_shm, self.global_ee_pos = init_SHM("global_ee_pos")
        self.global_ee_ori_shm, self.global_ee_ori = init_SHM("global_ee_ori")

    @property
    def _global_eef_pose_with_gripper_status_dict(self):
        return dict(
            base_pose=self.base_mocap_pose[:],
            local_ee_posi=self.local_ee_pos[:],
            local_ee_ori=self.local_ee_ori[:],
            global_ee_posi=self.global_ee_pos[:],
            global_ee_ori=self.global_ee_ori[:],
            gripper_status=self.is_gripper_close[:],  # 1: closed, 0: open
        )

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

    def _update_arm_info(self, enforce=False):
        if (self._deviation_t > 10) or enforce:
            self._compute_global_ee_pose()
            self._deviation_t = 0
        else:
            self._deviation_t += 1

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

        ##############################
        # Global Kinova EEF Pose
        self.global_ee_pos[:] = global_ee_pos
        self.global_ee_ori[:] = global_ee_ori
        ##############################


class TeleOpServer(object):
    def __init__(
        self,
        debug=False,
        kinova_ip="192.168.1.10",
        tcp_port=10000,
    ):
        self.debug = debug  # debug mode

        kinova_router = DeviceConnection.createTcpConnection(kinova_ip, tcp_port)
        self.arm = KinovaArm(kinova_router.__enter__(), router_real_time=None)
        print("init Kinova Arm.")

        self._init_SM_to_local()
        print(
            f"Inited pose_dict from SHM: {self._global_eef_pose_with_gripper_status_dict}"
        )
        print(
            f"{bcolors.OKGREEN}  In this class, fin_gripper_pose is not used! Directly control the Kinova endeffector pose. {bcolors.ENDC}"
        )

        try:
            self.run()
        finally:
            create_all_sm_topics(just_close=True)

    def reset(self):
        self.last_arm_vel_cmd_time = time.time()
        self.arm_cmd_vel_interval = 2
        self.enable_arm_vel_cmd_stop_callback = False
        self._deviation_t = 0

    def _init_SM_to_local(self):
        #### BASE--MOCAP [START] ####
        self.base_mocap_pose_shm, self.base_mocap_pose = init_SHM("base_mocap_pose")
        #### BASE--MOCAP [END] ####

        self.local_ee_pos_shm, self.local_ee_pos = init_SHM("local_ee_pos")
        self.local_ee_ori_shm, self.local_ee_ori = init_SHM("local_ee_ori")

        self.is_gripper_close_shm, self.is_gripper_close = init_SHM("is_gripper_close")

        # Additional
        self.global_ee_pos_shm, self.global_ee_pos = init_SHM("global_ee_pos")
        self.global_ee_ori_shm, self.global_ee_ori = init_SHM("global_ee_ori")

    @property
    def _global_eef_pose_with_gripper_status_dict(self):
        return dict(
            base_pose=self.base_mocap_pose[:],
            local_ee_posi=self.local_ee_pos[:],
            local_ee_ori=self.local_ee_ori[:],
            global_ee_posi=self.global_ee_pos[:],
            global_ee_ori=self.global_ee_ori[:],
            gripper_status=self.is_gripper_close[:],  # 1: closed, 0: open
        )

    def run(self):
        hostname = "0.0.0.0"
        port = 6040
        listener = Listener((hostname, port), authkey=b"123456")
        while True:
            print(f"[{hostname}:{port}] waiting for connection...")
            conn = listener.accept()
            print(f"[{hostname}:{port}] connected.")

            self.reset()
            while True:
                try:
                    if not conn.poll():
                        self._update_arm_info()
                    else:
                        req = conn.recv()
                        data = True
                        rtype = req["type"]
                        if (
                            ("update" in rtype)
                            or ("refresh" in rtype)
                            or ("set" in rtype)
                            or (" base" in rtype)
                        ):
                            color_print(
                                f"Currently skip those requests: {rtype} for the TeleopServer"
                            )
                        elif rtype == "get_global_eef_pose_with_gripper_status":
                            data = self._global_eef_pose_with_gripper_status_dict
                        elif rtype == "teleop_arm_vel":
                            gripper_cmd = req["gripper_cmd"]
                            if gripper_cmd == "open":
                                self._arm_open_gripper(enforce=False)
                            elif gripper_cmd == "close":
                                self._arm_close_gripper(enforce=False)

                            # rotate the gloabl command into local command
                            beta = np.pi - self.base_mocap_pose[-1]
                            R_global_in_local = R.from_euler(
                                "z", beta
                            ).as_matrix()  # EXTRINSIC FORMAT

                            vel_cmd_xyz_in_global = req["target_arm_xyz_vel"]
                            vel_cmd_xyz_in_local = (
                                R_global_in_local @ vel_cmd_xyz_in_global
                            )

                            # Rotation
                            vel_cmd_ori_rotvec_rad_in_local = req[
                                "target_arm_ori_rotvec_rad_vel"
                            ]

                            color_print(
                                f"DID NOT CONVERT ROTATION CMD FROM GLOBAL FRAME TO LOCAL FRAME YET",
                                color=bcolors.FAIL,
                            )

                            R_ori_delta_in_global = R.from_rotvec(
                                vel_cmd_ori_rotvec_rad_in_local
                            ).as_matrix()

                            R_ori_delta_in_local = R_ori_delta_in_global

                            vel_cmd_ori_deg_in_local = R.from_matrix(
                                R_ori_delta_in_local
                            ).as_euler(
                                "xyz", degrees=True
                            )  # euler in deg

                            arm_cmd_vel_interval = req["vel_cmd_duration"]

                            self.arm.command_velocity(
                                vel_cmd_xyz=vel_cmd_xyz_in_local,
                                vel_cmd_ori_deg=vel_cmd_ori_deg_in_local,
                                vel_cmd_duration=arm_cmd_vel_interval,
                                verbose=True if self.debug else False,
                            )
                            self.last_arm_vel_cmd_time = time.time()
                            self.arm_cmd_vel_interval = arm_cmd_vel_interval
                            self.enable_arm_vel_cmd_stop_callback = True
                            self._deviation_t += 10
                            # direct return and wait for the next command
                        elif rtype == "arm_move_precise":
                            target_arm_pos = req["target_arm_pos"]
                            target_arm_ori_deg = req["target_arm_ori_deg"]
                            joint_deg = self.arm.compute_ik(
                                target_arm_pos, target_arm_ori_deg
                            )  # pos in meter, ang in degree
                            self._arm_joint_ctrl(joint_deg)

                        elif rtype == "arm_move_precise_in_joint_space":
                            joint_deg = req["joint_deg"]
                            self._arm_joint_ctrl(joint_deg)

                        elif rtype == "arm_open_gripper":
                            self._arm_open_gripper(enforce=True)
                        elif rtype == "arm_close_gripper":
                            self._arm_close_gripper(enforce=True)
                        elif "stop" in rtype:
                            self.arm.stop()
                        elif "arm_home" in rtype:
                            self._arm_open_gripper(enforce=True)
                            self.arm.home()
                        else:
                            color_print(
                                f"Unknown request type {rtype} for the TeleopServer",
                                color=bcolors.FAIL,
                            )
                        conn.send(data)
                except (ConnectionResetError, EOFError):
                    self.arm.stop()
                    self.reset()
                    break
            print(f"[{hostname}:{port}] disconnected.")

    def _arm_joint_ctrl(self, joint_deg):
        self.enable_arm_vel_cmd_stop_callback = False
        wait = False
        run_ok = self.arm.move_angular(joint_deg.tolist(), wait)
        assert run_ok, f"Failed to execute arm_cmd_precise: with joint_deg={joint_deg}"
        self._update_arm_info(enforce=True)

    def _arm_open_gripper(self, enforce=False):
        if self.is_gripper_close[0] == 1 or enforce:
            self.arm.open_gripper()
            color_print("Open Gripper.", color=bcolors.BOLD)
            self.is_gripper_close[0] = 0

    def _arm_close_gripper(self, enforce=False):
        if self.is_gripper_close[0] == 0 or enforce:
            self.arm.close_gripper()
            color_print("Open Gripper.", color=bcolors.HEADER)
            self.is_gripper_close[0] = 1

    def _update_arm_info(self, enforce=False):
        if (self._deviation_t > 30) or enforce:
            self._update_arm_local_pose()
            self._deviation_t = 0
        else:
            self._deviation_t += 1

        if self.enable_arm_vel_cmd_stop_callback:
            elapsed_time = time.time() - self.last_arm_vel_cmd_time
            if elapsed_time > 1.1 * self.arm_cmd_vel_interval:
                color_print(f"elapsed_time. {elapsed_time}", color=bcolors.BOLD)
                self.arm.command_zero_velocity()
                self.enable_arm_vel_cmd_stop_callback = False

    def _update_arm_local_pose(self):
        local_ee_pos, local_ee_ori_deg = self.arm._get_robot_pose()
        self.local_ee_pos[:] = local_ee_pos
        self.local_ee_ori[:] = np.deg2rad(local_ee_ori_deg)


@click.command()
@click.option("--debug", is_flag=True, default=False)
@click.option("--base_pose_override", type=str, default=None)
def main(debug, base_pose_override):
    create_all_sm_topics()

    debug = True

    if base_pose_override is not None:
        # Must return the shm and data to keep the shared memory alive
        base_mocap_pose_shm, base_mocap_pose = init_SHM("base_mocap_pose")
        base_mocap_pose[:] = np.array(
            [float(s.strip()) for s in base_pose_override[1:-1].split(",")]
        )
    else:
        mocap_process = Process(target=MoCapMPAgent)
    teleop_process = Process(target=TeleOpServer, args=(debug,))
    global_eef_update_process = Process(target=GlobalEEFUpdateLoop, args=(debug,))
    if base_pose_override is None:
        mocap_process.start()
    teleop_process.start()
    global_eef_update_process.start()
    # All processes are daemon, so no need to join


if __name__ == "__main__":
    main()
