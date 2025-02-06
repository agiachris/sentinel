import click
import numpy as np
import time

# Safety + Exceptions
from sentinel.envs.real_mobile.utils.common.exceptions import SafetyCheckError
from sentinel.envs.real_mobile.utils.common.convert_coords import (
    local_to_global_pose,
)
from sentinel.envs.real_mobile.utils.perception.frame_converter import (
    CoordFrameConverter,
    wrap_angle,
)

# Network communication
from multiprocessing.connection import Listener
from multiprocessing.connection import Client
from sentinel.envs.real_mobile.utils.control.connection import DeviceConnection

# Multi-threading

# Onboard robot Base and Arm
from sentinel.envs.real_mobile.utils.control.mobile_base import MobileBase
from sentinel.envs.real_mobile.utils.control.kinova import KinovaArm

# Localizer
from sentinel.envs.real_mobile.utils.perception.logitech_v2.robot_pose_estimator_clean import (
    init_pose_estimator,
)


class KinovaWithBaseServer(object):
    ARM_MOUNTING_HEIGHT = 0.02  # 0.335
    ARM_REACH_APPROX = 1.5
    BASE_SAFE_MARGIN = 0.15
    BASE_VEL_THRESH = 0.05
    MIN_DIST_TO_FLOOR = 0.005
    MIN_EE_DIST = 0.20
    EE_XY_LOW = np.array([0.45, -0.15])
    EE_XY_HIGH = np.array([0.75, 0.15])
    BASE_TGT_LOWS = np.array([-1.5, -1.2])
    BASE_TGT_HIGHS = np.array([1.7, 1.2])

    BASE_DIFF_THRESH = np.array([0.015, 0.015, 0.02])
    BASE_MOVE_TIMEOUT = 10.0
    BASE_STEPS = 200

    def __init__(
        self,
        hostname="0.0.0.0",
        port=6040,
        kinova_ip="192.168.1.10",
        tcp_port=10000,
        base_name="bot3",
        rest_base_pose=np.array([0.0, 0.0, 0.0]),
        debug=True,
    ):
        assert kinova_ip is not None
        assert base_name is not None
        self.listener = Listener((hostname, port), authkey=b"123456")
        self.rest_base_pose = rest_base_pose
        self.debug = debug

        # Arm
        try:
            kinova_router = DeviceConnection.createTcpConnection(kinova_ip, tcp_port)
            self.arm = KinovaArm(kinova_router.__enter__(), router_real_time=None)
        except:
            self.arm = None
            print("Failed to connect to Kinova arm.")

        # Base + localizer
        try:
            self.poseEstimator = init_pose_estimator()
            self.base = MobileBase(name=base_name)
            self.coord_frame_converter = CoordFrameConverter()
            print("Init localizer")
        except:
            self.base = None
            print("Failed to init localizer")

        self.latest_base_get_xy_heading = None
        self._base_update_odometry_from_cameras()
        assert self.safety_check(), "Safety check failed."

        self.run()

    def _base_odom_pose(self):
        local_pose = self.base.get_pose()
        return self._base_local2global(local_pose)

    def _base_update_odometry_from_cameras(self):
        curr_odom_pose = self._base_odom_pose()
        curr_perc_pose = self._base_pose(refresh=True)
        # curr_perc_pose = self._base_pose(refresh=False)

        curr_perc_pose = np.array(curr_perc_pose)
        curr_odom_pose = np.array(curr_odom_pose)
        self.coord_frame_converter.update(curr_perc_pose, curr_odom_pose)
        return curr_perc_pose

    def _base_pose_cam_to_local_odom(self, target_cam_pose):
        # Step 1: Update odometry from cameras
        self._base_update_odometry_from_cameras()
        # Step 2: Convert target pose from camera frame to odom frame
        target_odom_global_pose = self.coord_frame_converter.convert_pose(
            target_cam_pose
        )
        target_odom_local_pose = self._base_global2local(target_odom_global_pose)

        # print(">>>" * 10)
        # print(f"cam_target_pose: {cam_target_pose}, odom_global_pose: {odom_global_pose}, odom_local_pose: {target_odom_local_pose}")
        # print("<<<" * 10)

        return np.array(target_odom_local_pose)

    def run(self):
        listener = self.listener
        while True:
            # Connect to clients
            address, port = listener.address
            print(f"[{address}:{port}] waiting for connection...")
            # Only one client can connect for now
            conn = listener.accept()
            print(f"[{address}:{port}] connected.")

            while True:
                try:
                    if not conn.poll():
                        # no request from client -- keep updating base pose
                        self.latest_base_get_xy_heading = self._base_pose(refresh=True)
                        continue
                    else:
                        # received request from client
                        req = conn.recv()
                        print(f"Received request {req}")
                        data = self.get_data(req)
                        conn.send(data)
                except (ConnectionResetError, EOFError):
                    break

            print(f"[{address}:{port}] disconnected.")

    def get_data(self, req):
        rtype = req["type"]
        print(f"[Request] {req}")

        arm_only = "arm" in rtype and not "base" in rtype
        base_only = "base" in rtype and not "arm" in rtype
        whole_robot = (not arm_only) and (not base_only)
        is_getter = "get" in rtype

        if is_getter:
            if arm_only:
                return self.get_arm_info(req)
            elif base_only:
                return self.get_base_info(req)
            else:
                raise ValueError(f"Request type {rtype} not recognized.")
        else:
            if arm_only:
                return self.execute_arm_command(req)
            elif base_only:
                return self.execute_base_command(req)
            elif whole_robot:
                return self.whole_robot_command(req)
            else:
                raise ValueError(f"Request type {rtype} not recognized.")

    def get_arm_info(self, req):
        rtype = req["type"]

        if rtype == "arm_get_robot_pose":
            data = self.arm._get_robot_pose()
        elif rtype == "arm_get_joint_pose":
            data = self.arm.get_joint_pose()
        elif rtype == "arm_get_local_ee_pose":
            data = self._local_ee_pose()
        elif rtype == "arm_get_global_ee_pose":
            base_pose = self._base_pose()
            data = self._global_ee_pose(base_pose)
        else:
            raise ValueError(f"Request type {rtype} not recognized.")
        return data

    def get_base_info(self, req):
        rtype = req["type"]

        if rtype == "base_get_pose_from_cam":
            data = self._base_pose()
        elif rtype == "base_get_pose_from_odom":
            data = self._base_odom_pose()
        elif rtype == "base_get_velocity":
            data = self.base.get_velocity()
        else:
            raise ValueError(f"Request type {rtype} not recognized.")
        return data

    def execute_arm_command(self, req):
        rtype = req["type"]
        if rtype == "arm_move_angular":
            joint_qpos = req["qpos"]
            wait = req["wait"]
            reach_time = req["reach_time"]
            data = self.arm.move_angular(joint_qpos, wait, reach_time)
        elif rtype == "arm_command_velocity":
            tgt_velocity = req["vel"]
            duration = req["duration"]
            wait = req["wait"]
            data = self.arm.command_velocity(tgt_velocity, duration, wait)
        elif rtype == "arm_home":
            data = self.arm.open_gripper()
            data &= self.arm.home()
        elif rtype == "arm_open_gripper":
            data = self.arm.open_gripper()
        elif rtype == "arm_close_gripper":
            data = self.arm.close_gripper()
        elif rtype == "arm_stop":
            data = self.arm.stop()
        elif rtype == "arm_compute_ik":
            data = self.arm.compute_ik(req["pos"], req["ang"])
        elif rtype == "arm_move_directly":
            joints_deg = self.arm.compute_ik(req["pos"], req["ang"])
            wait = req["wait"]
            reach_time = req["reach_time"]
            data = self.arm.move_angular(joints_deg.tolist(), wait, reach_time)
        else:
            raise ValueError(f"Request type {rtype} not recognized.")

        return data

    def execute_base_command(self, req):
        rtype = req["type"]
        assert self.base is not None, "Base is not initialized."

        if rtype == "base_set_rest_pose":
            self.rest_base_pose = req["rest_base_pose"]
            data = True
        elif rtype == "base_stop":
            data = self.base.stop()
        elif rtype == "base_resume":
            data = self.base.go()
        elif rtype == "base_set_ref_angle":
            self.base.set_ref_ang()
            data = True
        elif rtype == "base_move":
            global_pos = req["global_pos"]
            block = req.get("block", False)
            target_pose = global_pos
            odom_target_pose = self._base_pose_cam_to_local_odom(target_pose)
            run_ok = self.base.goto_pose(odom_target_pose, block=block)
            data = run_ok
        elif rtype == "base_move_directly":
            if self.base._stopped:
                self.base.go()  # resume

            # target_pose
            target_pose = req["target_pose"]
            max_wait_time = req.get("max_wait_time", self.BASE_MOVE_TIMEOUT)
            start_time = time.time()
            while True:
                # current_pose
                curr_base_pose = self._base_pose(refresh=False)
                if time.time() - start_time > max_wait_time:
                    run_ok = False
                    assert run_ok, "Base move timeout."
                    break
                # Move all bases.
                target_pose[2] = wrap_angle(target_pose[2], curr_base_pose[2])
                target_diff = target_pose - curr_base_pose
                pose_diff_ok = np.all(np.abs(target_diff) <= self.BASE_DIFF_THRESH)
                curr_vel = self.base.get_velocity()
                vel_is_small = np.all(np.abs(curr_vel) <= self.BASE_VEL_THRESH)
                if pose_diff_ok and vel_is_small:  # move done
                    run_ok = True
                    break
                else:
                    # move base
                    odom_target_pose = self._base_pose_cam_to_local_odom(target_pose)
                    run_ok = self.base.goto_pose(odom_target_pose, block=False)

                    # print(run_ok, target_pose, curr_base_pose, target_diff, pose_diff_ok, curr_vel, vel_is_small)
                    if not run_ok:
                        break
            data = run_ok
        else:
            raise ValueError(f"Request type {rtype} not recognized.")

        return data

    def whole_robot_command(self, req):
        rtype = req["type"]
        if rtype == "home":
            base_offset = req.get("base_offset", np.zeros(3))
            base_only = req.get("base_only", False)
            arm_only = req.get("arm_only", False)
            data = self._home(base_offset, base_only, arm_only)
        elif rtype == "freeze":
            data = self._freeze()
        elif rtype == "transmit_speed_test":
            data = req.get("transmit_data", np.zeros(7))
        elif rtype == "safety_check":
            data = self.safety_check()
        elif rtype == "move_base_and_arm":
            target_arm_vel = req["target_arm_vel"]
            target_base_pose = req["target_base_pose"]
            odom_target_base_pose = self._base_pose_cam_to_local_odom(target_base_pose)

            duration = req["duration"]
            wait = req["wait"]
            multiple_thread = req.get("multiple_thread", False)

            data = self.arm.command_velocity(target_arm_vel, duration, wait)
            data &= self.base.goto_pose(odom_target_base_pose, block=False)
        else:
            raise ValueError(f"Request type {rtype} not recognized.")
        return data

    # Private methods
    ## Base + Localizer
    def _base_local2global(self, local_pos):
        ix, iy, ia = self.rest_base_pose
        lx, ly, la = local_pos
        pos_x = ix + lx * np.cos(ia) - ly * np.sin(ia)
        pos_y = iy + lx * np.sin(ia) + ly * np.cos(ia)
        ang = ia + la
        return np.array([pos_x, pos_y, ang])

    def _base_global2local(self, global_pos):
        ix, iy, ia = self.rest_base_pose
        gx, gy, ga = global_pos
        pos_x = (gx - ix) * np.cos(ia) + (gy - iy) * np.sin(ia)
        pos_y = -(gx - ix) * np.sin(ia) + (gy - iy) * np.cos(ia)
        ang = ga - ia
        return np.array([pos_x, pos_y, ang])

    def _home(self, base_offset=np.zeros(3), base_only=False, arm_only=False):
        assert not (base_only and arm_only)
        run_ok = True
        if not base_only:
            run_ok &= self.arm.open_gripper()
            run_ok &= self.arm.home()
            if not run_ok:
                return False
        if not arm_only:
            target_pose = base_offset
            odom_target_pose = self._base_pose_cam_to_local_odom(target_pose)
            run_ok &= self.base.goto_pose(odom_target_pose)
        return run_ok

    ## Both Arm and Base
    def _freeze(self):
        print(f"[kinova with base] freezing robot...", end=" ")
        self.arm.stop()
        self.base.stop()
        print("done.")

    def _base_pose(self, refresh=False):
        if refresh:
            self.latest_base_get_xy_heading = self.poseEstimator.get_xy_heading()
        return self.latest_base_get_xy_heading

    def _global_ee_pose(self, base_pose):
        base_xy = base_pose[0:2]
        local_ee_posi, local_ee_ori = self._local_ee_pose()

        global_ee_posi, global_ee_ori, _ = local_to_global_pose(
            local_ee_posi,
            local_ee_ori,
            base_xy=base_xy,
            base_rot=base_pose[2],
            height_offset=self.ARM_MOUNTING_HEIGHT,
        )

        return np.array(global_ee_posi), np.array(global_ee_ori)

    def _local_ee_pose(self):
        local_ee_pos, local_ee_ori_deg = self.arm._get_robot_pose()
        local_ee_ori = local_ee_ori_deg / 180 * np.pi
        return np.array(local_ee_pos), np.array(local_ee_ori)

    def safety_check(self):
        try:
            st = time.time()
            # Get latest base poses
            base_pose = self._base_pose(refresh=True)
            self._assert_safety(base_pose)
            if self.debug:
                print(f"assert safety took {time.time() - st:.3f}s")
            return True
        except SafetyCheckError:
            print("Assert safety error!")
            self._freeze()
            return False

    def _assert_safety(self, base_pose):
        xy_lows = self.BASE_TGT_LOWS - self.BASE_SAFE_MARGIN
        xy_highs = self.BASE_TGT_HIGHS + self.BASE_SAFE_MARGIN
        all_ok = True

        base_xy = base_pose[0:2]
        if (base_xy < xy_lows).any() or (base_xy > xy_highs).any():
            all_ok = False
        assert all_ok, "Base pose out of range."

        global_ee_posi, global_ee_ori = self._global_ee_pose(base_pose)

        if global_ee_posi[2] < self.MIN_DIST_TO_FLOOR:
            all_ok = False
        assert all_ok, "End effector is too close to the floor."


class KinovaWithBaseClient(object):
    def __init__(self, ip, port, rest_base_pose=np.array([0.0, 0.0, 0.0])):
        self.conn = Client((ip, port), authkey=b"123456")

    def send_request(self, rtype, args=dict()):
        req = {"type": rtype}
        req.update(args)
        self.conn.send(req)
        return self.conn.recv()

    # Unit tests
    def transmit_speed_test(self, transmit_data=np.zeros(7)):
        req = dict(transmit_data=transmit_data)
        return self.send_request("transmit_speed_test", req)

    def safety_check(self):
        return self.send_request("safety_check")

    # Base + Localizer

    ## Query Base Info
    def base_get_pose_from_cam(self):
        # x, y, heading (in radian)
        return self.send_request("base_get_pose_from_cam")

    def base_get_pose_from_odom(self):
        return self.send_request("base_get_pose_from_odom")

    def base_get_velocity(self):
        return self.send_request("base_get_velocity")

    ## Execute Base Commands
    ## Query Base Info

    def base_stop(self):
        return self.send_request("base_stop")

    def base_resume(self):
        return self.send_request("base_resume")

    ## Execute Base Commands
    def move_base_and_arm(
        self,
        target_arm_vel,
        target_base_pose,
        duration,
        wait=False,
        multiple_thread=False,
        max_wait_time=0.01,
    ):
        req = dict(
            target_arm_vel=target_arm_vel,
            target_base_pose=target_base_pose,
            duration=duration,
            wait=wait,
            multiple_thread=multiple_thread,
            max_wait_time=max_wait_time,
        )
        return self.send_request("move_base_and_arm", req)

    def base_move(self, global_pos, block=True):
        req = dict(global_pos=global_pos, block=block)
        return self.send_request("base_move", req)

    def base_move_directly(self, target_pose, max_wait_time=10.0):
        req = dict(target_pose=target_pose, max_wait_time=max_wait_time)
        return self.send_request("base_move_directly", req)

    def base_set_ref_angle(self):
        return self.send_request("base_set_ref_angle")

    # Kinova Arm
    ## Querry Arm Info
    def arm_get_robot_pose(self):
        return self.send_request("arm_get_robot_pose")

    def arm_get_joint_pose(self):
        return self.send_request("arm_get_joint_pose")

    def arm_get_local_ee_pose(self):
        return self.send_request("arm_get_local_ee_pose")

    def arm_get_global_ee_pose(self):
        return self.send_request("arm_get_global_ee_pose")

    ## Execute Arm Commands
    def arm_home(self):
        return self.send_request("arm_home")

    def arm_open_gripper(self):
        return self.send_request("arm_open_gripper")

    def arm_close_gripper(self):
        return self.send_request("arm_close_gripper")

    def arm_stop(self):
        return self.send_request("arm_stop")

    def arm_command_velocity(self, tgt_velocity, duration, wait=True):
        req = dict(vel=tgt_velocity, duration=duration, wait=wait)
        return self.send_request("arm_command_velocity", req)

    def arm_move_angular(self, joint_positions, wait=True, reach_time=None):
        print(f'[move arm angular] {joint_positions} ({"wait" if wait else "async"})')
        req = dict(qpos=joint_positions, wait=wait, reach_time=reach_time)
        return self.send_request("arm_move_angular", req)

    def arm_compute_ik(self, pos, ang):
        req = dict(pos=pos, ang=ang)
        return self.send_request("arm_compute_ik", req)

    def arm_move_directly(self, pos, ang, wait=True, reach_time=None):
        req = dict(pos=pos, ang=ang, wait=wait, reach_time=reach_time)
        return self.send_request("arm_move_directly", req)


@click.command()
@click.option("--bot_num", type=int, default=3)
def main(bot_num):
    router, router_real_time = None, None
    try:
        server = KinovaWithBaseServer()
    finally:
        if router is not None:
            router.__exit__()
        if router_real_time is not None:
            router_real_time.__exit__()


if __name__ == "__main__":
    main()
