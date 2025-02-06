"""
Real Environment for Mobile Kinova.

Global Coords in SRC: x points toward restroom, y points toward the bed, z points up.

Constants:
- ARM_MOUNTING_HEIGHT: height of the arm mounting plate from the ground.
    - with base: 0.335
    - without base: 0.02

Authors: @Zi-ang-Cao

Date: 2024-05-11
"""

import numpy as np
import time
from threading import Thread
from multiprocessing.pool import ThreadPool

from sentinel.envs.real_mobile.utils.common.init import init_robot


DEBUG = 0
dbprint = print if DEBUG == 1 else lambda *args: ()

from sentinel.utils.traj_vis.traj_vis_utils import (
    convert_state_to_poses,
    extract_ac_traj,
)
import numpy as np


class RobotEnv(object):
    ARM_MOUNTING_HEIGHT = 0.335  # 0.335 with base; 0.02 without base
    ARM_REACH_APPROX = 1.5
    BASE_SAFE_MARGIN = 0.15
    BASE_VEL_THRESH = 0.05
    MIN_DIST_TO_FLOOR = 0.005
    MIN_EE_DIST = 0.20
    EE_XY_LOW = np.array([0.45, -0.15])
    EE_XY_HIGH = np.array([0.75, 0.15])
    BASE_TGT_LOWS = np.array([-1.5, -1.2])
    BASE_TGT_HIGHS = np.array([1.7, 1.2])

    def __init__(self, args):
        self.args = args
        self.num_eef = args.num_eef
        self.dof = args.dof
        self.rng = np.random.RandomState(seed=args.seed)
        assert self.dof in [7]  # only 7 DoF is supported for now

        self.max_episode_length = args.max_episode_length

        self.debug = True
        self._last_action_time = None

        # Control strategy
        self.use_pos_actions = args.use_pos_actions

        self._init_robots()

    def _init_robots(self):
        args = self.args

        self.freq = args.robot_info.freq
        self.robot_configs = args.robot_info.info
        self.flip_agents = args.robot_info.flip_agents
        self.obs_offset = (
            args.robot_info.obs_offset
            if hasattr(args.robot_info, "obs_offset")
            else np.array([0.0, 0.0, 0.0])
        )
        assert self.flip_agents == False, "Flip agents is not supported for now."

        # self.zero_out_ac_rot = args.zero_out_ac_rot

        self.randomize_xy = args.robot_info.randomize_xy
        self.randomize_rotation = args.robot_info.randomize_rotation

        self.grasping_strategy = args.robot_info.grasping_strategy
        self.task_name = args.robot_info.task_name
        self.floor_height = (
            args.robot_info.floor_height
            if hasattr(args.robot_info, "floor_height")
            else None
        )

        self.robots = init_robot(self.robot_configs, self.debug)
        self._gripper_poses = [0] * len(self.robots)

    def reset(self):
        self._t = 0
        self.max_ee_dist = None

        arm_preinit_posis = []
        arm_prinit_rot_degrees = []

        arm_preinit_posis_t2 = []
        arm_prinit_rot_degrees_t2 = []

        arm_preinit_joint_rads = []

        base_preinit_poses = []
        base_preinit_poses_t2 = []

        fin_gripper_offsets = []

        arm_limit_list = (
            []
        )  # list of np.zeros((3, 2))    # [x, y, z] limits <low, high>

        base_corner_xys_list = (
            []
        )  # list of np.zeros((4, 2))    # [x, y] corner coordinates

        # Compute preinit arm and base poses
        for config in self.robot_configs:
            # Fin gripper offset
            fin_gripper_offset = config["fin_gripper_offset"]
            fin_gripper_offsets.append(np.array(fin_gripper_offset))

            # Arm workspace limits
            arm_limit = config["arm_limit"]
            arm_limit_list.append(np.array(arm_limit).reshape(3, 2))

            # Base workspace limits
            base_corner_xys = config["base_corner_xys"]
            base_corner_xys_list.append(np.array(base_corner_xys).reshape(4, 2))

            # Arm pose
            preinit_arm_posi = config["preinit_arm_posi"]
            preinit_arm_rot_degree = config["preinit_arm_rot_degree"]

            arm_preinit_posis.append(preinit_arm_posi)
            arm_prinit_rot_degrees.append(preinit_arm_rot_degree)

            if "preinit_arm_posi_t2" in config:
                preinit_arm_posi_t2 = config["preinit_arm_posi_t2"]
                preinit_arm_rot_degree_t2 = config["preinit_arm_rot_degree_t2"]
                arm_preinit_posis_t2.append(preinit_arm_posi_t2)
                arm_prinit_rot_degrees_t2.append(preinit_arm_rot_degree_t2)

            if "preinit_arm_joint_rad" in config:
                arm_preinit_joint_rads.append(config["preinit_arm_joint_rad"])

            # Base pose
            preinit_base_pose = np.array(config["preinit_base_pose"])
            if not "preinit_arm_posi_t2" in config:
                preinit_base_pose[:2] += self.rng.randn(2) * self.randomize_xy
                preinit_base_pose[2] += (
                    self.rng.randn() * self.randomize_rotation / 180 * np.pi
                )
            base_preinit_poses.append(preinit_base_pose)

            if "preinit_base_pose_t2" in config:
                preinit_base_pose_t2 = np.array(config["preinit_base_pose_t2"])
                if not "preinit_arm_posi_t2" in config:
                    preinit_base_pose_t2[:2] += self.rng.randn(2) * self.randomize_xy
                    preinit_base_pose_t2[2] += (
                        self.rng.randn() * self.randomize_rotation / 180 * np.pi
                    )
                base_preinit_poses_t2.append(preinit_base_pose_t2)

        # Update workspace limits
        self._update_workspace(arm_limit_list, base_corner_xys_list)

        # Release/Open grippers and move to arm home pose
        ok = self._home_arms()

        # Move to preinit base poses
        self._resume_bases()
        ok &= self._preinit_bases(base_preinit_poses)
        input("Press [enter] when base is done")

        if len(base_preinit_poses_t2) > 0:
            ok &= self._preinit_bases(base_preinit_poses_t2)
            time.sleep(2)
            input("Press [enter] when base_preinit_poses_t2 is done")

        # Move to preinit arm poses
        self._update_fin_gripper_offsets(fin_gripper_offsets)

        # ok &= self._preinit_arm_in_joint_rads(arm_preinit_joint_rads)
        if len(arm_preinit_joint_rads) > 0:
            ok &= self._preinit_arm_in_joint_rads(arm_preinit_joint_rads)
        else:
            ok &= self._preinit_arms(arm_preinit_posis, arm_prinit_rot_degrees)

        if len(arm_preinit_posis_t2) > 0:
            input("Press [enter] to approach a close floor gripper pose")
            ok &= self._preinit_arms(arm_preinit_posis_t2, arm_prinit_rot_degrees_t2)

        # if self.task_name in ["load_shoes", "fold"]:
        if self.task_name in ["load_shoes", "fold", "lift", "unfold"]:
            input("Press [enter] to close gripper")
            for robot in self.robots:
                robot.arm_close_gripper()
            self._gripper_poses = [1] * len(self.robots)

        assert ok, "Failed to reset robots."

        self._t = 0
        return self._get_obs()

    def teleop_step(
        self,
        non_scaled_acs,
        vel_cmd_duration=0.1,
    ):
        if self.num_eef == 1:
            robot = self.robots[0]
            raw_ac = non_scaled_acs[0]
            self._teleop_one_robot_a_step(robot, raw_ac, vel_cmd_duration)
        else:
            assert (
                self.num_eef >= 2
            ), "When num_eef > 1, the robot should be able to handle multiple eef trajectories."
            teleop_thread_list = []
            for i, robot in enumerate(self.robots):
                raw_ac = non_scaled_acs[i]
                teleop_thread = Thread(
                    target=self._teleop_one_robot_a_step,
                    args=(robot, raw_ac, vel_cmd_duration),
                )
                teleop_thread_list.append(teleop_thread)

            for thread in teleop_thread_list:
                thread.start()

            for thread in teleop_thread_list:
                thread.join()

        self._t += 1

        done = self._t >= self.max_episode_length

        if done:
            for i, robot in enumerate(self.robots):
                robot.arm_stop()
                print(f"[teleop_step] >>>>>> robot.arm_stop()!!!!!!!!")

        # # sleep for step_interval
        # logging.info(f"[eval step] step {self._t} done")

        return self._get_obs(), 0.0, done, {}

    def _teleop_one_robot_a_step(self, robot, raw_ac, vel_cmd_duration):
        target_gripper_status = raw_ac[0]

        # eef_delta_pose = raw_ac[1:7] * (1.0 / vel_cmd_duration)
        eef_delta_pose = raw_ac[1:7]

        target_arm_xyz_vel = eef_delta_pose[:3]

        # delta axis-angle in rad to delta euler in deg
        target_arm_ori_rotvec_rad_vel = eef_delta_pose[3:]  # rotvector in rad

        robot.teleop_arm_vel(
            target_arm_xyz_vel=target_arm_xyz_vel,
            target_arm_ori_rotvec_rad_vel=target_arm_ori_rotvec_rad_vel,
            vel_cmd_duration=vel_cmd_duration,
            gripper_cmd="close" if target_gripper_status > 0.5 else "open",
        )

    def step(
        self,
        non_scaled_acs,
        ac_ref_state,
        pred_horizon=16,
        num_skip_steps=0,
        ac_horizon=8,
        step_interval=0.1,
        velocity_cmd=True,
        ac_time=None,
        arm_control_freq=125,  # for teleop
        #  arm_control_freq=50,
        fix_base=False,
    ):
        """
        velocity action:
            - non_scaled_acs = (pred_horizon, self.num_eef, self.dof)
            - ac_ref_state = (self.num_eef, 13)
        """

        assert (
            velocity_cmd
        ), "Only velocity command [delta xyz/velocity] is supported for now."

        """
        if self.zero_out_ac_rot and self.dof == 7:
            non_scaled_acs[:, :, -3:] = 0.0
        """

        # Velocity action + reference state => expected state
        """
        if self.task_name == "laundry_load":
            print(f"[step.py] >>>>>> num_skip_steps: {num_skip_steps}")
            ac_horizon = 4
        """
        if self.task_name == "fold":
            non_scaled_acs[:, :, 0] = 1.0

        # if self.task_name == "lift":
        #     non_scaled_acs[:, :, 4:] = 0.0 * non_scaled_acs[:, :, 4:]

        # Find expected states (pred_horizon, self.num_eef, 13)
        expected_states = extract_ac_traj(
            ac_ref_state,
            non_scaled_acs,
            num_skip_steps=num_skip_steps,
            ac_horizon=ac_horizon,
        )

        print(f"[step.py] >>>>>> num_skip_steps: {num_skip_steps}")
        # Extracted 8-steps in expected states -- consider num_skip_steps
        # expected_states = expected_states[num_skip_steps : num_skip_steps + ac_horizon]

        # NOTE: Clip the expected states to the max episode length
        expected_states = expected_states[: (self.max_episode_length - self._t)]

        ac_horizon = len(expected_states)

        print(f"[step.py] >>>>>> Clipped new ac_horizon: {ac_horizon}")

        # The first item in target_eef_traj and gripper_status_traj is the current pose
        # Which will be re-updated in the onboard control loop
        target_eef_traj = np.zeros((ac_horizon + 1, self.num_eef, 6))
        gripper_status_traj = np.zeros((ac_horizon + 1, self.num_eef))
        step_interval_traj = np.zeros((ac_horizon, self.num_eef))

        for i, state in enumerate(expected_states):
            expected_poses, expected_gripper_status = convert_state_to_poses(state)
            target_eef_traj[i + 1] = expected_poses
            gripper_status_traj[i + 1] = (
                (expected_gripper_status >= 0.5).astype(int).flatten()
            )
            if i == 0:
                local_step_interval = step_interval * (num_skip_steps + 1)
            else:
                local_step_interval = step_interval

            step_interval_traj[i] = local_step_interval

        if self.task_name == "packing":
            if self._t == 0:
                fix_base = True
                if np.any(gripper_status_traj > 0.5):
                    # find the first item and its index that gripper_status_traj > 0.5
                    first_item_idx = np.where(gripper_status_traj > 0.5)[0][0]
                    # Then set the gripper_status_traj to 1 for all the items after the first item
                    gripper_status_traj[first_item_idx + 1 :] = 1
            elif self._t < 16:
                fix_base = False
                gripper_status_traj = np.ones_like(gripper_status_traj)
        if self.task_name == "load_shoes":
            # fix_base = False
            fix_base = True
            if self._t >= 8:
                fix_base = False
        elif self.task_name == "close_luggage":
            fix_base = False
        elif self.task_name == "laundry_load":
            fix_base = False
            # if self._t <= 23:
            #     gripper_status_traj = np.ones_like(gripper_status_traj)
            if self._t == 0:
                try:
                    # find the first item and its index that gripper_status_traj > 0.5
                    first_item_idx = np.where(gripper_status_traj > 0.5)[0][0]
                    # Then set the gripper_status_traj to 1 for all the items after the first item
                    gripper_status_traj[first_item_idx + 1 :] = 1
                except:
                    print(
                        f"[step.py] >>>>>> gripper_status_traj: {gripper_status_traj}"
                    )

            elif self._t > 5 and self._t < 30:
                gripper_status_traj = np.ones_like(gripper_status_traj)
            else:
                fix_base = False
            # fix_base = False
        elif self.task_name == "lift":
            if self._t < 2:
                fix_base = True
            else:
                fix_base = False
        elif self.task_name == "unfold":
            fix_base = False
            gripper_status_traj = np.ones_like(gripper_status_traj)
        elif self.task_name == "make_bed":
            fix_base = False
            if self._t < 25:
                gripper_status_traj = np.ones_like(gripper_status_traj)
        elif self.task_name == "teleop":
            fix_base = True

        # take care of logging
        # def logging_fn(traj, start_t, freq, ac_horizon):
        #     for ac_idx in range(1, ac_horizon):
        #         logging_traj = deepcopy(np.array(traj[ac_idx:]))
        #         logging_traj[:, :, 1:4] += self.obs_offset
        #         # logging.info(f"[eval step] step {start_t + ac_idx} ac_time [{ac_time:.3f}] " + \
        #                     #  f"poses {logging_traj.round(3).tolist()}")
        #         time.sleep(1.0 / self.freq)
        #         # logging.info(f"[eval step] step {start_t + ac_idx} done")

        # logging_thread = Thread(target=logging_fn, args=(target_eef_traj, self._t, self.freq, ac_horizon))
        # logging_thread.start()

        print(
            f"[step.py] >>>>>> step_interval_traj: {step_interval_traj}; target_eef_traj: {target_eef_traj}"
        )
        st = time.time()
        ok = True
        if self.num_eef == 1:
            for i, robot in enumerate(self.robots):
                ok &= robot.track_eef_traj(
                    target_eef_traj[:, i],
                    gripper_status_traj[:, i],
                    step_interval_traj[:, i],
                    arm_control_freq,
                    fix_base=fix_base,
                    env_name=self.task_name,
                )
        else:
            assert (
                self.num_eef >= 2
            ), "When num_eef > 1, the robot should be able to handle multiple eef trajectories."
            track_eef_traj_thread_list = []
            for i, robot in enumerate(self.robots):
                track_eef_traj_thread = Thread(
                    target=robot.track_eef_traj,
                    args=(
                        target_eef_traj[:, i],
                        gripper_status_traj[:, i],
                        step_interval_traj[:, i],
                        arm_control_freq,
                        fix_base,
                        self.task_name,
                    ),
                )
                track_eef_traj_thread_list.append(track_eef_traj_thread)

            for thread in track_eef_traj_thread_list:
                thread.start()

            for thread in track_eef_traj_thread_list:
                thread.join()

        print(f"[step.py] >>>>>> robot.track_eef_traj take time: {time.time() - st}")

        self._t += ac_horizon
        # logging_thread.join(timeout=max(0, time.time() - (st + 1.0 / self.freq * ac_horizon)))

        self._gripper_poses = gripper_status_traj[-1]

        done = self._t >= self.max_episode_length

        if self.task_name == "lift" and self._t >= 20:
            if np.all(target_eef_traj[-1, :, 1] > -0.51):
                print(
                    f"[step.py] >>>>>> target_eef_traj[-1, :, 1]: {target_eef_traj[-1, :, 1]}"
                )
                if np.all(gripper_status_traj[-1] < 0.5):
                    print(
                        f"[step.py] >>>>>> gripper_status_traj[-1] < 0.5: {gripper_status_traj[-1] < 0.5}"
                    )
                    done = True

        if done:
            if self.num_eef == 1:
                for i, robot in enumerate(self.robots):
                    self._pre_ending(0, robot)
            else:
                assert (
                    self.num_eef >= 2
                ), "When num_eef > 1, the robot should be able to handle multiple eef trajectories."
                pre_ending_thread_list = []
                for i, robot in enumerate(self.robots):
                    pre_ending_thread = Thread(
                        target=self._pre_ending,
                        args=(
                            i,
                            robot,
                        ),
                    )
                    pre_ending_thread_list.append(pre_ending_thread)

                for thread in pre_ending_thread_list:
                    thread.start()

                for thread in pre_ending_thread_list:
                    thread.join()

        # sleep for step_interval
        # logging.info(f"[eval step] step {self._t} done")

        return self._get_obs(), 0.0, done, {}

    def _pre_ending(self, index, robot):
        robot.stop()
        time.sleep(0.1)

        robot.arm_open_gripper()
        time.sleep(1)

        if self.task_name == "laundry_load":
            end_base_pose = np.array([2.4, -2.86, 3.14])
            robot.base_move_precise(end_base_pose)
            time.sleep(1)

            """
            end_base_pose = np.array([2.4, -2.86, 3.14])
            robot.base_move_precise(end_base_pose)
            time.sleep(3)

            end_posi = np.array([0.77, 0.0, 0.30])
            # end_ori_deg = np.array([108.7, -87.6, 66.7])
            end_ori_deg = np.array([90, 0, 90])
            robot.arm_move_precise(end_posi, end_ori_deg)
            time.sleep(3)
            """
        elif self.task_name == "make_bed":
            """
            if index == 0:
                end_posi = np.array([0.5, 0., 0.6])
            else:
                end_posi = np.array([0.5, 0., 0.6])
            end_ori_deg = np.array([90, 0, 90])
            robot.arm_move_precise(end_posi, end_ori_deg)
            """
            robot.arm_open_gripper()
            base_pose = robot.get_global_eef_pose_with_gripper_status()["base_pose"]
            target_base_pose = base_pose.copy()
            if index == 0:
                target_base_pose[1] = 3.4
            else:
                target_base_pose[1] = 0.3
            robot.base_move_precise(target_base_pose)
            time.sleep(3)
        elif self.task_name == "fold":
            robot.arm_open_gripper()
            # latest_info = robot.get_global_eef_pose_with_gripper_status()
            time.sleep(3)
        elif self.task_name == "lift":
            pre_end_joint_rads = np.array(
                [[0.418, 0.595, 1.856, 5.856, 1.13, 5.57, 1.357]]
            )
            robot.arm_move_precise_in_joint_space(pre_end_joint_rads)
            time.sleep(2)

        # elif self.task_name == "lift":
        #     latest_info = robot.get_global_eef_pose_with_gripper_status()
        #     end_posi = latest_info["local_ee_posi"]
        #     end_ori_deg = np.rad2deg(latest_info["local_ee_ori"])
        #     end_ori_deg[0] = np.min([end_ori_deg[0]-30, 20])
        #     robot.arm_move_precise(end_posi, end_ori_deg)
        #     time.sleep(3)

    def _get_obs(self):
        # A fake observation
        return True

    def _home_arms(self):
        pool_results = []
        pool = ThreadPool()
        for i, robot in enumerate(self.robots):
            # Reset to Arm home
            pool_results.append(pool.apply_async(robot.arm_home))
            robot._arm_gripper_is_closed = False
        pool.close()  # Must call pool.close() before pool.join()
        pool.join()
        assert all(
            [res.ready() for res in pool_results]
        ), "Failed to move arm to home pose."
        pool.close()
        return True

    def _resume_bases(self):
        for i, robot in enumerate(self.robots):
            # Initialize base pose
            robot.base_resume()

            # refresh ref angle
            robot.base_refresh_ref_angle()

    def _update_workspace(self, arm_limit_list, base_corner_xys_list):
        for i, robot in enumerate(self.robots):
            # Update workspace limits
            robot.update_env_name(self.task_name)
            robot.arm_update_workspace(arm_limit_list[i])
            robot.base_update_workspace(base_corner_xys_list[i])

    def _update_fin_gripper_offsets(self, fin_gripper_offsets=[]):
        """
        fin_gripper_offsets : [np.array([x, y, z]), np.array([x, y, z]) ...]
        """
        for i, robot in enumerate(self.robots):
            # 0.04 for parallel gripper; 0.09 for fin gripper
            robot.arm_update_fin_gripper_offset(fin_gripper_offsets[i])

    def _preinit_bases(self, base_preinit_poses):
        """
        x, y, theta
        """
        for i, robot in enumerate(self.robots):
            robot.base_move_precise(base_preinit_poses[i])
        # Move to preinit base poses
        # pool_results = []
        # pool = ThreadPool()
        # for i, robot in enumerate(self.robots):
        #     # Move Base to preinit pose
        #     pool_results.append(pool.apply_async(robot.base_move_precise, (base_preinit_poses[i])))
        # pool.close()    # Must call pool.close() before pool.join()
        # pool.join()
        # assert all([res.ready() for res in pool_results]), "Failed to move base precisely to preinit pose."
        return True

    def _preinit_arm_in_joint_rads(self, preinit_arm_joint_rads):
        for i, robot in enumerate(self.robots):
            print(
                f"[_preinit_arm_in_joint_rads] >>>>>> preinit_arm_joint_rads: {preinit_arm_joint_rads[i]}"
            )
            robot.arm_move_precise_in_joint_space(preinit_arm_joint_rads[i])
            if self.task_name == "lift":
                time.sleep(2)
        return True
        # pool_results = []
        # pool = ThreadPool()
        # for i, robot in enumerate(self.robots):
        #     # Move Arm to preinit pose
        #     pool_results.append(
        #         pool.apply_async(
        #             robot.arm_move_precise_in_joint_space,
        #             (preinit_arm_joint_rads[i],),
        #         )
        #     )
        # pool.close()  # Must call pool.close() before pool.join()
        # pool.join()  # start the worker processes
        # assert all(
        #     [res.ready() for res in pool_results]
        # ), "Failed to move arm precisely to preinit pose."
        # return True

    def _preinit_arms(self, arm_preinit_posis, arm_prinit_rot_degrees):
        pool_results = []
        pool = ThreadPool()
        for i, robot in enumerate(self.robots):
            # Move Arm to preinit pose
            pool_results.append(
                pool.apply_async(
                    robot.arm_move_precise,
                    (arm_preinit_posis[i], arm_prinit_rot_degrees[i]),
                )
            )
        pool.close()  # Must call pool.close() before pool.join()
        pool.join()  # start the worker processes
        assert all(
            [res.ready() for res in pool_results]
        ), "Failed to move arm precisely to preinit pose."
        return True

    def compute_reward(self):
        return 0.0
