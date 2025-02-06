import numpy as np
from omegaconf import OmegaConf
import click


from sentinel.envs.real_mobile.robot_env import RobotEnv


def make_config(bot_num, pose, heading, arm_pose, freq):
    addresses = [dict(ip=f"iprl-bot{bot_num}", port=6100)]

    target_arm_pose = arm_pose.tolist()
    target_arm_pose[-1] -= RobotEnv.ARM_MOUNTING_HEIGHT

    conf = OmegaConf.create(
        dict(
            seed=42,
            num_points=1024,
            max_episode_length=30,
            use_pos_actions=True,
            prompt_lists=dict(),
            robot_info=dict(
                task_name="fold",
                keyboard=False,
                xy_action_scale=1.0,
                z_action_scale=1.0,
                randomize_xy=0.0,
                randomize_rotation=0.0,
                use_dummy_pc=True,
                freq=freq,
                flip_agents=False,
                grasping_strategy="none",
                cam_ids=[21172477],
                open_gripper_eventually=False,
                move_arms_fn="oneshot",
                spin_sleep_t=0.001,
                camera=dict(camera_name="logitech_v2", addresses=addresses),
                info=[
                    dict(
                        robot_name=f"mobile_base{bot_num}",
                        rest_base_pose=pose.tolist(),
                        rest_arm_pos=target_arm_pose,
                        rest_arm_rot=[1.57, 0.0, 1.57],
                    )
                ],
            ),
        )
    )

    return conf


@click.command()
@click.option("-b", "--bot", type=int, default=1, help="Bot number (IPRL Bot ?).")
@click.option(
    "-p", "--pose", type=str, default="[-1.23, 0.61, 1.57]", help="Initial arm position"
)
def main(bot, pose):
    pose = np.array([float(s.strip()) for s in pose[1:-1].split(",")])
    heading = -np.array([np.cos(pose[-1]), np.sin(pose[-1])])
    config = make_config(bot, pose, heading, np.array([0.6, 0.0, 0.6]), 1)

    env = RobotEnv(config)
    env._last_base_poses = env._get_base_poses()
    env._last_local_ee_poss, env._last_local_ee_oris = env._get_local_ee_poses(
        env._last_base_poses
    )
    env._update_odometry_from_cameras()
    env._resume_all_robots()
    env._move_bases(pose[None], max_wait_time=10)
    print("Done!")


if __name__ == "__main__":
    main()
