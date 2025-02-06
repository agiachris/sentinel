import numpy as np
from omegaconf import OmegaConf
import time
import click

from sentinel.envs.real_mobile.robot_env import color, RobotEnv


def make_config(bot_num, pose, heading, arm_pose, freq):
    addresses = [dict(ip=f"iprl-bot{bot_num}", port=6100)]

    target_arm_pose = arm_pose.tolist()
    target_arm_pose[-1] -= RobotEnv.ARM_MOUNTING_HEIGHT

    conf = OmegaConf.create(
        dict(
            seed=42,
            num_points=1024,
            max_episode_length=30,
            use_pos_actions=False,
            prompt_lists=dict(fold=["white cabinet"]),
            robot_info=dict(
                task_name="fold",
                keyboard=False,
                xy_action_scale=1.0,
                z_action_scale=1.0,
                randomize_xy=0.0,
                randomize_rotation=0.0,
                use_dummy_pc=True,
                # use_dummy_pc=False,
                freq=freq,
                flip_agents=False,
                grasping_strategy="none",
                cam_ids=[21172477],
                open_gripper_eventually=False,
                move_arms_fn="oneshot",
                spin_sleep_t=0.1,
                onboard=True,
                camera=dict(camera_name="logitech_v2", addresses=addresses),
            ),
        )
    )

    return conf


@click.command()
@click.option("-b", "--bot", type=int, default=1, help="Bot number (IPRL Bot ?).")
@click.option(
    "-p", "--pose", type=str, default="[-1.23, 0.61, 1.57]", help="Initial arm position"
)
@click.option("-a", "--arm_pose", type=str, default="[0.61, 0.0, 0.6]")
@click.option("-d", "--distance", type=float, default=0.61)
@click.option("-t", "--horizon", type=int, default=30)
@click.option("-f", "--freq", type=float, default=4.0)
def main(bot, pose, arm_pose, distance, horizon, freq):
    pose = np.array([float(s.strip()) for s in pose[1:-1].split(",")])
    arm_pose = np.array([float(s.strip()) for s in arm_pose[1:-1].split(",")])
    assert arm_pose[1] == 0, "This code does not support horizontal arm offsets."
    heading = -np.array([np.cos(pose[-1]), np.sin(pose[-1])])
    config = make_config(bot, pose, heading, arm_pose, freq)

    curr_pos = np.array(
        [
            pose[0] + heading[0] * arm_pose[0],
            pose[1] + heading[1] * arm_pose[0],
            arm_pose[2],
        ]
    )
    v = np.array([heading[0] * distance, heading[1] * distance, -0.2]) / (horizon // 2)

    env = RobotEnv(config)
    env.reset()
    for i in range(horizon):
        st = time.time()
        # target_pos = (curr_pos + v) if i < horizon // 2 else (curr_pos - v)
        target_pos = (curr_pos + v) if i < horizon // 2 else (curr_pos - v)

        print(
            f"{color.BOLD}{color.CYAN}[[main]] Step {i}{color.END} Target: {target_pos.round(2)}"
        )
        env.step(np.concatenate([[1.0], target_pos]))
        curr_pos = target_pos
        time_past = time.time() - st
        print(
            f"{color.BOLD}{color.YELLOW}[[main]] Time Past: {time_past:.3f}s{color.END}"
        )
    env._stop_all_robots()
    print("Done!")


if __name__ == "__main__":
    main()
