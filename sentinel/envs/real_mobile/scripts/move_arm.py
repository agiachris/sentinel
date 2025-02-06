import time
import click
import numpy as np

from sentinel.envs.real_mobile.utils.common.constant import color
from sentinel.envs.real_mobile.utils.control.kinova_with_base_server import (
    KinovaWithBaseClient,
)


@click.command()
@click.option("-b", "--bot", type=int, default=1, help="Bot number (IPRL Bot ?).")
@click.option("-a", "--arm_pose", type=str, default="[0.50, 0.0, 0.42]")
@click.option("-w", "--wait", type=int, default=0)
@click.option("-o", "--onboard", type=int, default=0)
@click.option("-l", "--localizer", type=int, default=0)
@click.option("-t", "--horizon", type=int, default=30)
def main(bot, arm_pose, wait, horizon, onboard, localizer):
    arm_pose = np.array([float(s.strip()) for s in arm_pose[1:-1].split(",")])
    assert (
        arm_pose[1] == 0.0
    ), f"This code does not support horizontal arm offsets. for {arm_pose}"

    if onboard:
        robot_client = KinovaWithBaseClient(ip="127.0.0.1", port=6040)
    else:
        assert bot in [3], f"bot number {bot} is not supported"
        robot_client = KinovaWithBaseClient(ip=f"iprl-bot{bot}", port=6040)

    curr_xyz = arm_pose
    path = np.array([0.2, 0.0, -0.2])
    delta_xyz = np.array([path[0], path[1], path[2]]) / (horizon // 2)
    ori = np.array([90, 0, 90])

    robot_client.arm_move_directly(curr_xyz, ori, True, 5.0)

    wait = False
    reach_time = None if wait else 3.0

    for i in range(horizon):
        st = time.time()
        target_xyz = (
            (curr_xyz + delta_xyz) if i < horizon // 2 else (curr_xyz - delta_xyz)
        )
        print(f"{color.BOLD}{color.RED}[[main]] One-shot {color.END}")
        robot_client.arm_move_directly(target_xyz, ori, wait, reach_time)
        print(
            f"{color.BOLD}{color.CYAN}[[main]] move_arm_directly (with wait={wait}) takes {time.time()-st} seconds{color.END}"
        )
        curr_xyz = target_xyz
        st = time.time()
        # Simulate the PC's processing time
        time.sleep(0.5)
        print(
            f"{color.BOLD}{color.YELLOW}[[main]] Simulate the PC's processing time (with wait={wait}) takes {time.time()-st} seconds{color.END}"
        )

        if localizer:
            st = time.time()
            xy_heading = robot_client.base_get_pose_from_cam()
            print(
                f"{color.BOLD}{color.RED}[[main]] localizer takes {time.time()-st} seconds{color.END}"
            )

    print("Done!")


if __name__ == "__main__":
    main()
