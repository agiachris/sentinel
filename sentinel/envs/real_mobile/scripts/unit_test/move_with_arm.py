import time
import click
import numpy as np

from sentinel.envs.real_mobile.utils.common.constant import color
from sentinel.envs.real_mobile.utils.control.kinova_with_base_py_client import (
    KinovaWithBasePyClient,
)


@click.command()
@click.option("-b", "--bot", type=int, default=2, help="Bot number (IPRL Bot ?).")
# @click.option("-a", "--arm_pose", type=str, default="[0.57, 0.0, 0.44]")
@click.option("-a", "--arm_pose", type=str, default="[0.57, 0.0, 0.20]")
@click.option("-p", "--base_pose", type=str, default="[0, 0, 0.0]")
@click.option(
    "-o", "--onboard", is_flag=True, default=False, help="flag to enable onboard mode."
)
@click.option("-t", "--horizon", type=int, default=40)
@click.option("-f", "--freq", type=int, default=20.0)
def main(bot, arm_pose, base_pose, onboard, horizon, freq):
    arm_pose_start = np.array([float(s.strip()) for s in arm_pose[1:-1].split(",")])
    rest_base_pose = np.array([float(s.strip()) for s in base_pose[1:-1].split(",")])

    assert arm_pose_start.shape == (3,), f"Invalid arm_pose: {arm_pose_start}"
    assert rest_base_pose.shape == (3,), f"Invalid base_pose: {rest_base_pose}"

    if onboard:
        robot_client = KinovaWithBasePyClient(
            name="unit_Test", ip="127.0.0.1", port=6040, rest_base_pose=rest_base_pose
        )
    else:
        assert bot in [1, 2, 3], f"bot number {bot} is not supported"
        robot_client = KinovaWithBasePyClient(
            name="unit_Test",
            ip=f"iprl-bot{bot}",
            port=6040,
            rest_base_pose=rest_base_pose,
        )

    input(f"arm_pose_start={arm_pose_start}, Press Enter to continue...")
    ori = np.array([90, 0, 90])
    robot_client.arm_move_precise(arm_pose_start, ori)

    # robot_client.home() # home both arm and base
    robot_client.base_move_precise(rest_base_pose)

    input("[after separate cmd] Press Enter to continue...")

    whole_robot_pose = robot_client.get_whole_robot_pose()
    print(
        f"{color.BOLD}{color.CYAN}[[whole_robot_pose]] {color.END} OUTPUT: {whole_robot_pose}"
    )

    input("[Querry whole_robot_pose is Done] Press Enter to continue...")

    duration = 1.0 / freq

    # Delta_xyz to reference velocity
    # arm_ref_delta = np.array([0.0, 0.0, -0.2]) / (horizon // 2)
    # arm_ref_v = arm_ref_delta * freq / 1

    # base_ref_delta = np.array([0.3, 0.0, 0.0]) / (horizon // 2)
    base_pose = rest_base_pose

    latest_arm_xyz = arm_pose_start

    half_horizon = horizon // 2
    for i in range(horizon):
        start_time = time.time()
        if i < half_horizon:
            arm_xyz_v = np.array([1, 0.0, 0]) / (half_horizon * duration)
            arm_ori_v = np.array([np.rad2deg(-0.5), 0.0, np.rad2deg(0.5)])

            # base_pose = base_pose + base_ref_delta
            base_pose = rest_base_pose + np.array([-0.3, 0.0, 0.0])
            expected_base_xyz = (
                rest_base_pose + np.array([-0.3, 0.0, 0.0]) * (i + 1) / half_horizon
            )
        else:
            arm_xyz_v = -1 * np.array([1, 0.0, 0.0]) / (half_horizon * duration)
            arm_ori_v = -1 * np.array([np.rad2deg(-0.5), 0.0, np.rad2deg(0.5)])
            # base_pose = base_pose - base_ref_delta
            base_pose = rest_base_pose - np.array([-0.3, 0.0, 0.0])
            expected_base_xyz = (
                rest_base_pose
                - np.array([-0.3, 0.0, 0.0]) * (i - half_horizon + 1) / half_horizon
            )

        expected_global_arm_xyz = (
            arm_xyz_v * duration + latest_arm_xyz + expected_base_xyz
        )

        start_time = time.time()
        # base_pose = rest_base_pose
        whole_robot_pose = robot_client.move_base_and_arm(
            base_pose, arm_xyz_v, arm_ori_v, duration
        )

        print(f"Time: {time.time() - start_time}")
        print(
            f"{color.BOLD}{color.CYAN}[[move_base_and_arm]]{color.END} OUTPUT: {whole_robot_pose}"
        )

        latest_arm_xyz = expected_global_arm_xyz

        if i == half_horizon - 1:
            arm_xyz_v = np.array([0, 0, 0])
            arm_ori_v = np.array([0, 0, 0])
            whole_robot_pose = robot_client.move_base_and_arm(
                base_pose, arm_xyz_v, arm_ori_v, duration
            )
            input("[After half_horizon] Press Enter to continue...")

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 / freq - elapsed_time))
        print(
            f"{color.BOLD}{color.YELLOW}[Fake sleep 0.1 s to simulate the obs processing time] {i}{color.END}"
        )

    time.sleep(0.1)
    # END
    # arm_xyz_v = np.array([0, 0, 0])
    # base_pose = rest_base_pose
    # arm_ori_v = np.array([0, 0, 0])
    # whole_robot_pose = robot_client.move_base_and_arm(base_pose, arm_xyz_v, arm_ori_v)
    # print(f"{color.BOLD}{color.CYAN}[[move_base_and_arm]]{color.END} OUTPUT: {whole_robot_pose}")
    # time.sleep(0.1)

    arm_xyz_v = np.array([0, 0, 0])
    arm_ori_v = np.array([0, 0, 0])
    whole_robot_pose = robot_client.move_base_and_arm(
        base_pose, arm_xyz_v, arm_ori_v, duration
    )

    input("Press Enter to continue...")

    print("Done!")


if __name__ == "__main__":
    main()
