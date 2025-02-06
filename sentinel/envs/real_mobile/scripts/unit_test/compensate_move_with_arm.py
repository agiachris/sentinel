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
@click.option("-a", "--arm_pose", type=str, default="[0.97, 0.0, 0.20]")
@click.option("-p", "--base_pose", type=str, default="[0.0, 0.0, 0.0]")
@click.option(
    "-o", "--onboard", is_flag=True, default=False, help="flag to enable onboard mode."
)
@click.option("-t", "--horizon", type=int, default=30)
@click.option("-f", "--freq", type=int, default=10.0)
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

    robot_client.arm_home()
    time.sleep(1)

    robot_client.arm_move_precise(arm_pose_start, ori)

    # robot_client.home() # home both arm and base
    robot_client.base_move_precise(rest_base_pose)

    input("[after separate cmd] Press Enter to continue...")
    expected_base_xyz, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori = (
        robot_client.get_whole_robot_pose()
    )
    print(
        f"{color.BOLD}{color.CYAN}>>>>>>>>[[get_whole_robot_pose]] {color.END} OUTPUT: {expected_base_xyz}"
    )
    print(
        f"{color.BOLD}{color.CYAN}[local_ee_pos = {local_ee_pos}, [global_ee_pos = {global_ee_pos}] {color.END}"
    )

    input("[after separate cmd] Press Enter to continue...")

    whole_robot_pose = robot_client.get_whole_robot_pose()
    print(
        f"{color.BOLD}{color.CYAN}[[whole_robot_pose]] {color.END} OUTPUT: {whole_robot_pose}"
    )

    input("[Querry whole_robot_pose is Done] Press Enter to continue...")

    duration = 1.0 / freq

    i = 0
    # while True:
    for i in range(horizon * 2):
        start_time = time.time()
        if i < horizon - 1:
            arm_xyz_v = np.array([0.0, 0.0, 0])
            arm_ori_v = np.array([0.0, 0.0, 0.0])
            base_pose = rest_base_pose + np.array([-0.3, -0.3, 0.0])
            # expected_base_xyz = rest_base_pose + np.array([-0.3, 0.0, 0.0]) * (i+1) / horizon
            expected_base_xyz = rest_base_pose
            # expected_base_xyz, _, _, _, _ = robot_client.get_whole_robot_pose()
        else:
            base_pose = rest_base_pose + np.array([-0.3, -0.3, 0.0])
            print(f"{color.BOLD}{color.RED}[[SHOULD TOTALLY STOPED]]{color.END}")
        expected_base_xyz, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori = (
            robot_client.get_whole_robot_pose()
        )
        print(
            f"{color.BOLD}{color.CYAN}>>>>>>>>> [[get_whole_robot_pose]] {color.END} OUTPUT: {expected_base_xyz}"
        )
        print(
            f"{color.BOLD}{color.CYAN}[local_ee_pos = {local_ee_pos}, [global_ee_pos = {global_ee_pos}] {color.END}"
        )

        # NOTE: HARD CODED -- rotate 180 about z axis!!!!!
        expected_global_arm_xyz = -1 * arm_pose_start

        ok = robot_client.compensate_move_base_and_arm(
            base_pose,
            arm_xyz_v,
            arm_ori_v,
            duration,
            expected_base_xyz,
            expected_global_arm_xyz,
        )

        print(
            f"{color.BOLD}{color.CYAN}[compensate_move_base_and_arm] takes {time.time() - start_time} {color.END} OUTPUT: {ok}"
        )

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 1 / freq - elapsed_time))
        print(
            f"{color.BOLD}{color.YELLOW}[sleep to match {1/freq} s {freq} control freq] {i}{color.END}"
        )

    robot_client.stop()
    expected_base_xyz, local_ee_pos, local_ee_ori, global_ee_pos, global_ee_ori = (
        robot_client.get_whole_robot_pose()
    )
    print(
        f"{color.BOLD}{color.CYAN}>>>>>>>>> [[get_whole_robot_pose]] {color.END} OUTPUT: {expected_base_xyz}"
    )
    print(
        f"{color.BOLD}{color.CYAN}[local_ee_pos = {local_ee_pos}, [global_ee_pos = {global_ee_pos}] {color.END}"
    )

    input("Press Enter to continue...")

    print("Done!")


if __name__ == "__main__":
    main()
