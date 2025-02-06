from sentinel.envs.real_mobile.utils.common.info import ROBOT_INFO
from sentinel.envs.real_mobile.utils.perception.cameras import get_camera_by_name
from sentinel.envs.real_mobile.utils.control.kinova_with_base_py_client import (
    KinovaWithBasePyClient,
)


def init_rendering(camera_config):
    print(f"[env utils] initializing camera.")
    camera = get_camera_by_name(camera_config)
    print(f"[env utils] camera ready.")
    return camera, camera_config


def init_robot(robot_config, debug=False):
    robots = []
    for robot_config_item in robot_config:
        if debug:
            print("Processing robot config", robot_config_item)
        robot_info = ROBOT_INFO[robot_config_item["robot_name"]]

        robot = KinovaWithBasePyClient(
            name=robot_config_item["robot_name"],
            ip=robot_info["ip"],
            port=robot_info["port"],
            rest_base_pose=robot_config_item["preinit_base_pose"],
        )

        robot.arm_close_gripper()
        robot.arm_open_gripper()

        if debug:
            print("Init done for robot with info", robot_info)
        robots.append(robot)
    return robots
