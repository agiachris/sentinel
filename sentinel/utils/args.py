"""
Command line arguments for camera and deform utils.

@contactrika, @yjy0625

"""

import argparse
import logging
import sys
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(parent=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="PTL", add_help=False)

    # Main/demo args.
    parser.add_argument(
        "--task_name", type=str, default="folding-v1", help="Name of the task"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--seed_env", type=int, default=666, help="Random seed for environment"
    )
    parser.add_argument(
        "--seed_act", type=int, default=6666, help="Random seed for action"
    )
    parser.add_argument(
        "--seed_cam", type=int, default=66666, help="Random seed for camera"
    )
    parser.add_argument("--vis", action="store_true", help="Whether to visualize")
    parser.add_argument(
        "--debug", action="store_true", help="Whether to print debug info"
    )
    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument(
        "--sim_frequency",
        type=int,
        default=500,
        help="Number of simulation steps per second",
    )  # 250-1K
    parser.add_argument("--sim_gravity", type=float, default=-9.8, help="Gravity")
    parser.add_argument(
        "--max_episode_length", type=int, default=100, help="Max episode length"
    )
    parser.add_argument("--dof", type=int, default=4, help="Action dim")
    parser.add_argument("--num_eef", type=int, default=2, help="Action dim")
    parser.add_argument(
        "--use_pos_actions", type=int, default=0, help="Whether to use position actions"
    )
    parser.add_argument("--ac_repeat", type=float, default=10, help="Action repeat")
    parser.add_argument("--ac_noise", type=float, default=0.0, help="Action noise")
    parser.add_argument("--ac_scale", type=float, default=1.0, help="Action scale")
    parser.add_argument(
        "--floor_texture_fname",
        type=str,
        default=None,
        help="File name for the floor texture from data/planes",
    )
    parser.add_argument("--gripper_name", type=str, default="suction")
    # Rigid obj args.
    parser.add_argument(
        "--rigid_custom_obj",
        type=str,
        default=None,
        help="Obj file for a custom rigid object in the scene",
    )
    parser.add_argument(
        "--rigid_exact_scale",
        type=float,
        nargs=3,
        default=None,
        help="Scaling for the custom rigid object",
    )
    # Anchor/grasping args.
    parser.add_argument(
        "--direct_control",
        action="store_true",
        help="Override anchor pos instead of " "interpreting actions as forces",
    )
    # deform/SoftBody obj args.
    parser.add_argument("--randomize_rotation", action="store_true")
    parser.add_argument("--exact_rotation", type=float, default=None)
    parser.add_argument("--randomize_articulation", action="store_true")
    parser.add_argument(
        "--deform_obj",
        type=str,
        default="cloth/cloth.obj",
        help="Obj file for deform item",
    )
    parser.add_argument(
        "--deform_init_pos",
        type=float,
        nargs=3,
        default=[0, 0, 0.65],
        help="Initial pos for the center of the deform object",
    )
    parser.add_argument(
        "--deform_init_ori",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Initial orientation for deform (in Euler angles)",
    )
    parser.add_argument(
        "--deform_scale", type=float, default=1.0, help="Scaling for the deform object"
    )
    parser.add_argument("--deform_randomize_scale", action="store_true")
    parser.add_argument(
        "--deform_noise",
        type=float,
        default=0.0,
        help="Add noise to deform point cloud (0.01 ok)",
    )
    parser.add_argument(
        "--deform_bending_stiffness",
        type=float,
        default=0.01,
        help="deform spring elastic stiffness (k)",
    )  # 1.0-300.0
    parser.add_argument(
        "--deform_damping_stiffness",
        type=float,
        default=1.0,
        help="deform spring damping stiffness (c)",
    )
    parser.add_argument(
        "--deform_elastic_stiffness",
        type=float,
        default=300.0,
        help="deform spring elastic stiffness (k)",
    )  # 1.0-300.0
    parser.add_argument(
        "--deform_friction_coeff",
        type=float,
        default=10.0,
        help="deform friction coefficient",
    )
    parser.add_argument("--deform_exact_scale", type=float, nargs=2, default=None)
    parser.add_argument("--randomize_robot_eef_pos", action="store_true")
    parser.add_argument("--robot_eef_pos_x_low", type=float, default=-0.025)
    parser.add_argument("--robot_eef_pos_x_high", type=float, default=0.025)
    parser.add_argument("--robot_eef_pos_y_low", type=float, default=-0.025)
    parser.add_argument("--robot_eef_pos_y_high", type=float, default=0.025)

    parser.add_argument("--randomize_rigid_pos", action="store_true")
    parser.add_argument("--rigid_pos_x_low", type=float, default=-0.1)
    parser.add_argument("--rigid_pos_x_high", type=float, default=0.1)
    parser.add_argument("--rigid_pos_y_low", type=float, default=0.0)
    parser.add_argument("--rigid_pos_y_high", type=float, default=0.1)

    parser.add_argument("--randomize_dynamics_noise", action="store_true")
    parser.add_argument(
        "--dynamics_noise_scheduler", type=str, default="GaussianNoiseScheduler"
    )
    parser.add_argument(
        "--dynamics_noise_scheduler_kwargs",
        type=json.loads,
        default='{"mean": 0, "std_dev": 0.05}',
    )

    # Camera args.
    parser.add_argument(
        "--cam_resolution", type=int, default=240, help="Point cloud resolution"
    )
    parser.add_argument(
        "--cam_rec_interval",
        type=int,
        default=1,
        help="How many steps to skip between each cam shot",
    )
    parser.add_argument("--cam_randomize", action="store_true")
    parser.add_argument(
        "--cam_num_views", type=int, default=1, help="Number of views to sample."
    )
    # Data generation.
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--data_num_objects", type=int, default=1)
    parser.add_argument("--data_repeat_object", type=int, default=1)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--data_rew_threshold", type=float, default=0.5)
    parser.add_argument("--deform_asset_dir", type=str, default=None)
    parser.add_argument("--policy", type=str, default="demo")
    parser.add_argument("--cam_pitches", type=int, nargs="*", default=[-90])
    parser.add_argument("--cam_yaws", type=int, nargs="*", default=[0])
    parser.add_argument("--cam_dist", type=float, default=1.0)
    parser.add_argument("--cam_fov", type=float, default=30)

    parser.add_argument("--scale_low", type=float, default=1.0)
    parser.add_argument("--scale_high", type=float, default=2.0)
    parser.add_argument("--scale_aspect_limit", type=float, default=100.0)
    parser.add_argument("--uniform_scaling", action="store_true")
    parser.add_argument("--scale_mode", type=str, default="sim")
    parser.add_argument("--speed_multiplier", type=float, default=1.0)
    parser.add_argument("--compute_vel_actions_from_pos", action="store_true")

    parser.add_argument("--mm_ac_scale", type=float, default=1)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=2.0)

    # Logging.
    parser.add_argument("--wandb_logging", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="jingyuny")

    args, unknown = parser.parse_known_args()
    return args, unknown
