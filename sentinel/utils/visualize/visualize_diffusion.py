import numpy as np
import os
import torch
import click
import cv2
from hydra import compose, initialize
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sentinel.utils.visualize.visualize_eval_pc import (
    parse_obs_node,
    parse_policy_node,
    parse_real_eval_mp,
    find_item_in_list,
)
from sentinel.bc.agents.sim3_dp_agent import SIM3DPAgent

import mob_manip_vision.pose_utils as pose_utils

# define cmap
colors = ["#3498DB", "#1ABC9C", "#FFA556"]  # '#34495E',
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)


def load_video(file_path):
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    frames = []

    while True:
        # Read a new frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Append the frame (numpy array) to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    return frames


def find_policy_inference_offsets(policy_outputs, time):
    item, index = find_item_in_list(policy_outputs, time, index=1, index_offset=0)
    assert item is not None
    state = np.array(item[2])
    offsets = np.array(item[3])[:, :, 1:4]
    offsets = np.cumsum(np.vstack([state[:, :3][None], offsets]), axis=0)
    return state, offsets, index


def load_calib(calib_dir):
    file_paths = list(glob(os.path.join(calib_dir, "ZED_*_calib_depth.npz")))
    cam_ids = [int(f.split("/")[-1].split("_")[1]) for f in file_paths]
    print(cam_ids)
    intrinsics = {}
    for cam_idx, cam_id in enumerate(cam_ids):
        d = np.load(os.path.join(calib_dir, f"ZED_{cam_id:d}_calib_depth.npz"))
        intrinsics[cam_id] = np.array(
            [[d["fx"], 0.0, d["cx"]], [0.0, d["fy"], d["cy"]], [0.0, 0.0, 1.0]]
        )
    transforms, _ = pose_utils.load_transforms(cam_ids, calib_dir, recompute=False)
    return intrinsics[cam_ids[0]], transforms[cam_ids[0]]


def project_points(points_3d, intrinsics, extrinsics):
    # Convert 3D points to homogeneous coordinates
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_hom = np.hstack((points_3d, ones))

    # Transform points using extrinsic matrix
    points_camera = points_3d_hom @ np.linalg.inv(extrinsics).T

    # Project points using intrinsic matrix
    points_2d_hom = points_camera[:, :3] @ intrinsics.T

    # Convert from homogeneous coordinates to Cartesian coordinates
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2, np.newaxis]

    return points_2d


def plot_trajectory_on_numpy_image(fig, ax, trajectory):
    # Normalize the trajectory indices for colormap
    normalized_indices = np.linspace(0, 1, len(trajectory))

    # Plot the points on the image
    for i, point in enumerate(trajectory):
        color = cmap(normalized_indices[i])
        ax.scatter(point[0], point[1], color=color, s=2)


@click.command()
@click.option(
    "--in_dir",
    type=str,
)
@click.option("--step", type=int, default=3)
@click.option(
    "--calib_dir",
    type=str,
)
@click.option("--k1", type=int, default=0)
@click.option("--k2", type=int, default=3)
@click.option("--h", type=int, default=12)
@click.option("--x", type=float, default=0.0)
@click.option("--y", type=float, default=0.0)
@click.option("--t_offset", type=int, default=0)
@click.option("--pct_offset", type=int, default=0)
def main(in_dir, step, calib_dir, k1, k2, h, x, y, t_offset, pct_offset):
    relative_config_path = os.path.relpath(os.path.join(in_dir, ".hydra"))
    with initialize(
        version_base=None, config_path=relative_config_path, job_name="vis"
    ):
        cfg = compose(config_name="config")

    spin_obs_state_path = os.path.join(in_dir, "obs_spinner.log")
    real_eval_mp_path = os.path.join(in_dir, "real_eval_mp_dp.log")
    policy_agent_path = os.path.join(in_dir, "policy_spinner.log")

    actions = parse_real_eval_mp(real_eval_mp_path)
    start_time = actions[0][0]
    end_time = actions[-1][0]
    time_to_str = lambda t: int((t.timestamp() - start_time.timestamp()) * 1000)
    t = None
    for action in actions:
        if action[1] == str(step):
            t = action[0].timestamp()
    if t is None:
        assert False

    times, positions, pcs_fg, pcs_bg, colors_fg, colors_bg = parse_obs_node(
        spin_obs_state_path
    )

    video_path = os.path.join(in_dir, "recording01.mp4")
    frames = np.array(load_video(video_path))
    img_shape = frames.shape[1:]

    image = frames[step * 5 + t_offset]
    image_k1 = image.copy()
    image_k2 = image.copy()

    _, obs_index = find_item_in_list([(tt.timestamp(),) for tt in times], t)
    pc_t = pcs_fg[obs_index + pct_offset]

    policy_outputs = parse_policy_node(policy_agent_path)
    inf_state, inf_offsets, inf_index = find_policy_inference_offsets(policy_outputs, t)

    intrinsics, extrinsics = load_calib(calib_dir)
    cam_id = cfg.robot_info.cam_ids[0]
    torch_intrinsics = torch.zeros(3, 3, dtype=torch.float64)
    fig1, ax1 = plt.subplots()
    ax1.set_aspect("equal")
    ax1.imshow(image_k1[..., ::-1] / 255)
    ax1.axis("off")
    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)

    fig2, ax2 = plt.subplots()
    ax2.set_aspect("equal")
    ax2.imshow(image_k2[..., ::-1] / 255)
    ax2.axis("off")
    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)

    agent = SIM3DPAgent(cfg)
    action = torch.tensor(inf_offsets)[:h, :, :]
    for i in range(10):
        noise = torch.randn(action.shape, device="cpu")
        vec_k1 = agent.actor.noise_scheduler.add_noise(action, noise, torch.tensor(k1))
        vec_k1 = vec_k1.detach().cpu().numpy()
        vec_k2 = agent.actor.noise_scheduler.add_noise(action, noise, torch.tensor(k2))
        vec_k2 = vec_k2.detach().cpu().numpy()
        # traj_k1 = np.cumsum(np.vstack([inf_state[:, :3][None], vec_k1]), axis=0)
        # traj_k2 = np.cumsum(np.vstack([inf_state[:, :3][None], vec_k2]), axis=0)
        traj_k1 = vec_k1 + np.array([x, y, 0.0])
        traj_k2 = vec_k2 + np.array([x, y, 0])

        # assume single agent
        traj_k1_2d = project_points(traj_k1[:, 0], intrinsics, extrinsics)
        traj_k2_2d = project_points(traj_k2[:, 0], intrinsics, extrinsics)
        plot_trajectory_on_numpy_image(fig1, ax1, traj_k1_2d)
        plot_trajectory_on_numpy_image(fig2, ax2, traj_k2_2d)

    fig1.savefig("plot_k1.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    fig2.savefig("plot_k2.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig1)
    plt.close(fig2)

    np.savez("pc.npz", pc=pc_t)


if __name__ == "__main__":
    main()
