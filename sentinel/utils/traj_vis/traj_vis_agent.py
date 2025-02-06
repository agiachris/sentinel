import os
import numpy as np
from collections import deque
from sentinel.utils.process_real_pc import animate_pcs
import click
import click
import ast


class TrajectoryVisualizer:
    def __init__(
        self,
        task_name,
        base_data_dir,
        base_log_dir,
        episode_template="ep{:03d}_t{:04d}.npz",
    ):
        self.task_name = task_name
        self.data_dir = os.path.join(
            base_data_dir, self.task_name, "segmented/training_data"
        )
        self.log_dir = os.path.join(base_log_dir, self.task_name)
        self.fn_template = episode_template
        self.demo_ac_horizon = 16  # default horizon for demo trajectory
        os.makedirs(self.log_dir, exist_ok=True)

    def extract_demo_traj(self, ep_id, start_time_step, demo_ac_horizon=None):
        if demo_ac_horizon is None:
            demo_ac_horizon = self.demo_ac_horizon

        expected_states_dq = deque(maxlen=demo_ac_horizon)
        for ac_ix in range(start_time_step + 1, start_time_step + 1 + demo_ac_horizon):
            obs_t = np.load(
                os.path.join(self.data_dir, self.fn_template.format(ep_id, ac_ix))
            )
            expected_states_dq.append(obs_t["eef_pos"])
        return np.array(expected_states_dq)

    def visualize_demo_trajectory(self, ep_id, start_time_step, demo_ac_horizon=None):
        obs_t2 = np.load(
            os.path.join(self.data_dir, self.fn_template.format(ep_id, start_time_step))
        )
        pc_components = [obs_t2["pc"]]
        color_components = [np.zeros_like(obs_t2["pc"])]

        demo_traj_states = self.extract_demo_traj(
            ep_id, start_time_step, demo_ac_horizon
        )
        for states in demo_traj_states:
            eef_poses = states
            gripper_actions = states[:, -1]
            for hand_idx, (eef_pose, gripper_action) in enumerate(
                zip(eef_poses, gripper_actions)
            ):
                eef_vis_pts = np.ones([31, 3]) * eef_pose[:3][None]
                eef_vis_colors = np.zeros([31, 3])
                axis_dir = np.zeros([3, 3])
                axis_dir[0] = eef_pose[3:6]
                axis_dir[1] = np.cross(eef_pose[6:9], eef_pose[3:6])
                axis_dir[2] = eef_pose[6:9]
                for i in range(3):
                    eef_vis_pts[1 + i * 10 : 11 + i * 10] += (
                        axis_dir[[i]] * (np.arange(10)[:, None] + 1) / 200
                    )
                eef_vis_colors[1:11, 0] = 255
                eef_vis_colors[11:21, 1] = 255
                eef_vis_colors[21:31, 2] = 255
                if hand_idx == 1:
                    eef_vis_colors[1:11, 1] = 255
                    eef_vis_colors[11:21, 2] = 255
                    eef_vis_colors[21:31, 0] = 255

                if gripper_action < 0.5:
                    eef_vis_colors /= 2
                pc_components.append(eef_vis_pts)
                color_components.append(eef_vis_colors)

        vis_pcs = np.concatenate(pc_components)
        vis_colors = np.concatenate(color_components)
        full_html_path = os.path.join(
            self.log_dir,
            f"debug_{self.task_name}_ep_{ep_id}_ac_{demo_ac_horizon}_from_t_{start_time_step}.html",
        )
        animate_pcs(
            vis_pcs[None, ...],
            vis_colors[None, ...],
            save_path=full_html_path,
        )
        print(f"Saved visualization to {full_html_path}")


def parse_list(ctx, param, value):
    try:
        return ast.literal_eval(value)
    except:
        raise click.BadParameter("List must be a valid Python list.")


@click.command()
@click.option(
    "--task_name", default="0424_cuppnp_d100_10hz", help="Name of the task to process."
)
@click.option(
    "--base_data_dir",
    help="Base directory for data.",
)
@click.option(
    "--base_log_dir",
    help="Base directory for logs.",
)
@click.option(
    "--episode_template",
    default="ep{:03d}_t{:04d}.npz",
    help="Template for episode file names.",
)
@click.option(
    "--step_t2", default=5, type=int, help="Time step t2 to start visualizing from."
)
@click.option(
    "--demo_ac_horizon", default=50, type=int, help="Action Horizon to visualize"
)
@click.option(
    "--ep_id_list",
    default="[1]",
    required=True,
    callback=parse_list,
    help="List of promotional items.",
)
def main(
    task_name,
    base_data_dir,
    base_log_dir,
    episode_template,
    step_t2,
    demo_ac_horizon,
    ep_id_list,
):
    visualizer = TrajectoryVisualizer(
        task_name, base_data_dir, base_log_dir, episode_template
    )
    for ep_id in ep_id_list:
        visualizer.visualize_demo_trajectory(ep_id, step_t2, demo_ac_horizon)
        print(
            f"Visualization complete for task {task_name} starting from step {step_t2}."
        )


if __name__ == "__main__":
    main()
