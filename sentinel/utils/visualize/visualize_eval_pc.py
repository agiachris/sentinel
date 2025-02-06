import os
import json
import click
import re
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import bisect


def parse_datetime(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")


def parse_obs_node(log_path):
    pattern_with_color = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):INFO:\[done\] state: (.*) pc: (.*) pc_bg: (.*) colors: (.*) bg_colors: (.*)"
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):INFO:\[done\] state: (.*) pc: (.*) pc_bg: (.*)"
    times = []
    positions = []
    pcs = []
    pcs_bg = []
    colors = []
    bg_colors = []
    with open(log_path, "r") as file:
        for line in file:
            match_with_color = re.search(pattern_with_color, line)
            match = re.search(pattern, line)
            if match_with_color or match:
                if match_with_color:
                    timestamp, state_str, pc_str, pc_bg_str, color_str, bg_color_str = (
                        match_with_color.groups()
                    )
                else:
                    timestamp, state_str, pc_str, pc_bg_str = match.groups()
                    color_str, bg_color_str = None, None
                timestamp = parse_datetime(timestamp)
                state = json.loads(state_str)
                positions.append(state[:][:3])
                pcs.append(np.array(json.loads(pc_str)))
                pcs_bg.append(np.array(json.loads(pc_bg_str)))
                if color_str is not None:
                    colors.append(np.array(json.loads(color_str)))
                    bg_colors.append(np.array(json.loads(bg_color_str)))
                times.append(timestamp)
    if len(colors) == 0:
        colors = None
        bg_colors = None
    return np.array(times), np.array(positions), pcs, pcs_bg, colors, bg_colors


def parse_real_eval_mp(log_path):
    pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\].*step (\d+) ac_time \[(.*)\] poses (\[.*\])"
    actions = []
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                timestamp, step, ac_time_str, action_str = match.groups()
                timestamp = parse_datetime(timestamp)
                expected_poses = json.loads(action_str)
                actions.append(
                    (timestamp, step, float(ac_time_str), np.array(expected_poses))
                )
    return actions


def parse_policy_node(log_path):
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):INFO:\[done\] time: \[(.*)\] state: (.*) output: (.*)"
    policy_outputs = []
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                timestamp, time_str, state_str, output_str = match.groups()
                timestamp = parse_datetime(timestamp)
                state = json.loads(state_str)[:][:3]
                if "nan" in output_str:
                    continue
                outputs = json.loads(output_str)
                policy_outputs.append((timestamp, float(time_str), state, outputs))
    return policy_outputs


def load_point_cloud(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)[0]
    data_np = np.array([data["x"], data["y"], data["z"]]).T
    indices = (data_np[:, -1] > -0.1) * (data_np[:, 1] > -0.6) * (data_np[:, 0] > -0.5)
    data_np = data_np[indices]
    return data_np, np.array(data["marker"]["color"])[indices]


def find_item_in_list(policy_outputs, time, index=0, index_offset=1):
    """
    Finds the policy output item that happens immediately before the given time.

    Parameters:
        policy_outputs (list): A list of tuples where each tuple contains (datetime, state, actions).
        time (datetime): The datetime object to find the closest policy output before it.

    Returns:
        The closest policy output item before the given time, or None if no such item exists.
    """
    # Extract the times from policy_outputs for binary search
    times = [item[index] for item in policy_outputs]

    # Use bisect to find the position to insert the given time, then subtract one to find the nearest less time
    index = bisect.bisect_left(times, time) - index_offset

    # Check if the index is valid (not before the start of the list)
    if index >= 0:
        return policy_outputs[index], index
    else:
        return None, -1


def find_policy_inference_data(policy_outputs, time):
    item, index = find_item_in_list(policy_outputs, time, index=1, index_offset=0)
    assert item is not None
    state = np.array(item[2])
    offsets = np.array(item[3])[:, :, 1:4]
    traj = np.cumsum(np.vstack([state[:, :3][None], offsets]), axis=0), index
    return traj


def find_env_data(action_data, time, state):
    item, index = find_item_in_list(action_data, time)
    assert item is not None
    expected_positions = np.array(item[-1])[:, :, :3]
    traj = np.concatenate([state[:, :3][None], expected_positions]), index
    return traj


@click.command()
@click.option("--in_dir", type=str, default=None)
def main(in_dir):
    # Define paths to the log files
    spin_obs_state_path = os.path.join(in_dir, "obs_spinner.log")
    real_eval_mp_path = os.path.join(in_dir, "real_eval_mp_dp.log")
    policy_agent_path = os.path.join(in_dir, "policy_spinner.log")

    # list of (timestamp, action)
    actions = parse_real_eval_mp(real_eval_mp_path)
    start_time = actions[0][0]
    end_time = actions[-1][0]
    time_to_str = lambda t: int((t.timestamp() - start_time.timestamp()) * 1000)

    times, positions, pcs_fg, pcs_bg, colors_fg, colors_bg = parse_obs_node(
        spin_obs_state_path
    )
    # positions[:, :, 0] = positions[:, :, 0] + 0.91
    # positions[:, :, 1] = positions[:, :, 1] + 0.91
    plot_times = [t for t in times if start_time <= t and t <= end_time]
    plot_indices = [
        i for i in range(len(times)) if times[i] >= start_time and times[i] <= end_time
    ]

    # list of (timestamp, state, action list)
    policy_outputs = parse_policy_node(policy_agent_path)

    # point cloud with shape (N, 3)
    if colors_fg is None:
        colors_fg, colors_bg = [], []
        for i in range(len(pcs_fg)):
            colors_fg.append(np.ones_like(pcs_fg[i]) * np.array([[0.0, 0.0, 0.0]]))
            colors_bg.append(np.ones_like(pcs_bg[i]) * np.array([[1.0, 0.75, 0.0]]))

    fig = go.Figure()

    # Add state marker to each frame
    frames = []
    env_steps = []
    inf_data, inf_index = None, None
    for i in plot_indices:
        time = times[i]
        frame_data = []
        for j in range(positions.shape[1]):
            frame_data += [
                go.Scatter3d(
                    x=positions[:, j, 0],
                    y=positions[:, j, 1],
                    z=positions[:, j, 2],
                    mode="lines",
                    line=dict(
                        width=3,
                        color=np.arange(len(positions)) / len(positions),
                        colorscale="viridis",
                    ),
                    name="Full EEF Trajectory",
                )
            ]
        frame_data += [
            go.Scatter3d(
                x=pcs_fg[i][:, 0],
                y=pcs_fg[i][:, 1],
                z=pcs_fg[i][:, 2],
                mode="markers",
                marker=dict(size=2, color=colors_fg[i]),
                name="Point Cloud (FG)",
            ),
            go.Scatter3d(
                x=pcs_bg[i][:, 0],
                y=pcs_bg[i][:, 1],
                z=pcs_bg[i][:, 2],
                mode="markers",
                marker=dict(size=2, color=colors_bg[i]),
                name="Point Cloud (BG)",
            ),
        ]

        # added eef pos visualization
        eef_pos = positions[i]
        for j in range(len(eef_pos)):
            frame_data.append(
                go.Scatter3d(
                    x=[eef_pos[j][0]],
                    y=[eef_pos[j][1]],
                    z=[eef_pos[j][2]],
                    marker=dict(size=6, color="red"),
                    text=[f"Current state: {eef_pos[j].round(3)}"],
                    name="Current EEF Pos",
                )
            )

        # add action visualization
        env_data, env_index = find_env_data(actions, time, eef_pos)
        env_step = int(actions[env_index][1])
        """
        for j in range(env_data.shape[1]):
            frame_data.append(
                go.Scatter3d(
                    x=env_data[:, j, 0],
                    y=env_data[:, j, 1],
                    z=env_data[:, j, 2],
                    mode="lines+markers",
                    name="Env Actions",
                    marker=dict(size=2.5, color="cyan"),
                )
            )
        """

        # add policy inference information
        env_ac_time = actions[env_index][2]
        inf_data, inf_index = find_policy_inference_data(policy_outputs, env_ac_time)
        for j in range(inf_data.shape[1]):
            frame_data.append(
                go.Scatter3d(
                    x=inf_data[:, j, 0],
                    y=inf_data[:, j, 1],
                    z=inf_data[:, j, 2],
                    mode="lines",
                    name="Policy Inference",
                    marker=dict(size=3, color="orange"),
                )
            )

        frames.append(go.Frame(data=frame_data, name=time_to_str(time)))
        env_steps.append(env_step)
    fig.update(frames=frames)

    fig.update(data=frames[0].data)

    # slider
    slider = dict(
        steps=[
            {
                "method": "animate",
                "args": [
                    [time_to_str(time)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"T = {time_to_str(time)/1000:.3f}s; env step = {env_steps[k]}",
            }
            for k, time in enumerate(plot_times)
        ]
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="Robot State and Actions Visualization",
        sliders=[slider],
    )

    fig.update_scenes(
        # xaxis=dict(range=[2.0, 6.0]),
        # yaxis=dict(range=[-5., -1.]),
        xaxis=dict(range=[-3.0, 3]),
        yaxis=dict(range=[-2.0, 4.0]),
        zaxis=dict(range=[0.0, 2.0]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.25),
    )

    fig.write_html(os.path.join(in_dir, f"vis_pc.html"))


if __name__ == "__main__":
    main()
