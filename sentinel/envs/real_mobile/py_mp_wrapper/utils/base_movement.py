import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from IPython.display import Image
import os


def find_intersection_points(intersection):
    intersection_points = []
    if not intersection.is_empty:
        if intersection.geom_type == "Point":
            intersection_points.append(intersection)
        elif intersection.geom_type == "MultiPoint":
            intersection_points.extend(intersection.geoms)
        elif intersection.geom_type == "LineString":
            intersection_points.extend(
                [Point(coords) for coords in intersection.coords]
            )
        elif isinstance(intersection, MultiLineString):
            for linestring in intersection.geoms:
                intersection_points.extend(
                    [Point(coords) for coords in linestring.coords]
                )
    return intersection_points


def move_robot_base_in_xy(
    workspace,
    target_eef_xy,
    cur_base_xy,
    primary_move_dir=None,
    arm_reaching_recommend=0.8,  # 0.6
    arm_reaching_min=0.55,  # 0.45
    arm_reaching_max=1.1,  # 0.9
    vis=False,
):

    # Normalize the primary move direction
    if primary_move_dir is not None:
        # Will be used later to move the base in the primary move direction!
        primary_move_dir = primary_move_dir / np.linalg.norm(primary_move_dir)

    # Calculate distance and inverse direction
    dist = np.linalg.norm(target_eef_xy - cur_base_xy)
    target_dir = (target_eef_xy - cur_base_xy) / dist

    # Create circles and ring
    center = Point(target_eef_xy)
    outer_circle = center.buffer(arm_reaching_max)
    inner_circle = center.buffer(arm_reaching_min)
    preferred_circle = center.buffer(arm_reaching_recommend)

    # Check whether the cur_base_xy is within those circles
    too_close = inner_circle.contains(Point(cur_base_xy))
    too_far_away = not outer_circle.contains(Point(cur_base_xy))

    # Pre-init the variables to return
    new_base_xy = None
    new_base_xy_candidate = None

    if not too_far_away:
        # The current base position is within the outer circle
        if too_close:
            # The current base position is within the arm reaching boundary
            new_base_xy_candidate = cur_base_xy
        else:
            # Move backward from the target position
            new_base_xy_candidate = target_eef_xy - target_dir * arm_reaching_recommend
    else:
        # The current base position is too far away from the target position
        # Need to generate a line from cur_base_xy to target_eef_xy
        line = LineString([cur_base_xy, target_eef_xy])
        intersection = line.intersection(preferred_circle)
        intersection_points = find_intersection_points(intersection)

        for point in intersection_points:
            if not inner_circle.contains(point):
                new_base_xy_candidate = np.array(point.coords[0])
                break

    if new_base_xy_candidate is None:
        # If no valid candidate was found, use the current base position
        new_base_xy_candidate = cur_base_xy

    if workspace.contains(Point(new_base_xy_candidate)):
        new_base_xy = new_base_xy_candidate
    else:
        # Use new_base_xy_candidate as the starting point to find the closest point in the workspace
        new_base_xy = np.array(
            workspace.exterior.interpolate(
                workspace.exterior.project(Point(new_base_xy_candidate))
            ).coords[0]
        )

    new_dist = np.linalg.norm(target_eef_xy - new_base_xy)

    # store the new_base_xy, new_dist, outer_circle, inner_circle into a dict()
    base_motion_info = {"new_base_xy": new_base_xy}
    base_motion_info.update(
        {
            "new_dist": new_dist,
        }
    )

    if vis:
        # If visualization is enabled, include additional details in the info dictionary
        base_motion_info.update(
            {"outer_circle": outer_circle, "inner_circle": inner_circle}
        )

    return base_motion_info


class BaseMoveStrategyVis:
    def __init__(self, workspace):
        self.workspace = Polygon(workspace)
        self.workspace_coords = np.array(workspace)

    def plan(self, current_base_pose, target_eef_posis, primary_move_dir, gif_path):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            self.workspace_coords[:, 0],
            self.workspace_coords[:, 1],
            "bo",
            label="Workspace Points",
        )
        sc_target = ax.scatter([], [], color="m", label="Target Position")
        sc_base = ax.scatter(
            [], [], color="c", label="New Base Position", marker="s", s=200
        )
        distance_vector = None
        move_vector = None
        outer_circle_patch = None
        inner_circle_patch = None
        distance_text = ax.text(
            0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
        )
        ax.legend()
        ax.set_xlim(
            self.workspace_coords[:, 0].min() - 1, self.workspace_coords[:, 0].max() + 1
        )
        ax.set_ylim(
            self.workspace_coords[:, 1].min() - 1, self.workspace_coords[:, 1].max() + 1
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Robot Workspace and Movements")
        ax.grid(True)

        def init():
            sc_target.set_offsets(np.empty((0, 2)))
            sc_base.set_offsets(np.empty((0, 2)))
            distance_text.set_text("")
            return sc_target, sc_base, distance_text

        def update(frame):
            nonlocal current_base_pose, distance_vector, move_vector, outer_circle_patch, inner_circle_patch
            target_eef_xy = target_eef_posis[frame]
            base_move_info = move_robot_base_in_xy(
                self.workspace,
                target_eef_xy,
                current_base_pose,
                primary_move_dir,
                vis=True,
            )
            new_base_pose, new_dist, outer_circle, inner_circle = (
                base_move_info["new_base_xy"],
                base_move_info["new_dist"],
                base_move_info["outer_circle"],
                base_move_info["inner_circle"],
            )
            sc_target.set_offsets([target_eef_xy])
            sc_base.set_offsets([new_base_pose])

            # Update the distance vector
            if distance_vector is not None:
                distance_vector.remove()
            distance_vector = ax.arrow(
                new_base_pose[0],
                new_base_pose[1],
                target_eef_xy[0] - new_base_pose[0],
                target_eef_xy[1] - new_base_pose[1],
                head_width=0.05,
                color="b",
            )

            # Plot the primary move direction as a red dashed arrow
            if move_vector is not None:
                move_vector.remove()
            move_vector = ax.arrow(
                current_base_pose[0],
                current_base_pose[1],
                primary_move_dir[0],
                primary_move_dir[1],
                head_width=0.05,
                color="r",
                linestyle="dashed",
            )

            # Plot the movement from the previous step to the current step
            ax.plot(
                [current_base_pose[0], new_base_pose[0]],
                [current_base_pose[1], new_base_pose[1]],
                "g-",
            )

            # Plot the outer circle
            if outer_circle_patch is not None:
                outer_circle_patch.remove()
            outer_circle_patch = plt.Polygon(
                np.array(outer_circle.exterior.coords),
                color="blue",
                alpha=0.3,
                label="Outer Circle",
            )
            ax.add_patch(outer_circle_patch)

            # Plot the inner circle
            if inner_circle_patch is not None:
                inner_circle_patch.remove()
            inner_circle_patch = plt.Polygon(
                np.array(inner_circle.exterior.coords),
                color="green",
                alpha=0.3,
                label="Inner Circle",
            )
            ax.add_patch(inner_circle_patch)

            # Update the distance text
            distance_text.set_text(f"Distance: {new_dist:.2f}")

            current_base_pose = new_base_pose
            return (
                sc_target,
                sc_base,
                distance_vector,
                move_vector,
                outer_circle_patch,
                inner_circle_patch,
                distance_text,
            )

        ani = FuncAnimation(
            fig,
            update,
            frames=len(target_eef_posis),
            init_func=init,
            blit=False,
            repeat=False,
        )

        try:
            # Ensure the directory exists before saving the animation
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            # Save the animation as a GIF
            ani.save(gif_path, writer=PillowWriter(fps=1))
        except Exception as e:
            print(f"Error saving animation: {e}")

        # Close the figure to prevent the static plot from showing
        plt.close(fig)

        # Check if the file was created successfully before trying to display it
        if os.path.exists(gif_path):
            return Image(filename=gif_path)
        else:
            print(f"Error: File {gif_path} was not created.")
            return None
