import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorsys

matplotlib.use("Agg")


def vectors_to_colors(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / np.maximum(
        norms[:, np.newaxis], 1e-6
    )  # Avoid division by zero

    azimuths = np.arctan2(normalized_vectors[:, 1], normalized_vectors[:, 0])
    polar_angles = np.arccos(normalized_vectors[:, 2])

    colors = []
    for azimuth, polar_angle, norm in zip(azimuths, polar_angles, norms):
        if norm < 1e-6:  # Check if the vector is close to zero
            colors.append([0.0, 0.0, 0.0])  # Set color to black for zero vectors
        else:
            hue = (azimuth + np.pi) / (2 * np.pi)
            saturation = polar_angle / np.pi
            value = 1.0

            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([r, g, b])

    return np.clip(np.array(colors), 0.0, 1.0)


def set_axes_equal(ax, canvas_radius=None):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    if canvas_radius is not None:
        ax.set_xlim3d([-canvas_radius, canvas_radius])
        ax.set_ylim3d([-canvas_radius, canvas_radius])
        ax.set_zlim3d([-canvas_radius, canvas_radius])
    else:
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def set_background_blank(ax):
    # Hide grid lines
    ax.grid(False)
    ax.set_axis_off()
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def plot_points(
    points, save_path, marker_size=60, color=None, blank_bg=False, canvas_radius=None
):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if blank_bg:
        set_background_blank(ax)
    color = color if color is not None else np.ones_like(points) * 0.3
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=marker_size,
        c=color,
        edgecolors="white",
        linewidth=0.5,
        alpha=1,
        antialiased=True,
    )
    set_axes_equal(ax, canvas_radius)
    fig.tight_layout()
    fig.savefig(save_path)


def plot_points_grid(
    points,
    save_path=None,
    marker_size=60,
    color=None,
    blank_bg=False,
    canvas_radius=None,
    titles=None,
    arrows=None,
    parameterize_scale_by_color=False,
):
    if len(points[0].shape) == 2:
        n_plot = len(points)
        fig = plt.figure(figsize=(5 * n_plot, 5))
        for i in range(n_plot):
            ax = fig.add_subplot(1, n_plot, i + 1, projection="3d")
            if blank_bg:
                set_background_blank(ax)
            color = (
                color if color is not None else [np.ones_like(p) for p in points] * 0.3
            )
            if parameterize_scale_by_color:
                sizes = marker_size * (1.0 + 1.0 / 0.9 * (0.9 - color[i][:, 0]))
            else:
                sizes = marker_size
            ax.scatter(
                points[i][:, 0],
                points[i][:, 1],
                points[i][:, 2],
                s=sizes,
                c=color[i],
                edgecolors="white",
                alpha=1,
                antialiased=True,
            )
            if arrows is not None and len(arrows[i]) > 0:
                arrow_list = np.nan_to_num(arrows[i])
                ax.quiver(
                    arrow_list[0][:, 0],
                    arrow_list[0][:, 1],
                    arrow_list[0][:, 2],
                    arrow_list[1][:, 0],
                    arrow_list[1][:, 1],
                    arrow_list[1][:, 2],
                    color=vectors_to_colors(arrow_list[1]),
                    alpha=1.0,
                    lw=2,
                )
            if titles is not None:
                ax.set_title(titles[i], y=1.0, pad=-20, fontsize=20)
            set_axes_equal(ax, canvas_radius)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    elif len(points.shape) == 4:
        n_plot, m_plot = points.shape[0], points.shape[1]
        fig = plt.figure(figsize=(5 * m_plot, 5 * n_plot))
        for i in range(n_plot):
            for j in range(m_plot):
                ax = fig.add_subplot(
                    n_plot, m_plot, i * m_plot + j + 1, projection="3d"
                )
                if blank_bg:
                    set_background_blank(ax)
                color = color if color is not None else np.ones_like(points) * 0.3
                ax.scatter(
                    points[i, j, :, 0],
                    points[i, j, :, 1],
                    points[i, j, :, 2],
                    s=marker_size,
                    c=color[i, j],
                    edgecolors="white",
                    linewidth=0.5,
                    alpha=1,
                    antialiased=True,
                )
                set_axes_equal(ax, canvas_radius)
    else:
        raise NotImplementedError("Dimension not match!")
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
    else:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data
