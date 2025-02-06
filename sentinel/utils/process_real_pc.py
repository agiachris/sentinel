import os
import time
import numpy as np
import cv2
import deva
import click
import torch
from omegaconf import OmegaConf
from glob import glob
import open3d as o3d

# plotting
import imageio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# depth inference
import mob_manip_vision.tri_inference as tri_inference
import mob_manip_vision.pose_utils as pose_utils

from sentinel.utils.depth_to_3d import make_pointcloud

from multiprocessing.pool import ThreadPool


def animate_pcs(
    pcs,
    colors=None,
    arrow_bases=None,
    arrow_heads=None,
    arrow_colors=None,
    save_path=None,
):
    # Create a fixed coordinate limit
    coord_limit = 6.0

    if colors is None:
        colors = np.zeros_like(pcs)

    # Create the Plotly figure and subplot
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    scatter = go.Scatter3d(
        x=[], y=[], z=[], mode="markers", marker=dict(size=5, color=[])
    )

    # Set the initial data for the scatter plot
    scatter.x = pcs[0][:, 0]
    scatter.y = pcs[0][:, 1]
    scatter.z = pcs[0][:, 2]
    scatter.marker.color = colors[0]
    scatter.text = [
        f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}\n{rgb_to_hsv(c)}"
        for x, y, z, c in zip(pcs[0][:, 0], pcs[0][:, 1], pcs[0][:, 2], colors[0])
    ]

    # Add the scatter plot to the figure
    fig.add_trace(scatter)

    # Add arrows if needed
    arrow_data = []
    if arrow_bases is not None:
        for i in range(len(arrow_bases)):
            arrow_data.append([])
            for j in range(len(arrow_bases[i])):
                line_points = np.array(
                    [arrow_bases[i][j], arrow_heads[i][j], np.array([np.nan] * 3)]
                )
                arrow_data[-1].append(
                    go.Scatter3d(
                        x=[arrow_bases[i][j, 0]],
                        y=[arrow_bases[i][j, 1]],
                        z=[arrow_bases[i][j, 2]],
                        marker=dict(size=7.5, color="#4285f4"),
                        text=[
                            f"Gripper {i} Position: ({arrow_bases[i][j, 0]:.2f}, {arrow_bases[i][j, 1]:.2f}, {arrow_bases[i][j, 2]:.2f})"
                        ],
                    )
                )
                color_a2s = (
                    lambda x: f"rgb({int(x[0] * 255)},{int(x[1] * 255)},{int(x[2] * 255)})"
                )
                arrow_data[-1].append(
                    go.Scatter3d(
                        x=line_points[:, 0],
                        y=line_points[:, 1],
                        z=line_points[:, 2],
                        line=dict(color=color_a2s(arrow_colors[i][j]), width=10),
                        mode="lines",
                    )
                )

            arrow_data[-1].append(
                go.Cone(
                    x=arrow_heads[i][:, 0],
                    y=arrow_heads[i][:, 1],
                    z=arrow_heads[i][:, 2],
                    u=(arrow_heads - arrow_bases)[i][:, 0],
                    v=(arrow_heads - arrow_bases)[i][:, 1],
                    w=(arrow_heads - arrow_bases)[i][:, 2],
                    sizemode="scaled",
                    sizeref=0.2,
                    showscale=False,
                    cmin=0,
                    cmax=1,
                    colorscale=[[0, "rgb(0,0,0)"], [1, "rgb(0,0,0)"]],
                )
            )
        for d in arrow_data[0]:
            fig.add_trace(d)

    # Set axis limits
    fig.update_scenes(
        xaxis=dict(range=[-coord_limit, coord_limit]),
        yaxis=dict(range=[-coord_limit, coord_limit]),
        zaxis=dict(range=[-coord_limit, coord_limit]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update(layout_showlegend=False)

    # Create the animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=pc[:, 0],
                    y=pc[:, 1],
                    z=pc[:, 2],
                    marker=dict(size=2.5, color=c),
                    text=[
                        f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}\n{rgb_to_hsv(color)}"
                        for x, y, z, color in zip(pc[:, 0], pc[:, 1], pc[:, 2], c)
                    ],
                )
            ],
            name=str(i),
        )
        for i, (pc, c) in enumerate(zip(pcs[1:], colors[1:]), start=1)
    ]

    # Add arrows to animated frames if needed
    if len(arrow_data) > 1:
        for i in range(len(arrow_bases) - 1):
            frames[i].data = [frames[i].data[0]] + arrow_data[i + 1]

    # Add the frames to the figure
    fig.update(frames=frames)

    buttons = dict(
        type="buttons",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                    ),
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                ],
            ),
        ],
    )

    # Update layout settings
    fig.update_layout(updatemenus=[buttons], uirevision=True)

    slider = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "text-before-value-on-display",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }
    for i, frame in enumerate(frames):
        step = dict(
            method="update",
            args=[
                [i],
                {
                    "frame": {"duration": 500, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 500},
                },
                {"title": f"Timestep {i + 1} / {len(fig.data)}"},
            ],
            label=i,
        )
        slider["steps"].append(step)
    fig.update_layout(sliders=[slider])

    # Show the interactive plot
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def set_background_blank(ax):
    # Hide grid lines
    # ax.grid(False)
    # ax.set_axis_off()
    # Hide axes ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    # ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


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


def animate_pcs_mpl(
    all_pcs, all_colors=None, save_path=None, bg_color="white", azim=120
):
    images = []
    if all_colors is None:
        all_colors = [None] * len(all_pcs)
    for points, colors in zip(all_pcs, all_colors):
        if colors is None:
            colors = np.zeros_like(points)
        fig = plt.figure(figsize=(5, 5), facecolor=bg_color)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_facecolor(bg_color)
        set_background_blank(ax)
        ax.view_init(45.0, azim=azim)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1,
            c=colors / 255,
            alpha=1,
            antialiased=True,
        )
        set_axes_equal(ax)
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_np = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        images.append(image_np)
        plt.close(fig)

    imageio.mimsave(save_path, images, duration=1000 / 5)


def rotate_point_cloud_around_y_axis(
    point_cloud, degrees, pivot=np.array([0.0, 0.0, 0.0])
):
    # Convert degrees to radians
    radians = np.radians(degrees)

    # Define the rotation matrix around the y-axis
    rotation_matrix = np.array(
        [
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)],
        ]
    )

    # Apply the rotation to the point cloud
    rotated_point_cloud = (
        np.dot(point_cloud - pivot[None], rotation_matrix.T) + pivot[None]
    )

    return rotated_point_cloud


class RealPCProcessor(object):
    def __init__(
        self, model_path, calib_dir, prompt_list, num_points=8192, tracker_crop=None
    ):
        self.model = self._load_model(model_path)
        self.model.eval()
        self.calib_dir = calib_dir
        self.cam_ids = self._get_cam_ids(calib_dir)
        self.intrinsics, self.transforms = self._load_calib(calib_dir)
        if torch.cuda.is_available():
            self.transforms_torch = {
                k: torch.tensor(v, dtype=torch.float64, device="cuda")
                for k, v in self.transforms.items()
            }
        self.num_points = num_points
        self._t = 0

        # initialize object tracker
        print("[process_real_pc.py] Loading object tracker...")
        from sentinel.vision_module.agents.deva_agent import VideoTrackAgent

        self.tracker = VideoTrackAgent(
            self._make_deva_config(), prompt_list, crop=tracker_crop
        )
        print("[process_real_pc.py] Finished loading object tracker.")

    def reset(self):
        self.tracker._t = 0

    def _make_deva_config(self):
        deva_save_dir = os.path.join(os.path.dirname(deva.__file__), "../saves")
        deva_config = OmegaConf.create(
            dict(
                detection_every=100,
                size=480,
                mem_every=10,
                chunk_size=2,
                model=os.path.join(deva_save_dir, "DEVA-propagation.pth"),
                GROUNDING_DINO_CONFIG_PATH=os.path.join(
                    deva_save_dir, "GroundingDINO_SwinT_OGC.py"
                ),
                GROUNDING_DINO_CHECKPOINT_PATH=os.path.join(
                    deva_save_dir, "groundingdino_swint_ogc.pth"
                ),
                SAM_CHECKPOINT_PATH=os.path.join(deva_save_dir, "sam_vit_h_4b8939.pth"),
                MOBILE_SAM_CHECKPOINT_PATH=os.path.join(deva_save_dir, "mobile_sam.pt"),
            )
        )
        return deva_config

    def _get_cam_ids(self, calib_dir):
        calib_files = list(glob(os.path.join(calib_dir, "ZED_**depth.npz")))
        cam_ids = [int(fn.split("_")[-3]) for fn in calib_files]
        return cam_ids

    def _load_calib(self, calib_dir):
        intrinsics = {}
        for cam_idx, cam_id in enumerate(self.cam_ids):
            d = np.load(os.path.join(calib_dir, f"ZED_{cam_id:d}_calib_depth.npz"))
            intrinsics[cam_id] = np.array(
                [[d["fx"], 0.0, d["cx"]], [0.0, d["fy"], d["cy"]], [0.0, 0.0, 1.0]]
            )
        transforms, _ = pose_utils.load_transforms(
            self.cam_ids, calib_dir, recompute=False
        )
        print("[process_real_pc.py] Loaded calibration matrices")
        return intrinsics, transforms

    def _load_model(self, model_path):
        print("Loading TRI image model from", model_path)
        # There is currently an issue in PyTorch 1.13 and later that causes the
        # stereo inference to lock up on the second inference attempt. This hack
        # seems to resolve the issue.
        torch._C._jit_set_nvfuser_enabled(False)
        # Load model and move to GPU.
        model = torch.jit.load(model_path)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        print("Loading done.")
        return model

    def preprocess_image(self, img):
        height, width, _ = img.shape
        assert height % 2 == 0
        assert width % 2 == 0
        height_delta = (height % 32) // 2
        width_delta = (width % 32) // 2
        cropped = img[
            height_delta : height - height_delta, width_delta : width - width_delta, :
        ]
        return cropped

    def tracker_step(self, img, return_numpy=False):
        with torch.no_grad():
            seg_mask = self.tracker.step(img, return_numpy=return_numpy)
        return seg_mask.detach()

    def model_fwd(self, left_tensor, right_tensor, disparity):
        with torch.no_grad():
            output, _ = self.model(left_tensor, right_tensor, num_disparities=disparity)
        return output["disparity"]

    def rgb2pc(
        self,
        left,
        right,
        cam_matrix,
        disparity=384,
        apply_filter=True,
        num_points=None,
        return_bg=False,
        verbose=False,
    ):
        # Convert inputs from Numpy arrays in 0 to 255 to PyTorch tensors in 0 to 1.
        if verbose:
            tt = time.time()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        left_tensor = torch.from_numpy(left.astype(np.uint8))
        right_tensor = torch.from_numpy(right.astype(np.uint8))
        if torch.cuda.is_available():
            left_tensor = left_tensor.cuda()
            right_tensor = right_tensor.cuda()
        left_tensor = left_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255
        right_tensor = (
            right_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255
        )
        if verbose:
            print(
                f"[process_real_pc.py => rgb2pc] move image to torch took {time.time() - tt:.3f}s"
            )

            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[process_real_pc.py => rgb2pc] move image to torch took {time_elapsed} ms in CUDA-SYNC Timer"
            )
            tt = time.time()
            start.record()

        # TODO: Parallelize self.tracker.step and self.model.forward
        # manager = Manager()
        # queue = manager.Queue()

        # Create processes
        # with torch.no_grad():
        # TODO: Parallelize self.tracker.step and self.model.forward
        # manager = Manager()
        # queue = manager.Queue()

        USE_THREAD = False
        # USE_THREAD = True
        if USE_THREAD:
            pool = ThreadPool()
            seg_mask_result = pool.apply_async(self.tracker_step, args=(left, False))
            disparity_result = pool.apply_async(
                self.model_fwd, args=(left_tensor, right_tensor, disparity)
            )
            pool.close()
            while not all([res.ready() for res in [seg_mask_result, disparity_result]]):
                time.sleep(0.001)
            seg_mask = seg_mask_result.get()
            disparity_output = disparity_result.get()
        else:
            seg_mask = self.tracker_step(left, return_numpy=False)
            disparity_output = self.model_fwd(left_tensor, right_tensor, disparity)

        if verbose:
            print(
                f"[process_real_pc.py => rgb2pc]  <USE_THREAD={USE_THREAD}> tracker_step + model_fwd took {time.time() - tt:.3f}s"
            )
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[process_real_pc.py => rgb2pc] <USE_THREAD={USE_THREAD}> tracker_step + model_fwd took {time_elapsed} ms in CUDA-SYNC Timer"
            )
            tt = time.time()
            start.record()

        # # Start processes
        # tracker_process.start()
        # model_process.start()

        # # Wait for both processes to finish
        # tracker_process.join()
        # model_process.join()
        # # Retrieve results from the queue
        # results = []
        # while not queue.empty():
        #     import pdb; pdb.set_trace()
        #     seg_mask = queue[0]
        #     disparity = queue[1]

        # # Mask objects in the image; returns result with shape (H, W)
        # with torch.no_grad():
        #     seg_mask = self.tracker.step(left, return_numpy=False)
        # if verbose:
        #     print(
        #         f"[process_real_pc.py => rgb2pc] tracker took {time.time() - tt:.3f}s"
        #     )
        #     end.record()
        #     torch.cuda.synchronize()  # Wait for all operations to complete
        #     # Calculate time elapsed
        #     time_elapsed = start.elapsed_time(end)  # Time in milliseconds
        #     print(f"[process_real_pc.py => rgb2pc] tracker took {time_elapsed} ms in CUDA-SYNC Timer")
        #     tt = time.time()
        #     start.record()

        if apply_filter:
            # note: this might take time too
            seg_mask_flattened = seg_mask.detach().flatten()
            sampled_indices = torch.argwhere(seg_mask_flattened).flatten()
            if num_points is not None:
                sampled_indices = sampled_indices[
                    torch.randint(len(sampled_indices), (num_points,)).to(
                        sampled_indices.device
                    )
                ]
            if return_bg:
                bg_indices = torch.argwhere(seg_mask_flattened == 0).flatten()
                if num_points is not None:
                    bg_indices = bg_indices[
                        torch.randint(len(bg_indices), (num_points * 2,)).to(
                            bg_indices.device
                        )
                    ]
                sampled_indices = torch.cat([sampled_indices, bg_indices])
            colors = left.reshape(-1, 3)
            colors = colors[sampled_indices.cpu().numpy()]
        else:
            sampled_indices = None
            colors = left
        seg_mask = seg_mask.detach().cpu().numpy()

        # if verbose:
        #     print(
        #         f"[process_real_pc.py => rgb2pc] apply filter took {time.time() - tt:.3f}s"
        #     )
        #     end.record()
        #     torch.cuda.synchronize()  # Wait for all operations to complete
        #     # Calculate time elapsed
        #     time_elapsed = start.elapsed_time(end)  # Time in milliseconds
        #     print(f"[process_real_pc.py => rgb2pc] apply filter took {time_elapsed} ms in CUDA-SYNC Timer")

        #     tt = time.time()
        #     start.record()

        # # Do forward pass on model and get output.
        # with torch.no_grad():
        #     # import pdb; pdb.set_trace()

        #     # # downsacle left_tensor and right_tensor to 1/4
        #     # # Calculate the starting indices for the bottom right quarter
        #     # height_start = left_tensor.size(2) // 2  # 1216 / 2 = 608
        #     # width_start = -320  # 2208 / 2 = 1104
        #     # # left_tensor = torch.nn.functional.interpolate(left_tensor, scale_factor=0.25, mode='bilinear', align_corners=False)
        #     # # right_tensor = torch.nn.functional.interpolate(right_tensor, scale_factor=0.25, mode='bilinear', align_corners=False)
        #     # left_tensor = left_tensor[:, :, height_start:, width_start:]
        #     # right_tensor = right_tensor[:, :, height_start:, width_start:]

        #     output, _ = self.model(left_tensor, right_tensor, num_disparities=disparity)

        # if verbose:
        #     print(f"[process_real_pc.py => rgb2pc] fwd took {time.time() - tt:.3f}s")
        #     end.record()
        #     torch.cuda.synchronize()  # Wait for all operations to complete
        #     # Calculate time elapsed
        #     time_elapsed = start.elapsed_time(end)  # Time in milliseconds
        #     print(f"[process_real_pc.py => rgb2pc] fwd took {time_elapsed} ms in CUDA-SYNC Timer")

        #     tt = time.time()
        #     start.record()

        # disparity = output["disparity"]

        # upsample disparity to original size
        # disparity = torch.nn.functional.interpolate(output["disparity"], scale_factor=4, mode='bilinear', align_corners=False)

        if verbose:
            print(
                f"[process_real_pc.py => rgb2pc] getting disparity_output took {time.time() - tt:.3f}s"
            )
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[process_real_pc.py => rgb2pc] getting disparity_output took {time_elapsed} ms in CUDA-SYNC Timer"
            )

            tt = time.time()
            start.record()

        pc_xyz = make_pointcloud(cam_matrix, disparity_output, sampled_indices)
        if verbose:
            print(
                f"[process_real_pc.py => rgb2pc] depth2pc took {time.time() - tt:.3f}s"
            )
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[process_real_pc.py => rgb2pc] getting disparity_output took {time_elapsed} ms in CUDA-SYNC Timer"
            )

            tt = time.time()
            start.record()

        if apply_filter:
            pass
        else:
            pc_xyz = pc_xyz[0].permute(1, 2, 0)
        if verbose:
            print(
                f"[process_real_pc.py => rgb2pc] process pc took {time.time() - tt:.3f}s"
            )
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[process_real_pc.py => rgb2pc] process pc took {time_elapsed} ms in CUDA-SYNC Timer"
            )

            tt = time.time()
            start.record()

        if return_bg:
            pc_fg_xyz, pc_bg_xyz = pc_xyz[:num_points], pc_xyz[num_points:]
            fg_colors, bg_colors = colors[:num_points], colors[num_points:]
            return pc_fg_xyz, pc_bg_xyz, fg_colors, bg_colors, seg_mask
        else:
            return pc_xyz, colors, seg_mask

    def transform_pc(self, points, transform):
        assert torch.is_tensor(points)
        assert torch.is_tensor(transform)
        assert transform.shape == (4, 4)
        assert points.shape[1] == 3
        return torch.matmul(points, transform[:3, :3].T) + transform[:3, 3][None]

    def remove_pc_outliers(self, pcs, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs)
        cl, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcs, colors = pcs[idx], colors[idx]
        nonzero_idxs = np.where(np.any(colors != 0, axis=1))[0]
        pcs, colors = pcs[nonzero_idxs], colors[nonzero_idxs]
        return pcs, colors

    def set_queries(self, queries):
        self.tracker.update_sub_prompts(queries)

    def color_filter_and_make_pc(
        self,
        imgs,
        cam_id,
        view="side",
        verbose=False,
        raw_pcs=None,
        raw_colors=None,
        return_bg=False,
    ):
        """
        Total time taken: 126 ms!
        """

        # start_total = torch.cuda.Event(enable_timing=True)
        # end_total = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start_total.record()

        # make side length multiples of 32
        shape_before = imgs[0].shape
        imgs = [self.preprocess_image(img) for img in imgs]
        shape_after = imgs[1].shape

        # adjust intrinsics matrix
        intrinsics = self.intrinsics[cam_id]
        torch_intrinsics = torch.zeros(3, 3, dtype=torch.float64)
        if torch.cuda.is_available():
            torch_intrinsics = torch_intrinsics.cuda()
        torch_intrinsics[0, 0] = (
            intrinsics[0, 0] * shape_after[1] / 2208
        )  # shape_before[1]
        torch_intrinsics[1, 1] = (
            intrinsics[1, 1] * shape_after[0] / 1242
        )  # shape_before[0]
        torch_intrinsics[0, 2] = (
            intrinsics[0, 2] * shape_before[1] / 2208
        )  # shape_before[1]
        torch_intrinsics[1, 2] = (
            intrinsics[1, 2] * shape_before[0] / 1242
        )  # shape_before[0]
        torch_intrinsics[2, 2] = 1.0

        assert len(imgs) == 2

        # end.record()
        # torch.cuda.synchronize()  # Wait for all operations to complete
        # # Calculate time elapsed
        # time_elapsed = start_total.elapsed_time(end)  # Time in milliseconds
        # print(f"[process_real_pc.py] init torch_intrinsics took {time_elapsed} ms in CUDA-SYNC Timer")

        # Timing start

        # st = time.time()

        # start.record()
        # print(">>>>>start_total.record()")

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        if raw_pcs is None or (type(raw_pcs) is list and raw_pcs[0] is None):
            with torch.no_grad():
                ret_list = self.rgb2pc(
                    imgs[0],
                    imgs[1],
                    torch_intrinsics,
                    num_points=self.num_points,
                    return_bg=return_bg,
                    verbose=verbose,
                )
                if return_bg:
                    pcs, bg_pcs, colors, bg_colors, seg_mask = ret_list
                else:
                    pcs, colors, seg_mask = ret_list
        else:
            pcs, colors = raw_pcs.reshape(-1, 3), raw_colors.reshape(-1, 3)
            seg_mask = None
        if verbose:
            # print(f"[process_real_pc.py] rgb2pc took {time.time() - st:.3f}s")
            # end.record()
            # torch.cuda.synchronize()  # Wait for all operations to complete
            # # Calculate time elapsed
            # time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            # print(f"[process_real_pc.py] rgb2pc took {time_elapsed} ms in CUDA-SYNC Timer")
            st = time.time()

            # start.record()
        # transforms_torch = torch.tensor(self.transforms[cam_id].astype(np.float32), device=pcs.device)
        # if verbose:
        #     print(f"[process_real_pc.py] transform+crop.0 took {time.time() - st:.3f}s")
        #     st = time.time()
        pcs = self.transform_pc(pcs, self.transforms_torch[cam_id])
        if verbose:
            print(f"[process_real_pc.py] transform+crop.0 took {time.time() - st:.3f}s")
            # end.record()
            # torch.cuda.synchronize()  # Wait for all operations to complete
            # # Calculate time elapsed
            # time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            # print(f"[process_real_pc.py] transform+crop.0 took {time_elapsed} ms in CUDA-SYNC Timer")
            st = time.time()
            # start.record()

        if return_bg:
            bg_pcs = self.transform_pc(bg_pcs, self.transforms_torch[cam_id])

        if verbose:
            print(f"[process_real_pc.py] transform+crop.1 took {time.time() - st:.3f}s")
            # end.record()
            # torch.cuda.synchronize()  # Wait for all operations to complete
            # # Calculate time elapsed
            # time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            # print(f"[process_real_pc.py] transform+crop.1 took {time_elapsed} ms in CUDA-SYNC Timer")
            st = time.time()
            # start.record()

        pcs, colors = self.remove_pc_outliers(pcs.detach().cpu(), colors)
        if verbose:
            print(f"[process_real_pc.py] outlier removal took {time.time() - st:.3f}s")
            # end.record()
            # torch.cuda.synchronize()  # Wait for all operations to complete
            # # Calculate time elapsed
            # time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            # print(f"[process_real_pc.py] outlier removal took {time_elapsed} ms in CUDA-SYNC Timer")
            st = time.time()
            # # start.record()

        if verbose:
            print(
                f"[process_real_pc.py] (t = {self._t}) color filtering got {len(pcs)} points."
            )

        self._t += 1
        assert torch.is_tensor(pcs)

        # end_total.record()
        # torch.cuda.synchronize()  # Wait for all operations to complete
        # # Calculate time elapsed
        # time_elapsed = start_total.elapsed_time(end_total)  # Time in milliseconds
        # print(f"[process_real_pc.py] Entire function took {time_elapsed} ms in CUDA-SYNC Timer")

        if return_bg:
            return (
                imgs,
                pcs.cpu().numpy(),
                bg_pcs.cpu().numpy(),
                colors,
                bg_colors,
                seg_mask,
            )
        else:
            return imgs, pcs.cpu().numpy(), colors, seg_mask


@click.command()
@click.option(
    "--model_path",
)
@click.option(
    "--in_dir",
    type=str,
)
@click.option(
    "--calib_dir",
)
@click.option("--cam_ids", default="21172477", type=str)
@click.option(
    "--output_dir",
)
@click.option("--prompt", default="brown box", type=str)
@click.option("--num_frames", default=11, type=int)
def main(model_path, in_dir, calib_dir, cam_ids, output_dir, prompt, num_frames):
    cam_id_str = cam_ids.replace(",", "_")
    cam_ids = [int(s) for s in cam_ids.split(",")]
    os.makedirs(output_dir, exist_ok=True)
    num_img_files = len(glob(os.path.join(in_dir, f"ZED_{cam_ids[0]}_left_*.png")))
    pc_processor = RealPCProcessor(model_path, calib_dir, prompt)
    all_pcs, all_colors = [], []
    for i in range(min(num_img_files, num_frames)):
        cam_pcs, cam_colors = [], []
        for cam_id in cam_ids:
            paths = [
                os.path.join(in_dir, f"ZED_{cam_id}_left_{i:04d}.png"),
                os.path.join(in_dir, f"ZED_{cam_id}_right_{i:04d}.png"),
            ]
            images = [
                cv2.imread(os.path.join(in_dir, path))[..., ::-1] for path in paths
            ]
            assert int(paths[0].split("_")[-3]) == cam_id
            masked, pcs, colors = pc_processor.color_filter_and_make_pc(
                images, cam_id, view="side" if cam_id == 21172477 else "top"
            )
            masked = np.concatenate(masked, axis=1)[..., ::-1]
            cv2.imwrite(
                os.path.join(os.getcwd(), "masked_{cam_id}_{i}.jpg"), masked.copy()
            )
            cam_pcs.append(pcs)
            cam_colors.append(colors)
        pcs = np.concatenate(cam_pcs)
        colors = np.concatenate(cam_colors)
        out_fn = f"ZED_{cam_id_str}_{i:04d}.ply"
        tri_inference.write_pointcloud(
            pcs, colors[..., ::-1], os.path.join(output_dir, out_fn)
        )
        all_pcs.append(pcs)
        all_colors.append(colors)
    animate_pcs(
        all_pcs,
        all_colors,
        save_path=os.path.join(output_dir, f"episode_{cam_id_str}.html"),
    )
    animate_pcs_mpl(
        all_pcs,
        all_colors,
        save_path=os.path.join(output_dir, f"episode_{cam_id_str}.gif"),
    )


if __name__ == "__main__":
    main()
