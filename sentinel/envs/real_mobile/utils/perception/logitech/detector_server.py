# Author: Jimmy Wu
# Date: September 2022

import argparse
import math
import time
from multiprocessing import Process
import cv2 as cv
import numpy as np

import sentinel.envs.real_mobile.utils.perception.logitech.constants as constants
import sentinel.envs.real_mobile.utils.perception.logitech.utils as utils
from sentinel.envs.real_mobile.utils.perception.logitech.camera_client import (
    CameraClient,
)
from sentinel.envs.real_mobile.utils.perception.logitech.camera_server import (
    CameraServer,
)
from sentinel.envs.real_mobile.utils.perception.logitech.server import Server


def get_angle_offsets():
    corners = [(0, 1), (1, 1), (1, 0), (0, 0)]
    offsets = {}
    for i, corner1 in enumerate(corners):
        for j, corner2 in enumerate(corners):
            if i != j:
                offsets[(i, j)] = -math.atan2(
                    corner2[1] - corner1[1], corner2[0] - corner1[0]
                )
    return offsets


class Detector:
    def __init__(self, placement, serial, port, scale_factor=0.2):
        assert placement in {"top", "bottom", "middle"}
        self.placement = placement

        # Camera
        self.camera_center, self.camera_corners = utils.get_camera_alignment_params(
            serial
        )
        self.camera_client = CameraClient(port)

        # Aruco marker detection
        cv.setNumThreads(4)  # Based on 12 CPUs
        self.marker_dict = cv.aruco.getPredefinedDictionary(constants.MARKER_DICT_ID)
        self.marker_dict.bytesList = self.marker_dict.bytesList[constants.MARKER_IDS]
        self.detector_params = cv.aruco.DetectorParameters()
        self.detector_params.minCornerDistanceRate = 0.2  # Reduce false positives
        self.marker_detector = cv.aruco.ArucoDetector(
            self.marker_dict, self.detector_params
        )

        # Pose transformation
        cam_corners_f32 = np.array(self.camera_corners, dtype=np.float32)
        self.pose_transform_mat = self.compute_pose_transform_matrix(cam_corners_f32)
        self.height_ratio = (
            constants.CAMERA_HEIGHT - constants.ROBOT_HEIGHT
        ) / constants.CAMERA_HEIGHT
        self.angle_offsets = get_angle_offsets()
        self.position_offset = constants.ROBOT_DIAG / 2 - constants.MARKER_PARAMS[
            "sticker_length"
        ] / math.sqrt(2)

        # Image transformation
        self.pixels_per_m = constants.PIXELS_PER_M
        self.floor_width = constants.FLOOR_WIDTH
        self.floor_length = constants.FLOOR_LENGTH
        self.image_width = round(scale_factor * self.pixels_per_m * self.floor_width)
        self.image_height = round(
            scale_factor * self.pixels_per_m * self.floor_length / 2
        )
        self.img_transform_mat = self.compute_img_transform_matrix(
            cam_corners_f32, self.image_width, self.image_height
        )

    @property
    def image_size(self):
        return np.array([self.image_height, self.image_width])

    def compute_pose_transform_matrix(self, src_points):
        if self.placement == "top":
            # Top left, top right, bottom right, bottom left
            dst_points = np.array(
                [
                    [-(constants.FLOOR_WIDTH / 2), constants.FLOOR_LENGTH / 2],
                    [constants.FLOOR_WIDTH / 2, constants.FLOOR_LENGTH / 2],
                    [constants.FLOOR_WIDTH / 2, 0],
                    [-(constants.FLOOR_WIDTH / 2), 0],
                ],
                dtype=np.float32,
            )
        elif self.placement == "bottom":
            dst_points = np.array(
                [
                    [-(constants.FLOOR_WIDTH / 2), 0],
                    [constants.FLOOR_WIDTH / 2, 0],
                    [constants.FLOOR_WIDTH / 2, -(constants.FLOOR_LENGTH / 2)],
                    [-(constants.FLOOR_WIDTH / 2), -(constants.FLOOR_LENGTH / 2)],
                ],
                dtype=np.float32,
            )
        elif self.placement == "middle":
            dst_points = np.array(
                [
                    [-(constants.FLOOR_WIDTH / 2), constants.FLOOR_LENGTH / 4],
                    [constants.FLOOR_WIDTH / 2, constants.FLOOR_LENGTH / 4],
                    [constants.FLOOR_WIDTH / 2, -(constants.FLOOR_LENGTH / 4)],
                    [-(constants.FLOOR_WIDTH / 2), -(constants.FLOOR_LENGTH / 4)],
                ],
                dtype=np.float32,
            )

        return cv.getPerspectiveTransform(src_points, dst_points).astype(np.float32)

    def compute_img_transform_matrix(self, src_points, image_width, image_height):
        dst_points = np.array(
            [
                [0, 0],
                [image_width, 0],
                [image_width, image_height],
                [0, image_height],
            ],
            dtype=np.float32,
        )
        return cv.getPerspectiveTransform(src_points, dst_points)

    def get_poses_from_markers(self, corners, indices, debug=False):
        data = {"poses": {}, "single_marker_robots": set()}
        if indices is None:
            return data

        # Convert marker corners from pixel coordinates to real-world coordinates
        corners = np.concatenate(corners, axis=1).squeeze(0)
        camera_center = np.array(self.camera_center, dtype=np.float32)
        corners = camera_center + self.height_ratio * (corners - camera_center)
        corners = np.c_[corners, np.ones(corners.shape[0], dtype=np.float32)]
        corners = corners @ self.pose_transform_mat.T
        corners = (corners[:, :2] / corners[:, 2:]).reshape(-1, 4, 2)

        # Compute marker positions
        centers = corners.mean(axis=1)

        # Compute marker headings, making sure to deal with wraparound
        diffs = (corners - centers.reshape(-1, 1, 2)).reshape(-1, 2)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0]).reshape(-1, 4) + np.radians(
            [-135, -45, 45, 135], dtype=np.float32
        )
        angles1 = np.mod(angles + math.pi, 2 * math.pi) - math.pi
        angles2 = np.mod(angles, 2 * math.pi)
        headings = np.where(
            angles1.std(axis=1) < angles2.std(axis=1),
            angles1.mean(axis=1),
            np.mod(angles2.mean(axis=1) + math.pi, 2 * math.pi) - math.pi,
        )

        # Compute robot poses using marker centers
        positions = centers.copy()
        indices = indices.squeeze(1)
        robot_indices = np.floor_divide(indices, 4)
        for robot_idx in np.unique(robot_indices):
            robot_idx = robot_idx.item()
            robot_mask = robot_indices == robot_idx
            indices_robot = np.mod(indices[robot_mask], 4)
            centers_robot = centers[robot_mask]
            positions_robot = centers_robot.copy()

            # Compute robot heading
            single_marker = robot_mask.sum() == 1
            if single_marker:
                # Use heading of the single visible marker
                heading = headings[robot_mask].item()
            else:
                # Compute heading using pairs of marker centers
                headings_robot = []
                for i, idx1 in enumerate(indices_robot):
                    for j, idx2 in enumerate(indices_robot):
                        if j <= i:
                            continue
                        if idx1 == idx2:  # Caused by false positives
                            continue
                        dx = centers_robot[j][0] - centers_robot[i][0]
                        dy = centers_robot[j][1] - centers_robot[i][1]
                        heading = math.atan2(dy, dx) + self.angle_offsets[(idx1, idx2)]
                        # heading = (heading + math.pi) % (2 * math.pi) - math.pi
                        headings_robot.append(heading)
                if len(headings_robot) == 0:  # Caused by false positives
                    continue
                headings_robot = np.array(headings_robot)
                headings1 = np.mod(headings_robot + math.pi, 2 * math.pi) - math.pi
                headings2 = np.mod(headings_robot, 2 * math.pi)
                if headings1.std() < headings2.std():
                    heading = headings1.mean()
                else:
                    heading = (headings2.mean() + math.pi) % (2 * math.pi) - math.pi
                # heading = np.array(headings_robot, dtype=np.float32).mean()

            # Compute robot position using marker position offsets
            angles = (
                heading
                + np.radians([-45, -135, 135, 45], dtype=np.float32)[indices_robot]
            )
            positions_robot[:, 0] += self.position_offset * np.cos(angles)
            positions_robot[:, 1] += self.position_offset * np.sin(angles)
            position = positions_robot.mean(axis=0)
            positions[robot_mask] = positions_robot

            # Store robot pose
            data["poses"][robot_idx] = (position[0], position[1], heading)
            if single_marker:
                data["single_marker_robots"].add(robot_idx)

        if debug:
            data["debug_data"] = list(
                zip(indices.tolist(), centers.tolist(), positions.tolist())
            )

        return data

    def get_poses(self, debug=False):
        image = self.camera_client.get_image()

        # Detect markers
        corners, indices, _ = self.marker_detector.detectMarkers(image)

        if debug:
            image_copy = image.copy()  # 0.2 ms
            if indices is not None:
                cv.drawDetectedMarkers(image_copy, corners, indices)
            cv.imshow(f"Detections ({self.placement})", image_copy)  # 0.3 ms

        return self.get_poses_from_markers(corners, indices, debug=debug)

    def get_image(self):
        image = self.camera_client.get_image()
        img_size = (self.image_width, self.image_height)
        image = cv.warpPerspective(image, self.img_transform_mat, img_size)
        return image

    def pixel2pos(self, pixel):
        pos = self.pose_transform_mat @ np.r_[pixel, [1]].reshape(3, 1)
        pos = pos.flatten()
        pos = np.array([pos[0] / pos[2], pos[1] / pos[2]])
        return pos

    def pos2pixel(self, pos):
        pos_homo = np.r_[pos[:2], [1]].reshape(3, 1)
        mat = self.img_transform_mat @ np.linalg.inv(self.pose_transform_mat)
        pixel = mat @ pos_homo
        pixel = np.array([pixel[0] / pixel[2], pixel[1] / pixel[2]]).round()
        return pixel.flatten()


class DetectorServer(Server):
    def __init__(
        self, top_only=False, debug=False, port_offset=0, scale_factor=0.2, **kwargs
    ):
        self._start_port = kwargs["start_port"]
        kwargs["start_port"] += 5 + port_offset
        print(f'[detector server] starting at port {kwargs["start_port"]}')
        super().__init__(**kwargs)
        self.debug = debug
        self.scale_factor = scale_factor
        self.pixels_per_m = constants.PIXELS_PER_M
        self.floor_width = constants.FLOOR_WIDTH
        self.floor_length = constants.FLOOR_LENGTH
        if top_only:
            self.detectors = [
                Detector(
                    "middle", "E4298F4E", self._start_port + port_offset, scale_factor
                )
            ]
        else:
            self.detectors = [
                Detector(
                    "top", "E4298F4E", self._start_port + port_offset, scale_factor
                ),
                Detector(
                    "bottom",
                    "099A11EE",
                    self._start_port + port_offset + 2,
                    scale_factor,
                ),
            ]

    def get_image(self):
        images = [d.get_image() for d in self.detectors]
        image = np.concatenate(images)
        return image

    def get_base_poses(self):
        data = {"poses": {}}
        if self.debug:
            data["debug_data"] = []
        for detector in self.detectors:
            new_data = detector.get_poses(debug=self.debug)
            for robot_idx, pose in new_data["poses"].items():
                if (
                    robot_idx in data["poses"]
                    and robot_idx in new_data["single_marker_robots"]
                ):
                    continue  # Single marker pose estimates are not as reliable
                data["poses"][robot_idx] = pose  # Bottom detector takes precedence
            if "debug_data" in new_data:
                data["debug_data"].extend(new_data["debug_data"])
        if self.debug:
            cv.waitKey(1)
        return data

    def get_data(self, req):
        request_type = req["type"]
        assert request_type in ["image", "pose", "pos2pixel", "pixel2pos"]
        if request_type == "image":
            data = {"image": self.get_image()}
        elif request_type == "pose":
            data = self.get_base_poses()
        elif request_type == "pos2pixel":
            res = None
            position = req["pos"]
            scaled_pixels_per_m = self.pixels_per_m * self.scale_factor
            pix_x = int((position[0] + self.floor_length / 2) * scaled_pixels_per_m)
            pix_y = int((self.floor_width / 2 - position[1]) * scaled_pixels_per_m)
            data = {"pixel": np.array([pix_x, pix_y])}
        elif request_type == "pixel2pos":
            pixel = req["pixel"]
            x, y = pixel[0], pixel[1]
            scaled_pixels_per_m = self.pixels_per_m * self.scale_factor
            pos_x = x / scaled_pixels_per_m - self.floor_length / 2
            pos_y = self.floor_width / 2 - y / scaled_pixels_per_m
            data = {"pos": np.array([pos_x, pos_y])}
        return data

    def clean_up(self):
        for detector in self.detectors:
            detector.camera_client.close()
        cv.destroyAllWindows()


def main(args):
    # Start camera servers
    start_port = args.start_port

    def start_camera_server(serial, port):
        CameraServer(serial, start_port=port, n_conns=args.cam_server_n_conn).run()

    if not args.no_cam_server:
        for serial, port in [
            ("E4298F4E", args.start_port + args.port_offset),
            ("099A11EE", args.start_port + args.port_offset + 2),
        ]:
            p = Process(target=start_camera_server, args=(serial, port))
            p.daemon = True
            p.start()
            if args.top_only:
                break

    # Wait for camera servers to be ready
    time.sleep(1.5)

    # Start detector server
    DetectorServer(
        top_only=args.top_only,
        debug=args.debug,
        start_port=args.start_port,
        port_offset=args.port_offset,
        n_conns=args.cam_server_n_conn,
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cam-server-n-conn", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=6000)
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--no-cam-server", action="store_true")
    main(parser.parse_args())
