"""
The facing up localization camera

Author: @ziangcao, @jingyuny
"""

import os
import cv2
import json
import socket
import numpy as np
from typing import Tuple

from sentinel.envs.real_mobile.utils.perception.logitech.utils import get_video_cap
from sentinel.envs.real_mobile.utils.perception.logitech_v2.get_cam_serial_number import (
    get_serial_number,
)

CAM_MOUNT_ON_FRONT = False

VEC_CAM_TO_BOT = {
    # NOTE: the offset convention follows the camera!!!
    # The x axis aligns with the line from robot center to the camera
    # The z axis is pointing upward
    # The y axis is determinated by the right-hand rule
    # theta is the along z axis rotation from the x axis (if mounting on the left side of robot, theta = np.pi/2)
    "iprl-bot1": (
        np.array([0.29, -0.02, 0])
        if CAM_MOUNT_ON_FRONT
        else np.array([0.26, 0, np.pi / 2])
    ),  # B66E281E + Temporary value -- Fake for now
    "iprl-bot2": (
        np.array([0.29, -0.02, 0])
        if CAM_MOUNT_ON_FRONT
        else np.array([0.26, 0, np.pi / 2])
    ),  # DDCD281E + Temporary value -- Fake for now
    "iprl-bot3": (
        np.array([0.27, 0.01, 0])
        if CAM_MOUNT_ON_FRONT
        else np.array([0.27, 0, np.pi / 2])
    ),  #
}


class RobotPoseEstimatorClean(object):
    def __init__(
        self,
        cam_serial: str,
        cam_matrix_path: str,
        marker_positions_path: str,
        cam_height=0.08,
        cam_id=0,
    ):
        self.cam_height = cam_height
        self.cam_serial = cam_serial  # Camera serial number for debugging

        # Initialization logic remains the same up  to video capture initialization
        # Load camera intrinsic parameters
        fs = cv2.FileStorage(cam_matrix_path, cv2.FILE_STORAGE_READ)
        self.image_width = int(fs.getNode("image_width").real())
        self.image_height = int(fs.getNode("image_height").real())
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("distortion_coefficients").mat()

        # Load marker positions
        with open(marker_positions_path, "r") as file:
            self.marker_positions = json.load(file)

        # Precompute marker corner positions in the world coordinates
        self.marker_length = 0.18  # Marker side length in meters
        self.marker_corners = self._get_marker_corners()

        # Initialize video capture conditionally
        self.cap = get_video_cap(
            cam_serial, self.image_width, self.image_height, new_camera=True
        )

        try:
            self.cam2bot = VEC_CAM_TO_BOT[socket.gethostname()]
        except KeyError:
            print(
                f"Camera to robot vector not found for {socket.gethostname()}. Using default value."
            )
            self.cam2bot = VEC_CAM_TO_BOT["iprl-bot3"]

    def _get_marker_corners(self):
        # Assuming square markers and all markers are placed with the same rotation on the ceiling
        half_length = self.marker_length / 2
        corners = {}
        for marker_id, pos in self.marker_positions.items():
            corners[marker_id] = np.array(
                [
                    [pos[0] + half_length, pos[1] + half_length, pos[2]],  # Top-left
                    [pos[0] - half_length, pos[1] + half_length, pos[2]],  # Top-right
                    [
                        pos[0] - half_length,
                        pos[1] - half_length,
                        pos[2],
                    ],  # Bottom-right
                    [pos[0] + half_length, pos[1] - half_length, pos[2]],  # Bottom-left
                ]
            )
        return corners

    def readPose_from_uv_map(self, observed_uv, world_xy, H, W):
        # [1] Establish the UV map from the observed_uv to the world_xy
        # Assuming a planar surface, we use findHomography to establish a mapping.
        # This function finds a perspective transformation between two planes.
        H_mat, _ = cv2.findHomography(observed_uv, world_xy)

        # [2] Find the center of the image plane in the UV map
        center_uv = np.array([W / 2, H / 2], dtype="float32").reshape(
            -1, 2
        )  # Reshape to (1, 2) for a single point

        forward_uv = np.array([W / 2, 0.0], dtype="float32").reshape(
            -1, 2
        )  # Reshape to (1, 2) for a single point

        # [3] Map the center of the image plane to the world coordinates
        # Apply the homography to the image center, ensuring it has the correct shape (1, N, 2)
        center_world = cv2.perspectiveTransform(center_uv[None, :, :], H_mat)
        center_world = center_world[0][0]  # Extract the transformed point

        forward_world = cv2.perspectiveTransform(forward_uv[None, :, :], H_mat)
        forward_world = forward_world[0][0]  # Extract the transformed point

        # [4] Calculate the heading vector (pointing to the up direction of camera plane!)
        # In currect mounting, it's pointing toward the center of the robot
        heading_vec = forward_world - center_world
        heading_vec = heading_vec / np.linalg.norm(
            heading_vec
        )  # Normalize the heading vector

        # Assuming the camera's height (Z coordinate) is known and constant, represented by self.cam_height
        # The function now returns the XY coordinates from the world space and the camera's height as Z
        return (
            np.array([center_world[0], center_world[1], self.cam_height]),
            heading_vec,
        )

    def get_camera_pose_from_uv_map(self, frame=None, with_center=True) -> tuple:
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        H = self.camera_matrix[1, 2] * 2.0
        W = self.camera_matrix[0, 2] * 2.0

        # Detect markers in the image
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

        if ids is not None:
            observed_corners = []
            world_corners = []
            for i, marker_id in enumerate(ids.flatten()):
                if str(marker_id) in self.marker_positions:

                    detected_corners = corners[i].reshape(-1, 2)  # (1, 4, 2) -> (4, 2)
                    ref_corners = self.marker_corners[str(marker_id)].reshape(
                        -1, 3
                    )  # (4, 3) -> (4, 3)

                    observed_corners.extend(detected_corners)
                    world_corners.extend(ref_corners)

                    if with_center:
                        detected_center = np.mean(
                            detected_corners, axis=0
                        )  # (4, 2) -> (2,)
                        ref_center = np.mean(ref_corners, axis=0)  # (4, 3) -> (3,)
                        observed_corners.append(detected_center)
                        world_corners.append(ref_center)

            # If we have corresponding corners, proceed with pose estimation
            if observed_corners and world_corners:
                # Initialize arrays for 2D pose estimation
                observed_uv = np.array(observed_corners)[
                    :, :2
                ]  # Take only the XY coordinates
                world_xy = np.array(world_corners)[
                    :, :2
                ]  # Take only the XY coordinates

                # Theoretically, construct a uv_map from the observed_uv to the world_xy
                # Then, directly find the center of image in uv space, and read out the pose from the uv_map
                tvec, heading_vec = self.readPose_from_uv_map(
                    observed_uv, world_xy, H, W
                )  # Placeholder for actual implementation
                rvec = np.zeros((3, 1))
                success = True

                if success:
                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    return (R, tvec), heading_vec
                else:
                    print("Pose estimation failed")
                    return None
            else:
                print("No matching markers found")
                return None
        else:
            print("No markers detected")
            return None

    def get_robot_pose(
        self, camera_pose: Tuple[np.ndarray, np.ndarray], heading_vec: np.ndarray
    ) -> Tuple[float, float, float]:
        if camera_pose is None:
            print("Camera pose is not available.")
            return None
        R, tvec = camera_pose

        cam_position = tvec.flatten()

        # The heading vector is pointing to the center of the robot
        inner_dir = heading_vec
        tangent_dir = np.array([inner_dir[1], -inner_dir[0]])
        robot_position = (
            cam_position[:2]
            + self.cam2bot[0] * inner_dir
            + self.cam2bot[1] * tangent_dir
        )

        # Compute the robot's heading angle based on the heading vector
        heading_angle = np.arctan2(heading_vec[1], heading_vec[0]) - self.cam2bot[2]

        # Clip the angle to the range [-pi, pi]
        if heading_angle < -np.pi:
            heading_angle += np.pi * 2
        elif heading_angle > np.pi:
            heading_angle -= np.pi * 2

        # Return the robot's pose: [x, y, heading]
        return robot_position[0], robot_position[1], heading_angle

    def get_xy_heading(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        camera_pose, heading_vec = self.get_camera_pose_from_uv_map(frame)
        robot_pose = self.get_robot_pose(camera_pose, heading_vec)
        return robot_pose

    def __del__(self):
        if hasattr(self, "cap"):
            if self.cap is not None:
                self.cap.release()


def init_pose_estimator():
    cam_serial = get_serial_number()
    curr_dir = os.path.dirname(__file__)
    cam_config_path = os.path.join(curr_dir, "configs", f"{cam_serial}.yml")
    marker_pos_path = os.path.join(curr_dir, "configs", "marker_positions.json")
    pose_estimator = RobotPoseEstimatorClean(
        cam_serial, cam_config_path, marker_pos_path
    )
    return pose_estimator
