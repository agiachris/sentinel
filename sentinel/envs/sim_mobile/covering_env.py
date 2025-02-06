import os
import pybullet
import numpy as np
import trimesh

from sentinel.envs.sim_mobile.base_env import BaseEnv
from sentinel.envs.sim_mobile.utils.init_utils import scale_mesh, rotate_around_z


DEBUG = 1
dbprint = print if DEBUG == 1 else lambda *args: ()


class CoveringEnv(BaseEnv):
    @property
    def name(self):
        return "covering"

    @property
    def robot_config(self):
        # Randomize robot end-effector positions.
        if getattr(self.args, "randomize_robot_eef_pos", False):
            offset = self.args.robot_eef_max_offset
            x_delta, y_delta = self.rng.rand(2) * 2 * offset - offset
        else:
            x_delta, y_delta = 0.0, 0.0
        self._robot_eef_x_delta = x_delta
        self._robot_eef_y_delta = y_delta

        if (
            not hasattr(self.args, "init_pose_mode")
        ) or self.args.init_pose_mode == "sim":
            init_base_pos = np.array([[-0.2, -0.1, 0.005], [+0.2, -0.1, 0.005]])
        else:
            init_base_pos = np.array([[-0.2, 0.1, 0.005], [+0.2, 0.1, 0.005]])
        init_base_pos[:, :2] *= self._soft_object_scale[None]
        init_base_pos[0, 0] -= 0.7 / np.sqrt(2)
        init_base_pos[1, 0] += 0.7 / np.sqrt(2)
        if (
            not hasattr(self.args, "init_pose_mode")
        ) or self.args.init_pose_mode == "sim":
            init_base_pos[0, 1] -= 0.7 / np.sqrt(2)
            init_base_pos[1, 1] -= 0.7 / np.sqrt(2)
        else:
            init_base_pos[0, 1] += 0.7 / np.sqrt(2)
            init_base_pos[1, 1] += 0.7 / np.sqrt(2)
        init_base_pos = rotate_around_z(init_base_pos, self._object_rotation[-1])
        init_base_pos[:, :2] += self.scene_offset[None]
        init_base_pos[:, :2] += np.array([x_delta, y_delta])
        if (
            not hasattr(self.args, "init_pose_mode")
        ) or self.args.init_pose_mode == "sim":
            init_base_rot = [
                self._object_rotation[-1] + np.pi * 1.25,
                self._object_rotation[-1] - np.pi * 0.25,
            ]
        else:
            init_base_rot = [
                self._object_rotation[-1] + np.pi * 0.75,
                self._object_rotation[-1] + np.pi * 0.25,
            ]

        rest_arm_pos = np.array([0.7, 0.0, 0.01 - self.ARM_MOUNTING_HEIGHT])
        rest_arm_rot = np.array([np.pi * 0.6, 0.0, np.pi / 2])
        robots = [
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[0, 0], init_base_pos[0, 1], init_base_rot[0]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
            {
                "sim_robot_name": "kinova",
                "rest_base_pose": np.array(
                    [init_base_pos[1, 0], init_base_pos[1, 1], init_base_rot[1]]
                ),
                "rest_arm_pos": rest_arm_pos,
                "rest_arm_rot": rest_arm_rot,
            },
        ]
        if (hasattr(self.args, "init_pose_mode")) and self.args.init_pose_mode != "sim":
            robots = robots[::-1]
        return robots

    def _randomize_object_scales(self):
        # override object scale variable so default size is equal to source size
        super()._randomize_object_scales()
        if self.scale_mode == "real_src":
            self._rigid_object_scale *= np.array([1.0, 0.7, 0.5]) * 1
            self._soft_object_scale *= np.array([0.6875, 0.6875]) * 1
        elif self.scale_mode == "real_target":
            self._rigid_object_scale *= np.array([5.45, 5.1, 2.6])
            self._soft_object_scale *= np.array([3.275, 2.975])
        else:
            assert self.scale_mode == "sim"

    @property
    def default_camera_config(self):
        cfg = super().default_camera_config
        cfg["pitch"] = -75
        cfg["distance"] = np.max(self._soft_object_scale) * 2
        cfg["target"] = [self.scene_offset[0], self.scene_offset[1], 0.1]
        return cfg

    @property
    def anchor_config(self):
        self._corner_positions = np.array(
            [[0.0, 0.0, 0.02], [0.0, 0.3, 0.02], [0.3, 0.0, 0.02]]
            # [[-1.5, -1.2, 0], [-1.5, 1.2, 0], [1.7, -1.2, 0], [1.7, 1.2, 0]]
        )
        anchors = [
            {
                "pos": self._corner_positions[i],
                "radius": 0.02,
                "rgba": (0.25, 0.52, 0.95, 1.0),
            }
            for i in range(3)
        ]
        return []
        # return anchors if DEBUG else []

    @property
    def rigid_objects(self):
        self._box_size = 0.05
        obj_path = os.path.join(self.data_path, "covering/box.obj")
        obj_path = scale_mesh(obj_path, self._rigid_object_scale)
        obj_path = "/".join(obj_path.split("/")[-2:])

        # Randomize rigid object position.
        if getattr(self.args, "randomize_rigid_pos", False):
            x_low, x_high = self.args.rigid_pos_x_low, self.args.rigid_pos_x_high
            x_delta = self.rng.rand() * (x_high - x_low) + x_low
            y_low, y_high = self.args.rigid_pos_y_low, self.args.rigid_pos_y_high
            y_delta = self.rng.rand() * (y_high - y_low) + y_low
        else:
            x_delta = 0.0
            y_delta = 0.0
        self._rigid_pos_x_delta = x_delta
        self._rigid_pos_y_delta = y_delta

        pos = [
            self.scene_offset[0] + x_delta,
            self.scene_offset[1] + y_delta,
            self._box_size * self._rigid_object_scale[-1],
        ]

        return [
            {
                "path": obj_path,
                "scale": 1.0,
                "pos": pos,
                "orn": self._object_rotation,
            }
        ]

    @property
    def soft_objects(self):
        # Default soft dynamics.
        deform_mass = 0.2
        self.args.deform_bending_stiffness = float(0.01)
        self.args.deform_damping_stiffness = float(1.0)
        self.args.deform_elastic_stiffness = float(100.0)
        self.args.deform_friction_coeff = float(1.0)

        # Randomize soft dynamics.
        if getattr(self.args, "randomize_soft_dynamics", False):
            delta = self.rng.rand(5) * 2 - 1
            deform_mass += self.args.deform_mass_percent * deform_mass * delta[0]
            self.args.deform_bending_stiffness += float(
                self.args.deform_bending_stiffness_percent
                * self.args.deform_bending_stiffness
                * delta[1]
            )
            self.args.deform_damping_stiffness += float(
                self.args.deform_damping_stiffness_percent
                * self.args.deform_damping_stiffness
                * delta[2]
            )
            self.args.deform_elastic_stiffness += float(
                self.args.deform_elastic_stiffness_percent
                * self.args.deform_elastic_stiffness
                * delta[3]
            )
            self.args.deform_friction_coeff += float(
                self.args.deform_friction_coeff_percent
                * self.args.deform_friction_coeff
                + delta[4]
            )

        obj_path = os.path.join(self.data_path, "folding/towel.obj")
        obj_path = scale_mesh(
            obj_path,
            np.array([self._soft_object_scale[0], self._soft_object_scale[1], 1.0]),
        )
        obj_path = "/".join(obj_path.split("/")[-2:])

        ang = self._object_rotation[-1]
        if (
            not hasattr(self.args, "init_pose_mode")
        ) or self.args.init_pose_mode == "sim":
            init_y = (
                -0.3 * self._soft_object_scale[1] * np.cos(ang) + self.scene_offset[1]
            )
        else:
            init_y = (
                0.3 * self._soft_object_scale[1] * np.cos(ang) + self.scene_offset[1]
            )
        return [
            {
                "path": obj_path,
                "scale": 2.0,
                "pos": [
                    0.3 * self._soft_object_scale[1] * np.sin(ang)
                    + self.scene_offset[0],
                    init_y,
                    0.001,
                ],
                "orn": self._object_rotation,
                "mass": deform_mass,
                "collision_margin": 0.001,
            }
        ]

    def compute_reward(self):
        obj_id = self.soft_ids[0]
        cloth_mesh_xyzs = np.array(self.sim.getMeshData(obj_id)[1])
        cloth_vol = trimesh.convex.convex_hull(cloth_mesh_xyzs)
        rigid_mesh_xyzs = self._get_rigid_body_mesh(self.rigid_ids[0])
        rigid_vol = trimesh.convex.convex_hull(rigid_mesh_xyzs)
        rigid_volume = rigid_vol.volume
        intersect_volume = rigid_vol.intersection(cloth_vol).volume
        return intersect_volume / rigid_volume

    def _reset_sim(self):
        super()._reset_sim()

        # make floor friction higher
        pybullet.changeDynamics(self.rigid_ids[0], -1, lateralFriction=0.3)

        # preload box mesh vertices
        # this is required because pybullet by default doesn't save all vertices
        # of the object
        obj_path = os.path.join(self.data_path, "covering/box.obj")
        mesh = trimesh.load(obj_path).vertices * self._rigid_object_scale
        self._box_vertices = mesh

        if DEBUG:
            # texture_id = self.sim.loadTexture("textures/dark_carpet.jpg")
            texture_id = self.sim.loadTexture("textures/dark_gray_fabric.jpg")
        else:
            texture_id = self.sim.loadTexture("textures/comforter.png")

        self.sim.changeVisualShape(
            self.soft_ids[0], -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id
        )

        if (
            not hasattr(self.args, "init_pose_mode")
        ) or self.args.init_pose_mode != "sim":
            self._move_grippers([1, 1])

    def _get_rigid_body_mesh(self, obj_id):
        assert obj_id == self.rigid_ids[0]
        mesh = self._box_vertices.copy()
        mesh = rotate_around_z(mesh, self._object_rotation[-1])
        obj_pos, _ = self.sim.getBasePositionAndOrientation(obj_id)
        mesh += np.array(obj_pos)[None]
        return mesh
