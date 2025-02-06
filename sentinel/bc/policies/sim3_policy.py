import torch
from torch import nn
import torch.nn.functional as F

from sentinel.point_feat.core.lib_vec.sim3_encoder import SIM3Vec4Latent
from sentinel.point_feat.core.lib_vec.vec_layers import VecLinNormAct as VecLNA
from sentinel.point_feat.core.lib_vec.vec_layers import VecLinear


class SIM3Policy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.device = device
        self.feat_net = SIM3Vec4Latent(**cfg.model.encoder)

        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.encoder_out_dim = cfg.model.encoder.c_dim
        self.hidden_dim = cfg.model.encoder.c_dim
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.eef_dim = cfg.env.eef_dim

        # Notations:
        # - B = batch size
        # - E = number of end-effectors
        # - M = num points in the point cloud

        # FC layer outputs (1) offset vector; (2) gripper action

        # use per-point feature to predict offset vectors and heatmap values
        # input: (B, C + E, 3, M); output: (B, 1 or 2, 3, M)
        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        vnla_cfg = dict(mode="so3", act_func=act_func, return_tuple=False)

        num_vector_input_dims = self.encoder_out_dim + self.num_eef * (
            self.eef_dim // 3
        )
        num_scalar_input_dims = self.num_eef if self.dof >= 4 else 0

        self.xyz_fc = nn.Sequential(
            VecLNA(
                num_vector_input_dims,
                self.hidden_dim,
                s_in=num_scalar_input_dims,
                **vnla_cfg
            ),
            VecLNA(self.hidden_dim, self.hidden_dim, **vnla_cfg),
            VecLinear(self.hidden_dim, 1 if self.dof <= 4 else 2),
        )
        # use invariant feature to predict gripper open/close
        # input: (B, C + E, 3); output: (B, num_eef)
        if self.dof > 3:
            self.grip_fc = nn.Sequential(
                VecLNA(
                    num_vector_input_dims,
                    self.hidden_dim,
                    s_in=num_scalar_input_dims,
                    **vnla_cfg
                ),
                VecLNA(self.hidden_dim, self.hidden_dim, **vnla_cfg),
                VecLinear(self.hidden_dim, 0, s_out=self.num_eef),
            )

    def _convert_state_to_vec(self, state):
        # state format for 3d and 4d actions: eef_pos
        # state format for 7d actions: eef_pos, eef_rot_x, eef_rot_z, gravity_dir, gripper_pose, [optional] goal_pos
        # input: (B, E * eef_dim)
        # output: (B, ?, 3) [need norm] + (B, ?, 3) [does not need norm] + maybe (B, E)
        if self.dof == 3:
            return state.view(state.shape[0], -1, 3), None, None
        elif self.dof == 4:
            state = state.view(state.shape[0], self.num_eef, -1)
            assert state.shape[-1] in [4, 7]
            eef_pos = state[:, :, :3]
            scalar_state = state[:, :, 3]
            if state.shape[-1] == 7:
                goal_pos = state[:, :, -3:]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=1)
            else:
                vec_state_pos = eef_pos
            return vec_state_pos, None, scalar_state
        else:
            state = state.view(state.shape[0], self.num_eef, -1)
            assert state.shape[-1] in [13, 16]
            eef_pos = state[:, :, :3]
            dir1 = state[:, :, 3:6]
            dir2 = state[:, :, 6:9]
            gravity_dir = state[:, :, 9:12]
            gripper_pose = state[:, :, 12]

            if state.shape[-1] > 13:
                goal_pos = state[:, :, 13:16]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=1)
            else:
                vec_state_pos = eef_pos
            vec_state_dir = torch.cat([dir1, dir2, gravity_dir], dim=1)
            scalar_state = gripper_pose
            return vec_state_pos, vec_state_dir, scalar_state

    def forward(self, obs, predict_action=True):
        pc = obs["pc"]
        z_pos, z_dir, z_scalar = obs["z_pos"], obs["z_dir"], obs["z_scalar"]

        # get encoder features
        with torch.no_grad():
            feat_dict = self.feat_net(
                pc[:, None], target_norm=self.pc_scale, ret_perpoint_feat=True
            )
        point_feat = feat_dict["per_point_so3"]  # (B, C, 3, M)
        B, _, _, M = point_feat.shape

        # get low-level state features
        # z_pos (B, ?, 3) [need norm] + z_dir (B, ?, 3) [does not need norm] + z_scalar (B, E)
        z_pos = (z_pos - feat_dict["center"]) / feat_dict["scale"][
            :, None, None
        ]  # (B, ?, 3)
        if self.dof == 7:
            z_state = torch.cat([z_pos, z_dir], dim=1)
        else:
            z_state = z_pos
        z = torch.cat([point_feat, z_state[..., None].repeat(1, 1, 1, M)], dim=1)

        if self.dof > 3:
            gripper_ac = F.sigmoid(
                self.grip_fc((torch.cat([feat_dict["so3"], z_state], dim=1), z_scalar))[
                    1
                ]
            )

        offsets, _ = self.xyz_fc(
            (z, z_scalar[..., None].repeat(1, 1, M) if z_scalar is not None else None)
        )  # (B, 1 or 2, 3, M)
        offsets = offsets.reshape(B, -1, M).permute(0, 2, 1)  # [B, M, 3 or 6]
        if self.ac_mode == "abs":
            center = feat_dict["center"]
        else:
            center = 0
        offsets[:, :, :3] = (
            offsets[:, :, :3] * feat_dict["scale"][:, None, None] + center
        )

        if predict_action:
            ac = []
            for i in range(self.num_eef):
                if self.dof > 3:
                    ac.append(gripper_ac[:, [i]])  # gripper action
                dists = torch.linalg.norm(z_pos[:, [i]] - pc, dim=-1)
                closest_idx = torch.argmin(dists, dim=1)
                selected_offsets = offsets[torch.arange(B), closest_idx]  # (B, 3 or 6)
                ac.append(selected_offsets)  # target vel

            ac = torch.cat(ac, dim=1)

        ret = {
            "offsets": offsets,  # (B, M, 3 or 6)
            "global_feat": feat_dict["so3"],
            "local_feat": point_feat,
            "min_scale": torch.min(feat_dict["scale"]),
            "mean_scale": torch.mean(feat_dict["scale"]),
            "max_scale": torch.max(feat_dict["scale"]),
        }
        if self.dof > 3:
            ret["gripper_ac"] = gripper_ac  # (B, E)
        if predict_action:
            ret["ac"] = ac  # (B, E * dof)
        return ret
