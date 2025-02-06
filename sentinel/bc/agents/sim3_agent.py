import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import wandb

from sentinel.utils.norm import Normalizer
from sentinel.bc.policies.sim3_policy import SIM3Policy
from sentinel.bc.utils import to_torch
from sentinel.utils.vis_points import plot_points_grid
from sentinel.utils.diffusion.lr_scheduler import get_scheduler


def find_closest_point_indices(pcs, query_points):
    # Initialize an array to store the closest point indices for each item in the batch
    results = np.zeros((len(query_points),), dtype=int)
    for i, (pc, query_point) in enumerate(zip(pcs, query_points)):
        kdtree = KDTree(pc)
        _, index = kdtree.query(query_point, k=1)
        results[i] = index
    return results


def find_closest_points(point_cloud, end_effectors, num_closest_points=20):
    batch_size, num_points, _ = point_cloud.size()
    _, num_end_effectors, _ = end_effectors.size()

    # Calculate distances between each end-effector and all points
    end_effectors = end_effectors.view(batch_size * num_end_effectors, 1, 3)
    point_cloud = point_cloud.view(batch_size, 1, num_points, 3)

    distances = torch.norm(
        end_effectors - point_cloud, dim=3
    )  # (batch_size*num_end_effectors, 1, num_points)
    _, closest_indices = distances.topk(
        num_closest_points, dim=2, largest=False
    )  # (batch_size*num_end_effectors, 1, num_closest_points)

    # Gather the closest points
    closest_points = torch.gather(
        point_cloud.expand(-1, num_closest_points, -1, -1),
        dim=2,
        index=closest_indices.expand(-1, -1, -1, 3),
    )

    # Reshape to (batch_size, num_end_effectors, num_closest_points, 3)
    closest_points = closest_points.view(
        batch_size, num_end_effectors, num_closest_points, 3
    )

    return closest_points


def fit_plane(points):
    # Fit a plane to the points using Singular Value Decomposition (SVD)
    batch_size, num_end_effectors, num_closest_points, _ = points.size()
    points = points.view(-1, num_closest_points, 3)

    centroid = points.mean(dim=1, keepdim=True)
    centered_points = points - centroid
    covariance_matrix = torch.bmm(centered_points.transpose(1, 2), centered_points)
    _, _, V = torch.svd(covariance_matrix)

    normal = V[:, :, -1]  # The normal of the plane is the last column of V
    return normal


def mirror_point(point, normal, origin):
    # Calculate the reflection of the point with respect to the plane
    # point: (batch_size, num_end_effectors, 3)
    # normal: (batch_size, num_end_effectors, 3)
    # origin: (batch_size, num_end_effectors, 3)

    # Calculate the vector from the origin to the point
    v = point - origin

    # Calculate the projection of v onto the plane
    projection = v - torch.sum(v * normal, dim=2, keepdim=True) * normal

    # Calculate the mirrored point
    mirrored_point = point - 2 * projection

    return mirrored_point


class SIM3Agent(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.actor = SIM3Policy(cfg, device=cfg.device).to(cfg.device)
        if cfg.mode == "train":
            self.optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=cfg.training.lr
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=cfg.data.dataset.num_training_steps,
            )
        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.eef_dim = cfg.env.eef_dim
        self.num_points = cfg.data.dataset.num_points
        self.shuffle_pc = cfg.data.dataset.shuffle_pc
        self.demo_freq = cfg.env.args.freq

        self.w_o = cfg.training.offset_loss_weight
        self.w_g = cfg.training.gripper_loss_weight
        self.sigma = cfg.training.sigma

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

        self.pca = PCA(n_components=3)
        self._pca_initialized = False
        self._pca_range = None

        self.global_pca = PCA(n_components=3)
        self._global_pca_initialized = False
        self._global_pca_range = None

        self.pc_normalizer = None
        self.state_normalizer = None
        self.ac_normalizer = None

    def _init_normalizers(self, batch):
        if self.ac_normalizer is None:
            gt_action = batch["action"]
            flattened_gt_action = gt_action.view(-1, self.dof)
            if self.dof == 7:
                indices = [[0], [1, 2, 3], [4, 5, 6]]
            elif self.dof == 4:
                indices = [[0], [1, 2, 3]]
            else:
                indices = None
            ac_normalizer = Normalizer(
                flattened_gt_action, symmetric=True, indices=indices
            )
            self.ac_normalizer = Normalizer(
                {
                    "min": ac_normalizer.stats["min"].tile((self.num_eef,)),
                    "max": ac_normalizer.stats["max"].tile((self.num_eef,)),
                }
            )
            print(f"Action normalization stats: {self.ac_normalizer.stats}")
        if self.state_normalizer is None:
            # dof layout: maybe gripper open/close, xyz, maybe rot
            if self.dof == 3:
                self.state_normalizer = ac_normalizer
            else:
                self.state_normalizer = Normalizer(
                    {
                        "min": ac_normalizer.stats["min"][1:4],
                        "max": ac_normalizer.stats["max"][1:4],
                    }
                )
            self.actor.state_normalizer = self.state_normalizer
        if self.pc_normalizer is None:
            self.pc_normalizer = self.state_normalizer
            self.actor.pc_normalizer = self.pc_normalizer

        # compute action scale relative to point cloud scale
        pc = batch["pc"].reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        ac_scale = ac_normalizer.stats["max"].max()
        self.pc_scale = pc_scale / ac_scale
        self.actor.pc_scale = self.pc_scale
        print("=" * 20)
        print(f" => PC scale = {self.pc_scale}")
        print("=" * 20)

    def train(self, training=True):
        self.actor.train(training)

    def act(self, obs, return_dict=False):
        self.train(False)

        assert isinstance(obs["pc"][0], np.ndarray)
        if len(obs["state"].shape) == 2:
            assert len(obs["pc"].shape) == 2  # (N, 3)
            obs["pc"] = [obs["pc"]]
            for k in obs:
                if k != "pc" and isinstance(obs[k], np.ndarray):
                    obs[k] = obs[k][None]
            has_batch_dim = False
        elif len(obs["state"].shape) == 3:
            assert len(obs["pc"][0].shape) == 2  # (B, N, 3)
            has_batch_dim = True
        else:
            raise ValueError("Input format not recognized.")

        ac_dim = self.num_eef * self.dof
        batch_size = len(obs["pc"])

        state = obs["state"].reshape(
            tuple(obs["state"].shape[:1]) + (-1,)
        )  # (B, num_eef * eef_dim)

        # process the point clouds
        # some point clouds might be invalid
        # if this occurs, exclude these batch items
        xyzs = []
        ac = np.zeros([batch_size, ac_dim])
        if return_dict:
            ac_dict = []
            for i in range(batch_size):
                ac_dict.append(None)
        forward_idxs = list(np.arange(batch_size))
        for batch_idx, xyz in enumerate(obs["pc"]):
            if not batch_idx in forward_idxs:
                xyzs.append(np.zeros((self.num_points, 3)))
            elif xyz.shape[0] == 0:
                # no points in point cloud, return no-op action
                forward_idxs.remove(batch_idx)
                xyzs.append(np.zeros((self.num_points, 3)))
            elif self.shuffle_pc:
                choice = np.random.choice(xyz.shape[0], self.num_points, replace=True)
                xyz = xyz[choice, :]
                xyzs.append(xyz)
            else:
                step = xyz.shape[0] // self.num_points
                xyz = xyz[::step, :][: self.num_points]
                xyzs.append(xyz)

        if len(forward_idxs) > 0:
            torch_obs = dict(
                pc=torch.tensor(np.array(xyzs)[forward_idxs]).to(self.device).float(),
                state=torch.tensor(state[forward_idxs]).to(self.device).float(),
            )
            for k in obs:
                if not k in ["pc", "state"] and isinstance(obs[k], np.ndarray):
                    torch_obs[k] = (
                        torch.tensor(obs[k][forward_idxs]).to(self.device).float()
                    )

            # normalize obs
            torch_obs = self._normalize_obs(torch_obs)

            raw_ac_dict = self.actor(torch_obs)
        else:
            raw_ac_dict = torch.zeros((batch_size, ac_dim)).to(self.actor.device)
        for i, idx in enumerate(forward_idxs):
            if return_dict:
                ac_dict[idx] = {
                    k: v[i]
                    for k, v in raw_ac_dict.items()
                    if len(v.shape) > 0 and v.shape[0] == len(obs["pc"])
                }
            unnormed_action = (
                self.ac_normalizer.unnormalize(raw_ac_dict["ac"][i])
                .detach()
                .cpu()
                .numpy()
            )
            ac[idx] = unnormed_action

        if not has_batch_dim:
            ac = ac[0]
            if return_dict:
                ac_dict = ac_dict[0]
        if return_dict:
            return ac, ac_dict
        else:
            return ac

    def _normalize_obs(self, obs):
        state = obs["state"]
        state = state.view(state.shape[0], self.num_eef, -1)
        z_pos, z_dir, z_scalar = self.actor._convert_state_to_vec(state)
        z_pos = self.state_normalizer.normalize(z_pos)

        pc = self.pc_normalizer.normalize(obs["pc"])

        return dict(pc=pc, z_pos=z_pos, z_dir=z_dir, z_scalar=z_scalar)

    def update(self, batch, vis=False):
        self.train()
        batch = to_torch(batch, self.device)

        B = batch["pc"].shape[0]

        # normalize pc
        if self.ac_normalizer is None:
            self._init_normalizers(batch)
        obs = self._normalize_obs(dict(pc=batch["pc"], state=batch["eef_pos"]))

        # normalize and process action
        gt_action = self.ac_normalizer.normalize(batch["action"])
        gt_action = gt_action.view(B, self.num_eef, self.dof)
        if self.dof > 3:
            gt_gripper_ac = gt_action[:, :, 0].reshape(B, -1)  # (B, E)
        gt_offset = gt_action[
            :, :, (0 if self.dof == 3 else 1) :
        ]  # (B, num_eef, dof - 1)

        # get policy prediction
        ac_dict = self.actor(obs)
        offsets = ac_dict["offsets"]  # (B, num_eef, dof - 1)
        if self.dof > 3:
            gripper_ac = ac_dict["gripper_ac"]

        # compute ground truth one-hot heatmap
        repeated_pc = np.repeat(obs["pc"][:, None].cpu().numpy(), self.num_eef, 1)
        repeated_pc = repeated_pc.reshape(B * self.num_eef, -1, 3)
        reshaped_gt_xyz = (
            obs["z_pos"][:, : self.num_eef, :3]
            .reshape(B * self.num_eef, 3)
            .cpu()
            .numpy()
        )
        gt_indices = find_closest_point_indices(repeated_pc, reshaped_gt_xyz)
        gt_indices = torch.tensor(gt_indices).to(self.device)  # (B * E)

        # calculate weight on offset loss
        # weight locations close to EEF positions higher
        offset_loss = 0
        global_offset_weights = torch.zeros((B, self.num_points), device=offsets.device)
        for i in range(self.num_eef):
            dist_pc_eef = torch.linalg.norm(
                obs["pc"] - obs["z_pos"][:, [i], :3], dim=-1
            )  # (B, M)
            offset_weights = torch.exp(-(dist_pc_eef**2) / (2 * self.sigma))
            global_offset_weights += offset_weights
            offset_weights = offset_weights / offset_weights.sum(1)[:, None]
            offset_loss = offset_loss + (
                (offset_weights[:, :, None] * (offsets - gt_offset[:, [i]]) ** 2)
                .sum(1)
                .mean()
            )
        vis_offset_weights = (
            global_offset_weights
            / torch.max(global_offset_weights, dim=1, keepdim=True)[0]
        )

        # BCE on gripper actions
        if self.dof > 3:
            gripper_loss = self.bce_loss(gripper_ac, gt_gripper_ac)

        loss = offset_loss * self.w_o
        if self.dof > 3:
            loss = loss + gripper_loss * self.w_g

        # Compute raw ac loss
        raw_ac_loss = torch.abs(ac_dict["ac"] - gt_action.reshape(B, -1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        if not self._pca_initialized:
            self._fit_pca(ac_dict["local_feat"])
        if not self._global_pca_initialized:
            self._fit_global_pca(ac_dict["global_feat"])

        # visualize offset maps
        if vis:
            ac_dict["pc"] = obs["pc"]
            if "rgb" in batch:
                image = batch["rgb"][0].detach().cpu().numpy()
            else:
                image = np.zeros((256, 256, 3))
            vis_pc_img = self.visualize_sample(
                image,
                {
                    k: v[0]
                    for k, v in ac_dict.items()
                    if len(v.shape) > 0 and len(v) == B
                },
                obs["z_pos"][0][: self.num_eef, :3].detach().cpu().numpy(),
                gt_offset=gt_offset[0][:, :3],
                offset_weights=vis_offset_weights[0][:, None][:, [0, 0, 0]],
            )

        metrics = {
            "offset_loss": offset_loss,
            "loss": loss,
            "raw_ac_loss": raw_ac_loss,
            "gt_vel_norm": np.linalg.norm(
                gt_action[:, 0, (1 if self.dof >= 4 else 0) :]
                .reshape(-1, 3)
                .detach()
                .cpu()
                .numpy(),
                axis=1,
            ).mean(),
            "gt_offset_vel_norm": np.linalg.norm(
                gt_offset[:, 0].detach().cpu().numpy(),
                axis=1,
            ).mean(),
            "pred_offset_vel_norm": np.linalg.norm(
                offsets[torch.arange(B), gt_indices.reshape(B, self.num_eef)[:, 0]]
                .detach()
                .cpu()
                .numpy(),
                axis=1,
            ).mean(),
            "min_scale": ac_dict["min_scale"].detach().cpu().numpy(),
            "mean_scale": ac_dict["mean_scale"].detach().cpu().numpy(),
            "max_scale": ac_dict["max_scale"].detach().cpu().numpy(),
            "gt_ac_scale": torch.norm(gt_action, dim=-1).detach().cpu().numpy().mean(),
        }
        if self.dof > 3:
            metrics["gripper_loss"] = gripper_loss
        if vis:
            metrics.update({"vis_pc": vis_pc_img})
        return metrics

    def visualize_sample(
        self,
        rgb,
        ac_dict,
        eef_pos,
        gt_offset=None,
        offset_weights=None,
        global_features=None,
        return_wandb_image=True,
    ):
        offsets = ac_dict["offsets"][:, :3]
        pc = ac_dict["pc"]
        if torch.is_tensor(pc):
            pc = pc.cpu().numpy()

        vis_pts, vis_colors, titles, arrows = [], [], [], []

        # filter indices with large predicted gripper values
        vis_weights = np.zeros([pc.shape[0]])  # (M)
        for i in range(self.num_eef):
            dist_pc_eef = np.linalg.norm(pc - eef_pos[[i]], axis=-1)  # (M)
            vis_weights += np.exp(-(dist_pc_eef**2) / (2 * self.sigma))
        idxs = np.where(vis_weights > 0.5)[0][::10]

        # visualize local features
        if self._pca_initialized:
            local_features = ac_dict["local_feat"]  # [C, 3, M]
            vis_features = self._inference_pca(local_features)
            vis_pts.append(pc)
            vis_colors.append(vis_features)
            titles.append("Local Features")
            arrows.append([])

        # visualize_global features
        if global_features is not None and self._global_pca_initialized:
            vis_global_features = self._inference_global_pca(np.array(global_features))
            vis_pts.append(vis_global_features)
            colors = np.full((len(vis_global_features), 3), 0.8)
            colors[-1] = 0.0
            vis_colors.append(colors)
            titles.append("Global Features")
            arrows.append([])

        # visualize ground truth offsets
        if gt_offset is not None:
            vis_gt_offset_np = gt_offset.detach().cpu().numpy() / self.demo_freq
            vis_pts.append(pc)
            vis_colors.append(np.repeat([[0.8, 0.8, 0.8]], len(pc), axis=0))
            titles.append("GT Offset")
            arrows.append(np.array([eef_pos, vis_gt_offset_np * 10]))
        if offset_weights is not None:
            vis_pts.append(pc)
            vis_colors.append(0.8 - offset_weights.detach().cpu().numpy() * 0.8)
            titles.append("Offset Weights")
            arrows.append([])

        # visualize predicted offsets
        vis_offset_np = offsets.detach().cpu().numpy() / self.demo_freq
        vis_pts.append(pc)
        vis_colors.append(np.repeat([[0.8, 0.8, 0.8]], len(pc), axis=0))
        titles.append("Predicted Action")
        arrows.append(np.array([pc[idxs], vis_offset_np[idxs] * 10]))

        vis_pc_img = plot_points_grid(
            vis_pts,
            color=vis_colors,
            blank_bg=True,
            marker_size=40,
            titles=titles,
            arrows=arrows,
            parameterize_scale_by_color=True,
        )
        target_rgb_shape = (
            int(rgb.shape[1] / rgb.shape[0] * vis_pc_img.shape[0]),
            vis_pc_img.shape[0],
        )
        rgb = cv2.resize(rgb.copy(), target_rgb_shape)
        vis_pc_img = np.concatenate([rgb, vis_pc_img], axis=1)
        if return_wandb_image:
            vis_pc_img = wandb.Image(vis_pc_img)
        return vis_pc_img

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            pc_normalizer=self.pc_normalizer.state_dict(),
            state_normalizer=self.state_normalizer.state_dict(),
            ac_normalizer=self.ac_normalizer.state_dict(),
            pc_scale=self.pc_scale,
        )
        torch.save(state_dict, save_path)

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        self.state_normalizer = Normalizer(state_dict["state_normalizer"])
        self.actor.state_normalizer = self.state_normalizer
        self.ac_normalizer = Normalizer(state_dict["ac_normalizer"])
        if self.actor.obs_mode.startswith("pc"):
            self.pc_normalizer = self.state_normalizer
            self.actor.pc_normalizer = self.pc_normalizer
        self.actor.load_state_dict(state_dict["actor"])
        self.pc_scale = state_dict["pc_scale"]
        self.actor.pc_scale = self.pc_scale

    def _fit_pca(self, features):
        B, C, _, M = features.shape
        features = torch.norm(features, dim=-2)
        features = features.transpose(1, 2).contiguous()
        features = features.detach().cpu().numpy()
        features_flattened = features.reshape(-1, C)
        self.pca.fit(features_flattened)
        pca_features = self.pca.transform(features_flattened)
        self._pca_range = (pca_features.min(axis=0), pca_features.max(axis=0))
        self._pca_initialized = True

    def _inference_pca(self, features):
        C, _, M = features.shape
        features = torch.norm(features, dim=-2)
        features = features.transpose(0, 1).contiguous()
        features = features.detach().cpu().numpy()
        features_flattened = features.reshape(-1, C)
        feats = self.pca.transform(features_flattened)
        feats = (feats - self._pca_range[0][None]) / (
            self._pca_range[1] - self._pca_range[0]
        )[None]
        feats = np.clip(feats, 0.0, 1.0)
        feats = feats.reshape(M, 3)
        return feats

    def _fit_global_pca(self, features):
        B, C, _ = features.shape
        features = torch.norm(features, dim=-1)
        features = features.detach().cpu().numpy()
        self.global_pca.fit(features)
        pca_features = self.pca.transform(features)
        self._global_pca_range = (pca_features.min(axis=0), pca_features.max(axis=0))
        self._global_pca_initialized = True

    def _inference_global_pca(self, features):
        features = np.linalg.norm(features, axis=-1)
        feats = self.global_pca.transform(features)
        feats = (feats - self._global_pca_range[0][None]) / (
            self._global_pca_range[1] - self._global_pca_range[0]
        )[None]
        feats = np.clip(feats, 0.0, 1.0)
        return feats
