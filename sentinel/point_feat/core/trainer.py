import os, sys
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from pytorch3d.transforms import RotateAxisAngle

from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sentinel.point_feat.utils.vis_points import (
    plot_points_grid,
    plot_points_seq,
)


class Trainer(object):
    """Trainer object for ONet 4D.
    The Trainer object obtains methods to perform a train and eval step
    as well as to visualize the current training state.

    Args:
        model (nn.Module): model to train
        optimizer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        threshold (float): threshold value for decision boundary
    """

    def __init__(self, model, optimizer, cfg, device="cuda"):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.vis_dir = os.path.join(cfg["training"]["out_dir"], "vis")
        os.makedirs(self.vis_dir, exist_ok=True)

    def train_step(self, data):
        """Performs a train step.

        Args:
            data (dict): training data
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if torch.isnan(loss):
            print(f"Loss is nan!")
            import pdb

            pdb.set_trace()
        else:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        """Performs a validation step.

        Args:
            data (dict): validation data
        """
        self.model.eval()
        device = self.device
        inputs = data.get("inputs", torch.empty(1, 0)).to(device)
        eval_dict = {}
        with torch.no_grad():
            loss = self.compute_loss(data)

        eval_dict["loss"] = loss.mean().item()
        return eval_dict

    def compute_loss(self, data):
        loss = self.model.get_loss(data)

        return loss

    def PCA(self, features):
        B, N, C = features.shape
        features_flattened = features.reshape(-1, C)
        pca = PCA(n_components=3)
        pca.fit(features_flattened)

        pca_features = pca.transform(features_flattened)
        pca_features = (pca_features - pca_features.min()) / (
            pca_features.max() - pca_features.min()
        )
        pca_features = pca_features.reshape(B, N, 3)

        return pca_features

    def visualize(self, data, it):
        """Performs a visualization step.

        Args:
            data (dict): visualization data
        """
        cfg_vis = self.cfg["vis"]
        xyz = data["xyz"].to(self.device)
        mesh = data["mesh"].to(self.device) if "mesh" in data else xyz

        z = self.model.encode_pcl(xyz, ret_perpoint_feat=cfg_vis["vis_enc_feat"])
        if cfg_vis["vis_enc_feat"]:
            if "per_point_so3" in z:
                enc_feat = z["per_point_so3"]  # [B, C, 3, N]
                B, C, _, N = enc_feat.shape
                if cfg_vis["vis_enc_mode"] == "nocs":
                    enc_feat = enc_feat.reshape(B, -1, N)
                elif cfg_vis["vis_enc_mode"] == "norm":
                    enc_feat = torch.norm(enc_feat, dim=2)
                enc_feat = enc_feat.transpose(1, 2).contiguous()  # [B, N, 3C]
            elif "per_point" in z:
                enc_feat = z["per_point"]  # [B, N, C]
            enc_feat = enc_feat.detach().cpu().numpy()
            enc_feat_pca = self.PCA(enc_feat)

        pcl_feat = self.model.decode(xyz, z, seg=None)  # [B, N, C]
        xyz = xyz.detach().cpu().numpy()
        pcl_feat = pcl_feat.detach().cpu().numpy()

        mesh_idxs = data["mesh_idxs"].to(self.device) if "mesh_idxs" in data else None
        mesh_feat = self.model.decode(mesh, z, seg=mesh_idxs)  # [B, M, C]
        mesh = mesh.detach().cpu().numpy()
        mesh_feat = mesh_feat.detach().cpu().numpy()

        N = pcl_feat.shape[1]
        feats = np.concatenate([pcl_feat, mesh_feat], axis=1)
        feats_pca = self.PCA(feats)
        pcl_feat_pca = feats_pca[:, :N]
        mesh_feat_pca = feats_pca[:, N:]

        if len(xyz.shape) == 4:
            B, T, N, _ = xyz.shape
            xyz = xyz.reshape(B, T, -1, 3)
            mesh = mesh.reshape(B, T, -1, 3)
            pcl_feat_pca = pcl_feat_pca.reshape(B, T, -1, 3)
            mesh_feat_pca = mesh_feat_pca.reshape(B, T, -1, 3)
            if cfg_vis["vis_enc_feat"]:
                enc_feat_pca = enc_feat_pca.reshape(B, T, -1, 3)

        if cfg_vis["vis_enc_feat"]:
            save_path_enc = os.path.join(self.vis_dir, f"feat_it{it}_enc.png")
            plot_points_grid(xyz, save_path_enc, color=enc_feat_pca, blank_bg=True)

        save_path_pcl = os.path.join(self.vis_dir, f"feat_it{it}_pcl.png")
        plot_points_grid(xyz, save_path_pcl, color=pcl_feat_pca, blank_bg=True)

        save_path_mesh = os.path.join(self.vis_dir, f"feat_it{it}_mesh.png")
        plot_points_grid(mesh, save_path_mesh, color=mesh_feat_pca, blank_bg=True)

        return 0

    def visualize_seq(self, cfg_vis, data, it, save_gif=True):
        """Performs a visualization step.

        Args:
            data (dict): visualization data
        """
        xyz = data["xyz"].to(self.device)
        mesh = data["mesh"].to(self.device)

        T = xyz.shape[1]

        if "mesh_idxs" in data:
            mesh_idxs = data["mesh_idxs"].to(self.device)
        else:
            mesh_idxs = None

        if cfg_vis["apply_transforms"]:
            period = cfg_vis["transforms"]["period"]
            xyz = xyz.repeat(1, period, 1, 1)
            mesh = mesh.repeat(1, period, 1, 1)
            if mesh_idxs is not None:
                mesh_idxs = mesh_idxs.repeat(1, period, 1, 1)

            for p in range(period):
                if p % 2 == 1:
                    xyz[:, T * p : T * (p + 1)] = torch.flip(
                        xyz[:, T * p : T * (p + 1)], [1]
                    )
                    mesh[:, T * p : T * (p + 1)] = torch.flip(
                        mesh[:, T * p : T * (p + 1)], [1]
                    )
                    if mesh_idxs is not None:
                        mesh_idxs[:, T * p : T * (p + 1)] = torch.flip(
                            mesh_idxs[:, T * p : T * (p + 1)], [1]
                        )

            if cfg_vis["transforms"]["apply_rot"]:
                trot = RotateAxisAngle(
                    angle=torch.arange(cfg_vis["t_steps"] * period)
                    * 360
                    / (cfg_vis["t_steps"] * period),
                    axis="Z",
                    degrees=True,
                )
                rot = trot.get_matrix()[:, :3, :3].to(self.device)

                xyz = torch.einsum("btnj, tij -> btni", xyz, rot).contiguous()
                mesh = torch.einsum("btnj, tij -> btni", mesh, rot).contiguous()

            if cfg_vis["transforms"]["apply_scale"]:
                min_scale = cfg_vis["transforms"]["min_scale"]
                max_scale = cfg_vis["transforms"]["max_scale"]
                scales = torch.linspace(
                    max_scale, min_scale, cfg_vis["t_steps"] * period // 2
                )
                scales = torch.cat([scales, torch.flip(scales, [0])]).to(self.device)

                xyz = xyz * scales[None, :, None, None]
                mesh = mesh * scales[None, :, None, None]

        pcl_feat = []
        mesh_feat = []
        T0 = cfg_vis["t_batchsize"]
        for p in range(xyz.shape[1] // T0):
            xyz_batch = xyz[:, T0 * p : T0 * (p + 1)].contiguous()
            mesh_batch = mesh[:, T0 * p : T0 * (p + 1)].contiguous()

            B, T0, N, _ = xyz_batch.shape
            B, T0, M, _ = mesh_batch.shape

            z = self.model.encode_pcl(xyz_batch)

            pcl_feat_batch = self.model.decode(xyz_batch, z, seg=None)  # [B, N, C]

            mesh_idxs_batch = (
                mesh_idxs[:, T0 * p : T0 * (p + 1)].contiguous()
                if mesh_idxs is not None
                else None
            )
            mesh_feat_batch = self.model.decode(
                mesh_batch, z, seg=mesh_idxs_batch
            )  # [B, M, C]

            pcl_feat_batch = pcl_feat_batch.detach().cpu().numpy()
            mesh_feat_batch = mesh_feat_batch.detach().cpu().numpy()

            pcl_feat.append(pcl_feat_batch.reshape(B, T0, N, -1))
            mesh_feat.append(mesh_feat_batch.reshape(B, T0, M, -1))

        pcl_feat = np.concatenate(pcl_feat, axis=1).reshape(B * xyz.shape[1], N, -1)
        mesh_feat = np.concatenate(mesh_feat, axis=1).reshape(B * mesh.shape[1], M, -1)

        xyz = xyz.detach().cpu().numpy()
        mesh = mesh.detach().cpu().numpy()

        N = pcl_feat.shape[1]
        feats = np.concatenate([pcl_feat, mesh_feat], axis=1)
        feats_pca = self.PCA(feats)
        pcl_feat_pca = feats_pca[:, :N]
        mesh_feat_pca = feats_pca[:, N:]

        if len(xyz.shape) == 4:
            B, T, N, _ = xyz.shape
            xyz = xyz.reshape(B, T, -1, 3)
            mesh = mesh.reshape(B, T, -1, 3)
            pcl_feat_pca = pcl_feat_pca.reshape(B, T, -1, 3)
            mesh_feat_pca = mesh_feat_pca.reshape(B, T, -1, 3)

        save_path_pcl = os.path.join(f"{self.vis_dir}_seq", f"feat_it{it}_pcl")
        os.makedirs(save_path_pcl, exist_ok=True)
        plot_points_seq(
            xyz,
            save_path_pcl,
            color=pcl_feat_pca,
            blank_bg=True,
            canvas_radius=0.4,
            save_gif=save_gif,
            fps=cfg_vis["fps"],
        )

        save_path_mesh = os.path.join(f"{self.vis_dir}_seq", f"feat_it{it}_mesh")
        os.makedirs(save_path_mesh, exist_ok=True)
        plot_points_seq(
            mesh,
            save_path_mesh,
            color=mesh_feat_pca,
            blank_bg=True,
            canvas_radius=0.4,
            save_gif=save_gif,
            fps=cfg_vis["fps"],
        )

        return 0

    def evaluate(self, val_loader):
        """Performs an evaluation.
        Args:
            val_loader (dataloader): Pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
