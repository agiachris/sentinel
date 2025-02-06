from typing import Dict, Union

import copy
import hydra
import torch
import numpy as np
from torch import nn

from sentinel.point_feat.core.lib_pn.pointnet_encoder import PointNetEncoder
from sentinel.utils.diffusion.ema_model import EMAModel
from sentinel.utils.diffusion.conditional_unet1d import ConditionalUnet1D
from sentinel.utils.diffusion.resnet_with_gn import get_resnet, replace_bn_with_gn


class DPPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim = cfg.model.hidden_dim
        self.obs_mode = cfg.model.obs_mode
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        self.num_eef = cfg.env.num_eef
        self.eef_dim = cfg.env.eef_dim
        self.dof = cfg.env.dof
        if cfg.model.obs_mode == "state":
            self.obs_dim = self.num_eef * self.eef_dim
        elif cfg.model.obs_mode == "rgb":
            self.obs_dim = 512 + self.num_eef * self.eef_dim
        else:
            self.obs_dim = hidden_dim + self.num_eef * self.eef_dim
        self.action_dim = self.dof * cfg.env.num_eef

        if self.obs_mode.startswith("pc"):
            self.encoder = PointNetEncoder(
                h_dim=hidden_dim,
                c_dim=hidden_dim,
                num_layers=cfg.model.encoder.backbone_args.num_layers,
            )
        elif self.obs_mode == "rgb":
            self.encoder = replace_bn_with_gn(get_resnet("resnet18"))
        else:
            self.encoder = nn.Identity()
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            diffusion_step_embed_dim=self.obs_dim * self.obs_horizon,
            global_cond_dim=self.obs_dim * self.obs_horizon,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DP Policy with {num_parameters} parameters")

        # Utilities for out-of-distribution metrics.
        self.rec_timesteps = [
            t - 1 for t in getattr(cfg.model, "rec_depths", [self.num_diffusion_iters])
        ]
        assert (
            len(self.rec_timesteps) > 0
            and max(self.rec_timesteps) < self.num_diffusion_iters
        )

        self.noise_pred_samples = getattr(
            cfg.model, "noise_pred_samples", self.num_diffusion_iters
        )
        assert self.noise_pred_samples <= self.num_diffusion_iters

    def _init_torch_compile(self):
        if self.cfg.model.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.noise_pred_net_handle = torch.compile(self.noise_pred_net)

    def forward(self, obs, predict_action=True, debug=False, return_dict=False):
        # assumes that observation has format:
        # - pc: [BS, obs_horizon, num_pts, 3]
        # - state: [BS, obs_horizon, obs_dim]
        # returns:
        # - action: [BS, pred_horizon, ac_dim]
        pc = obs["pc"]
        state = obs["state"]
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        pc_shape = pc.shape
        batch_size = pc.shape[0]

        # Encoder forward pass.
        ema_nets = self.ema.averaged_model
        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode == "rgb":
                rgb = obs["rgb"]
                rgb_shape = rgb.shape
                flattened_rgb = rgb.reshape(
                    batch_size * self.obs_horizon, *rgb_shape[-3:]
                )
                z = ema_nets["encoder"](flattened_rgb.permute(0, 3, 1, 2))
            else:
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
            z = feat = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)
        obs_cond = z.reshape(batch_size, -1)

        initial_noise_scale = 0.0 if debug else 1.0
        noisy_action = (
            torch.randn((batch_size, self.pred_horizon, self.action_dim)).to(
                self.device
            )
            * initial_noise_scale
        )
        curr_action = noisy_action

        # Reverse diffusion process.
        noise_preds = np.zeros((self.num_diffusion_iters, *curr_action.shape))
        diff_actions = np.zeros((self.num_diffusion_iters, *curr_action.shape))
        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_diffusion_iters)
        for i, k in enumerate(self.noise_scheduler.timesteps):
            diff_actions[-i - 1] = curr_action.detach().cpu().numpy()
            noise_pred = ema_nets["noise_pred_net"](
                sample=curr_action, timestep=k, global_cond=obs_cond
            )
            noise_preds[-i - 1] = noise_pred.detach().cpu().numpy()

            curr_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=curr_action
            ).prev_sample

        # Return action.
        ret = dict(ac=curr_action)  # (B, H, D)
        if return_dict:
            ret.update(
                dict(
                    obs_cond_vec=obs_cond.detach().cpu().numpy(),
                    noise_preds=noise_preds,  # (T, B, H, D)
                    diff_actions=diff_actions,  # (T, B, H, D)
                )
            )
            if self.obs_mode != "state":
                ret.update(dict(feat=feat.detach().cpu().numpy()))

        return ret

    def reconstruct_action(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Diffusion reconstruction of actions."""
        pc = obs["pc"]
        state = obs["state"]
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        pc_shape = pc.shape
        batch_size = pc.shape[0]

        # Encoder forward pass.
        ema_nets = self.ema.averaged_model
        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode == "rgb":
                rgb = obs["rgb"]
                rgb_shape = rgb.shape
                flattened_rgb = rgb.reshape(
                    batch_size * self.obs_horizon, *rgb_shape[-3:]
                )
                z = ema_nets["encoder"](flattened_rgb.permute(0, 3, 1, 2))
            else:
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
            z = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)
        obs_cond = z.reshape(batch_size, -1)

        # Action reconstruction.
        noise_preds = []
        recs = torch.zeros(
            (len(self.rec_timesteps), *action.shape), device=self.device
        ).float()
        for i, t in enumerate(self.rec_timesteps):
            # Forward diffusion process.
            noise = torch.randn(action.shape, device=self.device)
            timesteps = torch.ones((batch_size,), device=self.device).long() * t
            noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
            curr_action = noisy_action

            # Reverse diffusion process.
            noise_preds.append(np.zeros((t + 1, *curr_action.shape)))
            self.noise_scheduler.set_timesteps(
                timesteps=[_t for _t in range(t + 1)][::-1]
            )
            for j, k in enumerate(self.noise_scheduler.timesteps):
                noise_pred = ema_nets["noise_pred_net"](
                    sample=curr_action, timestep=k, global_cond=obs_cond
                )
                noise_preds[-1][-j - 1] = noise_pred.detach().cpu().numpy()

                curr_action = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=curr_action
                ).prev_sample

            recs[i] = curr_action

        # Return action reconstructions.
        ret = dict(ac=recs)  # (R, B, H, D)
        if return_dict:
            ret.update(
                dict(
                    noise_preds=noise_preds,  # List[(T, B, H, D)], len(List) = R
                )
            )

        return ret

    def predict_noise(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Noise prediction for actions."""
        pc = obs["pc"]
        state = obs["state"]
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        pc_shape = pc.shape
        batch_size = pc.shape[0]

        # Encoder forward pass.
        ema_nets = self.ema.averaged_model
        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode == "rgb":
                rgb = obs["rgb"]
                rgb_shape = rgb.shape
                flattened_rgb = rgb.reshape(
                    batch_size * self.obs_horizon, *rgb_shape[-3:]
                )
                z = ema_nets["encoder"](flattened_rgb.permute(0, 3, 1, 2))
            else:
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
            z = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)
        obs_cond = z.reshape(batch_size, -1)

        noise_labels = torch.randn(
            (self.noise_pred_samples, *action.shape), device=self.device
        ).float()
        noise_preds = torch.zeros(
            (self.noise_pred_samples, *action.shape), device=self.device
        ).float()
        timesteps = torch.zeros(
            (self.noise_pred_samples, batch_size), device=self.device
        ).long()
        for i in range(batch_size):
            timesteps[:, i] = torch.randperm(self.num_diffusion_iters)[
                : self.noise_pred_samples
            ]

        # Noise prediction.
        for i in range(self.noise_pred_samples):
            noisy_actions = self.noise_scheduler.add_noise(
                action, noise_labels[i], timesteps[i]
            )
            noise_preds[i] = ema_nets["noise_pred_net"](
                sample=noisy_actions, timestep=timesteps[i], global_cond=obs_cond
            )

        # Return noise predictions.
        ret = dict(
            noise_preds=noise_preds,  # (S, B, H, D)
            noise_labels=noise_labels,  # (S, B, H, D)
        )
        if return_dict:
            ret.update(
                dict(
                    timesteps=timesteps.detach().cpu().numpy(),  # (S, B)
                )
            )

        return ret

    def reconstruct_noise(
        self,
        action: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Reconstruct noise for actions."""
        pc = obs["pc"]
        state = obs["state"]
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        pc_shape = pc.shape
        batch_size = pc.shape[0]

        # Encoder forward pass.
        ema_nets = self.ema.averaged_model
        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode == "rgb":
                rgb = obs["rgb"]
                rgb_shape = rgb.shape
                flattened_rgb = rgb.reshape(
                    batch_size * self.obs_horizon, *rgb_shape[-3:]
                )
                z = ema_nets["encoder"](flattened_rgb.permute(0, 3, 1, 2))
            else:
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
            z = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)
        obs_cond = z.reshape(batch_size, -1)

        assert action.shape[0] == self.num_diffusion_iters
        curr_action = action[-1]

        noise_preds = torch.zeros_like(action, device=self.device).float()
        diff_actions = np.zeros_like(action.detach().cpu())
        self.noise_scheduler.set_timesteps(num_inference_steps=self.num_diffusion_iters)
        for i, k in enumerate(self.noise_scheduler.timesteps):
            diff_actions[-i - 1] = curr_action.detach().cpu().numpy()
            noise_preds[-i - 1] = ema_nets["noise_pred_net"](
                sample=action[-i - 1], timestep=k, global_cond=obs_cond
            )

            curr_action = self.noise_scheduler.step(
                model_output=noise_preds[-i - 1], timestep=k, sample=action[-i - 1]
            ).prev_sample

        # Return action.
        ret = dict(noise_preds=noise_preds)  # (T, B, H, D)
        if return_dict:
            ret.update(
                dict(
                    diff_actions=diff_actions,  # (T, B, H, D)
                )
            )

        return ret

    def step_ema(self):
        self.ema.step(self.nets)
