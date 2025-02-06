import torch
from torch import nn
import numpy as np
import time

from sentinel.utils.norm import Normalizer
from sentinel.bc.utils import to_torch
from sentinel.bc.agents.dp_agent import DPAgent
from sentinel.bc.policies.sim3_dp_policy import SIM3DPPolicy


class SIM3DPAgent(DPAgent):
    def _init_actor(self):
        self.actor = SIM3DPPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)
        self.pc_scale = None

    def _init_normalizers(self, batch):
        if self.use_normalization:
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
        if self.use_normalization:
            pc = batch["pc"].reshape(-1, self.num_points, 3)
            centroid = pc.mean(1, keepdim=True)
            centered_pc = pc - centroid
            pc_scale = centered_pc.norm(dim=-1).mean()
            ac_scale = ac_normalizer.stats["max"].max()
            self.pc_scale = pc_scale / ac_scale
            self.actor.pc_scale = self.pc_scale
        else:
            self.pc_scale = 1.0
            self.actor.pc_scale = 1.0
        print("=" * 20)
        print(f" => PC scale = {self.pc_scale}")
        print("=" * 20)

    def update(self, batch, vis=False):
        self.train()

        if self.cfg.debug.debug_speed:
            tt = time.time()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        # rgb = batch["rgb"]
        state = batch["eef_pos"]
        gt_action = batch["action"]  # torch.Size([32, 16, self.num_eef * self.dof])

        if self.pc_scale is None:
            self._init_normalizers(batch)
        if self.use_normalization:
            pc = self.pc_normalizer.normalize(pc)
            gt_action = self.ac_normalizer.normalize(gt_action)

        pc_shape = pc.shape
        batch_size = B = pc.shape[0]
        Ho = self.obs_horizon
        Hp = self.pred_horizon

        if self.obs_mode == "state":
            z_pos, z_dir, z_scalar = self.actor._convert_state_to_vec(state)
            if self.use_normalization:
                z_pos = self.state_normalizer.normalize(z_pos)
            if self.dof > 4:
                z = torch.cat([z_pos, z_dir], dim=-2)
            else:
                z = z_pos
        else:
            feat_dict = self.actor.encoder_handle(pc, target_norm=self.pc_scale)

            center = (
                feat_dict["center"].reshape(B, Ho, 1, 3)[:, [-1]].repeat(1, Ho, 1, 1)
            )
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Ho, 1, 1)
            z_pos, z_dir, z_scalar = self.actor._convert_state_to_vec(state)
            if self.use_normalization:
                z_pos = self.state_normalizer.normalize(z_pos)
            z_pos = (z_pos - center) / scale
            z = feat_dict["so3"]
            z = z.reshape(B, Ho, -1, 3)
            if self.dof > 4:
                z = torch.cat([z, z_pos, z_dir], dim=-2)
            else:
                z = torch.cat([z, z_pos], dim=-2)
        obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), (
            z_scalar.reshape(B, -1) if z_scalar is not None else None
        )

        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] obs processing took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        if self.obs_mode.startswith("pc"):
            if self.ac_mode == "abs":
                center = (
                    feat_dict["center"]
                    .reshape(B, Ho, 1, 3)[:, [-1]]
                    .repeat(1, Hp, 1, 1)
                )
            else:
                center = 0
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Hp, 1, 1)
            gt_action = gt_action.reshape(B, Hp, self.num_eef, self.dof)
            if self.dof == 4:
                gt_action = torch.cat(
                    [gt_action[..., :1], (gt_action[..., 1:] - center) / scale], dim=-1
                )
                gt_action = gt_action.reshape(B, Hp, -1)
            elif self.dof == 3:
                gt_action = (gt_action - center) / scale
            elif self.dof == 7:
                gt_action = torch.cat(
                    [
                        gt_action[..., :1],
                        (gt_action[..., 1:4] - center) / scale,
                        gt_action[..., 4:],
                    ],
                    dim=-1,
                )
                gt_action = gt_action.reshape(B, Hp, -1)
            else:
                raise ValueError(f"Dof {self.dof} not supported.")
        vec_eef_action, vec_gripper_action = self.actor._convert_action_to_vec(
            gt_action, batch
        )
        if self.dof != 7:
            noise = torch.randn(gt_action.shape, device=self.device)
            vec_eef_noise, vec_gripper_noise = self.actor._convert_action_to_vec(
                noise, batch
            )  # to debug
        else:
            vec_eef_noise = torch.randn_like(vec_eef_action, device=self.device)
            vec_gripper_noise = torch.randn_like(vec_gripper_action, device=self.device)
        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] action to vec took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        if vec_gripper_action is not None:
            noisy_eef_actions = self.actor.noise_scheduler.add_noise(
                vec_eef_action, vec_eef_noise, timesteps
            )
            noisy_gripper_actions = self.actor.noise_scheduler.add_noise(
                vec_gripper_action, vec_gripper_noise, timesteps
            )

            vec_eef_noise_pred, vec_gripper_noise_pred = (
                self.actor.noise_pred_net_handle(
                    noisy_eef_actions.permute(0, 3, 1, 2),
                    timesteps,
                    scalar_sample=noisy_gripper_actions.permute(0, 2, 1),
                    cond=obs_cond_vec,
                    scalar_cond=obs_cond_scalar,
                )
            )
            vec_eef_noise_pred = vec_eef_noise_pred.permute(0, 2, 3, 1)
            vec_gripper_noise_pred = vec_gripper_noise_pred.permute(0, 2, 1)
            if self.dof != 7:
                noise_pred = self.actor._convert_action_to_scalar(
                    vec_eef_noise_pred, vec_gripper_noise_pred, batch=batch
                ).view(noise.shape)
        else:
            noisy_eef_actions = self.actor.noise_scheduler.add_noise(
                vec_eef_action, vec_eef_noise, timesteps
            )

            vec_noise_pred = self.actor.noise_pred_net_handle(
                noisy_eef_actions.permute(0, 3, 1, 2),
                timesteps,
                cond=obs_cond_vec,
                scalar_cond=obs_cond_scalar,
            )[0].permute(0, 2, 3, 1)
            if self.dof != 7:
                noise_pred = self.actor._convert_action_to_scalar(
                    vec_noise_pred, batch=batch
                ).view(noise.shape)

        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] running noise pred net took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        if self.dof == 7:
            n_vec = np.prod(vec_eef_noise_pred.shape)
            n_sca = np.prod(vec_gripper_noise_pred.shape)
            k = (n_vec) / (n_vec + n_sca)
            loss = nn.functional.mse_loss(
                vec_eef_noise_pred, vec_eef_noise
            ) * k + nn.functional.mse_loss(
                vec_gripper_noise_pred, vec_gripper_noise
            ) * (
                1 - k
            )
        else:
            loss = nn.functional.mse_loss(noise_pred, noise)
        if torch.isnan(loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] compute loss took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] optim step took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        self.actor.step_ema()

        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] ema step took {time_elapsed} ms on GPU and {(time.time() - tt) * 1000:.2f}ms on CPU"
            )
            tt = time.time()
            start.record()

        metrics = {
            "loss": loss,
            "normalized_gt_ac_max": np.max(
                np.abs(vec_eef_action.reshape(-1, 3).detach().cpu().numpy()), axis=0
            ).mean(),
        }
        if self.dof == 7:
            metrics.update(
                {
                    "mean_gt_eef_noise_norm": np.linalg.norm(
                        vec_eef_noise.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_pred_eef_noise_norm": np.linalg.norm(
                        vec_eef_noise_pred.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_gt_gripper_noise_norm": np.linalg.norm(
                        vec_gripper_noise.detach().cpu().numpy(), axis=1
                    ).mean(),
                    "mean_pred_gripper_noise_norm": np.linalg.norm(
                        vec_gripper_noise_pred.detach().cpu().numpy(), axis=1
                    ).mean(),
                }
            )
        else:
            metrics.update(
                {
                    "mean_gt_noise_norm": np.linalg.norm(
                        noise.reshape(gt_action.shape[0], -1).detach().cpu().numpy(),
                        axis=1,
                    ).mean(),
                    "mean_pred_noise_norm": np.linalg.norm(
                        noise_pred.reshape(gt_action.shape[0], -1)
                        .detach()
                        .cpu()
                        .numpy(),
                        axis=1,
                    ).mean(),
                }
            )
        if self.cfg.debug.debug_speed:
            end.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Calculate time elapsed
            time_elapsed = start.elapsed_time(end)  # Time in milliseconds
            print(
                f"[sim3_dp_agent.py] preparing metrics took {time_elapsed} ms in CUDA-SYNC Timer"
            )
            tt = time.time()
            start.record()

        return metrics

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
            pc_scale=self.pc_scale,
        )
        if self.use_normalization:
            state_dict.update(
                dict(
                    pc_normalizer=self.pc_normalizer.state_dict(),
                    state_normalizer=self.state_normalizer.state_dict(),
                    ac_normalizer=self.ac_normalizer.state_dict(),
                )
            )
        torch.save(state_dict, save_path)

    def fix_checkpoint_keys(self, state_dict):
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            if "encoder.encoder" in k:
                fixed_k = k.replace("encoder.encoder", "encoder")
            else:
                fixed_k = k
            if "handle" in k:
                continue
            fixed_state_dict[fixed_k] = v
        return fixed_state_dict

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        if self.use_normalization:
            self.state_normalizer = Normalizer(state_dict["state_normalizer"])
            self.actor.state_normalizer = self.state_normalizer
            self.ac_normalizer = Normalizer(state_dict["ac_normalizer"])
            if self.obs_mode.startswith("pc"):
                self.pc_normalizer = self.state_normalizer
                self.actor.pc_normalizer = self.pc_normalizer
        del self.actor.encoder_handle
        del self.actor.noise_pred_net_handle
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()
        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )
        self.pc_scale = state_dict["pc_scale"]
        self.actor.pc_scale = self.pc_scale
        self.ckpt = save_path
