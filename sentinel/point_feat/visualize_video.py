import os
import numpy as np
import torch
from torch import optim
from tensorboardX import SummaryWriter
import argparse

from sentinel.point_feat.utils import config
from sentinel.point_feat.utils.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description="Train the point feature extraction model."
)
parser.add_argument(
    "--config", default="configs/default.yaml", type=str, help="Path to config file."
)
args = parser.parse_args()


cfg = config.load_config(args.config, "configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Shorthands
out_dir = cfg["training"]["out_dir"]
batch_size_vis = cfg["training"]["batch_size_vis"]
lr = cfg["training"]["learning_rate"]

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Dataset
cfg["data"]["dataset"]["t_steps"] = cfg["vis_video"]["t_steps"]
val_dataset = config.get_dataset(cfg, "val")


# Dataloader for visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_vis, shuffle=False, drop_last=True
)


# Model
model = config.get_model(cfg, device=device)


# Get optimizer and trainer
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = config.get_trainer(model, optimizer, cfg, device=device)


# Load pre-trained model is existing
kwargs = {
    "model": model,
    "optimizer": optimizer,
}
checkpoint_io = CheckpointIO(
    out_dir,
    initialize_from=cfg["training"]["initialize_from"],
    initialization_file_name=cfg["training"]["initialization_file_name"],
    **kwargs
)
try:
    load_dict = checkpoint_io.load("model.pt")
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get("epoch_it", -1)
it = load_dict.get("it", -1)
metric_val_best = load_dict.get("loss_val_best", np.inf)

logger = SummaryWriter(os.path.join(out_dir, "logs"))


# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print("Total number of parameters: %d" % nparameters)


for idx, data in enumerate(vis_loader):
    if idx >= cfg["vis_video"]["num_samples"]:
        break
    trainer.visualize_seq(cfg["vis_video"], data, idx, save_gif=True)
    print("Batch %d visualized" % idx)
