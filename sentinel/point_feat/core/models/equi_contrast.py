import os, sys
from torch import nn

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib_vec.sim3_encoder import SIM3Vec4Latent


class EquiContrast(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = {
            "sim3_encoder": SIM3Vec4Latent,
        }[
            cfg["encoder_class"]
        ](**cfg["encoder"])

    def encode_pcl(self, xyz, ret_perpoint_feat=False, target_norm=1.0):
        ret = self.encoder(
            xyz, ret_perpoint_feat=ret_perpoint_feat, target_norm=target_norm
        )

        return ret
