import numpy as np
import torch
import torch.nn as nn

from lightning_modules import AutoEncoderModule
import config


class InferenceEncoder:
    def __init__(
        self,
        front_ae_path=config.FRONTAE_PATH,
        side_ae_path=config.SIDEAE_PATH,
        device=torch.device("cuda"),
    ) -> None:
        self.front_ae = AutoEncoderModule().to(device)
        ckpt = torch.load(front_ae_path, map_location=device)
        self.front_ae.load_state_dict(ckpt["state_dict"])
        self.front_ae.eval()

        self.side_ae = AutoEncoderModule().to(device)
        ckpt = torch.load(side_ae_path, map_location=device)
        self.side_ae.load_state_dict(ckpt["state_dict"])
        self.side_ae.eval()

    def inference(self, front, side):
        with torch.no_grad():
            front = self.front_ae(front)
            side = self.side_ae(side)
        return torch.cat([front, side], dim=1)


if __name__ == "__main__":
    inference = InferenceEncoder()
    pred = inference.inference(
        torch.zeros((16, 1, 512, 512)), torch.zeros((16, 1, 512, 512))
    )
    print(pred.shape)
