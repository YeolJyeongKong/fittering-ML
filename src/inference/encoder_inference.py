import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from src.models.lightning_modules import AutoEncoderModule, CombAutoEncoderModule
from extras import paths


class InferenceEncoder:
    def __init__(
        self,
        model_path,
        device=torch.device("cuda"),
    ) -> None:
        self.autoencoder = CombAutoEncoderModule().to(device)
        ckpt = torch.load(model_path, map_location=device)
        self.autoencoder.load_state_dict(ckpt["state_dict"])
        self.autoencoder.eval()

    def inference(self, front, side):
        with torch.no_grad():
            front = self.front_ae(front)
            side = self.side_ae(side)
        return torch.cat([front, side], dim=1)


def encode_(module, dataloader, dataset_len, batch_size, device):
    x = np.empty((dataset_len, 515))
    y = np.empty((dataset_len, 8))

    for idx, batch in tqdm(
        enumerate(dataloader),
        desc="image encoding...",
        total=(dataset_len // batch_size),
    ):
        with torch.no_grad():
            front_z, side_z = module(
                batch["front"].to(device), batch["side"].to(device)
            )
        z = torch.cat([front_z, side_z], dim=1)

        x_tensor = torch.cat(
            [
                z.cpu(),
                batch["height"].unsqueeze(dim=1),
                batch["weight"].unsqueeze(dim=1),
                batch["sex"].unsqueeze(dim=1),
            ],
            dim=1,
        )
        x_arr = x_tensor.cpu().numpy()
        x[idx * batch_size : (idx + 1) * batch_size] = x_arr

        y_arr = batch["meas"].cpu().numpy()
        y[idx * batch_size : (idx + 1) * batch_size] = y_arr

    return x, y


def encode(module, dm, device):
    module.eval()
    module = module.to(device)

    dm.train_ratio = 1.0
    dm.shuffle = False
    dm.batch_size = 32

    train_x, train_y = encode_(
        module, dm.train_dataloader(), len(dm.train_dataset), dm.batch_size, device
    )
    test_x, test_y = encode_(
        module, dm.test_dataloader(), len(dm.test_dataset), dm.batch_size, device
    )

    return (train_x, train_y), (test_x, test_y)
