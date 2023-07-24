import os
import glob
import pandas as pd
from PIL import Image
import cv2
import wandb
import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize, Compose
from torchvision.transforms import functional as F
from data.preprocessing import *

from utils.visualize import Beta2Verts
from utils.predict_measure import Beta2Measurements

from extras import paths


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples):
        super().__init__()
        self.val_samples = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        front = self.val_samples["front"].to(pl_module.device)
        side = self.val_samples["side"].to(pl_module.device)

        front_logits = pl_module.front_autoencoder(front)
        side_logits = pl_module.side_autoencoder(side)

        front_pred = (front_logits > 0.5).float()
        side_pred = (side_logits > 0.5).float()

        trainer.logger.experiment.log(
            {
                "input front image": [wandb.Image(i[0]) for i in front],
                "output front image": [wandb.Image(o[0]) for o in front_pred],
                "input side image": [wandb.Image(i[0]) for i in side],
                "output side image": [wandb.Image(o[0]) for o in side_pred],
            }
        )


class BetaPredictionLogger(Callback):
    def __init__(self, val_samples, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.beta2verts = Beta2Verts(batch_size=batch_size, device=device)

    def on_validation_epoch_end(self, trainer, pl_module):
        front, side, height, betas = (
            self.val_samples["front_image"],
            self.val_samples["side_image"],
            self.val_samples["height"],
            self.val_samples["betas"],
        )
        image = torch.cat((front, side), dim=1).to(pl_module.device)
        height = height.to(pl_module.device)
        logits = pl_module(image, height)

        target_verts = self.beta2verts.beta2verts(betas)
        pred_verts = self.beta2verts.beta2verts(logits)
        trainer.logger.experiment.log(
            {
                "target shape": [
                    wandb.Object3D(target_vert) for target_vert in target_verts
                ],
                "output shape": [wandb.Object3D(pred_vert) for pred_vert in pred_verts],
            }
        )


class MeasurementsLogger(Callback):
    def __init__(self, val_samples, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.val_samples = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        front, side, height, meas = (
            self.val_samples["front"],
            self.val_samples["side"],
            self.val_samples["height"],
            self.val_samples["meas"],
        )
        front, side = front.to(pl_module.device), side.to(pl_module.device)
        height = height.to(pl_module.device)
        logits = pl_module(front, side, height)

        pred_meas = pd.DataFrame(logits.cpu().numpy())
        pred_meas.columns = config.MEASUREMENTS_ORDER
        pred_meas["idx"] = self.val_samples["idx"]

        target_meas = pd.DataFrame(meas.cpu().numpy())
        target_meas.columns = config.MEASUREMENTS_ORDER
        target_meas["idx"] = self.val_samples["idx"]

        trainer.logger.experiment.log(
            {
                "predict measurements": wandb.Table(dataframe=pred_meas),
                "target measurements": wandb.Table(dataframe=target_meas),
            }
        )


class RealDataPredictLogger(Callback):
    def __init__(self, real_user_dir):
        user_dirs = glob.glob(os.path.join(real_user_dir, "*"))
        len_user = len(user_dirs)
        self.front_tensor = torch.zeros((len_user, 1, 512, 512))
        self.side_tensor = torch.zeros((len_user, 1, 512, 512))
        self.height_tensor = torch.zeros((len_user, 1))
        self.target = np.zeros((len_user, 8))

        transform = Compose(
            [
                ToTensor(),
                BinTensor(threshold=0.9),
                Lambda(crop_true),
                Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                Lambda(morphology),
            ]
        )

        for idx, user_dir in enumerate(glob.glob(os.path.join(real_user_dir, "*"))):
            front = transform(
                cv2.imread(os.path.join(user_dir, "front.jpg"), cv2.IMREAD_GRAYSCALE)
            )
            side = transform(
                cv2.imread(os.path.join(user_dir, "side.jpg"), cv2.IMREAD_GRAYSCALE)
            )
            with open(os.path.join(user_dir, "meas.txt"), "r") as f:
                meas = np.array(list(map(float, f.readline().strip().split())))

            self.target[idx, :] = meas
            self.front_tensor[idx, :, :, :] = front
            self.side_tensor[idx, :, :, :] = side
            self.height_tensor[idx] = torch.tensor(meas[0])

    def on_validation_epoch_end(self, trainer, pl_module):
        front, side = self.front_tensor.to(pl_module.device), self.side_tensor.to(
            pl_module.device
        )
        height = self.height_tensor.to(pl_module.device)
        logits = pl_module(front, side, height)

        pred_meas = pd.DataFrame(logits.cpu().numpy())
        pred_meas.columns = config.MEASUREMENTS_ORDER

        target_meas = pd.DataFrame(self.target)
        target_meas.columns = config.MEASUREMENTS_ORDER

        trainer.logger.experiment.log(
            {
                "real predict measurements": wandb.Table(dataframe=pred_meas),
                "real target measurements": wandb.Table(dataframe=target_meas),
            }
        )
