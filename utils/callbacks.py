import pandas as pd
import wandb
import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

from utils.visualize import Beta2Verts
from utils.predict import Beta2Measurements

import config

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples):
        super().__init__()
        self.val_samples = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        x = torch.cat((self.val_samples['front_image'], self.val_samples['side_image']), dim=1)
        x = x.to(pl_module.device)

        logits = pl_module.autoencoder(x)
        preds = (logits > 0.5).float()
        
        trainer.logger.experiment.log({
            "input front image": [wandb.Image(x_[0])
                                for x_ in x], 
            "input side image": [wandb.Image(x_[1]) 
                                for x_ in x],
            "output front image": [wandb.Image(pred[0]) 
                                for pred in preds],
            "output side image": [wandb.Image(pred[1]) 
                                for pred in preds],
        })

class BetaPredictionLogger(Callback):
    def __init__(self, val_samples, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.beta2verts = Beta2Verts(batch_size=batch_size, device=device)

    def on_validation_epoch_end(self, trainer, pl_module):
        front, side, height, betas =\
              self.val_samples['front_image'], self.val_samples['side_image'], \
                self.val_samples['height'], self.val_samples['betas']
        image = torch.cat((front, side), dim=1).to(pl_module.device)
        height = height.to(pl_module.device)
        logits = pl_module(image, height)

        target_verts = self.beta2verts.beta2verts(betas)
        pred_verts = self.beta2verts.beta2verts(logits)
        trainer.logger.experiment.log({
            'target shape': [wandb.Object3D(target_vert) for target_vert in target_verts], 
            'output shape': [wandb.Object3D(pred_vert) for pred_vert in pred_verts]
        })


class MeasurementsLogger(Callback):
    def __init__(self, val_samples, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.val_samples = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        front, side, height, meas =\
              self.val_samples['front'], self.val_samples['side'], \
                self.val_samples['height'], self.val_samples['meas']
        front, side = front.to(pl_module.device), side.to(pl_module.device)
        height = height.to(pl_module.device)
        logits = pl_module(front, side, height)

        pred_meas = pd.DataFrame(logits.cpu().numpy())
        pred_meas.columns = config.MEASUREMENTS_ORDER

        target_meas = pd.DataFrame(meas.cpu().numpy())
        target_meas.columns = config.MEASUREMENTS_ORDER

        trainer.logger.experiment.log({
            'predict measurements': wandb.Table(dataframe=pred_meas), 
            'target measurements': wandb.Table(dataframe=target_meas)
        })