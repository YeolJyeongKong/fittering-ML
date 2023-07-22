import os
import json
import time
from PIL import Image
import pickle
import numpy as np
import cv2
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize, Compose
from torchvision.transforms import functional as F

from src.models.lightning_modules import CNNForwardModule
from src.data.datamodule import DataModule
from src.data.preprocessing import *
from src.inference.encoder_inference import InferenceEncoder
from src.inference.segment_inference import InferenceSegment

from extras import paths, constant


class Inference:
    def __init__(
        self,
        cnnmodel_path=paths.CNNMODEL_PATH,
        segmodel_path=paths.SEGMODEL_PATH,
        device=torch.device("cpu"),
    ):
        self.device = device

        self.model_cnn = CNNForwardModule()
        ckpt = torch.load(cnnmodel_path, map_location=device)
        self.model_cnn.load_state_dict(ckpt["state_dict"])
        self.model_cnn.eval()

        self.segment = InferenceSegment(segmodel_path=segmodel_path, device=self.device)

        self.transform = Compose(
            [
                ToTensor(),
                BinTensor(threshold=0.9),
                Lambda(crop_true),
                Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                Lambda(morphology),
            ]
        )

    def predict(self, front, side, height):
        front_mask = self.segment.predict(front)
        side_mask = self.segment.predict(side)

        front = torch.unsqueeze(self.transform(front_mask), dim=0)
        side = torch.unsqueeze(self.transform(side_mask), dim=0)
        height = torch.unsqueeze(torch.tensor(height), dim=0)

        with torch.no_grad():
            pred = self.model_cnn(front, side, height).numpy()
        meas = {name: pred_ for name, pred_ in zip(paths.MEASUREMENTS_ORDER, pred[0])}

        return json.dumps(str(meas))


class Evaluate:
    def __init__(
        self,
        segmodel_path=paths.SEGMODEL_PATH,
        front_ae_path=paths.FRONTAE_PATH,
        side_ae_path=paths.SIDEAE_PATH,
        regression_path=paths.REGRESSION_PATH,
        device=torch.device("cpu"),
    ):
        self.device = device

        self.transform = Compose(
            [
                ToTensor(),
                BinTensor(threshold=0.9),
                Lambda(crop_true),
                Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
            ]
        )

        self.segment = InferenceSegment(segmodel_path=segmodel_path, device=self.device)

        self.encoder = InferenceEncoder(front_ae_path, side_ae_path, device=self.device)

        self.regression = pickle.load(open(regression_path, "rb"))

    def predict(self, front, side, height, weight, sex):
        front_masked = self.segment.predict(front)
        side_masked = self.segment.predict(side)

        front = torch.unsqueeze(self.transform(front_masked), dim=0)
        side = torch.unsqueeze(self.transform(side_masked), dim=0)
        height = torch.tensor(height).reshape(1, 1)
        weight = torch.tensor(weight).reshape(1, 1)
        sex = torch.tensor(sex).reshape(1, 1)

        with torch.no_grad():
            z = self.encoder.inference(front, side)

        z_ = torch.cat([z, height, weight, sex], dim=1).numpy()
        y = self.regression.predict(z_)
        meas = {name: pred_ for name, pred_ in zip(constant.MEASUREMENTS_ORDER, y[0])}

        return json.dumps(str(meas))


if __name__ == "__main__":
    front_bin_image = Image.open(os.path.join(paths.SECRET_USER_DIR, "0", "front.jpg"))
    side_bin_image = Image.open(os.path.join(paths.SECRET_USER_DIR, "0", "side.jpg"))
    eval = Evaluate()
    meas = eval.predict(front_bin_image, side_bin_image, 181, 65, 1)  # 남자는 1 여자는 0
    # print(time.time() - t)
    print(json.loads(meas))
