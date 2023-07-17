import os
import json
from PIL import Image
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from lightning_modules import CNNForwardModule
from datamodule import DataModule

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize, Compose
from torchvision.transforms import functional as F
from data.preprocessing import *
from encoder_inference import InferenceEncoder

import sys
sys.path.append("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting")
from opensrc.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from opensrc.SemanticGuidedHumanMatting import utils
from opensrc.SemanticGuidedHumanMatting import inference

import config
import time


class Inference:
    def __init__(self, cnnmodel_path=config.CNNMODEL_PATH, segmodel_path=config.SEGMODEL_PATH):
        device = torch.device('cpu')

        self.model_cnn = CNNForwardModule()
        ckpt = torch.load(cnnmodel_path, map_location=device)
        self.model_cnn.load_state_dict(ckpt['state_dict'])
        self.model_cnn.eval()

        model = HumanMatting(backbone='resnet50')
        self.model = nn.DataParallel(model).eval()
        self.model.load_state_dict(torch.load(segmodel_path, map_location=device))
        
        self.transform = Compose([
                            ToTensor(),
                            BinTensor(threshold=0.9), 
                            Lambda(crop_true), 
                            Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                            Lambda(morphology),
                        ])

    def predict(self, front, side, height):
        front_pred_alpha, front_pred_mask = inference.single_inference(self.model, front)
        side_pred_alpha, side_pred_mask = inference.single_inference(self.model, side)

        front = torch.unsqueeze(self.transform(front_pred_mask), dim=0)
        side = torch.unsqueeze(self.transform(side_pred_mask), dim=0)
        height = torch.unsqueeze(torch.tensor(height), dim=0)

        with torch.no_grad():
            pred = self.model_cnn(front, side, height).numpy()
        meas = {name: pred_ for name, pred_ in zip(config.MEASUREMENTS_ORDER, pred[0])}
        
        return json.dumps(str(meas))


class Inferencev2:
    def __init__(self, segmodel_path=config.SEGMODEL_PATH):
        self.device = torch.device('cpu')

        self.transform = Compose([
                            ToTensor(),
                            BinTensor(threshold=0.9), 
                            Lambda(crop_true), 
                            Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                            Lambda(morphology),
                        ])

        seg_model = HumanMatting(backbone='resnet50')
        self.seg_model = nn.DataParallel(seg_model).eval()
        self.seg_model.load_state_dict(torch.load(segmodel_path, map_location=self.device))

        self.encoder = InferenceEncoder(device=self.device)

        self.regression = pickle.load(open(config.REGRESSION_PATH, "rb"))


    def predict(self, front, side, height):
        front_pred_alpha, front_pred_mask = inference.single_inference(self.seg_model, front, self.device)
        side_pred_alpha, side_pred_mask = inference.single_inference(self.seg_model, side, self.device)

        front = torch.unsqueeze(self.transform(front_pred_mask), dim=0)
        side = torch.unsqueeze(self.transform(side_pred_mask), dim=0)
        height = torch.tensor(height).reshape(1, 1)

        with torch.no_grad():
            z = self.encoder.inference(front, side)

        z_ = torch.cat([z, height], dim=1).numpy()
        y = self.regression.predict(z_)
        meas = {name: pred_ for name, pred_ in zip(config.MEASUREMENTS_ORDER, y[0])}
        
        return json.dumps(str(meas))



if __name__ == "__main__":
    front_bin_image = Image.open(os.path.join(config.SECRET_USER_DIR, '1', "front.jpg"))
    side_bin_image = Image.open(os.path.join(config.SECRET_USER_DIR, '1', "side.jpg"))
    inf = Inferencev2()
    meas = inf.predict(front_bin_image, side_bin_image, 165)
    # print(time.time() - t)
    print(json.loads(meas))
