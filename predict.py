import os
import json
from PIL import Image
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


def predict(front_image, side_image, height):
    device = torch.device('cpu')

    model_cnn = CNNForwardModule()
    ckpt = torch.load(config.CNNMODEL_PATH, map_location=device)
    model_cnn.load_state_dict(ckpt['state_dict'])
    model_cnn.eval()

    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).eval()
    model.load_state_dict(torch.load(config.SEGMODEL_PATH, map_location=device))

    transform = Compose([
                            ToTensor(),
                            BinTensor(threshold=0.9), 
                            Lambda(crop_true), 
                            Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                            Lambda(morphology),
                        ])

    front_pred_alpha, front_pred_mask = inference.single_inference(model, front_image)
    side_pred_alpha, side_pred_mask = inference.single_inference(model, side_image)

    front = torch.unsqueeze(transform(front_pred_mask), dim=0)
    side = torch.unsqueeze(transform(side_pred_mask), dim=0)
    height = torch.unsqueeze(torch.tensor(height), dim=0)

    with torch.no_grad():
        pred = model_cnn(front, side, height).numpy()
    meas = {name: pred_ for name, pred_ in zip(config.MEASUREMENTS_ORDER, pred[0])}
    
    return json.dumps(str(meas))


def predict_time(front_image, side_image, height):
    device = torch.device('cpu')

    ta = time.time()
    model_cnn = CNNForwardModule()
    ckpt = torch.load(config.CNNMODEL_PATH, map_location=device)
    model_cnn.load_state_dict(ckpt['state_dict'])
    model_cnn.eval()

    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).eval()
    model.load_state_dict(torch.load(config.SEGMODEL_PATH, map_location=device))
    tb = time.time()

    transform = Compose([
                            ToTensor(),
                            BinTensor(threshold=0.9), 
                            Lambda(crop_true), 
                            Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
                            Lambda(morphology),
                        ])

    front_pred_alpha, front_pred_mask = inference.single_inference(model, front_image)
    side_pred_alpha, side_pred_mask = inference.single_inference(model, side_image)
    tc = time.time()

    front = torch.unsqueeze(transform(front_pred_mask), dim=0)
    side = torch.unsqueeze(transform(side_pred_mask), dim=0)
    height = torch.unsqueeze(torch.tensor(height), dim=0)

    with torch.no_grad():
        pred = model_cnn(front, side, height).numpy()
    td = time.time()
    meas = {name: pred_ for name, pred_ in zip(config.MEASUREMENTS_ORDER, pred[0])}
    
    return ta, tb, tc, td
    # return json.dumps(str(meas))
    # return "success"


if __name__ == "__main__":
    front_bin_image = Image.open(os.path.join(config.SECRET_USER_DIR, '0', "front.jpg"))
    side_bin_image = Image.open(os.path.join(config.SECRET_USER_DIR, '0', "front.jpg"))
    inf = Inference(cnnmodel_path='./model_weights/epoch=19-step=160000.ckpt')
    meas = inf.predict(front_bin_image, side_bin_image, 181)
    # print(time.time() - t)
    print(json.loads(meas))
