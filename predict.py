import json
from PIL import Image
import numpy as np
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
                        ])

    front_pred_alpha, front_pred_mask = inference.single_inference(model, front_image)
    side_pred_alpha, side_pred_mask = inference.single_inference(model, side_image)

    front = torch.unsqueeze(transform(front_pred_alpha), dim=0)
    side = torch.unsqueeze(transform(side_pred_alpha), dim=0)
    height = torch.unsqueeze(torch.tensor(height), dim=0)

    with torch.no_grad():
        pred = model_cnn(front, side, height).numpy()

    meas = {name: pred_ for name, pred_ in zip(config.MEASUREMENTS_ORDER, pred[0])}
    
    return json.dumps(str(meas))


if __name__ == "__main__":
    front_bin_image = Image.open("/home/shin/Documents/image_data/front.jpg")
    side_bin_image = Image.open("/home/shin/Documents/image_data/side.jpg")
    meas = predict(front_bin_image, side_bin_image, 181)
    print(json.loads(meas))
