import os
os.environ["PL_DEVICE"] = "cpu"
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from lightning_modules import CNNForwardModule
import torchvision.transforms.functional as F

from custom_utils.predict import Beta2Measurements
from preprocess import transform

import sys
# sys.path.append("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting")
from opensrc.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from opensrc.SemanticGuidedHumanMatting import utils
from opensrc.SemanticGuidedHumanMatting import inference

def predict(front_image, side_image, height, 
            model_cnn_ckpt_path='/home/shin/VScodeProjects/fittering-ML/wandb-lightning/ktb7gxc2/checkpoints/epoch=19-step=10000.ckpt', 
            segmentation_ckpt_path='/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth'):
    model_cnn = CNNForwardModule(device=torch.device('cpu'))
    model_cnn.load_state_dict(torch.load(model_cnn_ckpt_path, map_location=torch.device('cpu')))
    model_cnn.eval()

    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).eval()
    model.load_state_dict(torch.load(segmentation_ckpt_path, map_location=torch.device('cpu')))

    front_pred_alpha, front_pred_mask = inference.single_inference(model, front_image)
    side_pred_alpha, side_pred_mask = inference.single_inference(model, side_image)

    front_bin_image = torch.tensor(front_pred_alpha[np.newaxis, ...])
    side_bin_image = torch.tensor(side_pred_alpha[np.newaxis, ...])
    front = transform(front_bin_image)
    side = transform(side_bin_image)

    image = torch.cat((front, side), dim=0).reshape(1, 2, 512, 512)

    with torch.no_grad():
        pred = model_cnn(image, torch.tensor(height))

    beta2meas = Beta2Measurements(device=torch.device("cpu"))
    meas = beta2meas.predict(pred)
    
    return meas


if __name__ == "__main__":
    front_bin_image = Image.open("./images/front.jpg")
    side_bin_image = Image.open("./images/side.jpg")


    meas = predict(front_bin_image, side_bin_image, 181,
                   model_cnn_ckpt_path='./weights/CNNForward.pth', 
                   segmentation_ckpt_path='./weights/SGHM-ResNet50.pth')
    print(meas)
