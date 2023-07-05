from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from lightning_modules import CNNForwardModule
from datamodule import DataModule
import torchvision.transforms.functional as F

from utils.predict import Beta2Measurements

import sys
# sys.path.append("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting")
from opensrc.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from opensrc.SemanticGuidedHumanMatting import utils
from opensrc.SemanticGuidedHumanMatting import inference

def predict(front_image, side_image, height, 
            model_cnn_ckpt_path='/home/shin/VScodeProjects/fittering-ML/wandb-lightning/ktb7gxc2/checkpoints/epoch=19-step=10000.ckpt', 
            segmentation_ckpt_path='/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth'):
    model_cnn = CNNForwardModule(device=torch.device('cpu')).\
        load_from_checkpoint(model_cnn_ckpt_path)
    model_cnn.eval()

    model = HumanMatting(backbone='resnet50')
    model = nn.DataParallel(model).eval()
    model.load_state_dict(torch.load(segmentation_ckpt_path))

    front_pred_alpha, front_pred_mask = inference.single_inference(model, front_image)
    side_pred_alpha, side_pred_mask = inference.single_inference(model, side_image)


    dm = DataModule(batch_size=1)

    front_bin_image = torch.tensor(front_pred_alpha[np.newaxis, ...])
    side_bin_image = torch.tensor(side_pred_alpha[np.newaxis, ...])
    front = dm.transform(front_bin_image)
    side = dm.transform(side_bin_image)

    image = torch.cat((front, side), dim=0).reshape(1, 2, 512, 512)

    with torch.no_grad():
        pred = model_cnn(image, torch.tensor(height))

    beta2meas = Beta2Measurements(device=torch.device("cpu"))
    meas = beta2meas.predict(pred)
    
    return meas


if __name__ == "__main__":
    front_bin_image = Image.open("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting/image_data/IMG_4251.jpg")
    side_bin_image = Image.open("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting/image_data/IMG_4254.jpg")


    meas = predict(front_bin_image, side_bin_image, 1.81,
                   model_cnn_ckpt_path=)
    print(meas)
