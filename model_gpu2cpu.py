from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from lightning_modules import CNNForwardModule
from datamodule import DataModule
import torchvision.transforms.functional as F

from utils.predict import Beta2Measurements

import sys
sys.path.append("/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting")
from opensrc.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from opensrc.SemanticGuidedHumanMatting import utils
from opensrc.SemanticGuidedHumanMatting import inference

model_cnn = CNNForwardModule(device=torch.device('cpu')).\
    load_from_checkpoint("/home/shin/VScodeProjects/fittering-ML/fittering-ML-inference/weights/CNNForward.ckpt")

torch.save(model_cnn.state_dict(), '/home/shin/VScodeProjects/fittering-ML/fittering-ML-inference/weights/CNNForward.pth')

