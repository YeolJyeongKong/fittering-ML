"""
Example Test:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

Example Evaluation:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --gt-dir "PATH_TO_GT_ALPHA_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

"""

import argparse
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.model import HumanSegment, HumanMatting
import utils
import inference


# --------------- Main ---------------

# directory & path
ROOT_DIR = '/home/shin/VScodeProjects/fittering-ML/opensrc/SemanticGuidedHumanMatting'
weight_path = os.path.join(ROOT_DIR, 'pretrained/SGHM-ResNet50.pth')
images_dir = os.path.join(ROOT_DIR, '/home/shin/Documents/image_data')
result_dir = os.path.join(ROOT_DIR, 'result')

# save image 
save_image = True


# Load Model
model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).cuda().eval()
model.load_state_dict(torch.load(weight_path))
print("Load checkpoint successfully ...")


# Load Images
image_list = sorted([*glob.glob(os.path.join(images_dir, '*.jpg'), recursive=True),
                    *glob.glob(os.path.join(images_dir, '*.png'), recursive=True)])

num_image = len(image_list)
print("Find ", num_image, " images")

metric_mad = utils.MetricMAD()
metric_mse = utils.MetricMSE()
metric_grad = utils.MetricGRAD()
metric_conn = utils.MetricCONN()
metric_iou = utils.MetricIOU()

mean_mad = 0.0
mean_mse = 0.0
mean_grad = 0.0
mean_conn = 0.0
mean_iou = 0.0

# Process 
for i in range(num_image):
    image_path = image_list[i]
    image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]
    print(i, '/', num_image, image_name)

    with Image.open(image_path) as img:
        img = img.convert("RGB")

    # inference
    pred_alpha, pred_mask = inference.single_inference(model, img)
    plt.imsave(os.path.join(result_dir, image_name + '.jpg'), np.squeeze(pred_mask), cmap='gray')
    # Image.fromarray(np.squeeze(pred_mask*255)).convert('L').save(os.path.join(result_dir, image_name + '.jpg'))

    # save results
    # if save_image:
    #     if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
    #     save_path = result_dir + '/' + image_name + '.png'
    #     Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)
    # plt.imshow(pred_alpha, cmap='gray')
    # plt.show()

print("Total mean mad ", mean_mad/num_image, " mean mse ", mean_mse/num_image, " mean grad ", \
    mean_grad/num_image, " mean conn ", mean_conn/num_image, " mean iou ", mean_iou/num_image)
