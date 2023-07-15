import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn

from opensrc.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from opensrc.SemanticGuidedHumanMatting import utils
from opensrc.SemanticGuidedHumanMatting import inference

import config
import time

t = time.time()
# model load
model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).eval()
model.load_state_dict(torch.load(config.SEGMODEL_PATH, map_location=torch.device('cpu')))

# image, height load
user_id = '1'
secret_user_dir = os.path.join(config.SECRET_USER_DIR, user_id)
front = Image.open(os.path.join(secret_user_dir, 'front.jpg'))
side = Image.open(os.path.join(secret_user_dir, 'side.jpg'))
with open(os.path.join(secret_user_dir, 'meas.txt'), 'r') as f:
    meas = f.readline().strip()


# 예측 과정
front_pred_alpha, front_pred_mask = inference.single_inference(model, front)
side_pred_alpha, side_pred_mask = inference.single_inference(model, side)

front_pred = front_pred_mask[:, :, 0]
side_pred = side_pred_mask[:, :, 0]
# side_pred = np.fliplr(side_pred)


# 예측 저장
real_user_dir = os.path.join(config.REAL_USER_DIR, user_id)
if os.path.exists(real_user_dir):
    shutil.rmtree(real_user_dir)
os.mkdir(real_user_dir)

plt.imsave(os.path.join(real_user_dir, 'front.jpg'), front_pred, cmap='gray')
plt.imsave(os.path.join(real_user_dir, 'side.jpg'), side_pred, cmap='gray')
with open(os.path.join(real_user_dir, 'meas.txt'), 'w') as f:
    f.write(meas)

print(time.time() - t)