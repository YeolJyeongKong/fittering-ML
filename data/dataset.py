import sys
sys.path.append('/home/shin/VScodeProjects/fittering-ML')
import os
from glob import glob
import json
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

import config

from data.augmentation import AugmentBetasCam
from data.preprocessing import *

class BinaryImageBetaDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=None) -> None:
        super().__init__()
        self.transform = transform
        self.augment = augment

        self.image_dir = os.path.join(data_dir, 'images')
        json_dir = os.path.join(data_dir, 'json')
        self.json_pathes = glob(os.path.join(json_dir, '*.json'))

    def __len__(self):
        return len(self.json_pathes)

    def __getitem__(self, idx):
        json_path = self.json_pathes[idx]
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        front_pil = Image.open(os.path.join(self.image_dir, f'front_{json_data["idx"]}.jpg'))
        side_pil = Image.open(os.path.join(self.image_dir, f'side_{json_data["idx"]}.jpg'))
        height = torch.tensor(json_data['height'])
        measurement_lst = torch.tensor([json_data[mea_name] for mea_name in config.MEASUREMENTS_ORDER])

        if self.transform:
            front_image = self.transform(front_image)
            side_image = self.transform(side_image)

        return {'front_image': front_image, 
                'side_image': side_image, 
                'height': height, 
                'meas': measurement_lst}

if __name__ == "__main__":
    os.chdir("/home/shin/VScodeProjects/fittering-ML")
    data_transforms = transforms.Compose([
        transforms.Lambda(crop_true),
        transforms.ToPILImage(),
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
        transforms.Lambda(convert_multiclass_to_binary_labels_torch)
    ])
    dataset = BinaryImageBetaDataset(ord_data_path="/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/data/amass_up3d_3dpw_train.npz", 
                                     augment=AugmentBetasCam(device=torch.device('cuda'), t_z_range=[0, 0], t_xy_std=0), 
                                     transform=data_transforms)
    data = dataset.__getitem__(0)
    print(data['height'])
    print(data['betas'])
    fig, axes = plt.subplots(1, 2)
    # front_image = crop_resize_binary_image(data['front_image'])
    axes[0].imshow(data["front_image"][0].cpu().numpy(), cmap='gray')
    axes[1].imshow(data['side_image'][0].cpu().numpy(), cmap='gray')
    plt.show()
