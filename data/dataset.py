import sys
sys.path.append('/home/shin/VScodeProjects/fittering-ML')
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from data.augmentation import AugmentBetasCam
from data.preprocessing import *

class BinaryImageBetaDataset(Dataset):
    def __init__(self, ord_data_path, data_range=None, transform=None, augment=None) -> None:
        super().__init__()
        self.transform = transform
        self.augment = augment

        data = np.load(ord_data_path)
        if data_range:
            self.ord_shapes = data['shapes'][data_range[0]:data_range[1]]
        else:
            self.ord_shapes = data['shapes']

    def __len__(self):
        return len(self.ord_shapes)

    def __getitem__(self, idx):
        ord_shape = self.ord_shapes[idx][np.newaxis, ...]

        assert ord_shape.shape == (1, 10), \
            f"shape of ord_shape: {ord_shape.shape} | expected shape:{(1, 10)}"
        
        front_image, side_image, height, betas = self.augment.render_image(ord_shape)

        if self.transform:
            front_image = self.transform(front_image)
            side_image = self.transform(side_image)

        return {'front_image': front_image, 
                'side_image': side_image, 
                'height': height, 
                'betas': betas[0]}

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
