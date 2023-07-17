import sys

sys.path.append("/home/shin/VScodeProjects/fittering-ML")
import os
from glob import glob
import json
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize
from torchvision.transforms import functional as F

import config

from data.preprocessing import *


class BinaryImageMeasDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.transform = transform

        self.image_dir = os.path.join(data_dir, "images")
        json_dir = os.path.join(data_dir, "json")
        self.json_pathes = glob(os.path.join(json_dir, "*.json"))

    def __len__(self):
        return len(self.json_pathes)

    def __getitem__(self, idx):
        json_path = self.json_pathes[idx]
        with open(json_path, "r") as f:
            json_data = json.load(f)
        front_pil = Image.open(
            os.path.join(self.image_dir, f'front_{json_data["idx"]}.jpg')
        )
        side_pil = Image.open(
            os.path.join(self.image_dir, f'side_{json_data["idx"]}.jpg')
        )
        height = torch.tensor(json_data["height"])
        measurement_lst = torch.tensor(
            [json_data[mea_name] for mea_name in config.MEASUREMENTS_ORDER]
        )

        front_arr = np.array(front_pil)
        side_arr = np.array(side_pil)

        if self.transform:
            front = self.transform(front_arr)
            side = self.transform(side_arr)

        return {
            "front": front,
            "side": side,
            "height": height,
            "meas": measurement_lst,
            "idx": json_data["idx"],
        }


class AihubDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.transform = transform
        self.json_pathes = glob(os.path.join(data_dir, "*/*.json"))

    def __len__(self):
        return len(self.json_pathes)

    @staticmethod
    def preprocessing(arr):
        arr = np.fliplr(arr)
        arr = np.rot90(arr)
        return arr

    def __getitem__(self, idx):
        json_path = self.json_pathes[idx]
        with open(json_path, "r") as f:
            json_data = json.load(f)
        front_dir = json_data["input"]["front"]
        side_dir = json_data["input"]["side"]

        front_arr = plt.imread(front_dir)
        front_arr = AihubDataset.preprocessing(front_arr)
        side_arr = plt.imread(side_dir)
        side_arr = AihubDataset.preprocessing(side_arr)

        if self.transform:
            front, side = self.transform(front_arr), self.transform(side_arr)

        return front, side


if __name__ == "__main__":
    # dataset = BinaryImageMeasDataset(
    #     data_dir=config.GEN_TRAIN_DIR,
    #     transform=transforms.Compose(
    #         [
    #             ToTensor(),
    #             BinTensor(threshold=0.5),
    #             Lambda(crop_true),
    #             Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
    #         ]
    #     ),
    # )
    # data_dict = dataset.__getitem__(0)
    # print(data_dict["front"].float().dtype)
    dataset = AihubDataset(
        "/home/shin/VScodeProjects/fittering-ML/data/source/aihub", transform=ToTensor()
    )

    front, side = dataset.__getitem__(0)
    print(front.shape, side.shape)
