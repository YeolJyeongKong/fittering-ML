import sys
import os
from tqdm import tqdm
from glob import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize
from torchvision.transforms import functional as F

from extras import paths, constant
from src.data.preprocess import *


class BinaryImageMeasDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.transform = transform

        self.image_dir = os.path.join(data_dir, "masked")
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
            [json_data[mea_name] for mea_name in paths.MEASUREMENTS_ORDER]
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
        self.json_pathes = glob(os.path.join(data_dir, "*.json"))

    def __len__(self):
        return len(self.json_pathes)

    def __getitem__(self, idx):
        json_path = self.json_pathes[idx]
        json_idx = json_path.split("/")[-1][:-5]
        with open(json_path, "r") as f:
            json_data = json.load(f)
        front_dir = json_data["input"]["front"]
        side_dir = json_data["input"]["side"]

        front = Image.open(front_dir).convert("L")
        front = np.array(front)
        side = Image.open(side_dir).convert("L")
        side = np.array(side)

        if self.transform:
            front, side = self.transform(front), self.transform(side)

        height = torch.tensor(json_data["input"]["height"])
        weight = torch.tensor(json_data["input"]["weight"])
        sex = torch.tensor(json_data["input"]["sex"])

        measurement_lst = torch.tensor(
            [json_data["output"][mea_name] for mea_name in constant.MEASUREMENTS_ORDER]
        )

        return {
            "front": front,
            "side": side,
            "height": height,
            "weight": weight,
            "sex": sex,
            "meas": measurement_lst,
            "idx": json_idx,
        }


class AihubOriDataset(Dataset):
    def __init__(self, aihub_data_dir=paths.AIHUB_DATA_DIR, transform=None):
        super().__init__()
        self.transform = transform
        self.json_pathes = glob(os.path.join(aihub_data_dir, "*/*.json"))

    def __len__(self):
        return len(self.json_pathes)

    def __getitem__(self, index) -> Any:
        json_path = self.json_pathes[index]
        with open(json_path, "r") as f:
            json_data = json.load(f)

        original_front_path = json_data["original_front_path"]
        original_side_path = json_data["original_side_path"]

        front = rotate(plt.imread(original_front_path))
        side = rotate(plt.imread(original_side_path))
        front_shape = front.shape[:2]
        side_shape = side.shape[:2]

        if self.transform:
            front = self.transform(front).unsqueeze(dim=0)
            side = self.transform(side).unsqueeze(dim=0)

        front_side = torch.cat((front, side), dim=0)

        return (
            front_side,
            [json_data["input"]["front"], json_data["input"]["side"]],
            [front_shape, side_shape],
        )
