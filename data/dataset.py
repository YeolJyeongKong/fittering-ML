import sys

sys.path.append("/home/shin/VScodeProjects/fittering-ML")
import os
from tqdm import tqdm
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
        self.json_pathes = glob(os.path.join(data_dir, "*.json"))

    def __len__(self):
        return len(self.json_pathes)

    @staticmethod
    def preprocessing(arr):
        arr = np.array(arr)
        arr = np.mean(arr, axis=-1) / 255.0
        return arr

    def __getitem__(self, idx):
        json_path = self.json_pathes[idx]
        json_idx = json_path.split("/")[-1][:-5]
        with open(json_path, "r") as f:
            json_data = json.load(f)
        front_dir = json_data["input"]["front"]
        side_dir = json_data["input"]["side"]

        front = Image.open(front_dir)
        front = AihubDataset.preprocessing(front)
        side = Image.open(side_dir)
        side = AihubDataset.preprocessing(side)

        if self.transform:
            front, side = self.transform(front).float(), self.transform(side).float()

        height = torch.tensor(json_data["input"]["height"])
        weight = torch.tensor(json_data["input"]["weight"])
        sex = torch.tensor(json_data["input"]["sex"])

        measurement_lst = torch.tensor(
            [json_data["output"][mea_name] for mea_name in config.MEASUREMENTS_ORDER]
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
    # dataset = AihubDataset(
    #     "/home/shin/VScodeProjects/fittering-ML/data/source/aihub",
    #     transform = transforms.Compose([
    #             ToTensor(),
    #             BinTensor(threshold=0.5),
    #             Lambda(crop_true),
    #             Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
    #         ])
    # )
    transform = transforms.Compose(
        [
            ToTensor(),
            BinTensor(threshold=0.5),
            Lambda(crop_true),
            Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
        ]
    )
    dataset = AihubDataset(
        data_dir=os.path.join(config.AIHUB_DATA_DIR, "train"), transform=transform
    )
    # data = dataset.__getitem__(0)
    # plt.imsave("sample.jpg", data["front"][0], cmap="gray")
    # print()

    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        meas = data["meas"].numpy()
        if np.isnan(meas).sum() > 0:
            print(data["idx"])

    # img = plt.imread('/media/shin/T7/dataset/data/training/image_csv/F010/Image/01_06_F010_11_mask.jpg')
    # img = np.mean(img, axis=-1) / 255.
    # transform = transforms.Compose([
    #             ToTensor(),
    #             BinTensor(threshold=0.5),
    #             Lambda(crop_true),
    #             Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
    #         ])
    # plt.imshow(transform(img)[0], cmap='gray')
    # plt.show()
