import os
import json
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.datamodule import DataModule
from src.data.preprocess import *
from data.dataset import BinaryImageMeasDataset, AihubDataset, AihubOriDataset
from extras import paths, constant


class GenDataset:
    def __init__(
        self,
        gen_params,
        data_size=1e5,
        train_ratio=0.8,
        ord_data_path=paths.SYNTHETIC_ORIGINAL_DIR,
        gen_data_dir=paths.SYNTHETIC_DATA_DIR,
        device=torch.device("cuda"),
    ):
        self.augment = AugmentBetasCam(device=torch.device("cuda"), **gen_params)
        ord_data = np.load(ord_data_path)["shapes"]
        ord_data = ord_data[
            np.random.choice(ord_data.shape[0], size=int(data_size), replace=False)
        ]
        split_idx = int(data_size * train_ratio)
        self.train_data = ord_data[:split_idx]
        self.test_data = ord_data[split_idx:]
        self._make_dir(gen_data_dir)

    @staticmethod
    def _check_make_dir(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def _make_dir(self, gen_data_dir):
        GenDataset._check_make_dir(gen_data_dir)

        gen_train_dir = os.path.join(gen_data_dir, "train")
        gen_test_dir = os.path.join(gen_data_dir, "test")
        GenDataset._check_make_dir(gen_train_dir)
        GenDataset._check_make_dir(gen_test_dir)

        self.gen_train_image_dir = os.path.join(gen_train_dir, "images")
        self.gen_train_json_dir = os.path.join(gen_train_dir, "json")

        self.gen_test_image_dir = os.path.join(gen_test_dir, "images")
        self.gen_test_json_dir = os.path.join(gen_test_dir, "json")

        GenDataset._check_make_dir(self.gen_train_image_dir)
        GenDataset._check_make_dir(self.gen_train_json_dir)
        GenDataset._check_make_dir(self.gen_test_image_dir)
        GenDataset._check_make_dir(self.gen_test_json_dir)

    def generate(self):
        for i in tqdm(range(len(self.train_data)), desc="generate train data"):
            shape = self.train_data[i : i + 1]
            front_image, side_image, measurements = self.augment.generate(shape)
            measurements["idx"] = i
            with open(os.path.join(self.gen_train_json_dir, f"{i}.json"), "w") as fp:
                json.dump(measurements, fp)

            Image.fromarray(front_image[0].cpu().numpy() * 255).convert("L").save(
                os.path.join(self.gen_train_image_dir, f"front_{i}.jpg")
            )
            Image.fromarray(side_image[0].cpu().numpy() * 255).convert("L").save(
                os.path.join(self.gen_train_image_dir, f"side_{i}.jpg")
            )

        for i in tqdm(range(len(self.test_data)), desc="generate test data"):
            shape = self.test_data[i : i + 1]
            front_image, side_image, measurements = self.augment.generate(shape)
            measurements["idx"] = i
            with open(os.path.join(self.gen_test_json_dir, f"{i}.json"), "w") as fp:
                json.dump(measurements, fp)

            Image.fromarray(front_image[0].cpu().numpy() * 255).convert("L").save(
                os.path.join(self.gen_test_image_dir, f"front_{i}.jpg")
            )
            Image.fromarray(side_image[0].cpu().numpy() * 255).convert("L").save(
                os.path.join(self.gen_test_image_dir, f"side_{i}.jpg")
            )


class GenSegmentation:
    def __init__(
        self,
        data_dir=paths.AIHUB_DATA_DIR,
        transform=None,
        seg_model=None,
        device=torch.device("cuda"),
    ) -> None:
        self.dataset = AihubOriDataset(data_dir, transform=transform)

        self.seg_model = seg_model
        self.seg_model.eval()
        self.seg_model.load_state_dict(
            torch.load(paths.SEGMODEL_PATH, map_location=torch.device("cuda"))
        )
        self.device = device

    @staticmethod
    def save_image_(idx, front_side_masked, shape, paths):
        image = F.resize(front_side_masked[idx].cpu(), size=shape[idx]).numpy()[0]
        plt.imsave(paths[idx], image, cmap="gray")

    def generate(self):
        for idx in tqdm(range(len(self.dataset)), desc="generate masked dataset"):
            front_side, paths, shape = self.dataset.__getitem__(idx)
            with torch.no_grad():
                front_side_masked = self.seg_model(front_side.to(self.device))
            for i in range(2):
                GenSegmentation.save_image_(i, front_side_masked, shape, paths)


if __name__ == "__main__":
    # gen_params = {
    #        'pose_std': 0.01,
    #         'betas_std_vect': 2.0,
    #         'K_std': 1,
    #         't_xy_std': 0.1,
    #         't_z_range': [-0.5, 0.5],
    #         'theta_std': 3,
    # }
    # gen_dataset = GenDataset(gen_params)
    # gen_dataset.generate()

    model_yaml_path = "/home/shin/VScodeProjects/fittering-ML/configs/model/v1.yaml"
    model_cfg = OmegaConf.load(model_yaml_path)
    seg_model = hydra.utils.instantiate(model_cfg.segment)

    preprocess_yaml_path = (
        "/home/shin/VScodeProjects/fittering-ML/configs/preprocess/v1.yaml"
    )
    preprocess_cfg = OmegaConf.load(preprocess_yaml_path)
    preprocess = hydra.utils.instantiate(preprocess_cfg.segment)

    gen = GenSegmentation(transform=preprocess, seg_model=seg_model)
    gen.generate()
