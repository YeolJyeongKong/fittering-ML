import os
import json
import numpy as np
import torch
import paths.config as config
from tqdm import tqdm
from PIL import Image
from data.data_augmentation.augmentation import AugmentBetasCam
from src.data.datamodule import DataModule
from data.dataset import BinaryImageMeasDataset, AihubDataset
from torch.utils.data import DataLoader
from demo.encoder_inference import InferenceEncoder


class GenDataset:
    def __init__(
        self,
        gen_param,
        data_size=1e5,
        train_ratio=0.8,
        ord_data_path=config.ORD_DATA_PATH,
        gen_data_dir=config.GEN_DATA_DIR,
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


class GenEncodeDataset:
    def __init__(
        self,
        encode_data_dir,
        dataset_mode="aihub",
        batch_size=16,
        device=torch.device("cuda"),
    ):
        self.encode_data_dir = encode_data_dir
        GenEncodeDataset._check_make_dir(self.encode_data_dir)
        self.batch_size = batch_size
        self.device = device

        dm = DataModule(batch_size=batch_size, dataset_mode=dataset_mode)
        transform = dm.transform
        self.batch_size = batch_size
        train_dataset = AihubDataset(
            data_dir=os.path.join(config.AIHUB_DATA_DIR, "train"), transform=transform
        )
        test_dataset = AihubDataset(
            data_dir=os.path.join(config.AIHUB_DATA_DIR, "test"), transform=transform
        )

        self.train_len = len(train_dataset)
        self.test_len = len(test_dataset)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        self.inference_encoder = InferenceEncoder(device=torch.device("cuda"))

    @staticmethod
    def _check_make_dir(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def generate_(self, len_dataset, dataloader, name):
        total_input_arr = np.empty((len_dataset, 515))
        total_output_arr = np.empty((len_dataset, 8))

        for idx, batch in tqdm(
            enumerate(dataloader),
            desc=f"generate {name}",
            total=(len_dataset // self.batch_size),
        ):
            pred = self.inference_encoder.inference(
                batch["front"].to(self.device), batch["side"].to(self.device)
            )
            input_tensor = torch.cat(
                [
                    pred.cpu(),
                    batch["height"].unsqueeze(dim=1),
                    batch["weight"].unsqueeze(dim=1),
                    batch["sex"].unsqueeze(dim=1),
                ],
                dim=1,
            )
            input_arr = input_tensor.cpu().numpy()
            total_input_arr[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ] = input_arr

            output_arr = batch["meas"].cpu().numpy()
            total_output_arr[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ] = output_arr

        return total_input_arr, total_output_arr

    def generate(self):
        train_x, train_y = self.generate_(
            self.train_len, self.train_dataloader, "train"
        )
        test_x, test_y = self.generate_(self.test_len, self.test_dataloader, "test")
        np.savez(os.path.join(self.encode_data_dir, "train"), x=train_x, y=train_y)
        np.savez(os.path.join(self.encode_data_dir, "test"), x=test_x, y=test_y)


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
    gen_dataset = GenEncodeDataset(
        encode_data_dir=os.path.join(config.AIHUB_DATA_DIR, "encode"),
        dataset_mode="aihub",
        batch_size=32,
    )
    gen_dataset.generate()
