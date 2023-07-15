import os
import json
import numpy as np
import torch
import config
from tqdm import tqdm
from PIL import Image
from data.augmentation import AugmentBetasCam


class GenDataset:
    def __init__(self, gen_param, data_size=1e5, train_ratio=0.8, 
                 ord_data_path=config.ORD_DATA_PATH, 
                 gen_data_dir=config.GEN_DATA_DIR,
                 device=torch.device('cuda')):
        self.augment = AugmentBetasCam(device=torch.device('cuda'), **gen_params)
        ord_data = np.load(ord_data_path)['shapes']
        ord_data = ord_data[np.random.choice(ord_data.shape[0], size=int(data_size), replace=False)]
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

        gen_train_dir = os.path.join(gen_data_dir, 'train')
        gen_test_dir = os.path.join(gen_data_dir, 'test')
        GenDataset._check_make_dir(gen_train_dir)
        GenDataset._check_make_dir(gen_test_dir)

        self.gen_train_image_dir = os.path.join(gen_train_dir, 'images')
        self.gen_train_json_dir = os.path.join(gen_train_dir, 'json')

        self.gen_test_image_dir = os.path.join(gen_test_dir, 'images')
        self.gen_test_json_dir = os.path.join(gen_test_dir, 'json')

        GenDataset._check_make_dir(self.gen_train_image_dir)
        GenDataset._check_make_dir(self.gen_train_json_dir)
        GenDataset._check_make_dir(self.gen_test_image_dir)
        GenDataset._check_make_dir(self.gen_test_json_dir)
            
        
    def generate(self):
        for i in tqdm(range(len(self.train_data)), desc="generate train data"):
            shape = self.train_data[i:i+1]
            front_image, side_image, measurements = self.augment.generate(shape)
            measurements['idx'] = i
            with open(os.path.join(self.gen_train_json_dir, f'{i}.json'), 'w') as fp:
                json.dump(measurements, fp)

            Image.fromarray(front_image[0].cpu().numpy()*255).convert('L').save(os.path.join(self.gen_train_image_dir, f"front_{i}.jpg"))
            Image.fromarray(side_image[0].cpu().numpy()*255).convert('L').save(os.path.join(self.gen_train_image_dir, f"side_{i}.jpg"))
        
        for i in tqdm(range(len(self.test_data)), desc="generate test data"):
            shape = self.test_data[i:i+1]
            front_image, side_image, measurements = self.augment.generate(shape)
            measurements['idx'] = i
            with open(os.path.join(self.gen_test_json_dir, f'{i}.json'), 'w') as fp:
                json.dump(measurements, fp)
                
            Image.fromarray(front_image[0].cpu().numpy()*255).convert('L').save(os.path.join(self.gen_test_image_dir, f"front_{i}.jpg"))
            Image.fromarray(side_image[0].cpu().numpy()*255).convert('L').save(os.path.join(self.gen_test_image_dir, f"side_{i}.jpg"))
            


if __name__ == "__main__":
    gen_params = {
           'pose_std': 0.01,
            'betas_std_vect': 2.0, 
            'K_std': 1, 
            't_xy_std': 0.1, 
            't_z_range': [-0.5, 0.5], 
            'theta_std': 3,
    }
    gen_dataset = GenDataset(gen_params)
    gen_dataset.generate()