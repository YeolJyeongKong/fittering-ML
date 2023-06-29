import torch
import os
import numpy as np

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplpytorch.display_utils import display_model

class Beta2Verts:
    def __init__(self, gender='neural', batch_size=None, device=None,
                 smpl_model_path='/home/shin/VScodeProjects/fittering-ML/data/additional/smpl/SMPL_NEUTRAL.pkl', 
                 smpl_mean_params_path='/home/shin/VScodeProjects/fittering-ML/data/additional/neutral_smpl_mean_params_6dpose.npz'):

        self.smpl_layer = SMPL_Layer(center_idx=0, gender=gender, model_path=smpl_model_path).cuda()
        neutral_mean = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/additional/neutral_smpl_mean_params_6dpose.npz", mmap_mode='r')
        # self.mean_pose = torch.tensor(neutral_mean['shape']).view(1, -1).repeat(batch_size, 1)
        self.mean_pose = torch.zeros((batch_size, 72)).cuda()

    def beta2verts(self, betas):
        verts, Jtr = self.smpl_layer(self.mean_pose, th_betas=betas)
        verts = verts.cpu().numpy()
        return verts
        