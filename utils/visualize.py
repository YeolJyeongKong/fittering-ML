import torch
import os
import numpy as np

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplpytorch.display_utils import display_model

class Beta2Image:
    def __init__(self, gender='neural', 
                 smpl_model_path='/home/shin/VScodeProjects/fittering-ML/data/additional/smpl/SMPL_NEUTRAL.pkl', 
                 smpl_mean_params_path='/home/shin/VScodeProjects/fittering-ML/data/additional/neutral_smpl_mean_params_6dpose.npz'):

        self.smpl_layer = SMPL_Layer(center_idx=0, gender=gender, model_path=smpl_model_path)
        neutral_mean = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/additional/neutral_smpl_mean_params_6dpose.npz", mmap_mode='r')
        self.mean_pose = torch.tensor(neutral_mean['shape']).view(1, -1)

    def beta2image(self, betas):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

        display_model(
            {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath='/home/shin/VScodeProjects/fittering-ML/modeling/image.png',
            # savepath=None,
            show=True)


if __name__ == '__main__':
    print(os.getcwd)
    cuda = False
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_path='/home/shin/VScodeProjects/fittering-ML/data/additional/smpl/SMPL_NEUTRAL.pkl')

    pose_params = torch.zeros((1, 72))

    neutral = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/additional/neutral_smpl_mean_params_6dpose.npz", mmap_mode='r')
    shape_params = torch.tensor(neutral['shape']).view(1, -1)

    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

    # Draw output vertices and joints
    display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath='/home/shin/VScodeProjects/fittering-ML/modeling/image.png',
        # savepath=None,
        show=True)
