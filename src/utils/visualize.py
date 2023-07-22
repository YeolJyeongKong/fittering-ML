import torch
import os
import numpy as np

from src.data.synthesize.smpl_augmentation import front_side_pose
from src.data.synthesize.smpl_official import SMPL
from extras import paths

# from smplpytorch.pytorch.smpl_layer import SMPL_Layer
# from smplpytorch.display_utils import display_model


class Beta2Verts:
    def __init__(
        self,
        gender="neural",
        batch_size=None,
        device=None,
        smpl_model_path="/home/shin/VScodeProjects/fittering-ML/data/additional/smpl/SMPL_NEUTRAL.pkl",
        smpl_mean_params_path="/home/shin/VScodeProjects/fittering-ML/data/additional/neutral_smpl_mean_params_6dpose.npz",
    ):
        # self.smpl_layer = SMPL_Layer(center_idx=0, gender=gender, model_path=smpl_model_path).cuda()
        # neutral_mean = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/additional/neutral_smpl_mean_params_6dpose.npz", mmap_mode='r')
        # self.mean_pose = torch.tensor(neutral_mean['shape']).view(1, -1).repeat(batch_size, 1)
        self.mean_pose = torch.zeros((batch_size, 72)).cuda()
        self.smpl_model = SMPL(
            model_path=config.SMPL_MODEL_DIR,
            J_REGRESSOR_EXTRA_PATH=config.J_REGRESSOR_EXTRA_PATH,
            COCOPLUS_REGRESSOR_PATH=config.COCOPLUS_REGRESSOR_PATH,
            H36M_REGRESSOR_PATH=config.H36M_REGRESSOR_PATH,
            batch_size=1,
        ).cuda()

    def beta2verts(self, betas):
        pose_rotmats, glob_rotmats = front_side_pose(self.mean_pose, di=0)

        smpl_output = self.smpl_model(
            body_pose=pose_rotmats.contiguous(),
            global_orient=glob_rotmats.contiguous(),
            betas=betas,
            pose2rot=False,
        )

        vertices = smpl_output.vertices  # batch_size, 6890, 3

        # verts, Jtr = self.smpl_layer(self.mean_pose, th_betas=betas)
        vertices = vertices.cpu().numpy()
        return vertices
