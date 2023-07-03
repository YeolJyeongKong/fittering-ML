import sys
import pandas as pd
import torch
sys.path.append("/home/shin/VScodeProjects/fittering-ML")
from data.smpl_official import SMPL
from smplx.lbs import batch_rodrigues

from utils.measure import MeasureVerts
from utils.measurement_definitions import *
import config

class Beta2Measurements:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL(model_path=config.SMPL_MODEL_DIR, J_REGRESSOR_EXTRA_PATH=config.J_REGRESSOR_EXTRA_PATH, 
                                COCOPLUS_REGRESSOR_PATH=config.COCOPLUS_REGRESSOR_PATH, 
                                H36M_REGRESSOR_PATH=config.H36M_REGRESSOR_PATH).to(self.device)

    @staticmethod
    def pose2rotmats(mean_pose):
        pose_rotmats = batch_rodrigues(mean_pose[:, 3:].contiguous().view(-1, 3))
        pose_rotmats = pose_rotmats.view(-1, 23, 3, 3)

        glob_rotmats = batch_rodrigues(mean_pose[:, :3].contiguous().view(-1, 3))
        glob_rotmats = glob_rotmats.unsqueeze(1)
        
        return pose_rotmats, glob_rotmats

    def predict(self, betas):
        batch_size = betas.shape[0]
        mean_pose = torch.zeros((batch_size, 72)).to(self.device)
        betas = betas.to(self.device)

        pose_rotmats, glob_rotmats = Beta2Measurements.pose2rotmats(mean_pose) 
        smpl_output = self.smpl_model(body_pose=pose_rotmats, global_orient=glob_rotmats, 
                                      betas=betas, pose2rot=False)
        
        measurements_dict = []
        for i in range(batch_size):
            measurements = MeasureVerts.verts2meas(smpl_output.vertices[i], smpl_output.joints[i])
            measurements_dict.append(measurements)
            
        measurements_df = pd.DataFrame(measurements_dict)
        return measurements_df


if __name__ == "__main__":
    beta2meas = Beta2Measurements(device=torch.device("cuda"))
    print(beta2meas.predict(betas=torch.zeros((10, 10))))