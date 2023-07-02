import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplpytorch.display_utils import display_model
# from STRAPS_3DHumanShapePose.data.synthetic_training_dataset import SyntheticTrainingDataset

neutral = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/additional/neutral_smpl_mean_params_6dpose.npz", mmap_mode='r')

if __name__ == '__main__':
    print(os.getcwd)
    cuda = False
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_path='/home/shin/VScodeProjects/fittering-ML/data/additional/smpl/SMPL_NEUTRAL.pkl')

    # Generate random pose and shape parameters
    # pose_params = torch.rand(batch_size, 72) * 0.2
    # shape_params = torch.rand(batch_size, 10) * 0.03

    pose_params = torch.zeros((1, 72))
    shape_params = torch.tensor(neutral['shape']).view(1, -1)

    # dataset = SyntheticTrainingDataset(npz_path="./STRAPS_3DHumanShapePose/data/amass_up3d_3dpw_train.npz", 
    #                                    params_from='3dpw') #['all', 'h36m', 'up3d', '3dpw', 'not_amass']
    # print(len(dataset))
    # data = dataset.__getitem__(100)
    # pose_params = data['pose'].view(1, -1)
    # shape_params = data['shape'].view(1, -1)

    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
    verts = verts.numpy()[0]
    print(verts.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2])
    plt.show()

    # with open("/home/shin/VScodeProjects/fittering-ML/mesh.obj", 'w') as f:
    #     for vert in verts:
    #         f.write(f"{vert[0]} {vert[1]} {vert[2]}\n")


    # Draw output vertices and joints
    # display_model(
    #     {'verts': verts.cpu().detach(),
    #      'joints': Jtr.cpu().detach()},
    #     model_faces=smpl_layer.th_faces,
    #     with_joints=True,
    #     kintree_table=smpl_layer.kintree_table,
    #     savepath='/home/shin/VScodeProjects/fittering-ML/modeling/image.png',
    #     # savepath=None,
    #     show=True)
