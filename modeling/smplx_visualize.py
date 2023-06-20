import torch
import os

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplpytorch.display_utils import display_model
# from STRAPS_3DHumanShapePose.data.synthetic_training_dataset import SyntheticTrainingDataset


if __name__ == '__main__':
    print(os.getcwd)
    cuda = False
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='female',
        model_root='./smplpytorch/smplpytorch/native/models')

    # Generate random pose and shape parameters
    # pose_params = torch.rand(batch_size, 72) * 0.2
    # shape_params = torch.rand(batch_size, 10) * 0.03

    pose_params = torch.zeros((1, 72))
    shape_params = torch.tensor([[0.04792313692484085,
        0.26527075040151804,
        0.7512205727618557,
        0.2978821327866097,
        0.1866084982418149,
        0.3341038157933196,
        0.8757696215861747,
        0.09900752184889527,
        0.7856228425535512,
        0.7729162775177757]])

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

    # Draw output vertices and joints
    display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        # savepath='image.png',
        savepath=None,
        show=True)
