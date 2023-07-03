import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
sys.path.append("/home/shin/VScodeProjects/fittering-ML")
from data.smpl_augmentation import augment_smpl
from data.cam_utils import get_intrinsics_matrix, perspective_project_torch
from data.cam_augmentation import augment_cam_t
from data.renderer import renderer
from data.smpl_official import SMPL
from data.proxy_rep_augmentation import augment_proxy_representation
from data.label_conversions import convert_multiclass_to_binary_labels_torch, convert_2Djoints_to_gaussian_heatmaps_torch
import config

import pytorch3d
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

class AugmentBetasCam:
    def __init__(self, device: torch.device = torch.device('cpu'), 
                 img_wh: int = 512,
                 betas_std_vect: Union[float, List[float]] = 1.5, 
                 K_std=1, t_xy_std=0.05, t_z_range=[-5, 5]) -> None:
        self.device = device

        if not isinstance(betas_std_vect, list):
            betas_std_vect = [betas_std_vect] * 10
        
        # betas augmentation
        delta_betas_std_vector = torch.tensor(betas_std_vect, device=device).float()
        self.smpl_augment_params = {'augment_shape': True,
                            'delta_betas_distribution': 'normal',
                            'delta_betas_std_vector': delta_betas_std_vector,
                            'delta_betas_range': [0, 0]}

        self.img_wh = img_wh

        # camera augmentation 
        self.K_std = K_std
        self.t_xy_std = t_xy_std
        self.t_z_range = t_z_range
        
        smpl_mean_params = np.load(config.SMPL_MEAN_PARAMS_PATH)
        self.mean_shape = torch.from_numpy(smpl_mean_params['shape']).float().to(self.device)

        # smpl_model
        self.smpl_model = SMPL(model_path=config.SMPL_MODEL_DIR, J_REGRESSOR_EXTRA_PATH=config.J_REGRESSOR_EXTRA_PATH, 
                               COCOPLUS_REGRESSOR_PATH=config.COCOPLUS_REGRESSOR_PATH, 
                               H36M_REGRESSOR_PATH=config.H36M_REGRESSOR_PATH, batch_size=1).to(self.device)


        self.proxy_rep_augment_params = {'remove_appendages': config.remove_appendages,
                            'deviate_joints2D': config.deviate_joints2D,
                            'deviate_verts2D': config.deviate_verts2D,
                            'occlude_seg': config.occlude_seg,
                            'remove_appendages_classes': config.remove_appendages_classes,
                            'remove_appendages_probabilities': config.remove_appendages_probabilities,
                            'delta_j2d_dev_range': config.delta_j2d_dev_range,
                            'delta_j2d_hip_dev_range': config.delta_j2d_hip_dev_range,
                            'delta_verts2d_dev_range': config.delta_verts2d_dev_range,
                            'occlude_probability': config.occlude_probability,
                            'occlude_box_dim': config.occlude_box_dim}
        
        self.R_z_180 = torch.tensor([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.]
        ])

        faces = np.load(config.SMPL_FACES_PATH)
        self.faces =torch.from_numpy(faces.astype(np.int)).to(self.device)
        self.pose = torch.zeros((1, 72)).to(self.device)

    def aug_betas(self, betas: np.ndarray):
        assert betas.shape == (1, 10), \
            f"betas shape: {betas.shape} but expected shape: {(1, 10)}"
        
        betas_aug, front_target_pose_rotmats, front_target_glob_rotmats, side_target_pose_rotmats, side_target_glob_rotmats = \
            augment_smpl(
                torch.from_numpy(betas.astype(np.float32)).to(self.device),
                self.pose,
                self.mean_shape,
                self.smpl_augment_params
            )

        return betas_aug, front_target_pose_rotmats, front_target_glob_rotmats, side_target_pose_rotmats, side_target_glob_rotmats

    def get_height(self, vertices_shaped):
        vertices_shaped = vertices_shaped.detach().cpu().numpy().astype(np.float64)
        sorted_Y = vertices_shaped[vertices_shaped[:, 1].argsort()]

        vertex_with_smallest_y = sorted_Y[:1]
        vertex_with_biggest_y = sorted_Y[6889:]
        # Simulate the 'floor' by setting the x and z coordinate to 0.
        vertex_on_floor = np.array([0, vertex_with_smallest_y[0, 1], 0])
        stature = np.linalg.norm(
            np.subtract(
                vertex_with_biggest_y.view(dtype="f8"),
                vertex_on_floor.view(dtype="f8"),
            )
        )
        return stature.astype(np.float32)

    def aug_cam(self):
        focal_length = 5000. + np.random.randn() * self.K_std
        cam_K = get_intrinsics_matrix(self.img_wh, self.img_wh, focal_length)
        cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(self.device)
        cam_K = cam_K[None, :, :].expand(1, -1, -1)

        cam_t = np.array([0., 0, 2.5])
        cam_t = torch.from_numpy(cam_t).float().to(self.device)
        cam_t = cam_t[None, :].expand(1, -1)
        cam_t = augment_cam_t(cam_t, xy_std=self.t_xy_std, delta_z_range=self.t_z_range)

        cam_front_R = torch.tensor([[[-1.,  0.,  0.],
                                     [ 0.,  1.,  0.],
                                     [ 0.,  0., -1.]]])
        
        cam_side_R = torch.tensor([[[0.,  0.,  1.],
                                    [0.,  1.,  0.],
                                    [-1., 0.,  0.]]])
        return cam_K, cam_t, (cam_front_R, cam_side_R)
    
    def render_image_(self, betas_rotats, cam_params, di):
        betas_aug, target_pose_rotmats, target_glob_rotmats = betas_rotats
        cam_K, cam_t, cam_R = cam_params
        cam_R = cam_R[di]
        
        target_smpl_output = self.smpl_model(body_pose=target_pose_rotmats,
                                global_orient=target_glob_rotmats,
                                betas=betas_aug,
                                pose2rot=False)

        target_vertices = target_smpl_output.vertices # 1, 6890, 3
        silhoute_image = renderer(target_vertices, self.faces, cam_R, cam_t, self.device)

        return silhoute_image, target_vertices

    def render_image(self, betas):
        cam_K, cam_t, cam_R = self.aug_cam()

        betas_aug, front_target_pose_rotmats, front_target_glob_rotmats, side_target_pose_rotmats, side_target_glob_rotmats = self.aug_betas(betas)
        front_image, vertices = self.render_image_((betas_aug, front_target_pose_rotmats, front_target_glob_rotmats), 
                           (cam_K, cam_t, cam_R), di=0)
        
        side_image, _ = self.render_image_((betas_aug, side_target_pose_rotmats, side_target_glob_rotmats), 
                           (cam_K, cam_t, cam_R), di=1)
        height = self.get_height(vertices[0])
        return front_image, side_image, height, betas_aug

if __name__ == "__main__":
    data = np.load("/home/shin/VScodeProjects/fittering-ML/modeling/STRAPS-3DHumanShapePose/data/amass_up3d_3dpw_train.npz")
    ord_shapes = data['shapes']

    augment = AugmentBetasCam(device=torch.device('cuda'), t_z_range=[0, 0], t_xy_std=0)
                            # betas_std_vect=0, K_std=0, t_xy_std=0, t_z_range=[0, 0])
    front, side, height, betas = augment.render_image(ord_shapes[0:1])
    print(height)
    print(betas.shape)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(front[0].cpu().numpy(), cmap='gray')
    axes[1].imshow(side[0].cpu().numpy(), cmap='gray')
    print(front.nonzero(as_tuple=True))
    plt.show()