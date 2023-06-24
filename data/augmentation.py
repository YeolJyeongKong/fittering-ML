import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

from data.smpl_augmentation import augment_smpl
from data.cam_utils import get_intrinsics_matrix, perspective_project_torch
from data.cam_augmentation import augment_cam_t
from data.nmr_renderer import NMRRenderer
from data.smpl_official import SMPL
from data.proxy_rep_augmentation import augment_proxy_representation
from data.label_conversions import convert_multiclass_to_binary_labels_torch, convert_2Djoints_to_gaussian_heatmaps_torch


class AugmentBetasCam:
    def __init__(self, device: torch.device = torch.device('cpu'), 
                 img_wh: int = 512,
                 betas_std_vect: Union[float, List[float]] = 1.5, 
                 K_std=1, t_xy_std=0.05, t_z_range=[-5, 5],
                 smpl_mean_params_path: str = "/home/shin/VScodeProjects/fittering-ML/data/additional/neutral_smpl_mean_params_6dpose.npz", 
                 SMPL_MODEL_DIR: str = "/home/shin/VScodeProjects/fittering-ML/data/additional/smpl", 
                 J_REGRESSOR_EXTRA_PATH: str = "/home/shin/VScodeProjects/fittering-ML/data/additional/J_regressor_extra.npy", 
                 COCOPLUS_REGRESSOR_PATH: str = "/home/shin/VScodeProjects/fittering-ML/data/additional/cocoplus_regressor.npy", 
                 H36M_REGRESSOR_PATH: str = "/home/shin/VScodeProjects/fittering-ML/data/additional/J_regressor_h36m.npy") -> None:
        self.device = device

        if not isinstance(betas_std_vect, list):
            betas_std_vect = [betas_std_vect] * 10
        
        # betas augmentation
        delta_betas_std_vector = torch.tensor(betas_std_vect, device=device).float()
        self.smpl_augment_params = {'augment_shape': True,
                            'delta_betas_distribution': 'normal',
                            'delta_betas_std_vector': delta_betas_std_vector,
                            'delta_betas_range': [0, 0]}
        
        self.target_pose = torch.zeros((1, 72)).to(device)

        self.img_wh = img_wh

        # camera augmentation 
        self.K_std = K_std
        self.t_xy_std = t_xy_std
        self.t_z_range = t_z_range
        
        smpl_mean_params = np.load(smpl_mean_params_path)
        self.mean_shape = torch.from_numpy(smpl_mean_params['shape']).float().to(self.device)

        # smpl_model
        self.smpl_model = SMPL(model_path=SMPL_MODEL_DIR, J_REGRESSOR_EXTRA_PATH=J_REGRESSOR_EXTRA_PATH, 
                               COCOPLUS_REGRESSOR_PATH=COCOPLUS_REGRESSOR_PATH, 
                               H36M_REGRESSOR_PATH=H36M_REGRESSOR_PATH, batch_size=1).to(self.device)
        
        
        ##
        self.ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
        self.ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))

        # Indices to get the 14 LSP joints from the 17 H36M joints
        self.H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
        self.H36M_TO_J14 = self.H36M_TO_J17[:14]

        
        remove_appendages = False
        deviate_joints2D = False
        deviate_verts2D = False
        occlude_seg = False
        remove_appendages_classes = [1, 2, 3, 4, 5, 6]
        remove_appendages_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
        delta_j2d_dev_range = [-8, 8]
        delta_j2d_hip_dev_range = [-8, 8]
        delta_verts2d_dev_range = [-0.01, 0.01]
        occlude_probability = 0.5
        occlude_box_dim = 48

        self.proxy_rep_augment_params = {'remove_appendages': remove_appendages,
                            'deviate_joints2D': deviate_joints2D,
                            'deviate_verts2D': deviate_verts2D,
                            'occlude_seg': occlude_seg,
                            'remove_appendages_classes': remove_appendages_classes,
                            'remove_appendages_probabilities': remove_appendages_probabilities,
                            'delta_j2d_dev_range': delta_j2d_dev_range,
                            'delta_j2d_hip_dev_range': delta_j2d_hip_dev_range,
                            'delta_verts2d_dev_range': delta_verts2d_dev_range,
                            'occlude_probability': occlude_probability,
                            'occlude_box_dim': occlude_box_dim}
        
        self.R_z_180 = torch.tensor([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.]
        ])

    def aug_betas(self, betas: np.ndarray):
        assert betas.shape == (1, 10), \
            f"betas shape: {betas.shape} but expected shape: {(1, 10)}"
        
        betas_aug, target_pose_rotmats, target_glob_rotmats = augment_smpl(
            torch.from_numpy(betas.astype(np.float32)).to(self.device),
            self.target_pose[:, 3:],
            self.target_pose[:, :3],
            self.mean_shape,
            self.smpl_augment_params
        )

        return betas_aug, target_pose_rotmats, target_glob_rotmats

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
        return stature

    def aug_cam(self):
        focal_length = 5000. + np.random.randn() * self.K_std
        cam_K = get_intrinsics_matrix(self.img_wh, self.img_wh, focal_length)
        cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(self.device)
        cam_K = cam_K[None, :, :].expand(1, -1, -1)

        cam_t = np.array([0., 0, 25.])
        cam_t = torch.from_numpy(cam_t).float().to(self.device)
        cam_t = cam_t[None, :].expand(1, -1)
        cam_t = augment_cam_t(cam_t, xy_std=self.t_xy_std, delta_z_range=self.t_z_range)

        cam_front_R = (torch.eye(3) @ self.R_z_180).to(self.device)[None, :, :].expand(1, -1, -1)
        cam_side_R = (torch.tensor([
            [0., 0., 1.], 
            [0., 1., 0.],
            [-1., 0., 0.]
        ]) @ self.R_z_180).to(self.device)[None, :, :].expand(1, -1, -1)
        return cam_K, cam_t, (cam_front_R, cam_side_R)
    
    def render_image_(self, betas_rotats, cam_params, di):
        betas_aug, target_pose_rotmats, target_glob_rotmats = betas_rotats
        cam_K, cam_t, cam_R = cam_params
        cam_R = cam_R[di]
        nmr_parts_renderer = NMRRenderer(1,
                                    cam_K,
                                    cam_R,
                                    self.img_wh,
                                    rend_parts_seg=True).to(self.device)
        
        target_smpl_output = self.smpl_model(body_pose=target_pose_rotmats,
                                global_orient=target_glob_rotmats,
                                betas=betas_aug,
                                pose2rot=False)

        target_vertices = target_smpl_output.vertices
        target_joints_all = target_smpl_output.joints

        target_joints_h36m = target_joints_all[:, self.ALL_JOINTS_TO_H36M_MAP, :]
        target_joints_h36mlsp = target_joints_h36m[:, self.H36M_TO_J14, :]
        target_joints_coco = target_joints_all[:, self.ALL_JOINTS_TO_COCO_MAP, :]
        target_joints2d_coco = perspective_project_torch(target_joints_coco, cam_R,
                                                            cam_t,
                                                            cam_K=cam_K)
        target_reposed_smpl_output = self.smpl_model(betas=betas_aug)
        target_reposed_vertices = target_reposed_smpl_output.vertices

        input = nmr_parts_renderer(target_vertices, cam_t)

        input, target_joints2d_coco_input = augment_proxy_representation(input,
                                                                    target_joints2d_coco,
                                                                    self.proxy_rep_augment_params)
        
        input = convert_multiclass_to_binary_labels_torch(input)
        input = input.unsqueeze(1)
        # height = self.get_height(target_reposed_vertices[0])
        # j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
        #                                                             self.img_wh)
        return input, target_reposed_vertices


    def render_image(self, betas):
        betas_aug, target_pose_rotmats, target_glob_rotmats = self.aug_betas(betas)
        cam_K, cam_t, cam_R = self.aug_cam()

        front_image, vertices = self.render_image_((betas_aug, target_pose_rotmats, target_glob_rotmats), 
                           (cam_K, cam_t, cam_R), di=0)
        side_image, _ = self.render_image_((betas_aug, target_pose_rotmats, target_glob_rotmats), 
                           (cam_K, cam_t, cam_R), di=1)
        height = self.get_height(vertices[0])
        return front_image[0], side_image[0], height, betas_aug

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