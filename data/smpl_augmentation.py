import torch
import numpy as np
from smplx.lbs import batch_rodrigues


def uniform_sample_shape(batch_size, mean_shape, delta_betas_range):
    """
    Uniform sampling of shape parameter deviations from the mean.
    """
    device = mean_shape.device
    l, h = delta_betas_range
    delta_betas = (h-l)*torch.rand(batch_size, 10, device=device) + l
    shape = delta_betas + mean_shape
    return shape  # (bs, 10)


def normal_sample_shape(batch_size, mean_shape, std_vector):
    """
    Gaussian sampling of shape parameter deviations from the mean.
    """
    device = mean_shape.device
    delta_betas = torch.randn(batch_size, 10, device=device)*std_vector
    shape = delta_betas + mean_shape
    return shape


def augment_smpl(orig_shape, pose,
                 mean_shape,
                 smpl_augment_params):
    """
    Augments SMPL shape parameters. Also converts SMPL pose parameters (in axis angle form)
    to rotation matrices.
    :param orig_shape: original shape.
    :param pose: pose parameters for the body (excluding root orientation).
    :param global_orients: root orientation.
    :param mean_shape: mean SMPL shape.
    :param smpl_augment_params: dict containing parameters for SMPL augmentation.
    """
    augment_shape = smpl_augment_params['augment_shape']
    delta_betas_distribution = smpl_augment_params['delta_betas_distribution']  # 'normal' or 'uniform' shape sampling distribution
    delta_betas_range = smpl_augment_params['delta_betas_range']  # Range of uniformly-distributed shape parameters.
    delta_betas_std_vector = smpl_augment_params['delta_betas_std_vector']  # std of normally-distributed the shape parameters.

    batch_size = orig_shape.shape[0]
    if augment_shape:
        assert delta_betas_distribution in ['uniform', 'normal']
        if delta_betas_distribution == 'uniform':
            new_shape = uniform_sample_shape(batch_size, mean_shape, delta_betas_range)
        elif delta_betas_distribution == 'normal':
            assert delta_betas_std_vector is not None
            new_shape = normal_sample_shape(batch_size, mean_shape, delta_betas_std_vector)
    else:
        new_shape = orig_shape

    front_pose_rotmats, front_glob_rotmats = front_side_pose(pose, di=0)
    side_pose_rotmats, side_glob_rotmats = front_side_pose(pose, di=1)

    return new_shape, front_pose_rotmats, front_glob_rotmats, side_pose_rotmats, side_glob_rotmats


def front_side_pose(pose, di):
    pose_ = pose.clone().detach()
    if di == 1:
        pose_[0, 41] = -70 / 180 * np.pi
        pose_[0, 44] = 70 / 180 * np.pi

    pose_rotmats = pose_[:, 3:]
    global_orients = pose_[:, :3]

    pose_rotmats = batch_rodrigues(pose_rotmats.contiguous().view(-1, 3))
    pose_rotmats = pose_rotmats.view(-1, 23, 3, 3)

    glob_rotmats = batch_rodrigues(global_orients.contiguous().view(-1, 3))
    glob_rotmats = glob_rotmats.unsqueeze(1)

    return pose_rotmats, glob_rotmats


if __name__ == "__main__":
    batch_size = 16
    mean_pose = torch.zeros((batch_size, 72))
    front_side_pose(mean_pose, di=0)
