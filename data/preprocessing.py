from typing import Any
import torch
import torchvision.transforms.functional as F 
import numpy as np
import torch.nn.functional as nn_f

    
def crop_true(image):
    image = image.cpu().type(torch.FloatTensor)
    assert image.ndim == 3, f'image.ndim must be 3 but get {image.ndim}'
    assert image.shape[0] == 1, "image must be binary image"
    _, y, x = image.nonzero(as_tuple=True)
    y = torch.sort(y)[0]
    x = torch.sort(x)[0]

    image = image[:, y[0]:y[-1], x[0]:x[-1]]
    max_size = torch.max(torch.tensor(image.shape))
    padding_height = (max_size - image.shape[1])
    padding_height = (padding_height // 2, padding_height // 2) if padding_height % 2 == 0\
        else (padding_height // 2 + 1, padding_height // 2)
    padding_width = (max_size - image.shape[2])
    padding_width = (padding_width // 2, padding_width // 2) if padding_width % 2 == 0\
        else (padding_width // 2 + 1, padding_width // 2)
    
    padded_image = nn_f.pad(image, pad=padding_width+padding_height)
    return padded_image

def convert_multiclass_to_binary_labels_torch(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = torch.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels


class BinTensor(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        """
        Converts multiclass segmentation labels into a binary mask.
        """
        if tensor.ndim == 2:
            tensor = torch.unsqueeze(tensor, 0)
        bin_tensor = torch.zeros_like(tensor)
        bin_tensor[tensor > self.threshold] = 1

        return bin_tensor