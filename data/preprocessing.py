from typing import Any
import torch
import torchvision.transforms.functional as F 
import numpy as np
import torch.nn.functional as nn_f

class TrueCrop(object):
    def __init__(self, di) -> None:
        self.di = di
    def __call__(self, image) -> Any:
        image = image.cpu().type(torch.FloatTensor)
        assert image.ndim == 3, f"image.ndim must be 3 but get {image.ndim}"
        assert image.shape[0] == 1, "image must be binary image"
        _, y, x = image.nonzero(as_tuple=True)
        y = torch.sort(y)[0]
        x = torch.sort(x)[0]
        if self.di == 0:
            image = image[:, y[0]:y[-1], x[0]:x[-1]]
        elif self.di == 1:
            image = image[:, y[0]:y[-1], :]
        else:
            raise Exception("di must be in (0, 1)")
        return image
    
def crop_true(image, di):
    image = image.cpu().type(torch.FloatTensor)
    assert image.ndim == 3, f"image.ndim must be 3 but get {image.ndim}"
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

def crop_resize_binary_image(image: torch.Tensor, resize_wh=512) -> torch.Tensor:
    assert image.ndim == 3, f"image.ndim must be 3 but get {image.ndim}"
    _, y, x = image.nonzero(as_tuple=True)
    y = torch.sort(y)[0]
    x = torch.sort(x)[0]
    image = image[:, y[0]:y[-1], x[0]:x[-1]]
    image = F.resize(image, size=512)
    print(image.shape)
    # image = F.to_pil_image(image.cpu().type(torch.FloatTensor))
    # image = F.resize(image, (512, 512))
    # image = totensor(np.array(image))
    return image

def PILfromtensor(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def totensor(array: np.ndarray, device=torch.device('cuda')):
    return torch.from_numpy(array).float().to(device)