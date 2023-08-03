from typing import Any
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import torch.nn.functional as nn_f


class Numpy2Tensor(object):
    def __init__(self) -> None:
        self.totensor = ToTensor()

    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        arr = arr.astype(np.float32)
        tensor = self.totensor(arr)
        return tensor


class Normalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / 255.0


def rotate(img: np.ndarray) -> np.ndarray:
    img = np.fliplr(img)
    img = np.rot90(img)
    return img


def image2numpy(path):
    image = Image.open(path).convert("L")
    image = np.array(image)
    return image


def crop_true(image):
    print(image.shape)
    image = image.cpu().type(torch.FloatTensor)
    assert image.ndim == 3, f"image.ndim must be 3 but get {image.ndim}"
    assert image.shape[0] == 1, "image must be binary image"
    _, y, x = image.nonzero(as_tuple=True)
    y = torch.sort(y)[0]
    x = torch.sort(x)[0]

    image = image[:, y[0] : y[-1], x[0] : x[-1]]
    max_size = torch.max(torch.tensor(image.shape))
    padding_height = max_size - image.shape[1]
    padding_height = (
        (padding_height // 2, padding_height // 2)
        if padding_height % 2 == 0
        else (padding_height // 2 + 1, padding_height // 2)
    )
    padding_width = max_size - image.shape[2]
    padding_width = (
        (padding_width // 2, padding_width // 2)
        if padding_width % 2 == 0
        else (padding_width // 2 + 1, padding_width // 2)
    )

    padded_image = nn_f.pad(image, pad=padding_width + padding_height)
    return padded_image


class Crop(object):
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def __call__(self, image: torch.Tensor) -> Any:
        _, y, x = (image > self.threshold).nonzero(as_tuple=True)
        y = torch.sort(y)[0]
        x = torch.sort(x)[0]

        image = image[:, y[0] : y[-1], x[0] : x[-1]]
        max_size = torch.max(torch.tensor(image.shape))
        padding_height = max_size - image.shape[1]
        padding_height = (
            (padding_height // 2, padding_height // 2)
            if padding_height % 2 == 0
            else (padding_height // 2 + 1, padding_height // 2)
        )
        padding_width = max_size - image.shape[2]
        padding_width = (
            (padding_width // 2, padding_width // 2)
            if padding_width % 2 == 0
            else (padding_width // 2 + 1, padding_width // 2)
        )

        padded_image = nn_f.pad(image, pad=padding_width + padding_height)
        return padded_image


def crop(image: torch.Tensor):
    _, y, x = (image > 0.5).nonzero(as_tuple=True)
    y = torch.sort(y)[0]
    x = torch.sort(x)[0]

    image = image[:, y[0] : y[-1], x[0] : x[-1]]
    max_size = torch.max(torch.tensor(image.shape))
    padding_height = max_size - image.shape[1]
    padding_height = (
        (padding_height // 2, padding_height // 2)
        if padding_height % 2 == 0
        else (padding_height // 2 + 1, padding_height // 2)
    )
    padding_width = max_size - image.shape[2]
    padding_width = (
        (padding_width // 2, padding_width // 2)
        if padding_width % 2 == 0
        else (padding_width // 2 + 1, padding_width // 2)
    )

    padded_image = nn_f.pad(image, pad=padding_width + padding_height)
    return padded_image


def morphology(image):
    image = image.numpy()[0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
    return torch.tensor(image[np.newaxis, ...])
