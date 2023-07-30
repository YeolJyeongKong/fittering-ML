from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

import torchvision.transforms.functional as F


def preprocess_segment(img):
    img = F.pil_to_tensor(img)
    img = F.resize(img, size=(512, 512))
    img = img.unsqueeze(dim=0).float() / 255.0
    return img


def to_bytearray(pred, size):
    masked = F.resize(pred, size=size[::-1])
    masked = F.to_pil_image(masked).convert("RGB")
    buffered = BytesIO()
    masked.save(buffered, format="JPEG")
    pred_str = buffered.getvalue()
    return pred_str
