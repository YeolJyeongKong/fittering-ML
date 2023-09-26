import requests
import re
import asyncio
import aiohttp
from io import BytesIO
import PIL
from PIL import Image
import torch
import torchvision.transforms.functional as F


def local_check():
    req = requests.get("http://ipconfig.kr")
    host_ip = re.search(r"IP Address : (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", req.text)[
        1
    ]
    if host_ip == "220.72.106.46":
        return True
    else:
        return False


def mean_nframe_encoded(imgs_encoded, nframes):
    new_imgs_encoded = []
    start_idx = 0
    for nframe in nframes:
        end_idx = start_idx + nframe
        sub_tensor = imgs_encoded[start_idx:end_idx]
        mean_tensor = torch.mean(sub_tensor, dim=0, keepdim=True)
        new_imgs_encoded.append(mean_tensor)
        start_idx = end_idx
    return torch.cat(new_imgs_encoded, dim=0)


def to_bytearray(pred, size):
    masked = F.resize(pred, size=size[::-1])
    masked = F.to_pil_image(masked).convert("RGB")
    buffered = BytesIO()
    masked.save(buffered, format="JPEG")
    pred_str = buffered.getvalue()
    return pred_str


async def load_img_(url, preprocess):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            obj = await response.read()

    image_obj = Image.open(BytesIO(obj))
    if isinstance(image_obj, PIL.GifImagePlugin.GifImageFile):
        imgs_lst = []
        for i in range(image_obj.n_frames):
            image_obj.seek(i)
            imgs_lst.append(preprocess(image_obj.convert("RGB"))[:3])
        return torch.stack(imgs_lst, dim=0), image_obj.n_frames

    return preprocess(Image.open(BytesIO(obj)))[:3].unsqueeze(0), 1


async def load_img(urls, preprocess):
    tasks = [load_img_(url, preprocess) for url in urls]
    img_nframe_lst = await asyncio.gather(*tasks)
    imgs, nframes = [], []
    for img, nframe in img_nframe_lst:
        imgs.append(img)
        nframes.append(nframe)
    return torch.cat(imgs, dim=0), nframes
