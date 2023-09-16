import requests
import re
import torch


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
