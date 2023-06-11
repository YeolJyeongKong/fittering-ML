import re
pkgs = None
with open("./modeling/Human-Segmentation-PyTorch/requirements.txt", 'r') as f:
    pkgs = list(map(lambda x: re.sub(r'\n', '', x).split("=")[0], f.readlines()))
with open("./modeling/Human-Segmentation-PyTorch/requirements.txt", 'w') as f:
    for pkg in pkgs:
        f.write(pkg + '\n')