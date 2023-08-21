from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree

import wandb
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
import cv2


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def print_wandb_run(cfg: DictConfig):
    with open(Path(cfg.paths.output_dir, "wandb_run.log"), "w") as file:
        file.write(f"run url: {wandb.run.get_url()}\n")
        file.write(f"run id: {wandb.run.id}\n")
        file.write(f"run name: {wandb.run.name}")


def draw_bbox(img, pred_bbox, label_bbox):
    img = img.copy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    cv2.rectangle(
        img,
        (int(pred_bbox[0] * w), int(pred_bbox[1] * h)),
        (int(pred_bbox[2] * w), int(pred_bbox[3] * h)),
        (0, 0, 1.0),
        thickness=1,
    )
    cv2.rectangle(
        img,
        (int(label_bbox[0] * w), int(label_bbox[1] * h)),
        (int(label_bbox[2] * w), int(label_bbox[3] * h)),
        (0, 1.0, 0),
        thickness=1,
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
