import os
import argparse
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src import utils
from extras import paths
from src.inference import encoder_inference
from src.trainer import human_size_trainer, product_encode_trainer


def human_size_main(cfg: DictConfig) -> None:
    segment_tag = human_size_trainer.save_segment(cfg, saved=True)

    module, datamodule, autoencoder_tag = human_size_trainer.train_AutoEncoderModule(
        cfg
    )
    (train_x, train_y), (test_x, test_y) = encoder_inference.encode(
        module, datamodule, device=torch.device("cuda")
    )

    regression_tag = human_size_trainer.train_Regression(
        (train_x, train_y), (test_x, test_y), cfg
    )

    bentofile = OmegaConf.load(paths.BENTOFILE_DEFAULT_PATH)
    bentofile.service = "serving.bentoml.service_human_size:svc"
    bentofile.models = [
        segment_tag,
        autoencoder_tag,
        regression_tag,
    ]
    bentofile.docker.env.OUTPUT_DIR = os.path.relpath(cfg.paths.output_dir, root_dir)

    return bentofile


def product_encode_main(cfg: DictConfig):
    product_tag = product_encode_trainer.train_ProductModule(cfg)
    bentofile = OmegaConf.load(paths.BENTOFILE_DEFAULT_PATH)
    bentofile.service = "serving.bentoml.service_fashion_cbf:svc"
    bentofile.models = [product_tag]
    output_dir = os.path.relpath(cfg.paths.output_dir, root_dir)
    bentofile.include += [output_dir]
    bentofile.docker.env.OUTPUT_DIR = output_dir

    return bentofile


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    if cfg.train == "human_size":
        bentofile = human_size_main(cfg)

    elif cfg.train == "product_encode":
        bentofile = product_encode_main(cfg)

    bentofile_path = os.path.join(cfg.paths.output_dir, "bentofile.yaml")
    OmegaConf.save(config=bentofile, f=bentofile_path)


if __name__ == "__main__":
    main()
