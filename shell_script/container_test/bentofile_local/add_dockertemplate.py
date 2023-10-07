from omegaconf import OmegaConf
import os
import argparse
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
parser = argparse.ArgumentParser()
parser.add_argument("--bentofile_name", dest="bentofile_name", type=str, default=None)
args = parser.parse_args()

bentofile_path = os.path.join(root_dir, "bentofile", args.bentofile_name)
bentofile = OmegaConf.load(bentofile_path)
bentofile.docker.dockerfile_template = (
    "./shell_script/container_test/bentofile_local/Dockerfile.template"
)

OmegaConf.save(
    bentofile, f"./shell_script/container_test/bentofile_local/{args.bentofile_name}"
)
