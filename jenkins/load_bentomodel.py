import subprocess
import argparse
import bentoml
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--bentofile-path", required=True)
parser.add_argument("-s3", "--s3-dir", required=True)

args = parser.parse_args()
s3_dir = args.s3_dir
bentofile_path = args.bentofile_path
target_bentomodels = OmegaConf.load(bentofile_path).models

for target_bentomodel in target_bentomodels:
    model_name = target_bentomodel.split(":")[0]
    try:
        bentoml.models.get(target_bentomodel)
    except bentoml.exceptions.NotFound:
        try:
            bentoml.models.delete(model_name)
        except bentoml.exceptions.NotFound:
            pass
        bentoml.models.import_model(f"{s3_dir}/{model_name}.bentomodel")
