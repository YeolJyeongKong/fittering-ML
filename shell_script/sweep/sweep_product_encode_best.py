import os
import wandb
import shutil

api = wandb.Api()
sweep = api.sweep(f"sinjy1203/product_encode_sweep/{os.environ['SWEEP_ID']}")
bentofile_path = f"./outputs/{sweep.best_run().name}/bentofile.yaml"
best_bentofile_path = "bentofile/sweep_best/bentofile_product_encode.yaml"

shutil.copy(bentofile_path, best_bentofile_path)
