import os
import pyrootutils

ROOT_DIR = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------ Additional files ------------------------
ADDITIONAL_DIR = os.path.join(ROOT_DIR, "./extras/smpl/additional")
SMPL_MODEL_DIR = os.path.join(ADDITIONAL_DIR, "models")
SMPL_FACES_PATH = os.path.join(ADDITIONAL_DIR, "smpl_faces.npy")
SMPL_MEAN_PARAMS_PATH = os.path.join(
    ADDITIONAL_DIR, "neutral_smpl_mean_params_6dpose.npz"
)
J_REGRESSOR_EXTRA_PATH = os.path.join(ADDITIONAL_DIR, "J_regressor_extra.npy")
COCOPLUS_REGRESSOR_PATH = os.path.join(ADDITIONAL_DIR, "cocoplus_regressor.npy")
H36M_REGRESSOR_PATH = os.path.join(ADDITIONAL_DIR, "J_regressor_h36m.npy")
VERTEX_TEXTURE_PATH = os.path.join(ADDITIONAL_DIR, "vertex_texture.npy")
CUBE_PARTS_PATH = os.path.join(ADDITIONAL_DIR, "cube_parts.npy")
SEGMENTATION_PATH = os.path.join(ADDITIONAL_DIR, "smpl_body_parts_2_faces.json")

# ------------------------ model weights ------------------------
MODEL_WEIGHTS_DIR = os.path.join(ROOT_DIR, "./model_weights")

SEGMODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "SGHM-ResNet50.pth")

# ------------------------ data ------------------------
# fashion dataset
FASHION_DATA_DIR = (
    "/media/shin/T7/fashion_dataset/Category_and_Attribute_Prediction_Benchmark"
)

# data root dir
DATA_DIR = os.path.join(ROOT_DIR, "./data")

# aihub data
AIHUB_DATA_DIR = os.path.join(DATA_DIR, "aihub")

AIHUB_ENCODED_DIR = os.path.join(AIHUB_DATA_DIR, "encoded")
AIHUB_MASKED_DIR = os.path.join(AIHUB_DATA_DIR, "masked_images")
AIHUB_TRAIN_DIR = os.path.join(AIHUB_DATA_DIR, "train")
AIHUB_TEST_DIR = os.path.join(AIHUB_DATA_DIR, "test")

# synthetic data
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")

SYNTHETIC_ORIGINAL_DIR = os.path.join(SYNTHETIC_DATA_DIR, "original")
SYNTHETIC_ENCODED_DIR = os.path.join(SYNTHETIC_DATA_DIR, "encoded")
SYNTHETIC_TRAIN_DIR = os.path.join(SYNTHETIC_DATA_DIR, "train")
SYNTHETIC_TEST_DIR = os.path.join(SYNTHETIC_DATA_DIR, "test")

# real user dir
REAL_USER_DIR = "/home/shin/Documents/real_user"

# secret data dir
SECRET_USER_DIR = "/home/shin/Documents/secret_data"

# s3 access key
S3_ACCESS_KEY_PATH = (
    "/home/shin/Documents/aws/aws_access_key/LocalS3AccessUser_accessKeys.csv"
)

# bentofile_defualt path
BENTOFILE_DEFAULT_PATH = os.path.join(ROOT_DIR, "bentofile/bentofile_default.yaml")

# best sweep bentofile path
BENTOFILE_BEST_SWEEP_PATH = os.path.join(ROOT_DIR, "bentofile")
