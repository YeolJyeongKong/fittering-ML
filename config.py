import os

# ------------------------ Paths ------------------------
# Additional files
ADDITIONAL_DIR = "/home/shin/VScodeProjects/fittering-ML/data/additional"
SMPL_MODEL_DIR = os.path.join(ADDITIONAL_DIR, 'smpl')
SMPL_FACES_PATH = os.path.join(ADDITIONAL_DIR, 'smpl_faces.npy')
SMPL_MEAN_PARAMS_PATH = os.path.join(ADDITIONAL_DIR, 'neutral_smpl_mean_params_6dpose.npz')
J_REGRESSOR_EXTRA_PATH = os.path.join(ADDITIONAL_DIR, 'J_regressor_extra.npy')
COCOPLUS_REGRESSOR_PATH = os.path.join(ADDITIONAL_DIR, 'cocoplus_regressor.npy')
H36M_REGRESSOR_PATH = os.path.join(ADDITIONAL_DIR, 'J_regressor_h36m.npy')
VERTEX_TEXTURE_PATH = os.path.join(ADDITIONAL_DIR, 'vertex_texture.npy')
CUBE_PARTS_PATH = os.path.join(ADDITIONAL_DIR, 'cube_parts.npy')
SEGMENTATION_PATH = os.path.join(ADDITIONAL_DIR, 'smpl_body_parts_2_faces.json')

# data dir
DATA_DIR = '/home/shin/VScodeProjects/fittering-ML/data/source'
# ordinary data path
ORD_DATA_PATH = os.path.join(DATA_DIR, 'amass_up3d_3dpw_train.npz')
# generated data path
GEN_DATA_DIR = os.path.join(DATA_DIR, 'gen_data')
GEN_TRAIN_DIR = os.path.join(GEN_DATA_DIR, 'train')
GEN_TEST_DIR = os.path.join(GEN_DATA_DIR, 'test')


# ------------------------ label order ------------------------
MEASUREMENTS_ORDER = [
        "height",
        "chest circumference",
        "waist circumference",
        "hip circumference",
        "thigh left circumference",
        "arm left length",
        "inside leg height",
        "shoulder breadth",
]

# ------------------------ Constants ------------------------
FOCAL_LENGTH = 5000.
REGRESSOR_IMG_WH = 512

# ------------------------ Joint label conventions ------------------------
# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.
# You may wish to use different subsets in accordance with your training data/inference needs.

# The joints superset is broken down into: 45 SMPL joints (24 standard + additional fingers/toes/face),
# 9 extra joints, 19 cocoplus joints and 17 H36M joints.
# The 45 SMPL joints are converted to COCO joints with the map below.
# (Not really sure how coco and cocoplus are related.)

# Indices to get 17 COCO joints and 17 H36M joints from joints superset.
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


remove_appendages = False
deviate_joints2D = False
deviate_verts2D = False
occlude_seg = False
remove_appendages_classes = [1, 2, 3, 4, 5, 6]
remove_appendages_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
delta_j2d_dev_range = [-8, 8]
delta_j2d_hip_dev_range = [-8, 8]
delta_verts2d_dev_range = [-0.01, 0.01]
occlude_probability = 0.5
occlude_box_dim = 48

