# ------------------------ fashion-cbf ------------------------
INFERENCE_BATCH_SIZE = 1
TOP_K = 12
RECOMMENDATION_N = 12


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

# aihub
CLOTH_CATEGORY = ["동복", "춘추복1", "춘추복2", "측정복", "하복1", "하복2"]
AIHUB_OUTPUT_ORDER = ["키", "젖가슴둘레", "허리둘레", "엉덩이둘레", "넙다리둘레", "팔길이", "엉덩이높이", "어깨사이너비"]

# ------------------------ Constants ------------------------
FOCAL_LENGTH = 5000.0
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


# ------------------------ aws ------------------------
BUCKET_NAME_HUMAN = "fittering-images-bucket"
S3_BUCKET_PATH_BODY = "body/"
S3_BUCKET_PATH_SILHOUETTE = "silhouette/"
S3_URL_HUMAN = "https://fittering-measurements-images.s3.ap-northeast-2.amazonaws.com"

S3_BUCKET_NAME_PRODUCT = "fittering-images-bucket"
S3_BUCKET_PATH_PRODUCT = "images/"


# ------------------------ milvus ------------------------
MILVUS_COLLECTION_NAME = "image_embedding_collection"
# MILVUS_COLLECTION_NAME = "image_embedding_collection_test"
MILVUS_PRIVATE_HOST = "172.31.3.122"
MILVUS_PUBLIC_HOST = "13.125.214.45"
MILVUS_PORT = "19530"
