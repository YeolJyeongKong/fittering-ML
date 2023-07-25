import torch
import torch.nn as nn

from src.SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
from src.SemanticGuidedHumanMatting import utils
from src.SemanticGuidedHumanMatting import inference
from extras import paths


class InferenceSegment:
    def __init__(self, segmodel_path=paths.SEGMODEL_PATH, device=torch.device("cpu")):
        self.device = device

        seg_model = HumanMatting(backbone="resnet50")
        self.seg_model = nn.DataParallel(seg_model).eval()
        self.seg_model.load_state_dict(
            torch.load(segmodel_path, map_location=self.device)
        )

    def predict(self, img):
        pred_alpha, pred_mask = inference.single_inference(
            self.seg_model, img, self.device
        )

        return pred_alpha
