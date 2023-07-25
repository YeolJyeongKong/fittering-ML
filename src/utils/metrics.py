import numpy as np
import torch
from torchmetrics import Metric

from utils.measure import MeasureVerts
from utils.predict_measure import Beta2Measurements


class AccuracyBinaryImage:
    def __init__(self):
        self.total_score = 0
        self.total_cnt = 0

    def metric(self, pred, label):
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        score = np.mean(((pred > 0.5).astype(np.int) == label).astype(np.int))
        self.total_score += score
        self.total_cnt += 1
        return score

    def compute_score(self):
        tmp_cnt, tmp_score = self.total_cnt, self.total_score
        self.total_cnt = 0
        self.total_score = 0
        return tmp_score / tmp_cnt


class MeasureMAE(Metric):
    def __init__(self, device):
        super().__init__()
        self.beta2meas = Beta2Measurements(device)
        self.add_state("measure_mae", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_betas: torch.Tensor, target_betas: torch.Tensor):
        preds_meas = self.beta2meas.predict(preds_betas)
        target_meas = self.beta2meas.predict(target_betas)

        total_mae = np.mean(np.abs(preds_meas.values - target_meas.values))

        self.measure_mae += total_mae.astype(np.longlong)
        self.total += 1

    def compute(self):
        return self.measure_mae / self.total
