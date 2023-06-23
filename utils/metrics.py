import numpy as np
class AccuracyBinaryImage:
    def __init__(self):
        self.total_score = 0
        self.total_cnt = 0

    def metric(self, pred, label):
        pred = pred.detach().cpu().numpy()
        label = pred.detach().cpu().numpy()
        score = np.mean(((pred > 0.5).astype(np.int) == label).astype(np.int))
        self.total_score += score
        self.total_cnt += 1
        return score
    
    def compute_score(self):
        tmp_cnt, tmp_score = self.total_cnt, self.total_score
        self.total_cnt = 0
        self.total_score = 0
        return tmp_score / tmp_cnt
