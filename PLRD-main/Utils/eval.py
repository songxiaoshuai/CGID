import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def cluster_acc(y_true, y_pred):
    # mapping[[0,1],……] The first element is preds, and the second element is the true of the mapping
    mapping, w = compute_best_mapping(y_true, y_pred)
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size, mapping

def cluster_F1(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    F1_score = f1_score(y_true, y_pred_aligned, average='weighted')
    return F1_score

def cluster_precision(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    precision = precision_score(y_true, y_pred_aligned, average='weighted')
    return precision

def cluster_recall(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    recall = recall_score(y_true, y_pred_aligned, average='weighted')
    return recall

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class ClassifyMetrics:
    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)

        t = targets.cpu().numpy()
        p = preds.cpu().numpy()
        acc = round(accuracy_score(t, p) * 100, 2)
        f1 = round(f1_score(t, p, average='weighted') * 100, 2)
        precision = round(precision_score(t, p, average='weighted') * 100, 2)
        recall = round(recall_score(t, p, average='weighted') * 100, 2)

        # reset
        self.preds = []
        self.targets = []

        return {"acc": acc, "f1": f1, "pre": precision, "rec": recall,
                "n_preds_right": int(accuracy_score(t, p) * len(preds)),
                "n_preds": len(preds)}


class ClusterMetrics:
    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)

        # Filter out preds within [target_min, target_max], and preds outside this range must have predicted errors
        select = ((preds >= targets.min()) & (preds <= targets.max()))
        preds_select = preds[select] - targets.min()
        targets_select = targets[select] - targets.min()

        n_select = len(preds_select)
        n_all = len(preds)
        t = targets_select.cpu().numpy()
        p = preds_select.cpu().numpy()
        acc, nmi, ari, f1, precision, recall = 0, 0, 0, 0, 0, 0

        if n_select > 0:
            accuracy, mapping = cluster_acc(t, p)
            acc = round(accuracy * (n_select / n_all) * 100, 2)
            nmi = round(nmi_score(t, p) * (n_select / n_all) * 100, 2)
            ari = round(ari_score(t, p) * (n_select / n_all) * 100, 2)
            f1 = round(cluster_F1(t, p) * (n_select / n_all) * 100, 2)
            precision = round(cluster_precision(t, p) * (n_select / n_all) * 100, 2)
            recall = round(cluster_recall(t, p) * (n_select / n_all) * 100, 2)
            # Store the mapping relationship between preds and targets
            map_dict = {}
            for i, j in mapping:
                map_dict[i + targets.min().item()] = j + targets.min().item()

        # reset
        self.preds = []
        self.targets = []

        return {"acc": acc, "nmi": nmi, "ari": ari, "f1": f1, "pre": precision, "rec": recall,
                "n_preds_right": int(accuracy * n_select) if n_select > 0 else 0,
                "n_preds": n_all, "map_dict": map_dict if n_select > 0 else {}}
