import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def preds_map(preds, targets, map_dim, preds_offset, targets_offset):
    """
    Mapping Pred Optimal to Target based on Hungarian Algorithm
    map_dim: number of calsses to map
    preds_offset: Starting the preds from 0 through preds_offset
    targets_offset: Starting the targets from 0 through preds_offset
    """
    # convert tensor to numpy
    preds = preds.cpu().numpy().astype(np.int64)
    targets = targets.cpu().numpy().astype(np.int64)
    # building a mapping matrix
    matrix = np.zeros((map_dim, map_dim), dtype=np.int64)
    for i in range(preds.size):
        matrix[preds[i] - preds_offset, targets[i] - targets_offset] += 1
    # optimal transmission
    pred_class, pred_class_map = linear_sum_assignment(matrix.max() - matrix)
    # building mapping dictionary
    preds_map_dict = {}
    for i in range(map_dim):
        preds_map_dict[int(pred_class[i] + preds_offset)] = pred_class_map[i] + targets_offset
    return preds_map_dict


def set_seed(seed_value):
    random.seed(seed_value) 
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 

    if torch.cuda.is_available():
        print('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False