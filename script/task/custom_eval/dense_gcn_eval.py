import torch
import numpy as np


def geometric_segmentation_accuracy(pred, gt, num_classes):  
    Is = np.empty(num_classes)
    Us = np.empty(num_classes)

    pred_np = pred.cpu().numpy()
    target_np = gt.cpu().numpy()

    for cl in range(num_classes):
        cur_gt_mask = (target_np == cl)
        cur_pred_mask = (pred_np == cl)
        I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
        U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
        Is[cl] = I
        Us[cl] = U

    return Is, Us
