import numpy as np

from numpy import count_nonzero as nnz


def compute_iou(gt, pred):
    """Compute the iou between a prediction and the ground truth.
    gt: Ground truth (np.array).
    pred: Prediction (np.array).
	Returns:
	iou: Intersection over union.
    """
    inter = np.count_nonzero(gt & pred)
    union = np.count_nonzero(gt | pred)
    iou = inter/float(union)
    return iou


def compute_accuracy(gt, pred):
    """Compute the iou between a prediction and the ground truth.
    gt: Ground truth (np.array).
    pred: Prediction (np.array).
	Returns:
	acc: Accuracy.
    """
    ref = np.count_nonzero(gt == pred)
    acc = ref/float(gt.size)
    return acc