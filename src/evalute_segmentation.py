import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label


def calculate_metrics(predicted, manual):
    """
    Calculate TP, FP, FN, TN, Dice coefficient, and Hausdorff distance
    between predicted and manual segmentations.

    Parameters:
    predicted (ndarray): 3D array of predicted segmentation
    manual (ndarray): 3D array of manual segmentation

    Returns:
    dict: Dictionary containing TP, FP, FN, TN, Dice coefficient, and Hausdorff distance
    """

    # Ensure the input arrays are binary
    predicted = (predicted > 0).astype(int)
    manual = (manual > 0).astype(int)

    # Calculate TP, FP, FN, TN
    TP = np.sum((predicted == 1) & (manual == 1))
    FP = np.sum((predicted == 1) & (manual == 0))
    FN = np.sum((predicted == 0) & (manual == 1))
    TN = np.sum((predicted == 0) & (manual == 0))

    # Calculate Dice coefficient
    dice = (2.0 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    # Calculate Hausdorff distance
    def surface_distances(pred, true):
        pred_surface = label(pred) & ~label(np.roll(pred, 1, axis=0)) & ~label(np.roll(pred, -1, axis=0)) & \
                       ~label(np.roll(pred, 1, axis=1)) & ~label(np.roll(pred, -1, axis=1)) & \
                       ~label(np.roll(pred, 1, axis=2)) & ~label(np.roll(pred, -1, axis=2))
        true_surface = label(true) & ~label(np.roll(true, 1, axis=0)) & ~label(np.roll(true, -1, axis=0)) & \
                       ~label(np.roll(true, 1, axis=1)) & ~label(np.roll(true, -1, axis=1)) & \
                       ~label(np.roll(true, 1, axis=2)) & ~label(np.roll(true, -1, axis=2))

        pred_points = np.column_stack(np.where(pred_surface))
        true_points = np.column_stack(np.where(true_surface))

        return directed_hausdorff(pred_points, true_points)[0], directed_hausdorff(true_points, pred_points)[0]

    hausdorff_distance = max(surface_distances(predicted, manual))

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Dice coefficient": dice,
        "Hausdorff distance": hausdorff_distance
    }

