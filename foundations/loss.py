import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        sum_ln=0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(len(y_true)):
            if y_true[i]==1:
                sum_ln=sum_ln+np.log(y_pred_clipped[i])
            else:
                sum_ln=sum_ln+np.log(1-y_pred_clipped[i])

        loss_f=-sum_ln/len(y_true)
        return round(loss_f,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_f = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return round(loss_f,4)

