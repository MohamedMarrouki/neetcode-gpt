import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        y_pred=[]
        y_pred=np.dot(X,weights)
        y_pred_r=[]
        for i in range(len(y_pred)):
            y_pred_r.append(round(y_pred[i],5))

        return y_pred_r

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        MSE = np.mean((model_prediction-ground_truth)**2)
        return round(MSE,5)
