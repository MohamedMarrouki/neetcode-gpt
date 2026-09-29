import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x_arr = np.array(x, dtype=np.float64)
        W1_arr = np.array(W1, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        W2_arr = np.array(W2, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_arr = np.array(y_true, dtype=np.float64)

        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)

        z2 = np.dot(W2_arr, a1) + b2_arr

        loss_f = np.mean((z2 - y_arr) ** 2)

        
        dz2 = 2 * (z2 - y_arr) / len(y_arr)

        db2 = dz2
        dW2 = np.outer(dz2, a1)
        da1 = np.dot(dz2, W2_arr)
        dz1 = da1 * (z1 > 0).astype(float)
        db1 = dz1


        dW1 = np.outer(dz1, x_arr)

        return { 'loss':np.round(loss_f,4),
                'dW2': np.round(dW2, 4).tolist(),
                'dW1': np.round(dW1, 4).tolist(),
                'db1': np.round(db1, 4).tolist(),

                'db2': np.round(db2, 4).tolist()}
