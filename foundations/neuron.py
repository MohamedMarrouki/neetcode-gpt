import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        output_before_act=2.3
        output=1.3
        output_before_act=np.dot(x,w)+b
        if activation=='sigmoid':
            output=1/(1+np.exp(-output_before_act))
            return float(round(output,5))
        elif activation=='relu':
            output=max(0,output_before_act)
            return float(round(output,5))
        else:
            return (-1)
