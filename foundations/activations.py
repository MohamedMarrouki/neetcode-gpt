import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        res=[]
        for i in range(len(z)):
            x=1/(1+np.exp(-z[i]))
            res.append(round(x,5))

        return res


    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        res=[]
        for i in range(len(z)):
            x=max(0.0,z[i])
            res.append(x)

        return res
