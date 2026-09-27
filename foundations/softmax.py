import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res=[]
        # sum_e=0
        # for i in range(len(z)):
        #     sum_e=sum_e+np.exp(z[i])
        # for j in range(len(z)):
        #     x=np.exp(z[j])/round(sum_e,4)
        #     res.append(round(x,4))
        exp_z = np.exp(z - np.max(z))
        res = exp_z / np.sum(exp_z)
        
        return np.round(res, 4)


