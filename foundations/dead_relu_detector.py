import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.ReLU):

                    # Check if each neuron outputs 0
                    # for ALL samples in the batch
                    dead = (x == 0).all(dim=0)

                    # Fraction of dead neurons
                    dead_fraction = dead.float().mean().item()

                    dead_fractions.append(
                        round(dead_fraction, 4)
                    )

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return "healthy"

        # 1. Severe neuron death
        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

        # 2. First layer affected
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Increasing neuron death with depth
        increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if (
            len(dead_fractions) > 1
            and increasing
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # 4 & 5. Healthy
        return "healthy"
