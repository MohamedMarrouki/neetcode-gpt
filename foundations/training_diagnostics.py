import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []

        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    # Mean and standard deviation
                    mean = x.mean().item()
                    std = x.std().item()

                    # A neuron is dead if its output <= 0
                    # for every sample in the batch
                    dead = (x <= 0).all(dim=0)
                    dead_fraction = dead.float().mean().item()

                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(
                            dead_fraction, 4
                        )
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        # Forward pass
        y_hat = model(x)

        # Compute loss
        loss = nn.MSELoss()(y_hat, y)

        # Backward pass
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(
                        torch.norm(grad).item(), 4
                    )
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        if any(
            s["dead_fraction"] > 0.5
            for s in activation_stats
        ):
            return "dead_neurons"

        # 2. Exploding gradients
        if any(
            s["norm"] > 1000
            for s in gradient_stats
        ):
            return "exploding_gradients"

        # 3. Vanishing gradients (last layer)
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation standard deviation
        if any(
            s["std"] < 0.1
            for s in activation_stats
        ):
            return "vanishing_gradients"

        if any(
            s["std"] > 10.0
            for s in activation_stats
        ):
            return "exploding_gradients"

        # 5. Healthy network
        return "healthy"
