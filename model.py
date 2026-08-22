from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class PC2FoodNet(nn.Module):
    """Physics-Constrained, Confidence-Calibrated multi-task food model.

    It predicts food class, volume, weight, and energy without receiving the
    ground-truth label at inference. Class probabilities define expected
    density and kcal/g priors; learnable residual gates correct those priors.
    """

    def __init__(self, num_classes: int, densities: torch.Tensor, kcal_per_g: torch.Tensor,
                 pretrained: bool = True, projection_dim: int = 512,
                 dropout: float = 0.30, residual_bound: float = 0.20):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim, num_classes)

        def regression_head(out_dim: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(projection_dim, projection_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim // 2, out_dim),
            )

        self.volume_head = regression_head(2)       # raw mean, log variance
        self.weight_direct_head = regression_head(2)
        self.energy_direct_head = regression_head(2)
        self.weight_residual_head = regression_head(1)
        self.energy_residual_head = regression_head(1)
        self.weight_gate_head = regression_head(1)
        self.energy_gate_head = regression_head(1)

        self.residual_bound = float(residual_bound)
        self.register_buffer("densities", densities.float())
        self.register_buffer("kcal_per_g", kcal_per_g.float())

    @staticmethod
    def _positive(raw: torch.Tensor) -> torch.Tensor:
        # Interpret the raw output as an unconstrained log-scale parameter.
        # This avoids requiring a linear head to emit values in the hundreds.
        return torch.expm1(F.softplus(raw).clamp(max=10.0)) + 1e-4

    def initialize_regression_biases(self, median_volume: float, median_weight: float,
                                     median_energy: float) -> None:
        def inverse_positive(value: float) -> float:
            """
            Invert the full _positive activation:
              _positive(x) = expm1(softplus(x).clamp(max=10)) + 1e-4

            Steps (in reverse):
              1. y     = value - 1e-4          (undo the shift)
              2. sp    = log1p(y).clamp(max=10) (undo expm1  → log1p)
              3. bias  = softplus⁻¹(sp)         (undo softplus → log(expm1(·)))
            """
            y  = torch.tensor(float(value)) - 1e-4
            sp = torch.log1p(y.clamp_min(1e-8)).clamp(max=10.0)
            # softplus⁻¹: for large sp use sp directly (numerically stable)
            bias = torch.where(sp > 20, sp, torch.log(torch.expm1(sp.clamp_min(1e-6))))
            return float(bias)

        for head, value in [
            (self.volume_head, median_volume),
            (self.weight_direct_head, median_weight),
            (self.energy_direct_head, median_energy),
        ]:
            final = head[-1]
            with torch.no_grad():
                final.bias[0] = inverse_positive(value)
                final.bias[1] = -1.0

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.projector(self.backbone(images))
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)

        volume_raw = self.volume_head(features)
        weight_raw = self.weight_direct_head(features)
        energy_raw = self.energy_direct_head(features)

        volume = self._positive(volume_raw[:, 0])
        direct_weight = self._positive(weight_raw[:, 0])
        direct_energy = self._positive(energy_raw[:, 0])

        expected_density = probs @ self.densities
        expected_kcal_per_g = probs @ self.kcal_per_g

        physics_weight = volume * expected_density
        weight_correction = torch.exp(
            self.residual_bound * torch.tanh(self.weight_residual_head(features).squeeze(1))
        )
        physics_weight = physics_weight * weight_correction
        weight_gate = torch.sigmoid(self.weight_gate_head(features).squeeze(1))
        weight = weight_gate * direct_weight + (1.0 - weight_gate) * physics_weight

        physics_energy = weight * expected_kcal_per_g
        energy_correction = torch.exp(
            self.residual_bound * torch.tanh(self.energy_residual_head(features).squeeze(1))
        )
        physics_energy = physics_energy * energy_correction
        energy_gate = torch.sigmoid(self.energy_gate_head(features).squeeze(1))
        energy = energy_gate * direct_energy + (1.0 - energy_gate) * physics_energy

        return {
            "features": features,
            "logits": logits,
            "probs": probs,
            "volume": volume,
            "weight": weight,
            "energy": energy,
            "direct_weight": direct_weight,
            "direct_energy": direct_energy,
            "physics_weight": physics_weight,
            "physics_energy": physics_energy,
            "expected_density": expected_density,
            "expected_kcal_per_g": expected_kcal_per_g,
            "weight_gate": weight_gate,
            "energy_gate": energy_gate,
            "logvar_volume": volume_raw[:, 1].clamp(-6.0, 6.0),
            "logvar_weight": weight_raw[:, 1].clamp(-6.0, 6.0),
            "logvar_energy": energy_raw[:, 1].clamp(-6.0, 6.0),
        }
