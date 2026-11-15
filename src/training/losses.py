from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss with per-sample margin_alpha of shape (B,),
    numerical stability for Euclidean distance near zero,
    and optional anti-collapse latent variance regularization.
    """

    def __init__(
        self,
        eps: float = 1e-12,
        var_reg_weight: float = 0.0,
        var_reg_target: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.var_reg_weight = float(var_reg_weight)
        self.var_reg_target = float(var_reg_target)

    def forward(
        self,
        z_A: torch.Tensor,
        z_P: torch.Tensor,
        z_N: torch.Tensor,
        margin_alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the adaptive Triplet Margin Loss for each element in the batch.
        Returns (loss, stats) where loss is a differentiable scalar.
        """
        assert z_A.ndim == 2 and z_A.shape == z_P.shape == z_N.shape, (
            f"Incompatible embedding shapes: A={tuple(z_A.shape)}, P={tuple(z_P.shape)}, N={tuple(z_N.shape)}."
        )
        margins = margin_alpha.to(device=z_A.device, dtype=z_A.dtype).view(-1)
        assert margins.shape[0] == z_A.shape[0], (
            f"margin_alpha batch size ({margins.shape[0]}) != z_A batch size ({z_A.shape[0]})."
        )

        d_ap = torch.sqrt(torch.sum((z_A - z_P) ** 2, dim=-1) + self.eps)
        d_an = torch.sqrt(torch.sum((z_A - z_N) ** 2, dim=-1) + self.eps)

        losses = F.relu(d_ap - d_an + margins)
        triplet_loss = torch.mean(losses)

        loss = triplet_loss
        if self.var_reg_weight > 0.0 and z_A.shape[0] > 1:
            z_all = torch.cat([z_A, z_P, z_N], dim=0)
            std_per_dim = torch.sqrt(z_all.var(dim=0, unbiased=False) + self.eps)
            var_loss = torch.mean(F.relu(self.var_reg_target - std_per_dim))
            loss = triplet_loss + self.var_reg_weight * var_loss

        with torch.no_grad():
            triplet_acc = float(torch.mean((d_ap < d_an).float()).item())
            margin_acc = float(torch.mean((d_ap + margins < d_an).float()).item())
            stats = {
                "loss": float(triplet_loss.item()),
                "triplet_acc": triplet_acc,
                "margin_satisfied_acc": margin_acc,
                "mean_d_ap": float(d_ap.mean().item()),
                "mean_d_an": float(d_an.mean().item()),
            }
        return loss, stats

