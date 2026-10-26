"""
Loss functions from Traffic_Model, adapted to the BasicTS calling convention.

BasicTS calls loss functions with keyword arguments matching the forward_return dict:
    loss_func(prediction=..., target=..., null_val=...)

The original Traffic_Model code uses y_pred / y_true parameter names.
This module re-exports the same logic with BasicTS-compatible signatures.
"""

import numpy as np
import torch


def masked_mae_loss(prediction: torch.Tensor, target: torch.Tensor,
                    null_val: float = 0.0) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = target != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction - target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_trend_loss(prediction: torch.Tensor, target: torch.Tensor,
                      null_val: float = 0.0, lambda_trend: float = 0.5) -> torch.Tensor:
    """
    Point-wise MAE + first-order difference MAE.

    Penalises both absolute value errors and trend-direction errors.
    Delta[t] = pred[t] - pred[t-1] is constrained to match the true Delta[t].

    Args:
        prediction: Model predictions, shape [B, L, N, C].
        target: Ground truth, shape [B, L, N, C].
        null_val: Value representing missing data.
        lambda_trend: Weight for the first-order difference term.
    """
    if np.isnan(null_val):
        mask = (~torch.isnan(target)).float()
    else:
        mask = (target != null_val).float()

    # element-wise MAE
    elem_loss = (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp(min=1)

    # first-order difference along time dimension (dim=1)
    pred_diff = prediction[:, 1:, :, :] - prediction[:, :-1, :, :]
    true_diff = target[:, 1:, :, :] - target[:, :-1, :, :]
    diff_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]
    trend_loss = (torch.abs(pred_diff - true_diff) * diff_mask).sum() / diff_mask.sum().clamp(min=1)

    return elem_loss + lambda_trend * trend_loss
