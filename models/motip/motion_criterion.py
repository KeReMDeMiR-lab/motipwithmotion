# models/motip/motion_criterion.py
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionConsistencyLoss(nn.Module):
    """
    Ensures motion predictions are consistent with actual movements
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, predicted_motion, actual_boxes, masks):
        """
        predicted_motion: (B, G, T, N, 4) - predicted box positions
        actual_boxes: (B, G, T, N, 4) - actual box positions
        masks: (B, G, T, N)
        """
        # Only compute loss for valid boxes
        valid_mask = ~masks
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted_motion.device)
        
        # L1 loss for position prediction
        position_loss = F.l1_loss(
            predicted_motion[valid_mask],
            actual_boxes[valid_mask],
            reduction='mean'
        )
        
        # Smoothness loss for motion
        motion_diff = predicted_motion[:, :, 1:] - predicted_motion[:, :, :-1]
        motion_diff_actual = actual_boxes[:, :, 1:] - actual_boxes[:, :, :-1]
        
        valid_motion_mask = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
        
        if valid_motion_mask.sum() > 0:
            smoothness_loss = F.l1_loss(
                motion_diff[valid_motion_mask],
                motion_diff_actual[valid_motion_mask],
                reduction='mean'
            )
        else:
            smoothness_loss = torch.tensor(0.0, device=predicted_motion.device)
        
        return self.weight * (position_loss + 0.5 * smoothness_loss)


def build_motion_criterion(config: dict):
    return MotionConsistencyLoss(
        weight=config.get("MOTION_LOSS_WEIGHT", 1.0)
    )