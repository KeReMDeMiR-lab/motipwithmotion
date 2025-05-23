# models/motip/id_decoder_with_motion.py
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
import einops
from models.motip.id_decoder_with_motion import IDDecoder


class IDDecoderWithMotion(IDDecoder):
    def __init__(
            self,
            feature_dim: int,
            id_dim: int,
            motion_dim: int,
            ffn_dim_ratio: int,
            num_layers: int,
            head_dim: int,
            num_id_vocabulary: int,
            rel_pe_length: int,
            use_aux_loss: bool,
            use_shared_aux_head: bool,
            motion_weight: float = 0.5,
    ):
        # Adjust dimensions to account for motion features
        super().__init__(
            feature_dim=feature_dim + motion_dim,
            id_dim=id_dim,
            ffn_dim_ratio=ffn_dim_ratio,
            num_layers=num_layers,
            head_dim=head_dim,
            num_id_vocabulary=num_id_vocabulary,
            rel_pe_length=rel_pe_length,
            use_aux_loss=use_aux_loss,
            use_shared_aux_head=use_shared_aux_head,
        )
        self.motion_dim = motion_dim
        self.motion_weight = motion_weight
        
        # Motion-aware attention gate
        self.motion_gate = nn.Sequential(
            nn.Linear(motion_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_info, use_decoder_checkpoint):
        # Extract motion features if available
        trajectory_motion = seq_info.get("trajectory_motion_features", None)
        unknown_motion = seq_info.get("unknown_motion_features", None)
        
        if trajectory_motion is not None and unknown_motion is not None:
            # Apply motion gating
            trajectory_gate = self.motion_gate(trajectory_motion)
            unknown_gate = self.motion_gate(unknown_motion)
            
            # Weight motion contribution
            trajectory_motion = trajectory_motion * trajectory_gate * self.motion_weight
            unknown_motion = unknown_motion * unknown_gate * self.motion_weight
            
            # Concatenate motion features with appearance features
            seq_info["trajectory_features"] = torch.cat([
                seq_info["trajectory_features"], 
                trajectory_motion
            ], dim=-1)
            seq_info["unknown_features"] = torch.cat([
                seq_info["unknown_features"], 
                unknown_motion
            ], dim=-1)
        
        # Call parent forward method
        return super().forward(seq_info, use_decoder_checkpoint)