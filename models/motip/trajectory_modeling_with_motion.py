# models/motip/trajectory_modeling_with_motion.py
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
import einops
from models.ffn import FFN
from models.motip.motion_modeling import MotionFeatureExtractor, KalmanFilter


class TrajectoryModelingWithMotion(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            motion_dim: int = 128,
            use_kalman: bool = True,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.use_kalman = use_kalman

        # Original appearance adapter
        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        
        # Motion feature extractor
        self.motion_extractor = MotionFeatureExtractor(
            feature_dim=feature_dim,
            motion_dim=motion_dim,
            num_motion_layers=3,
        )
        
        # Kalman filter for motion prediction
        if self.use_kalman:
            self.kalman_filter = KalmanFilter(state_dim=8, obs_dim=4)
            self.kalman_states = {}  # Store states per trajectory
            self.kalman_covariances = {}  # Store covariances per trajectory
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + motion_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
        )
        
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_info):
        trajectory_features = seq_info["trajectory_features"]
        trajectory_boxes = seq_info.get("trajectory_boxes", None)
        trajectory_masks = seq_info.get("trajectory_masks", None)
        
        B, G, T, N, _ = trajectory_features.shape
        
        # Apply appearance adapter
        trajectory_features = trajectory_features + self.adapter(trajectory_features)
        trajectory_features = self.norm(trajectory_features)
        
        # Extract motion features if boxes are available
        if trajectory_boxes is not None and trajectory_masks is not None:
            motion_features = self.motion_extractor(
                trajectory_features, trajectory_boxes, trajectory_masks
            )
            
            # Apply Kalman filtering for motion prediction
            if self.use_kalman and not self.training:
                # Initialize or update Kalman states
                for b in range(B):
                    for g in range(G):
                        for n in range(N):
                            key = (b, g, n)
                            if key not in self.kalman_states:
                                # Initialize state [x, y, w, h, vx, vy, vw, vh]
                                self.kalman_states[key] = torch.zeros(8, device=trajectory_boxes.device)
                                self.kalman_states[key][:4] = trajectory_boxes[b, g, -1, n]
                                self.kalman_covariances[key] = torch.eye(8, device=trajectory_boxes.device)
                            
                            # Update with current observation
                            obs = trajectory_boxes[b, g, -1, n]
                            if not trajectory_masks[b, g, -1, n]:
                                state, cov = self.kalman_filter(
                                    self.kalman_states[key].unsqueeze(0).unsqueeze(0),
                                    self.kalman_covariances[key].unsqueeze(0).unsqueeze(0),
                                    obs.unsqueeze(0).unsqueeze(0)
                                )
                                self.kalman_states[key] = state.squeeze()
                                self.kalman_covariances[key] = cov.squeeze()
            
            # Fuse appearance and motion features
            trajectory_features = self.feature_fusion(
                torch.cat([trajectory_features, motion_features], dim=-1)
            )
        
        # Apply FFN
        trajectory_features = trajectory_features + self.ffn(trajectory_features)
        trajectory_features = self.ffn_norm(trajectory_features)
        
        seq_info["trajectory_features"] = trajectory_features
        return seq_info




# # Copyright (c) Ruopeng Gao. All Rights Reserved.

# import torch.nn as nn

# from models.ffn import FFN


# class TrajectoryModeling(nn.Module):
#     def __init__(
#             self,
#             detr_dim: int,
#             ffn_dim_ratio: int,
#             feature_dim: int,
#     ):
#         super().__init__()

#         self.detr_dim = detr_dim
#         self.ffn_dim_ratio = ffn_dim_ratio
#         self.feature_dim = feature_dim

#         self.adapter = FFN(
#             d_model=detr_dim,
#             d_ffn=detr_dim * ffn_dim_ratio,
#             activation=nn.GELU(),
#         )
#         self.norm = nn.LayerNorm(feature_dim)
#         self.ffn = FFN(
#             d_model=feature_dim,
#             d_ffn=feature_dim * ffn_dim_ratio,
#             activation=nn.GELU(),
#         )
#         self.ffn_norm = nn.LayerNorm(feature_dim)

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         pass

#     def forward(self, seq_info):
#         trajectory_features = seq_info["trajectory_features"]
#         trajectory_features = trajectory_features + self.adapter(trajectory_features)
#         trajectory_features = self.norm(trajectory_features)
#         trajectory_features = trajectory_features + self.ffn(trajectory_features)
#         trajectory_features = self.ffn_norm(trajectory_features)
#         seq_info["trajectory_features"] = trajectory_features
#         return seq_info
