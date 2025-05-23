# models/motip/motion_modeling.py
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from models.misc import _get_clones
from models.ffn import FFN


class MotionFeatureExtractor(nn.Module):
    """
    Extracts motion features from consecutive frames using:
    1. Optical flow-like features from consecutive bounding boxes
    2. Velocity and acceleration estimation
    3. Motion pattern encoding
    """
    def __init__(
        self,
        feature_dim: int = 256,
        motion_dim: int = 128,
        num_motion_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        
        # Box motion encoder (velocity, acceleration)
        self.box_motion_encoder = nn.Sequential(
            nn.Linear(8, motion_dim),  # dx, dy, dw, dh, ddx, ddy, ddw, ddh
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(motion_dim, motion_dim),
        )
        
        # Appearance motion encoder (feature difference)
        self.appearance_motion_encoder = nn.Sequential(
            nn.Linear(feature_dim, motion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(motion_dim, motion_dim),
        )
        
        # Motion pattern encoder
        self.motion_pattern_layers = _get_clones(
            nn.TransformerEncoderLayer(
                d_model=motion_dim,
                nhead=8,
                dim_feedforward=motion_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_motion_layers
        )
        
        # Fusion layer
        self.motion_fusion = nn.Sequential(
            nn.Linear(motion_dim * 2, motion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(motion_dim, motion_dim),
        )
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def compute_box_motion(self, boxes, masks):
        """
        Compute velocity and acceleration from bounding boxes
        boxes: (B, G, T, N, 4)
        masks: (B, G, T, N)
        """
        B, G, T, N, _ = boxes.shape
        
        # Compute velocity (first-order difference)
        velocity = torch.zeros(B, G, T, N, 4, device=boxes.device)
        velocity[:, :, 1:] = boxes[:, :, 1:] - boxes[:, :, :-1]
        
        # Compute acceleration (second-order difference)
        acceleration = torch.zeros(B, G, T, N, 4, device=boxes.device)
        acceleration[:, :, 2:] = velocity[:, :, 2:] - velocity[:, :, 1:-1]
        
        # Concatenate motion features
        motion_features = torch.cat([velocity, acceleration], dim=-1)  # (B, G, T, N, 8)
        
        # Mask invalid motions
        motion_features = motion_features * (~masks).unsqueeze(-1).float()
        
        return motion_features
    
    def forward(self, features, boxes, masks):
        """
        Extract motion features from consecutive frames
        features: (B, G, T, N, D)
        boxes: (B, G, T, N, 4)
        masks: (B, G, T, N)
        """
        B, G, T, N, D = features.shape
        
        # Compute box-based motion features
        box_motion = self.compute_box_motion(boxes, masks)
        box_motion_encoded = self.box_motion_encoder(box_motion)  # (B, G, T, N, motion_dim)
        
        # Compute appearance-based motion features
        appearance_diff = torch.zeros_like(features)
        appearance_diff[:, :, 1:] = features[:, :, 1:] - features[:, :, :-1]
        appearance_motion_encoded = self.appearance_motion_encoder(appearance_diff)  # (B, G, T, N, motion_dim)
        
        # Reshape for temporal modeling
        box_motion_flat = einops.rearrange(box_motion_encoded, 'b g t n d -> (b g n) t d')
        appearance_motion_flat = einops.rearrange(appearance_motion_encoded, 'b g t n d -> (b g n) t d')
        mask_flat = einops.rearrange(masks, 'b g t n -> (b g n) t')
        
        # Encode motion patterns
        for layer in self.motion_pattern_layers:
            box_motion_flat = layer(box_motion_flat, src_key_padding_mask=mask_flat)
            appearance_motion_flat = layer(appearance_motion_flat, src_key_padding_mask=mask_flat)
        
        # Reshape back
        box_motion_pattern = einops.rearrange(box_motion_flat, '(b g n) t d -> b g t n d', b=B, g=G, n=N)
        appearance_motion_pattern = einops.rearrange(appearance_motion_flat, '(b g n) t d -> b g t n d', b=B, g=G, n=N)
        
        # Fuse motion features
        motion_features = self.motion_fusion(
            torch.cat([box_motion_pattern, appearance_motion_pattern], dim=-1)
        )
        
        return motion_features


class KalmanFilter(nn.Module):
    """
    Learnable Kalman Filter for motion prediction
    """
    def __init__(self, state_dim: int = 8, obs_dim: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Learnable transition and observation matrices
        self.F = nn.Parameter(torch.eye(state_dim))  # State transition
        self.H = nn.Parameter(torch.zeros(obs_dim, state_dim))  # Observation model
        self.H.data[:obs_dim, :obs_dim] = torch.eye(obs_dim)
        
        # Learnable noise covariances
        self.Q = nn.Parameter(torch.eye(state_dim) * 0.1)  # Process noise
        self.R = nn.Parameter(torch.eye(obs_dim) * 0.1)    # Measurement noise
        
    def forward(self, state, covariance, observation=None):
        """
        Kalman filter prediction and update
        state: (B, N, state_dim)
        covariance: (B, N, state_dim, state_dim)
        observation: (B, N, obs_dim) or None
        """
        # Prediction step
        state_pred = torch.matmul(self.F.unsqueeze(0).unsqueeze(0), state.unsqueeze(-1)).squeeze(-1)
        cov_pred = torch.matmul(torch.matmul(self.F.unsqueeze(0).unsqueeze(0), covariance), 
                                self.F.t().unsqueeze(0).unsqueeze(0)) + self.Q
        
        if observation is not None:
            # Update step
            y = observation - torch.matmul(self.H.unsqueeze(0).unsqueeze(0), state_pred.unsqueeze(-1)).squeeze(-1)
            S = torch.matmul(torch.matmul(self.H.unsqueeze(0).unsqueeze(0), cov_pred), 
                           self.H.t().unsqueeze(0).unsqueeze(0)) + self.R
            K = torch.matmul(torch.matmul(cov_pred, self.H.t().unsqueeze(0).unsqueeze(0)), torch.inverse(S))
            
            state = state_pred + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
            covariance = torch.matmul(torch.eye(self.state_dim).unsqueeze(0).unsqueeze(0).to(K.device) - 
                                    torch.matmul(K, self.H.unsqueeze(0).unsqueeze(0)), cov_pred)
        else:
            state = state_pred
            covariance = cov_pred
            
        return state, covariance