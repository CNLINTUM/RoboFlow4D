"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""
import math

import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


def sinusoidal_position_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Standard transformer-style sinusoidal positional encoding.
    Returns [length, dim].
    """
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * -(math.log(10000.0) / dim)
    )  # [dim/2]

    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, dim]


class Action_policy(nn.Module):
    """
    Action head: 4D points flow + robot state -> chunked actions.

    Inputs:
        - flow_4d: [B, T, P, 3]  (P≈100, T≈20)
        - robot_states: [B, proprio_dim]

    Output:
        - actions: [B, T, action_dim]
    """

    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 55,
        point_feat_dim: int = 128,
        proprio_feat_dim: int = 128,
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.point_feat_dim = point_feat_dim
        self.proprio_feat_dim = proprio_feat_dim
        self.d_model = d_model

        # Encode each 3D point -> point_feat_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, point_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Encode robot proprio state -> proprio_feat_dim
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proprio_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Project (point_feat + proprio_feat) -> d_model for Transformer
        self.fuse_proj = nn.Linear(point_feat_dim + proprio_feat_dim, d_model)

        # Temporal Transformer over T ≈ 20
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Final linear to action_dim
        self.out_proj = MLPResNet(
            num_blocks=2, input_dim=d_model, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def forward(
        self,
        flow_4d: torch.Tensor,      # [B, T, P, 3]
        robot_states: torch.Tensor, # [B, proprio_dim]
    ) -> torch.Tensor:
        """
        Args:
            flow_4d: [B, T, P, 3]
            robot_states: [B, proprio_dim]
        Returns:
            actions: [B, T, action_dim]
        """
        B, T, P, C = flow_4d.shape
        assert C == 3, f"flow_4d last dim should be 3 (xyz), got {C}"
        assert robot_states.shape[0] == B, "Batch size of robot_states must match flow_4d"

        device = flow_4d.device
        dtype = flow_4d.dtype

        # --- 1) Point features per timestep: [B, T, P, 3] -> [B, T, point_feat_dim] ---
        points_flat = flow_4d.reshape(B * T * P, 3)
        point_feats_flat = self.point_mlp(points_flat)  # [B*T*P, point_feat_dim]
        point_feats = point_feats_flat.reshape(B, T, P, self.point_feat_dim)
        point_feats = point_feats.mean(dim=2)  # avg over P -> [B, T, point_feat_dim]

        # --- 2) Proprio features: [B, proprio_dim] -> [B, T, proprio_feat_dim] ---
        proprio_feats = self.proprio_mlp(robot_states)  # [B, proprio_feat_dim]
        proprio_feats = proprio_feats.unsqueeze(1).expand(B, T, self.proprio_feat_dim)

        # --- 3) Fuse & project to d_model ---
        fused = torch.cat([point_feats, proprio_feats], dim=-1)  # [B, T, point+proprio]
        x = self.fuse_proj(fused)  # [B, T, d_model]

        # --- 4) Add sinusoidal time positional encoding ---
        pe = sinusoidal_position_encoding(
            length=T,
            dim=self.d_model,
            device=device,
            dtype=dtype,
        )  # [T, d_model]
        x = x + pe.unsqueeze(0)  # [B, T, d_model]

        x = self.transformer(x)  # [B, T, d_model]

        # --- 6) Project to action_dim ---
        actions = self.out_proj(x)  # [B, T, action_dim]
        return actions
