"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AlignProjector(nn.Module):
    """
    calculate the alignment between LLM and VGGT embeddings.
    """
    def __init__(
            self, 
            llm_dim: int, 
            vggt_dim: int,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type

        self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None

        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor = None) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_emb, target_emb):
        if self.align_loss_type == "cosine":
            # project vla dimension and calculate align loss
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()  # mean for sequence length
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")



def resample_tokens_to_length(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    tokens: [B, L_src, D]
    Resample an approximately spatial token sequence to a target length.
    """
    B, L_src, D = tokens.shape
    H = int(math.sqrt(L_src))
    W = max(1, round(L_src / H))
    if H * W != L_src:
        # Pad to H*W.
        pad = H * W - L_src
        if pad > 0:
            pad_tok = tokens[:, -1:, :].expand(B, pad, D)
            tokens = torch.cat([tokens, pad_tok], dim=1)
        elif pad < 0:
            tokens = tokens[:, :H*W, :]
    feat = tokens.transpose(1, 2).reshape(B, D, H, W)      # [B,D,H,W]

    Ht = int(math.sqrt(target_len))
    Wt = max(1, round(target_len / Ht))
    while Ht * Wt < target_len: Wt += 1
    while Ht * Wt - Wt >= target_len and Ht > 1: Ht -= 1

    feat_rs = F.interpolate(feat, size=(Ht, Wt), mode="bilinear", align_corners=True)  # [B,D,Ht,Wt]
    out = feat_rs.flatten(2).transpose(1, 2)  # [B, Ht*Wt, D]
    if out.shape[1] > target_len:
        out = out[:, :target_len, :]
    elif out.shape[1] < target_len:
        need = target_len - out.shape[1]
        out = torch.cat([out, out[:, -1:, :].expand(B, need, D)], dim=1)
    return out  # [B, target_len, D]
