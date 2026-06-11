import math
import contextlib
from typing import Optional  
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, Dinov2Model
from torchvision.transforms import Normalize

from motion_module import get_motion_module

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class MultiBranchCondModulation(nn.Module):
    """
    Community-standard conditional modulation module.

    Structure:
        cond
          ↓
      shared MLP
          ↓
    ┌───────────────┬───────────────┬───────────────┐
    │ Temporal head │ Point head    │ Cross head    │
    │ (shift,scale, │ (shift,scale, │ (shift,scale, │
    │  gate)        │  gate)        │  gate)        │
    └───────────────┴───────────────┴───────────────┘
    """
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_gate = use_gate
        out_dim = hidden_dim * (3 if use_gate else 2)

        # Shared FiLM-style core.
        self.shared = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim * mlp_ratio),
            nn.SiLU(),
        )

        # Branch-specific heads.
        self.temporal = nn.Linear(cond_dim * mlp_ratio, out_dim)
        self.point    = nn.Linear(cond_dim * mlp_ratio, out_dim)
        self.cross    = nn.Linear(cond_dim * mlp_ratio, out_dim)

        # AdaLN-Zero / LayerScale init.
        for head in (self.temporal, self.point, self.cross):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _split(self, x):
        if self.use_gate:
            shift, scale, gate = x.chunk(3, dim=-1)
            return shift, scale, gate
        else:
            shift, scale = x.chunk(2, dim=-1)
            return shift, scale, None

    def forward(self, cond):
        """
        Args:
            cond: [B, cond_dim]

        Returns:
            dict with keys:
              - "temporal": (shift, scale, gate)
              - "point":    (shift, scale, gate)
              - "cross":    (shift, scale, gate)
        """
        h = self.shared(cond)

        return {
            "temporal": self._split(self.temporal(h)),
            "point":    self._split(self.point(h)),
            "cross":    self._split(self.cross(h)),
        }


class FactorizedDiTBlockV2(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim

        self.norm_t = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)

        self.attn_t = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_p = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_x = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x, B, K, N,
        mod,
        mem_tokens=None
    ):
        L = K * N
        x = x.view(B, K, N, self.dim)

        # -------- Temporal SA --------
        shift, scale, gate = mod["temporal"]
        xt = self.norm_t(x)
        xt = xt * (1 + scale[:, None, None]) + shift[:, None, None]
        xt = xt.permute(0, 2, 1, 3).reshape(B * N, K, self.dim)
        attn_t, _ = self.attn_t(xt, xt, xt)
        attn_t = attn_t.view(B, N, K, self.dim).permute(0, 2, 1, 3)
        x = x + gate[:, None, None] * attn_t

        # -------- Point SA --------
        shift, scale, gate = mod["point"]
        xp = self.norm_p(x)
        xp = xp * (1 + scale[:, None, None]) + shift[:, None, None]
        xp = xp.reshape(B * K, N, self.dim)
        attn_p, _ = self.attn_p(xp, xp, xp)
        attn_p = attn_p.view(B, K, N, self.dim)
        x = x + gate[:, None, None] * attn_p

        if mem_tokens is not None:
            shift, scale, gate = mod["cross"]
            q = self.norm_x(x.view(B, L, self.dim))
            q = q * (1 + scale[:, None]) + shift[:, None]
            cx, _ = self.attn_x(q, mem_tokens, mem_tokens)
            x = x.view(B, L, self.dim) + gate[:, None] * cx
            x = x.view(B, K, N, self.dim)

        # -------- MLP --------
        xm = self.norm_m(x)
        x = x + self.mlp(xm)

        return x.view(B, L, self.dim)

class FlowDiT(nn.Module):
    def __init__(
        self,
        k_steps,
        num_points,
        model_dim,
        num_layers,
        num_heads,
        cond_dim,
        num_mem_tokens=16,
        cross_layers=3,
    ):
        super().__init__()
        self.k = k_steps
        self.n = num_points
        self.d = model_dim
        self.cross_layers = cross_layers

        self.input_proj = nn.Linear(3, model_dim)
        self.time_pe = nn.Parameter(torch.randn(1, k_steps, model_dim) * 0.02)
        self.point_pe = nn.Parameter(torch.randn(1, num_points, model_dim) * 0.02)
        self.query_proj = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.query_gate = nn.Parameter(torch.tensor(0.0))

        self.timestep_emb = TimestepEmbedding(model_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, cond_dim),
        )

        self.blocks = nn.ModuleList([
            FactorizedDiTBlockV2(model_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.modulators = nn.ModuleList([
            MultiBranchCondModulation(model_dim, cond_dim)
            for _ in range(num_layers)
        ])

        # Memory-token path.
        self.mem_resampler = TokenResampler(cond_dim, num_latents=num_mem_tokens)
        self.mem_proj = nn.Linear(cond_dim, model_dim, bias=False)

        self.out_proj = nn.Linear(model_dim, 3)

    def forward(self, noisy_flow, timestep, cond_vec, mem_tokens, point_queries=None):
        B, K, N, _ = noisy_flow.shape

        x = self.input_proj(noisy_flow)
        x = x + self.time_pe[:, :K].unsqueeze(2) + self.point_pe[:, :N].unsqueeze(1)
        if point_queries is not None:
            q_feat = self.query_proj(torch.nan_to_num(point_queries.to(x.dtype), nan=0.0))
            x = x + torch.tanh(self.query_gate) * q_feat.unsqueeze(1)
        x = x.view(B, K * N, self.d)

        cond = self.timestep_mlp(self.timestep_emb(timestep)) + cond_vec

        mem_tokens = self.mem_resampler(mem_tokens)
        mem_tokens = self.mem_proj(mem_tokens)

        for i, (blk, mod) in enumerate(zip(self.blocks, self.modulators)):
            mods = mod(cond)
            use_cross = (i < self.cross_layers)
            x = blk(
                x, B, K, N,
                mods,
                mem_tokens if use_cross else None
            )

        return self.out_proj(x).view(B, K, N, 3)


# ==========================================
# 1. Enhanced pretrained conditioning encoder (SigLIP + DINOv2 dual backbone)
# ==========================================
class PretrainedConditioningEncoder(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        dinov2_model_name: str = "facebook/dinov2-base",
        fuse: str = "sum",
        max_frames: int = 32,
        freeze_siglip: bool = True,
        freeze_dino: bool = True,
    ):
        super().__init__()
        # Load both vision backbones.
        self.siglip_model = SiglipModel.from_pretrained(siglip_model_name)
        self.dinov2_model = Dinov2Model.from_pretrained(dinov2_model_name)
        self.fuse = fuse

        # Configure frozen/trainable parameters.
        self.freeze_siglip = bool(freeze_siglip)
        self.freeze_dino   = bool(freeze_dino)

        for p in self.siglip_model.parameters():
            p.requires_grad = (not self.freeze_siglip)
        for p in self.dinov2_model.parameters():
            p.requires_grad = (not self.freeze_dino)

        # Set the initial training/eval modes.
        if self.freeze_siglip:
            self.siglip_model.eval()
        if self.freeze_dino:
            self.dinov2_model.eval()

        # Infer feature dimensions.
        self.s_dim = self.siglip_model.vision_model.config.hidden_size
        self.d_dim = self.dinov2_model.config.hidden_size
        self.vision_hidden_dim = self.s_dim + self.d_dim

        # Global projection layers.
        s_embed_dim = self.siglip_model.config.vision_config.hidden_size
        total_global_dim = s_embed_dim + self.d_dim
        
        proj_dim = condition_dim // 2 if fuse == "concat" else condition_dim
        self.img_proj = nn.Linear(total_global_dim, proj_dim)
        self.txt_proj = nn.Linear(s_embed_dim, proj_dim)

        # Temporal processing.
        self._max_frames = int(max_frames)
        self.frame_pos_embed = nn.Embedding(self._max_frames, self.vision_hidden_dim)
        self.temporal_q = nn.Linear(self.vision_hidden_dim, self.vision_hidden_dim, bias=False)
        self.temporal_k = nn.Linear(self.vision_hidden_dim, self.vision_hidden_dim, bias=False)
        self.motion_ln = nn.LayerNorm(self.vision_hidden_dim)
        self.motion_scale = nn.Parameter(torch.tensor(0.0))

        # Normalization used to convert SigLIP-preprocessed pixels to DINOv2 input.
        # SigLIP: mean=0.5, std=0.5
        # DINOv2: ImageNet mean/std
        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep SigLIP in eval mode when frozen.
        if self.freeze_siglip:
            self.siglip_model.eval()
        else:
            self.siglip_model.train(mode)

        # DINO follows the requested train/eval mode unless it is frozen.
        if self.freeze_dino:
            self.dinov2_model.eval()
        else:
            self.dinov2_model.train(mode)
        return self

    def _vision_forward(self, pixel_values: torch.Tensor, need_tokens: bool = True):
        # pixel_values are expected to already use SigLIP preprocessing.
        is_video = (pixel_values.dim() == 5)
        if is_video:
            B, Kc, C, H, W = pixel_values.shape
            pv = pixel_values.view(B * Kc, C, H, W)
        else:
            B, Kc = pixel_values.shape[0], 1
            pv = pixel_values

        use_amp = (pv.device.type == "cuda")
        
        with torch.autocast(device_type=pv.device.type, dtype=torch.bfloat16, enabled=use_amp):
            # 1) SigLIP features.
            sig_ctx = torch.no_grad() if self.freeze_siglip else contextlib.nullcontext()
            with sig_ctx:
                s_out = self.siglip_model.vision_model(pixel_values=pv)
                # SigLIP has no CLS token here.
                s_tokens = s_out.last_hidden_state  # [BK, 196, Ds]
                s_global = s_out.pooler_output      # [BK, Ds]

            # 2) DINOv2 features.
            # SigLIP: x = (img - 0.5)/0.5 => img = x*0.5 + 0.5
            img_01 = pv * 0.5 + 0.5
            pv_dino = (img_01 - self.dino_mean) / self.dino_std

            dino_ctx = torch.no_grad() if self.freeze_dino else contextlib.nullcontext()
            with dino_ctx:
                d_out = self.dinov2_model(pixel_values=pv_dino)
                # Remove DINOv2 CLS token at index 0.
                d_tokens = d_out.last_hidden_state[:, 1:, :]  # [BK, 256, Dd]
                d_global = d_out.pooler_output

            # 3) Spatially align DINO tokens from 16x16 to SigLIP's 14x14 grid.
            if need_tokens:
                Ls = int(math.sqrt(s_tokens.shape[1]))  # sqrt(196)=14
                Ld = int(math.sqrt(d_tokens.shape[1]))  # sqrt(256)=16

                if Ls != Ld:
                    # [BK, Ld^2, Dd] -> [BK, Dd, Ld, Ld] -> resize -> [BK, Dd, Ls, Ls] -> [BK, Ls^2, Dd]
                    d_img = d_tokens.permute(0, 2, 1).reshape(pv.size(0), self.d_dim, Ld, Ld)
                    d_res = F.interpolate(d_img, size=(Ls, Ls), mode="bilinear", align_corners=False)
                    d_tokens = d_res.reshape(pv.size(0), self.d_dim, -1).permute(0, 2, 1)

            # 4) Fuse global features and tokens.
            combined_img = torch.cat([s_global, d_global], dim=-1).float()
            combined_tok = torch.cat([s_tokens, d_tokens], dim=-1).float() if need_tokens else None

        if is_video:
            combined_img = combined_img.view(B, Kc, -1).mean(dim=1)
        combined_img = F.normalize(combined_img, dim=-1)

        if not need_tokens:
            return combined_img, None

        if not is_video:
            return combined_img, combined_tok
        
        # Multi-frame temporal fusion.
        tok = combined_tok.view(B, Kc, combined_tok.shape[1], -1)
        fpos = self.frame_pos_embed.weight[:Kc]
        tok = tok + fpos.view(1, Kc, 1, -1)
        
        cur_idx = Kc - 1
        tok_cur = tok[:, cur_idx]
        if cur_idx > 0:
            tok_past = tok[:, :cur_idx]
            q = self.temporal_q(tok_cur)
            k = self.temporal_k(tok_past)
            score = (k * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(k.shape[-1])
            w = torch.softmax(score, dim=1)
            motion_agg = (w.unsqueeze(-1) * (tok_cur.unsqueeze(1) - tok_past)).sum(dim=1)
            tok_out = tok_cur + self.motion_scale * self.motion_ln(motion_agg)
        else:
            tok_out = tok_cur

        return combined_img, tok_out


    def forward(self, pixel_values, input_ids=None, attention_mask=None, drop_condition_mask=None, return_tokens=False):
        img_feats, tok = self._vision_forward(pixel_values, need_tokens=return_tokens)        
        with torch.no_grad():
            if input_ids is not None:
                txt_feats = self.siglip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()
            else:
                txt_feats = torch.zeros(img_feats.shape[0], self.s_dim, device=img_feats.device)

        img_c = self.img_proj(img_feats)
        txt_c = self.txt_proj(txt_feats)

        if self.fuse == "sum": cond = img_c + txt_c
        elif self.fuse == "concat": cond = torch.cat([img_c, txt_c], dim=-1)
        else:
            gate = torch.sigmoid((img_c * txt_c).sum(dim=-1, keepdim=True))
            cond = gate * img_c + (1 - gate) * txt_c

        if drop_condition_mask is not None:
            cond = cond.masked_fill(drop_condition_mask.to(torch.bool)[:, None], 0.0)

        return (cond, tok) if return_tokens else cond


# ==========================================
# 2. Main model: GenerativeFlowModel
# ==========================================
class GenerativeFlowModel(nn.Module):
    def __init__(
        self,
        k_steps: int,
        num_points: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        dinov2_model_name: str = "facebook/dinov2-base",
        fuse: str = "sum",
    ):
        super().__init__()
        condition_dim = model_dim if fuse != "concat" else (model_dim * 2)
        
        self.encoder = PretrainedConditioningEncoder(
            condition_dim, siglip_model_name, dinov2_model_name, fuse
        )
        
        Dv = self.encoder.vision_hidden_dim 

        self.feat_3d_head = TokensTo3DAttnPool(condition_dim, siglip_hidden_dim=Dv)

        self.cond_merge = nn.Linear(condition_dim * 2, condition_dim)

        self.dit = FlowDiT(k_steps, num_points, model_dim, num_layers, num_heads, condition_dim)

        self.mem_proj = nn.Linear(Dv, model_dim, bias=False)
        nn.init.xavier_uniform_(self.mem_proj.weight)

    # Expose 3D features for the optional alignment loss in training.
    def get_features_3d(self, pixel_values):
        # Get tokens from the vision encoder only.
        _, tokens = self.encoder(pixel_values, return_tokens=True)
        # Aggregate with the 3D feature head.
        feat_3d = self.feat_3d_head(tokens)
        return feat_3d

    def forward(
        self,
        image_pixels,
        instruction_input_ids,
        instruction_attention_mask=None,
        query_points=None,
        noisy_flow=None,
        timestep=None,
        drop_condition_mask=None,
    ):
        cond, tokens = self.encoder(
            pixel_values=image_pixels,
            input_ids=instruction_input_ids,
            attention_mask=instruction_attention_mask,
            return_tokens=True
        )

        feat_3d = self.feat_3d_head(tokens)
        mem_tokens = self.mem_proj(tokens)

        point_queries = None
        if query_points is not None:
            point_queries = query_points.to(cond.dtype).clamp_(0.0, 1.0)
            point_queries = point_queries * 2.0 - 1.0

        cond = self.cond_merge(torch.cat([cond, feat_3d], dim=-1))

        if drop_condition_mask is not None:
            m = drop_condition_mask.to(torch.bool)
            cond = cond.masked_fill(m.view(-1, 1), 0.0)
            mem_tokens = mem_tokens.masked_fill(m.view(-1, 1, 1), 0.0)
            if point_queries is not None:
                point_queries = point_queries.masked_fill(m.view(-1, 1, 1), 0.0)

        return self.dit(noisy_flow, timestep, cond, mem_tokens=mem_tokens, point_queries=point_queries)

class _ResamplerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.05):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.ctx_ln = nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        hidden = int(d_model * ffn_ratio)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = _MLP(d_model, hidden, d_model, dropout=dropout)

    def forward(self, queries, context_tokens, key_padding_mask=None):
        q = self.q_ln(queries)
        k = self.ctx_ln(context_tokens)
        attn_out, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        queries = queries + self.drop(attn_out)

        ffn_in = self.ffn_ln(queries)
        queries = queries + self.drop(self.ffn(ffn_in))
        return queries

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TokensTo3DAttnPool(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        num_heads: int = 8,
        n_queries: int = 8,
        num_layers: int = 3,
        siglip_hidden_dim: int = 768,
        ffn_ratio: float = 4.0,
        dropout: float = 0.05,
        ln_in: bool = True,
        aggregate: str = "flat",
        mlp_hidden: int = 1024,
    ):
        super().__init__()
        self.condition_dim = condition_dim
        self.d_model = siglip_hidden_dim
        self.n_queries = n_queries
        self.aggregate = aggregate

        self.ln_in = nn.LayerNorm(self.d_model) if ln_in else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, n_queries, self.d_model) * 0.02)

        self.blocks = nn.ModuleList([
            _ResamplerBlock(self.d_model, num_heads, ffn_ratio=ffn_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

        if aggregate == "flat":
            out_in_dim = n_queries * self.d_model
            self.out_norm = nn.LayerNorm(out_in_dim)
        else:
            out_in_dim = self.d_model
            self.out_norm = nn.LayerNorm(out_in_dim)

        self.proj = _MLP(out_in_dim, mlp_hidden, condition_dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        B, L, D = tokens.shape
        # assert D == self.d_model, f"D mismatch: got {D}, expect {self.d_model}"

        ctx = self.ln_in(tokens)
        q = self.queries.expand(B, -1, -1).contiguous()

        for blk in self.blocks:
            q = blk(q, ctx, key_padding_mask=key_padding_mask)

        if self.aggregate == "flat":
            x = self.out_norm(q.reshape(B, -1))
        else:
            x = self.out_norm(q.mean(dim=1))

        return self.proj(x)
    

class TokenResampler(nn.Module):
    def __init__(self, dim: int, num_latents: int = 16, num_heads: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, tokens):
        B = tokens.size(0)
        q = self.latents.expand(B, -1, -1)
        kv = self.ln(tokens)
        out, _ = self.attn(q, kv, kv)
        return out
