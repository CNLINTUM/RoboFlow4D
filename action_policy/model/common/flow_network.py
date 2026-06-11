import torch
from torch import nn

from model.common.network import create_mlp
from model.common.projection import FiLMLayer, ProjectionHead
from model.common.vision_transformer import Block, trunc_normal_
import math

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def posemb_sincos_3d(d, h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    Return shape: (d*h*w, dim).
    dim must be a multiple of 6:
      each axis (x,y,z) gets dim/6 frequencies;
      each axis uses sin+cos -> 2*(dim/6) = dim/3;
      all three axes sum to dim.
    """
    z, y, x = torch.meshgrid(
        torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 6) == 0, "feature dimension must be multiple of 6 for 3D sincos emb"

    n_freq = dim // 6  # Number of frequencies per axis.
    omega = torch.arange(n_freq, dtype=torch.float32) / (n_freq - 1 if n_freq > 1 else 1)
    omega = 1.0 / (temperature ** omega)  # (n_freq,)

    # flatten -> (DHW, 1) * (1, n_freq) => (DHW, n_freq)
    z = z.flatten()[:, None].to(torch.float32) * omega[None, :]
    y = y.flatten()[:, None].to(torch.float32) * omega[None, :]
    x = x.flatten()[:, None].to(torch.float32) * omega[None, :]

    pe = torch.cat(
        (x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()),
        dim=1
    )
    return pe.to(dtype)


class FlowMlp(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch):
        super(FlowMlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.mlp = nn.Sequential(
            *create_mlp(
                input_dim=input_dim,
                output_dim=output_dim,
                net_arch=net_arch,
            )
        )

    def forward(self, discriptors, flows):
        x = torch.cat([discriptors, flows], dim=-1)
        return self.mlp(x)


class FlowConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriptors, flows):
        return torch.cat([discriptors, flows], dim=-1)


class FlowPos2d(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=[224, 224],
        pre_zero_out=False,
        post_zero_out=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, embed_dim).reshape(*img_size, embed_dim)
        self.register_buffer("pos_2d", pos_2d)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        if self.pre_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors + self.pos_2d[flows[:, 1], flows[:, 0]]
        if self.post_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors.reshape(*discriptors_shape)

        return discriptors


class FlowPos2dConcat(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=[224, 224],
        pre_zero_out=False,
        post_zero_out=False,
        use_mix_in=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, embed_dim).reshape(*img_size, embed_dim)
        self.register_buffer("pos_2d", pos_2d)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out
        self.use_mix_in = use_mix_in
        if self.use_mix_in:
            self.mixin = nn.Linear(2 * embed_dim, 2 * embed_dim)

        else:
            self.mixin = nn.Identity()

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        assert discriptors_shape[-1] == self.embed_dim
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        if self.pre_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = torch.cat(
            [discriptors, self.pos_2d[flows[:, 1], flows[:, 0]]], dim=-1
        )
        discriptors = self.mixin(discriptors)
        if self.post_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors.reshape(
            *discriptors_shape[:-1], discriptors_shape[-1] * 2
        )

        return discriptors

class FlowPos3dConcat(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_freqs: int = 10,
        include_raw_xyz: bool = True,
        xyz_bounds=None,  # Optional normalization bounds: ((xmin,ymin,zmin),(xmax,ymax,zmax)).
        pre_zero_out=False,
        post_zero_out=False,
        use_mix_in=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        self.include_raw_xyz = include_raw_xyz
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out

        in_dim = (3 if include_raw_xyz else 0) + 3 * (2 * num_freqs)
        self.pos_proj = nn.Linear(in_dim, embed_dim)
        self.mixin = nn.Linear(2 * embed_dim, 2 * embed_dim) if use_mix_in else nn.Identity()

        if xyz_bounds is not None:
            xyz_min, xyz_max = xyz_bounds
            self.register_buffer("xyz_min", torch.tensor(xyz_min, dtype=torch.float32))
            self.register_buffer("xyz_max", torch.tensor(xyz_max, dtype=torch.float32))
        else:
            self.xyz_min = None
            self.xyz_max = None

    def _normalize_xyz(self, xyz):
        if self.xyz_min is None or self.xyz_max is None:
            return xyz
        denom = (self.xyz_max - self.xyz_min).clamp_min(1e-6)
        return (xyz - self.xyz_min) / denom

    def _fourier(self, xyz):
        xyz = self._normalize_xyz(xyz)
        freqs = (2.0 ** torch.arange(self.num_freqs, device=xyz.device, dtype=xyz.dtype)) * math.pi
        phase = xyz.unsqueeze(-1) * freqs  # (M,3,F)
        pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).reshape(xyz.shape[0], -1)
        if self.include_raw_xyz:
            pe = torch.cat([xyz, pe], dim=-1)
        return pe

    def forward(self, discriptors, flows_in):
        dis_shape = discriptors.shape
        assert dis_shape[-1] == self.embed_dim
        dis = discriptors.reshape(-1, self.embed_dim)

        fin = flows_in.reshape(-1, flows_in.shape[-1]).clone()
        if fin.shape[-1] >= 4:
            xyz = fin[:, :3]
            visible = fin[:, 3]
        else:
            xyz = fin[:, :3]
            visible = torch.ones(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)

        xyz = xyz.to(dis.dtype)

        if self.pre_zero_out:
            dis = dis * visible[:, None]

        pos_emb = self.pos_proj(self._fourier(xyz).to(dis.dtype))  # (M, D)
        out = torch.cat([dis, pos_emb], dim=-1)                    # (M, 2D)
        out = self.mixin(out)

        if self.post_zero_out:
            out = out * visible[:, None]

        return out.reshape(*dis_shape[:-1], 2 * self.embed_dim)

class FlowPos3dNoDesc(nn.Module):
    """
    Args:
        flows_in: Tensor with shape (..., 3) or (..., 4), storing xyz and an
            optional visibility flag.

    Returns:
        Tensor with shape (..., 2 * embed_dim), generated from Fourier features
        of xyz.
    """
    def __init__(
        self,
        embed_dim: int,
        num_freqs: int = 10,
        include_raw_xyz: bool = True,
        xyz_bounds=None,
        pre_zero_out=False,
        post_zero_out=False,
        use_mix_in=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        self.include_raw_xyz = include_raw_xyz
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out

        in_dim = (3 if include_raw_xyz else 0) + 3 * (2 * num_freqs)

        self.pos_proj = nn.Linear(in_dim, 2 * embed_dim)
        self.mixin = nn.Linear(2 * embed_dim, 2 * embed_dim) if use_mix_in else nn.Identity()

        if xyz_bounds is not None:
            xyz_min, xyz_max = xyz_bounds
            self.register_buffer("xyz_min", torch.tensor(xyz_min, dtype=torch.float32))
            self.register_buffer("xyz_max", torch.tensor(xyz_max, dtype=torch.float32))
        else:
            self.xyz_min = None
            self.xyz_max = None

    def _normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.xyz_min is None or self.xyz_max is None:
            return xyz
        denom = (self.xyz_max - self.xyz_min).clamp_min(1e-6)
        return (xyz - self.xyz_min) / denom

    def _fourier(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = self._normalize_xyz(xyz)
        freqs = (2.0 ** torch.arange(self.num_freqs, device=xyz.device, dtype=xyz.dtype)) * math.pi
        phase = xyz.unsqueeze(-1) * freqs  # (M,3,F)
        pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).reshape(xyz.shape[0], -1)
        if self.include_raw_xyz:
            pe = torch.cat([xyz, pe], dim=-1)
        return pe

    def forward(self, flows_in: torch.Tensor) -> torch.Tensor:
        # flows_in: (..., 3) or (..., 4)
        orig_shape = flows_in.shape
        fin = flows_in.reshape(-1, orig_shape[-1]).clone()

        if fin.shape[-1] >= 4:
            xyz = fin[:, :3]
            visible = fin[:, 3]
        else:
            xyz = fin[:, :3]
            visible = torch.ones(xyz.shape[0], device=xyz.device, dtype=xyz.dtype)

        # Fourier PE -> 2D feature
        xyz = xyz.to(dtype=fin.dtype)
        pe = self._fourier(xyz).to(dtype=fin.dtype)
        out = self.pos_proj(pe)  # (M, 2D)

        if self.pre_zero_out:
            out = out * visible[:, None]

        out = self.mixin(out)

        if self.post_zero_out:
            out = out * visible[:, None]

        return out.reshape(*orig_shape[:-1], 2 * self.embed_dim)

class FlowPos2dVisibilityConcat(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=[224, 224],
        pre_zero_out=False,
        post_zero_out=False,
        use_mix_in=False,
        visiablity_embed_dim=32,
        visiablity_apply_norm_layer=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, embed_dim).reshape(*img_size, embed_dim)
        self.register_buffer("pos_2d", pos_2d)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out
        self.use_mix_in = use_mix_in
        if self.use_mix_in:
            self.mixin = nn.Linear(2 * embed_dim, 2 * embed_dim)

        else:
            self.mixin = nn.Identity()
        self.visiablity_embed_dim = visiablity_embed_dim
        self.visiablity_proj = ProjectionHead(
            in_dim=1,
            out_dim=visiablity_embed_dim,
            nlayers=1,
            apply_norm_layer=visiablity_apply_norm_layer,
        )

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        assert discriptors_shape[-1] == self.embed_dim
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        if self.pre_zero_out:
            discriptors = discriptors * visiable[:, None]
        visiablity = self.visiablity_proj(visiable[:, None])  # (B,1) -> (B,32)
        discriptors = torch.cat(
            [discriptors, self.pos_2d[flows[:, 1], flows[:, 0]], visiablity], dim=-1
        )
        discriptors = self.mixin(discriptors)
        if self.post_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors.reshape(
            *discriptors_shape[:-1],
            discriptors_shape[-1] * 2 + self.visiablity_embed_dim,
        )

        return discriptors


class FlowPos2dVisibilityFiLM(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_size=[224, 224],
        use_mix_in=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, embed_dim).reshape(*img_size, embed_dim)
        self.register_buffer("pos_2d", pos_2d)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.use_mix_in = use_mix_in
        if self.use_mix_in:
            self.mixin = nn.Linear(2 * embed_dim, 2 * embed_dim)

        else:
            self.mixin = nn.Identity()
        self.visiablity_proj = FiLMLayer(
            in_dim=1,
            out_dim=2 * embed_dim,
        )

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        assert discriptors_shape[-1] == self.embed_dim
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        discriptors = torch.cat(
            [discriptors, self.pos_2d[flows[:, 1], flows[:, 0]]], dim=-1
        )
        discriptors = self.visiablity_proj(discriptors, visiable[:, None])
        discriptors = self.mixin(discriptors)
        discriptors = discriptors.reshape(
            *discriptors_shape[:-1], discriptors_shape[-1] * 2
        )
        return discriptors


class FlowPos2dEmd(nn.Module):
    def __init__(
        self,
        embed_dim,
        frequency_embedding_size=256,
        img_size=[224, 224],
        pre_zero_out=False,
        post_zero_out=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, frequency_embedding_size).reshape(
            *img_size, frequency_embedding_size
        )
        self.register_buffer("pos_2d", pos_2d)
        self.position_embed = nn.Sequential(
            nn.Linear(frequency_embedding_size, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        if self.pre_zero_out:
            discriptors = discriptors * visiable[:, None]
        pos_emd = self.pos_2d[flows[:, 1], flows[:, 0]]
        emd = self.position_embed(pos_emd)
        discriptors = discriptors + emd
        if self.post_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors.reshape(*discriptors_shape)

        return discriptors


class FlowPos2dEmdConcat(nn.Module):
    def __init__(
        self,
        embed_dim,
        frequency_embedding_size=256,
        img_size=[224, 224],
        pre_zero_out=False,
        post_zero_out=False,
        use_mix_in=False,
    ):
        super().__init__()
        pos_2d = posemb_sincos_2d(*img_size, frequency_embedding_size).reshape(
            *img_size, frequency_embedding_size
        )
        self.register_buffer("pos_2d", pos_2d)
        self.position_embed = nn.Sequential(
            nn.Linear(frequency_embedding_size, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pre_zero_out = pre_zero_out
        self.post_zero_out = post_zero_out
        self.use_mix_in = use_mix_in
        if self.use_mix_in:
            self.mixin = nn.Sequential(
                nn.Linear(2 * embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
            )
        else:
            self.mixin = nn.Identity()

    def forward(self, discriptors, flows_in):
        flows = flows_in.clone()
        discriptors_shape = discriptors.shape
        discriptors = discriptors.reshape(-1, self.embed_dim)
        flows = flows.reshape(-1, 3)
        flows[:, :2] = flows[:, :2] * self.img_size[0]
        visiable = flows[:, 2]
        flows = flows.long()
        flows = torch.clip(flows, min=0, max=self.img_size[0] - 1)
        if self.pre_zero_out:
            discriptors = discriptors * visiable[:, None]
        pos_emd = self.pos_2d[flows[:, 1], flows[:, 0]]
        emd = self.position_embed(pos_emd)
        discriptors = torch.cat([discriptors, emd], dim=-1)
        discriptors = self.mixin(discriptors)
        if self.post_zero_out:
            discriptors = discriptors * visiable[:, None]
        discriptors = discriptors.reshape(
            *discriptors_shape[:-1], discriptors_shape[-1] * 2
        )

        return discriptors


class FlowStateEncoder(nn.Module):
    def __init__(
        self,
        cross_attention,
        transformer_encoder,
        global_pool="token",
        proj_head=None,
    ) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.transformer_encoder = transformer_encoder
        self.global_pool = global_pool
        self.proj_head = proj_head
        if self.global_pool == "max_mean":
            assert proj_head is not None, "proj_head is required for max_mean pooling"

    def forward(self, flow_emb, initial_emd=None):
        if initial_emd is not None and self.cross_attention is not None:
            flow_emb = self.cross_attention(x=flow_emb, context=initial_emd)  # (B,N,K)
        flow_token = self.transformer_encoder(flow_emb)  # (B,N+1,K)
        if self.global_pool == "token":
            flow_state = flow_token[:, 0]
        elif self.global_pool == "mean":
            flow_state = flow_token.mean(dim=1)
        elif self.global_pool == "max":
            flow_state, _ = flow_token.max(dim=1)
        elif self.global_pool == "max_mean":
            mean_flow_state = flow_token.mean(dim=1)
            max_flow_state, _ = flow_token.max(dim=1)
            flow_state = torch.cat([mean_flow_state, max_flow_state], dim=-1)

        if self.proj_head is not None:
            flow_state = self.proj_head(flow_state)
        return flow_state


class FlowTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        dim_out=None,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        class_token=True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_out = dim_out
        self.class_token = class_token
        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # self.cls_embed = nn.Parameter(torch.zeros(1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        if self.class_token:
            trunc_normal_(self.cls_token, std=0.02)
        if self.dim_out is not None:
            self.proj_out = nn.Linear(embed_dim, self.dim_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, n, _ = x.shape
        if self.class_token:
            # add the [CLS] token to the embed patch tokens
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            # x[:, 0] = x[:, 0] + self.cls_embed
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.dim_out is not None:
            x = self.proj_out(x)

        return x
