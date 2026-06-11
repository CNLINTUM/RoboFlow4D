import numpy as np
import timm
import torch
from einops import rearrange
from model.common.utility.arr import stratified_random_sampling, uniform_sampling
from model.common.utility.model import freeze_model
from torch import nn


def _infer_vit_feature_dim(vit_module: nn.Module) -> int:
    dim = getattr(vit_module, "embed_dim", None)
    if dim is None:
        dim = getattr(vit_module, "num_features", None)
    assert dim is not None, "Could not infer the visual backbone feature dimension."
    return dim


def _infer_flow_token_dim(state_encoder: nn.Module) -> int:
    transformer = getattr(state_encoder, "transformer_encoder", None)
    dim = getattr(transformer, "dim_out", None)
    if dim is None:
        dim = getattr(transformer, "embed_dim", None)
    assert dim is not None, "Could not infer the flow token dimension from state_encoder."
    return dim


def _infer_plan_dim(plan_encoder: nn.Module) -> int:
    dim = getattr(plan_encoder, "dim_out", None)
    if dim is None:
        dim = getattr(plan_encoder, "embed_dim", None)
    assert dim is not None, "Could not infer the flow plan dimension from plan_encoder."
    return dim


class FlowDiffusionPolicy(nn.Module):
    def __init__(
        self,
        flow_encoder,
        state_encoder,
        plan_encoder,
        time_alignment_transformer,
        flow_proj_in,
        diffusion_policy,
        policy_condition_on_proprioception_proj=True,
        point_encoder=None,
        alignment_detach=True,
        predict_detach=True,
        proprioception_proj_in=None,
        vision_transformer_0=None,
        vision_transformer_0_kwargs=None,
        sampling_method="stratified",
        sampling_frame=48,
        proprioception_predictor=None,
        freeze_vit=False,
        plan_condition_type="initial",
        target_condition_type="initial",
        target_plan_drop_prob=0.0,
        flow_condition_source="pooled",
        freeze_text=True, lang_drop_prob=0.1,
        MeanPoolTextEncoder=None
    ) -> None:
        super().__init__()
        self.sampling_method = sampling_method
        self.sampling_frame = sampling_frame
        self.alignment_detach = alignment_detach
        self.predict_detach = predict_detach
        self.freeze_vit = freeze_vit
        self.proprioception_predictor = proprioception_predictor
        if vision_transformer_0 is not None:
            self.vision_transformer_0 = vision_transformer_0
        elif vision_transformer_0_kwargs is not None:
            self.vision_transformer_0 = timm.create_model(**vision_transformer_0_kwargs)
        else:
            self.vision_transformer_0 = None

        if self.vision_transformer_0 is not None and self.freeze_vit:
            freeze_model(self.vision_transformer_0)

        self.flow_encoder = flow_encoder
        self.state_encoder = state_encoder
        self.point_encoder = point_encoder
        self.plan_encoder = plan_encoder
        self.time_alignment_transformer = time_alignment_transformer
        self.flow_proj_in = flow_proj_in
        self.proprioception_proj_in = proprioception_proj_in
        self.policy_condition_on_proprioception_proj = (
            policy_condition_on_proprioception_proj
        )
        self.prop_detach_in_alignment = (
            True if policy_condition_on_proprioception_proj else False
        )
        self.diffusion_policy = diffusion_policy
        self.plan_condition_type = plan_condition_type
        self.target_condition_type = target_condition_type
        self.target_plan_drop = nn.Dropout(target_plan_drop_prob)
        self.flow_condition_source = flow_condition_source

        assert self.plan_condition_type in ["initial", "current", "none"]
        assert self.target_condition_type in ["initial", "current", "none"]
        assert self.flow_condition_source in ["pooled", "predict_plan", "cross_attn"]

        K = _infer_flow_token_dim(self.state_encoder)
        plan_dim = _infer_plan_dim(self.plan_encoder)
        Dv = _infer_vit_feature_dim(self.vision_transformer_0)

        self.global_img_proj = nn.Sequential(
            nn.LayerNorm(Dv),
            nn.Linear(Dv, K),
        )

        self.global_img_drop = nn.Dropout(p=0.1)

        self.freeze_text = freeze_text
        self.text_encoder = MeanPoolTextEncoder
        self.lang_proj = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
        )
        self.lang_drop = nn.Dropout(p=lang_drop_prob)

        # Mirror the paper's "MLP + attention pooling" style flow-plan encoder:
        # first refine per-frame flow tokens, then pool them into a fixed global f_flow.
        self.flow_token_mlp = nn.Sequential(
            nn.LayerNorm(K),
            nn.Linear(K, plan_dim),
            nn.GELU(),
            nn.Linear(plan_dim, plan_dim),
        )

    def forward(
        self,
        noisy_actions,
        timesteps,
        images,
        wrist_images,
        proprioception,
        flows,
        initial_query_points,
        target_flows,
        input_ids=None, attention_mask=None
    ):
        if input_ids is None or attention_mask is None:
            raise ValueError("Need input_ids and attention_mask for language conditioning")

        if self.freeze_text:
            self.text_encoder.eval()
            with torch.no_grad():
                text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # (B, Dt)
        else:
            text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # (B, Dt)

        lang_k = self.lang_proj(text_feat)   # (B, K)
        lang_k = self.lang_drop(lang_k)

        B, T, N, _ = flows.shape
        if self.sampling_method in ["stratified", "uniform"]:
            target_horizon = target_flows.shape[1]
            reference_arr = np.arange(target_horizon)
            if self.sampling_method == "stratified":
                _, sample_indices = stratified_random_sampling(
                    reference_arr, self.sampling_frame, return_indices=True
                )
            else:
                _, sample_indices = uniform_sampling(
                    reference_arr, self.sampling_frame, return_indices=True
                )
            target_flows = target_flows[:, sample_indices].clone()  # (B,T',N,3)

        images = normalize_images(images)
        if self.freeze_vit:
            self.vision_transformer_0.eval()
            with torch.no_grad():
                img_feat = self.vision_transformer_0(images)   
        else:
            img_feat = self.vision_transformer_0(images)       # (B,768)

        wrist_images = normalize_images(wrist_images)
        if self.freeze_vit:
            self.vision_transformer_0.eval()
            with torch.no_grad():
                wrist_img_feat = self.vision_transformer_0(wrist_images)   
        else:
            wrist_img_feat = self.vision_transformer_0(wrist_images)       # (B,768)

        feat = torch.stack([img_feat, wrist_img_feat], dim=1)
        img_k = self.global_img_proj(feat)
        img_k = self.global_img_drop(img_k)
        img_global_k = img_k.mean(dim=1)

        flow_embedding = self.flow_encoder(flows)  # flows: (B,T,N,3) -> (B,T,N,2D)
        flow_embedding = rearrange(flow_embedding, "B T N C -> (B T) N C", B=B)  # (B*T,N,2D)

        plan_embedding = self.state_encoder(flow_embedding)                     # (B*T,K)
        plan_embedding = rearrange(plan_embedding, "(B T) K -> B T K", B=B)         # (B,T,K)
        flow_tokens = self.flow_token_mlp(plan_embedding)                      # (B,T,plan_dim)

        if self.proprioception_proj_in is not None:
            prop_tok = self.proprioception_proj_in(proprioception.unsqueeze(1))
        else:
            raise ValueError("proprioception_proj_in is required to align proprioception to token dim.")

        img_tok = img_global_k.unsqueeze(1)  # (B,1,K)
        if self.alignment_detach:
            align_in = torch.cat(
                [
                    img_tok.detach(),
                    flow_tokens,
                ],
                dim=1,
            )
        else:
            align_in = torch.cat(
                [
                    img_tok,
                    flow_tokens,
                ],
                dim=1,
            )

        predict_plan, _ = self.time_alignment_transformer(align_in, return_cls_token=True)

        with torch.no_grad():
            target_flow_embedding = self.flow_encoder(target_flows)
            target_flow_embedding = rearrange(target_flow_embedding, "B T N C -> (B T) N C", B=B)
            target_plan_seq = self.state_encoder(target_flow_embedding)
            target_plan_seq = rearrange(target_plan_seq, "(B T) K -> B T K", B=B)
            target_flow_tokens = self.flow_token_mlp(target_plan_seq)
            target_plan, _ = self.plan_encoder(target_flow_tokens, return_cls_token=True)

        if (self.proprioception_proj_in is not None) and self.policy_condition_on_proprioception_proj:
            prop_cond = prop_tok.flatten(start_dim=1)  # (B,K)
        else:
            prop_cond = proprioception  # (B,prop_dim)

        img_cond = img_global_k  # (B, K)
        flow_context = None
        if self.flow_condition_source == "cross_attn":
            f_flow = None
            flow_context = self.target_plan_drop(flow_tokens)
            condition = torch.cat(
                    [
                        lang_k,
                        img_cond,
                        prop_cond,
                    ],
                    dim=1,
                )
        elif self.flow_condition_source == "predict_plan":
            f_flow = predict_plan
            f_flow = self.target_plan_drop(f_flow)
            condition = torch.cat(
                    [lang_k, img_cond, prop_cond, f_flow],
                    dim=1,
                )
        else:
            f_flow, _ = self.plan_encoder(flow_tokens, return_cls_token=True)
            f_flow = self.target_plan_drop(f_flow)
            condition = torch.cat(
                    [lang_k, img_cond, prop_cond, f_flow],
                    dim=1,
                )

        if flow_context is None:
            prediction = self.diffusion_policy(noisy_actions, timesteps, global_cond=condition)
        else:
            prediction = self.diffusion_policy(noisy_actions, timesteps, global_cond=condition, context=flow_context)

        return (
            prediction,
            predict_plan,
            target_plan.clone().detach(),
        )


def normalize_images(images):
    if images.max() > 1.5:
        images = images / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
