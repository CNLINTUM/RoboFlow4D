#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect parameter stats by high-level modules:
  - vision encoder
  - text encoder
  - 3D perceiver (TokensTo3DAttnPool)
  - FlowDiT
(+ others)

Usage:
  python inspect_flow_model_params.py \
    --model_py /path/to/model_siglip_improved_align_dino_refined.py \
    --ckpt /path/to/siglip_flow_futureK_best.pt \
    --detail \
    --save_json param_summary.json
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn


def import_model_module(model_py: str):
    model_py = str(model_py)
    p = Path(model_py)
    assert p.exists(), f"model_py not found: {p}"
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handle DDP: keys like "module.xxx"
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def count_params_named(named_params: List[Tuple[str, torch.Tensor]]) -> Tuple[int, int]:
    total = sum(p.numel() for _, p in named_params)
    trainable = sum(p.numel() for _, p in named_params if p.requires_grad)
    return total, trainable


def human_m(num: int) -> float:
    return num / 1e6


def classify_param(name: str) -> str:
    """
    Grouping policy based on your model code:
      - encoder = PretrainedConditioningEncoder (siglip + dinov2 + temporal + proj)
      - feat_3d_head = TokensTo3DAttnPool
      - dit = FlowDiT
    """
    # -------- vision encoder --------
    if (
        name.startswith("encoder.siglip_model.vision_model")
        or name.startswith("encoder.dinov2_model")
        or name.startswith("encoder.frame_pos_embed")
        or name.startswith("encoder.temporal_q")
        or name.startswith("encoder.temporal_k")
        or name.startswith("encoder.motion_ln")
        or name.startswith("encoder.motion_scale")
        or name.startswith("encoder.img_proj")
    ):
        return "vision_encoder"

    # -------- text encoder --------
    # SiglipModel text tower is typically "text_model"; projection is txt_proj
    if (
        name.startswith("encoder.siglip_model.text_model")
        or name.startswith("encoder.txt_proj")
    ):
        return "text_encoder"

    # -------- 3D perceiver --------
    if name.startswith("feat_3d_head"):
        return "perceiver_3d"

    # -------- FlowDiT --------
    if name.startswith("dit"):
        return "flow_dit"

    # -------- everything else (conditioning / query pooling / mem_proj etc.) --------
    return "others"


def summarize_groups(model: nn.Module, detail: bool = False, topk: int = 15) -> Dict[str, Any]:
    # collect per-group named params
    group_params: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    for n, p in model.named_parameters():
        g = classify_param(n)
        group_params[g].append((n, p))

    # ensure stable ordering
    group_order = ["vision_encoder", "text_encoder", "perceiver_3d", "flow_dit", "others"]

    # global stats
    all_named = list(model.named_parameters())
    total_all, trainable_all = count_params_named(all_named)

    summary: Dict[str, Any] = {
        "total_params": total_all,
        "trainable_params": trainable_all,
        "groups": {},
    }

    print("\n==================== Model Parameter Summary ====================")
    print(f"[ALL] total: {human_m(total_all):.2f} M | trainable: {human_m(trainable_all):.2f} M"
          f" | frozen: {100.0 * (1.0 - trainable_all / max(total_all,1)):.2f}%")

    for g in group_order:
        named = group_params.get(g, [])
        if not named:
            continue

        total_g, trainable_g = count_params_named(named)
        frac = 100.0 * total_g / max(total_all, 1)

        dtypes = Counter(str(p.dtype) for _, p in named)
        reqgrad = Counter(bool(p.requires_grad) for _, p in named)

        # largest params
        largest = sorted(named, key=lambda x: x[1].numel(), reverse=True)[:topk]
        largest_brief = [
            {"name": n, "shape": list(p.shape), "numel": int(p.numel()),
             "dtype": str(p.dtype), "requires_grad": bool(p.requires_grad)}
            for n, p in largest
        ]

        summary["groups"][g] = {
            "total_params": total_g,
            "trainable_params": trainable_g,
            "fraction_of_all_percent": frac,
            "dtype_counts": dict(dtypes),
            "requires_grad_counts": {str(k): int(v) for k, v in reqgrad.items()},
            "largest_params_topk": largest_brief,
        }

        print(f"\n[{g}]")
        print(f"  total: {human_m(total_g):.2f} M ({frac:.2f}%) | trainable: {human_m(trainable_g):.2f} M"
              f" | frozen: {100.0 * (1.0 - trainable_g / max(total_g,1)):.2f}%")
        print(f"  dtype: {dict(dtypes)}")
        print(f"  requires_grad: { {('trainable' if k else 'frozen'): v for k, v in reqgrad.items()} }")
        print("  largest params:")
        for item in largest_brief:
            print(f"    - {item['numel']:>10,d} | {str(item['dtype']):>12} | "
                  f"{'T' if item['requires_grad'] else 'F'} | {item['shape']} | {item['name']}")

        if detail:
            print("  ---- full list ----")
            for n, p in named:
                print(f"    {n:70s} | {list(p.shape)!s:>18s} | {p.numel():>10,d} | "
                      f"{str(p.dtype):>12s} | {'T' if p.requires_grad else 'F'}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", type=str, required=True,
                    help="path to model_siglip_improved_align_dino_refined.py")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="path to siglip_flow_futureK_best.pt (or a raw state_dict)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--detail", action="store_true", help="print every param entry (can be long)")
    ap.add_argument("--topk", type=int, default=15, help="top-k largest params shown per group")
    ap.add_argument("--save_json", type=str, default=None, help="optional output json path")
    # fallback config if you don't pass ckpt that contains cfg/k_steps/num_points
    ap.add_argument("--k_steps", type=int, default=20)
    ap.add_argument("--num_points", type=int, default=100)
    ap.add_argument("--model_dim", type=int, default=768)
    ap.add_argument("--num_layers", type=int, default=10)
    ap.add_argument("--num_heads", type=int, default=12)
    args = ap.parse_args()

    mod = import_model_module(args.model_py)
    assert hasattr(mod, "GenerativeFlowModel"), "GenerativeFlowModel not found in model_py"

    # build from ckpt if provided (preferred because it stores cfg/k_steps/num_points)
    cfg = dict(model_dim=args.model_dim, num_layers=args.num_layers, num_heads=args.num_heads)
    k_steps = args.k_steps
    num_points = args.num_points
    state_dict = None

    if args.ckpt is not None:
        ckpt_obj = torch.load(args.ckpt, map_location="cpu")
        if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
            state_dict = ckpt_obj["model"]
            if "cfg" in ckpt_obj and isinstance(ckpt_obj["cfg"], dict):
                cfg = ckpt_obj["cfg"]
            if "k_steps" in ckpt_obj:
                k_steps = int(ckpt_obj["k_steps"])
            if "num_points" in ckpt_obj:
                num_points = int(ckpt_obj["num_points"])
        elif isinstance(ckpt_obj, dict):
            # raw state_dict
            state_dict = ckpt_obj
        else:
            raise ValueError("Unsupported ckpt format")

    model = mod.GenerativeFlowModel(k_steps=k_steps, num_points=num_points, **cfg)
    if state_dict is not None:
        state_dict = strip_module_prefix(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] missing keys: {len(missing)} (show 20) => {missing[:20]}")
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)} (show 20) => {unexpected[:20]}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)

    # summarize
    summary = summarize_groups(model, detail=args.detail, topk=args.topk)

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] saved json => {outp}")


if __name__ == "__main__":
    main()

# python check_param.py \
# --model_py /path/to/Flow_Generation/3DFlowModel/model_siglip_improved_align_dino_refined.py \
# --ckpt /path/to/Flow_Generation/ckpts_from_scratch_all_dinov2/Flow_k20_n100_lr5e-05_final_lr5e-06_bs32_align_weight0.1_p_uncond0.02_snr_gamma0.0_cond_kframes4_train_noise_timesteps100_motion_moduleTrue_query_pointsTrue/siglip_flow_futureK_best.pt \
# --detail \
# --save_json param_summary.json