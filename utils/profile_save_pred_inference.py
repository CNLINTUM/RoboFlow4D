#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import SiglipProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]


def dynamic_import(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_py", type=str, required=True)
    parser.add_argument("--save_pred_py", type=str, default=str(REPO_ROOT / "3DFlowModel/predict_flow_hdf5.py"))
    parser.add_argument("--siglip_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--data_group", type=str, default="data")
    parser.add_argument("--demo_id", type=str, default="demo_0")
    parser.add_argument("--frames_key", type=str, default="frames_rgb")
    parser.add_argument("--query_key", type=str, default="query_xy_t0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--cond_k", type=int, default=4)
    parser.add_argument("--cond_stride", type=int, default=1)
    parser.add_argument("--k_steps", type=int, default=20)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument(
        "--disable_cfg",
        action="store_true",
        help="Force guidance_scale=1.0 so profiling measures a single conditional forward without CFG batch doubling.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def load_raw_batch(
    save_pred_mod,
    h5_path: Path,
    data_group: str,
    demo_id: str,
    frames_key: str,
    query_key: str,
    start_idx: int,
    batch_size: int,
    cond_k: int,
    cond_stride: int,
) -> Tuple[List[List[np.ndarray]], np.ndarray, str, Dict[str, Any]]:
    with h5py.File(str(h5_path), "r") as f:
        demo = f[data_group][demo_id]
        frames_ds = demo[frames_key]
        query_points = np.asarray(demo[query_key][:], dtype=np.float32)
        query_points = query_points * (256.0 / 518.0)

        total_frames = int(frames_ds.shape[0])
        batch_s = list(range(start_idx, min(total_frames, start_idx + batch_size)))
        if len(batch_s) != batch_size:
            raise ValueError(f"Requested batch_size={batch_size} at start_idx={start_idx}, but only got {len(batch_s)} samples")

        raw_frames_batch: List[List[np.ndarray]] = []
        cond_indices_batch: List[List[int]] = []
        for s in batch_s:
            idxs = save_pred_mod.build_cond_idxs(s, cond_k, cond_stride)
            cond_indices_batch.append(idxs)
            raw_frames_batch.append([np.asarray(frames_ds[i]) for i in idxs])

        instruction = save_pred_mod.normalize_instr_key(h5_path.stem)
        meta = {
            "total_frames": total_frames,
            "batch_s": batch_s,
            "cond_indices_batch": cond_indices_batch,
            "query_points_shape": list(query_points.shape),
            "instruction": instruction,
        }
    return raw_frames_batch, query_points, instruction, meta


def preprocess_batch(
    save_pred_mod,
    processor: SiglipProcessor,
    raw_frames_batch: List[List[np.ndarray]],
    query_points: np.ndarray,
    instruction: str,
    device: str,
):
    pixel_values = []
    for raw_frames in raw_frames_batch:
        imgs = [Image.fromarray(save_pred_mod._to_rgb_hwc(frame)) for frame in raw_frames]
        pv = processor(images=imgs, return_tensors="pt")["pixel_values"]
        pixel_values.append(pv)
    pixel_values = torch.stack(pixel_values, dim=0)

    txt = processor(text=[instruction], return_tensors="pt", padding=True)
    batch_size = pixel_values.shape[0]
    input_ids = txt["input_ids"].expand(batch_size, -1).contiguous()
    attn_mask = txt.get("attention_mask", torch.ones_like(txt["input_ids"])).expand(batch_size, -1).contiguous()
    qp = torch.from_numpy(query_points.astype(np.float32)).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    return (
        pixel_values.to(device, non_blocking=True),
        input_ids.to(device, non_blocking=True),
        attn_mask.to(device, non_blocking=True),
        qp.to(device, non_blocking=True),
    )


def run_sampling_once(
    save_pred_mod,
    model,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    query_points: torch.Tensor,
    k_steps: int,
    num_points: int,
    device: str,
    ddim_steps: int,
    guidance_scale: float,
):
    return save_pred_mod.predict_future_k_ddim_cfg_batch_tensor(
        model=model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attn_mask=attn_mask,
        query_points=query_points,
        k_steps=k_steps,
        num_points=num_points,
        device=device,
        num_train_timesteps=100,
        num_inference_steps=ddim_steps,
        guidance_scale=guidance_scale,
    )


def synchronize_if_cuda(device: str):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_e2e(
    save_pred_mod,
    model,
    processor,
    raw_frames_batch,
    query_points,
    instruction,
    device,
    k_steps,
    num_points,
    ddim_steps,
    guidance_scale,
    warmup,
    repeat,
):
    for _ in range(warmup):
        pv, ids, attn, qp = preprocess_batch(save_pred_mod, processor, raw_frames_batch, query_points, instruction, device)
        _ = run_sampling_once(save_pred_mod, model, pv, ids, attn, qp, k_steps, num_points, device, ddim_steps, guidance_scale)
        synchronize_if_cuda(device)

    latencies_ms = []
    peak_mem_gb = []
    for _ in range(repeat):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        pv, ids, attn, qp = preprocess_batch(save_pred_mod, processor, raw_frames_batch, query_points, instruction, device)
        out = run_sampling_once(save_pred_mod, model, pv, ids, attn, qp, k_steps, num_points, device, ddim_steps, guidance_scale)
        synchronize_if_cuda(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        if str(device).startswith("cuda"):
            peak_mem_gb.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
        else:
            peak_mem_gb.append(0.0)
        del pv, ids, attn, qp, out

    return {
        "latency_mean_ms": float(statistics.mean(latencies_ms)),
        "latency_std_ms": float(statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0),
        "latency_min_ms": float(min(latencies_ms)),
        "latency_max_ms": float(max(latencies_ms)),
        "peak_memory_gb_max": float(max(peak_mem_gb) if peak_mem_gb else 0.0),
        "repeat": int(repeat),
        "warmup": int(warmup),
    }


def benchmark_model_only(
    save_pred_mod,
    model,
    pixel_values,
    input_ids,
    attn_mask,
    query_points,
    device,
    k_steps,
    num_points,
    ddim_steps,
    guidance_scale,
    warmup,
    repeat,
):
    for _ in range(warmup):
        _ = run_sampling_once(save_pred_mod, model, pixel_values, input_ids, attn_mask, query_points, k_steps, num_points, device, ddim_steps, guidance_scale)
        synchronize_if_cuda(device)

    latencies_ms = []
    peak_mem_gb = []
    for _ in range(repeat):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        out = run_sampling_once(save_pred_mod, model, pixel_values, input_ids, attn_mask, query_points, k_steps, num_points, device, ddim_steps, guidance_scale)
        synchronize_if_cuda(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        if str(device).startswith("cuda"):
            peak_mem_gb.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
        else:
            peak_mem_gb.append(0.0)
        del out

    return {
        "latency_mean_ms": float(statistics.mean(latencies_ms)),
        "latency_std_ms": float(statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0),
        "latency_min_ms": float(min(latencies_ms)),
        "latency_max_ms": float(max(latencies_ms)),
        "peak_memory_gb_max": float(max(peak_mem_gb) if peak_mem_gb else 0.0),
        "repeat": int(repeat),
        "warmup": int(warmup),
    }


def profile_flops(
    save_pred_mod,
    model,
    pixel_values,
    input_ids,
    attn_mask,
    query_points,
    device,
    k_steps,
    num_points,
    ddim_steps,
    guidance_scale,
):
    if str(device).startswith("cuda"):
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]

    synchronize_if_cuda(device)
    with profile(activities=activities, record_shapes=False, with_flops=True, profile_memory=False) as prof:
        _ = run_sampling_once(
            save_pred_mod,
            model,
            pixel_values,
            input_ids,
            attn_mask,
            query_points,
            k_steps,
            num_points,
            device,
            ddim_steps,
            guidance_scale,
        )
        synchronize_if_cuda(device)

    total_flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_flops += int(evt.flops)

    return {
        "total_flops": int(total_flops),
        "total_gflops": float(total_flops / 1e9),
        "total_tflops": float(total_flops / 1e12),
        "equivalent_per_ddim_step_gflops": float((total_flops / max(1, ddim_steps)) / 1e9),
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    effective_guidance_scale = 1.0 if args.disable_cfg else float(args.guidance_scale)

    save_pred_py = Path(args.save_pred_py)
    save_pred_mod = dynamic_import(save_pred_py, "save_pred_bench_module")
    save_pred_mod.set_seed(args.seed)
    save_pred_mod.enable_sdpa()

    processor = SiglipProcessor.from_pretrained(args.siglip_name, use_fast=True)
    model, k_ckpt, n_ckpt, cfg = save_pred_mod.load_flow_model(
        ckpt_path=Path(args.ckpt),
        model_py=Path(args.model_py),
        device=args.device,
        k_steps_override=args.k_steps,
        num_points_override=None,
        strict=True,
    )
    num_points = int(n_ckpt)

    raw_frames_batch, query_points, instruction, meta = load_raw_batch(
        save_pred_mod=save_pred_mod,
        h5_path=Path(args.h5_path),
        data_group=args.data_group,
        demo_id=args.demo_id,
        frames_key=args.frames_key,
        query_key=args.query_key,
        start_idx=args.start_idx,
        batch_size=args.batch_size,
        cond_k=args.cond_k,
        cond_stride=args.cond_stride,
    )

    pv, ids, attn, qp = preprocess_batch(
        save_pred_mod=save_pred_mod,
        processor=processor,
        raw_frames_batch=raw_frames_batch,
        query_points=query_points,
        instruction=instruction,
        device=args.device,
    )

    e2e = benchmark_e2e(
        save_pred_mod=save_pred_mod,
        model=model,
        processor=processor,
        raw_frames_batch=raw_frames_batch,
        query_points=query_points,
        instruction=instruction,
        device=args.device,
        k_steps=args.k_steps,
        num_points=num_points,
        ddim_steps=args.ddim_steps,
        guidance_scale=effective_guidance_scale,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    model_only = benchmark_model_only(
        save_pred_mod=save_pred_mod,
        model=model,
        pixel_values=pv,
        input_ids=ids,
        attn_mask=attn,
        query_points=qp,
        device=args.device,
        k_steps=args.k_steps,
        num_points=num_points,
        ddim_steps=args.ddim_steps,
        guidance_scale=effective_guidance_scale,
        warmup=max(1, min(2, args.warmup)),
        repeat=max(1, min(5, args.repeat)),
    )

    flops = profile_flops(
        save_pred_mod=save_pred_mod,
        model=model,
        pixel_values=pv,
        input_ids=ids,
        attn_mask=attn,
        query_points=qp,
        device=args.device,
        k_steps=args.k_steps,
        num_points=num_points,
        ddim_steps=args.ddim_steps,
        guidance_scale=effective_guidance_scale,
    )

    result = {
        "h5_path": str(args.h5_path),
        "demo_id": args.demo_id,
        "instruction": instruction,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "start_idx": int(args.start_idx),
        "cond_k": int(args.cond_k),
        "cond_stride": int(args.cond_stride),
        "k_steps": int(args.k_steps),
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale_requested": float(args.guidance_scale),
        "guidance_scale_effective": float(effective_guidance_scale),
        "disable_cfg": bool(args.disable_cfg),
        "num_points": int(num_points),
        "model_cfg_keys": list(cfg.keys()),
        "input_shapes": {
            "pixel_values": list(pv.shape),
            "input_ids": list(ids.shape),
            "attention_mask": list(attn.shape),
            "query_points": list(qp.shape),
        },
        "batch_meta": meta,
        "e2e_latency": e2e,
        "model_only_latency": model_only,
        "flops": flops,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
