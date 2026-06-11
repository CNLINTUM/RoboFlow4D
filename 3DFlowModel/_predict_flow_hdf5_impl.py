"""
Internal implementation for the HDF5 flow inference entry point.

Predict future 3D point trajectories with a trained flow model and write them
back into each HDF5 demo. The default output key is `pre_point_traj`.

Example:
  python predict_flow_hdf5.py \
    --hdf5_root /path/to/Flow_data_features_hdf5 \
    --ckpt /path/to/ckpt.pt \
    --model_py 3DFlowModel/flow_model.py \
    --k_steps 20 --ddim_steps 50 --guidance_scale 2.0 \
    --frames_key frames_rgb --query_key query_points --instr_attr prompt \
    --out_key pre_point_traj --overwrite
"""

import os
import re
import sys
import math
import time
import json
import random
import argparse
import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import h5py
from PIL import Image

import cv2  # for PnP & projection utils

import torch
from transformers import SiglipProcessor
import matplotlib.pyplot as plt
from matplotlib import cm

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_sdpa():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception:
        pass


def _to_rgb_hwc(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    a = np.squeeze(a)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        if a.shape[-1] in (3, 4):
            a = a[..., :3]
        elif a.shape[0] in (3, 4):
            a = np.moveaxis(a, 0, -1)[..., :3]
        else:
            a2 = np.squeeze(a)
            if a2.ndim == 2:
                a = np.stack([a2, a2, a2], axis=-1)
            else:
                raise ValueError(f"Unexpected image shape after squeeze: {a.shape}")
    else:
        raise ValueError(f"Unexpected image shape: {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a

def build_cond_idxs(s: int, cond_k: int, cond_stride: int) -> List[int]:
    """Use the same history-frame sampling as training: step backward from s and pad with the first frame if needed."""
    idxs = list(range(max(0, s - (cond_k - 1) * cond_stride), s + 1, cond_stride))
    if len(idxs) > cond_k:
        idxs = idxs[-cond_k:]
    while len(idxs) < cond_k:
        idxs.insert(0, idxs[0])
    return idxs

def debounce_gripper_changes(gb: np.ndarray, debounce: int = 3) -> List[int]:
    """Return indices where a debounced new gripper state starts."""
    gb = np.asarray(gb, dtype=np.int32)
    if gb.size <= 1:
        return []
    if debounce <= 1:
        return (np.where(np.diff(gb) != 0)[0] + 1).astype(int).tolist()

    cur = int(gb[0])
    changes: List[int] = []
    i = 1
    while i < gb.size:
        if int(gb[i]) == cur:
            i += 1
            continue
        new = int(gb[i])
        ok = True
        for j in range(i, min(gb.size, i + debounce)):
            if int(gb[j]) != new:
                ok = False
                break
        if ok:
            changes.append(i)
            cur = new
            i += debounce
        else:
            i += 1
    return changes

def segment_gripper_state(actions: Optional[np.ndarray],
                          t_len: int,
                          debounce: int = 3) -> List[Tuple[int, int]]:
    """Build inclusive gripper-state segments covering [0, t_len - 1]."""
    t_len = int(t_len)
    if t_len <= 0:
        return []
    if actions is None or len(actions) == 0:
        return [(0, t_len - 1)]

    a = np.asarray(actions)
    if a.ndim == 2:
        g = a[:, -1]
    else:
        g = a.reshape(-1)
    usable = min(t_len, int(g.shape[0]))
    if usable <= 0:
        return [(0, t_len - 1)]

    gb = (np.nan_to_num(g[:usable].astype(np.float32)) > 0).astype(np.int32)
    change_idxs = debounce_gripper_changes(gb, debounce=debounce)
    boundaries = sorted(set([0] + change_idxs + [usable]))

    segments: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s0 = int(boundaries[i])
        s1 = int(boundaries[i + 1] - 1)
        if s1 >= s0:
            segments.append((s0, s1))
    if not segments:
        segments = [(0, usable - 1)]
    if usable < t_len:
        last_start, _ = segments[-1]
        segments[-1] = (last_start, t_len - 1)
    return segments

def find_segment_end(segments: List[Tuple[int, int]], s: int, t_len: int) -> int:
    if not segments:
        return max(0, int(t_len) - 1)
    s = int(s)
    for s0, s1 in segments:
        if s0 <= s <= s1:
            return int(s1)
    return int(segments[-1][1] if s > segments[-1][1] else segments[0][1])

def resample_on_traj(start: int,
                     end: int,
                     num: int,
                     gamma_s: float = 1.2,
                     gamma_e: float = 1.6,
                     min_unique_ratio: float = 0.7) -> np.ndarray:
    """Same endpoint-dense keyframe resampling used by flow model training."""
    start = int(start)
    end = int(end)
    num = int(num)
    if num <= 0:
        return np.zeros((0,), dtype=np.int64)
    if end <= start:
        return np.array([start] * num, dtype=np.int64)

    u = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
    w = np.empty_like(u)
    left = u <= 0.5
    w[left] = 0.5 * np.power(2.0 * u[left], gamma_s)
    w[~left] = 1.0 - 0.5 * np.power(2.0 * (1.0 - u[~left]), gamma_e)

    idxs = np.round(start + (end - start) * w).astype(np.int64)
    idxs[0] = start
    idxs[-1] = end
    idxs = np.maximum.accumulate(idxs)

    if np.unique(idxs).size < int(min_unique_ratio * num):
        idxs = np.round(np.linspace(start, end, num=num)).astype(np.int64)
        idxs[0] = start
        idxs[-1] = end
        idxs = np.maximum.accumulate(idxs)
    return idxs

def h5_take_time(ds, idxs: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Take possibly repeated time indices from an HDF5 dataset."""
    idxs = np.asarray(idxs, dtype=np.int64)
    if idxs.size == 0:
        return np.asarray(ds[:0], dtype=dtype)
    idxs = np.clip(idxs, 0, int(ds.shape[0]) - 1)
    uniq, inv = np.unique(idxs, return_inverse=True)
    buf = np.asarray(ds[uniq], dtype=dtype)
    return buf[inv]

def overwrite_h5_dataset(grp: h5py.Group,
                         name: str,
                         arr: np.ndarray,
                         compression: str = "gzip",
                         chunks: bool = True,
                         dtype=np.float32):
    """Overwrite a dataset inside a demo group by deleting and recreating it."""
    if name in grp:
        del grp[name]
    arr = np.asarray(arr, dtype=dtype)
    grp.create_dataset(name, data=arr, compression=compression, chunks=chunks)

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle the 'module.' prefix used by DDP checkpoints."""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

# -----------------------------
# Model loading
# -----------------------------
def dynamic_import_model(model_py: Path, module_name: str = "flow_model_module"):
    if not model_py.exists():
        raise FileNotFoundError(f"model_py not found: {model_py}")
    spec = importlib.util.spec_from_file_location(module_name, str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create spec for {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_flow_model(ckpt_path: Path,
                    model_py: Path,
                    device: str,
                    k_steps_override: Optional[int] = None,
                    num_points_override: Optional[int] = None,
                    strict: bool = False):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("cfg", {}) or {}

    k_steps = int(k_steps_override if k_steps_override is not None else ckpt.get("k_steps", 20))
    num_points = int(num_points_override if num_points_override is not None else ckpt.get("num_points", 128))

    model_module = dynamic_import_model(model_py)
    if not hasattr(model_module, "GenerativeFlowModel"):
        raise AttributeError(f"{model_py} does not define GenerativeFlowModel")

    model = model_module.GenerativeFlowModel(k_steps=k_steps, num_points=num_points, **cfg).to(device)

    state = ckpt.get("model", None)
    if state is None:
        state = ckpt.get("state_dict", None)
    if state is None and isinstance(ckpt, dict):
        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected):
        print(f"[WARN] load_state_dict strict={strict}: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) < 20 and len(unexpected) < 20:
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

    model.eval()
    return model, k_steps, num_points, cfg


def make_a_bar(num_train_timesteps: int, device,
               beta_start: float = 1e-4, beta_end: float = 1e-1):
    betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)  # a_bar[t], t in [0, num_train_timesteps-1]

@torch.inference_mode()
def predict_future_k_ddim_cfg_batch_tensor(
    model,
    pixel_values: torch.Tensor,      # [B, Kc, 3, 224, 224]
    input_ids: torch.Tensor,         # [B, L]
    attn_mask: torch.Tensor,         # [B, L]
    query_points: Optional[torch.Tensor],      # [B, N, 2/3] or None
    k_steps: int,
    num_points: int,
    device: str,
    num_train_timesteps: int = 50,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.0,
):
    B = pixel_values.shape[0]
    x_t = torch.randn(B, k_steps, num_points, 3, device=device)

    a_bar = make_a_bar(num_train_timesteps, device=device)
    ts = torch.linspace(num_train_timesteps - 1, 0, num_inference_steps, device=device).long()

    for i, t in enumerate(ts):
        t_int = int(t.item())
        t_vec = torch.full((B,), t_int, device=device, dtype=torch.long)

        if guidance_scale is None or guidance_scale == 1.0:
            v_hat = model(
                image_pixels=pixel_values,
                instruction_input_ids=input_ids,
                instruction_attention_mask=attn_mask,
                query_points=query_points,
                noisy_flow=x_t,
                timestep=t_vec,
                drop_condition_mask=torch.zeros(B, device=device, dtype=torch.bool),
            )
        else:
            x_in = torch.cat([x_t, x_t], dim=0)                 # [2B, ...]
            t_in = torch.cat([t_vec, t_vec], dim=0)             # [2B]
            drop_mask = torch.cat([
                torch.zeros(B, device=device, dtype=torch.bool),  # cond
                torch.ones(B,  device=device, dtype=torch.bool),  # uncond (drop all cond)
            ], dim=0)

            pv_in   = torch.cat([pixel_values, pixel_values], dim=0)
            ids_in  = torch.cat([input_ids, input_ids], dim=0)
            att_in  = torch.cat([attn_mask, attn_mask], dim=0)
            qp_in   = torch.cat([query_points, query_points], dim=0) if query_points is not None else None

            v_out = model(
                image_pixels=pv_in,
                instruction_input_ids=ids_in,
                instruction_attention_mask=att_in,
                query_points=qp_in,
                noisy_flow=x_in,
                timestep=t_in,
                drop_condition_mask=drop_mask,
            )
            v_cond, v_uncond = v_out.chunk(2, dim=0)
            v_hat = v_uncond + guidance_scale * (v_cond - v_uncond)

        # v-pred -> x0, eps
        s = torch.sqrt(a_bar[t])
        c = torch.sqrt(1.0 - a_bar[t])
        x0  = s * x_t - c * v_hat
        eps = c * x_t + s * v_hat

        # DDIM eta=0 update
        if i < len(ts) - 1:
            t_next = ts[i + 1]
            s_next = torch.sqrt(a_bar[t_next])
            c_next = torch.sqrt(1.0 - a_bar[t_next])
            x_t = s_next * x0 + c_next * eps
        else:
            x_t = x0

    return x_t.detach().float().cpu().numpy()



# -----------------------------
# Stitch full trajectory
# -----------------------------
from contextlib import nullcontext

@torch.inference_mode()
def encode_all_frames(processor: SiglipProcessor,
                      frames_ds,
                      chunk: int = 64) -> torch.Tensor:
    """Encode an HDF5 frames dataset into CPU SigLIP pixel values [T, 3, 224, 224]."""
    T = int(frames_ds.shape[0])
    out = []
    for i in range(0, T, chunk):
        arr = frames_ds[i:min(T, i + chunk)]  # numpy
        imgs = []
        for j in range(arr.shape[0]):
            rgb = _to_rgb_hwc(arr[j])
            imgs.append(Image.fromarray(rgb))
        pv = processor(images=imgs, return_tensors="pt")["pixel_values"]  # [b,3,224,224] CPU
        out.append(pv)
    return torch.cat(out, dim=0)

def infer_frame_hw(frames_shape) -> Tuple[int, int]:
    if len(frames_shape) < 3:
        raise ValueError(f"Unexpected frames shape: {frames_shape}")
    if len(frames_shape) >= 4:
        if int(frames_shape[-1]) in (1, 3, 4):
            return int(frames_shape[1]), int(frames_shape[2])
        if int(frames_shape[1]) in (1, 3, 4):
            return int(frames_shape[2]), int(frames_shape[3])
    return int(frames_shape[-3]), int(frames_shape[-2])


def normalize_query_points_for_model(
    query_points: np.ndarray,
    frame_hw: Tuple[int, int],
    spatracker_res: float = 518.0,
) -> np.ndarray:
    """Map query points to the [0, 1] coordinates used by flow training."""
    q = np.asarray(query_points, dtype=np.float32)
    if q.ndim not in (2, 3) or q.shape[-1] < 2:
        raise ValueError(f"query_points must be [N,2+] or [T,N,2+], got {q.shape}")
    q = np.nan_to_num(q[..., :2].copy(), nan=0.0, posinf=0.0, neginf=0.0)

    q_max = float(np.nanmax(q)) if q.size else 0.0
    q_min = float(np.nanmin(q)) if q.size else 0.0
    if q_max <= 2.0 and q_min >= -0.5:
        return np.clip(q, 0.0, 1.0).astype(np.float32)

    h, w = int(frame_hw[0]), int(frame_hw[1])
    if q_max > max(h, w) * 1.2:
        q[..., 0] *= float(w) / float(spatracker_res)
        q[..., 1] *= float(h) / float(spatracker_res)

    q[..., 0] /= max(float(w - 1), 1.0)
    q[..., 1] /= max(float(h - 1), 1.0)
    return np.clip(q, 0.0, 1.0).astype(np.float32)



def rescale(traj_uv: np.ndarray):
    scaled_traj_uv = traj_uv * 256 / 518
    return scaled_traj_uv

def solve_w2c_via_pnp(Pw: np.ndarray, uv: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Through PnP estimate world→camera extrinsics: X_cam = R * X_world + t
    Pw: (N,3) world points; uv: (N,2) pixels.
    Returns (R[3,3], t[3]).
    """
    assert Pw.shape[0] >= 6, "PnP needs >=6 points for stability"
    mask = np.isfinite(Pw).all(1) & np.isfinite(uv).all(1)
    Pw = Pw[mask].astype(np.float32)
    uv = uv[mask].astype(np.float32)
    if Pw.shape[0] < 6:
        raise RuntimeError("Not enough valid correspondences for PnP after filtering")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=Pw, imagePoints=uv, cameraMatrix=K, distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=3.0, iterationsCount=300
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(Pw, uv, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float32), tvec.reshape(3).astype(np.float32)


def project_world_to_pixels(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """K[3,3], R[3,3], t[3], Xw[N,3] -> uv[N,2]."""
    Xc = (R @ Xw.T + t[:, None]).T           # N×3
    z = np.clip(Xc[:, 2:3], 1e-6, None)
    uvw = (K @ Xc.T).T                       # N×3
    return uvw[:, :2] / z

def draw_pred_gt_tracks_on_image(
    img: np.ndarray,
    pred_uv: np.ndarray,          # [K, N, 2]
    gt_uv: Optional[np.ndarray],  # [K, N, 2] or None
    title: str,
    out_path: Path,
    stride: int = 1,
    cmap_name: str = "viridis",
    alpha_pred: float = 0.9,
    alpha_gt: float = 0.55,
):
    H, W = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    plt.imshow(img)

    # rescale to your pixel coord convention
    pred_uv = rescale(pred_uv)
    if gt_uv is not None:
        gt_uv = rescale(gt_uv)

    Kp = pred_uv.shape[0]
    colors = cm.get_cmap(cmap_name, max(Kp - 1, 1))

    for n in range(0, pred_uv.shape[1], stride):
        # ---------- GT (dashed) ----------
        if gt_uv is not None:
            pts_g = gt_uv[:, n, :]  # [K,2]
            m_g = np.isfinite(pts_g).all(axis=1)
            if m_g.sum() >= 2:
                plt.plot(pts_g[m_g, 0], pts_g[m_g, 1], '--', linewidth=1.6, alpha=alpha_gt, color='white')
                plt.plot(pts_g[m_g, 0], pts_g[m_g, 1], '--', linewidth=1.0, alpha=alpha_gt, color='black')

        # ---------- Pred (solid) ----------
        pts_p = pred_uv[:, n, :]  # [K,2]
        m_p = np.isfinite(pts_p).all(axis=1)
        if m_p.sum() >= 2:
            plt.plot(pts_p[m_p, 0], pts_p[m_p, 1], '-', linewidth=1.0, alpha=alpha_pred, color=colors(Kp - 2 if Kp >= 2 else 0))

            first_idx = np.where(m_p)[0][0]
            last_idx  = np.where(m_p)[0][-1]
            plt.scatter(pts_p[first_idx, 0], pts_p[first_idx, 1], s=10, c='white',
                        edgecolors='black', linewidths=0.3, alpha=0.8)
            plt.scatter(pts_p[last_idx, 0], pts_p[last_idx, 1], s=10, c='red',
                        edgecolors='black', linewidths=0.3, alpha=0.8)

    plt.text(
        8, 18, "solid=pred, dashed=GT",
        fontsize=9, color="white",
        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none")
    )

    plt.axis('off'); plt.tight_layout(pad=0)
    if title:
        plt.title(title, fontsize=10)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_tracks(
    images,
    traj_world_pred,          # [K, N, 3]
    intrinsics,
    P0_world,
    track2d,
    instruction,
    demo_id,
    step,
    traj_world_gt=None,       # [K, N, 3] or None
    k_steps=20,
    out_dir=None,
):
    K = intrinsics
    R, t = solve_w2c_via_pnp(P0_world, track2d, K)

    # ----- pred: world -> uv -----
    pred_uv = []
    for j in range(traj_world_pred.shape[0]):
        uv = project_world_to_pixels(K, R, t, traj_world_pred[j])  # [N,2]
        pred_uv.append(uv)
    pred_uv = np.stack(pred_uv, axis=0)  # [K,N,2]

    # ----- gt: world -> uv (optional) -----
    gt_uv = None
    if traj_world_gt is not None:
        gt_uv_list = []
        for j in range(traj_world_gt.shape[0]):
            uv = project_world_to_pixels(K, R, t, traj_world_gt[j])  # [N,2]
            gt_uv_list.append(uv)
        gt_uv = np.stack(gt_uv_list, axis=0)  # [K,N,2]

    safe_instruction = re.sub(r"[^\w\-_.() ]+", "_", str(instruction)).strip()
    debug_root = Path(out_dir) if out_dir is not None else Path("outputs/debug_tracks")
    path = debug_root / safe_instruction

    path.mkdir(parents=True, exist_ok=True)
    tracks_path = path / f"tracks_checks_{demo_id}_{step}.png"

    frame_np = images
    draw_pred_gt_tracks_on_image(
        frame_np,
        pred_uv=pred_uv,
        gt_uv=gt_uv,
        title=f"Pred+GT 2D tracks (K={k_steps})",
        out_path=tracks_path,
        stride=1,
        cmap_name="viridis",
        alpha_pred=0.9,
        alpha_gt=0.55,
    )


@torch.inference_mode()
def infer_pred_point_traj_for_demo(
    model,
    processor: SiglipProcessor,
    frames_ds,
    query_points_np: Optional[np.ndarray],
    instruction: str,
    k_steps: int,
    num_points: int,
    ddim_steps: int,
    cond_k: int,
    cond_stride: int,
    window_bs: int,
    device: str,
    guidance_scale
) -> np.ndarray:
    """
    Returns:
        pred_futures: [T, K, N, 3], where pred_futures[s] contains K resampled
        keyframes from the current frame to the end of the current gripper
        segment.
    """
    T = int(frames_ds.shape[0])
    qp_tensor = None
    qp_dynamic = False
    if query_points_np is not None:
        qp = np.asarray(query_points_np)
        if qp.ndim == 2:
            N = int(qp.shape[0])
            qp_tensor = torch.from_numpy(qp.astype(np.float32)).unsqueeze(0).to(device)
        elif qp.ndim == 3:
            if qp.shape[0] < T:
                raise ValueError(f"dynamic query_points has T={qp.shape[0]} but frames have T={T}")
            qp = qp[:T]
            N = int(qp.shape[1])
            qp_dynamic = True
            qp_tensor = torch.from_numpy(qp.astype(np.float32))
        else:
            raise ValueError(f"query_points must be [N,D] or [T,N,D], got {qp.shape}")

        if N != num_points:
            print(f"[WARN] num_points mismatch: model expects {num_points}, query_points has {N}. Use N={N}.")
            num_points = N
    else:
        N = int(num_points)

    # 1) encode frames once
    pixel_values_frames = encode_all_frames(processor, frames_ds, chunk=64)  # CPU [T,3,224,224]

    # 2) tokenize instruction once
    txt = processor(text=[instruction], return_tensors="pt", padding=True)
    input_ids_1 = txt["input_ids"].to(device)
    attn_1 = txt.get("attention_mask", torch.ones_like(input_ids_1)).to(device)

    pred_futures = np.full((T, k_steps, num_points, 3), np.nan, dtype=np.float32)

    all_s = list(range(T))
    for st in range(0, len(all_s), window_bs):
        print("\nProcessing window starting at index:", st)
        batch_s = all_s[st: st + window_bs]
        B = len(batch_s)

        idx_mat = torch.tensor([build_cond_idxs(s, cond_k, cond_stride) for s in batch_s], dtype=torch.long)
        pv = pixel_values_frames[idx_mat]  # CPU [B,Kc,3,224,224]
        pv = pv.to(device, non_blocking=True)

        input_ids = input_ids_1.expand(B, -1)
        attn_mask = attn_1.expand(B, -1)
        if qp_tensor is None:
            qp_batch = None
        elif qp_dynamic:
            qp_batch = qp_tensor[batch_s].to(device, non_blocking=True)
        else:
            qp_batch = qp_tensor.expand(B, -1, -1)

        with torch.no_grad():
            pred_windows = predict_future_k_ddim_cfg_batch_tensor(
                model=model,
                pixel_values=pv,
                input_ids=input_ids,
                attn_mask=attn_mask,
                query_points=qp_batch,
                k_steps=k_steps,
                num_points=num_points,
                device=device,
                num_train_timesteps=100,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale
            )

        for bi, s in enumerate(batch_s):
            pred_futures[s] = pred_windows[bi]  # [K, N, 3]
    return pred_futures

# -----------------------------
# HDF5 processing
# -----------------------------
def get_first_existing_key(grp: h5py.Group, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in grp:
            return k
    return None

def normalize_instr_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("_", " ").strip()

    # Remove prefixes such as "kitchen scene4", "living room scene5", or "study scene1".
    s = re.sub(r"^(?:kitchen|living room|study)\s+scene\s*\d+\s+", "", s)

    # Strip common "tracks" / "track(s)" suffixes, optionally with "demo".
    s = re.sub(r"(?:\s+demo)?\s+tracks?\s*$", "", s)

    # Then handle odd suffixes such as "demo", "demo 123", or "demo demo 123".
    s = re.sub(r"(?:\s|_)+(?:demo(?:\s|_)+demo|\bdemo)\s*\d+\s*$", "", s)
    s = re.sub(r"(?:\s|_)+demo\s*$", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def process_hdf5_file(
    h5_path: Path,
    model,
    processor: SiglipProcessor,
    args,
):    
    with h5py.File(str(h5_path), "r+") as f:
        if args.data_group not in f:
            return 0

        data_grp = f[args.data_group]
        demo_ids = list(data_grp.keys())
        if args.max_demos is not None:
            demo_ids = demo_ids[:args.max_demos]

        wrote = 0
        for demo_id in demo_ids:
            demo = data_grp[demo_id]

            frames_key = get_first_existing_key(demo, ["frames_rgb"])
            if frames_key is None:
                continue

            # query points are optional. Models trained without --query_points
            # should be sampled with args.no_query_points=True.
            qp_key = get_first_existing_key(demo, [args.query_key, "query_xy_t0"])
            if (not args.no_query_points) and qp_key is None:
                continue


            target_flows = demo["point_traj"]

            out_key = args.out_key
            if (out_key in demo) and (not args.overwrite):
                continue

            frames_ds = demo[frames_key]
            if args.no_query_points:
                query_points = None
                qp_key = None
            else:
                query_points = normalize_query_points_for_model(
                    demo[qp_key][:],
                    frame_hw=infer_frame_hw(frames_ds.shape),
                )

            intrinsics = demo['intrs2']

            P0_world = demo["point_traj"][0]  # do not mmap here since we slice twice

            track2d = demo['track2d'][0]

            x = demo["frames_rgb"][0]          # (3,256,256) float32         [131., 134., 135., ..., 108., 105., 106.],
            print(x.dtype, float(x.min()), float(x.max()), float(x.mean()))

            if query_points is not None:
                print("qp shape:", query_points.shape)
                qp_flat = query_points.reshape(-1, query_points.shape[-1])
                print("qp key:", qp_key, "dynamic:", query_points.ndim == 3)
                print("qp min:", qp_flat.min(axis=0), "qp max:", qp_flat.max(axis=0))
            else:
                print("qp: disabled (--no_query_points)")

            instruction = normalize_instr_key(h5_path.stem)
            pred_traj = infer_pred_point_traj_for_demo(
                model=model,
                processor=processor,
                frames_ds=frames_ds,
                query_points_np=query_points,
                instruction=str(instruction),
                k_steps=args.k_steps,
                num_points=args.num_points,
                ddim_steps=args.ddim_steps,
                cond_k=args.cond_k,
                cond_stride=args.cond_stride,
                window_bs=args.window_bs,
                device=args.device,
                guidance_scale=args.guidance_scale
            )

            if args.debug_tracks_dir is not None:
                T_gt = demo["point_traj"].shape[0]
                K = args.k_steps
                actions_np = np.asarray(demo["actions"][:]) if "actions" in demo else None
                segments = segment_gripper_state(actions_np, T_gt, debounce=3)

                for step in range(pred_traj.shape[0]):
                    # GT keyframes: [K, N, 3] aligned with pred_traj[step].
                    gt_world = None
                    if step < T_gt:
                        seg_end = find_segment_end(segments, step, T_gt)
                        key_idxs = resample_on_traj(step, seg_end, K)
                        key_idxs = np.clip(key_idxs, 0, T_gt - 1)
                        gt_world = h5_take_time(demo["point_traj"], key_idxs, dtype=np.float32)

                    draw_tracks(
                        _to_rgb_hwc(frames_ds[step]),
                        traj_world_pred=pred_traj[step],                 # [K,N,3]
                        traj_world_gt=gt_world,                          # [K,N,3] or None
                        intrinsics=intrinsics[0],
                        P0_world=P0_world,
                        track2d=track2d[:, :2],
                        instruction=instruction,
                        demo_id=demo_id,
                        step=step,
                        k_steps=K,
                        out_dir=args.debug_tracks_dir,
                    )

            if not args.dry_run:
                overwrite_h5_dataset(demo, out_key, pred_traj, compression="gzip", chunks=True)
                demo.attrs[f"{out_key}_ckpt"] = str(args.ckpt)
                demo.attrs[f"{out_key}_ddim_steps"] = int(args.ddim_steps)
                demo.attrs[f"{out_key}_guidance_scale"] = float(args.guidance_scale)
                demo.attrs[f"{out_key}_k_steps"] = int(args.k_steps)
                demo.attrs[f"{out_key}_query_points"] = bool(not args.no_query_points)
                demo.attrs[f"{out_key}_query_key"] = "" if qp_key is None else str(qp_key)
                demo.attrs[f"{out_key}_dynamic_query_points"] = bool(query_points is not None and query_points.ndim == 3)
                wrote += 1
        if not args.dry_run:
            f.flush()
        return wrote


def list_hdf5_files(root: Path) -> List[Path]:
    files = sorted([p for p in root.rglob("*.hdf5") if p.is_file()])
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", type=str, required=True, help="Directory searched recursively for HDF5 files.")
    parser.add_argument("--ckpt", type=str, required=True, help="flow diffusion model checkpoint")
    parser.add_argument("--model_py", type=str, default="flow_model.py",
                        help="Python file containing GenerativeFlowModel.")
    parser.add_argument("--siglip_name", type=str, default="google/siglip-base-patch16-224")

    parser.add_argument("--data_group", type=str, default="data", help="Top-level HDF5 group containing demos.")
    parser.add_argument("--frames_key", type=str, default="frames_rgb")
    parser.add_argument(
        "--query_key",
        type=str,
        default="query_points",
        help="Query point dataset. Use track2d for checkpoints trained with per-frame --query_points.",
    )
    parser.add_argument(
        "--no_query_points",
        action="store_true",
        help="Use query_points=None during sampling. Enable this for checkpoints trained without --query_points.",
    )
    parser.add_argument("--instr_attr", type=str, default="prompt", help="Demo attribute containing the instruction/prompt.")

    parser.add_argument("--out_key", type=str, default="pre_point_traj", help="Output dataset name written under each demo group.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output dataset if it already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Run inference without writing HDF5 outputs.")
    parser.add_argument(
        "--debug_tracks_dir",
        type=str,
        default=None,
        help="Optional directory for 2D track debug images. Disabled by default.",
    )

    parser.add_argument("--k_steps", type=int, default=None, help="Prediction horizon. Defaults to checkpoint metadata or 20.")
    parser.add_argument("--num_points", type=int, default=None, help="Number of query points. Defaults to checkpoint metadata or 128.")

    parser.add_argument("--cond_k", type=int, default=4)
    parser.add_argument("--cond_stride", type=int, default=1)
    parser.add_argument("--window_bs", type=int, default=150)

    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast with bfloat16 when available.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--norm_stats", type=str, default=None,
                        help="path to norm_stats_allpoints.json (same one used in training)")

    args = parser.parse_args()
    if args.norm_stats is not None:
        with open(args.norm_stats, "r") as rf:
            args.norm_stats = json.load(rf)

    set_seed(args.seed)
    enable_sdpa()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    hdf5_root = Path(args.hdf5_root)
    ckpt_path = Path(args.ckpt)
    model_py = Path(args.model_py)
    if not model_py.is_absolute() and not model_py.exists():
        here = Path(__file__).resolve().parent
        cand = here / model_py
        if cand.exists():
            model_py = cand
    args.model_py = str(model_py)

    print("[INFO] Loading SigLIP processor:", args.siglip_name)
    processor = SiglipProcessor.from_pretrained(args.siglip_name, use_fast=True)

    print("[INFO] Loading model from ckpt:", ckpt_path)
    model, k_ckpt, n_ckpt, cfg = load_flow_model(
        ckpt_path=ckpt_path,
        model_py=model_py,
        device=args.device,
        k_steps_override=args.k_steps,
        num_points_override=args.num_points,
        strict=True,
    )
    args.k_steps = int(args.k_steps if args.k_steps is not None else k_ckpt)
    args.num_points = int(args.num_points if args.num_points is not None else n_ckpt)

    print(f"[INFO] device={args.device}, k_steps={args.k_steps}, num_points={args.num_points}, out_key={args.out_key}")
    print(f"[INFO] model cfg keys: {list(cfg.keys())}")

    files = list_hdf5_files(hdf5_root)
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f"[INFO] Found {len(files)} hdf5 files under {hdf5_root}")

    total_wrote = 0
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        wrote = process_hdf5_file(fp, model, processor, args)
        total_wrote += wrote
        if wrote > 0:
            print(f"[{i}/{len(files)}] wrote {wrote} demos into {fp}")
    t1 = time.time()
    print(f"[DONE] total demos written: {total_wrote}, elapsed: {t1 - t0:.1f}s")
    if args.dry_run:
        print("[NOTE] dry_run=True: no file was modified.")


if __name__ == "__main__":
    main()
