#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow/Track generation with Grounded-SAM2 + SpatialTrackerV2,
now supporting **prompt lookup from a CSV** + **VGGT chunked inference**
+ **optional freeing of SAM2/GDINO after first-frame mask**.
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import sys as _sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import cv2
from PIL import Image
import h5py
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as T

import random


# ---------------------- Optional local repo resolution ----------------------
WORK = REPO_ROOT
GSA2_ROOT = WORK / "Grounded-SAM-2"
SPA_ROOT  = WORK / "SpaTrackerV2"

def _maybe_prepend_paths():
    paths_to_add = [str(REPO_ROOT)]
    if GSA2_ROOT.exists(): paths_to_add.append(str(GSA2_ROOT))
    if SPA_ROOT.exists():  paths_to_add.append(str(SPA_ROOT))
    for p in reversed(paths_to_add):
        if p not in _sys.path:
            _sys.path.insert(0, p)
_maybe_prepend_paths()

# -------------------------- External deps / models --------------------------
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model as gdino_load_model
from grounding_dino.groundingdino.util.inference import load_image as gdino_load_image
from grounding_dino.groundingdino.util.inference import predict  as gdino_predict

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.utils.visualizer import Visualizer


from utils.pooling_utils import custom_pooling # import pooling function


# --------------------------------- CLI -------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Process Libero HDF5 files to generate flow and tracking data (with CSV prompts).")
    # IO
    p.add_argument('--input_dirs', nargs='+', required=True, help='List of input directories containing Libero HDF5 files.')
    p.add_argument('--out_root', required=True, help='Root output folder to save results.')

    # GSAM2 paths
    p.add_argument('--gdino.config', dest='gdino_config', default=str(GSA2_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    p.add_argument('--gdino.ckpt',   dest='gdino_ckpt',   default=str(GSA2_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth"))
    p.add_argument('--sam2.config',  dest='sam2_config',  default=str(GSA2_ROOT / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"))
    p.add_argument('--sam2.ckpt',    dest='sam2_ckpt',    default=str(GSA2_ROOT / "checkpoints/sam2.1_hiera_large.pt"))

    # GSAM2 params
    p.add_argument('--multimask_output', action='store_true')
    p.add_argument('--box_thresh', type=float, default=0.35)   # 0.35
    p.add_argument('--text_thresh', type=float, default=0.30)  # 0.25
    p.add_argument(
        '--gripper_box_shift_y',
        type=float,
        default=60.0,
        help='Pixel y-offset applied to GroundingDINO gripper boxes before SAM. Use 0 for ManiSkill gripper-only tracks.',
    )
    p.add_argument(
        '--gripper_box_delta_h',
        type=float,
        default=-80.0,
        help='Pixel height delta applied to GroundingDINO gripper boxes before SAM. Use 0 for ManiSkill gripper-only tracks.',
    )
    p.add_argument(
        '--gripper_mask_keep_top_frac',
        type=float,
        default=1.0,
        help='For gripper boxes, keep only the top fraction of the SAM mask inside the adjusted box. Use <1 to avoid including objects below the gripper.',
    )

    # Prompt options
    p.add_argument(
        '--prompt',
        default='robot gripper',
        help='Grounding prompt for mask/query selection. Override this for non-gripper targets.',
    )
    p.add_argument('--prompt_from_text', action='store_true')
    p.add_argument('--prompt_max_words', type=int, default=6)
    p.add_argument('--add_prefix', default=None)

    # CSV prompt options
    p.add_argument('--prompt_csv', default=None, help='Path to CSV containing prompts.')
    p.add_argument('--prompt_csv_key_type', default='auto', choices=['auto','base_name','hdf5_stem','task_text','path','text'],
                   help='Which column to use as key for lookup (auto tries multiple).')
    p.add_argument('--prompt_csv_prompt_col', default='prompt', help='Name of the prompt column in CSV (default: prompt).')

    # Data
    p.add_argument('--image_key', default='base_camera')
    p.add_argument('--save_wrist', action='store_true')
    p.add_argument('--image_size', type=int, nargs=2, default=None)
    p.add_argument('--fps_stride', type=int, default=1)

    # Tracker
    p.add_argument('--track_mode', default='offline', choices=['offline','online'])
    p.add_argument('--grid_size', type=int, default=50)    # 50
    p.add_argument('--vo_points', type=int, default=100)  # 100
    p.add_argument(
        '--maniskill_filter_tracks',
        dest='maniskill_filter_tracks',
        action='store_true',
        default=True,
        help='Apply ManiSkill non-collapsing quality masks to vis / point_traj_valid_mask.',
    )
    p.add_argument(
        '--no_maniskill_filter_tracks',
        dest='maniskill_filter_tracks',
        action='store_false',
        help='Disable ManiSkill point quality masking.',
    )
    p.add_argument('--maniskill_filter_sor_k', type=int, default=8)
    p.add_argument('--maniskill_filter_sor_std_ratio', type=float, default=4.0)
    p.add_argument('--maniskill_filter_reproj_thresh_px', type=float, default=16.0)
    p.add_argument('--maniskill_filter_teleport_mad', type=float, default=8.0)
    p.add_argument('--maniskill_filter_min_keep_ratio', type=float, default=0.25)

    # Misc
    p.add_argument('--device', default='cuda')

    # Memory control / performance
    p.add_argument('--vggt_device', default=None, help='Device for VGGT (e.g., cuda:1). Default: follow --device.')
    p.add_argument('--vggt_amp', choices=['fp16','bf16','off'], default='fp16', help='AMP for VGGT forward.')
    p.add_argument('--vggt_chunk', type=int, default=48, help='Temporal chunk length for VGGT.')
    p.add_argument('--free_sam_after_mask', action='store_true', help='Free SAM2/GDINO right after first-frame mask.')

    # Misc
    p.add_argument('--save_visuals', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_demos', type=int, default=None, help='Optional cap for quick ManiSkill processing runs.')
    return p.parse_args()

# -------------------------------- Utils ------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _as_numpy_float32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _poses_to_4x4_numpy(x) -> np.ndarray:
    poses = _as_numpy_float32(x)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.shape[-2:] == (3, 4):
        bottom = np.zeros(poses.shape[:-2] + (1, 4), dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=-2)
    return poses.astype(np.float32)

def extract_task_name(hdf5_file: Path) -> str:
    return hdf5_file.stem.replace("_demo", "").replace("_", " ").strip()

# -------------------------- CSV prompt loader -------------------------------
def _to_lc(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None

def build_prompt_lookup_from_csv(csv_path: str,
                                 prompt_col_name: str = 'prompt') -> Dict[str, Dict[str,str]]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if prompt_col_name.lower() not in cols:
        raise ValueError(f"CSV must contain a '{prompt_col_name}' column, got: {list(df.columns)}")

    prompt_col = cols[prompt_col_name.lower()]
    spaces: Dict[str, Dict[str,str]] = {}

    def add_space(colkey: str, fn):
        colname = cols.get(colkey)
        if not colname: return
        space = {}
        for _, row in df.iterrows():
            key_raw = row.get(colname, None)
            if key_raw is None: continue
            key = fn(str(key_raw))
            pr  = row.get(prompt_col, None)
            if not isinstance(pr, str): continue
            space[_to_lc(key)] = pr.strip()
        if space:
            spaces[colkey] = space

    add_space('base_name', lambda s: s)
    add_space('hdf5_stem', lambda s: s)
    add_space('task_text', lambda s: s)
    add_space('text',      lambda s: s)

    # path space
    if 'path' in cols:
        space = {}
        colname = cols['path']
        for _, row in df.iterrows():
            pval = row.get(colname, None)
            pr   = row.get(prompt_col, None)
            if not isinstance(pval, str) or not isinstance(pr, str): continue
            path = Path(pval)
            candidates = {path.stem, path.name, path.with_suffix('').name}
            for cand in candidates:
                space[_to_lc(cand)] = pr.strip()
        if space:
            spaces['path'] = space

    return spaces

def lookup_prompt(spaces: Dict[str, Dict[str,str]],
                  key_type: str,
                  base_name: str, hdf5_stem: str, task_text: str) -> Optional[str]:
    cand_map = {
        'base_name': _to_lc(base_name),
        'hdf5_stem': _to_lc(hdf5_stem),
        'task_text': _to_lc(task_text),
        'path':      _to_lc(hdf5_stem),
        'text':      _to_lc(task_text),
    }

    def try_type(t):
        space = spaces.get(t, {})
        key = cand_map[t]
        if key and key in space:
            return space[key]
        return None

    if key_type != 'auto':
        return try_type(key_type)

    for t in ['base_name','hdf5_stem','path','task_text','text']:
        val = try_type(t); 
        if val is not None: return val
    return None

# ------------------------ Grounded-SAM2 helpers -----------------------------
@torch.inference_mode()
def init_gsam2(gdino_config: str, gdino_ckpt: str,
               sam2_config: str, sam2_ckpt: str, device: str):
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    config_dir = str(GSA2_ROOT / "sam2/configs")
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)

    if sam2_config.startswith(str(GSA2_ROOT)):
        sam2_config = os.path.relpath(sam2_config, config_dir)

    sam2_model = build_sam2(sam2_config, sam2_ckpt, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = gdino_load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_ckpt,
        device=device
    )
    return grounding_model, sam2_predictor

def add_pos_neg_points(input_boxes, h, w):
    boxes = np.asarray(input_boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None]
    if boxes.size == 0:
        return (
            np.zeros((0, 0, 2), dtype=np.float32),
            np.zeros((0, 0), dtype=np.int64),
        )

    coords_all = []
    labels_all = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = x1 + 0.5 * bw
        cy = y1 + 0.5 * bh

        # Use box-relative positive clicks. A fixed image-center click can lie
        # outside the detected gripper and make SAM return only a partial mask.
        pos = [
            (cx, cy),
            (cx - 0.20 * bw, cy),
            (cx + 0.20 * bw, cy),
            (cx, cy - 0.18 * bh),
        ]
        neg = [
            (x1, y1),
            (x1, y2),
            (x2, y1),
            (x2, y2),
            (cx, y2 - 0.08 * bh),
        ]
        pts = np.asarray(pos + neg, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
        coords_all.append(pts)
        labels_all.append(labels)

    return np.stack(coords_all, axis=0), np.stack(labels_all, axis=0)

@torch.inference_mode()
def run_gsam2_on_first_frame(grounding_model, sam2_predictor: SAM2ImagePredictor,
                             first_rgb: np.ndarray, prompt: str,
                             box_thresh: float, text_thresh: float,
                             multimask_output: bool,
                             tmp_img_path: Path,
                             device: str,
                             gripper_box_shift_y: float = 60.0,
                             gripper_box_delta_h: float = -80.0,
                             gripper_mask_keep_top_frac: float = 1.0):
    Image.fromarray(first_rgb).save(tmp_img_path)
    image_source, image = gdino_load_image(str(tmp_img_path))

    text = (prompt or '').strip().lower()
    if not text.endswith('.'):
        text += '.'

    use_cuda = (device.startswith('cuda') and torch.cuda.is_available())
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_cuda else torch.autocast(device_type='cpu', dtype=torch.bfloat16)

    sam2_predictor.set_image(image_source)
    boxes_cxcywh01, confidences, labels = gdino_predict(
        model=grounding_model, image=image, caption=text,
        box_threshold=box_thresh, text_threshold=text_thresh, device=device
    )

    h, w, _ = image_source.shape
    if boxes_cxcywh01 is None or boxes_cxcywh01.numel() == 0:
        return (torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0, 4), dtype=torch.float32),
                [])

    scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh01.dtype, device=boxes_cxcywh01.device)
    boxes_abs_cxcywh = boxes_cxcywh01 * scale

    # 5. Optional gripper-box heuristic. For ManiSkill gripper-only prompts this
    # should usually be disabled so the query mask covers the full gripper.
    def _is_gripper(name: str) -> bool:
        s = str(name).strip().lower().rstrip('.')
        return s == 'robotic gripper' or s == 'white robotic gripper' or s.startswith('robotic ')
    gripper_idx = [i for i, name in enumerate(labels) if _is_gripper(name)]

    if len(gripper_idx) > 0:
        gi = torch.as_tensor(gripper_idx, device=boxes_abs_cxcywh.device, dtype=torch.long)
        delta = torch.tensor([0.0, float(gripper_box_shift_y), 0.0, float(gripper_box_delta_h)],
                             device=boxes_abs_cxcywh.device,
                             dtype=boxes_abs_cxcywh.dtype)
        boxes_abs_cxcywh[gi] = boxes_abs_cxcywh[gi] + delta

    boxes_abs_cxcywh[:, 0] = boxes_abs_cxcywh[:, 0].clamp(0, w)
    boxes_abs_cxcywh[:, 1] = boxes_abs_cxcywh[:, 1].clamp(0, h)
    boxes_abs_cxcywh[:, 2] = boxes_abs_cxcywh[:, 2].clamp(1, w)
    boxes_abs_cxcywh[:, 3] = boxes_abs_cxcywh[:, 3].clamp(1, h)

    boxes_xyxy_pixels = box_convert(
        boxes=boxes_abs_cxcywh, in_fmt="cxcywh", out_fmt="xyxy"
    ).detach().cpu().numpy()

    point_coords_b, point_labels_b = add_pos_neg_points(boxes_xyxy_pixels, h, w)

    with ctx:
        if use_cuda and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        masks, scores, _ = sam2_predictor.predict(
            point_coords=point_coords_b,
            point_labels=point_labels_b,
            box=boxes_xyxy_pixels,
            multimask_output=multimask_output,
        )
    masks = torch.from_numpy(masks)
    if masks.dim() == 4:
        if multimask_output:
            scores_np = np.asarray(scores)
            best_per_box = scores_np.argmax(axis=1)
            masks_per_box = []
            for b, m_idx in enumerate(best_per_box):
                masks_per_box.append(masks[b, m_idx])
            masks = torch.stack(masks_per_box, dim=0)
        else:
            masks = masks[:, 0]

    keep_top_frac = float(gripper_mask_keep_top_frac)
    if gripper_idx and keep_top_frac < 0.999:
        keep_top_frac = max(0.05, min(1.0, keep_top_frac))
        for idx in gripper_idx:
            if idx >= masks.shape[0]:
                continue
            x1, y1, x2, y2 = boxes_xyxy_pixels[idx]
            y1, y2 = sorted((float(y1), float(y2)))
            cut_y = int(round(y1 + keep_top_frac * max(y2 - y1, 1.0)))
            cut_y = int(np.clip(cut_y, 0, h))
            masks[idx, cut_y:, :] = False

    boxes_cxcywh01_mod = boxes_abs_cxcywh / scale
    boxes_xywh01 = boxes_cxcywh01_mod.clone()
    boxes_xywh01[:, 0] -= boxes_cxcywh01_mod[:, 2] / 2.0
    boxes_xywh01[:, 1] -= boxes_cxcywh01_mod[:, 3] / 2.0
    boxes_xywh01 = boxes_xywh01.clamp(0, 1)

    phrases_all = [f"{name}({float(conf):.2f})"
                   for name, conf in zip(labels, confidences.detach().cpu().numpy())]

    conf_np = confidences.detach().cpu().numpy()
    best_box_idx = int(conf_np.argmax())

    masks = masks[best_box_idx:best_box_idx + 1]
    boxes_xywh01 = boxes_xywh01[best_box_idx:best_box_idx + 1]
    phrases = [phrases_all[best_box_idx]]

    return masks.bool(), boxes_xywh01.cpu(), phrases

@torch.inference_mode()
def extract_first_frame_save(video_t: torch.Tensor, out_path: Path) -> np.ndarray:
    first = video_t[0].permute(1,2,0).byte().cpu().numpy()
    Image.fromarray(first).save(out_path)
    return first


def visualize_like_spatrack2(out_dir: Path, video: torch.Tensor, tracks_2d: np.ndarray, visibility: np.ndarray):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    video = video.clone().detach()
    tracks = torch.from_numpy(tracks_2d)[...,:2] # torch.Size([84, 100, 2])
    vis    = torch.from_numpy(visibility)

    max_size = 336
    h, w = video.shape[2:]
    scale = min(max_size / h, max_size / w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        resize = T.Resize((new_h, new_w))
        video = resize(video)
        tracks = tracks.clone(); tracks[...,:2] = tracks[...,:2] * scale

    viser = Visualizer(save_dir=str(out_dir), grayscale=False, fps=10, pad_value=0, tracks_leave_trace=5)
    viser.visualize(video=video[None], tracks=tracks[None], visibility=vis[None], filename="test")

def module_device(obj, default=torch.device('cpu')):
    try:
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                return p.device
            for b in obj.buffers():
                return b.device
            return default
        if isinstance(obj, torch.Tensor):
            return obj.device
        if hasattr(obj, "model") and isinstance(obj.model, torch.nn.Module):
            return module_device(obj.model, default)
        if hasattr(obj, "module") and isinstance(obj.module, torch.nn.Module):
            return module_device(obj.module, default)
    except Exception:
        pass
    return default

# ------------------------------- I/O utils ---------------------------------
@torch.inference_mode()
def save_mask_products(out_dir: Path, masks: torch.Tensor, boxes_xywh01: torch.Tensor, phrases: list, first_rgb: np.ndarray, save_visuals: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    m = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
    m[torch.any(masks, dim=0)] = 255
    Image.fromarray(m.numpy()).save(out_dir / 'mask_binary.png')

    meta = [{'value': 0, 'label': 'background'}]
    for i, (phr, b) in enumerate(zip(phrases, boxes_xywh01)):
        label = phr.split('(')[0]
        score = float(phr.split('(')[1][:-1]) if '(' in phr else 0.0
        meta.append({'value': i+1, 'label': label, 'logit': score, 'box_xywh01': b.numpy().tolist()})
    with open(out_dir / 'mask.json', 'w') as f:
        json.dump(meta, f, indent=2)

    if save_visuals:
        overlay = first_rgb.copy()
        mask_np = m.numpy()
        overlay[mask_np > 0] = (0.7 * overlay[mask_np > 0] + 0.3 * np.array([0,255,0])).astype(np.uint8)
        cv2.imwrite(str(out_dir / 'viz_mask_overlay.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return m.numpy()

def save_first_frame_with_bboxes(out_dir: Path, first_rgb: np.ndarray, boxes_xywh01: torch.Tensor, phrases: list):
    if boxes_xywh01 is None or len(boxes_xywh01) == 0:
        return
    img = first_rgb.copy()
    h, w = img.shape[:2]

    boxes = boxes_xywh01.detach().cpu().numpy() if isinstance(boxes_xywh01, torch.Tensor) else np.asarray(boxes_xywh01)
    for i, b in enumerate(boxes):
        x, y, bw, bh = float(b[0]) * w, float(b[1]) * h, float(b[2]) * w, float(b[3]) * h
        x1 = int(round(np.clip(x,       0, w - 1)))
        y1 = int(round(np.clip(y,       0, h - 1)))
        x2 = int(round(np.clip(x + bw,  0, w - 1)))
        y2 = int(round(np.clip(y + bh,  0, h - 1)))

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = phrases[i] if i < len(phrases) else f"obj{i}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, ty), (x1 + tw + 4, ty + th + bl + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 2, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / 'first_frame_bbox.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ------------------------------- Time utils -------------------------------
def _pad_time(x: torch.Tensor, T_target: int, mode: str = 'edge') -> torch.Tensor:
    """Pad or truncate the time dimension to T_target.

    mode='ones' is useful for multiplicative confidence weights; mode='edge'
    repeats the final frame.
    """
    T = x.shape[0]
    if T == T_target:
        return x
    if T > T_target:
        return x[:T_target]
    pad_n = T_target - T
    if mode == 'ones':
        filler = torch.ones_like(x[:1]).expand(pad_n, *x.shape[1:])
    elif mode == 'zeros':
        filler = torch.zeros_like(x[:1]).expand(pad_n, *x.shape[1:])
    else:  # 'edge'
        filler = x[-1:].expand(pad_n, *x.shape[1:])
    return torch.cat([x, filler], dim=0)

def _canonize_vggt_chunk(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Canonicalize VGGT chunk outputs:
      poses_pred:  [t, 3, 4]
      intrs:       [t, 3, 3]
      points_map:  [t, H, W, C]  (THWC)
      unc_metric:  [t, 1, H, W]  (TCHW, one channel)
    """
    def squeeze_leading_one(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() >= 1 and x.size(0) == 1 else x

    def to_T1HW(x: torch.Tensor) -> torch.Tensor:
        # Normalize to [t, 1, H, W].
        if x.dim() == 5:          # [B,T,C,H,W]
            x = x.squeeze(0)      # -> [T,C,H,W]
        if x.dim() == 4:
                # [T,C,H,W] or [T,H,W,C]
            if x.shape[1] in (1,2,3,4) and x.shape[2] > 8 and x.shape[3] > 8:
                # [T,C,H,W]
                if x.shape[1] != 1:
                    x = x[:, :1]
                return x.contiguous()
            else:
                # Treat as [T,H,W,C].
                return x[..., :1].permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            # [T,H,W] or [C,H,W]
            if x.shape[0] in (1,2,3,4) and x.shape[1] > 8 and x.shape[2] > 8:
                # [C,H,W] -> [1,1,H,W]
                return x[:1].unsqueeze(0).contiguous()
            else:
                # [T,H,W] -> [T,1,H,W]
                return x.unsqueeze(1).contiguous()
        if x.dim() == 2:
            # [H,W] -> [1,1,H,W]
            return x.unsqueeze(0).unsqueeze(0).contiguous()
        raise RuntimeError(f"Unexpected tensor dim for unc_metric: {tuple(x.shape)}")

    out: Dict[str, torch.Tensor] = {}

    # poses_pred / intrs
    out["poses_pred"] = squeeze_leading_one(pred["poses_pred"]).contiguous()
    out["intrs"]      = squeeze_leading_one(pred["intrs"]).contiguous()
    
    # vggt features
    out["features"] = squeeze_leading_one(pred["features"]).contiguous()

    # points_map -> THWC
    pm = squeeze_leading_one(pred["points_map"]).contiguous()
    if pm.dim() == 4:
        if pm.shape[1] in (1,2,3,4) and pm.shape[2] > 8 and pm.shape[3] > 8:
            pm = pm.permute(0, 2, 3, 1).contiguous()  # TCHW -> THWC
    elif pm.dim() == 3:
            # [H,W,C] or [C,H,W]
        if pm.shape[0] in (1,2,3,4) and pm.shape[1] > 8 and pm.shape[2] > 8:
            pm = pm.permute(1, 2, 0).unsqueeze(0).contiguous()  # C H W -> 1 H W C
        else:
            pm = pm.unsqueeze(0).contiguous()  # H W C -> 1 H W C
    else:
        raise RuntimeError(f"Unexpected points_map shape: {tuple(pm.shape)}")
    out["points_map"] = pm

    # unc_metric -> [T,1,H,W]
    out["unc_metric"] = to_T1HW(pred["unc_metric"])

    return out

@torch.inference_mode()
def vggt_forward_chunked(vggt_front: VGGT4Track,
                         video_tensor: torch.Tensor,      # [T,3,H,W] on vggt_dev
                         vggt_dev: torch.device,
                         amp_mode: str = 'fp16',
                         chunk_len: int = 48,
                         overlap: int = 1) -> Dict[str, torch.Tensor]:
    """
    Run VGGT in temporal chunks with optional frame overlap.

    With overlap=1, each chunk includes the previous chunk's last frame and the
    overlap is dropped afterward so the output length stays exactly T.
    """
    assert video_tensor.dim() == 4 and video_tensor.shape[1] == 3, "video_tensor must be [T,3,H,W]"
    T_ = video_tensor.shape[0]
    outs_lists = {k: [] for k in ["poses_pred", "intrs", "points_map", "unc_metric", "features"]}
    amp_ctx = _get_amp_context(vggt_dev, amp_mode)

    s = 0
    while s < T_:
        e = min(T_, s + chunk_len)
        s_in = s if s == 0 else max(0, s - overlap)  # Include the previous context frame.
        clip = video_tensor[s_in:e].to(vggt_dev, non_blocking=True)  # [L(+overlap),3,H,W]
        need = clip.shape[0]
        cut  = 0 if s == 0 else overlap            # Drop the overlapping prefix after inference.

        with amp_ctx:
            pred_raw = vggt_front(clip[None] / 255.0)  # [1, L(+overlap), ...]
        pred = _canonize_vggt_chunk(pred_raw)

        # Pad branch outputs to the chunk length if needed, then drop overlap.
        for k in outs_lists.keys():
            x = pred[k]
            if x.shape[0] != need:
                pad_mode = 'ones' if k == 'unc_metric' else 'edge'
                x = _pad_time(x, need, mode=pad_mode)
            if cut:
                x = x[cut:]
            outs_lists[k].append(x.contiguous())

        s = e
        del pred_raw, pred, clip
        if str(vggt_dev).startswith('cuda'):
            torch.cuda.empty_cache()

    outs = {k: torch.cat(vs, dim=0) for k, vs in outs_lists.items()}
    # Align outputs to the full input sequence length.
    for k in outs.keys():
        outs[k] = outs[k][:T_].contiguous()
    return outs


def _get_amp_context(device: torch.device, amp_mode: str):
    use_cuda = (str(device).startswith('cuda') and torch.cuda.is_available())
    if not use_cuda or amp_mode == 'off':
        # cpu autocast for parity; dtype doesn't matter much here
        return torch.autocast(device_type='cpu', dtype=torch.bfloat16)
    if amp_mode == 'fp16':
        return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)

def tracker2img(items: np.ndarray):
    """Map tracker grid points to image coordinates."""
    tracker_size = 518
    h = 256
    w = 256
    scale = min(h / tracker_size, w / tracker_size)
    if scale < 1:
        items_scaled = items * scale
    return items_scaled

def _sample_points_from_mask(mask: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Sample diverse tracker-space query points from a binary mask."""
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    mask_bool = np.asarray(mask) > 0
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rng = np.random.default_rng(seed)
    replace = xs.size < count
    sel = rng.choice(xs.size, size=count, replace=replace)
    pts = np.stack([xs[sel], ys[sel]], axis=-1).astype(np.float32)
    pts += rng.uniform(-0.35, 0.35, size=pts.shape).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, mask_bool.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, mask_bool.shape[0] - 1)
    return pts

# project 3D to 2d pixels
def project_world_to_pixels(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """K[3,3], R[3,3], t[3], Xw[N,3] -> uv[N,2]."""
    Xc = (R @ Xw.T + t[:, None]).T           # N×3
    z = np.clip(Xc[:, 2:3], 1e-6, None)
    uvw = (K @ Xc.T).T                       # N×3
    return uvw[:, :2] / z

def _robust_upper(values: np.ndarray, mad_factor: float, min_margin: float = 1e-6) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("inf")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if sigma < min_margin:
        return med + min_margin
    return med + float(mad_factor) * sigma

def _first_frame_sor_keep(
    points0: np.ndarray,
    candidate_mask: np.ndarray,
    k: int,
    std_ratio: float,
) -> np.ndarray:
    """Return a conservative first-frame SOR mask without modifying points."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    keep = np.zeros_like(candidate_mask, dtype=bool)
    idx = np.flatnonzero(candidate_mask)
    if idx.size == 0:
        return keep
    if idx.size <= max(int(k), 2):
        keep[idx] = True
        return keep

    pts = np.asarray(points0[idx], dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    if not np.all(finite):
        idx = idx[finite]
        pts = pts[finite]
    if idx.size <= max(int(k), 2):
        keep[idx] = True
        return keep

    k_eff = min(max(int(k), 1), idx.size - 1)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    knn = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_knn = np.mean(knn, axis=1)

    med = np.median(mean_knn)
    mad = np.median(np.abs(mean_knn - med))
    robust_sigma = max(1.4826 * mad, 1e-8)
    thresh = med + float(std_ratio) * robust_sigma
    keep[idx[mean_knn <= thresh]] = True
    return keep

def _build_maniskill_noncollapsing_track_mask(
    point_traj: np.ndarray,
    track2d: np.ndarray,
    track2d_projected: np.ndarray,
    vis_pred: np.ndarray,
    tracker_mask: np.ndarray,
    query_xy_t0: np.ndarray,
    *,
    enabled: bool,
    sor_k: int,
    sor_std_ratio: float,
    reproj_thresh_px: float,
    teleport_mad: float,
    min_keep_ratio: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a visibility mask for ManiSkill gripper tracks without replacing any
    3D coordinates. This catches obvious background / off-mask / unstable
    tracks while preserving raw SpaTracker geometry for calibration.
    """
    point_traj = np.asarray(point_traj, dtype=np.float32)
    track2d = np.asarray(track2d, dtype=np.float32)[..., :2]
    track2d_projected = np.asarray(track2d_projected, dtype=np.float32)[..., :2]
    T_len, num_points = point_traj.shape[:2]

    vis2d = np.asarray(vis_pred, dtype=np.float32)
    if vis2d.ndim == 3 and vis2d.shape[-1] == 1:
        vis2d = vis2d[..., 0]
    vis2d = vis2d[:T_len, :num_points]
    base_vis = np.isfinite(vis2d) & (vis2d > 0.5)
    finite_track = np.isfinite(point_traj).all(axis=-1) & np.isfinite(track2d).all(axis=-1)

    if not enabled:
        valid = base_vis & finite_track
        return valid.astype(bool), {
            "enabled": False,
            "track_keep_ratio": 1.0,
            "valid_ratio": float(valid.mean()) if valid.size else 0.0,
            "fallback_used": False,
        }

    mask = np.asarray(tracker_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    mh, mw = mask.shape[:2]
    q = np.asarray(query_xy_t0, dtype=np.float32)[..., :2]
    qx = np.clip(np.round(q[:, 0]).astype(np.int64), 0, mw - 1)
    qy = np.clip(np.round(q[:, 1]).astype(np.int64), 0, mh - 1)
    initial_mask_keep = mask[qy, qx]

    finite_point = np.isfinite(point_traj).all(axis=-1)
    finite_track_all = finite_track.all(axis=0)
    candidate = initial_mask_keep & finite_point.all(axis=0) & finite_track_all

    sor_keep = _first_frame_sor_keep(
        point_traj[0],
        candidate_mask=candidate,
        k=int(sor_k),
        std_ratio=float(sor_std_ratio),
    )

    step_norm = np.linalg.norm(np.diff(point_traj, axis=0), axis=-1)
    max_step = np.nanmax(np.where(np.isfinite(step_norm), step_norm, np.nan), axis=0)
    if np.any(sor_keep & np.isfinite(max_step)):
        teleport_thresh = _robust_upper(max_step[sor_keep], mad_factor=float(teleport_mad), min_margin=1e-4)
    elif np.any(np.isfinite(max_step)):
        teleport_thresh = _robust_upper(max_step, mad_factor=float(teleport_mad), min_margin=1e-4)
    else:
        teleport_thresh = float("inf")
    teleport_keep = np.isfinite(max_step) & (max_step <= teleport_thresh)

    track_keep = sor_keep & teleport_keep
    fallback_used = False
    if track_keep.mean() < float(min_keep_ratio):
        # If SOR is too aggressive for a tiny gripper mask, fall back to the
        # mask + teleport checks rather than collapsing the trajectory.
        track_keep = candidate & teleport_keep
        fallback_used = True

    reproj_err = np.linalg.norm(track2d_projected[:T_len, :num_points] - track2d[:T_len, :num_points], axis=-1)
    reproj_ok = np.isfinite(reproj_err) & (reproj_err <= float(reproj_thresh_px))

    valid = base_vis & finite_track[:T_len, :num_points] & reproj_ok & track_keep[None, :]
    stats: Dict[str, Any] = {
        "enabled": True,
        "num_points": int(num_points),
        "initial_mask_keep_ratio": float(initial_mask_keep.mean()) if initial_mask_keep.size else 0.0,
        "sor_keep_ratio": float(sor_keep.mean()) if sor_keep.size else 0.0,
        "track_keep_ratio": float(track_keep.mean()) if track_keep.size else 0.0,
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "reproj_keep_ratio": float(reproj_ok.mean()) if reproj_ok.size else 0.0,
        "teleport_thresh": float(teleport_thresh),
        "teleport_keep_ratio": float(teleport_keep.mean()) if teleport_keep.size else 0.0,
        "fallback_used": bool(fallback_used),
        "mode": "mask_plus_sor_plus_teleport_plus_reprojection_no_replace",
    }
    return valid.astype(bool), stats
    
# --------------------------- Tracking / Flows -------------------
@torch.inference_mode()
def run_spatial_tracker(predictor: Predictor, vggt_front: VGGT4Track,
                        video_t: torch.Tensor, mask_binary: np.ndarray,
                        grid_size: int, device: str,
                        vggt_dev: Optional[torch.device] = None,
                        vggt_amp: str = 'fp16',
                        vggt_chunk: int = 48,
                        filter_cfg: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    video_t: [T, C, H, W] float32 on CPU (0~255)
    device:  for SpatialTracker/Predictor, e.g., 'cuda:0'
    vggt_dev: device for VGGT (e.g., torch.device('cuda:1'))
    """
    device = torch.device(device)
    vggt_dev = vggt_dev or next(vggt_front.parameters()).device

    # 1) Preprocess for VGGT on its device
    vggt_video = preprocess_image(video_t).to(vggt_dev, non_blocking=True)  # [T,3,H,W]
    preds = vggt_forward_chunked(
        vggt_front=vggt_front,
        video_tensor=vggt_video,
        vggt_dev=vggt_dev,
        amp_mode=vggt_amp,
        chunk_len=int(vggt_chunk),
    )

    # vggt_hidden  [T,Patch num, Dim]
    vggt_hidden = preds["features"].mean(1)  # [T, Dim]

    del vggt_video
    if str(vggt_dev).startswith('cuda'):
        torch.cuda.empty_cache()

    # 2) Move necessary outputs to predictor device
    extrs      = preds["poses_pred"].to(device).contiguous()
    intrs      = preds["intrs"].to(device).contiguous()
    points_map = preds["points_map"].to(device).contiguous()
    unc_conf   = preds["unc_metric"].to(device).contiguous()
    del preds
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    print("extrs", extrs.shape, "intrs", intrs.shape, "points_map", points_map.shape, "unc_conf", unc_conf.shape, "vggt_hidden", vggt_hidden.shape)

    # Keep predictor inputs separate from VGGT tensors to reduce peak memory.
    predictor_video = preprocess_image(video_t).to(device, non_blocking=True)  # [T,3,H,W] in [0,1]

    # 4) Build mask/grid
    if points_map.dim() == 4:   # [T, H, W, C]
        H, W = points_map.shape[1], points_map.shape[2]
    else:
        H, W = predictor_video.shape[2], predictor_video.shape[3]

    depth_tensor = points_map[..., 2] if points_map.shape[-1] == 3 else points_map[2]
    depths_np = _as_numpy_float32(depth_tensor)
    extrinsics_np = _poses_to_4x4_numpy(extrs)
    if unc_conf.dim() == 4:
        unc_metric = (unc_conf[:, 0] > 0.5).float()
    else:
        unc_metric = (unc_conf > 0.5).float()

    grid_pts = get_points_on_a_grid(grid_size, (H, W), device='cpu')
    
    mask = cv2.resize((mask_binary > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    g = grid_pts[0].long()
    keep = mask[g[...,1].numpy(), g[...,0].numpy()] > 0
    grid_pts = grid_pts[:, keep]

    target = int(predictor.spatrack.track_num)
    M = grid_pts.shape[1]
    if M == 0:
        sampled = _sample_points_from_mask(mask, target, seed=0)
        if sampled.shape[0] == 0:
            del extrs, intrs, points_map, unc_conf, predictor_video
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            return ({}, {})
        grid_pts = torch.from_numpy(sampled)[None].float()
        M = grid_pts.shape[1]

    if M > target:
        rng = np.random.default_rng(seed=0)
        sel = rng.choice(M, size=target, replace=False)
        grid_pts = grid_pts[:, sel]
    elif M < target:
        # Fill from mask pixels instead of duplicating existing grid points.
        # Duplicates collapse the 3D tracks; diverse queries keep a usable
        # fixed-size tensor while invalid points can still be masked by vis.
        supplement = _sample_points_from_mask(mask, target - M, seed=1)
        if supplement.shape[0] > 0:
            grid_pts = torch.cat([grid_pts, torch.from_numpy(supplement)[None].float()], dim=1)

    query_xyt = torch.cat([torch.zeros_like(grid_pts[..., :1]), grid_pts], dim=-1)[0].numpy()
    query_xy_t0 = grid_pts[0].cpu().numpy()

    use_cuda = (str(device).startswith('cuda') and torch.cuda.is_available())
    amp_ctx = _get_amp_context(device, 'bf16') if use_cuda else torch.autocast(device_type='cpu', dtype=torch.bfloat16)
    with amp_ctx:
        c2w_traj, intrs2, point_map, conf_depth, track3d_pred, track2d_pred, vis_pred, conf_pred, video_out = predictor.forward(
            predictor_video, depth=depth_tensor,
            intrs=intrs, extrs=extrs,
            queries=query_xyt,
            fps=1, full_point=False, iters_track=5,
            query_no_BA=True, fixed_cam=True, stage=1, unc_metric=unc_metric,
            support_frame=extrs.shape[0]-1, replace_ratio=0.2
        )
    # 5) Outputs. Keep raw SpaTracker/VGGT 3D points unchanged. ManiSkill
    # filtering below only changes visibility masks; it never copies/replaces
    # 3D coordinates, avoiding the point-collapse failure mode.
    point_traj = (
        torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[..., :3].cpu())
        + c2w_traj[:, :3, 3][:, None, :]
    )
    c2w_traj_np = c2w_traj.detach().float().cpu().numpy()
    point_traj = point_traj.cpu().numpy()
    # Keep the actual 2D tracks returned by SpaTracker. The projection is saved
    # only for diagnostics / consistency masking.
    track2d_projected = []
    for j in range(point_traj.shape[0]):
        Kj = intrs2[j].cpu().numpy() if intrs2.ndim == 3 else intrs2.cpu().numpy()
        c2w_j = c2w_traj_np[j]
        R_cw = c2w_j[:3, :3]
        t_cw = c2w_j[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        track2d_projected.append(project_world_to_pixels(Kj, R_wc, t_wc, point_traj[j]))
    track2d_projected = np.stack(track2d_projected, axis=0).astype(np.float32)
    track2d = track2d_pred.detach().float().cpu().numpy().astype(np.float32)

    vis_raw_np = vis_pred.cpu().numpy().astype(np.float32)
    filter_cfg = filter_cfg or {}
    valid_mask, filter_stats = _build_maniskill_noncollapsing_track_mask(
        point_traj=point_traj,
        track2d=track2d,
        track2d_projected=track2d_projected,
        vis_pred=vis_raw_np,
        tracker_mask=mask,
        query_xy_t0=query_xy_t0,
        enabled=bool(filter_cfg.get("enabled", True)),
        sor_k=int(filter_cfg.get("sor_k", 8)),
        sor_std_ratio=float(filter_cfg.get("sor_std_ratio", 4.0)),
        reproj_thresh_px=float(filter_cfg.get("reproj_thresh_px", 16.0)),
        teleport_mad=float(filter_cfg.get("teleport_mad", 8.0)),
        min_keep_ratio=float(filter_cfg.get("min_keep_ratio", 0.25)),
    )

    vis2d = vis_raw_np[..., 0] if vis_raw_np.ndim == 3 and vis_raw_np.shape[-1] == 1 else vis_raw_np
    vis_filtered = np.where(valid_mask, vis2d[: valid_mask.shape[0], : valid_mask.shape[1]], 0.0).astype(np.float32)
    vis_np = vis_filtered[..., None] if vis_raw_np.ndim == 3 and vis_raw_np.shape[-1] == 1 else vis_filtered

    p0_uv = track2d[0].copy()

    del extrs, intrs, points_map, unc_conf, depth_tensor, unc_metric, predictor_video
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    return (
        {'track2d': track2d, 'track2d_projected_from_point_traj': track2d_projected,
         'vis': vis_np, 'vis_spatracker': vis_raw_np, 'point_traj_valid_mask': valid_mask.astype(np.uint8),
         'point_traj': point_traj, 'point_traj_spatracker': point_traj,
         'point_traj_spatracker_raw': point_traj, 'grid_points_xy': grid_pts[0].cpu().numpy(),
         'frames_rgb': video_t.cpu().numpy(), "depths": depths_np, "extrinsics": extrinsics_np,
         "c2w": c2w_traj_np, "intrs2": intrs2.cpu().numpy(), "p0_uv": p0_uv,
         "query_xy_t0": query_xy_t0, "vggt_hidden": vggt_hidden.cpu().numpy()},
        {'video': video_out.detach().cpu(), 'point_filter_stats': filter_stats}
    )

def add_spatracker_point_traj_alias(tracks: Dict[str, np.ndarray], demo_grp: h5py.Group) -> None:
    if "point_traj" not in tracks:
        return
    tracks.setdefault("point_traj_spatracker", tracks["point_traj"])
    demo_grp.attrs["point_traj_active_source"] = "point_traj_spatracker"
    demo_grp.attrs["point_traj_mode"] = "spatracker"
    demo_grp.attrs["point_traj_modes_available"] = "point_traj_spatracker"
    demo_grp.attrs["point_traj_units"] = "spatracker_v2_relative"
    demo_grp.attrs["point_traj_coordinate_frame"] = "spatracker_v2_or_vggt"

# ---------------------------- Prompt Resolution ----------------------------
def _truncate_words(text: str, max_words: int) -> str:
    return ' '.join([w for w in text.replace('/', ' ').split() if w][:max_words])

def resolve_prompt(task_text: str, args,
                   csv_spaces: Optional[Dict[str, Dict[str,str]]],
                   hdf5_stem: str, base_name: str) -> str:
    if csv_spaces is not None:
        csv_prompt = lookup_prompt(csv_spaces, args.prompt_csv_key_type, base_name, hdf5_stem, task_text)
        if isinstance(csv_prompt, str) and csv_prompt.strip():
            val = csv_prompt.strip()
            if args.add_prefix:
                prefix = args.add_prefix.strip()
                val = f"{prefix if prefix.endswith('.') else prefix + '.'} {val}".strip()
            return val

    if args.prompt:
        val = args.prompt.strip()
        if args.add_prefix:
            prefix = args.add_prefix.strip()
            val = f"{prefix if prefix.endswith('.') else prefix + '.'} {val}".strip()
        return val

    if args.prompt_from_text and task_text:
        val = _truncate_words(task_text.strip(), args.prompt_max_words)
        if args.add_prefix:
            prefix = args.add_prefix.strip()
            val = f"{prefix if prefix.endswith('.') else prefix + '.'} {val}".strip()
        return val

    val = 'object'
    if args.add_prefix:
        prefix = args.add_prefix.strip()
        val = f"{prefix if prefix.endswith('.') else prefix + '.'} {val}".strip()
    return val

# --------------------------------- Main ------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    vggt_dev = torch.device(args.vggt_device) if args.vggt_device else torch.device(device)

    # CSV prompts (optional)
    csv_spaces = None
    if args.prompt_csv:
        if not os.path.exists(args.prompt_csv):
            raise FileNotFoundError(f"CSV file not found: {args.prompt_csv}")
        csv_spaces = build_prompt_lookup_from_csv(args.prompt_csv, prompt_col_name=args.prompt_csv_prompt_col)
        print(f"[CSV] Loaded prompt table from {args.prompt_csv}. Key spaces: {list(csv_spaces.keys())}")

    print("Initializing models...")
    grounding_model, sam2_predictor = init_gsam2(
        args.gdino_config, args.gdino_ckpt, args.sam2_config, args.sam2_ckpt, device
    )
    vggt_front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(vggt_dev)
    predictor = Predictor.from_pretrained(f"Yuxihenry/SpatialTrackerV2-{args.track_mode.capitalize()}")
    predictor.eval().to(device)
    predictor.spatrack.track_num = args.vo_points
    print("Models initialized.")

    # check devices
    print(
        "[Device-Check]",
        f"vggt_front={module_device(vggt_front)}",
        f"predictor={module_device(predictor)}",
        f"grounding_model={module_device(grounding_model)}",
        f"sam2_predictor={module_device(sam2_predictor)}",
    )

    from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

    def _slice_nested(x, sl):
        """Slice ManiSkill dict-of-arrays recursively."""
        if isinstance(x, dict):
            return {k: _slice_nested(v, sl) for k, v in x.items()}
        return x[sl]

    def _get_nested_strict(d, keys, ctx="root"):
        """Strict nested getter: missing key -> KeyError, wrong type -> TypeError."""
        cur = d
        path = ctx
        for k in keys:
            if not isinstance(cur, dict):
                raise TypeError(f"{path} must be dict, got {type(cur)}")
            if k not in cur:
                raise KeyError(f"Missing key '{k}' at {path}. Available keys={list(cur.keys())}")
            cur = cur[k]
            path = f"{path}.{k}"
        return cur

    def _pick_tracking_rgb_strict(obs_ep, camera_name: str):
        """
        STRICT: only use obs['sensor_data'][camera_name]['rgb'].
        Any mismatch -> raise (no fallback).
        """
        sensor_data = _get_nested_strict(obs_ep, ["sensor_data"], ctx="obs")
        cam = _get_nested_strict(sensor_data, [camera_name], ctx="obs.sensor_data")
        rgb = _get_nested_strict(cam, ["rgb"], ctx=f"obs.sensor_data.{camera_name}")
        if rgb is None:
            raise ValueError(f"obs.sensor_data.{camera_name}.rgb is None")
        return rgb  # expected shape (T,H,W,3)

    def _pick_wrist_rgb_strict(obs_ep):
        """
        STRICT (when args.save_wrist=True): require obs['sensor_data']['hand_camera']['rgb'].
        """
        sensor_data = _get_nested_strict(obs_ep, ["sensor_data"], ctx="obs")
        cam = _get_nested_strict(sensor_data, ["hand_camera"], ctx="obs.sensor_data")
        rgb = _get_nested_strict(cam, ["rgb"], ctx="obs.sensor_data.hand_camera")
        if rgb is None:
            raise ValueError("obs.sensor_data.hand_camera.rgb is None")
        return rgb

    total_demos_processed = 0
    total_demos_attempted = 0
    for input_dir_str in args.input_dirs:
        input_dir = Path(input_dir_str)
        if not input_dir.is_dir():
            print(f"Warning: Input directory not found, skipping: {input_dir}")
            continue

        # ManiSkill trajectory files are typically *.h5; keep *.hdf5 too
        hdf5_files = sorted(list(input_dir.glob("*.h5")) + list(input_dir.glob("*.hdf5")))
        print(f"\nFound {len(hdf5_files)} trajectory H5 files in {input_dir}.")

        for hdf5_file in tqdm(hdf5_files, desc=f"Processing files in {input_dir.name}"):
            if args.max_demos is not None and total_demos_attempted >= int(args.max_demos):
                break
            task_text = extract_task_name(hdf5_file)
            hdf5_stem = hdf5_file.stem

            # one input file -> one output dir + one aggregated tracks hdf5
            task_out_dir = Path(args.out_root) / hdf5_stem
            task_out_dir.mkdir(parents=True, exist_ok=True)
            out_h5_path = task_out_dir / f"{hdf5_stem}_tracks.hdf5"

            # ---- Load ManiSkill dataset ----
            ds = ManiSkillTrajectoryDataset(dataset_file=str(hdf5_file), load_count=-1, device=None)

            episodes = getattr(ds, "episodes", None)
            if episodes is None or len(episodes) == 0:
                raise RuntimeError(f"{hdf5_file} has no episodes.")

            # Build episode id list + choose test episode
            ep_ids = []
            for i, meta in enumerate(episodes):
                if not isinstance(meta, dict):
                    raise TypeError(f"ds.episodes[{i}] must be dict, got {type(meta)}")
                ep_ids.append(int(meta["episode_id"]))  # strict: require episode_id

            test_demo_id = random.choice(ep_ids)
            print(f"[Task {hdf5_stem}] {len(ep_ids)} episodes, test episode: {test_demo_id}")

            # ---- Open output hdf5 ----
            with h5py.File(out_h5_path, "w") as fout:
                fout.attrs["task_text"] = task_text
                fout.attrs["source_hdf5"] = str(hdf5_file)
                # In ManiSkill: treat image_key as *camera name* (e.g., base_camera / hand_camera)
                fout.attrs["image_key"] = args.image_key
                fout.attrs["test_demo_id"] = str(test_demo_id)

                data_grp = fout.create_group("data")

                stride = max(1, args.fps_stride)
                start = 0  # flat index into ds.obs/ds.actions arrays

                for i, meta in enumerate(episodes):
                    if args.max_demos is not None and total_demos_attempted >= int(args.max_demos):
                        break
                    T = int(meta["elapsed_steps"])
                    if T <= 0:
                        raise ValueError(f"Episode {i} has invalid elapsed_steps={T}")

                    ep_id = int(meta["episode_id"])
                    sl = slice(start, start + T, stride)

                    base_name = f"{hdf5_stem}_{ep_id}"
                    total_demos_attempted += 1

                    out_dir = task_out_dir / base_name
                    save_viz_for_this_demo = args.save_visuals
                    if save_viz_for_this_demo:
                        out_dir.mkdir(parents=True, exist_ok=True)

                    prompt = resolve_prompt(task_text, args, csv_spaces, hdf5_stem, base_name)
                    print(f"Processing episode: {base_name} | prompt: '{prompt}'")

                    demo_grp = data_grp.create_group(str(ep_id))
                    demo_grp.attrs["prompt"] = str(prompt) if prompt is not None else ""
                    demo_grp.attrs["has_tracks"] = False
                    demo_grp.attrs["used_camera"] = str(args.image_key)

                    if getattr(ds, "obs", None) is None:
                        raise RuntimeError(
                            f"{hdf5_file} has no obs (obs_mode might be 'none'). "
                            f"Tracking requires obs_mode with rgb."
                        )

                    obs_ep = _slice_nested(ds.obs, sl)

                    actions = np.asarray(ds.actions[sl])

                    qpos = _get_nested_strict(obs_ep, ["agent", "qpos"], ctx="obs")
                    robot_states = np.asarray(qpos)

                    wrist_frames = None
                    if args.save_wrist:
                        wrist_rgb = _pick_wrist_rgb_strict(obs_ep)         # (T,H,W,3)
                        wrist_rgb = np.asarray(wrist_rgb)
                        wrist_frames = np.transpose(wrist_rgb, (0, 3, 1, 2))  # (T,C,H,W)

                    demo_grp.create_dataset("robot_states", data=robot_states, compression="gzip")
                    demo_grp.create_dataset("actions", data=actions, compression="gzip")
                    if wrist_frames is not None:
                        demo_grp.create_dataset("wrist_frames", data=wrist_frames, compression="gzip")

                    video_frames = _pick_tracking_rgb_strict(obs_ep, camera_name=args.image_key)
                    video_frames = np.asarray(video_frames)  # (T,H,W,3) uint8
                    if video_frames.shape[0] == 0:
                        raise ValueError(f"{base_name}: video_frames has 0 frames")

                    if args.image_size:
                        w, h = args.image_size
                        video_frames = np.stack(
                            [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in video_frames],
                            axis=0,
                        )

                    video_t = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous().float()

                    # first frame path:
                    if save_viz_for_this_demo:
                        first_frame_path = out_dir / "first_frame.jpg"
                    else:
                        first_frame_path = task_out_dir / f"{base_name}_first_frame_tmp.jpg"

                    first_rgb = extract_first_frame_save(video_t, first_frame_path)

                    # ---- Grounded-SAM2 on first frame ----
                    if (grounding_model is None) or (sam2_predictor is None):
                        grounding_model, sam2_predictor = init_gsam2(
                            args.gdino_config, args.gdino_ckpt,
                            args.sam2_config, args.sam2_ckpt, device
                        )

                    masks, boxes, phrases = run_gsam2_on_first_frame(
                        grounding_model, sam2_predictor, first_rgb, prompt,
                        args.box_thresh, args.text_thresh, args.multimask_output,
                        first_frame_path, device,
                        gripper_box_shift_y=args.gripper_box_shift_y,
                        gripper_box_delta_h=args.gripper_box_delta_h,
                        gripper_mask_keep_top_frac=args.gripper_mask_keep_top_frac,
                    )

                    if not save_viz_for_this_demo:
                        try:
                            os.remove(first_frame_path)
                        except FileNotFoundError:
                            pass

                    if save_viz_for_this_demo and boxes is not None and len(boxes) > 0:
                        save_first_frame_with_bboxes(out_dir, first_rgb, boxes_xywh01=boxes, phrases=phrases)

                    # ---- Free SAM2/GDINO if requested ----
                    if args.free_sam_after_mask:
                        try:
                            del grounding_model
                            del sam2_predictor
                            grounding_model, sam2_predictor = None, None
                            if device.startswith("cuda"):
                                torch.cuda.empty_cache()
                            print("[Memory] Freed SAM2 & GroundingDINO from GPU.")
                        except Exception as e:
                            print("[Memory] Free SAM2/GDINO failed:", e)

                    if masks.shape[0] == 0:
                        print(f"  No masks from Grounded-SAM-2 for {base_name}. Skipping tracker.")
                        start += T
                        continue

                    # ---- binary mask ----
                    if save_viz_for_this_demo:
                        mask_binary = save_mask_products(
                            out_dir,
                            masks, boxes, phrases, first_rgb,
                            save_visuals=True,
                        )
                    else:
                        m = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
                        m[torch.any(masks, dim=0)] = 255
                        mask_binary = m.numpy()

                    # ---- SpatialTrackerV2 (VGGT chunked) ----
                    tracks, extra = run_spatial_tracker(
                        predictor, vggt_front, video_t, mask_binary, args.grid_size, device,
                        vggt_dev=vggt_dev, vggt_amp=args.vggt_amp, vggt_chunk=args.vggt_chunk,
                        filter_cfg={
                            "enabled": bool(args.maniskill_filter_tracks),
                            "sor_k": int(args.maniskill_filter_sor_k),
                            "sor_std_ratio": float(args.maniskill_filter_sor_std_ratio),
                            "reproj_thresh_px": float(args.maniskill_filter_reproj_thresh_px),
                            "teleport_mad": float(args.maniskill_filter_teleport_mad),
                            "min_keep_ratio": float(args.maniskill_filter_min_keep_ratio),
                        },
                    )

                    if tracks and "track2d" in tracks and tracks["track2d"].size > 0:
                        demo_grp.attrs["has_tracks"] = True
                        add_spatracker_point_traj_alias(tracks, demo_grp)
                        for stat_key, stat_val in extra.get("point_filter_stats", {}).items():
                            demo_grp.attrs[f"maniskill_point_filter_{stat_key}"] = stat_val

                        for key, arr in tracks.items():
                            if key.startswith(
                                (
                                    "point", "track", "vis", "grid", "frames",
                                    "intrs", "depth", "extr", "c2w", "p0_uv", "query_xy_t0", "vggt_hidden"
                                )
                            ):
                                demo_grp.create_dataset(key, data=arr, compression="gzip")

                        if save_viz_for_this_demo:
                            fr = first_rgb.copy()

                            tracker_size = extra["video"].shape[-1]
                            h, w = fr.shape[:2]
                            scale = min(h / tracker_size, w / tracker_size)
                            if scale < 1:
                                grid_points_scaled = tracks["grid_points_xy"] * scale
                                for (x, y) in grid_points_scaled:
                                    cv2.circle(fr, (int(x), int(y)), 2, (255, 0, 0), -1)

                            cv2.imwrite(
                                str(out_dir / "viz_tracks_firstframe.jpg"),
                                cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                            )

                            visualize_like_spatrack2(
                                out_dir,
                                video=extra["video"],
                                tracks_2d=tracks["track2d"],
                                visibility=tracks["vis"],
                            )
                            print(f"Saved test_video for task {hdf5_stem}, episode {ep_id} at {out_dir}")

                        total_demos_processed += 1
                    else:
                        print(f"  Skipping save for {base_name}: No points were successfully tracked.")

                    start += T
        if args.max_demos is not None and total_demos_attempted >= int(args.max_demos):
            break

    print(
        f"\nDone. Attempted {total_demos_attempted} demos, processed {total_demos_processed}. "
        f"Aggregated results are under: {args.out_root}"
    )


if __name__ == '__main__':
    main()
