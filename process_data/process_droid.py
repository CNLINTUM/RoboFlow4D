#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process DROID TFDS episodes into flow-tracking HDF5 files.

Two flow modes are supported:
  - gripper: use Grounded-DINO + SAM2 on the first frame, then track masked gripper points
  - object:  use the manipulated object mentioned in the instruction as the GSAM2 prompt
  - scene:   use SpatialTrackerV2's default scene-query sampling over the whole frame

Outputs are written as one HDF5 per episode under:
  out_root/<normalized_instruction>/episode_xxxxxx_tracks.hdf5

Each HDF5 uses the same nested structure expected by the existing training code:
  /data/<episode_id>/{frames_rgb, wrist_frames, robot_states, actions, point_traj, ...}
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys as _sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.ops import box_convert

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import tensorflow as tf
import tensorflow_datasets as tfds

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ---------------------- Optional local repo resolution ----------------------
WORK = REPO_ROOT
GSA2_ROOT = WORK / "Grounded-SAM-2"
SPA_ROOT = WORK / "SpaTrackerV2"

def _maybe_prepend_paths() -> None:
    paths_to_add = [str(REPO_ROOT)]
    if GSA2_ROOT.exists():
        paths_to_add.append(str(GSA2_ROOT))
    if SPA_ROOT.exists():
        paths_to_add.append(str(SPA_ROOT))
    for p in reversed(paths_to_add):
        if p not in _sys.path:
            _sys.path.insert(0, p)


_maybe_prepend_paths()


# -------------------------- External deps / models --------------------------
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.utils import get_points_on_a_grid

try:
    from utils.motion_filter import filter_points_moving_and_sor_firstframe

    _HAS_MOTION_FILTER = True
except Exception:
    _HAS_MOTION_FILTER = False


# =============================================================================
# Utilities
# =============================================================================
def overwrite_dataset(h5group, name: str, data, **create_kwargs):
    if name in h5group:
        del h5group[name]
    return h5group.create_dataset(name, data=data, **create_kwargs)


def module_device(m: torch.nn.Module) -> str:
    try:
        return str(next(m.parameters()).device)
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_amp_context(device: torch.device, amp_mode: str):
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if (not use_cuda) or amp_mode == "off":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    if amp_mode == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


def get_preprocess_geometry(
    raw_h: int,
    raw_w: int,
    mode: str = "crop",
    target_size: int = 518,
    keep_ratio: bool = False,
) -> Dict[str, float]:
    if mode not in {"crop", "pad"}:
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    if mode == "pad":
        if raw_w >= raw_h:
            resized_w = target_size
            resized_h = round(raw_h * (resized_w / raw_w) / 14) * 14
        else:
            resized_h = target_size
            resized_w = round(raw_w * (resized_h / raw_h) / 14) * 14
        pad_top = (target_size - resized_h) // 2
        pad_left = (target_size - resized_w) // 2
        crop_top = 0
        crop_left = 0
        out_h = target_size
        out_w = target_size
    else:
        resized_w = target_size
        resized_h = round(raw_h * (resized_w / raw_w) / 14) * 14
        pad_top = 0
        pad_left = 0
        crop_left = 0
        crop_top = 0
        out_h = resized_h
        out_w = resized_w
        if (not keep_ratio) and resized_h > target_size:
            crop_top = (resized_h - target_size) // 2
            out_h = target_size

    return {
        "scale_x": float(resized_w) / float(raw_w),
        "scale_y": float(resized_h) / float(raw_h),
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
        "crop_left": float(crop_left),
        "crop_top": float(crop_top),
        "out_h": float(out_h),
        "out_w": float(out_w),
    }


def map_xy_from_preprocessed_to_raw(xy: np.ndarray, geom: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32).copy()
    arr[..., 0] = (arr[..., 0] - geom["pad_left"] + geom["crop_left"]) / geom["scale_x"]
    arr[..., 1] = (arr[..., 1] - geom["pad_top"] + geom["crop_top"]) / geom["scale_y"]
    return arr


def map_query_xyt_from_preprocessed_to_raw(query_xyt: np.ndarray, geom: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(query_xyt, dtype=np.float32).copy()
    arr[..., 1:3] = map_xy_from_preprocessed_to_raw(arr[..., 1:3], geom)
    return arr


def map_intrinsics_from_preprocessed_to_raw(intrs: np.ndarray, geom: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(intrs, dtype=np.float32).copy()
    arr[..., 0, 2] = arr[..., 0, 2] - geom["pad_left"] + geom["crop_left"]
    arr[..., 1, 2] = arr[..., 1, 2] - geom["pad_top"] + geom["crop_top"]
    arr[..., 0, :] /= geom["scale_x"]
    arr[..., 1, :] /= geom["scale_y"]
    return arr


def map_depth_from_preprocessed_to_raw(
    depths: np.ndarray,
    geom: Dict[str, float],
    raw_h: int,
    raw_w: int,
) -> np.ndarray:
    arr = np.asarray(depths, dtype=np.float32)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError(f"Expected depth shape [T,H,W] or [H,W], got {arr.shape}")

    x_raw, y_raw = np.meshgrid(
        np.arange(raw_w, dtype=np.float32),
        np.arange(raw_h, dtype=np.float32),
    )
    map_x = x_raw * geom["scale_x"] + geom["pad_left"] - geom["crop_left"]
    map_y = y_raw * geom["scale_y"] + geom["pad_top"] - geom["crop_top"]
    valid = (
        (map_x >= 0)
        & (map_x <= arr.shape[2] - 1)
        & (map_y >= 0)
        & (map_y <= arr.shape[1] - 1)
    )

    mapped = np.empty((arr.shape[0], raw_h, raw_w), dtype=np.float32)
    for i in range(arr.shape[0]):
        mapped_i = cv2.remap(
            arr[i],
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mapped_i[~valid] = 0
        mapped[i] = mapped_i
    return mapped[0] if squeeze else mapped


def _truncate_words(text: str, max_words: int) -> str:
    toks = [w for w in re.split(r"\s+", text.strip()) if w]
    return " ".join(toks[:max_words])


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_str(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


def _repeat_by_stride(arr: np.ndarray, stride: int, T_full: int) -> np.ndarray:
    T_s = arr.shape[0]
    idx = np.arange(T_full) // stride
    idx = np.clip(idx, 0, T_s - 1)
    return arr[idx]


def _interp_by_stride(arr: np.ndarray, stride: int, T_full: int) -> np.ndarray:
    T_s = arr.shape[0]
    if T_s == 1:
        return np.repeat(arr, T_full, axis=0)

    t_src = np.arange(T_s, dtype=np.float32) * float(stride)
    t_tgt = np.arange(T_full, dtype=np.float32)
    orig_shape = arr.shape
    arr_flat = arr.reshape(T_s, -1)
    out_flat = np.empty((T_full, arr_flat.shape[1]), dtype=arr.dtype)
    for d in range(arr_flat.shape[1]):
        out_flat[:, d] = np.interp(t_tgt, t_src, arr_flat[:, d])
    return out_flat.reshape((T_full,) + orig_shape[1:])


def normalize_instr_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("_", " ").strip()
    s = re.sub(r"^(?:kitchen|living room|study)\s+scene\s*\d+\s+", "", s)
    s = re.sub(r"(?:\s+demo)?\s+tracks?\s*$", "", s)
    s = re.sub(r"(?:\s|_)+(?:demo(?:\s|_)+demo|\bdemo)\s*\d+\s*$", "", s)
    s = re.sub(r"(?:\s|_)+demo\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def instruction_to_dirname(instruction: str, max_len: int = 120) -> str:
    key = normalize_instr_key(instruction) or "unknown task"
    safe = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if not safe:
        safe = "unknown_task"
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
    head = safe[: max_len - len(digest) - 1].rstrip("_")
    return f"{head}_{digest}"


def _pick_instruction_from_step(step: Dict[str, Any]) -> str:
    for k in ["language_instruction", "language_instruction_2", "language_instruction_3"]:
        v = step.get(k, "")
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8", errors="ignore")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _normalize_instruction_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_object_phrase_from_instruction(instruction: str) -> str:
    text = _normalize_instruction_text(instruction)
    if not text:
        return ""

    verb_patterns = [
        r"turn off",
        r"turn on",
        r"pick up",
        r"pick",
        r"take out",
        r"take",
        r"put down",
        r"put",
        r"place",
        r"move",
        r"slide",
        r"use",
        r"open",
        r"close",
        r"grab",
        r"lift",
        r"pull",
        r"push",
    ]
    for pat in verb_patterns:
        m = re.match(rf"^\s*{pat}\s+(.*)$", text)
        if m:
            text = m.group(1).strip()
            break

    boundaries = [
        r"\bout of\b",
        r"\binto\b",
        r"\binside\b",
        r"\bonto\b",
        r"\bfrom\b",
        r"\bcloser to\b",
        r"\bnext to\b",
        r"\baway from\b",
        r"\bto\b",
        r"\bin\b",
        r"\bon\b",
        r"\boff\b",
        r"\bthen\b",
        r"\band then\b",
        r"\band finally\b",
        r"\band\b",
        r"\bfinally\b",
        r"\busing\b",
        r"\bwith\b",
    ]
    cut = len(text)
    for pat in boundaries:
        m = re.search(pat, text)
        if m:
            cut = min(cut, m.start())
    text = text[:cut].strip(" ,")

    text = re.sub(r"^(the|a|an|one|two|three|four)\s+", "", text).strip()
    return text


def resolve_prompt_for_episode(args, instruction: str, flow_mode: str) -> str:
    prompt_mode = getattr(args, "mask_prompt_mode", "auto")

    if args.prompt:
        text = args.prompt.strip()
    elif prompt_mode == "instruction_object" and instruction:
        text = extract_object_phrase_from_instruction(instruction)
    elif prompt_mode == "instruction_text" and instruction:
        text = _truncate_words(instruction, args.prompt_max_words)
    elif prompt_mode == "manual":
        text = ""
    elif flow_mode == "object" and instruction:
        text = extract_object_phrase_from_instruction(instruction)
    elif args.prompt_from_instruction and instruction:
        text = _truncate_words(instruction, args.prompt_max_words)
    else:
        text = ""

    prefix = (args.add_prefix or "").strip()
    if flow_mode == "object" and "gripper" in prefix.lower():
        prefix = ""
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."

    if not text:
        if flow_mode == "gripper":
            text = "robotic gripper"
        elif flow_mode == "object":
            text = "object"

    if prefix and text:
        return f"{prefix} {text}".strip()
    if prefix:
        return prefix
    if text:
        return text
    return "object"


def resolve_target_mode(args) -> str:
    return args.target_mode or args.flow_mode


def build_robot_state(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    joint = obs.get("joint_position", None)
    grip = obs.get("gripper_position", None)
    if joint is None or grip is None:
        return None
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    grip = np.asarray(grip, dtype=np.float32).reshape(-1)
    return np.concatenate([joint, grip], axis=0).astype(np.float32)


# =============================================================================
# Visualization helpers
# =============================================================================
def _color_for_id(i: int) -> Tuple[int, int, int]:
    h = (i * 37) % 180
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def overlay_mask_on_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = bgr.copy()
    m = (mask > 0).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
    if m.max() == 0:
        return out
    color = np.array([0, 255, 0], dtype=np.uint8)
    colored = np.zeros_like(out, dtype=np.uint8)
    colored[m > 0] = color
    return cv2.addWeighted(out, 1.0, colored, float(alpha), 0)


def draw_boxes_xywh01(bgr: np.ndarray, boxes_xywh01: np.ndarray, thickness: int = 2) -> np.ndarray:
    out = bgr.copy()
    if boxes_xywh01 is None or len(boxes_xywh01) == 0:
        return out
    H, W = out.shape[:2]
    for box in boxes_xywh01:
        x, y, w, h = [float(v) for v in box.tolist()]
        x1 = int(np.clip(x * W, 0, W - 1))
        y1 = int(np.clip(y * H, 0, H - 1))
        x2 = int(np.clip((x + w) * W, 0, W - 1))
        y2 = int(np.clip((y + h) * H, 0, H - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), thickness)
    return out


def safe_int_xy(uv: np.ndarray, W: int, H: int):
    if uv.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)
    x = uv[:, 0]
    y = uv[:, 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    xy = np.stack([x, y], axis=-1).astype(np.int32)
    return xy, valid


def render_tracks_video(
    frames_rgb: np.ndarray,
    track2d: np.ndarray,
    vis: Optional[np.ndarray],
    out_mp4: Path,
    fps: int = 10,
    stride: int = 2,
    max_points: int = 200,
    radius: int = 2,
    draw_trail: bool = False,
    trail_len: int = 15,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames_rgb.shape
    idx = np.arange(0, T, max(1, stride))

    N = track2d.shape[1]
    if max_points is not None and max_points > 0 and N > max_points:
        sel = np.linspace(0, N - 1, max_points).astype(np.int64)
    else:
        sel = np.arange(N, dtype=np.int64)

    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    colors = [_color_for_id(int(i)) for i in range(len(sel))]
    trails = [[] for _ in range(len(sel))] if draw_trail else None

    for t in idx:
        bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
        uv = track2d[t, sel]
        xy, valid = safe_int_xy(uv, W, H)
        if vis is not None:
            valid = valid & vis[t, sel].astype(bool)

        for j in range(len(sel)):
            if not valid[j]:
                continue
            x, y = int(xy[j, 0]), int(xy[j, 1])
            cv2.circle(bgr, (x, y), int(radius), colors[j], -1)
            if trails is not None:
                trails[j].append((x, y))
                if len(trails[j]) > trail_len:
                    trails[j] = trails[j][-trail_len:]
                if len(trails[j]) >= 2:
                    pts = np.asarray(trails[j], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bgr, [pts], isClosed=False, color=colors[j], thickness=1)

        cv2.putText(bgr, f"t={t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(bgr)

    vw.release()


# =============================================================================
# Episode materialization
# =============================================================================
def materialize_episode_arrays(
    ep: Dict[str, Any],
    image_key: str,
    save_wrist: bool = False,
    wrist_key: str = "wrist_image_left",
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    Dict[str, str],
    bool,
]:
    steps_obj = ep["steps"]

    if isinstance(steps_obj, tf.data.Dataset):
        step_iter = tfds.as_numpy(steps_obj)
    else:
        step_iter = iter(steps_obj)

    frames_ext_list = []
    frames_w_list = [] if save_wrist else None
    actions_list = []
    robot_states_list = []
    ee_states_list = []
    instruction = ""
    any_missing_ext = False
    any_missing_wrist = False
    any_missing_robot = False

    for st in step_iter:
        obs = st.get("observation", {})

        frame = obs.get(image_key, None)
        if frame is None:
            any_missing_ext = True
            frames_ext_list.append(None)
        else:
            frames_ext_list.append(frame)

        if save_wrist:
            wrist = obs.get(wrist_key, None)
            if wrist is None:
                any_missing_wrist = True
                frames_w_list.append(None)
            else:
                frames_w_list.append(wrist)

        action = st.get("action", None)
        actions_list.append(action)

        robot_state = build_robot_state(obs)
        if robot_state is None:
            any_missing_robot = True
            robot_states_list.append(None)
        else:
            robot_states_list.append(robot_state)

        ee_state = obs.get("cartesian_position", None)
        ee_states_list.append(None if ee_state is None else np.asarray(ee_state, dtype=np.float32))

        if not instruction:
            instruction = _pick_instruction_from_step(st)

    if len(actions_list) == 0 or any(a is None for a in actions_list):
        actions_np = np.zeros((0, 7), dtype=np.float32)
    else:
        actions_np = np.asarray(actions_list, dtype=np.float32)

    T = int(actions_np.shape[0])
    if T == 0:
        return None, None, actions_np, None, None, instruction, {}, True

    if any_missing_ext or any(f is None for f in frames_ext_list):
        frames_ext_np = None
    else:
        frames_ext_np = np.stack(frames_ext_list, axis=0).astype(np.uint8)

    frames_w_np = None
    if save_wrist:
        if any_missing_wrist or any(f is None for f in frames_w_list):
            frames_w_np = None
        else:
            frames_w_np = np.stack(frames_w_list, axis=0).astype(np.uint8)

    robot_states_np = None
    if not any_missing_robot and len(robot_states_list) == T and all(x is not None for x in robot_states_list):
        robot_states_np = np.stack(robot_states_list, axis=0).astype(np.float32)

    ee_states_np = None
    if len(ee_states_list) == T and all(x is not None for x in ee_states_list):
        ee_states_np = np.stack(ee_states_list, axis=0).astype(np.float32)

    meta: Dict[str, str] = {}
    em = ep.get("episode_metadata", {})
    if isinstance(em, dict):
        if "file_path" in em:
            meta["file_path"] = _as_str(em["file_path"])
        if "recording_folderpath" in em:
            meta["recording_folderpath"] = _as_str(em["recording_folderpath"])

    return frames_ext_np, frames_w_np, actions_np, robot_states_np, ee_states_np, instruction, meta, any_missing_ext


# =============================================================================
# Grounded-SAM2 helpers
# =============================================================================
@torch.inference_mode()
def init_gsam2(gdino_config: str, gdino_ckpt: str, sam2_config: str, sam2_ckpt: str, device: str):
    import hydra
    import inspect
    from hydra.core.global_hydra import GlobalHydra

    # Grounded-SAM-2 expects newer Hydra APIs; this keeps the local repo usable
    # in older TF environments where Hydra does not accept version_base.
    if "version_base" not in inspect.signature(hydra.initialize_config_module).parameters:
        _orig_init_module = hydra.initialize_config_module

        def _compat_init_module(config_module: str, version_base=None, job_name: str = "app"):
            return _orig_init_module(config_module=config_module, job_name=job_name)

        hydra.initialize_config_module = _compat_init_module

    if "version_base" not in inspect.signature(hydra.initialize_config_dir).parameters:
        _orig_init_dir = hydra.initialize_config_dir

        def _compat_init_dir(config_dir: str, version_base=None, job_name: str = "app"):
            return _orig_init_dir(config_dir=config_dir, job_name=job_name)

        hydra.initialize_config_dir = _compat_init_dir

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model as gdino_load_model

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
        device=device,
    )
    return grounding_model, sam2_predictor


def add_pos_neg_points(input_boxes: np.ndarray, h: int, w: int):
    pos_points = [(160, 32)]
    neg_points = [
        (174, 64),
        (160, 10),
        (160, 10),
        (input_boxes[0, 0], input_boxes[0, 1]),
        (input_boxes[0, 0], input_boxes[0, 3]),
        (input_boxes[0, 2], input_boxes[0, 1]),
        (input_boxes[0, 2], input_boxes[0, 3]),
    ]

    def clip_pts(pts):
        return [(float(max(0, min(w - 1, x))), float(max(0, min(h - 1, y)))) for x, y in pts]

    pos = clip_pts(pos_points)
    neg = clip_pts(neg_points)
    point_coords = np.asarray(pos + neg, dtype=np.float32)
    point_labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)

    B = input_boxes.shape[0]
    P = point_coords.shape[0]
    point_coords_b = np.broadcast_to(point_coords[None], (B, P, 2)).copy().astype(np.float32)
    point_labels_b = np.broadcast_to(point_labels[None], (B, P)).copy().astype(np.int64)
    return point_coords_b, point_labels_b


def _is_gripper_label(name: str) -> bool:
    s = str(name).strip().lower().rstrip(".")
    return (
        s == "robotic gripper"
        or s == "white robotic gripper"
        or s == "black robotic gripper"
        or s.startswith("robotic ")
    )


@torch.inference_mode()
def run_gsam2_on_first_frame(
    grounding_model,
    sam2_predictor,
    first_rgb: np.ndarray,
    prompt: str,
    box_thresh: float,
    text_thresh: float,
    multimask_output: bool,
    tmp_img_path: Path,
    device: str,
    gripper_box_delta_px: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    gripper_min_box_size_px: Tuple[float, float] = (0.0, 0.0),
):
    from grounding_dino.groundingdino.util.inference import load_image as gdino_load_image
    from grounding_dino.groundingdino.util.inference import predict as gdino_predict

    Image.fromarray(first_rgb).save(tmp_img_path)
    image_source, image = gdino_load_image(str(tmp_img_path))

    text = (prompt or "").strip().lower()
    if not text.endswith("."):
        text += "."

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_cuda else torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    sam2_predictor.set_image(image_source)
    boxes_cxcywh01, confidences, labels = gdino_predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=device,
    )

    h, w, _ = image_source.shape
    if boxes_cxcywh01 is None or boxes_cxcywh01.numel() == 0:
        return torch.zeros((0, h, w), dtype=torch.bool), np.zeros((0, 4), dtype=np.float32), []

    scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh01.dtype, device=boxes_cxcywh01.device)
    boxes_abs_cxcywh = boxes_cxcywh01 * scale

    gripper_idx = [i for i, name in enumerate(labels) if _is_gripper_label(name)]
    if gripper_idx:
        gi = torch.as_tensor(gripper_idx, device=boxes_abs_cxcywh.device, dtype=torch.long)
        delta = torch.tensor(
            gripper_box_delta_px,
            device=boxes_abs_cxcywh.device,
            dtype=boxes_abs_cxcywh.dtype,
        )
        boxes_abs_cxcywh[gi] = boxes_abs_cxcywh[gi] + delta
        min_w, min_h = gripper_min_box_size_px
        if min_w > 0:
            boxes_abs_cxcywh[gi, 2] = boxes_abs_cxcywh[gi, 2].clamp(min=float(min_w))
        if min_h > 0:
            boxes_abs_cxcywh[gi, 3] = boxes_abs_cxcywh[gi, 3].clamp(min=float(min_h))

    boxes_abs_cxcywh[:, 0] = boxes_abs_cxcywh[:, 0].clamp(0, w)
    boxes_abs_cxcywh[:, 1] = boxes_abs_cxcywh[:, 1].clamp(0, h)
    boxes_abs_cxcywh[:, 2] = boxes_abs_cxcywh[:, 2].clamp(1, w)
    boxes_abs_cxcywh[:, 3] = boxes_abs_cxcywh[:, 3].clamp(1, h)

    boxes_xyxy_pixels = box_convert(boxes=boxes_abs_cxcywh, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
    point_coords_b, point_labels_b = add_pos_neg_points(boxes_xyxy_pixels, h, w)

    with ctx:
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
            masks = torch.stack([masks[b, m_idx] for b, m_idx in enumerate(best_per_box)], dim=0)
        else:
            masks = masks[:, 0]

    boxes_cxcywh01_mod = boxes_abs_cxcywh / scale
    boxes_xywh01 = boxes_cxcywh01_mod.clone()
    boxes_xywh01[:, 0] -= boxes_cxcywh01_mod[:, 2] / 2.0
    boxes_xywh01[:, 1] -= boxes_cxcywh01_mod[:, 3] / 2.0
    boxes_xywh01 = boxes_xywh01.clamp(0, 1)

    conf_np = confidences.detach().cpu().numpy()
    phrases = [f"{name}({float(conf):.2f})" for name, conf in zip(labels, conf_np)]
    return masks.bool(), boxes_xywh01.cpu().numpy().astype(np.float32), phrases


def dilate_binary_mask(mask_binary: np.ndarray, radius_px: int) -> np.ndarray:
    radius_px = int(radius_px)
    if radius_px <= 0:
        return mask_binary
    mask = (np.asarray(mask_binary) > 0).astype(np.uint8)
    kernel_size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return (cv2.dilate(mask, kernel, iterations=1) * 255).astype(np.uint8)


# =============================================================================
# VGGT chunked forward
# =============================================================================
def _pad_time(x: torch.Tensor, T_target: int, mode: str = "edge") -> torch.Tensor:
    T = x.shape[0]
    if T == T_target:
        return x
    if T > T_target:
        return x[:T_target]
    pad_n = T_target - T
    if mode == "ones":
        filler = torch.ones_like(x[:1]).expand(pad_n, *x.shape[1:])
    elif mode == "zeros":
        filler = torch.zeros_like(x[:1]).expand(pad_n, *x.shape[1:])
    else:
        filler = x[-1:].expand(pad_n, *x.shape[1:])
    return torch.cat([x, filler], dim=0)


def _canonize_vggt_chunk(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def squeeze_leading_one(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() >= 1 and x.size(0) == 1 else x

    def to_T1HW(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.squeeze(0)
        if x.dim() == 4:
            if x.shape[1] in (1, 2, 3, 4) and x.shape[2] > 8 and x.shape[3] > 8:
                if x.shape[1] != 1:
                    x = x[:, :1]
                return x.contiguous()
            return x[..., :1].permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8:
                return x[:1].unsqueeze(0).contiguous()
            return x.unsqueeze(1).contiguous()
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0).contiguous()
        raise RuntimeError(f"Unexpected unc_metric shape: {tuple(x.shape)}")

    out: Dict[str, torch.Tensor] = {}
    out["poses_pred"] = squeeze_leading_one(pred["poses_pred"]).contiguous()
    out["intrs"] = squeeze_leading_one(pred["intrs"]).contiguous()
    out["features"] = squeeze_leading_one(pred["features"]).contiguous()

    pm = squeeze_leading_one(pred["points_map"]).contiguous()
    if pm.dim() == 4:
        if pm.shape[1] in (1, 2, 3, 4) and pm.shape[2] > 8 and pm.shape[3] > 8:
            pm = pm.permute(0, 2, 3, 1).contiguous()
    elif pm.dim() == 3:
        if pm.shape[0] in (1, 2, 3, 4) and pm.shape[1] > 8 and pm.shape[2] > 8:
            pm = pm.permute(1, 2, 0).unsqueeze(0).contiguous()
        else:
            pm = pm.unsqueeze(0).contiguous()
    else:
        raise RuntimeError(f"Unexpected points_map shape: {tuple(pm.shape)}")
    out["points_map"] = pm
    out["unc_metric"] = to_T1HW(pred["unc_metric"])
    return out


@torch.inference_mode()
def vggt_forward_chunked(
    vggt_front: VGGT4Track,
    video_tensor: torch.Tensor,
    vggt_dev: torch.device,
    amp_mode: str = "fp16",
    chunk_len: int = 48,
    overlap: int = 1,
) -> Dict[str, torch.Tensor]:
    assert video_tensor.dim() == 4 and video_tensor.shape[1] == 3
    T_ = video_tensor.shape[0]
    outs_lists = {k: [] for k in ["poses_pred", "intrs", "points_map", "unc_metric", "features"]}
    amp_ctx = _get_amp_context(vggt_dev, amp_mode)

    s = 0
    while s < T_:
        e = min(T_, s + chunk_len)
        s_in = s if s == 0 else max(0, s - overlap)
        clip = video_tensor[s_in:e].to(vggt_dev, non_blocking=True)
        need = clip.shape[0]
        cut = 0 if s == 0 else overlap

        with amp_ctx:
            pred_raw = vggt_front(clip[None] / 255.0)
        pred = _canonize_vggt_chunk(pred_raw)

        for k in outs_lists.keys():
            x = pred[k]
            if x.shape[0] != need:
                pad_mode = "ones" if k == "unc_metric" else "edge"
                x = _pad_time(x, need, mode=pad_mode)
            if cut:
                x = x[cut:]
            outs_lists[k].append(x.contiguous())

        s = e
        del pred_raw, pred, clip
        if str(vggt_dev).startswith("cuda"):
            torch.cuda.empty_cache()

    outs = {k: torch.cat(vs, dim=0) for k, vs in outs_lists.items()}
    for k in outs.keys():
        outs[k] = outs[k][:T_].contiguous()
    return outs


# =============================================================================
# Tracking helpers
# =============================================================================
def project_world_to_pixels(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    Xc = (R @ Xw.T + t[:, None]).T
    z = np.clip(Xc[:, 2:3], 1e-6, None)
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / z


def build_gripper_queries(
    mask_binary: np.ndarray,
    H: int,
    W: int,
    grid_size: int,
    target: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    grid_pts = get_points_on_a_grid(grid_size, (H, W), device="cpu")
    mask = cv2.resize((mask_binary > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    g = grid_pts[0].long()
    keep = mask[g[..., 1].numpy(), g[..., 0].numpy()] > 0
    grid_pts = grid_pts[:, keep]

    M = int(grid_pts.shape[1])
    if M == 0:
        return None, None, None

    if M > target:
        rng = np.random.default_rng(seed=0)
        sel = rng.choice(M, size=target, replace=False)
        grid_pts = grid_pts[:, sel]
    elif M < target:
        rep = np.random.choice(M, size=target - M, replace=True)
        grid_pts = torch.cat([grid_pts, grid_pts[:, rep]], dim=1)

    query_xyt = torch.cat([torch.zeros_like(grid_pts[..., :1]), grid_pts], dim=-1)[0].cpu().numpy().astype(np.float32)
    grid_xy = grid_pts[0].cpu().numpy().astype(np.float32)
    return query_xyt, grid_xy, grid_xy.copy()


def build_scene_queries(
    H: int,
    W: int,
    scene_grid_size: int,
) -> np.ndarray:
    grid_pts = get_points_on_a_grid(scene_grid_size, (H, W), device="cpu")
    scene_queries = torch.cat([torch.zeros_like(grid_pts[..., :1]), grid_pts], dim=-1)
    return scene_queries[0].cpu().numpy().astype(np.float32)


def compute_mean_vis(vis: np.ndarray, query_xyt: Optional[np.ndarray]) -> float:
    vis_np = np.asarray(vis, dtype=np.float32)
    if vis_np.ndim == 3 and vis_np.shape[-1] == 1:
        vis_np = vis_np[..., 0]
    if query_xyt is None or query_xyt.size == 0:
        return float(vis_np.mean())

    T, N = vis_np.shape
    q_t = np.clip(np.asarray(query_xyt[:, 0], dtype=np.int64), 0, T - 1)
    valid = np.arange(T)[:, None] >= q_t[None, :]
    if valid.any():
        return float(vis_np[valid].mean())
    return float(vis_np.mean())


@torch.inference_mode()
def run_spatial_tracker(
    predictor: Predictor,
    vggt_front: VGGT4Track,
    video_t: torch.Tensor,
    flow_mode: str,
    grid_size: int,
    scene_grid_size: int,
    device: str,
    mask_binary: Optional[np.ndarray] = None,
    vggt_dev: Optional[torch.device] = None,
    vggt_amp: str = "fp16",
    vggt_chunk: int = 48,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device_t = torch.device(device)
    vggt_dev = vggt_dev or next(vggt_front.parameters()).device
    raw_h, raw_w = int(video_t.shape[2]), int(video_t.shape[3])

    vggt_video = preprocess_image(video_t).to(vggt_dev, non_blocking=True)
    preds = vggt_forward_chunked(
        vggt_front=vggt_front,
        video_tensor=vggt_video,
        vggt_dev=vggt_dev,
        amp_mode=vggt_amp,
        chunk_len=int(vggt_chunk),
    )
    vggt_hidden = preds["features"].mean(1)

    del vggt_video
    if str(vggt_dev).startswith("cuda"):
        torch.cuda.empty_cache()

    extrs = preds["poses_pred"].to(device_t).contiguous()
    intrs = preds["intrs"].to(device_t).contiguous()
    points_map = preds["points_map"].to(device_t).contiguous()
    unc_conf = preds["unc_metric"].to(device_t).contiguous()
    del preds
    if str(device_t).startswith("cuda"):
        torch.cuda.empty_cache()

    predictor_video = preprocess_image(video_t).to(device_t, non_blocking=True)
    if points_map.dim() == 4:
        H, W = int(points_map.shape[1]), int(points_map.shape[2])
    else:
        H, W = int(predictor_video.shape[2]), int(predictor_video.shape[3])
    geom = get_preprocess_geometry(raw_h, raw_w, mode="crop", target_size=518, keep_ratio=False)
    if int(geom["out_h"]) != H or int(geom["out_w"]) != W:
        geom = {
            "scale_x": float(W) / float(raw_w),
            "scale_y": float(H) / float(raw_h),
            "pad_left": 0.0,
            "pad_top": 0.0,
            "crop_left": 0.0,
            "crop_top": 0.0,
            "out_h": float(H),
            "out_w": float(W),
        }

    depth_tensor = points_map[..., 2] if points_map.shape[-1] == 3 else points_map[:, 2]
    if unc_conf.dim() == 4:
        unc_metric_track = (unc_conf[:, 0] > 0.5).float()
    else:
        unc_metric_track = (unc_conf > 0.5).float()

    if flow_mode in {"gripper", "object"}:
        target = int(predictor.spatrack.track_num)
        if mask_binary is None:
            raise ValueError(f"mask_binary is required for {flow_mode} flow mode")
        query_xyt, grid_points_xy, query_xy_t0 = build_gripper_queries(mask_binary, H, W, grid_size, target)
        if query_xyt is None:
            del extrs, intrs, points_map, unc_conf, predictor_video
            if str(device_t).startswith("cuda"):
                torch.cuda.empty_cache()
            return {}, {}
    elif flow_mode == "scene":
        query_xyt = build_scene_queries(H, W, scene_grid_size)
        grid_points_xy = query_xyt[:, 1:3].copy()
        query_xy_t0 = grid_points_xy.copy()
    else:
        raise ValueError(f"Unsupported flow_mode: {flow_mode}")

    use_cuda = str(device_t).startswith("cuda") and torch.cuda.is_available()
    amp_ctx = _get_amp_context(device_t, "bf16") if use_cuda else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    with amp_ctx:
        c2w_traj, intrs2, point_map, conf_depth, track3d_pred, track2d_pred, vis_pred, conf_pred, video_out = predictor.forward(
            predictor_video,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=1,
            full_point=False,
            iters_track=5,
            query_no_BA=True,
            fixed_cam=True,
            stage=1,
            unc_metric=unc_metric_track,
            support_frame=extrs.shape[0] - 1,
            replace_ratio=0.2,
        )

    point_traj = torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[..., :3].cpu()) + c2w_traj[:, :3, 3][:, None, :]
    if _HAS_MOTION_FILTER and flow_mode in {"gripper", "object"}:
        point_traj, inlier_masks, moving_mask, motion_mag = filter_points_moving_and_sor_firstframe(
            point_traj,
            motion_thresh=0.25,
            k=64,
            std_ratio=2.5,
            replace_outliers=True,
        )

    point_traj_np = point_traj.cpu().numpy().astype(np.float32)
    c2w_traj_np = c2w_traj.detach().cpu().numpy().astype(np.float32)
    w2c_traj_np = torch.inverse(c2w_traj).detach().cpu().numpy().astype(np.float32)
    track2d_list = []
    for j in range(point_traj_np.shape[0]):
        Kj = intrs2[j].cpu().numpy() if intrs2.ndim == 3 else intrs2.cpu().numpy()
        Rj = w2c_traj_np[j, :3, :3]
        tj = w2c_traj_np[j, :3, 3]
        track2d_list.append(project_world_to_pixels(Kj, Rj, tj, point_traj_np[j]))
    track2d = np.stack(track2d_list, axis=0).astype(np.float32)
    track2d = map_xy_from_preprocessed_to_raw(track2d, geom)

    grid_points_xy_raw = map_xy_from_preprocessed_to_raw(grid_points_xy, geom)
    query_xyt_raw = map_query_xyt_from_preprocessed_to_raw(query_xyt, geom)
    query_xy_t0_raw = map_xy_from_preprocessed_to_raw(query_xy_t0, geom) if query_xy_t0 is not None else None
    if query_xy_t0_raw is not None and track2d.shape[0] > 0:
        track2d[0] = query_xy_t0_raw.astype(np.float32)
    intrs2_np = map_intrinsics_from_preprocessed_to_raw(intrs2.cpu().numpy().astype(np.float32), geom)
    depths_np = map_depth_from_preprocessed_to_raw(
        depth_tensor.detach().float().cpu().numpy().astype(np.float32),
        geom,
        raw_h,
        raw_w,
    )

    vis_np = vis_pred.cpu().numpy().astype(np.float32)
    if vis_np.ndim == 3 and vis_np.shape[-1] == 1:
        vis_np = vis_np[..., 0]

    tracks = {
        "track2d": track2d,
        "vis": vis_np,
        "point_traj": point_traj_np,
        "grid_points_xy": np.asarray(grid_points_xy_raw, dtype=np.float32),
        "query_xyt": np.asarray(query_xyt_raw, dtype=np.float32),
        "depths": depths_np.astype(np.float32),
        "extrinsics": c2w_traj_np,
        "intrs2": intrs2_np,
        "w2c": w2c_traj_np,
        "w2c_traj": w2c_traj_np,
        "c2w": c2w_traj_np,
        "c2w_traj": c2w_traj_np,
        "p0_uv": (query_xy_t0_raw if query_xy_t0_raw is not None else track2d[0]).copy().astype(np.float32),
        "vggt_hidden": vggt_hidden.cpu().numpy().astype(np.float32),
    }
    if query_xy_t0_raw is not None:
        tracks["query_xy_t0"] = np.asarray(query_xy_t0_raw, dtype=np.float32)

    del extrs, intrs, points_map, unc_conf, depth_tensor, unc_metric_track, predictor_video
    if str(device_t).startswith("cuda"):
        torch.cuda.empty_cache()

    return tracks, {"video": video_out.detach().cpu()}


def add_spatracker_point_traj_alias(tracks: Dict[str, np.ndarray], demo_grp: h5py.Group) -> None:
    if "point_traj" not in tracks:
        return
    tracks.setdefault("point_traj_spatracker", tracks["point_traj"])
    demo_grp.attrs["point_traj_active_source"] = "point_traj_spatracker"
    demo_grp.attrs["point_traj_mode"] = "spatracker"
    demo_grp.attrs["point_traj_modes_available"] = "point_traj_spatracker"
    demo_grp.attrs["point_traj_units"] = "spatracker_v2_relative"
    demo_grp.attrs["point_traj_coordinate_frame"] = "spatracker_v2_or_vggt"


# =============================================================================
# TFDS loader
# =============================================================================
def build_tfds_builder_from_dir(droid_dir: Path):
    droid_dir = droid_dir.expanduser().resolve()
    if (droid_dir / "dataset_info.json").exists():
        data_dir = droid_dir
    else:
        vers = sorted([p for p in droid_dir.iterdir() if p.is_dir() and re.match(r"^\d+\.\d+\.\d+$", p.name)])
        if not vers:
            raise FileNotFoundError(f"Cannot find TFDS directory with dataset_info.json under: {droid_dir}")
        data_dir = vers[-1]
    return tfds.builder_from_directory(str(data_dir)), data_dir


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Process DROID TFDS episodes into flow tracking HDF5 files.")
    p.add_argument("--droid_dir", required=True)
    p.add_argument("--out_root", required=True)

    p.add_argument("--split", default="train")
    p.add_argument("--max_episodes", type=int, default=-1)
    p.add_argument("--skip_existing", action="store_true")

    p.add_argument("--image_key", default="exterior_image_1_left")
    p.add_argument("--save_wrist", action="store_true")
    p.add_argument("--wrist_key", default="wrist_image_left")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Resize frames to (W H) before tracking.")
    p.add_argument("--fps_stride", type=int, default=1)

    p.add_argument("--target_mode", default=None, choices=["gripper", "object", "scene"], help="Preferred unified target selector.")
    p.add_argument("--flow_mode", default="gripper", choices=["gripper", "object", "scene"], help="Legacy alias of --target_mode.")
    p.add_argument("--grid_size", type=int, default=50, help="Grid size for masked gripper/object queries.")
    p.add_argument("--scene_grid_size", type=int, default=20, help="Regular scene-query grid size for scene mode.")
    p.add_argument("--track_mode", default="offline", choices=["offline", "online"])
    p.add_argument("--vo_points", type=int, default=400)
    p.add_argument("--min_vis_ratio", type=float, default=0.10)

    p.add_argument("--prompt", default=None)
    p.add_argument("--mask_prompt_mode", default="auto", choices=["auto", "manual", "instruction_text", "instruction_object"], help="How to resolve GSAM text prompts for gripper/object targets.")
    p.add_argument("--prompt_from_instruction", action="store_true", help="Legacy shortcut for --mask_prompt_mode instruction_text.")
    p.add_argument("--prompt_max_words", type=int, default=6)
    p.add_argument("--add_prefix", default=None, help="Optional prefix prepended to the resolved GSAM prompt.")

    p.add_argument("--gdino_config", default=str(GSA2_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    p.add_argument("--gdino_ckpt", default=str(GSA2_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth"))
    p.add_argument("--sam2_config", default=str(GSA2_ROOT / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"))
    p.add_argument("--sam2_ckpt", default=str(GSA2_ROOT / "checkpoints/sam2.1_hiera_large.pt"))
    p.add_argument("--box_thresh", type=float, default=0.35)
    p.add_argument("--text_thresh", type=float, default=0.30)
    p.add_argument("--multimask_output", action="store_true")
    p.add_argument("--free_sam_after_mask", action="store_true")
    p.add_argument(
        "--gripper_box_delta_px",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DW", "DH"),
        help="Optional cx/cy/w/h adjustment in pixels for gripper boxes before SAM2.",
    )
    p.add_argument(
        "--gripper_min_box_size_px",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("W", "H"),
        help="Minimum gripper box width/height in pixels before SAM2. Useful for tiny DROID gripper detections.",
    )
    p.add_argument(
        "--mask_dilate_px",
        type=int,
        default=0,
        help="Dilate the first-frame target mask by this pixel radius before query sampling.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vggt_device", default=None, help="e.g. cuda:1; if None, follow --device")
    p.add_argument("--vggt_amp", choices=["fp16", "bf16", "off"], default="fp16")
    p.add_argument("--vggt_chunk", type=int, default=48)

    p.add_argument("--save_first_mask", action="store_true")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--vis_dirname", default="_vis")
    p.add_argument("--vis_fps", type=int, default=10)
    p.add_argument("--vis_stride", type=int, default=2)
    p.add_argument("--vis_max_points", type=int, default=0, help="Maximum points to draw in MP4; <=0 means draw all.")
    p.add_argument("--vis_point_radius", type=int, default=2)
    p.add_argument("--vis_draw_trail", action="store_true")
    p.add_argument("--vis_trail_len", type=int, default=15)
    p.add_argument("--vis_mask_alpha", type=float, default=0.35)

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.prompt_from_instruction and args.mask_prompt_mode == "auto":
        args.mask_prompt_mode = "instruction_text"
    args.flow_mode = resolve_target_mode(args)
    set_seed(args.seed)

    out_root = Path(args.out_root).expanduser().resolve()
    _ensure_dir(out_root)

    device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    vggt_dev = torch.device(args.vggt_device) if args.vggt_device else torch.device(device)

    builder, real_dir = build_tfds_builder_from_dir(Path(args.droid_dir))
    ds = builder.as_dataset(split=args.split, shuffle_files=False)
    ds_np = tfds.as_numpy(ds)

    print("[TFDS] name:", builder.info.name, "version:", builder.info.version, "split:", args.split)
    print("[TFDS] data_dir:", real_dir)
    print("[Keys] image_key =", args.image_key, "| wrist_key =", args.wrist_key, "| save_wrist =", args.save_wrist)
    print("[Mode] flow_mode =", args.flow_mode)
    print("[Prompt] mask_prompt_mode =", args.mask_prompt_mode)

    grounding_model = None
    sam2_predictor = None
    if args.flow_mode in {"gripper", "object"}:
        print("Initializing GSAM2 models...")
        grounding_model, sam2_predictor = init_gsam2(
            args.gdino_config, args.gdino_ckpt, args.sam2_config, args.sam2_ckpt, device
        )

    print("Initializing SpatialTrackerV2 models...")
    vggt_front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(vggt_dev)
    predictor = Predictor.from_pretrained(f"Yuxihenry/SpatialTrackerV2-{args.track_mode.capitalize()}")
    predictor.eval().to(device)
    predictor.spatrack.track_num = int(args.vo_points)

    print(
        "[Device-Check]",
        "vggt_front=", module_device(vggt_front),
        "predictor=", module_device(predictor),
        "grounding_model=", module_device(grounding_model) if grounding_model is not None else "n/a",
        "sam2_predictor=", module_device(sam2_predictor) if sam2_predictor is not None else "n/a",
    )

    vis_root = out_root / args.vis_dirname
    if args.save_visuals:
        _ensure_dir(vis_root)

    stats = {
        "dataset": builder.info.name,
        "version": str(builder.info.version),
        "split": args.split,
        "flow_mode": args.flow_mode,
        "mask_prompt_mode": args.mask_prompt_mode,
        "image_key": args.image_key,
        "wrist_key": args.wrist_key if args.save_wrist else "",
        "fps_stride": int(args.fps_stride),
        "episodes_total": 0,
        "transitions_total": 0,
        "episodes_saved": 0,
        "transitions_saved": 0,
        "episodes_filtered": 0,
        "transitions_filtered": 0,
        "episodes_existing": 0,
        "transitions_existing": 0,
        "filtered_reasons": {
            "too_short": {"episodes": 0, "transitions": 0},
            "missing_instruction": {"episodes": 0, "transitions": 0},
            "missing_image_key": {"episodes": 0, "transitions": 0},
            "no_mask_first_frame": {"episodes": 0, "transitions": 0},
            "tracking_failed": {"episodes": 0, "transitions": 0},
            "low_visibility": {"episodes": 0, "transitions": 0},
            "exception": {"episodes": 0, "transitions": 0},
        },
    }

    def _count_transitions(T: int) -> int:
        return max(0, int(T) - 1)

    max_eps = args.max_episodes if args.max_episodes and args.max_episodes > 0 else None

    for ep_idx, ep in enumerate(tqdm(ds_np, desc="Processing DROID episodes")):
        if max_eps is not None and ep_idx >= max_eps:
            break

        (
            frames_ext,
            frames_wrist,
            actions,
            robot_states,
            ee_states,
            instruction,
            meta,
            any_missing_ext,
        ) = materialize_episode_arrays(
            ep,
            image_key=args.image_key,
            save_wrist=args.save_wrist,
            wrist_key=args.wrist_key,
        )

        stats["episodes_total"] += 1

        T_action = int(actions.shape[0])
        T_frame = int(frames_ext.shape[0]) if frames_ext is not None else T_action
        T_state = int(robot_states.shape[0]) if robot_states is not None else T_action
        T_raw = min(T_action, T_frame, T_state)
        tr_raw = _count_transitions(T_raw)
        stats["transitions_total"] += tr_raw

        if T_raw < 2:
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_raw
            stats["filtered_reasons"]["too_short"]["episodes"] += 1
            stats["filtered_reasons"]["too_short"]["transitions"] += tr_raw
            continue

        instruction = str(instruction or "").strip()
        if not instruction:
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_raw
            stats["filtered_reasons"]["missing_instruction"]["episodes"] += 1
            stats["filtered_reasons"]["missing_instruction"]["transitions"] += tr_raw
            continue

        if frames_ext is None or any_missing_ext:
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_raw
            stats["filtered_reasons"]["missing_image_key"]["episodes"] += 1
            stats["filtered_reasons"]["missing_image_key"]["transitions"] += tr_raw
            continue

        frames_ext = _to_uint8_rgb(frames_ext[:T_raw])
        actions = np.asarray(actions[:T_raw], dtype=np.float32)
        robot_states = np.asarray(robot_states[:T_raw], dtype=np.float32) if robot_states is not None else None
        ee_states = np.asarray(ee_states[:T_raw], dtype=np.float32) if ee_states is not None else None
        if frames_wrist is not None:
            frames_wrist = _to_uint8_rgb(frames_wrist[:T_raw])

        if args.image_size:
            w, h = args.image_size
            frames_ext = np.stack(
                [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames_ext],
                axis=0,
            )
            if frames_wrist is not None:
                frames_wrist = np.stack(
                    [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames_wrist],
                    axis=0,
                )

        T_full = int(frames_ext.shape[0])
        tr_full = _count_transitions(T_full)

        prompt = resolve_prompt_for_episode(args, instruction, args.flow_mode) if args.flow_mode in {"gripper", "object"} else ""
        ep_name = f"episode_{ep_idx:06d}"
        instr_dirname = instruction_to_dirname(instruction) if instruction else ep_name
        instr_out_dir = out_root / instr_dirname
        ep_out = instr_out_dir / f"{ep_name}_tracks.hdf5"
        if args.skip_existing and ep_out.exists():
            stats["episodes_existing"] += 1
            stats["transitions_existing"] += tr_full
            continue

        file_path = meta.get("file_path", "")
        recording_folderpath = meta.get("recording_folderpath", "")

        stride = max(1, int(args.fps_stride))
        video_frames = frames_ext[::stride]
        video_t = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous().float()

        mask_binary = None
        boxes_xywh01 = None
        phrases = None
        first_rgb = video_frames[0]

        if args.flow_mode in {"gripper", "object"}:
            if grounding_model is None or sam2_predictor is None:
                grounding_model, sam2_predictor = init_gsam2(
                    args.gdino_config, args.gdino_ckpt, args.sam2_config, args.sam2_ckpt, device
                )

            tmp_img_path = out_root / f"_tmp_first_frame_{os.getpid()}_{ep_idx}.jpg"
            try:
                masks, boxes_xywh01, phrases = run_gsam2_on_first_frame(
                    grounding_model,
                    sam2_predictor,
                    first_rgb=first_rgb,
                    prompt=prompt,
                    box_thresh=float(args.box_thresh),
                    text_thresh=float(args.text_thresh),
                    multimask_output=bool(args.multimask_output),
                    tmp_img_path=tmp_img_path,
                    device=device,
                    gripper_box_delta_px=tuple(args.gripper_box_delta_px),
                    gripper_min_box_size_px=tuple(args.gripper_min_box_size_px),
                )
            except Exception:
                stats["episodes_filtered"] += 1
                stats["transitions_filtered"] += tr_full
                stats["filtered_reasons"]["exception"]["episodes"] += 1
                stats["filtered_reasons"]["exception"]["transitions"] += tr_full
                try:
                    tmp_img_path.unlink()
                except Exception:
                    pass
                continue
            finally:
                try:
                    tmp_img_path.unlink()
                except Exception:
                    pass

            if args.free_sam_after_mask:
                del grounding_model
                del sam2_predictor
                grounding_model, sam2_predictor = None, None
                gc.collect()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

            if masks is None or int(masks.shape[0]) == 0:
                stats["episodes_filtered"] += 1
                stats["transitions_filtered"] += tr_full
                stats["filtered_reasons"]["no_mask_first_frame"]["episodes"] += 1
                stats["filtered_reasons"]["no_mask_first_frame"]["transitions"] += tr_full
                continue

            m = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
            m[torch.any(masks, dim=0)] = 255
            mask_binary = dilate_binary_mask(m.cpu().numpy(), int(args.mask_dilate_px))

        try:
            tracks, _extra = run_spatial_tracker(
                predictor,
                vggt_front,
                video_t=video_t,
                flow_mode=args.flow_mode,
                grid_size=int(args.grid_size),
                scene_grid_size=int(args.scene_grid_size),
                device=device,
                mask_binary=mask_binary,
                vggt_dev=vggt_dev,
                vggt_amp=args.vggt_amp,
                vggt_chunk=int(args.vggt_chunk),
            )
        except Exception:
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_full
            stats["filtered_reasons"]["exception"]["episodes"] += 1
            stats["filtered_reasons"]["exception"]["transitions"] += tr_full
            continue

        if not (tracks and "track2d" in tracks and isinstance(tracks["track2d"], np.ndarray) and tracks["track2d"].size > 0):
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_full
            stats["filtered_reasons"]["tracking_failed"]["episodes"] += 1
            stats["filtered_reasons"]["tracking_failed"]["transitions"] += tr_full
            continue

        mean_vis = compute_mean_vis(tracks["vis"], tracks.get("query_xyt"))
        if mean_vis < float(args.min_vis_ratio):
            stats["episodes_filtered"] += 1
            stats["transitions_filtered"] += tr_full
            stats["filtered_reasons"]["low_visibility"]["episodes"] += 1
            stats["filtered_reasons"]["low_visibility"]["transitions"] += tr_full
            continue

        if stride > 1:
            if "point_traj" in tracks:
                tracks["point_traj"] = _interp_by_stride(tracks["point_traj"], stride=stride, T_full=T_full).astype(np.float32)
            if "track2d" in tracks:
                tracks["track2d"] = _interp_by_stride(tracks["track2d"], stride=stride, T_full=T_full).astype(np.float32)
            if "vis" in tracks:
                tracks["vis"] = _repeat_by_stride(tracks["vis"], stride=stride, T_full=T_full).astype(np.float32)
            if "intrs2" in tracks:
                tracks["intrs2"] = _repeat_by_stride(tracks["intrs2"], stride=stride, T_full=T_full).astype(np.float32)
            if "depths" in tracks:
                tracks["depths"] = _repeat_by_stride(tracks["depths"], stride=stride, T_full=T_full).astype(np.float32)
            if "extrinsics" in tracks:
                tracks["extrinsics"] = _repeat_by_stride(tracks["extrinsics"], stride=stride, T_full=T_full).astype(np.float32)
            if "w2c" in tracks:
                tracks["w2c"] = _repeat_by_stride(tracks["w2c"], stride=stride, T_full=T_full).astype(np.float32)
            if "w2c_traj" in tracks:
                tracks["w2c_traj"] = _repeat_by_stride(tracks["w2c_traj"], stride=stride, T_full=T_full).astype(np.float32)
            if "c2w" in tracks:
                tracks["c2w"] = _repeat_by_stride(tracks["c2w"], stride=stride, T_full=T_full).astype(np.float32)
            if "c2w_traj" in tracks:
                tracks["c2w_traj"] = _repeat_by_stride(tracks["c2w_traj"], stride=stride, T_full=T_full).astype(np.float32)
            if "vggt_hidden" in tracks:
                tracks["vggt_hidden"] = _repeat_by_stride(tracks["vggt_hidden"], stride=stride, T_full=T_full).astype(np.float32)
            if "query_xyt" in tracks:
                tracks["query_xyt"] = tracks["query_xyt"].copy().astype(np.float32)
                tracks["query_xyt"][:, 0] *= float(stride)

        if "track2d" in tracks:
            tracks["p0_uv"] = tracks["track2d"][0].copy().astype(np.float32)

        if args.save_visuals:
            vis_dir = vis_root / instr_dirname
            _ensure_dir(vis_dir)
            try:
                first_bgr = cv2.cvtColor(frames_ext[0], cv2.COLOR_RGB2BGR)
                if mask_binary is not None:
                    first_bgr = overlay_mask_on_bgr(first_bgr, mask_binary, alpha=float(args.vis_mask_alpha))
                if boxes_xywh01 is not None and len(boxes_xywh01) > 0:
                    first_bgr = draw_boxes_xywh01(first_bgr, np.asarray(boxes_xywh01, dtype=np.float32))
                cv2.imwrite(str(vis_dir / f"{ep_name}_first.png"), first_bgr)

                render_tracks_video(
                    frames_rgb=frames_ext,
                    track2d=tracks["track2d"],
                    vis=tracks.get("vis"),
                    out_mp4=vis_dir / f"{ep_name}_tracks.mp4",
                    fps=int(args.vis_fps),
                    stride=int(args.vis_stride),
                    max_points=int(args.vis_max_points),
                    radius=int(args.vis_point_radius),
                    draw_trail=bool(args.vis_draw_trail),
                    trail_len=int(args.vis_trail_len),
                )

                meta_json = {
                    "episode_index": int(ep_idx),
                    "episode_id": ep_name,
                    "instruction": instruction,
                    "prompt": prompt,
                    "flow_mode": args.flow_mode,
                    "mask_prompt_mode": args.mask_prompt_mode,
                    "T": int(T_full),
                    "mean_vis": float(mean_vis),
                    "file_path": file_path,
                    "recording_folderpath": recording_folderpath,
                }
                with open(vis_dir / f"{ep_name}_meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta_json, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        _ensure_dir(instr_out_dir)
        with h5py.File(ep_out, "w") as fout:
            fout.attrs["dataset"] = builder.info.name
            fout.attrs["version"] = str(builder.info.version)
            fout.attrs["split"] = args.split
            fout.attrs["episode_index"] = int(ep_idx)
            fout.attrs["episode_id"] = ep_name
            fout.attrs["instruction"] = instruction
            fout.attrs["prompt"] = prompt
            fout.attrs["flow_mode"] = args.flow_mode
            fout.attrs["mask_prompt_mode"] = args.mask_prompt_mode
            fout.attrs["file_path"] = file_path
            fout.attrs["recording_folderpath"] = recording_folderpath
            fout.attrs["mean_vis"] = float(mean_vis)

            data_grp = fout.create_group("data")
            demo_grp = data_grp.create_group(ep_name)
            demo_grp.attrs["instruction"] = instruction
            demo_grp.attrs["prompt"] = prompt
            demo_grp.attrs["flow_mode"] = args.flow_mode
            demo_grp.attrs["mask_prompt_mode"] = args.mask_prompt_mode
            demo_grp.attrs["has_tracks"] = True
            add_spatracker_point_traj_alias(tracks, demo_grp)

            frames_rgb_chw = np.transpose(frames_ext, (0, 3, 1, 2)).astype(np.float32)
            overwrite_dataset(demo_grp, "frames_rgb", frames_rgb_chw, compression="gzip")
            overwrite_dataset(demo_grp, "actions", np.asarray(actions, dtype=np.float32), compression="gzip")

            if robot_states is not None:
                overwrite_dataset(demo_grp, "robot_states", np.asarray(robot_states, dtype=np.float32), compression="gzip")
            if ee_states is not None:
                overwrite_dataset(demo_grp, "ee_states", np.asarray(ee_states, dtype=np.float32), compression="gzip")
            if args.save_wrist and frames_wrist is not None:
                wrist_chw = np.transpose(frames_wrist, (0, 3, 1, 2)).astype(np.float32)
                overwrite_dataset(demo_grp, "wrist_frames", wrist_chw, compression="gzip")

            for key in ["track2d", "vis", "point_traj", "point_traj_spatracker", "grid_points_xy", "depths", "extrinsics", "intrs2", "w2c", "w2c_traj", "c2w", "c2w_traj", "p0_uv", "query_xy_t0", "query_xyt", "vggt_hidden"]:
                if key in tracks:
                    overwrite_dataset(demo_grp, key, np.asarray(tracks[key]), compression="gzip")

            if args.save_first_mask and mask_binary is not None:
                overwrite_dataset(demo_grp, "mask_first_frame", mask_binary.astype(np.uint8), compression="gzip")
            if boxes_xywh01 is not None and len(boxes_xywh01) > 0:
                overwrite_dataset(demo_grp, "first_frame_boxes_xywh01", np.asarray(boxes_xywh01, dtype=np.float32), compression="gzip")
            if phrases:
                dt = h5py.string_dtype(encoding="utf-8")
                overwrite_dataset(demo_grp, "first_frame_phrases", np.asarray([str(x) for x in phrases], dtype=object), dtype=dt)

        stats["episodes_saved"] += 1
        stats["transitions_saved"] += tr_full

    stats["episodes_filtered"] = int(stats["episodes_total"] - stats["episodes_saved"] - stats["episodes_existing"])
    stats["transitions_filtered"] = int(stats["transitions_total"] - stats["transitions_saved"] - stats["transitions_existing"])

    stats_path = out_root / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n========== DONE ==========")
    print("Saved episodes:", stats["episodes_saved"], "/", stats["episodes_total"])
    print("Existing episodes skipped:", stats["episodes_existing"])
    print("Existing transitions skipped:", stats["transitions_existing"])
    print("Saved transitions:", stats["transitions_saved"], "/", stats["transitions_total"])
    print("Filtered episodes:", stats["episodes_filtered"])
    print("Filtered transitions:", stats["transitions_filtered"])
    print("Filtered reasons:")
    for reason, cnts in stats["filtered_reasons"].items():
        if cnts["episodes"] > 0 or cnts["transitions"] > 0:
            print(f"  - {reason}: episodes={cnts['episodes']}, transitions={cnts['transitions']}")
    print("Stats written to:", stats_path)
    if args.save_visuals:
        print("Visualizations in:", vis_root)


if __name__ == "__main__":
    main()
