import copy
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import h5py
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SiglipProcessor
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
import wandb
wandb.util.working_set = lambda: ()  

_ACTION_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_ACTION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACTION_POLICY_ROOT))

from model.action_heads import Action_policy

processor = None

def get_text_processor():
    """Lazy-load in the main process; DataLoader workers should not import tokenizer state."""
    global processor
    if processor is None:
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast=True)
    return processor

def enable_sdpa():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception:
        pass

def normalize_instr_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("_", " ").strip()

    # Remove LIBERO scene prefixes and generated track/demo suffixes.
    s = re.sub(r"^(?:kitchen|living room|study)\s+scene\s*\d+\s+", "", s)
    s = re.sub(r"(?:\s+demo)?\s+tracks?\s*$", "", s)
    s = re.sub(r"(?:\s|_)+(?:demo(?:\s|_)+demo|\bdemo)\s*\d+\s*$", "", s)
    s = re.sub(r"(?:\s|_)+demo\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def debounce_gripper_changes(gb: np.ndarray, debounce: int = 3) -> List[int]:
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

def segment_gripper_state(actions: Optional[np.ndarray], t_len: int, debounce: int = 3) -> List[Tuple[int, int]]:
    t_len = int(t_len)
    if t_len <= 0:
        return []
    if actions is None or len(actions) == 0:
        return [(0, t_len - 1)]

    a = np.asarray(actions)
    g = a[:, -1] if a.ndim == 2 else a.reshape(-1)
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
    idxs = np.asarray(idxs, dtype=np.int64)
    idxs = np.clip(idxs, 0, int(ds.shape[0]) - 1)
    uniq, inv = np.unique(idxs, return_inverse=True)
    return np.asarray(ds[uniq], dtype=dtype)[inv]

def load_segment_keyframe_targets(grp: h5py.Group, s: int, k_steps: int, point_key: str = "point_traj") -> np.ndarray:
    point_ds = grp[point_key]
    t_len = int(point_ds.shape[0])
    actions = np.asarray(grp["actions"][:]) if "actions" in grp else None
    segments = segment_gripper_state(actions, t_len, debounce=3)
    seg_end = find_segment_end(segments, int(s), t_len)
    key_idxs = resample_on_traj(int(s), seg_end, int(k_steps))
    return h5_take_time(point_ds, key_idxs, dtype=np.float32)

def target_cache_key(k_steps: int) -> str:
    return f"target_point_traj_k{int(k_steps)}"

def transform_query_points(q: np.ndarray, train: bool, H: int = 256, W: int = 256) -> np.ndarray:
    if train:
        return aug_query_points(q, H=H, W=W)
    q = np.asarray(q, dtype=np.float32).copy() * (256.0 / 518.0)
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q

def _image_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Store image tensors compactly while preserving [0,1] and [0,255] sources."""
    if arr.dtype == np.uint8:
        return arr
    x = np.asarray(arr)
    scale = 255.0 if float(np.nanmax(x)) <= 1.5 else 1.0
    return np.clip(x * scale, 0, 255).round().astype(np.uint8)

def _dataset_chunks(name: str, shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if len(shape) == 0:
        return None
    if name in ("frames_rgb", "wrist_frames") and len(shape) == 4:
        return (1,) + tuple(shape[1:])
    if name.startswith("target_point_traj_k") and len(shape) == 4:
        return (1,) + tuple(shape[1:])
    if name.startswith("pre_point_traj") and len(shape) == 4:
        return (1,) + tuple(shape[1:])
    if name.startswith("point_traj") and len(shape) == 3:
        return (1,) + tuple(shape[1:])
    if len(shape) == 2:
        return (min(int(shape[0]), 1024), int(shape[1]))
    return tuple(max(1, int(x)) for x in shape)

def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        try:
            dst.attrs[key] = value
        except TypeError:
            dst.attrs[key] = str(value)

def _create_cached_dataset(grp: h5py.Group, name: str, data: np.ndarray, compression: Optional[str]):
    chunks = _dataset_chunks(name, tuple(data.shape))
    kwargs = {}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if compression is not None:
        kwargs["compression"] = compression
    grp.create_dataset(name, data=data, **kwargs)

def compute_target_cache(point_traj: np.ndarray, actions: Optional[np.ndarray], k_steps: int) -> np.ndarray:
    point_traj = np.asarray(point_traj, dtype=np.float32)
    t_len = int(point_traj.shape[0])
    max_s = (t_len - 1) - int(k_steps)
    out_len = max(0, max_s + 1)
    out = np.zeros((out_len, int(k_steps), point_traj.shape[1], point_traj.shape[2]), dtype=np.float32)
    if out_len == 0:
        return out
    segments = segment_gripper_state(actions, t_len, debounce=3)
    for s in range(out_len):
        seg_end = find_segment_end(segments, s, t_len)
        key_idxs = resample_on_traj(s, seg_end, int(k_steps))
        out[s] = point_traj[key_idxs]
    return out

def build_policy_training_cache(
    flow_root: str,
    cache_root: str,
    k_steps: int,
    compression: Optional[str] = "lzf",
    overwrite: bool = False,
    limit_files: Optional[int] = None,
) -> None:
    """Rewrite tracks HDF5 files into a training-friendly layout.

    The cache mirrors the source tree but stores per-frame chunks, uint8 images,
    and optional precomputed target keyframes for flow-conditioned training.
    """
    src_root = Path(flow_root)
    dst_root = Path(cache_root)
    h5_files = sorted(src_root.rglob("*_tracks.hdf5"))
    if limit_files is not None:
        h5_files = h5_files[: int(limit_files)]
    if not h5_files:
        raise RuntimeError(f"No *_tracks.hdf5 files found under {src_root}")

    compression = None if compression in (None, "none", "None") else compression
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE] Building policy cache: {src_root} -> {dst_root} ({len(h5_files)} files)")

    keys_to_copy = (
        "frames_rgb",
        "wrist_frames",
        "pre_point_traj",
        "point_traj",
        "actions",
        "robot_states",
        "query_xy_t0",
    )

    for file_idx, src_path in enumerate(h5_files, start=1):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        if dst_path.exists() and not overwrite:
            print(f"[CACHE] [{file_idx}/{len(h5_files)}] exists, skip {dst_path}", flush=True)
            continue

        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            tmp_path.unlink()

        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            _copy_attrs(src, dst)
            dst.attrs["policy_cache_version"] = 1
            dst.attrs["source_tracks_file"] = str(src_path)
            data_out = dst.create_group("data")
            data_in = src.get("data", {})
            for demo_id, demo in data_in.items():
                demo_out = data_out.create_group(str(demo_id))
                _copy_attrs(demo, demo_out)

                point_arr = None
                actions_arr = None
                for key in keys_to_copy:
                    if key not in demo:
                        continue
                    arr = np.asarray(demo[key])
                    if key in ("frames_rgb", "wrist_frames"):
                        arr = _image_to_uint8(arr)
                    elif key in ("actions", "robot_states"):
                        arr = arr.astype(np.float32, copy=False)
                    elif key in ("pre_point_traj", "point_traj", "query_xy_t0"):
                        arr = arr.astype(np.float32, copy=False)
                    _create_cached_dataset(demo_out, key, arr, compression=compression)
                    if key == "point_traj":
                        point_arr = arr
                    elif key == "actions":
                        actions_arr = arr

                tkey = target_cache_key(k_steps)
                if point_arr is not None:
                    targets = compute_target_cache(point_arr, actions_arr, k_steps)
                    _create_cached_dataset(demo_out, tkey, targets, compression=compression)

        if dst_path.exists():
            dst_path.unlink()
        tmp_path.replace(dst_path)
        print(f"[CACHE] [{file_idx}/{len(h5_files)}] wrote {dst_path}", flush=True)


def list_demo_dirs(flow_root: Path) -> List[Path]:
    """
    Recursively find all directories under flow_root that contain
    point_traj.npy, regardless of nesting depth. This supports both the
    original layout and the newer suite-prefixed layout.
    """
    demo_dirs: List[Path] = []
    # Each point_traj.npy parent directory is treated as one demo directory.
    for traj_file in sorted(flow_root.rglob("point_traj.npy")):
        demo_dirs.append(traj_file.parent)
    return demo_dirs


def load_mapping(mapping_csv: Optional[str], text_col: Optional[str], prompt_col: Optional[str]):
    if mapping_csv is None:
        return {}
    df = pd.read_csv(mapping_csv)
    cols = {c.lower(): c for c in df.columns}
    text = cols.get((text_col or "instruction").lower())
    prompt = cols.get((prompt_col or "prompt").lower())

    out = {}
    for _, row in df.iterrows():
        instr = str(row.get(text, "")).strip() if text else ""
        pr    = str(row.get(prompt, "")).strip() if prompt else ""
        key   = normalize_instr_key(instr)
        if key:
            out[key] = {"instruction": instr, "prompt": pr}
    return out


def augment_image_sequence_rlds_style_numpy(
    images_np: np.ndarray,
    image_size: Tuple[int, int] = (256, 256),
    train: bool = True,
    image_augment_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Apply sequence-consistent image augmentation.

    The input can be either TCHW or THWC, and the output keeps the same layout
    and dtype style. Evaluation mode only resizes frames.
    """
    assert images_np is not None and len(images_np) > 0, "images_np must be non-empty"
    if image_augment_kwargs is None:
        image_augment_kwargs = {}

    x = images_np
    assert x.ndim == 4, f"images_np must be 4D TCHW or THWC, got {x.shape}"
    input_is_thwc = (x.shape[-1] == 3)
    input_is_tchw = (x.shape[1] == 3)
    assert input_is_thwc or input_is_tchw, f"Cannot infer channel layout, shape={x.shape}"

    if input_is_thwc:
        x = np.transpose(x, (0, 3, 1, 2))  # THWC -> TCHW

    in_dtype = x.dtype
    is_uint8 = (in_dtype == np.uint8)

    float_in_255 = False
    if not is_uint8:
        vmax = float(np.max(x))
        float_in_255 = (vmax > 1.5)

    xt = torch.from_numpy(x)  # [T,C,H,W]
    if is_uint8:
        xt = xt.to(torch.float32) / 255.0
    else:
        xt = xt.to(torch.float32)
        if float_in_255:
            xt = xt / 255.0

    if not train:
        out = []
        for t in range(xt.shape[0]):
            img = F.resize(
                xt[t], image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            out.append(img)
        yt = torch.stack(out, dim=0)  # [T,C,H,W]
        yt = yt.clamp(0.0, 1.0)
        return _torch_to_numpy_like_input(yt, input_is_thwc, is_uint8, float_in_255)

    aug_order = image_augment_kwargs.get(
        "augment_order",
        [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    rrc_cfg = image_augment_kwargs.get("random_resized_crop", None)
    rb = image_augment_kwargs.get("random_brightness", None)    # [0.2]
    rc = image_augment_kwargs.get("random_contrast", None)      # [0.8, 1.2]
    rsat = image_augment_kwargs.get("random_saturation", None)  # [0.8, 1.2]
    rh = image_augment_kwargs.get("random_hue", None)           # [0.05]

    # Sample all random augmentation parameters once per sequence.
    # 1) RandomResizedCrop params (use first frame)
    if rrc_cfg is not None:
        scale = tuple(rrc_cfg.get("scale", [0.9, 0.9]))
        ratio = tuple(rrc_cfg.get("ratio", [1.0, 1.0]))
        i, j, h, w = RandomResizedCrop.get_params(xt[0], scale=scale, ratio=ratio)
        rrc_params = (i, j, h, w)
    else:
        rrc_params = None

    # 2) brightness
    if rb is not None and len(rb) > 0:
        delta = float(rb[0])
        brightness_factor = random.uniform(max(0.0, 1.0 - delta), 1.0 + delta)
    else:
        brightness_factor = None

    # 3) contrast
    if rc is not None and len(rc) == 2:
        contrast_factor = random.uniform(float(rc[0]), float(rc[1]))
    else:
        contrast_factor = None

    # 4) saturation
    if rsat is not None and len(rsat) == 2:
        saturation_factor = random.uniform(float(rsat[0]), float(rsat[1]))
    else:
        saturation_factor = None

    # 5) hue
    if rh is not None and len(rh) > 0:
        max_h = float(rh[0])
        hue_factor = random.uniform(-max_h, max_h)
    else:
        hue_factor = None

    # Apply transforms in augment_order to the full sequence.
    yt = xt
    for op in aug_order:
        if op == "random_resized_crop" and rrc_params is not None:
            top, left, height, width = rrc_params
            out = []
            for t in range(yt.shape[0]):
                img = F.resized_crop(
                    yt[t],
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    size=image_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                out.append(img)
            yt = torch.stack(out, dim=0)

        elif op == "random_brightness" and brightness_factor is not None:
            out = [F.adjust_brightness(yt[t], brightness_factor) for t in range(yt.shape[0])]
            yt = torch.stack(out, dim=0)

        elif op == "random_contrast" and contrast_factor is not None:
            out = [F.adjust_contrast(yt[t], contrast_factor) for t in range(yt.shape[0])]
            yt = torch.stack(out, dim=0)

        elif op == "random_saturation" and saturation_factor is not None:
            out = [F.adjust_saturation(yt[t], saturation_factor) for t in range(yt.shape[0])]
            yt = torch.stack(out, dim=0)

        elif op == "random_hue" and hue_factor is not None:
            out = [F.adjust_hue(yt[t], hue_factor) for t in range(yt.shape[0])]
            yt = torch.stack(out, dim=0)

        else:
            continue

    # If random_resized_crop was not applied, resize once at the end.
    if rrc_params is None:
        out = []
        for t in range(yt.shape[0]):
            img = F.resize(
                yt[t], image_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            out.append(img)
        yt = torch.stack(out, dim=0)

    yt = yt.clamp(0.0, 1.0)
    return _torch_to_numpy_like_input(yt, input_is_thwc, is_uint8, float_in_255)

def _torch_to_numpy_like_input(
    yt: torch.Tensor, input_is_thwc: bool, is_uint8: bool, float_in_255: bool
) -> np.ndarray:
    """yt: [T,C,H,W] float in [0,1] -> numpy with same layout & dtype style as input."""
    y = yt.detach().cpu()

    if is_uint8:
        y = (y * 255.0).round().clamp(0, 255).to(torch.uint8)
        out = y.numpy()
    else:
        if float_in_255:
            y = (y * 255.0)
        out = y.numpy().astype(np.float32)

    if input_is_thwc:
        out = np.transpose(out, (0, 2, 3, 1))  # TCHW -> THWC
    return out

def aug_query_points(
    q,                 # [N,2] float32 in pixel coords
    H=256, W=256,
    global_shift_std=1.0,   # pixels
    local_jitter_std=0.5,   # pixels
    p_outlier=0.02,
    do_shuffle=True,
    p_shuffle=0.1,
    enforce_change=False,
):
    q = q.copy() * (256.0 / 518.0)

    # (1) global + local jitter
    g = np.random.normal(scale=global_shift_std, size=(1, 2)).astype(np.float32)
    n = np.random.normal(scale=local_jitter_std, size=q.shape).astype(np.float32)
    q = q + g + n

    # (2) optional: inject a few outliers
    if p_outlier > 0:
        N = q.shape[0]
        out = (np.random.rand(N) < p_outlier)
        if out.any():
            q[out, 0] = np.random.uniform(0, W - 1, size=out.sum()).astype(np.float32)
            q[out, 1] = np.random.uniform(0, H - 1, size=out.sum()).astype(np.float32)

    # clamp
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)

    perm = None

    # (3) partial shuffle order
    if do_shuffle and p_shuffle > 0:
        N = q.shape[0]
        perm = np.arange(N)

        mask = (np.random.rand(N) < p_shuffle)
        idx = np.where(mask)[0]

        if idx.size > 1:
            shuffled = idx.copy()
            np.random.shuffle(shuffled)

            if enforce_change:
                for _ in range(10):
                    if not np.any(shuffled == idx):
                        break
                    np.random.shuffle(shuffled)
                if np.any(shuffled == idx):
                    shuffled = np.roll(idx, 1)

            perm[idx] = shuffled
            q = q[perm]

    return q


def augment_pred_flows(
    pred_flows: np.ndarray,
    rng: np.random.Generator | None = None,
    global_shift_std=0.1,
    point_bias_std=0.1,
    local_jitter_std=0.1,
    drift_step_std=3e-2,
    p_outlier=0.02,
    outlier_std=3e-2,
    p_dropout=0.05,
    do_shuffle=False,
    clip_abs=None,
    return_aux=False,
):
    assert pred_flows.ndim == 3 and pred_flows.shape[-1] == 3
    K, N, _ = pred_flows.shape
    x = pred_flows.astype(np.float32, copy=True)
    if rng is None:
        rng = np.random.default_rng()

    if global_shift_std > 0:
        g = rng.normal(0.0, global_shift_std, size=(1, 1, 3)).astype(np.float32)
        x += g

    if point_bias_std > 0:
        b = rng.normal(0.0, point_bias_std, size=(1, N, 3)).astype(np.float32)
        x += b

    if drift_step_std > 0:
        steps = rng.normal(0.0, drift_step_std, size=(K, N, 3)).astype(np.float32)
        drift = np.cumsum(steps, axis=0)
        x += drift

    if local_jitter_std > 0:
        j = rng.normal(0.0, local_jitter_std, size=(K, N, 3)).astype(np.float32)
        x += j

    out_mask = np.zeros((N,), dtype=bool)
    if p_outlier > 0:
        out_mask = rng.random(N) < p_outlier
        if out_mask.any():
            big = rng.normal(0.0, outlier_std, size=(1, out_mask.sum(), 3)).astype(np.float32)
            x[:, out_mask, :] += big

    drop_mask = np.zeros((N,), dtype=bool)
    if p_dropout > 0:
        drop_mask = rng.random(N) < p_dropout
        if drop_mask.any():
            src_pool = np.where(~drop_mask)[0]
            if src_pool.size > 0:
                src = rng.choice(src_pool, size=int(drop_mask.sum()), replace=True)
                x[:, drop_mask, :] = x[:, src, :]

    perm = np.arange(N)
    if do_shuffle:
        perm = rng.permutation(N)
        x = x[:, perm, :]

        out_mask = out_mask[perm]
        drop_mask = drop_mask[perm]

    # Optional clipping.
    if clip_abs is not None:
        x = np.clip(x, -float(clip_abs), float(clip_abs), out=x)

    if return_aux:
        aux = {
            "perm": perm,                 # Use this to synchronize other tensors when do_shuffle=True.
            "outlier_mask": out_mask,     # (N,)
            "drop_mask": drop_mask,       # (N,)
            "blend_lam": lam,
        }
        return x, aux
    return x


class FutureKDataset(Dataset):
    """
    Load data from the new *_tracks.hdf5 files instead of per-directory .npy
    files.

    Expected layout:
        flow_root/
          libero_object/
            <task_name>_demo/
              <task_name>_demo_tracks.hdf5
              <task_name>_demo_demo_xx/  (legacy per-demo directory; ignored)

    HDF5 structure:
        <task_name>_demo_tracks.hdf5
          / (root)
            attrs: task_text, source_hdf5, image_key, test_demo_id, ...
            /data
              /0
                attrs: prompt, has_tracks
                datasets:
                  point_traj   (T, N, 3)
                  vis          (T, N)
                  frames_rgb   (T, H, W, 3)
                  vggt_hidden  (T, L, D) or (T, D)
                  p0_world     (N, 3)
    """
    def __init__(
        self,
        flow_root: str,
        k_steps: int,
        num_points: int,
        use_flow_condition: bool = True,
        train: bool = True,
        augment_images: Optional[bool] = None,
        augment_flows: Optional[bool] = None,
        flow_aug_kwargs: Optional[Dict[str, float]] = None,
        flow_key: str = "pre_point_traj",
        target_flow_key: str = "point_traj",
        samples: Optional[List[Tuple[str, str, int]]] = None,
    ):
        super().__init__()
        self.root = Path(flow_root)
        self.k = k_steps
        self.n = num_points
        self.use_flow_condition = bool(use_flow_condition)
        self.train = bool(train)
        self.augment_images = self.train if augment_images is None else bool(augment_images)
        self.augment_flows = (self.train and self.use_flow_condition) if augment_flows is None else bool(augment_flows)
        self.flow_aug_kwargs = dict(flow_aug_kwargs or {})
        self.flow_key = str(flow_key)
        self.target_flow_key = str(target_flow_key)
        self.target_cache_key = target_cache_key(self.k)

        # Per-worker HDF5 file-handle cache.
        self._file_handles: Dict[str, h5py.File] = {}

        # Traverse *_tracks.hdf5 files and collect (h5_path, demo_id, s) windows.
        self.samples: List[Tuple[str, str, int]] = []
        if samples is not None:
            self.samples = list(samples)
            assert len(self.samples) > 0, "No valid windows for K-step future prediction."
            return

        # Only use processed track files and avoid raw, large demonstration HDF5 files.
        h5_files = sorted(self.root.rglob("*_tracks.hdf5"))
        if len(h5_files) == 0:
            raise RuntimeError(f"No *_tracks.hdf5 files found under {self.root}")

        for h5_path in h5_files:
            # Some files may be locked by another process; skip them.
            try:
                f = h5py.File(h5_path, "r")
            except (BlockingIOError, OSError) as e:
                print(f"[WARN] Cannot open {h5_path}: {e}. Skipping this file.")
                continue

            with f:
                if "data" not in f:
                    continue
                data_grp = f["data"]
                for demo_id, grp in data_grp.items():
                    # Skip demos explicitly marked as not having tracks.
                    if grp.attrs.get("has_tracks", True) is False:
                        continue
                    required = ["frames_rgb", "wrist_frames", "actions", "robot_states"]
                    if self.use_flow_condition:
                        required.append(self.flow_key)
                        if self.target_flow_key == "point_traj":
                            if self.target_cache_key not in grp:
                                required.append("point_traj")
                        else:
                            required.append(self.target_flow_key)
                    if any(key not in grp for key in required):
                        continue

                    lengths = [
                        int(grp["frames_rgb"].shape[0]),
                        int(grp["wrist_frames"].shape[0]),
                        int(grp["actions"].shape[0]),
                        int(grp["robot_states"].shape[0]),
                    ]
                    if self.use_flow_condition:
                        lengths.append(int(grp[self.flow_key].shape[0]))
                        if self.target_flow_key in grp:
                            lengths.append(int(grp[self.target_flow_key].shape[0]))
                    T_len = min(lengths)
                    # s ranges where s+K <= T-1  -> s <= T-K-1
                    max_s = (T_len - 1) - self.k
                    if max_s < 0:
                        continue

                    for s in range(0, max_s + 1):
                        self.samples.append((str(h5_path), str(demo_id), s))

        assert len(self.samples) > 0, "No valid windows for K-step future prediction."

    # DataLoader multiprocessing support: do not serialize open file handles.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handles"] = None   # Let each worker reopen its own HDF5 handles.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = {}

    def _get_file(self, path: str) -> h5py.File:
        """Cache HDF5 handles in the current process to avoid repeated open/close calls."""
        if self._file_handles is None:
            self._file_handles = {}
        fh = self._file_handles.get(path)
        if fh is None:
            fh = h5py.File(path, "r")
            self._file_handles[path] = fh
        return fh

    def __del__(self):
        if getattr(self, "_file_handles", None):
            for fh in self._file_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass

    def _to_rgb_chw(self, frame: np.ndarray) -> np.ndarray:
        """Robustly convert any frame to CHW uint8 RGB."""
        a = np.asarray(frame)
        a = np.squeeze(a)

        # Normalize all input layouts to HWC RGB before converting to uint8.
        if a.ndim == 2:
            # HW -> HWC
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3:
            if a.shape[-1] in (3, 4):
                # HWC or HW(A)
                a = a[..., :3]
            elif a.shape[0] in (3, 4):
                # CHW or (A)CHW
                a = np.moveaxis(a, 0, -1)[..., :3]  # -> HWC
            else:
                a2 = np.squeeze(a)
                if a2.ndim == 2:
                    a = np.stack([a2, a2, a2], axis=-1)
                else:
                    raise ValueError(f"Unexpected image shape after squeeze: {a.shape}")
        else:
            raise ValueError(f"Unexpected image shape: {a.shape}")

        # dtype -> uint8
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)

        # HWC -> CHW
        a = np.transpose(a, (2, 0, 1))  # [3, H, W]

        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        return a

    def _load_cond_frames(self, frames_ds, s: int) -> List[Image.Image]:
        """
        Load cond_k frames from the HDF5 frames_rgb dataset. Frames are sampled
        backward from s with cond_stride; if there are not enough frames, repeat
        the earliest available frame.
        """
        T = frames_ds.shape[0]
        idxs = list(range(
            max(0, s - (self.cond_k - 1) * self.cond_stride),
            s + 1,
            self.cond_stride,
        ))
        if len(idxs) > self.cond_k:
            idxs = idxs[-self.cond_k:]
        while len(idxs) < self.cond_k:
            idxs.insert(0, idxs[0])

        imgs: List[Image.Image] = []
        for idx in idxs:
            idx_clamped = min(idx, T - 1)
            arr = self._to_rgb_hwc(frames_ds[idx_clamped])
            imgs.append(Image.fromarray(arr))
        return imgs

    # Standard Dataset interface.
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        h5_path_str, demo_id, s = self.samples[i]
        f = self._get_file(h5_path_str)
        grp = f["data"][demo_id]

        task_dir_name = Path(h5_path_str).parent.name
        instruction = normalize_instr_key(task_dir_name)

        if self.use_flow_condition:
            pred_flows = np.asarray(grp[self.flow_key][s], dtype=np.float32)
            if self.augment_flows:
                pred_flows = augment_pred_flows(
                    pred_flows,
                    rng=None,
                    **self.flow_aug_kwargs,
                )

            if (
                self.target_flow_key == "point_traj"
                and self.target_cache_key in grp
                and s < int(grp[self.target_cache_key].shape[0])
            ):
                target_flows = np.asarray(grp[self.target_cache_key][s], dtype=np.float32)
            else:
                target_flows = load_segment_keyframe_targets(grp, s, self.k, point_key=self.target_flow_key)
        else:
            pred_flows = np.zeros((self.k, self.n, 3), dtype=np.float32)
            target_flows = np.zeros((self.k, self.n, 3), dtype=np.float32)

        images = grp["frames_rgb"]  # (T, 3, H, W) -> CHW uint8
        images = images[s:s+1]
        images = augment_image_sequence_rlds_style_numpy(images, train=self.augment_images)
        images = np.asarray(images, dtype=np.float32).squeeze(0)
        images = self._to_rgb_chw(images)

        wrist_images = grp["wrist_frames"] # (T, 3, H, W)
        wrist_images = wrist_images[s:s+1]
        wrist_images = augment_image_sequence_rlds_style_numpy(wrist_images, train=self.augment_images)
        wrist_images = np.asarray(wrist_images, dtype=np.float32).squeeze(0)

        if self.use_flow_condition and "query_xy_t0" in grp:
            initial_query_points = np.asarray(grp["query_xy_t0"], dtype=np.float32)
            initial_query_points = transform_query_points(initial_query_points, train=self.train)
        else:
            initial_query_points = np.zeros((self.n, 2), dtype=np.float32)

        robot_states = grp["robot_states"]      # (1, proprio_dim)
        robot_states = robot_states[s:s+1]
        robot_states = np.asarray(robot_states, dtype=np.float32).squeeze(0)

        action_chunk = grp["actions"]
        action_chunk = action_chunk[s:s+self.k]  # (T, action_dim)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)

        return {
            "images": torch.from_numpy(images).float(),
            "wrist_images": torch.from_numpy(wrist_images).float(),
            "proprios": torch.from_numpy(robot_states).float(),
            "flows": torch.from_numpy(pred_flows).float(),
            "initial_query_points": torch.from_numpy(initial_query_points).float(),
            "target_flows": torch.from_numpy(target_flows).float(),
            "actions": torch.from_numpy(action_chunk).float(),
            "instruction": instruction,
        }


def collate_fn(batch):
    instruction = [b["instruction"] for b in batch]
    images = torch.stack([b["images"] for b in batch], dim=0)  # [B, 3, H, W]
    wrist_images = torch.stack([b["wrist_images"] for b in batch], dim=0)  # [B, 3, H, W]
    proprios   = torch.stack([b["proprios"]   for b in batch], dim=0) # [1, proprio_dim]
    flows   = torch.stack([b["flows"]   for b in batch], dim=0)  # [B, K, N, 3]
    initial_query_points = torch.stack([b["initial_query_points"] for b in batch], dim=0)  # [B, N, 2]
    target_flows = torch.stack([b["target_flows"] for b in batch], dim=0)  # [B, K, N, 3]
    actions = torch.stack([b["actions"] for b in batch], dim=0)  # [B, K, action_dim]
    return images, wrist_images, proprios, flows, initial_query_points, target_flows, actions, instruction

def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying nn.Module if wrapped by DDP/DataParallel."""
    return m.module if hasattr(m, "module") else m

def train_one_epoch_dp(model, dataloader, noise_scheduler, optimizer, device, cfg, accum_steps=1, max_grad_norm=0.0, fp16=False, ema: Optional[Any]=None):
    model.train()
    total_loss, count = 0.0, 0

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    text_processor = get_text_processor()

    for step, batch in enumerate(dataloader):
        images, wrist_images, proprios, flows, initial_query_points, target_flows, action_gt, instruction = batch
        images = images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        proprios = proprios.to(device, non_blocking=True)
        flows = flows.to(device, non_blocking=True)
        initial_query_points = initial_query_points.to(device, non_blocking=True)
        target_flows = target_flows.to(device, non_blocking=True)
        action_gt = action_gt.to(device, non_blocking=True)
        proc_txt = text_processor(text=instruction, return_tensors="pt", padding=True)
        input_ids = proc_txt["input_ids"].to(device, non_blocking=True)
        pad_id = getattr(getattr(text_processor, "tokenizer", None), "pad_token_id", None)
        attn_mask = (input_ids != pad_id).long().to(device, non_blocking=True) if pad_id is not None else torch.ones_like(input_ids)


        bsz = action_gt.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()

        noise = torch.randn_like(action_gt)
        noisy_actions = noise_scheduler.add_noise(action_gt, noise, timesteps)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and device.type == "cuda")):
            noise_pred, predict_plan, target_plan = model(
                                # noisy_actions, timesteps, images, proprios, flows, initial_query_points, target_flows
                noisy_actions, timesteps, images, wrist_images, proprios, flows, initial_query_points, target_flows, input_ids, attn_mask
            )

            action_loss = torch.nn.functional.mse_loss(noise_pred, noise)

            align = torch.tensor(0., device=device)
            if getattr(cfg, "apply_alignment_loss", False):
                # align = torch.nn.functional.mse_loss(predict_plan, target_plan) # (B, latent_dim)
                # predict_plan = torch.nn.functional.normalize(predict_plan, dim=-1)
                # target_plan = torch.nn.functional.normalize(target_plan.detach(), dim=-1)
                align = torch.nn.functional.mse_loss(predict_plan, target_plan)   # Or use 1 - cosine_similarity.
                loss = action_loss + float(getattr(cfg, "alignment_loss_coef", 1.0)) * align
            else:
                loss = action_loss

            loss_scaled = loss / max(1, accum_steps)

        if scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step + 1) % accum_steps == 0:
            if max_grad_norm and max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # --- EMA update: run once per optimizer step (after the weights update) ---
            if ema is not None:
                ema.step(_unwrap_model(model))

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * bsz
        count += bsz

    return total_loss / max(1, count)

@torch.no_grad()
def evaluate_dp(model, dataloader, noise_scheduler, device, cfg, fp16=False):
    model.eval()
    total_diff_loss, total_l1, total_align, count = 0.0, 0.0, 0.0, 0
    text_processor = get_text_processor()

    # DDIMScheduler has alphas_cumprod
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)  # [T]
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    do_clip = bool(getattr(noise_scheduler.config, "clip_sample", False))
    clip_range = float(getattr(noise_scheduler.config, "clip_sample_range", 1.0))

    for batch in dataloader:
        images, wrist_images, proprios, flows, initial_query_points, target_flows, action_gt, instruction = batch
        images = images.to(device, non_blocking=True)
        wrist_images = wrist_images.to(device, non_blocking=True)
        proprios = proprios.to(device, non_blocking=True)
        flows = flows.to(device, non_blocking=True)
        initial_query_points = initial_query_points.to(device, non_blocking=True)
        target_flows = target_flows.to(device, non_blocking=True)
        action_gt = action_gt.to(device, non_blocking=True)

        # input_ids = input_ids.to(device, non_blocking=True)
        # attn_mask = attn_mask.to(device, non_blocking=True)
        proc_txt = text_processor(text=instruction, return_tensors="pt", padding=True)
        input_ids = proc_txt["input_ids"].to(device, non_blocking=True)
        pad_id = getattr(getattr(text_processor, "tokenizer", None), "pad_token_id", None)
        attn_mask = (input_ids != pad_id).long().to(device, non_blocking=True) if pad_id is not None else torch.ones_like(input_ids)


        bsz = action_gt.shape[0]
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        noise = torch.randn_like(action_gt)
        x_t = noise_scheduler.add_noise(action_gt, noise, t)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(fp16 and device.type == "cuda")):
            noise_pred, predict_plan, target_plan = model(
                x_t, t, images, wrist_images, proprios, flows, initial_query_points, target_flows, input_ids, attn_mask
            )

            # Diffusion training target: epsilon MSE.
            diff_loss = torch.nn.functional.mse_loss(noise_pred, noise)

            align = torch.tensor(0., device=device)
            # Optional alignment loss between the predicted and target action plans.
            if getattr(cfg, "apply_alignment_loss", False):
                align = torch.nn.functional.mse_loss(predict_plan, target_plan)
                loss = diff_loss + float(getattr(cfg, "alignment_loss_coef", 1.0)) * align
            else:
                loss = diff_loss

            # Action L1 metric: reconstruct x0_hat from x_t and the predicted noise.
            a = alphas_cumprod[t].view(bsz, *([1] * (action_gt.ndim - 1)))  # [B,1,1]
            sqrt_a = torch.sqrt(a)
            sqrt_oma = torch.sqrt(1.0 - a)

            if pred_type == "epsilon":
                x0_hat = (x_t - sqrt_oma * noise_pred) / (sqrt_a + 1e-8)
            elif pred_type == "v_prediction":
                # x0 = sqrt(a) * x_t - sqrt(1-a) * v
                x0_hat = sqrt_a * x_t - sqrt_oma * noise_pred
            else:
                raise ValueError(f"Unsupported prediction_type: {pred_type}")

            if do_clip:
                x0_hat = x0_hat.clamp(-clip_range, clip_range)

            action_l1 = torch.nn.functional.l1_loss(x0_hat, action_gt)

        total_diff_loss += diff_loss.item() * bsz
        total_align += align.item() * bsz
        total_l1 += action_l1.item() * bsz
        count += bsz

    
    # avg_diff = total_diff_loss / max(1, count)avg_loss = (total_diff_loss + float(getattr(cfg, "alignment_loss_coef", 1.0)) * total_align) / max(1, count)
    avg_diff = total_diff_loss / max(1, count)
    avg_align = total_align / max(1, count)
    avg_loss = (total_diff_loss + total_align) / max(1, count)
    avg_l1 = total_l1 / max(1, count)

    # Return metrics as a dict for clearer logging and downstream use.
    return {
        "loss_total": avg_loss,
        "loss_diffusion": avg_diff,
        "loss_align": avg_align,
        "action_l1": avg_l1,
    }


# Model: DiT backbone; model outputs v
def count_params(module: torch.nn.Module): # count total parameters & trainable parameters
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def human_m(x):  # Display parameter counts in millions.
    return x / 1e6

def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ======================
#  Main
# ======================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_root", type=str, required=True)

    parser.add_argument("--k_steps", type=int, default=20, help="Number of future flow keyframes")
    parser.add_argument("--num_points", type=int, default=100, help="Number of flow points")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Disable clipping with a non-positive value")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=5e-6)
    parser.add_argument("--anneal_lr", action="store_true", help="Enable learning rate annealing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./ckpts_improved")
    parser.add_argument("--cache_root", type=str, default=None, help="Optional optimized HDF5 policy cache root")
    parser.add_argument("--build_cache", action="store_true", help="Build/refresh --cache_root before training")
    parser.add_argument("--build_cache_only", action="store_true", help="Build --cache_root and exit")
    parser.add_argument("--cache_overwrite", action="store_true", help="Overwrite existing cache HDF5 files")
    parser.add_argument("--cache_compression", choices=["none", "lzf", "gzip"], default="lzf")
    parser.add_argument("--cache_limit_files", type=int, default=None, help="Debug helper: cache only first N files")
    parser.add_argument("--save_raw", action="store_true", help="Also evaluate and save the raw non-EMA model")
    parser.add_argument("--flow_key", type=str, default="pre_point_traj_metric",
                        help="HDF5 dataset key for predicted flow conditioning.")
    parser.add_argument("--target_flow_key", type=str, default="point_traj_metric",
                        help="HDF5 dataset key for target trajectories used by optional alignment.")
    parser.add_argument("--flow_aug", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable random augmentation on flow conditions during flow-policy training.")
    parser.add_argument("--flow_aug_scale", type=float, default=0.2,
                        help="Scale all flow augmentation std values. Use 0 to disable noise.")
    parser.add_argument("--flow_aug_global_shift_std", type=float, default=0.05)
    parser.add_argument("--flow_aug_point_bias_std", type=float, default=0.05)
    parser.add_argument("--flow_aug_local_jitter_std", type=float, default=0.05)
    parser.add_argument("--flow_aug_drift_step_std", type=float, default=3e-2)
    parser.add_argument("--save_latest", action=argparse.BooleanOptionalAction, default=True,
                        help="Overwrite latest checkpoint during training in addition to best checkpoint.")
    parser.add_argument("--latest_save_frequency", type=int, default=1,
                        help="Save latest checkpoint every N epochs; final epoch is always saved when --save_latest is enabled.")
    parser.add_argument(
        "--policy_cfg",
        type=str,
        default="model/model_res_dp_flow.yaml",
        help="Flow-conditioned policy config yaml.",
    )

    # --- [W&B] args ---
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="Action Policy Training", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_dir", type=str, default=None, help="W&B local dir for logs")


    args = parser.parse_args()
    policy_cfg_path = Path(args.policy_cfg)
    if not policy_cfg_path.is_absolute() and not policy_cfg_path.exists():
        policy_cfg_path = Path(__file__).resolve().parents[1] / policy_cfg_path
    args.policy_cfg = str(policy_cfg_path)
    if args.wandb_run is None:
        args.wandb_run = f"action_policy_{policy_cfg_path.stem}_k{args.k_steps}_bs{args.batch_size}_lr{args.lr}"

    print(f"[INFO] Loading policy cfg: {args.policy_cfg}")
    cfg_yaml = OmegaConf.load(args.policy_cfg)
    cfg = OmegaConf.merge(cfg_yaml)
    OmegaConf.resolve(cfg)
    use_flow_condition = bool(getattr(cfg, "use_flow_condition", True))
    print(f"[INFO] use_flow_condition={use_flow_condition}")

    data_root = args.flow_root
    if args.cache_root is not None:
        cache_root = Path(args.cache_root)
        need_build = args.build_cache or args.build_cache_only or not cache_root.exists()
        if need_build:
            build_policy_training_cache(
                flow_root=args.flow_root,
                cache_root=str(cache_root),
                k_steps=args.k_steps,
                compression=args.cache_compression,
                overwrite=args.cache_overwrite,
                limit_files=args.cache_limit_files,
            )
        data_root = str(cache_root)
        if args.build_cache_only:
            print(f"[CACHE] Build complete: {data_root}")
            return
    args.data_root = data_root

    set_seed_everywhere(args.seed)
    # --- [W&B] init ---
    wb = None
    if args.wandb:
        assert wandb is not None, "wandb not installed. pip install wandb"
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config=vars(args),
        )

    # Main training configuration.
    num_epochs = args.epochs
    # batch_size = args.batch_size
    # lr = args.lr
    # val_ratio = 0.1
    # l1_weight = 1.0

    # flow_ratio = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds_index = FutureKDataset(
        data_root,
        args.k_steps,
        args.num_points,
        use_flow_condition=use_flow_condition,
        train=True,
        augment_images=False,
        augment_flows=False,
        flow_key=args.flow_key,
        target_flow_key=args.target_flow_key,
    )
    n_train = int(len(ds_index) * 0.95); n_val = len(ds_index) - n_train
    split_generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds_index), generator=split_generator).tolist()
    train_samples = [ds_index.samples[i] for i in perm[:n_train]]
    val_samples = [ds_index.samples[i] for i in perm[n_train:]]
    flow_aug_kwargs = {
        "global_shift_std": float(args.flow_aug_scale) * float(args.flow_aug_global_shift_std),
        "point_bias_std": float(args.flow_aug_scale) * float(args.flow_aug_point_bias_std),
        "local_jitter_std": float(args.flow_aug_scale) * float(args.flow_aug_local_jitter_std),
        "drift_step_std": float(args.flow_aug_scale) * float(args.flow_aug_drift_step_std),
    }
    train_augment_flows = bool(use_flow_condition and args.flow_aug and args.flow_aug_scale != 0.0)
    print(f"[DATA] train_augment_flows={train_augment_flows} flow_aug_kwargs={flow_aug_kwargs if train_augment_flows else {}}")

    train_set = FutureKDataset(
        data_root,
        args.k_steps,
        args.num_points,
        use_flow_condition=use_flow_condition,
        train=True,
        augment_images=True,
        augment_flows=train_augment_flows,
        flow_aug_kwargs=flow_aug_kwargs,
        flow_key=args.flow_key,
        target_flow_key=args.target_flow_key,
        samples=train_samples,
    )
    val_set = FutureKDataset(
        data_root,
        args.k_steps,
        args.num_points,
        use_flow_condition=use_flow_condition,
        train=False,
        augment_images=False,
        augment_flows=False,
        flow_key=args.flow_key,
        target_flow_key=args.target_flow_key,
        samples=val_samples,
    )
    print(
        f"[DATA] root={data_root} samples={len(ds_index)} "
        f"train={len(train_set)} val={len(val_set)} use_flow={use_flow_condition}"
    )

    pw = args.workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=4 if pw else None,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.workers // 2) if pw else 0,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
        collate_fn=collate_fn,
    )

    def strip_prefix(state_dict, prefixes=("model.", "module.")):
        out = {}
        for k, v in state_dict.items():
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
            out[k] = v
        return out

    model = instantiate(cfg.model)
    model.to(device)

    # --------- EMA (Exponential Moving Average) ---------
    use_ema = bool(getattr(getattr(cfg, 'training', {}), 'use_ema', False))
    ema = None
    ema_model = None
    if use_ema:
        # Keep a separate smoothed copy of weights for eval/saving
        ema_model = copy.deepcopy(model)
        ema_model.to(device)
        ema = instantiate(cfg.ema, model=ema_model)
        print('[EMA] enabled. Will update EMA weights after each optimizer step.')

    tot, trn = count_params(model)
    print(f"Total parameters:     {human_m(tot):8.2f} M")
    print(f"Trainable parameters: {human_m(trn):8.2f} M")
    print(f"Frozen ratio:         {100.0 * (1.0 - trn / max(tot,1)):6.2f}%")

    noise_scheduler = instantiate(cfg.noise_scheduler)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    best_val_loss = float("inf")
    best_val_loss_ema = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_dp(
            model, train_loader, noise_scheduler, optimizer, device,
            cfg=cfg, accum_steps=args.accum_steps, max_grad_norm=args.max_grad_norm, fp16=args.fp16,
            ema=ema
        )

        val = None
        if args.save_raw or not (use_ema and ema_model is not None):
            val = evaluate_dp(model, val_loader, noise_scheduler, device, cfg=cfg, fp16=args.fp16)

        val_ema = None
        if use_ema and ema_model is not None:
            val_ema = evaluate_dp(ema_model, val_loader, noise_scheduler, device, cfg=cfg, fp16=args.fp16)

        if val_ema is not None:
            print(
                f"[Epoch {epoch}/{num_epochs}] train_loss={train_loss:.6f} | "
                f"[EMA] val_total={val_ema['loss_total']:.6f} "
                f"val_diff={val_ema['loss_diffusion']:.6f} "
                f"val_align{val_ema['loss_align']:.6f} "
                f"val_l1={val_ema['action_l1']:.6f}"
            )
        elif val is not None:
            print(
                f"[Epoch {epoch}/{num_epochs}] train_loss={train_loss:.6f} | "
                f"val_total={val['loss_total']:.6f} "
                f"val_diff={val['loss_diffusion']:.6f} "
                f"val_align{val['loss_align']:.6f} "
                f"val_l1={val['action_l1']:.6f}"
            )

        if wb is not None:
            log_payload = {
                "train/loss": float(train_loss),
                "epoch": int(epoch),
            }
            if val is not None:
                log_payload.update({
                    "val/loss_total": float(val["loss_total"]),
                    "val/loss_diffusion": float(val["loss_diffusion"]),
                    "val/action_l1": float(val["action_l1"]),
                    "val/align": float(val["loss_align"]),
                })
            if val_ema is not None:
                log_payload.update({
                    "val_ema/loss_total": float(val_ema["loss_total"]),
                    "val_ema/loss_diffusion": float(val_ema["loss_diffusion"]),
                    "val_ema/action_l1": float(val_ema["action_l1"]),
                    "val_ema/align": float(val_ema["loss_align"]),
                })
            wandb.log(log_payload, commit=True)

        if (args.save_raw or val_ema is None) and val is not None and float(val["action_l1"]) < best_val_loss:
            best_val_loss = float(val["action_l1"])
            save_dir = Path(str(args.save_dir))
            save_dir.mkdir(parents=True, exist_ok=True)   # Ensure the checkpoint directory exists.

            ckpt_path = save_dir / "best_action_policy.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": int(epoch),
                    "val_loss_total": float(val["loss_total"]),
                    "val_loss_diffusion": float(val["loss_diffusion"]),
                    "val_action_l1": float(val["action_l1"]),
                    "val_align": float(val["loss_align"]),
                },
                ckpt_path,
            )
            print(
                f"  >> Saved best model to {ckpt_path} "
                f"(val_total={val['loss_total']:.6f}, val_l1={val['action_l1']:.6f})"
            )

        if val_ema is not None and float(val_ema["action_l1"]) < best_val_loss_ema:
            best_val_loss_ema = float(val_ema["action_l1"])
            save_dir = Path(str(args.save_dir))
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path_ema = save_dir / "best_action_policy_ema.pt"
            torch.save(
                {
                    "model_state": ema_model.state_dict(),
                    "epoch": int(epoch),
                    "val_loss_total": float(val_ema["loss_total"]),
                    "val_loss_diffusion": float(val_ema["loss_diffusion"]),
                    "val_action_l1": float(val_ema["action_l1"]),
                    "val_align": float(val_ema["loss_align"]),
                },
                ckpt_path_ema,
            )
            print(
                f"  >> Saved best EMA model to {ckpt_path_ema} "
                f"(ema_val_total={val_ema['loss_total']:.6f}, ema_val_l1={val_ema['action_l1']:.6f})"
            )

        if args.save_latest:
            latest_freq = max(1, int(args.latest_save_frequency))
            if (epoch % latest_freq == 0) or (epoch == num_epochs):
                save_dir = Path(str(args.save_dir))
                save_dir.mkdir(parents=True, exist_ok=True)

                if (args.save_raw or val_ema is None) and val is not None:
                    latest_path = save_dir / "latest_action_policy.pt"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "epoch": int(epoch),
                            "val_loss_total": float(val["loss_total"]),
                            "val_loss_diffusion": float(val["loss_diffusion"]),
                            "val_action_l1": float(val["action_l1"]),
                            "val_align": float(val["loss_align"]),
                            "is_latest": True,
                        },
                        latest_path,
                    )

                if val_ema is not None:
                    latest_path_ema = save_dir / "latest_action_policy_ema.pt"
                    torch.save(
                        {
                            "model_state": ema_model.state_dict(),
                            "epoch": int(epoch),
                            "val_loss_total": float(val_ema["loss_total"]),
                            "val_loss_diffusion": float(val_ema["loss_diffusion"]),
                            "val_action_l1": float(val_ema["action_l1"]),
                            "val_align": float(val_ema["loss_align"]),
                            "is_latest": True,
                        },
                        latest_path_ema,
                    )
                    print(
                        f"  >> Saved latest EMA model to {latest_path_ema} "
                        f"(epoch={epoch}, ema_val_total={val_ema['loss_total']:.6f}, "
                        f"ema_val_l1={val_ema['action_l1']:.6f})"
                    )

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
