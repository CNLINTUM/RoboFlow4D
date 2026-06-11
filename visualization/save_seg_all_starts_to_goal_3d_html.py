#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export interactive 3D start-to-goal flow HTMLs for each gripper segment.

The output is a standalone Three.js page:
  phase 1: the 3D flow trail grows from current state to atomic goal
  phase 2: the gripper/object points move along the already-visible trail

This is intentionally separate from the 2D MP4 exporter so the old workflow
stays untouched.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

from common import (
    TrajectorySegmenter,
    pick_one_demo_id,
    task_name_from_h5,
    to_rgb_hwc_uint8,
)


DEPTH_KEYS = ("sim_depths", "depths", "depth", "depth_maps", "depth_video")
INTRINSICS_KEYS = ("sim_intrinsics", "intrinsics", "intrs2", "intrs")
CAM_TO_SCENE_KEYS = (
    "sim_T_base_cam",
    "T_base_cam",
    "T_world_cam",
    "c2w_traj",
    "cam2world_gl",
    "cam2world",
    "camera_poses",
    "c2w",
    "extrs",
    "extrinsics",
)
WORLD_TO_CAM_KEYS = ("w2c", "w2c_traj", "camera_extrinsics")
METRIC_INTRINSICS_KEYS = ("sim_intrinsics", "intrinsics")
METRIC_CAM_TO_SCENE_KEYS = (
    "sim_T_base_cam",
    "T_base_cam",
    "T_world_cam",
    # ManiSkill RGB-D metric trajectories are lifted with c2w_traj, so the
    # scene must use the same camera-to-world transform for alignment.
    "c2w_traj",
    "c2w",
)
SPATRACKER_INTRINSICS_KEYS = ("intrs2", "intrs", "intrinsics")
SPATRACKER_CAM_TO_SCENE_KEYS = (
    "c2w_traj",
    "cam2world_gl",
    "cam2world",
    "camera_poses",
    "c2w",
    "extrs",
    "extrinsics",
)
SPATRACKER_WORLD_TO_CAM_KEYS = ("w2c", "w2c_traj", "camera_extrinsics")
PRED_TRAJ_KEYS = ("pre_point_traj_metric", "pre_point_traj", "pred_point_traj")
GT_TRAJ_KEYS = ("point_traj_metric", "point_traj", "point_traj_base_metric")
FLOW_COLOR_MODES = ("rainbow", "index", "rgb", "time")
SCENE_COORD_MODES = ("auto", "metric", "spatracker", "camera")
FLOW_ALIGN_MODES = ("none", "start_translation", "endpoint_lerp", "per_frame_translation")
JET_STOPS = (
    (0.00, (0.03, 0.18, 0.95)),
    (0.18, (0.00, 0.66, 1.00)),
    (0.38, (0.00, 0.92, 0.40)),
    (0.58, (0.72, 0.96, 0.10)),
    (0.76, (1.00, 0.78, 0.05)),
    (0.90, (1.00, 0.34, 0.03)),
    (1.00, (0.86, 0.02, 0.02)),
)


def collect_start_indices(s0: int, seg_last: int, start_stride: int, max_start_frames: int) -> List[int]:
    starts = list(range(int(s0), int(seg_last) + 1, max(1, int(start_stride))))
    if max_start_frames is not None and int(max_start_frames) > 0:
        starts = starts[: int(max_start_frames)]
    return starts


def first_existing_key(grp: h5py.Group, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in grp:
            return key
    return None


def as_float_array(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def sanitize_name(text: str, max_len: int = 180) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return s[:max_len] if len(s) > max_len else s


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32)
    if out.ndim != 3 or out.shape[-1] < 3:
        raise ValueError(f"trajectory must be [K,N,3], got {out.shape}")
    out = out[..., :3]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_flow_xyz(
    grp: h5py.Group,
    traj_key: str,
    key_idxs: np.ndarray,
    start_t: int,
) -> Tuple[np.ndarray, str]:
    if traj_key == "auto":
        traj_key = first_existing_key(grp, GT_TRAJ_KEYS + PRED_TRAJ_KEYS) or ""
    if not traj_key:
        raise KeyError(f"no trajectory key found. available keys={list(grp.keys())}")
    if traj_key not in grp:
        raise KeyError(f"traj_key={traj_key!r} not found. available keys={list(grp.keys())}")

    ds = grp[traj_key]
    if ds.ndim == 3:
        idx = np.asarray(key_idxs, dtype=np.int64)
        uniq, inv = np.unique(idx, return_inverse=True)
        xyz_u = np.asarray(ds[uniq], dtype=np.float32)
        xyz = xyz_u[inv]
    elif ds.ndim == 4:
        # Common prediction layout: [T, K, N, 3].
        xyz = ds[int(start_t)]
        if xyz.shape[0] != len(key_idxs):
            src = np.linspace(0.0, 1.0, num=xyz.shape[0], dtype=np.float32)
            dst = np.linspace(0.0, 1.0, num=len(key_idxs), dtype=np.float32)
            flat = xyz.reshape(xyz.shape[0], -1)
            interp = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], axis=1)
            xyz = interp.reshape(len(key_idxs), xyz.shape[1], xyz.shape[2])
    else:
        raise ValueError(f"unsupported {traj_key} shape={ds.shape}")
    return normalize_xyz(xyz), traj_key


def apply_flow_visibility_mask(
    flow_xyz: np.ndarray,
    flow_colors: np.ndarray,
    grp: h5py.Group,
    key_idxs: np.ndarray,
    enabled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Hide tracks marked invalid by preprocessing without changing coordinates."""
    info: Dict[str, Any] = {
        "enabled": False,
        "disabled_by_user": not bool(enabled),
        "mask_key": "",
        "num_before": int(flow_xyz.shape[1]) if flow_xyz.ndim >= 2 else 0,
        "num_after": int(flow_xyz.shape[1]) if flow_xyz.ndim >= 2 else 0,
        "num_removed": 0,
        "fallback": False,
    }
    if not enabled:
        return flow_xyz, flow_colors, info
    if flow_xyz.ndim != 3 or flow_xyz.shape[1] == 0:
        return flow_xyz, flow_colors, info

    mask_key = first_existing_key(grp, ["point_traj_valid_mask", "vis"]) or ""
    if not mask_key:
        return flow_xyz, flow_colors, info

    ds = grp[mask_key]
    if ds.ndim < 2:
        return flow_xyz, flow_colors, info

    idx = np.asarray(key_idxs, dtype=np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)
    valid = np.asarray(ds[uniq], dtype=np.float32)[inv]
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    if valid.ndim != 2:
        return flow_xyz, flow_colors, info

    valid = valid[:, : flow_xyz.shape[1]]
    keep = (np.isfinite(valid) & (valid > 0.5)).mean(axis=0) >= 0.5
    if keep.size and valid.shape[0] > 0:
        keep &= valid[0] > 0.5
    min_keep = max(3, int(round(0.15 * flow_xyz.shape[1])))
    info.update(
        {
            "enabled": True,
            "mask_key": mask_key,
            "num_after": int(np.sum(keep)),
            "num_removed": int(flow_xyz.shape[1] - np.sum(keep)),
        }
    )
    if int(np.sum(keep)) < min_keep:
        info["fallback"] = True
        info["num_after"] = int(flow_xyz.shape[1])
        info["num_removed"] = 0
        return flow_xyz, flow_colors, info
    return flow_xyz[:, keep, :], flow_colors[keep], info


def load_track_colors(grp: h5py.Group, start_t: int, num_points: int) -> np.ndarray:
    if "track2d" not in grp or "frames_rgb" not in grp:
        return palette_colors(num_points)

    img = to_rgb_hwc_uint8(grp["frames_rgb"][int(start_t)])
    h, w = img.shape[:2]
    uv = np.asarray(grp["track2d"][int(start_t)], dtype=np.float32)
    if uv.ndim != 2 or uv.shape[0] < num_points or uv.shape[1] < 2:
        return palette_colors(num_points)

    uv = uv[:num_points, :2].copy()
    if np.nanmax(uv) > max(h, w) * 1.2:
        uv[:, 0] *= w / 518.0
        uv[:, 1] *= h / 518.0
    elif np.nanmax(uv) <= 2.0 and np.nanmin(uv) >= -0.5:
        uv[:, 0] *= max(w - 1, 1)
        uv[:, 1] *= max(h - 1, 1)

    x = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, h - 1)
    colors = img[y, x].astype(np.float32) / 255.0
    bad = ~np.isfinite(colors).all(axis=1)
    if np.any(bad):
        colors[bad] = palette_colors(int(np.sum(bad)))
    return colors.astype(np.float32)


def build_flow_colors(grp: h5py.Group, start_t: int, num_points: int, color_mode: str) -> np.ndarray:
    if color_mode == "rgb":
        return load_track_colors(grp, start_t=start_t, num_points=num_points)
    if color_mode == "rainbow":
        return spatial_rainbow_colors(grp, start_t=start_t, num_points=num_points)
    return palette_colors(num_points)


def robust_upper_threshold(values: np.ndarray, factor: float) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("inf")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad > 1e-8:
        return med + float(factor) * 1.4826 * mad
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(q75 - q25)
    if iqr > 1e-8:
        return float(q75) + float(factor) * iqr
    return float(np.max(vals))


def parse_index_list(value: str) -> List[int]:
    if not value:
        return []
    out: List[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def filter_flow_outliers(
    flow_xyz: np.ndarray,
    flow_colors: np.ndarray,
    enabled: bool,
    max_step: float,
    robust_factor: float,
    spatial_k: int,
    spatial_factor: float,
    min_keep_ratio: float,
    drop_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Remove query trajectories with non-finite values or isolated large jumps."""
    xyz = np.asarray(flow_xyz, dtype=np.float32)
    colors = np.asarray(flow_colors, dtype=np.float32)
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "num_before": int(xyz.shape[1]) if xyz.ndim >= 2 else 0,
        "num_after": int(xyz.shape[1]) if xyz.ndim >= 2 else 0,
        "num_removed": 0,
        "max_step_threshold": None,
        "path_threshold": None,
        "spatial_threshold": None,
        "num_spatial_removed": 0,
        "manual_drop_indices": [int(x) for x in drop_indices],
        "fallback": False,
    }
    if xyz.ndim != 3 or xyz.shape[1] == 0:
        return xyz, colors, info

    finite = np.isfinite(xyz).all(axis=(0, 2))
    keep = finite.copy()
    for idx in drop_indices:
        if 0 <= int(idx) < keep.shape[0]:
            keep[int(idx)] = False
    if not enabled:
        info["num_after"] = int(np.sum(keep))
        info["num_removed"] = int(xyz.shape[1] - np.sum(keep))
        info["kept_indices"] = [int(x) for x in np.where(keep)[0].tolist()]
        return xyz[:, keep, :], colors[keep], info
    if xyz.shape[0] > 1:
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        max_step_per_point = np.nanmax(steps, axis=0)
        path_len = np.nansum(steps, axis=0)

        thresholds: List[float] = []
        if max_step > 0:
            thresholds.append(float(max_step))
        if robust_factor > 0 and np.any(finite):
            thresholds.append(robust_upper_threshold(max_step_per_point[finite], robust_factor))
        if thresholds:
            step_thr = float(min(thresholds))
            keep &= max_step_per_point <= step_thr
            info["max_step_threshold"] = step_thr

        if robust_factor > 0 and np.any(finite):
            path_thr = robust_upper_threshold(path_len[finite], robust_factor)
            keep &= path_len <= path_thr
            info["path_threshold"] = float(path_thr)

    if spatial_factor > 0 and xyz.shape[1] >= max(4, int(spatial_k) + 2):
        candidate_idxs = np.where(keep)[0]
        if candidate_idxs.size >= max(4, int(spatial_k) + 2):
            candidate_xyz = xyz[:, candidate_idxs, :]
            k = int(np.clip(spatial_k, 1, max(1, candidate_idxs.size - 1)))
            worst_nn = np.zeros((candidate_idxs.size,), dtype=np.float32)
            for frame_xyz in candidate_xyz:
                dist = np.linalg.norm(frame_xyz[:, None, :] - frame_xyz[None, :, :], axis=-1)
                dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf)
                dist.sort(axis=1)
                nn_mean = dist[:, 1 : k + 1].mean(axis=1)
                worst_nn = np.maximum(worst_nn, nn_mean.astype(np.float32))
            spatial_thr = robust_upper_threshold(worst_nn, spatial_factor)
            spatial_bad = worst_nn > spatial_thr
            if np.any(spatial_bad):
                keep[candidate_idxs[spatial_bad]] = False
                info["num_spatial_removed"] = int(np.sum(spatial_bad))
            info["spatial_threshold"] = float(spatial_thr)

    min_keep = max(3, int(np.ceil(float(min_keep_ratio) * xyz.shape[1])))
    if int(np.sum(keep)) < min_keep:
        keep = finite
        info["fallback"] = True

    if not np.any(keep):
        keep = np.ones((xyz.shape[1],), dtype=bool)
        info["fallback"] = True

    info["num_after"] = int(np.sum(keep))
    info["num_removed"] = int(xyz.shape[1] - np.sum(keep))
    info["kept_indices"] = [int(x) for x in np.where(keep)[0].tolist()]
    return xyz[:, keep, :], colors[keep], info


def robust_flow_translation(ref_xyz: np.ndarray, flow_xyz: np.ndarray, max_offset: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    diff = np.asarray(ref_xyz, dtype=np.float32) - np.asarray(flow_xyz, dtype=np.float32)
    finite = np.isfinite(diff).all(axis=1)
    info: Dict[str, Any] = {
        "num_corr": int(np.sum(finite)),
        "offset_norm": 0.0,
        "skipped": False,
    }
    if int(np.sum(finite)) < 3:
        info["skipped"] = True
        info["reason"] = "too_few_corr"
        return np.zeros((3,), dtype=np.float32), info

    candidates = diff[finite]
    offset = np.nanmedian(candidates, axis=0).astype(np.float32)
    residual = np.linalg.norm(candidates - offset[None, :], axis=1)
    if residual.size >= 8:
        keep = residual <= np.nanpercentile(residual, 80.0)
        if int(np.sum(keep)) >= 3:
            candidates = candidates[keep]
            offset = np.nanmedian(candidates, axis=0).astype(np.float32)
            info["num_corr"] = int(candidates.shape[0])

    norm = float(np.linalg.norm(offset))
    info["offset_norm"] = norm
    if max_offset > 0 and norm > float(max_offset):
        info["skipped"] = True
        info["reason"] = f"offset_norm>{float(max_offset):.4f}"
        return np.zeros((3,), dtype=np.float32), info
    return offset, info


def align_flow_to_reference(
    flow_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    mode: str,
    max_offset: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(mode and mode != "none"),
        "mode": str(mode),
        "max_offset": float(max_offset),
        "skipped": False,
    }
    if not info["enabled"]:
        return flow_xyz, info
    if flow_xyz.shape != ref_xyz.shape:
        info["skipped"] = True
        info["reason"] = f"shape_mismatch:{flow_xyz.shape}!={ref_xyz.shape}"
        return flow_xyz, info

    aligned = np.asarray(flow_xyz, dtype=np.float32).copy()
    k_len = int(aligned.shape[0])
    if mode == "start_translation":
        offset, offset_info = robust_flow_translation(ref_xyz[0], aligned[0], max_offset=max_offset)
        aligned += offset[None, None, :]
        info["start"] = offset_info
        return aligned, info

    if mode == "endpoint_lerp":
        off0, info0 = robust_flow_translation(ref_xyz[0], aligned[0], max_offset=max_offset)
        off1, info1 = robust_flow_translation(ref_xyz[-1], aligned[-1], max_offset=max_offset)
        for k in range(k_len):
            alpha = 0.0 if k_len <= 1 else float(k) / float(k_len - 1)
            aligned[k] += ((1.0 - alpha) * off0 + alpha * off1)[None, :]
        info["start"] = info0
        info["end"] = info1
        return aligned, info

    if mode == "per_frame_translation":
        offsets = []
        skipped = 0
        for k in range(k_len):
            offset, offset_info = robust_flow_translation(ref_xyz[k], aligned[k], max_offset=max_offset)
            aligned[k] += offset[None, :]
            offsets.append(offset)
            skipped += int(bool(offset_info.get("skipped", False)))
        offsets_np = np.stack(offsets, axis=0) if offsets else np.zeros((0, 3), dtype=np.float32)
        norms = np.linalg.norm(offsets_np, axis=1) if offsets_np.size else np.zeros((0,), dtype=np.float32)
        info["num_skipped_frames"] = int(skipped)
        info["offset_norm_median"] = float(np.nanmedian(norms)) if norms.size else 0.0
        info["offset_norm_max"] = float(np.nanmax(norms)) if norms.size else 0.0
        return aligned, info

    info["skipped"] = True
    info["reason"] = f"unknown_mode:{mode}"
    return flow_xyz, info


def palette_colors(num_points: int) -> np.ndarray:
    scalar = np.linspace(0.0, 1.0, num=max(1, int(num_points)), dtype=np.float32)
    return apply_jet_colormap(scalar)[: int(num_points)]


def spatial_rainbow_colors(grp: h5py.Group, start_t: int, num_points: int) -> np.ndarray:
    scalar = read_initial_y_scalar(grp, start_t=start_t, num_points=num_points)
    if scalar is None:
        scalar = np.linspace(0.0, 1.0, num=max(1, int(num_points)), dtype=np.float32)
    else:
        scalar = normalize_scalar(scalar)
    return apply_jet_colormap(scalar)


def read_initial_y_scalar(grp: h5py.Group, start_t: int, num_points: int) -> Optional[np.ndarray]:
    candidates: List[np.ndarray] = []
    if "track2d" in grp:
        arr = np.asarray(grp["track2d"][int(start_t)], dtype=np.float32)
        candidates.append(arr)
    for key in ("query_xy_t0", "p0_uv", "grid_points_xy"):
        if key in grp:
            arr = np.asarray(grp[key], dtype=np.float32)
            if arr.ndim >= 3:
                arr = arr[int(np.clip(start_t, 0, arr.shape[0] - 1))]
            candidates.append(arr)

    for arr in candidates:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= num_points and arr.shape[1] >= 2:
            y = arr[:num_points, 1].astype(np.float32)
            if np.isfinite(y).any() and float(np.nanmax(y) - np.nanmin(y)) > 1e-6:
                return y
    return None


def normalize_scalar(scalar: np.ndarray) -> np.ndarray:
    x = np.asarray(scalar, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.nanpercentile(x[finite], 2.0))
    hi = float(np.nanpercentile(x[finite], 98.0))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1e-6
    out = (np.nan_to_num(x, nan=lo) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_jet_colormap(scalar: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(scalar, dtype=np.float32), 0.0, 1.0).reshape(-1)
    colors = np.zeros((u.shape[0], 3), dtype=np.float32)
    for i, val in enumerate(u):
        for j in range(len(JET_STOPS) - 1):
            a_u, a_c = JET_STOPS[j]
            b_u, b_c = JET_STOPS[j + 1]
            if val <= b_u:
                r = (float(val) - a_u) / max(1e-6, b_u - a_u)
                colors[i] = np.asarray(a_c, dtype=np.float32) * (1.0 - r) + np.asarray(b_c, dtype=np.float32) * r
                break
        else:
            colors[i] = np.asarray(JET_STOPS[-1][1], dtype=np.float32)
    return colors

def read_matrix_frame(grp: h5py.Group, keys: Sequence[str], frame_idx: int) -> Optional[np.ndarray]:
    key = first_existing_key(grp, keys)
    if key is None:
        return None
    arr = np.asarray(grp[key])
    if arr.ndim >= 3:
        arr = arr[int(np.clip(frame_idx, 0, arr.shape[0] - 1))]
    return np.asarray(arr, dtype=np.float32)


def scene_keysets(scene_coord_mode: str) -> Tuple[Sequence[str], Sequence[str], Sequence[str], bool]:
    mode = str(scene_coord_mode).lower()
    if mode == "metric":
        return METRIC_INTRINSICS_KEYS, METRIC_CAM_TO_SCENE_KEYS, (), False
    if mode == "spatracker":
        # Processed SpaTracker files save VGGT poses under c2w-style names when available.
        # If no such pose exists, falling back to camera-frame depth would misalign with
        # point_traj_spatracker, so require a pose and use sparse flow points instead.
        return SPATRACKER_INTRINSICS_KEYS, SPATRACKER_CAM_TO_SCENE_KEYS, SPATRACKER_WORLD_TO_CAM_KEYS, True
    if mode == "camera":
        return INTRINSICS_KEYS, (), (), False
    return INTRINSICS_KEYS, CAM_TO_SCENE_KEYS, WORLD_TO_CAM_KEYS, False


def read_cam_to_scene_frame(
    grp: h5py.Group,
    frame_idx: int,
    scene_coord_mode: str,
) -> Tuple[Optional[np.ndarray], str]:
    _, cam_to_scene_keys, world_to_cam_keys, require_pose = scene_keysets(scene_coord_mode)
    cam_to_scene = read_matrix_frame(grp, cam_to_scene_keys, frame_idx) if cam_to_scene_keys else None
    pose_key = first_existing_key(grp, cam_to_scene_keys) if cam_to_scene_keys else ""
    if cam_to_scene is None and world_to_cam_keys:
        E = read_matrix_frame(grp, world_to_cam_keys, frame_idx)
        pose_key = first_existing_key(grp, world_to_cam_keys) or ""
        if E is not None and E.shape[0] >= 4 and E.shape[1] >= 4:
            try:
                cam_to_scene = np.linalg.inv(E[:4, :4].astype(np.float32))
                pose_key = f"inverse({pose_key})"
            except np.linalg.LinAlgError:
                cam_to_scene = None
    if cam_to_scene is None and require_pose:
        return None, ""
    return cam_to_scene, pose_key or ""


def read_depth_frame(grp: h5py.Group, frame_idx: int, depth_key: str) -> Optional[np.ndarray]:
    if depth_key == "auto":
        depth_key = first_existing_key(grp, DEPTH_KEYS) or ""
    if not depth_key or depth_key not in grp:
        return None

    depth = np.asarray(grp[depth_key][int(frame_idx)], dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        return None
    return depth


def build_scene_from_depth(
    grp: h5py.Group,
    frame_idx: int,
    depth_key: str,
    scene_stride: int,
    max_scene_points: int,
    max_depth: float,
    scene_coord_mode: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    depth = read_depth_frame(grp, frame_idx, depth_key)
    if depth is None:
        return None, None, ""

    intrinsics_keys, _, _, require_pose = scene_keysets(scene_coord_mode)
    K = read_matrix_frame(grp, intrinsics_keys, frame_idx)
    if K is None or K.shape[0] < 3 or K.shape[1] < 3:
        return None, None, ""
    K = K[:3, :3].astype(np.float32)

    img = None
    if "frames_rgb" in grp:
        img = to_rgb_hwc_uint8(grp["frames_rgb"][int(frame_idx)])

    h, w = depth.shape
    stride = max(1, int(scene_stride))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx].reshape(-1)
    x_pix = xx.reshape(-1).astype(np.float32)
    y_pix = yy.reshape(-1).astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    if max_depth > 0:
        valid &= z <= float(max_depth)
    if np.any(valid):
        z_valid = z[valid]
        hi = np.nanpercentile(z_valid, 99.5)
        valid &= z <= float(hi)
    if not np.any(valid):
        return None, None, ""

    z = z[valid]
    x_pix = x_pix[valid]
    y_pix = y_pix[valid]

    if max_scene_points > 0 and z.shape[0] > int(max_scene_points):
        sel = np.linspace(0, z.shape[0] - 1, num=int(max_scene_points)).round().astype(np.int64)
        z = z[sel]
        x_pix = x_pix[sel]
        y_pix = y_pix[sel]

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return None, None, ""

    points = np.stack([(x_pix - cx) * z / fx, (y_pix - cy) * z / fy, z], axis=1).astype(np.float32)

    colors = np.full((points.shape[0], 3), 0.62, dtype=np.float32)
    if img is not None:
        ih, iw = img.shape[:2]
        sx = iw / float(w)
        sy = ih / float(h)
        xi = np.clip(np.round(x_pix * sx).astype(np.int64), 0, iw - 1)
        yi = np.clip(np.round(y_pix * sy).astype(np.int64), 0, ih - 1)
        colors = img[yi, xi].astype(np.float32) / 255.0

    cam_to_scene, _ = read_cam_to_scene_frame(grp, frame_idx, scene_coord_mode)
    if cam_to_scene is None and require_pose:
        return None, None, ""

    if cam_to_scene is not None and cam_to_scene.shape[0] >= 4 and cam_to_scene.shape[1] >= 4:
        cam_to_scene = cam_to_scene[:4, :4].astype(np.float32)
        try:
            homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
            points = (cam_to_scene @ homo.T).T[:, :3].astype(np.float32)
        except np.linalg.LinAlgError:
            pass

    return points, colors, depth_key


def build_sparse_scene(flow_xyz: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = flow_xyz[0].astype(np.float32)
    scene_colors = np.clip(colors.astype(np.float32) * 0.45 + 0.28, 0.0, 1.0)
    return points, scene_colors


def estimate_camera_depth_for_rgb_plane(
    reference_points: np.ndarray,
    cam_to_scene: Optional[np.ndarray],
    default_depth: float = 1.0,
) -> float:
    ref = np.asarray(reference_points, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[-1] < 3 or ref.shape[0] == 0:
        return float(default_depth)
    ref = ref[:, :3]
    finite = np.isfinite(ref).all(axis=1)
    if not np.any(finite):
        return float(default_depth)

    points = ref[finite]
    if cam_to_scene is not None and cam_to_scene.shape[0] >= 4 and cam_to_scene.shape[1] >= 4:
        try:
            scene_to_cam = np.linalg.inv(cam_to_scene[:4, :4].astype(np.float32))
            homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
            points_cam = (scene_to_cam @ homo.T).T[:, :3]
            z = points_cam[:, 2]
        except np.linalg.LinAlgError:
            z = points[:, 2]
    else:
        z = points[:, 2]

    z = z[np.isfinite(z) & (z > 1e-4)]
    if z.size == 0:
        return float(default_depth)
    return float(np.clip(np.nanmedian(z), 0.25, 3.0))


def build_scene_from_rgb_plane(
    grp: h5py.Group,
    frame_idx: int,
    scene_stride: int,
    max_scene_points: int,
    scene_coord_mode: str,
    reference_points: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Fallback scene for real videos that have RGB and camera poses but no depth.

    This is a visualization-only textured plane at the approximate flow depth,
    not a metric dense reconstruction. It gives real/DROID examples visible
    scene context when no depth map is available.
    """
    if "frames_rgb" not in grp:
        return None, None, ""

    img = to_rgb_hwc_uint8(grp["frames_rgb"][int(frame_idx)])
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None, None, ""

    intrinsics_keys, _, _, require_pose = scene_keysets(scene_coord_mode)
    K = read_matrix_frame(grp, intrinsics_keys, frame_idx)
    if K is None or K.shape[0] < 3 or K.shape[1] < 3:
        return None, None, ""
    K = K[:3, :3].astype(np.float32)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return None, None, ""

    cam_to_scene, _ = read_cam_to_scene_frame(grp, frame_idx, scene_coord_mode)
    if cam_to_scene is None and require_pose:
        return None, None, ""

    stride = max(1, int(scene_stride))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    x_pix = xx.reshape(-1).astype(np.float32)
    y_pix = yy.reshape(-1).astype(np.float32)
    if max_scene_points > 0 and x_pix.shape[0] > int(max_scene_points):
        sel = np.linspace(0, x_pix.shape[0] - 1, num=int(max_scene_points)).round().astype(np.int64)
        x_pix = x_pix[sel]
        y_pix = y_pix[sel]

    z = np.full_like(x_pix, estimate_camera_depth_for_rgb_plane(reference_points, cam_to_scene))
    points = np.stack([(x_pix - cx) * z / fx, (y_pix - cy) * z / fy, z], axis=1).astype(np.float32)
    xi = np.clip(np.round(x_pix).astype(np.int64), 0, w - 1)
    yi = np.clip(np.round(y_pix).astype(np.int64), 0, h - 1)
    colors = img[yi, xi].astype(np.float32) / 255.0

    if cam_to_scene is not None and cam_to_scene.shape[0] >= 4 and cam_to_scene.shape[1] >= 4:
        homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        points = (cam_to_scene[:4, :4].astype(np.float32) @ homo.T).T[:, :3].astype(np.float32)

    return points, colors.astype(np.float32), "rgb_plane"


def build_scene_sequence_from_depth(
    grp: h5py.Group,
    key_idxs: np.ndarray,
    depth_key: str,
    scene_stride: int,
    max_scene_points: int,
    max_depth: float,
    scene_coord_mode: str,
    fallback_points: np.ndarray,
    fallback_colors: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    scene_frames: List[np.ndarray] = []
    color_frames: List[np.ndarray] = []
    used_depth_key = ""
    last_points = np.asarray(fallback_points, dtype=np.float32)
    last_colors = np.asarray(fallback_colors, dtype=np.float32)

    for idx in np.asarray(key_idxs, dtype=np.int64).tolist():
        points, colors, this_depth_key = build_scene_from_depth(
            grp=grp,
            frame_idx=int(idx),
            depth_key=depth_key,
            scene_stride=scene_stride,
            max_scene_points=max_scene_points,
            max_depth=max_depth,
            scene_coord_mode=scene_coord_mode,
        )
        if points is None or colors is None:
            points, colors = last_points, last_colors
        else:
            points = np.asarray(points, dtype=np.float32)
            colors = np.asarray(colors, dtype=np.float32)
            last_points, last_colors = points, colors
            used_depth_key = this_depth_key or used_depth_key

        scene_frames.append(points)
        color_frames.append(colors)

    if not used_depth_key:
        used_depth_key = depth_key if depth_key != "auto" else ""
    return scene_frames, color_frames, used_depth_key


def read_camera_view(grp: h5py.Group, frame_idx: int, scene_coord_mode: str) -> Optional[Dict[str, Any]]:
    cam_to_scene, camera_key = read_cam_to_scene_frame(grp, frame_idx, scene_coord_mode)

    if cam_to_scene is None or cam_to_scene.shape[0] < 4 or cam_to_scene.shape[1] < 4:
        return None

    T = np.asarray(cam_to_scene[:4, :4], dtype=np.float32)
    origin = T[:3, 3]
    forward = T[:3, 2]
    up = -T[:3, 1]
    if np.linalg.norm(forward) < 1e-6 or np.linalg.norm(up) < 1e-6:
        return None
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)

    fov_deg = 55.0
    K = read_matrix_frame(grp, INTRINSICS_KEYS, frame_idx)
    img_h = None
    if "frames_rgb" in grp:
        try:
            img_h = int(to_rgb_hwc_uint8(grp["frames_rgb"][int(frame_idx)]).shape[0])
        except Exception:
            img_h = None
    if img_h is None:
        depth = read_depth_frame(grp, frame_idx, "auto")
        if depth is not None:
            img_h = int(depth.shape[0])
    if K is not None and K.shape[0] >= 3 and K.shape[1] >= 3 and img_h is not None:
        fy = float(K[1, 1])
        if abs(fy) > 1e-6:
            fov_deg = float(np.degrees(2.0 * np.arctan(0.5 * float(img_h) / fy)))

    return {
        "position": list3(origin, ndigits=6),
        "target": list3(origin + forward, ndigits=6),
        "up": list3(up, ndigits=6),
        "fov": fov_deg,
        "pose_key": camera_key or "",
    }


def list3(arr: np.ndarray, ndigits: int = 5) -> List[Any]:
    return np.round(np.asarray(arr, dtype=np.float32), ndigits).tolist()


def make_stage_payload(
    title: str,
    flow_xyz: np.ndarray,
    flow_colors: np.ndarray,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    scene_point_frames: Optional[Sequence[np.ndarray]],
    scene_color_frames: Optional[Sequence[np.ndarray]],
    meta: Dict[str, Any],
    fps: float,
    point_size: float,
    scene_point_size: float,
    line_opacity: float,
    flip_yz: bool,
    screen_space_points: bool,
    flow_color_mode: str,
    hide_flow_points: bool,
) -> Dict[str, Any]:
    return {
        "title": title,
        "flow_points": list3(flow_xyz),
        "flow_colors": list3(flow_colors),
        "scene_points": list3(scene_points),
        "scene_colors": list3(scene_colors),
        "scene_point_frames": [list3(x) for x in scene_point_frames] if scene_point_frames is not None else None,
        "scene_color_frames": [list3(x) for x in scene_color_frames] if scene_color_frames is not None else None,
        "meta": meta,
        "settings": {
            "fps": float(fps),
            "pointSize": float(point_size),
            "scenePointSize": float(scene_point_size),
            "lineOpacity": float(line_opacity),
            "flipYZ": bool(flip_yz),
            "screenSpacePoints": bool(screen_space_points),
            "flowColorMode": str(flow_color_mode),
            "hideFlowPoints": bool(hide_flow_points),
        },
    }


def write_staged_interactive_html(out_path: Path, stages: Sequence[Dict[str, Any]]) -> None:
    if len(stages) == 0:
        raise ValueError("staged HTML needs at least one stage payload")
    data_json = json.dumps(list(stages), ensure_ascii=True, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__STAGES_JSON__", data_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def write_interactive_html(
    out_path: Path,
    title: str,
    flow_xyz: np.ndarray,
    flow_colors: np.ndarray,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    scene_point_frames: Optional[Sequence[np.ndarray]],
    scene_color_frames: Optional[Sequence[np.ndarray]],
    meta: Dict[str, Any],
    fps: float,
    point_size: float,
    scene_point_size: float,
    line_opacity: float,
    flip_yz: bool,
    screen_space_points: bool,
    flow_color_mode: str,
    hide_flow_points: bool,
) -> Dict[str, Any]:
    payload = make_stage_payload(
        title=title,
        flow_xyz=flow_xyz,
        flow_colors=flow_colors,
        scene_points=scene_points,
        scene_colors=scene_colors,
        scene_point_frames=scene_point_frames,
        scene_color_frames=scene_color_frames,
        meta=meta,
        fps=fps,
        point_size=point_size,
        scene_point_size=scene_point_size,
        line_opacity=line_opacity,
        flip_yz=flip_yz,
        screen_space_points=screen_space_points,
        flow_color_mode=flow_color_mode,
        hide_flow_points=hide_flow_points,
    )
    write_staged_interactive_html(out_path, [payload])
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D Flow</title>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #f7f7f4;
      color: #1d2228;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body.bg-dark {
      background: #101113;
      color: #eef1f4;
    }
    body.bg-transparent {
      background: transparent;
    }
    #view {
      position: fixed;
      inset: 0;
    }
    .hud {
      position: fixed;
      left: 14px;
      top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      max-width: min(980px, calc(100vw - 28px));
      padding: 8px;
      border: 1px solid rgba(24, 30, 36, 0.16);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.84);
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 28px rgba(0,0,0,0.12);
      z-index: 2;
    }
    body.bg-dark .hud {
      border-color: rgba(255,255,255,0.13);
      background: rgba(18, 20, 24, 0.82);
      box-shadow: 0 10px 25px rgba(0,0,0,0.24);
    }
    .hud button, .hud select {
      height: 28px;
      border: 1px solid rgba(24, 30, 36, 0.18);
      border-radius: 6px;
      background: #ffffff;
      color: #1d2228;
      cursor: pointer;
      font-size: 12px;
    }
    .hud button {
      min-width: 58px;
      padding: 0 9px;
    }
    .hud select {
      max-width: 240px;
      padding: 0 7px;
    }
    body.bg-dark .hud button, body.bg-dark .hud select {
      border-color: rgba(255,255,255,0.18);
      background: #252a30;
      color: #eef1f4;
    }
    .hud button:hover, .hud select:hover {
      border-color: rgba(53, 194, 166, 0.7);
    }
    .hud input[type="range"] {
      width: min(30vw, 270px);
      accent-color: #35c2a6;
    }
    .ctrl {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      white-space: nowrap;
      color: #4b5560;
    }
    body.bg-dark .ctrl {
      color: #cfd6de;
    }
    .ctrl input[type="range"] {
      width: 92px;
    }
    .readout {
      min-width: 126px;
      font-size: 12px;
      color: #4b5560;
      text-align: right;
    }
    body.bg-dark .readout {
      color: #cfd6de;
    }
    .badge {
      position: fixed;
      right: 14px;
      bottom: 14px;
      max-width: min(70vw, 720px);
      padding: 7px 9px;
      border-radius: 8px;
      border: 1px solid rgba(24, 30, 36, 0.14);
      background: rgba(255, 255, 255, 0.72);
      color: #4b5560;
      font-size: 12px;
      z-index: 2;
    }
    body.bg-dark .badge {
      border-color: rgba(255,255,255,0.12);
      background: rgba(18, 20, 24, 0.72);
      color: #cfd6de;
    }
  </style>
</head>
<body>
  <div id="view"></div>
  <div class="hud">
    <select id="stage"></select>
    <button id="play">Pause</button>
    <button id="cameraView">Camera</button>
    <button id="fitView">Fit</button>
    <select id="background">
      <option value="white" selected>White</option>
      <option value="transparent">Transparent</option>
      <option value="dark">Dark</option>
    </select>
    <label class="ctrl">Flow <input id="flowSize" type="range" min="2" max="24" value="9" step="0.5"></label>
    <label class="ctrl">Scene <input id="sceneSize" type="range" min="0.5" max="14" value="5" step="0.1"></label>
    <label class="ctrl">Trail <input id="lineAlpha" type="range" min="0.05" max="1" value="0.78" step="0.01"></label>
    <label class="ctrl">Scene alpha <input id="sceneAlpha" type="range" min="0.05" max="1" value="0.92" step="0.01"></label>
    <input id="timeline" type="range" min="0" max="1" value="0" step="1">
    <div id="readout" class="readout"></div>
    <div id="pickReadout" class="readout"></div>
  </div>
  <div id="badge" class="badge"></div>

  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
  <script>
    const STAGES = __STAGES_JSON__;
    const view = document.getElementById("view");
    const stageSelect = document.getElementById("stage");
    const playButton = document.getElementById("play");
    const cameraViewButton = document.getElementById("cameraView");
    const fitViewButton = document.getElementById("fitView");
    const backgroundSelect = document.getElementById("background");
    const flowSizeInput = document.getElementById("flowSize");
    const sceneSizeInput = document.getElementById("sceneSize");
    const lineAlphaInput = document.getElementById("lineAlpha");
    const sceneAlphaInput = document.getElementById("sceneAlpha");
    const timeline = document.getElementById("timeline");
    const readout = document.getElementById("readout");
    const pickReadout = document.getElementById("pickReadout");
    const badge = document.getElementById("badge");

    let DATA = STAGES[0];
    let flow = [];
    let flowColors = [];
    let scenePoints = [];
    let sceneColors = [];
    let scenePointFrames = [];
    let sceneColorFrames = [];
    let K = 0;
    let N = 0;
    let totalFrames = 2;
    let phaseSplit = 1;
    let segmentCount = 0;
    let currentMarkerPoints = [];
    let currentMarkerStep = 0;
    let activeSceneFrame = -1;
    let currentFrame = 0;
    let isPlaying = true;
    let lastTime = performance.now();
    let frameCarry = 0;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    view.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.screenSpacePanning = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.78));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.72);
    keyLight.position.set(3, 4, 5);
    scene.add(keyLight);

    const sceneMaterial = new THREE.PointsMaterial({
      size: 5.0,
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      sizeAttenuation: false
    });
    const markerMaterial = new THREE.PointsMaterial({
      size: 9.0,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      sizeAttenuation: false
    });
    const goalMaterial = new THREE.PointsMaterial({
      size: 7.65,
      vertexColors: true,
      transparent: true,
      opacity: 0.86,
      sizeAttenuation: false
    });
    const lineMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.78
    });
    const centroidMaterial = new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.92 });

    const sceneCloud = new THREE.Points(new THREE.BufferGeometry(), sceneMaterial);
    const markers = new THREE.Points(new THREE.BufferGeometry(), markerMaterial);
    const goalCloud = new THREE.Points(new THREE.BufferGeometry(), goalMaterial);
    const trailLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial);
    trailLines.frustumCulled = false;
    const centroidMesh = new THREE.Mesh(new THREE.SphereGeometry(0.012, 24, 24), centroidMaterial);
    scene.add(sceneCloud);
    scene.add(trailLines);
    scene.add(goalCloud);
    scene.add(markers);
    scene.add(centroidMesh);

    for (let i = 0; i < STAGES.length; i++) {
      const opt = document.createElement("option");
      opt.value = String(i);
      const s = STAGES[i];
      opt.textContent = s.stage_label || `stage ${i}: t${s.meta.start_t} -> t${s.meta.goal_t}`;
      stageSelect.appendChild(opt);
    }

    function currentSettings() {
      return DATA.settings || {};
    }

    function tx(p) {
      const settings = currentSettings();
      return settings.flipYZ ? [p[0], -p[1], -p[2]] : [p[0], p[1], p[2]];
    }

    function txVector(p) {
      return tx(p);
    }

    function asVec3(p) {
      const q = tx(p);
      return new THREE.Vector3(q[0], q[1], q[2]);
    }

    function flatPoints(points) {
      const out = new Float32Array(points.length * 3);
      for (let i = 0; i < points.length; i++) {
        const p = tx(points[i]);
        out[i * 3] = p[0];
        out[i * 3 + 1] = p[1];
        out[i * 3 + 2] = p[2];
      }
      return out;
    }

    function flatColors(colors) {
      const out = new Float32Array(colors.length * 3);
      for (let i = 0; i < colors.length; i++) {
        out[i * 3] = colors[i][0];
        out[i * 3 + 1] = colors[i][1];
        out[i * 3 + 2] = colors[i][2];
      }
      return out;
    }

    const jetStops = [
      [0.00, [0.03, 0.18, 0.95]],
      [0.18, [0.00, 0.66, 1.00]],
      [0.38, [0.00, 0.92, 0.40]],
      [0.58, [0.72, 0.96, 0.10]],
      [0.76, [1.00, 0.78, 0.05]],
      [0.90, [1.00, 0.34, 0.03]],
      [1.00, [0.86, 0.02, 0.02]]
    ];

    function rainbowColor(u) {
      const t = Math.max(0, Math.min(1, Number.isFinite(u) ? u : 0));
      for (let i = 0; i < jetStops.length - 1; i++) {
        const a = jetStops[i];
        const b = jetStops[i + 1];
        if (t >= a[0] && t <= b[0]) {
          const r = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
          return [
            a[1][0] * (1 - r) + b[1][0] * r,
            a[1][1] * (1 - r) + b[1][1] * r,
            a[1][2] * (1 - r) + b[1][2] * r
          ];
        }
      }
      return jetStops[jetStops.length - 1][1];
    }

    function repeatedColor(color, n) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        out[i * 3] = color[0];
        out[i * 3 + 1] = color[1];
        out[i * 3 + 2] = color[2];
      }
      return out;
    }

    function pointColor(i) {
      const c = flowColors && flowColors[i] ? flowColors[i] : null;
      if (c && c.length >= 3) return c;
      return rainbowColor(i / Math.max(1, N - 1));
    }

    function indexedPointColors(n) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const c = pointColor(i);
        out[i * 3] = c[0];
        out[i * 3 + 1] = c[1];
        out[i * 3 + 2] = c[2];
      }
      return out;
    }

    function markerColorsForStep(n, stepIndex) {
      const mode = (currentSettings().flowColorMode || "rainbow").toLowerCase();
      if (mode === "time") {
        return repeatedColor(rainbowColor(stepIndex / Math.max(1, K - 1)), n);
      }
      return indexedPointColors(n);
    }

    function disposeGeometry(obj) {
      if (obj.geometry) obj.geometry.dispose();
    }

    function replacePointsGeometry(obj, positions, colors) {
      disposeGeometry(obj);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
      obj.geometry = g;
    }

    function buildLineGeometry() {
      segmentCount = Math.max(0, K - 1) * N;
      const linePositions = new Float32Array(segmentCount * 2 * 3);
      const lineColors = new Float32Array(segmentCount * 2 * 3);
      let cursor = 0;
      const mode = (currentSettings().flowColorMode || "rainbow").toLowerCase();
      for (let t = 0; t < K - 1; t++) {
        const timeC0 = rainbowColor(t / Math.max(1, K - 1));
        const timeC1 = rainbowColor((t + 1) / Math.max(1, K - 1));
        for (let i = 0; i < N; i++) {
          const trackColor = pointColor(i);
          const c0 = mode === "time" ? timeC0 : trackColor;
          const c1 = mode === "time" ? timeC1 : trackColor;
          const p0 = tx(flow[t][i]);
          const p1 = tx(flow[t + 1][i]);
          const base = cursor * 6;
          linePositions[base] = p0[0];
          linePositions[base + 1] = p0[1];
          linePositions[base + 2] = p0[2];
          linePositions[base + 3] = p1[0];
          linePositions[base + 4] = p1[1];
          linePositions[base + 5] = p1[2];
          lineColors[base] = c0[0];
          lineColors[base + 1] = c0[1];
          lineColors[base + 2] = c0[2];
          lineColors[base + 3] = c1[0];
          lineColors[base + 4] = c1[1];
          lineColors[base + 5] = c1[2];
          cursor++;
        }
      }
      disposeGeometry(trailLines);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      g.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
      trailLines.geometry = g;
    }

    function centroid(points) {
      const c = new THREE.Vector3();
      for (const pRaw of points) {
        const p = tx(pRaw);
        c.x += p[0];
        c.y += p[1];
        c.z += p[2];
      }
      c.multiplyScalar(1 / Math.max(1, points.length));
      return c;
    }

    function setMarkerPositions(points, stepIndex) {
      currentMarkerPoints = points || [];
      currentMarkerStep = stepIndex;
      const pos = markers.geometry.attributes.position.array;
      for (let i = 0; i < points.length; i++) {
        const p = tx(points[i]);
        pos[i * 3] = p[0];
        pos[i * 3 + 1] = p[1];
        pos[i * 3 + 2] = p[2];
      }
      markers.geometry.setAttribute("color", new THREE.BufferAttribute(markerColorsForStep(points.length, stepIndex), 3));
      markers.geometry.attributes.position.needsUpdate = true;
      markers.geometry.computeBoundingSphere();
      centroidMesh.position.copy(centroid(points));
    }

    function currentFlowIndices() {
      const ff = (DATA.meta && DATA.meta.flow_filter) ? DATA.meta.flow_filter : {};
      const kept = Array.isArray(ff.kept_indices) ? ff.kept_indices : null;
      if (kept && kept.length === N) return kept;
      return Array.from({ length: N }, (_, i) => i);
    }

    function pickNearestFlowPoint(clientX, clientY) {
      if (!currentMarkerPoints || currentMarkerPoints.length === 0) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const mouseX = clientX - rect.left;
      const mouseY = clientY - rect.top;
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);
      let bestIdx = -1;
      let bestDist = Infinity;
      const v = new THREE.Vector3();
      for (let i = 0; i < currentMarkerPoints.length; i++) {
        const p = tx(currentMarkerPoints[i]);
        v.set(p[0], p[1], p[2]).project(camera);
        if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) continue;
        const sx = (v.x * 0.5 + 0.5) * width;
        const sy = (-v.y * 0.5 + 0.5) * height;
        const d = Math.hypot(mouseX - sx, mouseY - sy);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) return;
      const origIdx = currentFlowIndices()[bestIdx];
      const pRaw = currentMarkerPoints[bestIdx];
      pickReadout.textContent = `picked display_idx=${bestIdx} orig_idx=${origIdx} step=${currentMarkerStep + 1}/${K} dist=${bestDist.toFixed(1)}px xyz=${pRaw.map(x => Number(x).toFixed(4)).join(",")}`;
    }

    function percentile(sortedValues, q) {
      if (sortedValues.length === 0) return 0;
      const pos = (sortedValues.length - 1) * q;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      const t = pos - lo;
      return sortedValues[lo] * (1 - t) + sortedValues[hi] * t;
    }

    function robustBoxFromData() {
      const xs = [];
      const ys = [];
      const zs = [];
      function addPoint(pRaw) {
        const p = tx(pRaw);
        if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) return;
        xs.push(p[0]);
        ys.push(p[1]);
        zs.push(p[2]);
      }
      for (const p of scenePoints) addPoint(p);
      for (const frame of flow) {
        for (const p of frame) addPoint(p);
      }
      if (xs.length === 0) return null;
      xs.sort((a, b) => a - b);
      ys.sort((a, b) => a - b);
      zs.sort((a, b) => a - b);
      const loQ = 0.01;
      const hiQ = 0.99;
      const mins = [percentile(xs, loQ), percentile(ys, loQ), percentile(zs, loQ)];
      const maxs = [percentile(xs, hiQ), percentile(ys, hiQ), percentile(zs, hiQ)];
      const center = new THREE.Vector3(
        0.5 * (mins[0] + maxs[0]),
        0.5 * (mins[1] + maxs[1]),
        0.5 * (mins[2] + maxs[2])
      );
      const size = new THREE.Vector3(
        Math.max(0.01, (maxs[0] - mins[0]) * 1.12),
        Math.max(0.01, (maxs[1] - mins[1]) * 1.12),
        Math.max(0.01, (maxs[2] - mins[2]) * 1.12)
      );
      return { center, size };
    }

    function updateCameraClipping() {
      const robust = robustBoxFromData();
      const radius = robust ? Math.max(robust.size.x, robust.size.y, robust.size.z, 0.01) : 2.0;
      camera.near = Math.max(radius / 2000, 0.0001);
      camera.far = Math.max(radius * 200, 10);
      camera.updateProjectionMatrix();
    }

    function fitCamera() {
      const robust = robustBoxFromData();
      const box = new THREE.Box3();
      if (robust) {
        box.setFromCenterAndSize(robust.center, robust.size);
      } else {
        box.expandByObject(sceneCloud);
        box.expandByObject(markers);
        box.expandByObject(goalCloud);
      }
      if (box.isEmpty()) {
        camera.position.set(0, 0, 2);
        controls.target.set(0, 0, 0);
        controls.update();
        return;
      }
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      const radius = Math.max(size.x, size.y, size.z, 0.01);
      controls.target.copy(center);
      camera.up.set(0, 1, 0);
      camera.fov = 55;
      camera.position.set(center.x + radius * 0.75, center.y + radius * 0.45, center.z + radius * 1.35);
      camera.near = Math.max(radius / 1000, 0.0001);
      camera.far = Math.max(radius * 100, 10);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function applySourceCamera() {
      const cam = DATA.meta ? DATA.meta.camera_view : null;
      if (!cam || !cam.position || !cam.target) {
        fitCamera();
        return;
      }
      const pos = asVec3(cam.position);
      const target = asVec3(cam.target);
      camera.position.copy(pos);
      controls.target.copy(target);
      if (cam.up) {
        const up = txVector(cam.up);
        camera.up.set(up[0], up[1], up[2]).normalize();
      }
      camera.fov = Number.isFinite(cam.fov) ? cam.fov : 55;
      camera.aspect = window.innerWidth / window.innerHeight;
      updateCameraClipping();
      controls.update();
    }

    function applyBackground(mode) {
      document.body.classList.toggle("bg-dark", mode === "dark");
      document.body.classList.toggle("bg-transparent", mode === "transparent");
      if (mode === "dark") {
        scene.background = new THREE.Color(0x101113);
        renderer.setClearColor(0x101113, 1);
        centroidMaterial.color.setHex(0xffffff);
      } else if (mode === "transparent") {
        scene.background = null;
        renderer.setClearColor(0xffffff, 0);
        centroidMaterial.color.setHex(0x111111);
      } else {
        scene.background = new THREE.Color(0xf7f7f4);
        renderer.setClearColor(0xf7f7f4, 1);
        centroidMaterial.color.setHex(0x111111);
      }
    }

    function applyControlValues() {
      const settings = currentSettings();
      const screenSpacePoints = settings.screenSpacePoints !== false;
      markerMaterial.size = parseFloat(flowSizeInput.value);
      goalMaterial.size = markerMaterial.size * 0.85;
      sceneMaterial.size = parseFloat(sceneSizeInput.value);
      lineMaterial.opacity = parseFloat(lineAlphaInput.value);
      sceneMaterial.opacity = parseFloat(sceneAlphaInput.value);
      markerMaterial.sizeAttenuation = !screenSpacePoints;
      goalMaterial.sizeAttenuation = !screenSpacePoints;
      sceneMaterial.sizeAttenuation = !screenSpacePoints;
      centroidMesh.scale.setScalar(Math.max(0.5, markerMaterial.size / 9.0));
      applyBackground(backgroundSelect.value);
    }

    function setControlDefaultsFromStage() {
      const settings = currentSettings();
      const screenSpacePoints = settings.screenSpacePoints !== false;
      const flowSize = screenSpacePoints ? (settings.pointSize > 0.1 ? settings.pointSize : 9.0) : settings.pointSize;
      const sceneSize = screenSpacePoints ? (settings.scenePointSize > 0.1 ? settings.scenePointSize : 5.0) : settings.scenePointSize;
      flowSizeInput.value = String(flowSize);
      sceneSizeInput.value = String(sceneSize);
      lineAlphaInput.value = String(settings.lineOpacity ?? 0.78);
      sceneAlphaInput.value = "0.92";
      applyControlValues();
    }

    function rebuildStageGeometry() {
      flow = DATA.flow_points || [];
      flowColors = DATA.flow_colors || [];
      scenePoints = DATA.scene_points || [];
      sceneColors = DATA.scene_colors || [];
      scenePointFrames = (DATA.scene_point_frames && DATA.scene_point_frames.length) ? DATA.scene_point_frames : [scenePoints];
      sceneColorFrames = (DATA.scene_color_frames && DATA.scene_color_frames.length) ? DATA.scene_color_frames : [sceneColors];
      K = flow.length;
      N = K > 0 ? flow[0].length : 0;
      totalFrames = Math.max(2, K * 2);
      phaseSplit = Math.max(1, K);
      timeline.max = String(totalFrames - 1);
      activeSceneFrame = -1;

      setSceneFrame(0);
      replacePointsGeometry(markers, new Float32Array(N * 3), markerColorsForStep(N, 0));
      replacePointsGeometry(goalCloud, flatPoints(K > 0 ? flow[K - 1] : []), markerColorsForStep(N, K - 1));
      buildLineGeometry();
      update(0);
    }

    function setSceneFrame(frameIdx) {
      const idx = Math.max(0, Math.min(scenePointFrames.length - 1, frameIdx));
      if (idx === activeSceneFrame) return;
      activeSceneFrame = idx;
      const pts = scenePointFrames[idx] || scenePoints;
      const cols = sceneColorFrames[idx] || sceneColors;
      scenePoints = pts;
      sceneColors = cols;
      replacePointsGeometry(sceneCloud, flatPoints(pts), flatColors(cols));
    }

    function update(frame) {
      currentFrame = Math.max(0, Math.min(totalFrames - 1, frame));
      let phaseText = "flow";
      if (currentFrame < phaseSplit) {
        const prefix = Math.min(K - 1, currentFrame);
        trailLines.geometry.setDrawRange(0, prefix * N * 2);
        setSceneFrame(0);
        setMarkerPositions(K > 0 ? flow[0] : [], 0);
      } else {
        phaseText = "move";
        trailLines.geometry.setDrawRange(0, segmentCount * 2);
        const moveFrame = Math.min(K - 1, currentFrame - phaseSplit);
        setSceneFrame(moveFrame);
        setMarkerPositions(K > 0 ? flow[moveFrame] : [], moveFrame);
      }
      timeline.value = String(currentFrame);
      const logical = currentFrame < phaseSplit ? currentFrame : Math.min(K - 1, currentFrame - phaseSplit);
      const meta = DATA.meta || {};
      readout.textContent = `${phaseText} ${logical + 1}/${K}  t${meta.start_t} -> t${meta.goal_t}`;
    }

    function setStage(index, resetCamera) {
      DATA = STAGES[Math.max(0, Math.min(STAGES.length - 1, index))];
      stageSelect.value = String(index);
      badge.textContent = DATA.title;
      frameCarry = 0;
      rebuildStageGeometry();
      setControlDefaultsFromStage();
      if (resetCamera) applySourceCamera();
    }

    stageSelect.addEventListener("change", () => {
      setStage(parseInt(stageSelect.value, 10), true);
    });

    playButton.addEventListener("click", () => {
      isPlaying = !isPlaying;
      playButton.textContent = isPlaying ? "Pause" : "Play";
      lastTime = performance.now();
    });

    cameraViewButton.addEventListener("click", () => applySourceCamera());
    fitViewButton.addEventListener("click", () => fitCamera());
    renderer.domElement.addEventListener("click", (event) => {
      pickNearestFlowPoint(event.clientX, event.clientY);
    });

    for (const el of [flowSizeInput, sceneSizeInput, lineAlphaInput, sceneAlphaInput, backgroundSelect]) {
      el.addEventListener("input", () => applyControlValues());
      el.addEventListener("change", () => applyControlValues());
    }

    timeline.addEventListener("input", () => {
      isPlaying = false;
      playButton.textContent = "Play";
      update(parseInt(timeline.value, 10));
    });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    applyBackground("white");
    setStage(0, true);

    function animate(now) {
      requestAnimationFrame(animate);
      controls.update();
      if (isPlaying) {
        const elapsed = (now - lastTime) / 1000;
        lastTime = now;
        const fps = Number(currentSettings().fps || 6.0);
        frameCarry += elapsed * fps;
        if (frameCarry >= 1) {
          const step = Math.floor(frameCarry);
          frameCarry -= step;
          update((currentFrame + step) % totalFrames);
        }
      } else {
        lastTime = now;
      }
      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def export_one_html(
    grp: h5py.Group,
    out_path: Path,
    title: str,
    key_idxs: np.ndarray,
    start_t: int,
    goal_t: int,
    traj_key: str,
    depth_key: str,
    scene_coord_mode: str,
    scene_stride: int,
    max_scene_points: int,
    max_depth: float,
    fps: float,
    point_size: float,
    scene_point_size: float,
    line_opacity: float,
    flip_yz: bool,
    screen_space_points: bool,
    dynamic_scene: bool,
    flow_color_mode: str,
    filter_flow_visibility_enabled: bool,
    filter_flow_outliers_enabled: bool,
    flow_max_step: float,
    flow_robust_factor: float,
    flow_spatial_k: int,
    flow_spatial_factor: float,
    flow_min_keep_ratio: float,
    drop_flow_indices: Sequence[int],
    flow_align_key: str,
    flow_align_mode: str,
    flow_align_max_offset: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    flow_xyz, used_traj_key = load_flow_xyz(grp, traj_key=traj_key, key_idxs=key_idxs, start_t=start_t)
    flow_align_info: Dict[str, Any] = {
        "enabled": bool(flow_align_key and flow_align_mode != "none"),
        "align_key": str(flow_align_key),
        "mode": str(flow_align_mode),
    }
    if flow_align_key and flow_align_mode != "none":
        try:
            ref_xyz, used_align_key = load_flow_xyz(grp, traj_key=flow_align_key, key_idxs=key_idxs, start_t=start_t)
            flow_xyz, flow_align_info = align_flow_to_reference(
                flow_xyz=flow_xyz,
                ref_xyz=ref_xyz,
                mode=str(flow_align_mode),
                max_offset=float(flow_align_max_offset),
            )
            flow_align_info["align_key"] = used_align_key
        except Exception as exc:
            flow_align_info["skipped"] = True
            flow_align_info["error"] = repr(exc)

    flow_colors = build_flow_colors(
        grp=grp,
        start_t=start_t,
        num_points=flow_xyz.shape[1],
        color_mode=flow_color_mode,
    )
    flow_xyz, flow_colors, visibility_filter_info = apply_flow_visibility_mask(
        flow_xyz=flow_xyz,
        flow_colors=flow_colors,
        grp=grp,
        key_idxs=key_idxs,
        enabled=bool(filter_flow_visibility_enabled),
    )
    flow_xyz, flow_colors, flow_filter_info = filter_flow_outliers(
        flow_xyz=flow_xyz,
        flow_colors=flow_colors,
        enabled=bool(filter_flow_outliers_enabled),
        max_step=float(flow_max_step),
        robust_factor=float(flow_robust_factor),
        spatial_k=int(flow_spatial_k),
        spatial_factor=float(flow_spatial_factor),
        min_keep_ratio=float(flow_min_keep_ratio),
        drop_indices=drop_flow_indices,
    )

    scene_points, scene_colors, used_depth_key = build_scene_from_depth(
        grp=grp,
        frame_idx=start_t,
        depth_key=depth_key,
        scene_stride=scene_stride,
        max_scene_points=max_scene_points,
        max_depth=max_depth,
        scene_coord_mode=scene_coord_mode,
    )
    if scene_points is None or scene_colors is None:
        scene_points, scene_colors, used_depth_key = build_scene_from_rgb_plane(
            grp=grp,
            frame_idx=start_t,
            scene_stride=scene_stride,
            max_scene_points=max_scene_points,
            scene_coord_mode=scene_coord_mode,
            reference_points=flow_xyz[0],
        )
    if scene_points is None or scene_colors is None:
        scene_points, scene_colors = build_sparse_scene(flow_xyz, flow_colors)
        used_depth_key = ""

    scene_point_frames = None
    scene_color_frames = None
    if dynamic_scene:
        scene_point_frames, scene_color_frames, seq_depth_key = build_scene_sequence_from_depth(
            grp=grp,
            key_idxs=key_idxs,
            depth_key=depth_key,
            scene_stride=scene_stride,
            max_scene_points=max_scene_points,
            max_depth=max_depth,
            scene_coord_mode=scene_coord_mode,
            fallback_points=scene_points,
            fallback_colors=scene_colors,
        )
        if seq_depth_key:
            used_depth_key = seq_depth_key

    camera_view = read_camera_view(grp, int(start_t), scene_coord_mode=scene_coord_mode)
    meta = {
        "start_t": int(start_t),
        "goal_t": int(goal_t),
        "key_idxs": [int(x) for x in key_idxs.tolist()],
        "traj_key": used_traj_key,
        "depth_key": used_depth_key,
        "scene_coord_mode": str(scene_coord_mode),
        "num_flow_points": int(flow_xyz.shape[1]),
        "flow_align": flow_align_info,
        "flow_filter": flow_filter_info,
        "flow_visibility_filter": visibility_filter_info,
        "num_scene_points": int(scene_points.shape[0]),
        "num_scene_frames": int(len(scene_point_frames)) if scene_point_frames is not None else 1,
        "dynamic_scene": bool(dynamic_scene),
        "camera_view": camera_view,
    }
    payload = write_interactive_html(
        out_path=out_path,
        title=title,
        flow_xyz=flow_xyz,
        flow_colors=flow_colors,
        scene_points=scene_points,
        scene_colors=scene_colors,
        scene_point_frames=scene_point_frames,
        scene_color_frames=scene_color_frames,
        meta=meta,
        fps=fps,
        point_size=point_size,
        scene_point_size=scene_point_size,
        line_opacity=line_opacity,
        flip_yz=flip_yz,
        screen_space_points=screen_space_points,
        flow_color_mode=flow_color_mode,
        hide_flow_points=False,
    )
    return meta, payload


def save_all_starts_to_goal_3d_html_for_one_demo(
    h5_path: Path,
    out_dir: Path,
    segmenter: TrajectorySegmenter,
    k_steps: int,
    demo_id: Optional[str],
    verbose: bool,
    traj_key: str,
    depth_key: str,
    scene_coord_mode: str,
    scene_stride: int,
    max_scene_points: int,
    max_depth: float,
    fps: float,
    point_size: float,
    scene_point_size: float,
    line_opacity: float,
    flip_yz: bool,
    screen_space_points: bool,
    dynamic_scene: bool,
    flow_color_mode: str,
    filter_flow_visibility_enabled: bool,
    filter_flow_outliers_enabled: bool,
    flow_max_step: float,
    flow_robust_factor: float,
    flow_spatial_k: int,
    flow_spatial_factor: float,
    flow_min_keep_ratio: float,
    drop_flow_indices: Sequence[int],
    flow_align_key: str,
    flow_align_mode: str,
    flow_align_max_offset: float,
    start_stride: int,
    max_start_frames: int,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "h5_path": str(h5_path),
        "task_name": task_name_from_h5(h5_path),
        "demo_id": None,
        "segments": [],
        "viz_cfg": {
            "traj_key": traj_key,
            "depth_key": depth_key,
            "scene_coord_mode": scene_coord_mode,
            "k_steps": int(k_steps),
            "fps": float(fps),
            "point_size": float(point_size),
            "scene_point_size": float(scene_point_size),
            "line_opacity": float(line_opacity),
            "flip_yz": bool(flip_yz),
            "screen_space_points": bool(screen_space_points),
            "dynamic_scene": bool(dynamic_scene),
            "flow_color_mode": str(flow_color_mode),
            "filter_flow_visibility": bool(filter_flow_visibility_enabled),
            "filter_flow_outliers": bool(filter_flow_outliers_enabled),
            "flow_max_step": float(flow_max_step),
            "flow_robust_factor": float(flow_robust_factor),
            "flow_spatial_k": int(flow_spatial_k),
            "flow_spatial_factor": float(flow_spatial_factor),
            "flow_min_keep_ratio": float(flow_min_keep_ratio),
            "drop_flow_indices": [int(x) for x in drop_flow_indices],
            "flow_align_key": str(flow_align_key),
            "flow_align_mode": str(flow_align_mode),
            "flow_align_max_offset": float(flow_align_max_offset),
            "scene_stride": int(scene_stride),
            "max_scene_points": int(max_scene_points),
            "max_depth": float(max_depth),
        },
    }

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{h5_path} has no 'data' group. keys={list(f.keys())}")

        data_grp = f["data"]
        if demo_id is None:
            demo_id = pick_one_demo_id(data_grp)
        if demo_id not in data_grp:
            raise KeyError(f"demo_id={demo_id} not in /data. available={list(data_grp.keys())[:10]}...")

        grp = data_grp[demo_id]
        manifest["demo_id"] = str(demo_id)

        if "frames_rgb" not in grp:
            raise KeyError(f"'frames_rgb' not found in demo group. keys={list(grp.keys())}")

        g = segmenter._load_gripper_signal(grp)
        t_len = int(grp["frames_rgb"].shape[0])
        if len(g) != t_len:
            t_len = min(t_len, len(g))
            g = g[:t_len]

        if verbose:
            gb = segmenter._binarize_gripper(g)
            change_idxs = segmenter._debounce_changes(gb, segmenter.gripper_debounce)
            print(
                f"[DBG] {manifest['task_name']} demo={demo_id} "
                f"T={t_len} num_changes={len(change_idxs)} change_idxs={change_idxs[:50]}"
            )

        segments = segmenter.seg_gripper_state(g, t_len)
        task_out = out_dir / manifest["task_name"] / f"demo_{demo_id}"
        task_out.mkdir(parents=True, exist_ok=True)
        stage_payloads: List[Dict[str, Any]] = []

        for si, (s0, s1_excl) in enumerate(segments):
            seg_last = int(s1_excl - 1)
            if seg_last < int(s0):
                continue

            seg_dir = task_out / f"seg_{si:03d}_s{int(s0):05d}_e{int(seg_last):05d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            seg_info: Dict[str, Any] = {
                "seg_idx": int(si),
                "start": int(s0),
                "end_exclusive": int(s1_excl),
                "end": int(seg_last),
                "len": int(s1_excl - s0),
                "start_htmls": [],
            }

            start_indices = collect_start_indices(
                s0=int(s0),
                seg_last=int(seg_last),
                start_stride=int(start_stride),
                max_start_frames=int(max_start_frames),
            )

            for start_t in start_indices:
                key_idxs = segmenter.resample_on_traj(int(start_t), int(seg_last), num=int(k_steps)).astype(np.int64)
                key_idxs = np.clip(key_idxs, 0, t_len - 1)
                html_path = seg_dir / f"start_t{int(start_t):05d}_to_goal_3d.html"
                title = f"{manifest['task_name']} demo={demo_id} seg={si} start={start_t} goal={seg_last}"
                start_info: Dict[str, Any] = {
                    "start_t": int(start_t),
                    "goal_t": int(seg_last),
                    "html_path": None,
                    "key_idxs": [int(x) for x in key_idxs.tolist()],
                }

                try:
                    meta, payload = export_one_html(
                        grp=grp,
                        out_path=html_path,
                        title=title,
                        key_idxs=key_idxs,
                        start_t=int(start_t),
                        goal_t=int(seg_last),
                        traj_key=str(traj_key),
                        depth_key=str(depth_key),
                        scene_coord_mode=str(scene_coord_mode),
                        scene_stride=int(scene_stride),
                        max_scene_points=int(max_scene_points),
                        max_depth=float(max_depth),
                        fps=float(fps),
                        point_size=float(point_size),
                        scene_point_size=float(scene_point_size),
                        line_opacity=float(line_opacity),
                        flip_yz=bool(flip_yz),
                        screen_space_points=bool(screen_space_points),
                        dynamic_scene=bool(dynamic_scene),
                        flow_color_mode=str(flow_color_mode),
                        filter_flow_visibility_enabled=bool(filter_flow_visibility_enabled),
                        filter_flow_outliers_enabled=bool(filter_flow_outliers_enabled),
                        flow_max_step=float(flow_max_step),
                        flow_robust_factor=float(flow_robust_factor),
                        flow_spatial_k=int(flow_spatial_k),
                        flow_spatial_factor=float(flow_spatial_factor),
                        flow_min_keep_ratio=float(flow_min_keep_ratio),
                        drop_flow_indices=drop_flow_indices,
                        flow_align_key=str(flow_align_key),
                        flow_align_mode=str(flow_align_mode),
                        flow_align_max_offset=float(flow_align_max_offset),
                    )
                    start_info["html_path"] = str(html_path)
                    start_info["meta"] = meta
                    payload["stage_label"] = f"stage {si}: t{int(start_t)} -> t{int(seg_last)}"
                    stage_payloads.append(payload)
                except Exception as exc:
                    start_info["error"] = repr(exc)
                    print(
                        f"[WARN] {manifest['task_name']} demo={demo_id} seg={si} "
                        f"start_t={start_t}: {repr(exc)}"
                    )

                seg_info["start_htmls"].append(start_info)

            manifest["segments"].append(seg_info)

        if stage_payloads:
            all_stage_path = task_out / "all_stages_3d.html"
            write_staged_interactive_html(all_stage_path, stage_payloads)
            manifest["all_stages_html_path"] = str(all_stage_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{manifest['task_name']}__demo_{manifest['demo_id']}__3d_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_root", type=str, required=True, help="hdf5 file or directory")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory for interactive HTMLs")
    parser.add_argument("--k_steps", type=int, default=20)

    parser.add_argument("--gripper_debounce", type=int, default=3)
    parser.add_argument("--keep_last_segment", action="store_true")
    parser.add_argument("--drop_last_segment", action="store_true")
    parser.add_argument("--min_seg_len", type=int, default=10)

    parser.add_argument("--gamma_s", type=float, default=1.2)
    parser.add_argument("--gamma_e", type=float, default=1.6)
    parser.add_argument("--min_unique_ratio", type=float, default=0.7)

    parser.add_argument("--demo_id", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--task_type", type=str, default="libero", choices=["libero", "maniskill", "real"])

    parser.add_argument("--traj_key", type=str, default="auto", help="point_traj_metric, point_traj, pre_point_traj_metric, pre_point_traj, or auto")
    parser.add_argument("--depth_key", type=str, default="auto", help="optional depth dataset for dense scene point cloud")
    parser.add_argument(
        "--scene_coord_mode",
        type=str,
        default="auto",
        choices=SCENE_COORD_MODES,
        help="Coordinate system for dense scene points: metric uses sim/base poses; spatracker uses VGGT/SpaTracker poses when present.",
    )
    parser.add_argument("--scene_stride", type=int, default=2)
    parser.add_argument("--max_scene_points", type=int, default=14000)
    parser.add_argument("--max_depth", type=float, default=-1.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--point_size", type=float, default=9.0)
    parser.add_argument("--scene_point_size", type=float, default=5.0)
    parser.add_argument("--line_opacity", type=float, default=0.78)
    parser.add_argument(
        "--flow_color_mode",
        type=str,
        default="rainbow",
        choices=FLOW_COLOR_MODES,
        help="rainbow: spatial jet-style colors from initial query y; index: jet-style colors by point index; rgb: sample RGB frame colors; time: color by flow step",
    )
    parser.add_argument(
        "--no_filter_flow_visibility",
        action="store_true",
        help="disable visual-only hiding of tracks marked invalid by vis/point_traj_valid_mask",
    )
    parser.add_argument("--no_filter_flow_outliers", action="store_true", help="disable visual-only filtering of flow trajectories with large jumps")
    parser.add_argument("--flow_max_step", type=float, default=0.30, help="drop flow points with any step larger than this value; <=0 disables this absolute cap")
    parser.add_argument("--flow_robust_factor", type=float, default=8.0, help="robust MAD/IQR factor for visual-only flow outlier filtering; <=0 disables robust filtering")
    parser.add_argument("--flow_spatial_k", type=int, default=5, help="kNN size for visual-only spatial flow outlier filtering")
    parser.add_argument("--flow_spatial_factor", type=float, default=6.0, help="robust factor for dropping spatially isolated flow points; <=0 disables spatial filtering")
    parser.add_argument("--flow_min_keep_ratio", type=float, default=0.5, help="fallback to unfiltered finite flow if filtering would keep fewer than this ratio")
    parser.add_argument("--drop_flow_indices", type=str, default="", help="comma-separated original query point indices to hide in visualization")
    parser.add_argument("--flow_align_key", type=str, default="", help="optional reference trajectory key used for visual translation alignment")
    parser.add_argument("--flow_align_mode", type=str, default="none", choices=FLOW_ALIGN_MODES)
    parser.add_argument("--flow_align_max_offset", type=float, default=0.40, help="skip visual alignment if a translation exceeds this many meters; <=0 disables the cap")
    parser.add_argument("--no_flip_yz", action="store_true", help="disable tapip3d-style y/z flip in the browser")
    parser.add_argument("--world_space_points", action="store_true", help="use perspective/world-unit point sizes instead of clearer screen-space point sizes")
    parser.add_argument("--static_scene", action="store_true", help="keep the scene point cloud fixed during the move phase")

    parser.add_argument("--start_stride", type=int, default=1)
    parser.add_argument("--max_start_frames", type=int, default=-1)

    args = parser.parse_args()

    flow_root = Path(args.flow_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_last = True
    if args.drop_last_segment:
        keep_last = False
    if args.keep_last_segment:
        keep_last = True

    drop_flow_indices = parse_index_list(str(args.drop_flow_indices))

    if args.task_type == "libero":
        plus_is_close = True
        normalize_gripper = False
    elif args.task_type == "maniskill":
        plus_is_close = False
        normalize_gripper = False
    else:
        plus_is_close = False
        normalize_gripper = True

    segmenter = TrajectorySegmenter(
        gripper_debounce=args.gripper_debounce,
        keep_last_segment=keep_last,
        min_seg_len=args.min_seg_len,
        gamma_s=args.gamma_s,
        gamma_e=args.gamma_e,
        min_unique_ratio=args.min_unique_ratio,
        plus_is_close=plus_is_close,
        normalize_gripper=normalize_gripper,
    )

    if not flow_root.exists():
        raise RuntimeError(f"flow_root not exists: {flow_root}")

    if flow_root.is_file():
        h5_files = [flow_root]
    else:
        h5_files = sorted(flow_root.rglob("*_tracks.hdf5"))
        if len(h5_files) == 0:
            h5_files = sorted(flow_root.rglob("*.hdf5")) + sorted(flow_root.rglob("*.h5"))

    if len(h5_files) == 0:
        raise RuntimeError(f"No hdf5/h5 files found under {flow_root}")

    if args.max_tasks is not None and args.max_tasks > 0:
        h5_files = h5_files[: args.max_tasks]

    all_manifests = []
    total_htmls = 0
    for h5_path in h5_files:
        try:
            manifest = save_all_starts_to_goal_3d_html_for_one_demo(
                h5_path=h5_path,
                out_dir=out_dir,
                segmenter=segmenter,
                k_steps=args.k_steps,
                demo_id=args.demo_id,
                verbose=bool(args.verbose),
                traj_key=str(args.traj_key),
                depth_key=str(args.depth_key),
                scene_coord_mode=str(args.scene_coord_mode),
                scene_stride=int(args.scene_stride),
                max_scene_points=int(args.max_scene_points),
                max_depth=float(args.max_depth),
                fps=float(args.fps),
                point_size=float(args.point_size),
                scene_point_size=float(args.scene_point_size),
                line_opacity=float(args.line_opacity),
                flip_yz=(not bool(args.no_flip_yz)),
                screen_space_points=(not bool(args.world_space_points)),
                dynamic_scene=(not bool(args.static_scene)),
                flow_color_mode=str(args.flow_color_mode),
                filter_flow_visibility_enabled=(not bool(args.no_filter_flow_visibility)),
                filter_flow_outliers_enabled=(not bool(args.no_filter_flow_outliers)),
                flow_max_step=float(args.flow_max_step),
                flow_robust_factor=float(args.flow_robust_factor),
                flow_spatial_k=int(args.flow_spatial_k),
                flow_spatial_factor=float(args.flow_spatial_factor),
                flow_min_keep_ratio=float(args.flow_min_keep_ratio),
                drop_flow_indices=drop_flow_indices,
                flow_align_key=str(args.flow_align_key),
                flow_align_mode=str(args.flow_align_mode),
                flow_align_max_offset=float(args.flow_align_max_offset),
                start_stride=int(args.start_stride),
                max_start_frames=int(args.max_start_frames),
            )
            all_manifests.append(manifest)
            num_ok = sum(
                1
                for seg in manifest["segments"]
                for start_info in seg.get("start_htmls", [])
                if start_info.get("html_path")
            )
            total_htmls += int(num_ok)
            print(
                f"[OK] {manifest['task_name']} demo={manifest['demo_id']} "
                f"segments={len(manifest['segments'])} start_to_goal_htmls={num_ok}"
            )
        except Exception as exc:
            print(f"[WARN] skip {h5_path}: {repr(exc)}")

    index_path = out_dir / "_all_tasks_3d_index.json"
    index_path.write_text(json.dumps(all_manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Saved to: {out_dir} total_htmls={total_htmls}")


if __name__ == "__main__":
    main()
