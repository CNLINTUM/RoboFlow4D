#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-align calibrated SpaTracker metric trajectories for training data.

The global calibrated trajectory

    point_traj_base_metric_from_spatracker

is smooth and metric-scale, but a single Sim(3) over a full demo can leave local
offsets in individual atomic stages. This script keeps the smooth calibrated
shape and applies either a per-stage endpoint translation correction:

    offset(t) = lerp(offset_start, offset_goal)

where each endpoint offset is a robust median between a metric reference
trajectory and the calibrated trajectory at the stage boundary, or a per-stage
robust Sim(3) correction:

    p_aligned = scale * R @ p_source + t

The Sim(3) mode is useful when the globally calibrated flow has the right
motion trend but a locally wrong gripper footprint size or orientation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


DEFAULT_REF_CANDIDATES = (
    "point_traj_base_metric_unfiltered",
    "point_traj_base_metric_depth_aligned",
    "point_traj_base_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write stage-aligned calibrated SpaTracker metric trajectories.")
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively.")
    parser.add_argument("--data_group", default="data")
    parser.add_argument("--demo_ids", nargs="*", default=None)
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--source_key", default="point_traj_base_metric_from_spatracker")
    parser.add_argument("--ref_key", default="auto", help="Metric endpoint reference; auto tries unfiltered/depth_aligned/filtered.")
    parser.add_argument("--ref_candidates", nargs="+", default=list(DEFAULT_REF_CANDIDATES))
    parser.add_argument("--vis_key", default="vis")
    parser.add_argument("--min_vis", type=float, default=0.5)
    parser.add_argument("--out_key", default="point_traj_base_metric_from_spatracker_stage_aligned")
    parser.add_argument(
        "--align_mode",
        default="translation",
        choices=("translation", "sim3", "endpoint_sim3"),
        help=(
            "Per-stage correction model. translation keeps the old endpoint lerp behavior; "
            "sim3 fits one transform over the full stage; endpoint_sim3 fits stage-boundary "
            "Sim(3) transforms and interpolates them."
        ),
    )

    parser.add_argument("--gripper_debounce", type=int, default=3)
    parser.add_argument("--keep_last_segment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_seg_len", type=int, default=10)
    parser.add_argument("--plus_is_close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_gripper", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--trim_percentile", type=float, default=80.0, help="Residual percentile kept when estimating endpoint offsets.")
    parser.add_argument("--max_offset_m", type=float, default=0.40, help="Skip an endpoint correction if its norm exceeds this value; <=0 disables.")
    parser.add_argument("--max_endpoint_p95_m", type=float, default=0.25, help="Skip an endpoint correction if endpoint residual p95 exceeds this value; <=0 disables.")
    parser.add_argument("--stage_sim3_iterations", type=int, default=256)
    parser.add_argument("--stage_sim3_threshold_m", type=float, default=0.04)
    parser.add_argument("--stage_sim3_min_corr", type=int, default=8)
    parser.add_argument("--stage_sim3_min_inliers", type=int, default=6)
    parser.add_argument("--stage_sim3_min_inlier_ratio", type=float, default=0.25)
    parser.add_argument("--stage_sim3_min_scale", type=float, default=0.4)
    parser.add_argument("--stage_sim3_max_scale", type=float, default=2.5)
    parser.add_argument("--stage_sim3_max_p95_m", type=float, default=0.25)
    parser.add_argument("--stage_sim3_max_correspondences", type=int, default=5000)
    parser.add_argument("--stage_sim3_seed", type=int, default=0)
    parser.add_argument("--set_active", action="store_true", help="Also copy out_key into point_traj.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--compression", default="gzip", help="HDF5 compression; use none/false/0 to disable.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def collect_track_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.tracks or []:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*_tracks.hdf5")))
        else:
            paths.append(p)
    for root in args.tracks_root or []:
        paths.extend(sorted(Path(root).rglob("*_tracks.hdf5")))

    uniq: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise ValueError("No track files found. Pass --tracks or --tracks_root.")
    return uniq


def sort_demo_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        if x.startswith("demo_"):
            tail = x.split("_")[-1]
            if tail.isdigit():
                return (0, int(tail))
        if x.isdigit():
            return (0, int(x))
        return (1, x)

    return sorted(ids, key=key_fn)


def select_demo_ids(data_grp: h5py.Group, requested: Optional[Sequence[str]], max_demos: Optional[int]) -> list[str]:
    ids = sort_demo_ids(data_grp.keys())
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def resolve_ref_key(group: h5py.Group, requested: str, candidates: Sequence[str]) -> str:
    if requested != "auto":
        if requested not in group:
            raise KeyError(f"ref_key={requested!r} not found. available={list(group.keys())}")
        return requested
    for key in candidates:
        if key in group:
            return key
    raise KeyError(f"No metric reference key found. tried={list(candidates)}, available={list(group.keys())}")


def binarize_gripper(g: np.ndarray, plus_is_close: bool) -> np.ndarray:
    g = np.nan_to_num(g.astype(np.float32))
    thr = 0.5 * (float(np.nanmin(g)) + float(np.nanmax(g)))
    if plus_is_close:
        return (g > thr).astype(np.int32)
    return (g < thr).astype(np.int32)


def debounce_changes(gb: np.ndarray, debounce: int) -> list[int]:
    if debounce <= 1:
        return (np.where(np.diff(gb) != 0)[0] + 1).astype(int).tolist()

    t_len = len(gb)
    cur = gb[0]
    changes: list[int] = []
    i = 1
    while i < t_len:
        if gb[i] == cur:
            i += 1
            continue
        new = gb[i]
        ok = True
        for j in range(i, min(t_len, i + debounce)):
            if gb[j] != new:
                ok = False
                break
        if ok:
            changes.append(i)
            cur = new
            i += debounce
        else:
            i += 1
    return changes


def segment_gripper(actions: np.ndarray, args: argparse.Namespace, t_len: int) -> list[tuple[int, int]]:
    if actions.ndim != 2 or actions.shape[0] < 1:
        return [(0, t_len)]
    g = actions[:t_len, -1].astype(np.float32)
    if args.normalize_gripper:
        g = (g - 0.5) * 2.0
    gb = binarize_gripper(g, plus_is_close=bool(args.plus_is_close))
    changes = debounce_changes(gb, int(args.gripper_debounce))
    boundaries = sorted(set([0] + [int(x) for x in changes] + [int(t_len)]))
    segments: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s0, s1 = boundaries[i], boundaries[i + 1]
        if s1 <= s0:
            continue
        if (i == len(boundaries) - 2) and (not args.keep_last_segment) and changes:
            continue
        if (s1 - s0) >= int(args.min_seg_len):
            segments.append((s0, s1))
    if not segments and t_len >= int(args.min_seg_len):
        segments = [(0, t_len)]
    return segments


def endpoint_offset(
    source: np.ndarray,
    ref: np.ndarray,
    vis: Optional[np.ndarray],
    frame_idx: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    src = np.asarray(source[frame_idx], dtype=np.float32)
    dst = np.asarray(ref[frame_idx], dtype=np.float32)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if vis is not None:
        finite &= np.asarray(vis[frame_idx], dtype=np.float32) >= float(args.min_vis)

    info: dict[str, float | int | bool | str] = {
        "frame": int(frame_idx),
        "num_corr": int(np.sum(finite)),
        "offset_norm": 0.0,
        "residual_median": float("nan"),
        "residual_p95": float("nan"),
        "skipped": False,
    }
    if int(np.sum(finite)) < 3:
        info["skipped"] = True
        info["reason"] = "too_few_corr"
        return np.zeros((3,), dtype=np.float32), info

    diff = dst[finite] - src[finite]
    offset = np.nanmedian(diff, axis=0).astype(np.float32)
    residual = np.linalg.norm(diff - offset[None, :], axis=1)
    if residual.size >= 8 and 0.0 < float(args.trim_percentile) < 100.0:
        keep = residual <= np.nanpercentile(residual, float(args.trim_percentile))
        if int(np.sum(keep)) >= 3:
            diff = diff[keep]
            offset = np.nanmedian(diff, axis=0).astype(np.float32)
            residual = np.linalg.norm(diff - offset[None, :], axis=1)
            info["num_corr"] = int(diff.shape[0])

    offset_norm = float(np.linalg.norm(offset))
    info["offset_norm"] = offset_norm
    info["residual_median"] = float(np.nanmedian(residual))
    info["residual_p95"] = float(np.nanpercentile(residual, 95))
    if args.max_offset_m > 0 and offset_norm > float(args.max_offset_m):
        info["skipped"] = True
        info["reason"] = "offset_too_large"
        return np.zeros((3,), dtype=np.float32), info
    if args.max_endpoint_p95_m > 0 and float(info["residual_p95"]) > float(args.max_endpoint_p95_m):
        info["skipped"] = True
        info["reason"] = "endpoint_residual_too_large"
        return np.zeros((3,), dtype=np.float32), info
    return offset, info


def fit_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * (src @ R.T) + t with Umeyama alignment."""
    if src.shape[0] < 3:
        raise ValueError("Need at least three correspondences for Sim(3)")

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    src_var = np.mean(np.sum(src_c * src_c, axis=1))
    if src_var < 1e-12:
        raise ValueError("Degenerate source correspondences")

    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, svals, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(svals * sign) / src_var)
    trans = dst_mean - scale * (src_mean @ rot.T)
    return scale, rot, trans


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    shape = points.shape
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = scale * (flat @ rot.T) + trans
    return out.reshape(shape).astype(np.float32)


def residuals(src: np.ndarray, dst: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pred = scale * (src @ rot.T) + trans
    return np.linalg.norm(pred - dst, axis=1)


def ransac_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    iterations: int,
    threshold: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(src.shape[0])
    if n < 3:
        raise ValueError("Need at least three correspondences")

    best_inliers = np.zeros(n, dtype=bool)
    best_score = (-1, np.inf)
    sample_size = min(4, n)

    for _ in range(max(1, int(iterations))):
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            scale, rot, trans = fit_sim3(src[idx], dst[idx])
        except ValueError:
            continue
        err = residuals(src, dst, scale, rot, trans)
        inliers = err <= float(threshold)
        score = (int(inliers.sum()), float(np.nanmedian(err[inliers])) if np.any(inliers) else np.inf)
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_inliers = inliers

    if int(best_inliers.sum()) < 3:
        scale, rot, trans = fit_sim3(src, dst)
        err = residuals(src, dst, scale, rot, trans)
        inliers = err <= float(threshold)
        return scale, rot, trans, inliers, err

    scale, rot, trans = fit_sim3(src[best_inliers], dst[best_inliers])
    err = residuals(src, dst, scale, rot, trans)
    inliers = err <= float(threshold)
    if int(inliers.sum()) >= 3:
        scale, rot, trans = fit_sim3(src[inliers], dst[inliers])
        err = residuals(src, dst, scale, rot, trans)
        inliers = err <= float(threshold)
    return scale, rot, trans, inliers, err


def stage_correspondences(
    source: np.ndarray,
    ref: np.ndarray,
    vis: Optional[np.ndarray],
    s0: int,
    s1_excl: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source[s0:s1_excl], dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(ref[s0:s1_excl], dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if vis is not None:
        v = np.asarray(vis[s0:s1_excl], dtype=np.float32).reshape(-1)
        mask &= v >= float(args.min_vis)

    idx = np.flatnonzero(mask)
    max_corr = int(args.stage_sim3_max_correspondences)
    if max_corr > 0 and idx.size > max_corr:
        idx = rng.choice(idx, size=max_corr, replace=False)
    return src[idx].astype(np.float64), dst[idx].astype(np.float64)


def segment_sim3_transform(
    source: np.ndarray,
    ref: np.ndarray,
    vis: Optional[np.ndarray],
    s0: int,
    s1_excl: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[Optional[tuple[float, np.ndarray, np.ndarray]], dict[str, float | int | bool | str]]:
    src, dst = stage_correspondences(source, ref, vis, s0, s1_excl, args, rng)
    info: dict[str, float | int | bool | str] = {
        "num_corr": int(src.shape[0]),
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "scale": float("nan"),
        "residual_median": float("nan"),
        "residual_p95": float("nan"),
        "fallback": False,
        "reason": "",
    }
    if int(src.shape[0]) < int(args.stage_sim3_min_corr):
        info["fallback"] = True
        info["reason"] = "too_few_corr"
        return None, info

    try:
        scale, rot, trans, inliers, err = ransac_sim3(
            src,
            dst,
            iterations=int(args.stage_sim3_iterations),
            threshold=float(args.stage_sim3_threshold_m),
            rng=rng,
        )
    except ValueError as exc:
        info["fallback"] = True
        info["reason"] = str(exc)
        return None, info

    num_inliers = int(np.sum(inliers))
    inlier_ratio = float(num_inliers / max(int(src.shape[0]), 1))
    inlier_err = err[inliers] if num_inliers > 0 else err
    info.update(
        {
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "scale": float(scale),
            "residual_median": float(np.nanmedian(inlier_err)),
            "residual_p95": float(np.nanpercentile(inlier_err, 95)),
        }
    )

    if num_inliers < int(args.stage_sim3_min_inliers):
        info["fallback"] = True
        info["reason"] = "too_few_inliers"
        return None, info
    if inlier_ratio < float(args.stage_sim3_min_inlier_ratio):
        info["fallback"] = True
        info["reason"] = "low_inlier_ratio"
        return None, info
    if not (float(args.stage_sim3_min_scale) <= float(scale) <= float(args.stage_sim3_max_scale)):
        info["fallback"] = True
        info["reason"] = "scale_out_of_range"
        return None, info
    if float(args.stage_sim3_max_p95_m) > 0 and float(info["residual_p95"]) > float(args.stage_sim3_max_p95_m):
        info["fallback"] = True
        info["reason"] = "residual_p95_too_large"
        return None, info
    return (float(scale), rot, trans), info


def trajectory_stats(traj: np.ndarray, prefix: str) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    point_max = np.nanmax(steps, axis=0)
    unique0 = np.unique(np.round(traj[0], 3), axis=0).shape[0]
    return {
        f"{prefix}_max_step_m": float(np.nanmax(steps)),
        f"{prefix}_p95_point_max_step_m": float(np.nanpercentile(point_max, 95)),
        f"{prefix}_median_point_max_step_m": float(np.nanmedian(point_max)),
        f"{prefix}_unique0_ratio": float(unique0 / max(traj.shape[1], 1)),
    }


def reference_error_stats(traj: np.ndarray, ref: np.ndarray, vis: Optional[np.ndarray], prefix: str) -> dict[str, float]:
    err = np.linalg.norm(np.asarray(traj, dtype=np.float32) - np.asarray(ref, dtype=np.float32), axis=-1)
    mask = np.isfinite(err)
    if vis is not None:
        mask &= np.asarray(vis, dtype=np.float32) >= 0.5
    vals = err[mask]
    if vals.size == 0:
        return {
            f"{prefix}_ref_median_m": float("nan"),
            f"{prefix}_ref_p95_m": float("nan"),
            f"{prefix}_ref_rmse_m": float("nan"),
        }
    return {
        f"{prefix}_ref_median_m": float(np.nanmedian(vals)),
        f"{prefix}_ref_p95_m": float(np.nanpercentile(vals, 95)),
        f"{prefix}_ref_rmse_m": float(np.sqrt(np.nanmean(vals * vals))),
    }


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, overwrite: bool, compression) -> None:
    if name in group:
        if not overwrite:
            raise ValueError(f"{name} already exists; pass --overwrite to replace it")
        del group[name]
    group.create_dataset(name, data=np.asarray(data), compression=compression)


def resolve_coordinate_frame(group: h5py.Group, source_key: str, ref_key: str) -> str:
    for key in (source_key, ref_key):
        attr = f"{key}_coordinate_frame"
        if attr in group.attrs:
            value = group.attrs[attr]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return "robot0_base"


def process_demo(
    track_file: Path,
    demo_id: str,
    group: h5py.Group,
    args: argparse.Namespace,
    compression,
) -> dict[str, object]:
    row: dict[str, object] = {
        "track_file": str(track_file),
        "demo_id": str(demo_id),
        "source_key": args.source_key,
        "out_key": args.out_key,
        "status": "ok",
        "error": "",
    }
    if args.source_key not in group:
        row["status"] = "skip"
        row["error"] = f"missing source_key {args.source_key}"
        return row
    try:
        ref_key = resolve_ref_key(group, args.ref_key, args.ref_candidates)
    except KeyError as exc:
        row["status"] = "skip"
        row["error"] = str(exc)
        return row
    row["ref_key"] = ref_key

    source = np.asarray(group[args.source_key], dtype=np.float32)
    ref = np.asarray(group[ref_key], dtype=np.float32)
    if source.shape != ref.shape or source.ndim != 3 or source.shape[-1] != 3:
        row["status"] = "skip"
        row["error"] = f"shape mismatch source={source.shape} ref={ref.shape}"
        return row

    t_len = int(source.shape[0])
    if "actions" in group:
        actions = np.asarray(group["actions"], dtype=np.float32)
        t_len = min(t_len, int(actions.shape[0]))
        segments = segment_gripper(actions, args, t_len)
    else:
        segments = [(0, t_len)]

    vis = None
    if args.vis_key and args.vis_key in group:
        vis_arr = np.asarray(group[args.vis_key], dtype=np.float32)
        if vis_arr.ndim == 3 and vis_arr.shape[-1] == 1:
            vis_arr = vis_arr[..., 0]
        if vis_arr.shape[:2] == source.shape[:2]:
            vis = vis_arr

    out = source.copy()
    offset_norms = []
    skipped_endpoints = 0
    sim3_scales: list[float] = []
    sim3_inlier_ratios: list[float] = []
    sim3_residual_medians: list[float] = []
    sim3_residual_p95s: list[float] = []
    sim3_fallback_segments = 0
    seed = int(args.stage_sim3_seed) + sum(ord(ch) for ch in f"{track_file}:{demo_id}")
    rng = np.random.default_rng(seed)
    for s0, s1_excl in segments:
        seg_last = int(s1_excl - 1)
        if seg_last < int(s0):
            continue
        if args.align_mode == "endpoint_sim3":
            tr0, sim3_info0 = segment_sim3_transform(source, ref, vis, int(s0), int(s0) + 1, args, rng)
            tr1, sim3_info1 = segment_sim3_transform(source, ref, vis, seg_last, seg_last + 1, args, rng)
            if tr0 is not None and tr1 is not None:
                sim3_scales.extend([float(sim3_info0.get("scale", tr0[0])), float(sim3_info1.get("scale", tr1[0]))])
                sim3_inlier_ratios.extend(
                    [
                        float(sim3_info0.get("inlier_ratio", 0.0)),
                        float(sim3_info1.get("inlier_ratio", 0.0)),
                    ]
                )
                sim3_residual_medians.extend(
                    [
                        float(sim3_info0.get("residual_median", float("nan"))),
                        float(sim3_info1.get("residual_median", float("nan"))),
                    ]
                )
                sim3_residual_p95s.extend(
                    [
                        float(sim3_info0.get("residual_p95", float("nan"))),
                        float(sim3_info1.get("residual_p95", float("nan"))),
                    ]
                )
                denom = max(seg_last - int(s0), 1)
                scale0, rot0, trans0 = tr0
                scale1, rot1, trans1 = tr1
                for t in range(int(s0), int(s1_excl)):
                    alpha = float(t - int(s0)) / float(denom)
                    aligned0 = apply_sim3(source[t], scale0, rot0, trans0)
                    aligned1 = apply_sim3(source[t], scale1, rot1, trans1)
                    out[t] = ((1.0 - alpha) * aligned0 + alpha * aligned1).astype(np.float32)
                continue
            sim3_fallback_segments += 1

        if args.align_mode == "sim3":
            transform, sim3_info = segment_sim3_transform(source, ref, vis, int(s0), int(s1_excl), args, rng)
            if transform is not None:
                scale, rot, trans = transform
                out[int(s0) : int(s1_excl)] = apply_sim3(source[int(s0) : int(s1_excl)], scale, rot, trans)
                sim3_scales.append(float(sim3_info.get("scale", scale)))
                sim3_inlier_ratios.append(float(sim3_info.get("inlier_ratio", 0.0)))
                sim3_residual_medians.append(float(sim3_info.get("residual_median", float("nan"))))
                sim3_residual_p95s.append(float(sim3_info.get("residual_p95", float("nan"))))
                continue
            sim3_fallback_segments += 1

        off0, info0 = endpoint_offset(source, ref, vis, int(s0), args)
        off1, info1 = endpoint_offset(source, ref, vis, seg_last, args)
        skipped_endpoints += int(bool(info0.get("skipped", False))) + int(bool(info1.get("skipped", False)))
        offset_norms.extend([float(info0.get("offset_norm", 0.0)), float(info1.get("offset_norm", 0.0))])
        denom = max(seg_last - int(s0), 1)
        for t in range(int(s0), int(s1_excl)):
            alpha = float(t - int(s0)) / float(denom)
            offset = (1.0 - alpha) * off0 + alpha * off1
            out[t] = source[t] + offset[None, :]

    row.update(
        {
            "align_mode": args.align_mode,
            "num_segments": int(len(segments)),
            "num_skipped_endpoints": int(skipped_endpoints),
            "offset_norm_median": float(np.nanmedian(offset_norms)) if offset_norms else 0.0,
            "offset_norm_p95": float(np.nanpercentile(offset_norms, 95)) if offset_norms else 0.0,
            "offset_norm_max": float(np.nanmax(offset_norms)) if offset_norms else 0.0,
            "num_sim3_segments": int(len(sim3_scales)),
            "num_sim3_fallback_segments": int(sim3_fallback_segments),
            "sim3_scale_median": float(np.nanmedian(sim3_scales)) if sim3_scales else float("nan"),
            "sim3_scale_p10": float(np.nanpercentile(sim3_scales, 10)) if sim3_scales else float("nan"),
            "sim3_scale_p90": float(np.nanpercentile(sim3_scales, 90)) if sim3_scales else float("nan"),
            "sim3_inlier_ratio_median": float(np.nanmedian(sim3_inlier_ratios)) if sim3_inlier_ratios else float("nan"),
            "sim3_residual_median_m": float(np.nanmedian(sim3_residual_medians)) if sim3_residual_medians else float("nan"),
            "sim3_residual_p95_m": float(np.nanmedian(sim3_residual_p95s)) if sim3_residual_p95s else float("nan"),
        }
    )
    row.update(reference_error_stats(source[:t_len], ref[:t_len], None if vis is None else vis[:t_len], "source"))
    row.update(reference_error_stats(out[:t_len], ref[:t_len], None if vis is None else vis[:t_len], "aligned"))
    row.update(trajectory_stats(source[:t_len], "source"))
    row.update(trajectory_stats(out[:t_len], "aligned"))

    if args.dry_run:
        return row

    write_dataset(group, args.out_key, out, args.overwrite, compression)
    group.attrs[f"{args.out_key}_source_key"] = args.source_key
    group.attrs[f"{args.out_key}_ref_key"] = ref_key
    if args.align_mode == "endpoint_sim3":
        group.attrs[f"{args.out_key}_alignment"] = (
            "stage_endpoint_sim3_lerp_with_translation_fallback"
            if sim3_fallback_segments
            else "stage_endpoint_sim3_lerp"
        )
    elif args.align_mode == "sim3":
        group.attrs[f"{args.out_key}_alignment"] = (
            "stage_sim3_with_translation_fallback" if sim3_fallback_segments else "stage_sim3"
        )
    else:
        group.attrs[f"{args.out_key}_alignment"] = "stage_endpoint_lerp_translation"
    group.attrs[f"{args.out_key}_align_mode"] = args.align_mode
    group.attrs[f"{args.out_key}_units"] = "meters"
    group.attrs[f"{args.out_key}_coordinate_frame"] = resolve_coordinate_frame(group, args.source_key, ref_key)
    for key, value in row.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            group.attrs[f"{args.out_key}_{key}"] = value

    if args.set_active:
        write_dataset(group, "point_traj", out, True, compression)
        group.attrs["point_traj_active_source"] = args.out_key
        group.attrs["point_traj_mode"] = "metric_stage_aligned"
        group.attrs["point_traj_units"] = "meters"
        group.attrs["point_traj_coordinate_frame"] = resolve_coordinate_frame(group, args.source_key, ref_key)

    modes = [
        key
        for key in (
            "point_traj_spatracker",
            "point_traj_base_metric",
            "point_traj_base_metric_from_spatracker",
            args.out_key,
        )
        if key in group
    ]
    if modes:
        group.attrs["point_traj_modes_available"] = ",".join(dict.fromkeys(modes))
    return row


def process_file(track_file: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    mode = "r" if args.dry_run else "r+"
    compression = h5_compression(args.compression)
    print(f"\n[tracks] {track_file}")
    with h5py.File(track_file, mode) as f:
        if args.data_group not in f:
            rows.append({"track_file": str(track_file), "demo_id": "", "status": "skip", "error": "missing data group"})
            print(f"  [WARN] missing /{args.data_group}")
            return
        demo_ids = select_demo_ids(f[args.data_group], args.demo_ids, args.max_demos)
        print(f"  demos: {len(demo_ids)}")
        for demo_id in demo_ids:
            row = process_demo(track_file, demo_id, f[args.data_group][demo_id], args, compression)
            rows.append(row)
            if row["status"] != "ok":
                print(f"  [WARN] {demo_id}: {row['error']}")
                continue
            if not args.quiet:
                if row.get("align_mode") in {"sim3", "endpoint_sim3"}:
                    print(
                        "  {demo}: segs={segs} sim3={sim3} fallback={fallback} scale_med={scale:.3f} "
                        "ref_med {before:.3f}->{after:.3f}m step_p95={step:.3f}m".format(
                            demo=demo_id,
                            segs=int(row["num_segments"]),
                            sim3=int(row["num_sim3_segments"]),
                            fallback=int(row["num_sim3_fallback_segments"]),
                            scale=float(row["sim3_scale_median"]),
                            before=float(row["source_ref_median_m"]),
                            after=float(row["aligned_ref_median_m"]),
                            step=float(row["aligned_p95_point_max_step_m"]),
                        )
                    )
                else:
                    print(
                        "  {demo}: segs={segs} offset_med={off:.3f}m "
                        "ref_med {before:.3f}->{after:.3f}m step_p95={step:.3f}m".format(
                            demo=demo_id,
                            segs=int(row["num_segments"]),
                            off=float(row["offset_norm_median"]),
                            before=float(row["source_ref_median_m"]),
                            after=float(row["aligned_ref_median_m"]),
                            step=float(row["aligned_p95_point_max_step_m"]),
                        )
                    )


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "track_file",
        "demo_id",
        "status",
        "error",
        "source_key",
        "ref_key",
        "out_key",
        "align_mode",
        "num_segments",
        "num_skipped_endpoints",
        "offset_norm_median",
        "offset_norm_p95",
        "offset_norm_max",
        "num_sim3_segments",
        "num_sim3_fallback_segments",
        "sim3_scale_median",
        "sim3_scale_p10",
        "sim3_scale_p90",
        "sim3_inlier_ratio_median",
        "sim3_residual_median_m",
        "sim3_residual_p95_m",
        "source_ref_median_m",
        "source_ref_p95_m",
        "source_ref_rmse_m",
        "aligned_ref_median_m",
        "aligned_ref_p95_m",
        "aligned_ref_rmse_m",
        "source_max_step_m",
        "source_p95_point_max_step_m",
        "source_median_point_max_step_m",
        "source_unique0_ratio",
        "aligned_max_step_m",
        "aligned_p95_point_max_step_m",
        "aligned_median_point_max_step_m",
        "aligned_unique0_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def print_summary(rows: list[dict[str, object]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n[DONE] demos={len(rows)} ok={len(ok)} skipped={len(rows)-len(ok)}")
    if not ok:
        return
    for key in (
        "source_ref_median_m",
        "aligned_ref_median_m",
        "source_ref_p95_m",
        "aligned_ref_p95_m",
        "aligned_p95_point_max_step_m",
        "offset_norm_median",
        "sim3_scale_median",
        "sim3_inlier_ratio_median",
        "sim3_residual_p95_m",
    ):
        vals = np.asarray([float(r[key]) for r in ok if key in r and r[key] != ""], dtype=np.float64)
        print(
            f"  {key}: median={np.nanmedian(vals):.4f} "
            f"p10={np.nanpercentile(vals, 10):.4f} p90={np.nanpercentile(vals, 90):.4f}"
        )


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for track_file in collect_track_files(args):
        process_file(track_file, args, rows)
    if args.report_csv:
        write_report(rows, Path(args.report_csv))
        print(f"[OK] report_csv={args.report_csv}")
    print_summary(rows)


if __name__ == "__main__":
    main()
