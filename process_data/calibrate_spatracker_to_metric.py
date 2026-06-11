#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate SpaTracker trajectories into robot-base metric coordinates.

SpaTracker trajectories are usually smooth, but their coordinate frame and
scale are relative. Metric RGB-D lifting gives robot-base meters, but individual
track/depth samples can be noisy near occlusions and depth discontinuities.

This script fits a robust Sim(3) transform

    p_metric ~= scale * R @ p_spatracker + t

from reliable SpaTracker/metric point correspondences, then applies it to the
full SpaTracker trajectory. It writes a new dataset by default instead of
overwriting existing metric trajectories.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


DEFAULT_TARGET_CANDIDATES = (
    "point_traj_base_metric_unfiltered",
    "point_traj_base_metric_depth_aligned",
    "point_traj_base_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustly align point_traj_spatracker to robot-base metric coordinates."
    )
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively.")
    parser.add_argument("--data_group", default="data")
    parser.add_argument("--demo_ids", nargs="*", default=None)
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--source_key", default="point_traj_spatracker")
    parser.add_argument(
        "--target_key",
        default="auto",
        help="Metric anchor trajectory key. Use auto to try unfiltered/depth_aligned/filtered metric keys.",
    )
    parser.add_argument("--target_candidates", nargs="+", default=list(DEFAULT_TARGET_CANDIDATES))
    parser.add_argument("--vis_key", default="vis")
    parser.add_argument("--min_vis", type=float, default=0.5)
    parser.add_argument("--out_key", default="point_traj_base_metric_from_spatracker")

    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_correspondences", type=int, default=20000)
    parser.add_argument("--min_correspondences", type=int, default=32)
    parser.add_argument(
        "--max_metric_point_step_m",
        type=float,
        default=0.30,
        help="Discard an entire point track as an anchor if its metric target has a larger one-frame jump; <=0 disables.",
    )
    parser.add_argument(
        "--first_frame_radius_quantile",
        type=float,
        default=0.98,
        help="Drop far first-frame metric anchors by radius quantile; <=0 or >=1 disables.",
    )

    parser.add_argument("--ransac_iters", type=int, default=512)
    parser.add_argument("--inlier_threshold_m", type=float, default=0.04)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set_active", action="store_true", help="Also copy out_key into point_traj.")
    parser.add_argument("--compression", default="gzip", help="HDF5 compression; use none/false/0 to disable.")
    parser.add_argument("--quiet", action="store_true", help="Only print per-file and final summary.")
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


def resolve_target_key(group: h5py.Group, requested: str, candidates: Sequence[str]) -> str:
    if requested != "auto":
        if requested not in group:
            raise KeyError(f"target_key={requested!r} not found. available={list(group.keys())}")
        return requested
    for key in candidates:
        if key in group:
            return key
    raise KeyError(f"No metric target key found. tried={list(candidates)}, available={list(group.keys())}")


def fit_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * (src @ R.T) + t with Umeyama alignment."""
    if src.shape[0] < 3:
        raise ValueError("Need at least three correspondences for Sim(3)")

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


def sim3_matrix(scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = (float(scale) * rot).astype(np.float32)
    mat[:3, 3] = trans.astype(np.float32)
    return mat


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
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least three correspondences")

    best_inliers = np.zeros(n, dtype=bool)
    best_err = np.full(n, np.inf, dtype=np.float64)
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
            best_err = err

    if best_inliers.sum() < 3:
        scale, rot, trans = fit_sim3(src, dst)
        err = residuals(src, dst, scale, rot, trans)
        inliers = err <= float(threshold)
        return scale, rot, trans, inliers, err

    scale, rot, trans = fit_sim3(src[best_inliers], dst[best_inliers])
    err = residuals(src, dst, scale, rot, trans)
    inliers = err <= float(threshold)
    if inliers.sum() >= 3:
        scale, rot, trans = fit_sim3(src[inliers], dst[inliers])
        err = residuals(src, dst, scale, rot, trans)
        inliers = err <= float(threshold)
    return scale, rot, trans, inliers, err


def finite_and_shape_ok(source: np.ndarray, target: np.ndarray) -> bool:
    return (
        source.shape == target.shape
        and source.ndim == 3
        and source.shape[-1] == 3
        and source.shape[0] >= 2
        and source.shape[1] >= 1
    )


def point_track_step_mask(target: np.ndarray, max_step_m: float) -> np.ndarray:
    if max_step_m <= 0:
        return np.ones(target.shape[1], dtype=bool)
    steps = np.linalg.norm(np.diff(target, axis=0), axis=-1)
    return np.nanmax(steps, axis=0) <= float(max_step_m)


def first_frame_radius_mask(target: np.ndarray, quantile: float) -> np.ndarray:
    if quantile <= 0.0 or quantile >= 1.0:
        return np.ones(target.shape[1], dtype=bool)
    first = target[0]
    finite = np.isfinite(first).all(axis=1)
    out = np.zeros(target.shape[1], dtype=bool)
    if finite.sum() < 4:
        out[finite] = True
        return out
    center = np.nanmedian(first[finite], axis=0)
    radius = np.linalg.norm(first - center[None, :], axis=1)
    cutoff = np.nanquantile(radius[finite], float(quantile))
    out = finite & (radius <= cutoff)
    return out


def visibility_mask(group: h5py.Group, vis_key: str, min_vis: float, shape: tuple[int, int]) -> np.ndarray:
    if not vis_key or vis_key not in group:
        return np.ones(shape, dtype=bool)
    vis = np.asarray(group[vis_key])
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]
    if vis.shape[:2] != shape:
        return np.ones(shape, dtype=bool)
    return np.asarray(vis, dtype=np.float32) >= float(min_vis)


def build_correspondences(
    group: h5py.Group,
    source: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    t_len, n_points = source.shape[:2]
    frame_mask = np.zeros(t_len, dtype=bool)
    frame_mask[:: max(1, int(args.frame_stride))] = True

    finite = np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1)
    finite &= visibility_mask(group, args.vis_key, args.min_vis, (t_len, n_points))

    track_keep = point_track_step_mask(target, float(args.max_metric_point_step_m))
    track_keep &= first_frame_radius_mask(target, float(args.first_frame_radius_quantile))
    finite &= track_keep[None, :]
    finite &= frame_mask[:, None]

    src = source[finite].astype(np.float64)
    dst = target[finite].astype(np.float64)
    before_subsample = int(src.shape[0])
    if src.shape[0] > int(args.max_correspondences):
        idx = rng.choice(src.shape[0], size=int(args.max_correspondences), replace=False)
        src = src[idx]
        dst = dst[idx]

    stats = {
        "num_candidate_corr": before_subsample,
        "num_corr": int(src.shape[0]),
        "num_track_points": int(n_points),
        "num_anchor_tracks_kept": int(track_keep.sum()),
        "anchor_track_keep_ratio": float(track_keep.sum() / max(n_points, 1)),
    }
    return src, dst, stats


def split_train_val(
    src: np.ndarray,
    dst: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = src.shape[0]
    if n < 8 or val_fraction <= 0:
        return src, dst, src, dst
    n_val = int(round(n * min(max(float(val_fraction), 0.0), 0.8)))
    n_val = min(max(n_val, 1), n - 3)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return src[train_idx], dst[train_idx], src[val_idx], dst[val_idx]


def err_stats(err: np.ndarray, prefix: str) -> dict[str, float]:
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {
            f"{prefix}_rmse_m": float("nan"),
            f"{prefix}_median_m": float("nan"),
            f"{prefix}_p95_m": float("nan"),
            f"{prefix}_max_m": float("nan"),
        }
    return {
        f"{prefix}_rmse_m": float(np.sqrt(np.nanmean(err * err))),
        f"{prefix}_median_m": float(np.nanmedian(err)),
        f"{prefix}_p95_m": float(np.nanpercentile(err, 95)),
        f"{prefix}_max_m": float(np.nanmax(err)),
    }


def trajectory_stats(traj: np.ndarray, prefix: str) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    max_step_per_point = np.nanmax(steps, axis=0)
    unique0 = np.unique(np.round(traj[0], 3), axis=0).shape[0]
    return {
        f"{prefix}_max_step_m": float(np.nanmax(steps)),
        f"{prefix}_p95_point_max_step_m": float(np.nanpercentile(max_step_per_point, 95)),
        f"{prefix}_median_point_max_step_m": float(np.nanmedian(max_step_per_point)),
        f"{prefix}_unique0_ratio": float(unique0 / max(traj.shape[1], 1)),
    }


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, overwrite: bool, compression) -> None:
    if name in group:
        if not overwrite:
            raise ValueError(f"{name} already exists; pass --overwrite to replace it")
        del group[name]
    group.create_dataset(name, data=np.asarray(data), compression=compression)


def process_demo(
    track_file: Path,
    demo_id: str,
    group: h5py.Group,
    args: argparse.Namespace,
    rng: np.random.Generator,
    compression,
) -> dict[str, object]:
    row: dict[str, object] = {
        "track_file": str(track_file),
        "demo_id": demo_id,
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
        target_key = resolve_target_key(group, args.target_key, args.target_candidates)
    except KeyError as exc:
        row["status"] = "skip"
        row["error"] = str(exc)
        return row

    row["target_key"] = target_key
    source = np.asarray(group[args.source_key], dtype=np.float32)
    target = np.asarray(group[target_key], dtype=np.float32)
    if not finite_and_shape_ok(source, target):
        row["status"] = "skip"
        row["error"] = f"shape mismatch source={source.shape} target={target.shape}"
        return row

    src_corr, dst_corr, corr_stats = build_correspondences(group, source, target, args, rng)
    row.update(corr_stats)
    if src_corr.shape[0] < int(args.min_correspondences):
        row["status"] = "skip"
        row["error"] = f"too few correspondences {src_corr.shape[0]} < {args.min_correspondences}"
        return row

    train_src, train_dst, val_src, val_dst = split_train_val(src_corr, dst_corr, args.val_fraction, rng)
    scale, rot, trans, train_inliers, train_err = ransac_sim3(
        train_src,
        train_dst,
        iterations=int(args.ransac_iters),
        threshold=float(args.inlier_threshold_m),
        rng=rng,
    )
    train_inlier_ratio = float(train_inliers.sum() / max(train_inliers.size, 1))
    val_err = residuals(val_src, val_dst, scale, rot, trans)
    val_inliers = val_err <= float(args.inlier_threshold_m)
    val_inlier_ratio = float(val_inliers.sum() / max(val_inliers.size, 1))
    all_err = residuals(src_corr, dst_corr, scale, rot, trans)
    all_inlier_ratio = float(np.mean(all_err <= float(args.inlier_threshold_m)))

    aligned = apply_sim3(source, scale, rot, trans)
    row.update(
        {
            "scale": float(scale),
            "train_corr": int(train_src.shape[0]),
            "val_corr": int(val_src.shape[0]),
            "train_inliers": int(train_inliers.sum()),
            "train_inlier_ratio": train_inlier_ratio,
            "val_inliers": int(val_inliers.sum()),
            "val_inlier_ratio": val_inlier_ratio,
            "all_inlier_ratio": all_inlier_ratio,
            "threshold_m": float(args.inlier_threshold_m),
        }
    )
    row.update(err_stats(train_err[train_inliers], "train_inlier"))
    row.update(err_stats(val_err[val_inliers], "val_inlier"))
    row.update(err_stats(train_err, "train_all"))
    row.update(err_stats(val_err, "val"))
    row.update(err_stats(all_err, "all"))
    row.update(trajectory_stats(target, "target"))
    row.update(trajectory_stats(aligned, "aligned"))

    if args.dry_run:
        return row

    write_dataset(group, args.out_key, aligned, args.overwrite, compression)
    write_dataset(group, f"{args.out_key}_sim3", sim3_matrix(scale, rot, trans), args.overwrite, compression)
    target_frame = group.attrs.get(f"{target_key}_coordinate_frame", "metric")
    group.attrs[f"{args.out_key}_source_key"] = args.source_key
    group.attrs[f"{args.out_key}_target_key"] = target_key
    group.attrs[f"{args.out_key}_units"] = "meters"
    group.attrs[f"{args.out_key}_coordinate_frame"] = target_frame
    group.attrs[f"{args.out_key}_alignment"] = "robust_sim3"
    group.attrs[f"{args.out_key}_scale"] = float(scale)
    group.attrs[f"{args.out_key}_rotation"] = rot.astype(np.float32)
    group.attrs[f"{args.out_key}_translation"] = trans.astype(np.float32)
    for key, value in row.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            group.attrs[f"{args.out_key}_{key}"] = value

    if args.set_active:
        write_dataset(group, "point_traj", aligned, True, compression)
        group.attrs["point_traj_active_source"] = args.out_key
        group.attrs["point_traj_mode"] = "metric_from_spatracker"
        group.attrs["point_traj_units"] = "meters"
        group.attrs["point_traj_coordinate_frame"] = target_frame

    modes = [
        key
        for key in ("point_traj_spatracker", "point_traj_world_metric", "point_traj_base_metric", args.out_key)
        if key in group
    ]
    if modes:
        group.attrs["point_traj_modes_available"] = ",".join(dict.fromkeys(modes))

    return row


def process_file(track_file: Path, args: argparse.Namespace, rows: list[dict[str, object]], rng: np.random.Generator) -> None:
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
            row = process_demo(track_file, demo_id, f[args.data_group][demo_id], args, rng, compression)
            rows.append(row)
            if row["status"] != "ok":
                print(f"  [WARN] {demo_id}: {row['error']}")
                continue
            if args.quiet:
                continue
            print(
                "  {demo}: scale={scale:.4g} inlier={inlier:.2f} "
                "val_med/p95={med:.3f}/{p95:.3f}m "
                "aligned_step_p95={step:.3f}m".format(
                    demo=demo_id,
                    scale=float(row["scale"]),
                    inlier=float(row["all_inlier_ratio"]),
                    med=float(row["val_median_m"]),
                    p95=float(row["val_p95_m"]),
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
        "target_key",
        "out_key",
        "scale",
        "num_candidate_corr",
        "num_corr",
        "num_track_points",
        "num_anchor_tracks_kept",
        "anchor_track_keep_ratio",
        "train_corr",
        "val_corr",
        "train_inliers",
        "train_inlier_ratio",
        "val_inliers",
        "val_inlier_ratio",
        "all_inlier_ratio",
        "threshold_m",
        "train_inlier_rmse_m",
        "train_inlier_median_m",
        "train_inlier_p95_m",
        "val_inlier_rmse_m",
        "val_inlier_median_m",
        "val_inlier_p95_m",
        "train_all_rmse_m",
        "train_all_median_m",
        "train_all_p95_m",
        "val_rmse_m",
        "val_median_m",
        "val_p95_m",
        "val_max_m",
        "all_rmse_m",
        "all_median_m",
        "all_p95_m",
        "all_max_m",
        "target_max_step_m",
        "target_p95_point_max_step_m",
        "target_median_point_max_step_m",
        "target_unique0_ratio",
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
    skipped = len(rows) - len(ok)
    print(f"\n[DONE] demos={len(rows)} ok={len(ok)} skipped={skipped}")
    if not ok:
        return
    for key in (
        "all_inlier_ratio",
        "val_inlier_ratio",
        "val_inlier_median_m",
        "val_inlier_p95_m",
        "val_median_m",
        "val_p95_m",
        "aligned_p95_point_max_step_m",
        "scale",
    ):
        vals = np.asarray([float(r[key]) for r in ok if key in r and np.isfinite(float(r[key]))], dtype=np.float64)
        if vals.size == 0:
            continue
        print(
            f"  {key}: median={np.median(vals):.4f} "
            f"p10={np.percentile(vals, 10):.4f} p90={np.percentile(vals, 90):.4f}"
        )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, object]] = []
    for track_file in collect_track_files(args):
        process_file(track_file, args, rows, rng)
    if args.report_csv:
        write_report(rows, Path(args.report_csv))
        print(f"[OK] report_csv={args.report_csv}")
    print_summary(rows)


if __name__ == "__main__":
    main()
