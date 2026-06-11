#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximate-convert point_traj to robot-base metric coordinates by alignment.

This is useful when you already have paired data:

    source: point_traj                  # arbitrary 3D frame
    target: point_traj_base_metric      # metric robot-base 3D frame

The script fits either:
    sim3:   Y ~= s * R @ X + t
    affine: Y ~= X @ A + b

and writes the aligned source trajectory to a new HDF5 dataset.

Important: this cannot recover absolute scale from point_traj alone. It needs
paired metric targets, or a previously saved transform, because monocular /
VGGT-style 3D trajectories can have arbitrary scale and coordinate frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit/apply point_traj -> point_traj_base_metric alignment."
    )
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more HDF5 track files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively for *_tracks.hdf5.")
    parser.add_argument("--demo_ids", nargs="*", default=None, help="Optional demo ids, e.g. demo_0 demo_1.")
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--source_key", default="point_traj")
    parser.add_argument("--target_key", default="point_traj_base_metric")
    parser.add_argument("--out_key", default="point_traj_base_metric_from_point_traj")
    parser.add_argument("--method", choices=("sim3", "affine"), default="sim3")
    parser.add_argument("--trim_percentile", type=float, default=95.0, help="Residual percentile kept during robust refit; <=0 disables.")
    parser.add_argument("--trim_iters", type=int, default=2)
    parser.add_argument("--min_points", type=int, default=12)

    parser.add_argument("--transform_json", default=None, help="Optional path to save fitted transforms as JSON.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", default="gzip", help="HDF5 compression; use none/false/0 to disable.")
    return parser.parse_args()


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


def get_demo_ids(data_grp: h5py.Group, requested: Optional[Sequence[str]], max_demos: Optional[int]) -> list[str]:
    ids = sort_demo_ids(data_grp.keys())
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def flatten_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"source shape {source.shape} != target shape {target.shape}")
    if source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError(f"trajectory must have last dim 3, got {source.shape}")
    x = source.reshape(-1, 3)
    y = target.reshape(-1, 3)
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[mask], y[mask]


def fit_affine(x: np.ndarray, y: np.ndarray) -> dict:
    xa = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    b, *_ = np.linalg.lstsq(xa, y, rcond=None)
    return {"method": "affine", "matrix": b[:3].tolist(), "translation": b[3].tolist()}


def apply_affine(source: np.ndarray, transform: dict) -> np.ndarray:
    a = np.asarray(transform["matrix"], dtype=np.float64)
    b = np.asarray(transform["translation"], dtype=np.float64)
    return (np.asarray(source, dtype=np.float64).reshape(-1, 3) @ a + b).reshape(source.shape).astype(np.float32)


def fit_sim3(x: np.ndarray, y: np.ndarray) -> dict:
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    x_c = x - mu_x
    y_c = y - mu_y
    cov = (y_c.T @ x_c) / float(x.shape[0])
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    var_x = np.mean(np.sum(x_c * x_c, axis=1))
    scale = float(np.trace(np.diag(svals) @ d) / max(var_x, 1e-12))
    t = mu_y - scale * (r @ mu_x)
    return {"method": "sim3", "scale": scale, "rotation": r.tolist(), "translation": t.tolist()}


def apply_sim3(source: np.ndarray, transform: dict) -> np.ndarray:
    s = float(transform["scale"])
    r = np.asarray(transform["rotation"], dtype=np.float64)
    t = np.asarray(transform["translation"], dtype=np.float64)
    flat = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    out = s * (r @ flat.T).T + t
    return out.reshape(source.shape).astype(np.float32)


def fit_transform(x: np.ndarray, y: np.ndarray, method: str) -> dict:
    if method == "affine":
        return fit_affine(x, y)
    if method == "sim3":
        return fit_sim3(x, y)
    raise ValueError(method)


def apply_transform(source: np.ndarray, transform: dict) -> np.ndarray:
    if transform["method"] == "affine":
        return apply_affine(source, transform)
    if transform["method"] == "sim3":
        return apply_sim3(source, transform)
    raise ValueError(transform["method"])


def residuals(x: np.ndarray, y: np.ndarray, transform: dict) -> np.ndarray:
    pred = apply_transform(x.astype(np.float32), transform).reshape(-1, 3).astype(np.float64)
    return np.linalg.norm(pred - y, axis=1)


def robust_fit(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    if x.shape[0] < int(args.min_points):
        raise ValueError(f"Need at least {args.min_points} valid points, got {x.shape[0]}")

    keep = np.ones(x.shape[0], dtype=bool)
    transform = fit_transform(x, y, args.method)
    if args.trim_percentile <= 0:
        return transform, keep

    for _ in range(max(0, int(args.trim_iters))):
        err = residuals(x[keep], y[keep], transform)
        cutoff = np.nanpercentile(err, float(args.trim_percentile))
        old_keep_indices = np.where(keep)[0]
        new_keep = np.zeros_like(keep)
        new_keep[old_keep_indices[err <= cutoff]] = True
        if int(new_keep.sum()) < int(args.min_points):
            break
        keep = new_keep
        transform = fit_transform(x[keep], y[keep], args.method)
    return transform, keep


def metrics(source: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    x, y = flatten_pair(source, target)
    p, _ = flatten_pair(pred, target)
    err = np.linalg.norm(p - y, axis=1)
    ss_res = float(np.sum((p - y) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    return {
        "num_points": int(x.shape[0]),
        "rms_m": float(np.sqrt(np.mean(err ** 2))),
        "median_m": float(np.nanmedian(err)),
        "p95_m": float(np.nanpercentile(err, 95)),
        "max_m": float(np.nanmax(err)),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
    }


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, overwrite: bool, compression) -> None:
    if name in group:
        if not overwrite:
            print(f"    skip existing {name}; pass --overwrite to replace")
            return
        del group[name]
    group.create_dataset(name, data=data, compression=compression)
    print(f"    wrote {name} {data.shape} {data.dtype}")


def process_file(track_file: Path, args: argparse.Namespace) -> list[dict]:
    mode = "r" if args.dry_run else "r+"
    compression = h5_compression(args.compression)
    records: list[dict] = []

    print(f"\n[tracks] {track_file}")
    with h5py.File(track_file, mode) as f:
        if "data" not in f:
            raise KeyError(f"{track_file} has no /data group. keys={list(f.keys())}")

        demo_ids = get_demo_ids(f["data"], args.demo_ids, args.max_demos)
        print(f"  demos: {len(demo_ids)}")
        for demo_id in demo_ids:
            grp = f[f"data/{demo_id}"]
            if args.source_key not in grp:
                print(f"  [WARN] {demo_id}: missing {args.source_key}, skip")
                continue
            if args.target_key not in grp:
                print(f"  [WARN] {demo_id}: missing {args.target_key}; cannot fit metric transform, skip")
                continue

            source = np.asarray(grp[args.source_key], dtype=np.float32)
            target = np.asarray(grp[args.target_key], dtype=np.float32)
            x, y = flatten_pair(source, target)
            transform, keep = robust_fit(x, y, args)
            pred = apply_transform(source, transform)
            stats = metrics(source, target, pred)
            stats["num_fit_points"] = int(keep.sum())
            stats["num_valid_points"] = int(x.shape[0])

            print(
                f"  {demo_id}: {args.source_key} -> {args.out_key} "
                f"method={args.method} fit={stats['num_fit_points']}/{stats['num_valid_points']} "
                f"rms={stats['rms_m']:.4f}m median={stats['median_m']:.4f}m "
                f"p95={stats['p95_m']:.4f}m R2={stats['r2']:.4f}"
            )

            record = {
                "track_file": str(track_file),
                "demo_id": demo_id,
                "source_key": args.source_key,
                "target_key": args.target_key,
                "out_key": args.out_key,
                "transform": transform,
                "metrics": stats,
            }
            records.append(record)

            if args.dry_run:
                continue

            write_dataset(grp, args.out_key, pred, args.overwrite, compression)
            grp.attrs[f"{args.out_key}_method"] = args.method
            grp.attrs[f"{args.out_key}_source_key"] = args.source_key
            grp.attrs[f"{args.out_key}_target_key"] = args.target_key
            grp.attrs[f"{args.out_key}_approximate_alignment"] = True
            grp.attrs[f"{args.out_key}_units"] = "meters_if_target_metric"
            for key, value in stats.items():
                grp.attrs[f"{args.out_key}_{key}"] = value

    return records


def main() -> None:
    args = parse_args()
    all_records = []
    for track_file in collect_track_files(args):
        all_records.extend(process_file(track_file, args))

    if args.transform_json:
        out_path = Path(args.transform_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote transform report: {out_path}")


if __name__ == "__main__":
    main()
