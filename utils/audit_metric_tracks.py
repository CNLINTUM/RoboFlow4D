#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and optionally delete low-quality metric trajectory demos.

The metric conversion can produce valid HDF5 keys while still leaving poor
training examples: duplicated replacement tracks, large single-frame jumps, or
too few inlier trajectories after filtering. This script reports those cases
and only deletes groups when both --delete and --yes are set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


DEFAULT_REQUIRED_KEYS = (
    "frames_rgb",
    "track2d",
    "vis",
    "point_traj_spatracker",
    "sim_depths",
    "sim_intrinsics",
    "sim_T_base_cam",
    "point_traj_base_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report or delete dirty metric-scale HDF5 demos.")
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively.")
    parser.add_argument("--data_group", default="data")
    parser.add_argument("--demo_ids", nargs="*", default=None)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--traj_key", default="point_traj_base_metric")
    parser.add_argument("--required_keys", nargs="+", default=list(DEFAULT_REQUIRED_KEYS))
    parser.add_argument("--round_decimals", type=int, default=3, help="Rounding for duplicate first-frame points.")

    parser.add_argument("--max_step_m", type=float, default=0.3001)
    parser.add_argument("--min_unique0_ratio", type=float, default=0.60)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.60)
    parser.add_argument("--max_changed_ratio", type=float, default=0.40)
    parser.add_argument("--require_filter_attrs", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--delete", action="store_true", help="Delete bad demo groups.")
    parser.add_argument("--yes", action="store_true", help="Required together with --delete.")
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


def select_demo_ids(data_grp: h5py.Group, requested: Optional[Sequence[str]], max_demos: Optional[int]) -> list[str]:
    ids = sort_demo_ids(data_grp.keys())
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def attr_float(group: h5py.Group, key: str) -> Optional[float]:
    if key not in group.attrs:
        return None
    value = group.attrs[key]
    try:
        return float(value)
    except Exception:
        return None


def audit_demo(group: h5py.Group, args: argparse.Namespace) -> dict:
    row: dict[str, object] = {}
    reasons: list[str] = []

    missing = [key for key in args.required_keys if key not in group]
    if missing:
        reasons.extend([f"missing:{key}" for key in missing])
        row["bad_reasons"] = ";".join(reasons)
        row["is_bad"] = True
        return row

    traj = np.asarray(group[args.traj_key], dtype=np.float32)
    row["shape"] = "x".join(str(x) for x in traj.shape)
    if traj.ndim != 3 or traj.shape[-1] != 3 or traj.shape[0] < 2 or traj.shape[1] == 0:
        reasons.append("bad_shape")
        row["bad_reasons"] = ";".join(reasons)
        row["is_bad"] = True
        return row

    num_points = int(traj.shape[1])
    finite = bool(np.isfinite(traj).all())
    if not finite:
        reasons.append("nonfinite")
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    max_step = float(np.nanmax(steps))
    max_step_per_point = np.nanmax(steps, axis=0)
    unique0 = int(np.unique(np.round(traj[0], int(args.round_decimals)), axis=0).shape[0])
    unique0_ratio = float(unique0 / max(num_points, 1))

    inlier = attr_float(group, f"{args.traj_key}_filter_inlier_count")
    changed = attr_float(group, f"{args.traj_key}_filter_changed_count")
    if inlier is None and args.require_filter_attrs:
        reasons.append("missing_filter_inlier_attr")
    if changed is None and args.require_filter_attrs:
        reasons.append("missing_filter_changed_attr")

    inlier_ratio = float(inlier / num_points) if inlier is not None else np.nan
    changed_ratio = float(changed / num_points) if changed is not None else np.nan

    if max_step > float(args.max_step_m):
        reasons.append("max_step")
    if unique0_ratio < float(args.min_unique0_ratio):
        reasons.append("low_unique0")
    if inlier is not None and inlier_ratio < float(args.min_inlier_ratio):
        reasons.append("low_inlier")
    if changed is not None and changed_ratio > float(args.max_changed_ratio):
        reasons.append("too_many_replaced")

    row.update(
        {
            "finite": finite,
            "num_points": num_points,
            "max_step_m": max_step,
            "p95_point_max_step_m": float(np.nanpercentile(max_step_per_point, 95)),
            "unique0": unique0,
            "unique0_ratio": unique0_ratio,
            "filter_inlier_count": "" if inlier is None else int(inlier),
            "filter_inlier_ratio": "" if inlier is None else inlier_ratio,
            "filter_changed_count": "" if changed is None else int(changed),
            "filter_changed_ratio": "" if changed is None else changed_ratio,
            "bad_reasons": ";".join(reasons),
            "is_bad": bool(reasons),
        }
    )
    return row


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "track_file",
        "demo_id",
        "is_bad",
        "bad_reasons",
        "shape",
        "finite",
        "num_points",
        "max_step_m",
        "p95_point_max_step_m",
        "unique0",
        "unique0_ratio",
        "filter_inlier_count",
        "filter_inlier_ratio",
        "filter_changed_count",
        "filter_changed_ratio",
        "action",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def process_file(track_file: Path, args: argparse.Namespace) -> tuple[int, int, list[dict]]:
    mode = "r+" if args.delete else "r"
    rows: list[dict] = []
    total = 0
    bad = 0
    with h5py.File(track_file, mode) as f:
        if args.data_group not in f:
            rows.append(
                {
                    "track_file": str(track_file),
                    "demo_id": "",
                    "is_bad": True,
                    "bad_reasons": f"missing_data_group:{args.data_group}",
                    "action": "file_skipped",
                }
            )
            return total, bad + 1, rows
        data_grp = f[args.data_group]
        to_delete: list[str] = []
        for demo_id in select_demo_ids(data_grp, args.demo_ids, args.max_demos):
            total += 1
            row = audit_demo(data_grp[demo_id], args)
            row["track_file"] = str(track_file)
            row["demo_id"] = str(demo_id)
            if row.get("is_bad"):
                bad += 1
                row["action"] = "delete" if args.delete else "report"
                if args.delete:
                    to_delete.append(str(demo_id))
            else:
                row["action"] = "keep"
            rows.append(row)
        for demo_id in to_delete:
            del data_grp[demo_id]
    return total, bad, rows


def main() -> None:
    args = parse_args()
    if args.delete and not args.yes:
        raise SystemExit("Refusing to delete without --yes. Re-run with --delete --yes after checking the report.")

    files = collect_track_files(args)
    all_rows: list[dict] = []
    total = 0
    bad = 0
    print(f"[INFO] files={len(files)} mode={'delete' if args.delete else 'report'} traj_key={args.traj_key}")
    for track_file in files:
        n, b, rows = process_file(track_file, args)
        total += n
        bad += b
        all_rows.extend(rows)
        if b:
            print(f"[BAD] {track_file} bad={b}/{n}")
            for row in [r for r in rows if r.get("is_bad")][:5]:
                print(f"  {row.get('demo_id')}: {row.get('bad_reasons')} ({row.get('action')})")
            if b > 5:
                print(f"  ... {b - 5} more")
    if args.report_csv:
        write_report(all_rows, Path(args.report_csv))
        print(f"[OK] report_csv={args.report_csv}")
    print(f"[DONE] inspected={total} bad={bad} keep={total - bad} action={'deleted' if args.delete else 'reported'}")


if __name__ == "__main__":
    main()
