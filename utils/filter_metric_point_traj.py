#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter existing metric point trajectories in processed track HDF5 files.

This is the lightweight companion to process_data/patch_libero_sim_depth_to_tracks.py.
It does not rerender MuJoCo depth. Instead, it reuses existing
point_traj_base_metric and applies the same moving + first-frame SOR +
teleport rejection + connected-component filtering used during metric patching.

By default, the original unfiltered metric trajectory is saved once as
point_traj_base_metric_unfiltered. Later runs read from that backup and
rewrite point_traj_base_metric, so threshold tuning is repeatable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

from process_data.patch_libero_sim_depth_to_tracks import filter_metric_point_traj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter point_traj_base_metric in existing track HDF5 files.")
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively for *_tracks.hdf5.")
    parser.add_argument("--data_group", default="data")
    parser.add_argument("--demo_ids", nargs="*", default=None)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--source_key", default="auto", help="Input key. auto uses backup_key if present, else metric_key.")
    parser.add_argument("--metric_key", default="point_traj_base_metric", help="Metric trajectory key to overwrite.")
    parser.add_argument(
        "--backup_key",
        default="point_traj_base_metric_unfiltered",
        help="Backup key for the unfiltered metric trajectory. Use 'none' to disable.",
    )
    parser.add_argument(
        "--sync_point_traj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also rewrite point_traj when it is active metric data.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--compression", default="gzip", help="HDF5 compression; use none/false/0 to disable.")

    parser.add_argument("--filter_metric_points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_motion_thresh", type=float, default=0.25)
    parser.add_argument("--filter_sor_k", type=int, default=64)
    parser.add_argument("--filter_sor_std_ratio", type=float, default=2.5)
    parser.add_argument("--filter_replace_outliers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_replace_mode", choices=("random", "nearest"), default="random")
    parser.add_argument("--filter_teleport_step_thresh", type=float, default=0.3)
    parser.add_argument("--filter_component_points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_component_k", type=int, default=5)
    parser.add_argument("--filter_component_factor", type=float, default=6.0)
    parser.add_argument("--filter_component_frame_stride", type=int, default=1)
    parser.add_argument("--filter_component_min_bad_frames", type=int, default=1)
    parser.add_argument("--filter_component_min_keep_ratio", type=float, default=0.55)
    parser.add_argument("--filter_trajectory_component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_seed", type=int, default=0)
    parser.add_argument("--filter_verbose", action="store_true")
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


def select_demo_ids(
    data_grp: h5py.Group,
    requested: Optional[Sequence[str]],
    max_demos: Optional[int],
) -> list[str]:
    ids = sort_demo_ids(data_grp.keys())
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def _decode_attr(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def is_metric_active(group: h5py.Group, metric_key: str) -> bool:
    active = str(_decode_attr(group.attrs.get("point_traj_active_source", ""))).strip()
    mode = str(_decode_attr(group.attrs.get("point_traj_mode", ""))).strip()
    if active == metric_key or mode == "metric":
        return True
    if "point_traj" in group and metric_key in group and group["point_traj"].shape == group[metric_key].shape:
        return bool(np.allclose(np.asarray(group["point_traj"]), np.asarray(group[metric_key]), equal_nan=True))
    return False


def write_dataset(group: h5py.Group, key: str, value: np.ndarray, compression) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=value, compression=compression)


def resolve_source_key(group: h5py.Group, args: argparse.Namespace) -> str:
    backup_key = None if args.backup_key.lower() in {"", "none", "false", "0"} else args.backup_key
    if args.source_key != "auto":
        return args.source_key
    if backup_key and backup_key in group:
        return backup_key
    return args.metric_key


def filter_demo(group: h5py.Group, demo_id: str, args: argparse.Namespace, compression) -> tuple[bool, dict]:
    source_key = resolve_source_key(group, args)
    if source_key not in group:
        return False, {"reason": f"missing_source:{source_key}"}

    backup_key = None if args.backup_key.lower() in {"", "none", "false", "0"} else args.backup_key
    source = np.asarray(group[source_key], dtype=np.float32)
    filtered, stats = filter_metric_point_traj(source, args)
    stats["source_key"] = source_key
    stats["metric_key"] = args.metric_key

    if args.dry_run:
        return True, stats

    if backup_key and backup_key not in group:
        write_dataset(group, backup_key, np.asarray(group[args.metric_key]), compression)
        group.attrs[f"{backup_key}_note"] = "original point_traj_base_metric before metric filtering"

    sync_active = is_metric_active(group, args.metric_key)
    write_dataset(group, args.metric_key, filtered, compression)
    group.attrs[f"{args.metric_key}_units"] = "meters"
    group.attrs[f"{args.metric_key}_coordinate_frame"] = "robot0_base"
    group.attrs[f"{args.metric_key}_filtered"] = True
    for key, value in stats.items():
        group.attrs[f"{args.metric_key}_filter_{key}"] = value

    if args.sync_point_traj and sync_active:
        write_dataset(group, "point_traj", filtered, compression)
        group.attrs["point_traj_active_source"] = args.metric_key
        group.attrs["point_traj_mode"] = "metric"
        group.attrs["point_traj_units"] = "meters"
        group.attrs["point_traj_coordinate_frame"] = "robot0_base"

    return True, stats


def process_file(track_file: Path, args: argparse.Namespace) -> tuple[int, int, int]:
    mode = "r" if args.dry_run else "r+"
    compression = h5_compression(args.compression)
    processed = 0
    skipped = 0
    changed = 0

    print(f"\n[tracks] {track_file}")
    with h5py.File(track_file, mode) as f:
        if args.data_group not in f:
            print(f"  [WARN] missing /{args.data_group}")
            return processed, skipped + 1, changed
        demo_ids = select_demo_ids(f[args.data_group], args.demo_ids, args.max_demos)
        print(f"  demos={len(demo_ids)} dry_run={args.dry_run}")
        for demo_id in demo_ids:
            ok, stats = filter_demo(f[args.data_group][demo_id], demo_id, args, compression)
            if not ok:
                skipped += 1
                print(f"  [SKIP] {demo_id}: {stats['reason']}")
                continue
            processed += 1
            changed += int(stats.get("changed_count", 0))
            print(
                f"  {demo_id}: src={stats.get('source_key')} "
                f"inliers={stats.get('inlier_count', 0)}/{stats.get('num_points', 0)} "
                f"teleport_bad={stats.get('teleport_bad_count', 0)} "
                f"component_bad={stats.get('component_bad_count', 0)} "
                f"max_step={stats.get('max_step_before_m', 0.0):.4f}->{stats.get('max_step_after_m', 0.0):.4f}m"
            )
    return processed, skipped, changed


def main() -> None:
    args = parse_args()
    files = collect_track_files(args)
    total_processed = 0
    total_skipped = 0
    total_changed = 0
    print(f"[INFO] files={len(files)}")
    for track_file in files:
        processed, skipped, changed = process_file(track_file, args)
        total_processed += processed
        total_skipped += skipped
        total_changed += changed
    print(
        f"\n[DONE] processed_demos={total_processed} "
        f"skipped={total_skipped} changed_points={total_changed}"
    )


if __name__ == "__main__":
    main()
