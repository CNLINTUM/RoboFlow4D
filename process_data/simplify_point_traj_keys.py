#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export processed HDF5 tracks with a minimal point trajectory naming scheme.

The public / training-facing convention is:

  point_traj               : original SpaTracker/VGGT-scale trajectory
  point_traj_metric        : recommended metric trajectory, in meters

All other point_traj* variants can be omitted from the exported files while
preserving the rest of each demo group, such as RGB frames, actions, track2d,
visibility, robot states, and model features.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively.")
    parser.add_argument("--out_root", default=None, help="Write a clean copy under this root.")
    parser.add_argument("--in_place", action="store_true", help="Rewrite keys inside the input files.")
    parser.add_argument("--data_group", default="data")
    parser.add_argument("--demo_ids", nargs="*", default=None)
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--spatracker_source_key", default="point_traj_spatracker")
    parser.add_argument("--metric_source_key", default="point_traj_base_metric_from_spatracker_stage_aligned")
    parser.add_argument("--metric_out_key", default="point_traj_metric")
    parser.add_argument(
        "--drop_incomplete_demo_groups",
        action="store_true",
        help=(
            "When exporting a clean copy, omit demo-like groups that do not contain all "
            "required trajectory source keys."
        ),
    )
    parser.add_argument(
        "--keep_extra_point_traj_keys",
        action="store_true",
        help="Keep legacy point_traj* datasets. By default they are omitted from clean copies or deleted in-place.",
    )
    parser.add_argument(
        "--clean_point_traj_attrs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace verbose point_traj* attrs with a compact naming summary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--compression", default="gzip", help="Compression for newly written trajectory keys.")
    return parser.parse_args()


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def sort_demo_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        tail = str(x).split("_")[-1]
        return (0, int(tail)) if tail.isdigit() else (1, str(x))

    return sorted([str(x) for x in ids], key=key_fn)


def collect_track_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    roots = [Path(x).resolve() for x in args.tracks_root or []]
    paths: list[Path] = []
    for item in args.tracks or []:
        p = Path(item)
        if p.is_dir():
            roots.append(p.resolve())
            paths.extend(sorted(p.rglob("*_tracks.hdf5")))
        else:
            paths.append(p)
    for root in roots:
        paths.extend(sorted(root.rglob("*_tracks.hdf5")))

    uniq: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise ValueError("No *_tracks.hdf5 files found.")
    return uniq, roots


def relative_to_any(path: Path, roots: Sequence[Path]) -> Path:
    rp = path.resolve()
    for root in sorted(roots, key=lambda x: len(str(x)), reverse=True):
        try:
            return rp.relative_to(root)
        except ValueError:
            continue
    return Path(path.name)


def get_demo_ids(data_group: h5py.Group, requested: Optional[Sequence[str]], max_demos: Optional[int]) -> list[str]:
    ids = sort_demo_ids(data_group.keys())
    if requested:
        keep = set(str(x) for x in requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def write_dataset(group: h5py.Group, key: str, value: np.ndarray, overwrite: bool, compression) -> None:
    if key in group:
        if not overwrite:
            raise ValueError(f"{group.name}/{key} exists; pass --overwrite")
        del group[key]
    group.create_dataset(key, data=np.asarray(value), compression=compression)


def copy_attrs(src: h5py.Group | h5py.File, dst: h5py.Group | h5py.File, clean_point_attrs: bool) -> None:
    for key, value in src.attrs.items():
        if clean_point_attrs and str(key).startswith("point_traj"):
            continue
        dst.attrs[key] = value


def set_clean_attrs(group: h5py.Group, args: argparse.Namespace, source_attrs=None) -> None:
    source_attrs = source_attrs or group.attrs

    def attr_value(name: str, default):
        return source_attrs[name] if name in source_attrs else default

    keys_available = ["point_traj", args.metric_out_key]
    group.attrs["point_traj_active_source"] = "point_traj"
    group.attrs["point_traj_mode"] = "spatracker"
    group.attrs["point_traj_units"] = "spatracker_v2_relative"
    group.attrs["point_traj_coordinate_frame"] = "spatracker_v2_or_vggt"
    group.attrs[f"{args.metric_out_key}_source_key"] = args.metric_source_key
    group.attrs[f"{args.metric_out_key}_alignment"] = attr_value(
        f"{args.metric_source_key}_alignment", "robust_sim3"
    )
    group.attrs[f"{args.metric_out_key}_units"] = "meters"
    group.attrs[f"{args.metric_out_key}_coordinate_frame"] = attr_value(
        f"{args.metric_source_key}_coordinate_frame", "robot0_base"
    )
    group.attrs["point_traj_keys_available"] = ",".join(keys_available)


def has_required_sources(group: h5py.Group, args: argparse.Namespace) -> tuple[bool, bool]:
    has_spatracker = args.spatracker_source_key in group
    has_metric = args.metric_source_key in group
    return has_spatracker, has_metric


def should_skip_legacy_point_key(name: str, args: argparse.Namespace) -> bool:
    if args.keep_extra_point_traj_keys:
        return False
    return str(name).startswith("point_traj")


def is_demo_like_group(group: h5py.Group) -> bool:
    return any(str(name).startswith("point_traj") for name in group.keys())


def copy_group_clean(src: h5py.Group, dst: h5py.Group, args: argparse.Namespace, compression) -> dict[str, int]:
    stats = {
        "groups": 1,
        "datasets": 0,
        "standardized": 0,
        "missing_metric": 0,
        "missing_spatracker": 0,
        "dropped_incomplete": 0,
    }
    has_spatracker, has_metric = has_required_sources(src, args)
    has_all_required = has_spatracker and has_metric
    if args.drop_incomplete_demo_groups and is_demo_like_group(src) and not has_all_required:
        stats["groups"] = 0
        stats["dropped_incomplete"] = 1
        stats["missing_spatracker"] = int(not has_spatracker)
        stats["missing_metric"] = int(not has_metric)
        stats["drop_group"] = 1
        return stats

    copy_attrs(src, dst, clean_point_attrs=bool(args.clean_point_traj_attrs))

    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(name)
            child_stats = copy_group_clean(item, child, args, compression)
            if child_stats.pop("drop_group", 0):
                del dst[name]
            for key, value in child_stats.items():
                stats[key] = stats.get(key, 0) + value
            continue

        if should_skip_legacy_point_key(name, args):
            continue
        src.copy(name, dst)
        stats["datasets"] += 1

    if has_spatracker or has_metric:
        if has_spatracker:
            write_dataset(dst, "point_traj", np.asarray(src[args.spatracker_source_key]), True, compression)
            stats["datasets"] += 1
        else:
            stats["missing_spatracker"] += 1
        if has_metric:
            write_dataset(dst, args.metric_out_key, np.asarray(src[args.metric_source_key]), True, compression)
            stats["datasets"] += 1
        else:
            stats["missing_metric"] += 1
        if has_all_required:
            set_clean_attrs(dst, args, source_attrs=src.attrs)
            stats["standardized"] += 1
    return stats


def export_clean_copy(src_file: Path, dst_file: Path, args: argparse.Namespace, compression) -> dict[str, int]:
    if dst_file.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_file} exists; pass --overwrite")
        dst_file.unlink()
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = dst_file.with_suffix(dst_file.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    with h5py.File(src_file, "r") as src, h5py.File(tmp_file, "w") as dst:
        copy_attrs(src, dst, clean_point_attrs=bool(args.clean_point_traj_attrs))
        stats = {
            "groups": 0,
            "datasets": 0,
            "standardized": 0,
            "missing_metric": 0,
            "missing_spatracker": 0,
            "dropped_incomplete": 0,
        }
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                child = dst.create_group(name)
                child_stats = copy_group_clean(item, child, args, compression)
                if child_stats.pop("drop_group", 0):
                    del dst[name]
                for key, value in child_stats.items():
                    stats[key] = stats.get(key, 0) + value
            else:
                if should_skip_legacy_point_key(name, args):
                    continue
                src.copy(name, dst)
                stats["datasets"] += 1
    shutil.move(str(tmp_file), str(dst_file))
    return stats


def standardize_in_place(track_file: Path, args: argparse.Namespace, compression) -> dict[str, int]:
    stats = {
        "standardized": 0,
        "missing_metric": 0,
        "missing_spatracker": 0,
        "deleted": 0,
    }
    mode = "r" if args.dry_run else "r+"
    with h5py.File(track_file, mode) as f:
        if args.data_group not in f:
            raise KeyError(f"{track_file} has no /{args.data_group}")
        demo_ids = get_demo_ids(f[args.data_group], args.demo_ids, args.max_demos)
        for demo_id in demo_ids:
            group = f[args.data_group][demo_id]
            has_spatracker, has_metric = has_required_sources(group, args)
            if not has_spatracker:
                stats["missing_spatracker"] += 1
            if not has_metric:
                stats["missing_metric"] += 1
            if args.dry_run or not (has_spatracker and has_metric):
                continue
            spatracker = np.asarray(group[args.spatracker_source_key])
            metric = np.asarray(group[args.metric_source_key])
            source_attrs = dict(group.attrs.items())
            for key in list(group.keys()):
                if should_skip_legacy_point_key(key, args):
                    del group[key]
                    stats["deleted"] += 1
            write_dataset(group, "point_traj", spatracker, True, compression)
            write_dataset(group, args.metric_out_key, metric, True, compression)
            if args.clean_point_traj_attrs:
                for attr in list(group.attrs.keys()):
                    if str(attr).startswith("point_traj"):
                        del group.attrs[attr]
            set_clean_attrs(group, args, source_attrs=source_attrs)
            stats["standardized"] += 1
    return stats


def main() -> None:
    args = parse_args()
    if bool(args.out_root) == bool(args.in_place):
        raise ValueError("Choose exactly one of --out_root or --in_place.")
    compression = h5_compression(args.compression)
    track_files, roots = collect_track_files(args)
    totals: dict[str, int] = {}

    for src_file in track_files:
        if args.out_root:
            rel = relative_to_any(src_file, roots)
            dst_file = Path(args.out_root) / rel
            print(f"[copy] {src_file} -> {dst_file}")
            if args.dry_run:
                continue
            stats = export_clean_copy(src_file, dst_file, args, compression)
        else:
            print(f"[in-place] {src_file}")
            stats = standardize_in_place(src_file, args, compression)
        print("  " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value

    print("[DONE] " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))


if __name__ == "__main__":
    main()
