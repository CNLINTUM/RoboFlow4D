#!/usr/bin/env python3
"""Build clean metric-scale LIBERO track files with one command.

This is a thin pipeline wrapper around the lower-level processing scripts:

1. replay LIBERO simulator states to add RGB-D metric anchors;
2. calibrate smooth SpaTracker trajectories into metric coordinates;
3. apply stage-level alignment;
4. export a compact training-facing HDF5 copy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks_root", required=True, help="Processed LIBERO tracks root from process_libero_hdf5.py.")
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"), help="LIBERO repo root, or set LIBERO_ROOT.")
    parser.add_argument("--out_root", required=True, help="Output root for clean training HDF5 tracks.")
    parser.add_argument("--demo_ids", nargs="*", default=None, help="Optional demo ids to process.")
    parser.add_argument("--max_demos", type=int, default=None, help="Debug option: process only the first N demos per file.")
    parser.add_argument("--camera_name", default=None, help="Optional LIBERO camera override.")
    parser.add_argument("--stage_align_mode", default="translation", choices=("translation", "sim3", "endpoint_sim3"))
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_depth", action="store_true", help="Skip simulator RGB-D replay if metric anchors already exist.")
    parser.add_argument("--skip_global_align", action="store_true")
    parser.add_argument("--skip_stage_align", action="store_true")
    parser.add_argument("--keep_extra_point_traj_keys", action="store_true")
    return parser.parse_args()


def add_common_demo_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.demo_ids:
        cmd.extend(["--demo_ids", *[str(x) for x in args.demo_ids]])
    if args.max_demos is not None:
        cmd.extend(["--max_demos", str(args.max_demos)])
    return cmd


def add_overwrite(cmd: list[str], overwrite: bool) -> list[str]:
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n[{name}]")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    if not args.libero_root and not args.skip_depth:
        raise ValueError("Pass --libero_root or set LIBERO_ROOT, unless --skip_depth is used.")

    python = sys.executable
    tracks_root = str(Path(args.tracks_root))
    out_root = str(Path(args.out_root))
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    if not args.skip_depth:
        cmd = [
            python,
            str(HERE / "patch_libero_sim_depth_to_tracks.py"),
            "--tracks_root",
            tracks_root,
            "--libero_root",
            str(Path(args.libero_root)),
            "--point_traj_metric",
        ]
        if args.camera_name:
            cmd.extend(["--camera_name", str(args.camera_name)])
        if args.dry_run:
            cmd.append("--dry_run")
        add_common_demo_args(cmd, args)
        add_overwrite(cmd, args.overwrite)
        run_step("1/4 RGB-D metric anchors", cmd, env)

    if not args.skip_global_align:
        cmd = [
            python,
            str(HERE / "calibrate_spatracker_to_metric.py"),
            "--tracks_root",
            tracks_root,
            "--compression",
            str(args.compression),
        ]
        if args.dry_run:
            cmd.append("--dry_run")
        add_common_demo_args(cmd, args)
        add_overwrite(cmd, args.overwrite)
        run_step("2/4 global metric calibration", cmd, env)

    if not args.skip_stage_align:
        cmd = [
            python,
            str(HERE / "stage_align_metric_from_spatracker.py"),
            "--tracks_root",
            tracks_root,
            "--align_mode",
            str(args.stage_align_mode),
            "--compression",
            str(args.compression),
        ]
        if args.dry_run:
            cmd.append("--dry_run")
        add_common_demo_args(cmd, args)
        add_overwrite(cmd, args.overwrite)
        run_step("3/4 stage metric alignment", cmd, env)

    cmd = [
        python,
        str(HERE / "simplify_point_traj_keys.py"),
        "--tracks_root",
        tracks_root,
        "--out_root",
        out_root,
        "--compression",
        str(args.compression),
        "--drop_incomplete_demo_groups",
    ]
    if args.keep_extra_point_traj_keys:
        cmd.append("--keep_extra_point_traj_keys")
    if args.dry_run:
        cmd.append("--dry_run")
    add_common_demo_args(cmd, args)
    add_overwrite(cmd, args.overwrite)
    run_step("4/4 clean training export", cmd, env)

    print(f"\n[DONE] clean metric tracks: {out_root}")


if __name__ == "__main__":
    main()
