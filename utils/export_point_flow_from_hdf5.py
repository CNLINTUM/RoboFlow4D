#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export per-frame 2D/3D point positions and flows from *_tracks.hdf5 files.

For each demo, save:
  track2d.npy      [T, N, 2]
  flow2d.npy       [T, N, 2] where flow2d[t] = track2d[t+1] - track2d[t], last frame is 0
  point_traj.npy   [T, N, 3] if present
  flow3d.npy       [T, N, 3] if point_traj exists
  meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files or directories. Directories are searched recursively for *_tracks.hdf5.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save exported numpy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def iter_h5_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix in {".h5", ".hdf5"}:
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("*_tracks.hdf5")))
            continue
        raise FileNotFoundError(f"Input not found: {path}")
    uniq = sorted({p.resolve() for p in files})
    if not uniq:
        raise RuntimeError("No *_tracks.hdf5 files found in inputs")
    return uniq


def forward_diff_keep_last_zero(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] >= 2:
        out[:-1] = arr[1:] - arr[:-1]
    return out


def save_array(path: Path, arr: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def export_one_demo(h5_path: Path, demo_id: str, demo_grp: h5py.Group, out_dir: Path, overwrite: bool) -> None:
    demo_out = out_dir / h5_path.stem / str(demo_id)
    demo_out.mkdir(parents=True, exist_ok=True)

    meta = {
        "source_h5": str(h5_path),
        "demo_id": str(demo_id),
        "has_tracks": bool(demo_grp.attrs.get("has_tracks", False)),
        "keys": list(demo_grp.keys()),
    }

    if "track2d" in demo_grp:
        track2d = np.asarray(demo_grp["track2d"][:], dtype=np.float32)
        if track2d.ndim != 3 or track2d.shape[-1] < 2:
            raise ValueError(f"Unexpected track2d shape in {h5_path} demo={demo_id}: {track2d.shape}")
        track2d = track2d[..., :2]
        flow2d = forward_diff_keep_last_zero(track2d)
        save_array(demo_out / "track2d.npy", track2d, overwrite=overwrite)
        save_array(demo_out / "flow2d.npy", flow2d, overwrite=overwrite)
        meta["track2d_shape"] = list(track2d.shape)
        meta["flow2d_shape"] = list(flow2d.shape)

    if "point_traj" in demo_grp:
        point_traj = np.asarray(demo_grp["point_traj"][:], dtype=np.float32)
        if point_traj.ndim != 3 or point_traj.shape[-1] < 3:
            raise ValueError(f"Unexpected point_traj shape in {h5_path} demo={demo_id}: {point_traj.shape}")
        point_traj = point_traj[..., :3]
        flow3d = forward_diff_keep_last_zero(point_traj)
        save_array(demo_out / "point_traj.npy", point_traj, overwrite=overwrite)
        save_array(demo_out / "flow3d.npy", flow3d, overwrite=overwrite)
        meta["point_traj_shape"] = list(point_traj.shape)
        meta["flow3d_shape"] = list(flow3d.shape)

    meta_path = demo_out / "meta.json"
    if overwrite or not meta_path.exists():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    h5_files = iter_h5_files(args.inputs)

    print(f"[INFO] Found {len(h5_files)} track hdf5 file(s)")
    for h5_path in h5_files:
        print(f"[READ] {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                print(f"[WARN] skip {h5_path}: no /data group")
                continue
            for demo_id, demo_grp in f["data"].items():
                export_one_demo(
                    h5_path=h5_path,
                    demo_id=demo_id,
                    demo_grp=demo_grp,
                    out_dir=out_dir,
                    overwrite=bool(args.overwrite),
                )
                print(f"[OK] {h5_path.name} demo={demo_id}")


if __name__ == "__main__":
    main()
