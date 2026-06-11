#!/usr/bin/env python3
"""Build a ManiSkill gripper-rigid metric flow from query pixels and TCP poses.

Some early ManiSkill processed files saved correct ``query_xy_t0`` points but
incorrect ``track2d`` / projected ``point_traj``. For gripper-centric demos this
utility provides a visualization-oriented repair:

1. Lift ``query_xy_t0`` (or ``grid_points_xy``) from frame 0 using RGB-D depth.
2. Express those points in the initial TCP frame.
3. Move them through the episode with ``obs/extra/tcp_pose``.

The result is a metric trajectory in the ManiSkill world frame. It is meant for
gripper-point visualization when tracker output is known to be corrupted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

from process_data.convert_track2d_to_point_traj_base_metric import (
    choose_patch_candidate,
    track_xy_to_image_pixels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True, help="Processed ManiSkill *_tracks.hdf5 file.")
    p.add_argument("--rgbd_h5", required=True, help="Matching ManiSkill RGB-D replay HDF5.")
    p.add_argument("--demo_ids", nargs="*", default=None)
    p.add_argument("--traj_ids", nargs="*", default=None)
    p.add_argument("--camera_name", default="base_camera")
    p.add_argument("--query_key", default="query_xy_t0", choices=("query_xy_t0", "grid_points_xy", "p0_uv"))
    p.add_argument(
        "--query_uv_mode",
        default="518",
        choices=("auto", "pixels", "518", "normalized"),
        help="Coordinate scale for query_key. ManiSkill SpaTracker queries are usually in 518x518 model coordinates.",
    )
    p.add_argument("--out_key", default="point_traj_tcp_rigid_metric")
    p.add_argument("--depth_patch_radius", type=int, default=3)
    p.add_argument("--depth_patch_percentile", type=float, default=20.0)
    p.add_argument("--replace_point_traj_base_metric", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--compression", default="gzip")
    return p.parse_args()


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def write_dataset(group: h5py.Group, key: str, value: np.ndarray, overwrite: bool, compression) -> None:
    if key in group:
        if not overwrite:
            print(f"    skip existing {key}; pass --overwrite")
            return
        del group[key]
    group.create_dataset(key, data=value, compression=compression)
    print(f"    wrote {key} {value.shape} {value.dtype}")


def sort_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        tail = str(x).split("_")[-1]
        return (0, int(tail)) if tail.isdigit() else (1, str(x))

    return sorted([str(x) for x in ids], key=key_fn)


def numeric_suffix(text: str, fallback: int) -> int:
    tail = str(text).split("_")[-1]
    return int(tail) if tail.isdigit() else int(fallback)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose7_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    T[:3, 3] = pose[:3]
    return T


def read_raw(rgbd: h5py.File, traj_id: str, camera_name: str):
    depth_key = f"{traj_id}/obs/sensor_data/{camera_name}/depth"
    K_key = f"{traj_id}/obs/sensor_param/{camera_name}/intrinsic_cv"
    E_key = f"{traj_id}/obs/sensor_param/{camera_name}/extrinsic_cv"
    tcp_key = f"{traj_id}/obs/extra/tcp_pose"
    for key in (depth_key, K_key, E_key, tcp_key):
        if key not in rgbd:
            raise KeyError(f"Missing {key}")

    depth = np.squeeze(np.asarray(rgbd[depth_key], dtype=np.float32))
    if np.nanmedian(depth[depth > 0]) > 20.0:
        depth = depth / 1000.0
    K = np.asarray(rgbd[K_key], dtype=np.float32)
    E34 = np.asarray(rgbd[E_key], dtype=np.float32)
    w2c = np.tile(np.eye(4, dtype=np.float32), (E34.shape[0], 1, 1))
    w2c[:, :3, :4] = E34
    c2w = np.linalg.inv(w2c).astype(np.float32)
    tcp = np.asarray(rgbd[tcp_key], dtype=np.float32)
    return depth.astype(np.float32), K, w2c, c2w, tcp


def lift_initial_queries(
    query_xy: np.ndarray,
    depth0: np.ndarray,
    K0: np.ndarray,
    c2w0: np.ndarray,
    radius: int,
    percentile: float,
    uv_mode_requested: str,
):
    xy_px, uv_mode = track_xy_to_image_pixels(
        query_xy[None, :, :2],
        depth0.shape[0],
        depth0.shape[1],
        uv_mode_requested,
    )
    xy_px = xy_px[0]

    points = np.empty((xy_px.shape[0], 3), dtype=np.float32)
    sampled_xy = np.empty_like(xy_px, dtype=np.float32)
    offsets = np.empty((xy_px.shape[0],), dtype=np.float32)
    for i, xy in enumerate(xy_px):
        xy_i, _z, world_i, offset_i = choose_patch_candidate(
            depth=depth0,
            xy_one=xy,
            K=K0,
            T_base_cam=c2w0,
            mode="patch_percentile",
            radius=radius,
            percentile=percentile,
            prev_base=None,
            temporal_pixel_weight=0.0,
        )
        sampled_xy[i] = xy_i
        points[i] = world_i
        offsets[i] = offset_i
    stats = {
        "uv_mode": uv_mode,
        "sample_pixel_offset_mean": float(np.nanmean(offsets)),
        "sample_pixel_offset_p95": float(np.nanpercentile(offsets, 95)),
        "sample_pixel_offset_max": float(np.nanmax(offsets)),
    }
    return points, sampled_xy, stats


def project_world_to_pixels(points_world: np.ndarray, intrinsics: np.ndarray, w2c_traj: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c_traj = np.asarray(w2c_traj, dtype=np.float32)
    t_len, num_points, _ = points_world.shape
    out = np.empty((t_len, num_points, 2), dtype=np.float32)
    for t in range(t_len):
        homo = np.concatenate([points_world[t], np.ones((num_points, 1), dtype=np.float32)], axis=1)
        cam = (w2c_traj[t] @ homo.T).T[:, :3]
        z = np.clip(cam[:, 2], 1e-6, None)
        out[t, :, 0] = intrinsics[t, 0, 0] * cam[:, 0] / z + intrinsics[t, 0, 2]
        out[t, :, 1] = intrinsics[t, 1, 1] * cam[:, 1] / z + intrinsics[t, 1, 2]
    return out


def process_demo(group: h5py.Group, rgbd: h5py.File, demo_id: str, traj_id: str, args: argparse.Namespace):
    if args.query_key not in group:
        raise KeyError(f"{demo_id}: missing {args.query_key}")

    depth, K, w2c, c2w, tcp = read_raw(rgbd, traj_id, args.camera_name)
    query = np.asarray(group[args.query_key], dtype=np.float32)
    if query.ndim != 2 or query.shape[-1] < 2:
        raise ValueError(f"{demo_id}: {args.query_key} must be [N,2+], got {query.shape}")

    t_len = min(int(group["frames_rgb"].shape[0]), depth.shape[0], tcp.shape[0])
    initial_world, sampled_xy, stats = lift_initial_queries(
        query_xy=query,
        depth0=depth[0],
        K0=K[0],
        c2w0=c2w[0],
        radius=int(args.depth_patch_radius),
        percentile=float(args.depth_patch_percentile),
        uv_mode_requested=str(args.query_uv_mode),
    )

    tcp_T = np.stack([pose7_to_mat(x) for x in tcp[:t_len]], axis=0).astype(np.float32)
    inv_tcp0 = np.linalg.inv(tcp_T[0])
    initial_h = np.concatenate([initial_world, np.ones((initial_world.shape[0], 1), dtype=np.float32)], axis=1)
    local_h = (inv_tcp0 @ initial_h.T).T
    traj = np.empty((t_len, initial_world.shape[0], 3), dtype=np.float32)
    for t in range(t_len):
        traj[t] = (tcp_T[t] @ local_h.T).T[:, :3]
    track2d = project_world_to_pixels(traj, K[:t_len], w2c[:t_len])

    print(
        f"  {demo_id} <- {traj_id}: {args.query_key}->{args.out_key} {traj.shape} "
        f"({stats['uv_mode']} offset mean/p95/max="
        f"{stats['sample_pixel_offset_mean']:.2f}/{stats['sample_pixel_offset_p95']:.2f}/{stats['sample_pixel_offset_max']:.2f}px)"
    )
    return traj, sampled_xy, track2d, stats


def main() -> None:
    args = parse_args()
    compression = h5_compression(args.compression)
    mode = "r" if args.dry_run else "r+"
    with h5py.File(args.rgbd_h5, "r") as rgbd, h5py.File(args.tracks, mode) as f:
        demo_ids = sort_ids(args.demo_ids or f["data"].keys())
        raw_trajs = sort_ids(args.traj_ids or [k for k in rgbd.keys() if str(k).startswith("traj_")])
        for i, demo_id in enumerate(demo_ids):
            group = f[f"data/{demo_id}"]
            raw_idx = numeric_suffix(demo_id, i)
            traj_id = raw_trajs[0] if len(raw_trajs) == 1 else f"traj_{raw_idx}"
            if traj_id not in rgbd:
                print(f"  [WARN] {demo_id}: missing raw {traj_id}; skip")
                continue
            try:
                traj, sampled_xy, track2d, stats = process_demo(group, rgbd, demo_id, traj_id, args)
            except Exception as exc:
                print(f"  [WARN] {demo_id}: {exc!r}")
                continue
            if args.dry_run:
                continue
            write_dataset(group, args.out_key, traj, args.overwrite, compression)
            write_dataset(group, f"{args.out_key}_sampled_xy_t0", sampled_xy, args.overwrite, compression)
            write_dataset(group, f"{args.out_key}_track2d", track2d, args.overwrite, compression)
            group.attrs[f"{args.out_key}_units"] = "meters"
            group.attrs[f"{args.out_key}_coordinate_frame"] = "maniskill_world"
            group.attrs[f"{args.out_key}_source"] = "query_xy_t0 lifted at t0 and moved rigidly by obs/extra/tcp_pose"
            for key, value in stats.items():
                group.attrs[f"{args.out_key}_{key}"] = value
            if args.replace_point_traj_base_metric:
                write_dataset(group, "point_traj_base_metric", traj, True, compression)
                group.attrs["point_traj_base_metric_units"] = "meters"
                group.attrs["point_traj_base_metric_coordinate_frame"] = "maniskill_world"
                group.attrs["point_traj_base_metric_source"] = args.out_key


if __name__ == "__main__":
    main()
