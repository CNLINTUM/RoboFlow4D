#!/usr/bin/env python3
"""Patch ManiSkill processed tracks with simulator RGB-D calibration.

The normal SpaTracker preprocessing may only save RGB frames and tracker-scale
3D points. This utility reads a ManiSkill replay generated with
``--obs-mode rgbd`` and writes metric depth, intrinsics, camera poses, and a
metric 3D trajectory lifted from ``track2d`` back into the processed track HDF5.

The lifted trajectory is saved in the ManiSkill world frame, in meters. By
default it is written both as ``point_traj_world_metric`` and
``point_traj_base_metric`` so existing visualization/training utilities can use
the familiar metric key.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", nargs="*", required=True, help="Processed ManiSkill *_tracks.hdf5 files.")
    p.add_argument("--rgbd_h5", required=True, help="ManiSkill replay HDF5 saved with --obs-mode rgbd.")
    p.add_argument("--demo_ids", nargs="*", default=None, help="Processed demo ids, e.g. 0 demo_0.")
    p.add_argument("--traj_ids", nargs="*", default=None, help="Raw replay trajectory ids, e.g. traj_0.")
    p.add_argument("--camera_name", default="base_camera")
    p.add_argument("--track_key", default="track2d")
    p.add_argument("--uv_mode", default="auto", choices=("auto", "pixels", "518", "normalized"))
    p.add_argument("--out_key", default="point_traj_base_metric")
    p.add_argument("--world_key", default="point_traj_world_metric")
    p.add_argument("--replace_point_traj", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--compression", default="gzip", help="HDF5 compression; use none/false/0 to disable.")
    return p.parse_args()


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def write_dataset(group: h5py.Group, key: str, value: np.ndarray, overwrite: bool, compression) -> None:
    if key in group:
        if not overwrite:
            print(f"    skip existing {key}; pass --overwrite to replace")
            return
        del group[key]
    group.create_dataset(key, data=value, compression=compression)
    print(f"    wrote {key} {value.shape} {value.dtype}")


def sort_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        tail = x.split("_")[-1]
        if tail.isdigit():
            return (0, int(tail))
        return (1, x)

    return sorted(ids, key=key_fn)


def pick_demo_ids(data_group: h5py.Group, requested: Optional[Sequence[str]]) -> list[str]:
    ids = sort_ids(data_group.keys())
    if requested:
        wanted = set(str(x) for x in requested)
        ids = [x for x in ids if x in wanted]
    return ids


def numeric_suffix(text: str, fallback: int) -> int:
    tail = str(text).split("_")[-1]
    return int(tail) if tail.isdigit() else int(fallback)


def read_raw_rgbd(
    rgbd: h5py.File,
    traj_id: str,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sensor_root = f"{traj_id}/obs/sensor_data/{camera_name}"
    param_root = f"{traj_id}/obs/sensor_param/{camera_name}"
    required = [
        f"{sensor_root}/depth",
        f"{sensor_root}/rgb",
        f"{param_root}/intrinsic_cv",
        f"{param_root}/extrinsic_cv",
        f"{param_root}/cam2world_gl",
    ]
    missing = [k for k in required if k not in rgbd]
    if missing:
        raise KeyError(f"Missing RGB-D keys for {traj_id}/{camera_name}: {missing}")

    depth = np.asarray(rgbd[f"{sensor_root}/depth"], dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 3:
        raise ValueError(f"depth must be [T,H,W] after squeeze, got {depth.shape}")

    # ManiSkill stores visual depth in millimeters for rgbd observations.
    if np.nanmedian(depth[depth > 0]) > 20.0:
        depth = depth / 1000.0

    rgb = np.asarray(rgbd[f"{sensor_root}/rgb"])
    K = np.asarray(rgbd[f"{param_root}/intrinsic_cv"], dtype=np.float32)
    E34 = np.asarray(rgbd[f"{param_root}/extrinsic_cv"], dtype=np.float32)
    cam2world_gl = np.asarray(rgbd[f"{param_root}/cam2world_gl"], dtype=np.float32)

    E = np.tile(np.eye(4, dtype=np.float32), (E34.shape[0], 1, 1))
    E[:, :3, :4] = E34
    c2w = np.linalg.inv(E).astype(np.float32)
    return depth.astype(np.float32), rgb, K.astype(np.float32), E.astype(np.float32), c2w, cam2world_gl.astype(np.float32)


def bilinear_sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    x = np.clip(xy[:, 0].astype(np.float64), 0.0, w - 1.0)
    y = np.clip(xy[:, 1].astype(np.float64), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    same_x = x0 == x1
    same_y = y0 == y1
    wa[same_x & same_y] = 1.0
    wb[same_x & same_y] = 0.0
    wc[same_x & same_y] = 0.0
    wd[same_x & same_y] = 0.0
    wa[same_x & ~same_y] = y1[same_x & ~same_y] - y[same_x & ~same_y]
    wb[same_x & ~same_y] = y[same_x & ~same_y] - y0[same_x & ~same_y]
    wc[same_x & ~same_y] = 0.0
    wd[same_x & ~same_y] = 0.0
    wa[~same_x & same_y] = x1[~same_x & same_y] - x[~same_x & same_y]
    wc[~same_x & same_y] = x[~same_x & same_y] - x0[~same_x & same_y]
    wb[~same_x & same_y] = 0.0
    wd[~same_x & same_y] = 0.0

    return (wa * depth[y0, x0] + wb * depth[y1, x0] + wc * depth[y0, x1] + wd * depth[y1, x1]).astype(np.float32)


def track_xy_to_image_pixels(track_xy: np.ndarray, height: int, width: int, uv_mode: str) -> tuple[np.ndarray, str]:
    xy = np.asarray(track_xy, dtype=np.float32)[..., :2].copy()
    if uv_mode == "518":
        xy[..., 0] *= float(width) / 518.0
        xy[..., 1] *= float(height) / 518.0
        return xy, "scaled_from_518"
    if uv_mode == "normalized":
        xy[..., 0] *= max(width - 1, 1)
        xy[..., 1] *= max(height - 1, 1)
        return xy, "scaled_from_normalized"
    if uv_mode == "pixels":
        return xy, "unchanged_pixels"

    finite = xy[np.isfinite(xy)]
    if finite.size == 0:
        return xy, "unchanged_all_nan"
    mn = float(np.nanmin(xy))
    mx = float(np.nanmax(xy))
    if mx > max(height, width) * 1.2:
        xy[..., 0] *= float(width) / 518.0
        xy[..., 1] *= float(height) / 518.0
        return xy, "scaled_from_518"
    if mx <= 2.0 and mn >= -0.5:
        xy[..., 0] *= max(width - 1, 1)
        xy[..., 1] *= max(height - 1, 1)
        return xy, "scaled_from_normalized"
    return xy, "unchanged_pixels"


def infer_uv_mode_from_group(group: h5py.Group, height: int, width: int, requested: str) -> str:
    """Resolve processed tracker UV scale before lifting points with RGB-D depth.

    Some ManiSkill files store SpaTracker query points in the model resolution
    (518x518), while the rendered RGB-D frames are 256x256. Looking only at
    ``track2d`` can miss this when all selected tracks happen to lie in the
    lower coordinate range, so use the original query grid as an extra hint.
    """
    if requested != "auto":
        return requested

    for key in ("query_xy_t0", "p0_uv", "grid_points_xy"):
        if key not in group:
            continue
        uv = np.asarray(group[key], dtype=np.float32)
        if uv.ndim >= 3:
            uv = uv[0]
        if uv.ndim < 2 or uv.shape[-1] < 2:
            continue
        finite = uv[..., :2][np.isfinite(uv[..., :2])]
        if finite.size == 0:
            continue
        if float(np.nanmax(finite)) > max(height, width) * 1.2:
            return "518"
        if float(np.nanmax(finite)) <= 2.0 and float(np.nanmin(finite)) >= -0.5:
            return "normalized"
    return "auto"


def lift_track2d_to_world(
    track2d: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    c2w_traj: np.ndarray,
    uv_mode: str,
) -> tuple[np.ndarray, str, dict[str, float]]:
    track2d = np.asarray(track2d, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    c2w_traj = np.asarray(c2w_traj, dtype=np.float32)

    if track2d.ndim != 3 or track2d.shape[-1] < 2:
        raise ValueError(f"track2d must be [T,N,2+], got {track2d.shape}")
    t_len = track2d.shape[0]
    for name, arr in (("depths", depths), ("intrinsics", intrinsics), ("c2w_traj", c2w_traj)):
        if arr.shape[0] != t_len:
            raise ValueError(f"track2d length {t_len} != {name} length {arr.shape[0]}")

    height, width = int(depths.shape[1]), int(depths.shape[2])
    xy_px, resolved_uv_mode = track_xy_to_image_pixels(track2d, height, width, uv_mode)
    out = np.empty(track2d.shape[:-1] + (3,), dtype=np.float32)
    reproj_errs = []
    all_depth = []

    for t in range(t_len):
        xy = xy_px[t]
        z = bilinear_sample_depth(depths[t], xy)
        K = intrinsics[t]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        if abs(fx) < 1e-8 or abs(fy) < 1e-8:
            raise ValueError(f"invalid intrinsics at t={t}: fx={fx}, fy={fy}")

        x_cam = (xy[:, 0] - cx) * z / fx
        y_cam = (xy[:, 1] - cy) * z / fy
        p_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1).astype(np.float32)
        p_world_h = (c2w_traj[t] @ p_cam[..., None])[..., 0]
        out[t] = p_world_h[:, :3]
        all_depth.append(z)

        try:
            w2c = np.linalg.inv(c2w_traj[t])
            p_cam_check = (w2c @ p_world_h[..., None])[..., 0]
            u = fx * p_cam_check[:, 0] / p_cam_check[:, 2] + cx
            v = fy * p_cam_check[:, 1] / p_cam_check[:, 2] + cy
            reproj_errs.append(np.sqrt((u - xy[:, 0]) ** 2 + (v - xy[:, 1]) ** 2))
        except np.linalg.LinAlgError:
            pass

    depth_cat = np.concatenate(all_depth, axis=0)
    stats = {
        "depth_min": float(np.nanmin(depth_cat)),
        "depth_max": float(np.nanmax(depth_cat)),
        "xyz_min": float(np.nanmin(out)),
        "xyz_max": float(np.nanmax(out)),
    }
    if reproj_errs:
        err = np.concatenate(reproj_errs, axis=0)
        stats["reproj_median_px"] = float(np.nanmedian(err))
        stats["reproj_p95_px"] = float(np.nanpercentile(err, 95))
        stats["reproj_max_px"] = float(np.nanmax(err))
    return out, resolved_uv_mode, stats


def ensure_spatracker_backup(group: h5py.Group, compression) -> None:
    if "point_traj_spatracker" in group or "point_traj" not in group:
        return
    group.create_dataset("point_traj_spatracker", data=np.asarray(group["point_traj"]), compression=compression)
    group.attrs["point_traj_spatracker_note"] = "copied from point_traj before ManiSkill RGB-D metric patch"
    print(f"    wrote point_traj_spatracker {group['point_traj_spatracker'].shape} {group['point_traj_spatracker'].dtype}")


def process_track_file(track_file: Path, rgbd: h5py.File, args: argparse.Namespace) -> None:
    compression = h5_compression(args.compression)
    mode = "r" if args.dry_run else "r+"
    raw_traj_ids = sort_ids(args.traj_ids or [k for k in rgbd.keys() if str(k).startswith("traj_")])
    if not raw_traj_ids:
        raise ValueError("No raw traj_* groups found in RGB-D replay file.")

    print(f"\n[tracks] {track_file}")
    with h5py.File(track_file, mode) as f:
        if "data" not in f:
            raise KeyError(f"{track_file} has no /data group.")
        demo_ids = pick_demo_ids(f["data"], args.demo_ids)
        print(f"  demos: {demo_ids}")
        for idx, demo_id in enumerate(demo_ids):
            grp = f[f"data/{demo_id}"]
            if args.track_key not in grp:
                print(f"  [WARN] {demo_id}: missing {args.track_key}; skip")
                continue
            raw_idx = numeric_suffix(demo_id, idx)
            if len(raw_traj_ids) == 1:
                traj_id = raw_traj_ids[0]
            elif raw_idx < len(raw_traj_ids):
                traj_id = raw_traj_ids[raw_idx]
            else:
                print(f"  [WARN] {demo_id}: no matching raw traj for index {raw_idx}; skip")
                continue

            depth, rgb, K, w2c, c2w, cam2world_gl = read_raw_rgbd(rgbd, traj_id, args.camera_name)
            track2d = np.asarray(grp[args.track_key], dtype=np.float32)
            t_len = min(track2d.shape[0], depth.shape[0], K.shape[0], c2w.shape[0])
            if t_len <= 0:
                print(f"  [WARN] {demo_id}: empty overlap with {traj_id}; skip")
                continue

            depth = depth[:t_len]
            K = K[:t_len]
            w2c = w2c[:t_len]
            c2w = c2w[:t_len]
            cam2world_gl = cam2world_gl[:t_len]
            rgb = rgb[:t_len]
            track2d = track2d[:t_len]

            demo_uv_mode = infer_uv_mode_from_group(grp, int(depth.shape[1]), int(depth.shape[2]), args.uv_mode)
            point_world, resolved_uv_mode, stats = lift_track2d_to_world(
                track2d=track2d,
                depths=depth,
                intrinsics=K,
                c2w_traj=c2w,
                uv_mode=demo_uv_mode,
            )
            print(
                f"  {demo_id} <- {traj_id}: depth={depth.shape} K={K.shape} "
                f"{args.track_key}->{args.out_key} {point_world.shape} ({resolved_uv_mode})"
            )
            print_stats = {
                "reproj_median_px": float("nan"),
                "reproj_p95_px": float("nan"),
                "reproj_max_px": float("nan"),
                **stats,
            }
            print(
                "    depth=[{depth_min:.4f},{depth_max:.4f}]m "
                "xyz=[{xyz_min:.4f},{xyz_max:.4f}]m "
                "reproj median/p95/max={reproj_median_px:.2e}/{reproj_p95_px:.2e}/{reproj_max_px:.2e}px".format(
                    **print_stats,
                )
            )
            if args.dry_run:
                continue

            ensure_spatracker_backup(grp, compression)
            write_dataset(grp, "depths", depth, args.overwrite, compression)
            write_dataset(grp, "intrinsics", K, args.overwrite, compression)
            write_dataset(grp, "w2c_traj", w2c, args.overwrite, compression)
            write_dataset(grp, "c2w_traj", c2w, args.overwrite, compression)
            write_dataset(grp, "cam2world_gl", cam2world_gl, args.overwrite, compression)
            write_dataset(grp, "maniskill_rgb", rgb, args.overwrite, compression)
            if args.world_key:
                write_dataset(grp, args.world_key, point_world, args.overwrite, compression)
            write_dataset(grp, args.out_key, point_world, args.overwrite, compression)

            grp.attrs[f"{args.out_key}_units"] = "meters"
            grp.attrs[f"{args.out_key}_coordinate_frame"] = "maniskill_world"
            grp.attrs[f"{args.out_key}_uv_mode"] = resolved_uv_mode
            grp.attrs[f"{args.out_key}_depth_key"] = "depths"
            grp.attrs[f"{args.out_key}_intrinsics_key"] = "intrinsics"
            grp.attrs[f"{args.out_key}_camera_pose_key"] = "c2w_traj"
            grp.attrs["maniskill_rgbd_source_hdf5"] = str(args.rgbd_h5)
            grp.attrs["maniskill_rgbd_raw_traj_id"] = traj_id
            grp.attrs["point_traj_modes_available"] = ",".join(
                [k for k in ("point_traj_spatracker", "point_traj_world_metric", "point_traj_base_metric") if k in grp]
            )
            if args.replace_point_traj:
                write_dataset(grp, "point_traj", point_world, True, compression)
                grp.attrs["point_traj_active_source"] = args.out_key
                grp.attrs["point_traj_mode"] = "metric"
                grp.attrs["point_traj_units"] = "meters"
                grp.attrs["point_traj_coordinate_frame"] = "maniskill_world"


def main() -> None:
    args = parse_args()
    with h5py.File(args.rgbd_h5, "r") as rgbd:
        for item in args.tracks:
            p = Path(item)
            files = sorted(p.rglob("*_tracks.hdf5")) if p.is_dir() else [p]
            for track_file in files:
                process_track_file(track_file, rgbd, args)


if __name__ == "__main__":
    main()
