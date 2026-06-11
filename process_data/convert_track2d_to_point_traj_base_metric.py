#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert 2D point tracks to metric 3D point trajectories in robot-base frame.

This script does not replay LIBERO / MuJoCo. It assumes the track HDF5 already
contains metric camera information, for example:

    track2d
    sim_depths        or depths
    sim_intrinsics    or intrinsics
    sim_T_base_cam    or T_base_cam

For each tracked pixel (u, v), it samples depth z, back-projects the point into
the camera frame using K, then transforms it to robot base:

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    p_base = T_base_cam @ [x_cam, y_cam, z, 1]

The output point_traj_base_metric is in meters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np


DEPTH_KEYS = ("sim_depths", "depths", "depth", "depth_maps", "depth_video")
INTRINSICS_KEYS = ("sim_intrinsics", "intrinsics", "intrs2", "intrs")
T_BASE_CAM_KEYS = ("sim_T_base_cam", "T_base_cam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift track2d to point_traj_base_metric using metric depth / intrinsics / T_base_cam."
    )
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively for *_tracks.hdf5.")
    parser.add_argument("--demo_ids", nargs="*", default=None, help="Optional demo ids, e.g. demo_0 demo_1.")
    parser.add_argument("--max_demos", type=int, default=None)

    parser.add_argument("--track_key", default="track2d")
    parser.add_argument("--depth_key", default="auto")
    parser.add_argument("--intrinsics_key", default="auto")
    parser.add_argument("--T_base_cam_key", default="auto")
    parser.add_argument("--out_key", default="point_traj_base_metric")

    parser.add_argument(
        "--uv_mode",
        default="auto",
        choices=("auto", "pixels", "518", "normalized"),
        help="Coordinate system of track2d. auto detects pixels / SpaTracker-518 / normalized.",
    )
    parser.add_argument(
        "--depth_sample_mode",
        default="bilinear",
        choices=("bilinear", "patch_min", "patch_median", "patch_percentile", "patch_temporal"),
        help=(
            "How to sample depth at each track point. patch_temporal searches a local depth patch "
            "and chooses the candidate most consistent with the previous 3D point."
        ),
    )
    parser.add_argument("--depth_patch_radius", type=int, default=3, help="Patch radius in pixels for patch_* sampling.")
    parser.add_argument(
        "--depth_patch_percentile",
        type=float,
        default=20.0,
        help="Depth percentile used by patch_percentile and the first frame of patch_temporal.",
    )
    parser.add_argument(
        "--depth_temporal_pixel_weight",
        type=float,
        default=0.005,
        help="Meters of penalty per pixel offset when choosing patch_temporal candidates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--replace_point_traj",
        action="store_true",
        help="Replace point_traj with the metric result, backing up old point_traj as point_traj_original.",
    )
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


def first_existing_key(grp: h5py.Group, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in grp:
            return key
    return None


def resolve_key(grp: h5py.Group, requested: str, candidates: Sequence[str], label: str) -> str:
    if requested != "auto":
        if requested not in grp:
            raise KeyError(f"{label} key {requested!r} not found. available={list(grp.keys())}")
        return requested
    key = first_existing_key(grp, candidates)
    if key is None:
        raise KeyError(f"No {label} key found. tried={candidates}, available={list(grp.keys())}")
    return key


def h5_compression(name: str):
    return None if str(name).lower() in {"", "none", "false", "0"} else name


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, overwrite: bool, compression) -> None:
    if name in group:
        if not overwrite:
            print(f"    skip existing {name}; pass --overwrite to replace")
            return
        del group[name]
    group.create_dataset(name, data=data, compression=compression)
    print(f"    wrote {name} {data.shape} {data.dtype}")


def _decode_attr(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def ensure_spatracker_point_traj(group: h5py.Group, compression) -> None:
    if "point_traj_spatracker" in group:
        return
    if "point_traj_original" in group:
        group.create_dataset("point_traj_spatracker", data=np.asarray(group["point_traj_original"]), compression=compression)
        group.attrs["point_traj_spatracker_note"] = "copied from point_traj_original"
        print(f"    wrote point_traj_spatracker {group['point_traj_spatracker'].shape} {group['point_traj_spatracker'].dtype}")
        return
    if "point_traj" not in group:
        return
    active = str(_decode_attr(group.attrs.get("point_traj_active_source", "")))
    if active == "point_traj_base_metric":
        return
    group.create_dataset("point_traj_spatracker", data=np.asarray(group["point_traj"]), compression=compression)
    group.attrs["point_traj_spatracker_note"] = "copied from point_traj before metric conversion"
    print(f"    wrote point_traj_spatracker {group['point_traj_spatracker'].shape} {group['point_traj_spatracker'].dtype}")


def update_point_traj_modes_attr(group: h5py.Group) -> None:
    modes = [k for k in ("point_traj_spatracker", "point_traj_base_metric") if k in group]
    if modes:
        group.attrs["point_traj_modes_available"] = ",".join(modes)


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

    return (
        wa * depth[y0, x0]
        + wb * depth[y1, x0]
        + wc * depth[y0, x1]
        + wd * depth[y1, x1]
    ).astype(np.float32)


def lift_pixels_to_base(xy: np.ndarray, z: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    x_cam = (xy[:, 0] - K[0, 2]) * z / K[0, 0]
    y_cam = (xy[:, 1] - K[1, 2]) * z / K[1, 1]
    p_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    p_base_h = (T_base_cam @ p_cam[..., None])[..., 0]
    return p_base_h[:, :3].astype(np.float32)


def depth_patch_candidates(
    depth: np.ndarray,
    xy_one: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    x = float(np.clip(xy_one[0], 0.0, w - 1.0))
    y = float(np.clip(xy_one[1], 0.0, h - 1.0))
    r = max(0, int(radius))
    cx = int(round(x))
    cy = int(round(y))
    x0, x1 = max(0, cx - r), min(w - 1, cx + r)
    y0, y1 = max(0, cy - r), min(h - 1, cy + r)

    ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    zs = depth[ys, xs].reshape(-1).astype(np.float32)
    pix = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.float32)
    valid = np.isfinite(zs) & (zs > 0)
    if not np.any(valid):
        z = bilinear_sample_depth(depth, xy_one[None, :])
        pix = xy_one[None, :].astype(np.float32)
        return pix, z, lift_pixels_to_base(pix, z, K, T_base_cam)

    pix = pix[valid]
    zs = zs[valid]
    base = lift_pixels_to_base(pix, zs, K, T_base_cam)
    return pix, zs, base


def choose_patch_candidate(
    depth: np.ndarray,
    xy_one: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    mode: str,
    radius: int,
    percentile: float,
    prev_base: Optional[np.ndarray],
    temporal_pixel_weight: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    pix, zs, base = depth_patch_candidates(depth, xy_one, K, T_base_cam, radius)
    pixel_dist = np.linalg.norm(pix - xy_one[None, :], axis=1)

    if mode == "patch_min":
        target = float(np.nanmin(zs))
        score = np.abs(zs - target) + 1e-4 * pixel_dist
    elif mode == "patch_median":
        target = float(np.nanmedian(zs))
        score = np.abs(zs - target) + 1e-4 * pixel_dist
    elif mode == "patch_percentile" or prev_base is None or not np.all(np.isfinite(prev_base)):
        target = float(np.nanpercentile(zs, np.clip(float(percentile), 0.0, 100.0)))
        score = np.abs(zs - target) + 1e-4 * pixel_dist
    else:
        score = np.linalg.norm(base - prev_base[None, :], axis=1) + float(temporal_pixel_weight) * pixel_dist

    best = int(np.nanargmin(score))
    return pix[best], float(zs[best]), base[best].astype(np.float32), float(pixel_dist[best])


def track_xy_to_image_pixels(track_xy: np.ndarray, height: int, width: int, uv_mode: str) -> tuple[np.ndarray, str]:
    xy = np.asarray(track_xy, dtype=np.float32).copy()
    if xy.shape[-1] < 2:
        raise ValueError(f"track2d must have last dim >= 2, got {xy.shape}")
    xy = xy[..., :2]

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


def lift_track2d_to_base(
    track2d: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    T_base_cams: np.ndarray,
    uv_mode: str,
    depth_sample_mode: str,
    depth_patch_radius: int,
    depth_patch_percentile: float,
    depth_temporal_pixel_weight: float,
) -> tuple[np.ndarray, str, dict[str, float]]:
    track2d = np.asarray(track2d, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    T_base_cams = np.asarray(T_base_cams, dtype=np.float32)

    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if depths.ndim != 3:
        raise ValueError(f"depths must be [T,H,W], got {depths.shape}")
    if track2d.ndim != 3 or track2d.shape[-1] < 2:
        raise ValueError(f"track2d must be [T,N,2+], got {track2d.shape}")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be [T,3,3], got {intrinsics.shape}")
    if T_base_cams.ndim != 3 or T_base_cams.shape[-2:] != (4, 4):
        raise ValueError(f"T_base_cam must be [T,4,4], got {T_base_cams.shape}")

    t_len = track2d.shape[0]
    for name, arr in (("depths", depths), ("intrinsics", intrinsics), ("T_base_cam", T_base_cams)):
        if arr.shape[0] != t_len:
            raise ValueError(f"track2d length {t_len} != {name} length {arr.shape[0]}")

    height, width = int(depths.shape[1]), int(depths.shape[2])
    track2d_px, resolved_uv_mode = track_xy_to_image_pixels(track2d, height, width, uv_mode)
    out = np.empty(track2d_px.shape[:-1] + (3,), dtype=np.float32)
    sampled_xy = np.empty_like(track2d_px, dtype=np.float32)

    reproj_errs = []
    depth_values = []
    pixel_offsets = []
    for t in range(t_len):
        xy = track2d_px[t]
        K = intrinsics[t]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if abs(fx) < 1e-8 or abs(fy) < 1e-8:
            raise ValueError(f"invalid intrinsics at t={t}: fx={fx}, fy={fy}")

        if depth_sample_mode == "bilinear":
            z = bilinear_sample_depth(depths[t], xy)
            sampled_xy[t] = xy
            out[t] = lift_pixels_to_base(xy, z, K, T_base_cams[t])
            pixel_offsets.append(np.zeros((xy.shape[0],), dtype=np.float32))
        else:
            z = np.empty((xy.shape[0],), dtype=np.float32)
            offsets = np.empty((xy.shape[0],), dtype=np.float32)
            for n in range(xy.shape[0]):
                prev = out[t - 1, n] if t > 0 else None
                xy_n, z_n, base_n, offset_n = choose_patch_candidate(
                    depth=depths[t],
                    xy_one=xy[n],
                    K=K,
                    T_base_cam=T_base_cams[t],
                    mode=depth_sample_mode,
                    radius=int(depth_patch_radius),
                    percentile=float(depth_patch_percentile),
                    prev_base=prev,
                    temporal_pixel_weight=float(depth_temporal_pixel_weight),
                )
                sampled_xy[t, n] = xy_n
                z[n] = z_n
                out[t, n] = base_n
                offsets[n] = offset_n
            pixel_offsets.append(offsets)
        depth_values.append(z)

        try:
            T_cam_base = np.linalg.inv(T_base_cams[t])
            p_base_h = np.concatenate([out[t], np.ones((out.shape[1], 1), dtype=np.float32)], axis=-1)
            p_cam_check = (T_cam_base @ p_base_h[..., None])[..., 0]
            u = K[0, 0] * p_cam_check[:, 0] / p_cam_check[:, 2] + K[0, 2]
            v = K[1, 1] * p_cam_check[:, 1] / p_cam_check[:, 2] + K[1, 2]
            reproj_errs.append(np.sqrt((u - xy[:, 0]) ** 2 + (v - xy[:, 1]) ** 2))
        except np.linalg.LinAlgError:
            pass

    depth_cat = np.concatenate(depth_values, axis=0)
    offset_cat = np.concatenate(pixel_offsets, axis=0)
    stats = {
        "depth_min": float(np.nanmin(depth_cat)),
        "depth_max": float(np.nanmax(depth_cat)),
        "xyz_min": float(np.nanmin(out)),
        "xyz_max": float(np.nanmax(out)),
        "sample_pixel_offset_mean": float(np.nanmean(offset_cat)),
        "sample_pixel_offset_p95": float(np.nanpercentile(offset_cat, 95)),
        "sample_pixel_offset_max": float(np.nanmax(offset_cat)),
    }
    if reproj_errs:
        err = np.concatenate(reproj_errs, axis=0)
        stats["reproj_median_px"] = float(np.nanmedian(err))
        stats["reproj_p95_px"] = float(np.nanpercentile(err, 95))
        stats["reproj_max_px"] = float(np.nanmax(err))
    for key in ("reproj_median_px", "reproj_p95_px", "reproj_max_px"):
        stats.setdefault(key, float("nan"))
    return out, resolved_uv_mode, stats


def process_file(track_file: Path, args: argparse.Namespace) -> None:
    mode = "r" if args.dry_run else "r+"
    compression = h5_compression(args.compression)

    print(f"\n[tracks] {track_file}")
    with h5py.File(track_file, mode) as f:
        if "data" not in f:
            raise KeyError(f"{track_file} has no /data group. keys={list(f.keys())}")

        demo_ids = get_demo_ids(f["data"], args.demo_ids, args.max_demos)
        print(f"  demos: {len(demo_ids)}")
        for demo_id in demo_ids:
            grp = f[f"data/{demo_id}"]
            if args.track_key not in grp:
                print(f"  [WARN] {demo_id}: missing {args.track_key}, skip")
                continue

            depth_key = resolve_key(grp, args.depth_key, DEPTH_KEYS, "depth")
            intrinsics_key = resolve_key(grp, args.intrinsics_key, INTRINSICS_KEYS, "intrinsics")
            T_key = resolve_key(grp, args.T_base_cam_key, T_BASE_CAM_KEYS, "T_base_cam")

            point_traj_base, resolved_uv_mode, stats = lift_track2d_to_base(
                track2d=np.asarray(grp[args.track_key]),
                depths=np.asarray(grp[depth_key]),
                intrinsics=np.asarray(grp[intrinsics_key]),
                T_base_cams=np.asarray(grp[T_key]),
                uv_mode=args.uv_mode,
                depth_sample_mode=args.depth_sample_mode,
                depth_patch_radius=args.depth_patch_radius,
                depth_patch_percentile=args.depth_patch_percentile,
                depth_temporal_pixel_weight=args.depth_temporal_pixel_weight,
            )

            print(
                f"  {demo_id}: {args.track_key}+{depth_key}+{intrinsics_key}+{T_key} "
                f"-> {args.out_key} {point_traj_base.shape} ({resolved_uv_mode})"
            )
            print(
                "    depth=[{depth_min:.4f},{depth_max:.4f}]m "
                "xyz=[{xyz_min:.4f},{xyz_max:.4f}]m "
                "sample_offset mean/p95/max={sample_pixel_offset_mean:.2f}/{sample_pixel_offset_p95:.2f}/{sample_pixel_offset_max:.2f}px "
                "reproj median/p95/max={reproj_median_px:.2e}/{reproj_p95_px:.2e}/{reproj_max_px:.2e}px".format(**stats)
            )

            if args.dry_run:
                continue

            ensure_spatracker_point_traj(grp, compression)
            write_dataset(grp, args.out_key, point_traj_base, args.overwrite, compression)
            grp.attrs[f"{args.out_key}_uv_mode"] = resolved_uv_mode
            grp.attrs[f"{args.out_key}_depth_sample_mode"] = args.depth_sample_mode
            grp.attrs[f"{args.out_key}_depth_patch_radius"] = int(args.depth_patch_radius)
            grp.attrs[f"{args.out_key}_depth_patch_percentile"] = float(args.depth_patch_percentile)
            grp.attrs[f"{args.out_key}_depth_temporal_pixel_weight"] = float(args.depth_temporal_pixel_weight)
            grp.attrs[f"{args.out_key}_depth_key"] = depth_key
            grp.attrs[f"{args.out_key}_intrinsics_key"] = intrinsics_key
            grp.attrs[f"{args.out_key}_T_base_cam_key"] = T_key
            grp.attrs[f"{args.out_key}_units"] = "meters"
            grp.attrs[f"{args.out_key}_coordinate_frame"] = "robot0_base"
            update_point_traj_modes_attr(grp)

            if args.replace_point_traj:
                if "point_traj" in grp and "point_traj_original" not in grp:
                    grp.copy("point_traj", "point_traj_original")
                write_dataset(grp, "point_traj", point_traj_base, True, compression)
                grp.attrs["point_traj_active_source"] = args.out_key
                grp.attrs["point_traj_mode"] = "metric"
                grp.attrs["point_traj_units"] = "meters"
                grp.attrs["point_traj_coordinate_frame"] = "robot0_base"


def main() -> None:
    args = parse_args()
    for track_file in collect_track_files(args):
        process_file(track_file, args)


if __name__ == "__main__":
    main()
