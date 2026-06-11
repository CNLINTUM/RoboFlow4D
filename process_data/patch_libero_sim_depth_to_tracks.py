#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch LIBERO track HDF5 files with simulator metric depth and camera poses.

This script replays the saved MuJoCo states from the original LIBERO HDF5 file,
renders true simulator depth, and writes the result back into the processed
``*_tracks.hdf5`` file.

Important convention:
    The existing LIBERO preprocessing rotates RGB frames by 180 degrees using
    ``frame[::-1, ::-1]``. By default this script applies the same rotation to
    depth and changes K / camera pose so ``track2d`` can be lifted directly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.convert_track2d_to_point_traj_base_metric import (  # noqa: E402
    lift_track2d_to_base as lift_track2d_to_base_metric,
)


KNOWN_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
}

IMAGE_KEY_TO_CAMERA = {
    "agentview_rgb": "agentview",
    "agentview_image": "agentview",
    "eye_in_hand_rgb": "robot0_eye_in_hand",
    "robot0_eye_in_hand_rgb": "robot0_eye_in_hand",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add simulator metric depths / intrinsics / T_base_cam to LIBERO track HDF5 files."
    )
    p.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    p.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively for *_tracks.hdf5.")
    p.add_argument("--source_hdf5", default=None, help="Override source LIBERO HDF5. Default: tracks root attr source_hdf5.")
    default_libero_root = os.environ.get("LIBERO_ROOT")
    p.add_argument(
        "--libero_root",
        default=default_libero_root,
        required=default_libero_root is None,
        help="Path to the LIBERO repo root. Can also be provided with LIBERO_ROOT.",
    )
    p.add_argument("--camera_name", default=None, help="LIBERO camera name. Default inferred from image_key.")
    p.add_argument("--image_key", default=None, help="Processed image key, used only to infer camera_name.")
    p.add_argument("--demo_ids", nargs="*", default=None, help="Optional demo ids, e.g. demo_0 demo_1.")
    p.add_argument("--max_demos", type=int, default=None)
    p.add_argument("--max_stride_search", type=int, default=20)
    p.add_argument("--limit_frames", type=int, default=None, help="Dry-run only: render at most this many frames per demo.")
    p.add_argument("--dry_run", action="store_true", help="Render and report shapes without writing to HDF5.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite sim_* datasets if they already exist.")
    p.add_argument(
        "--save_state_indices",
        action="store_true",
        help="Optionally save sim_state_indices for debugging frame-to-state alignment.",
    )
    p.add_argument(
        "--write_standard_keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write depths/intrinsics/T_world_cam/T_base_cam when absent.",
    )
    p.add_argument(
        "--overwrite_standard_keys",
        action="store_true",
        help="Overwrite standard keys such as depths or T_base_cam. Use carefully if VGGT depths already exist.",
    )
    p.add_argument(
        "--match_processed_rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the existing LIBERO preprocessing rotation frame[::-1, ::-1].",
    )
    p.add_argument(
        "--point_traj_metric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lift track2d into robot-base metric xyz and save point_traj_base_metric.",
    )
    p.add_argument(
        "--depth_sample_mode",
        default="bilinear",
        choices=("bilinear", "patch_min", "patch_median", "patch_percentile", "patch_temporal"),
        help=(
            "Depth sampling mode for --point_traj_metric. patch_temporal searches a local depth patch "
            "and prefers candidates that are temporally continuous in 3D."
        ),
    )
    p.add_argument("--depth_patch_radius", type=int, default=3)
    p.add_argument("--depth_patch_percentile", type=float, default=20.0)
    p.add_argument("--depth_temporal_pixel_weight", type=float, default=0.005)
    p.add_argument(
        "--replace_point_traj",
        action="store_true",
        help="Replace point_traj with point_traj_base_metric, preserving the old one as point_traj_spatracker.",
    )
    p.add_argument(
        "--filter_metric_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter point_traj_base_metric with moving + first-frame SOR + teleport rejection.",
    )
    p.add_argument(
        "--filter_motion_thresh",
        type=float,
        default=0.25,
        help="Motion threshold passed to filter_points_moving_and_sor_firstframe.",
    )
    p.add_argument(
        "--filter_sor_k",
        type=int,
        default=64,
        help="KNN count for the first-frame SOR metric-point filter.",
    )
    p.add_argument(
        "--filter_sor_std_ratio",
        type=float,
        default=2.5,
        help="SOR threshold std-ratio for the metric-point filter.",
    )
    p.add_argument(
        "--filter_replace_outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace filtered-out trajectories with inlier trajectories instead of only returning masks.",
    )
    p.add_argument(
        "--filter_replace_mode",
        choices=("random", "nearest"),
        default="random",
        help="Replacement source selection for filtered-out metric trajectories.",
    )
    p.add_argument(
        "--filter_teleport_step_thresh",
        type=float,
        default=0.3,
        help="Single-step metric jump threshold in meters for teleport rejection.",
    )
    p.add_argument(
        "--filter_component_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter small disconnected trajectory/endpoint components after teleport/SOR filtering.",
    )
    p.add_argument("--filter_component_k", type=int, default=5, help="KNN count for component outlier filtering.")
    p.add_argument(
        "--filter_component_factor",
        type=float,
        default=6.0,
        help="Robust threshold factor for component outlier filtering.",
    )
    p.add_argument(
        "--filter_component_frame_stride",
        type=int,
        default=1,
        help="Frame stride used by frame-wise component filtering.",
    )
    p.add_argument(
        "--filter_component_min_bad_frames",
        type=int,
        default=1,
        help="Drop a point if it is outside the main component for at least this many sampled frames.",
    )
    p.add_argument(
        "--filter_component_min_keep_ratio",
        type=float,
        default=0.55,
        help="Disable component filtering for a demo if it would keep fewer than this fraction of points.",
    )
    p.add_argument(
        "--filter_trajectory_component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run connected-component filtering in full trajectory space.",
    )
    p.add_argument("--filter_seed", type=int, default=0, help="Torch RNG seed used when --filter_replace_mode=random.")
    p.add_argument(
        "--filter_verbose",
        action="store_true",
        help="Let utils.motion_filter print per-demo detailed filtering stats.",
    )
    p.add_argument("--compression", default="gzip", help="HDF5 compression for written datasets; use 'none' to disable.")
    return p.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


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
    uniq = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise ValueError("No track files found. Pass --tracks or --tracks_root.")
    return uniq


def infer_source_hdf5(track_file: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    with h5py.File(track_file, "r") as f:
        if "source_hdf5" not in f.attrs:
            raise ValueError(f"{track_file} has no root attr source_hdf5; pass --source_hdf5.")
        return Path(str(_decode_attr(f.attrs["source_hdf5"])))


def infer_image_key(track_file: Path, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    with h5py.File(track_file, "r") as f:
        if "image_key" in f.attrs:
            return str(_decode_attr(f.attrs["image_key"]))
    return None


def infer_camera_name(track_file: Path, args: argparse.Namespace) -> str:
    if args.camera_name:
        return args.camera_name
    image_key = infer_image_key(track_file, args.image_key)
    if image_key in IMAGE_KEY_TO_CAMERA:
        return IMAGE_KEY_TO_CAMERA[image_key]
    raise ValueError(
        f"Could not infer camera name from image_key={image_key!r}. Pass --camera_name, e.g. agentview."
    )


def infer_suite_from_source(source_hdf5: Path) -> Optional[str]:
    for part in source_hdf5.parts:
        if part in KNOWN_SUITES:
            return part
        if part.endswith("_no_noops") and part[: -len("_no_noops")] in KNOWN_SUITES:
            return part[: -len("_no_noops")]
    return None


def infer_bddl_path(source_hdf5: Path, libero_root: Path) -> Path:
    suite = infer_suite_from_source(source_hdf5)
    task_name = source_hdf5.stem
    if task_name.endswith("_demo"):
        task_name = task_name[: -len("_demo")]

    bddl_root = libero_root / "libero" / "libero" / "bddl_files"
    if suite is not None:
        candidate = bddl_root / suite / f"{task_name}.bddl"
        if candidate.exists():
            return candidate

    matches = list(bddl_root.glob(f"*/{task_name}.bddl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {task_name}.bddl under {bddl_root}")
    raise ValueError(f"Ambiguous BDDL for {task_name}: {matches}")


def get_track_demo_ids(track_file: Path, requested: Optional[Iterable[str]], max_demos: Optional[int]) -> list[str]:
    with h5py.File(track_file, "r") as f:
        ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]) if x.startswith("demo_") else x)
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[:max_demos]
    return ids


def infer_render_size(track_file: Path, source_hdf5: Path, demo_ids: list[str], image_key: Optional[str]) -> tuple[int, int]:
    with h5py.File(track_file, "r") as f:
        for demo_id in demo_ids:
            grp = f[f"data/{demo_id}"]
            if "frames_rgb" in grp:
                shape = grp["frames_rgb"].shape
                if len(shape) == 4:
                    # Processed tracks usually store frames as (T, H, W, C).
                    # Some datasets may use (T, C, H, W). Infer the channel
                    # axis so the renderer gets H x W, not W x C.
                    if shape[-1] in (1, 3, 4):
                        return int(shape[1]), int(shape[2])
                    if shape[1] in (1, 3, 4):
                        return int(shape[2]), int(shape[3])
                    return int(shape[-3]), int(shape[-2])
            if "track2d" in grp:
                # Most LIBERO processed files are 256x256; fall through for exact source image if possible.
                break
    if image_key:
        with h5py.File(source_hdf5, "r") as f:
            for demo_id in demo_ids:
                key = f"data/{demo_id}/obs/{image_key}"
                if key in f:
                    h, w = f[key].shape[1:3]
                    return int(h), int(w)
    return 256, 256


def infer_state_indices(
    src_demo: h5py.Group,
    track_demo: h5py.Group,
    max_stride_search: int,
) -> tuple[np.ndarray, str]:
    src_len = int(src_demo["states"].shape[0])
    target_len = None
    for key in ("track2d", "frames_rgb", "actions", "robot_states"):
        if key in track_demo:
            target_len = int(track_demo[key].shape[0])
            break
    if target_len is None:
        target_len = src_len

    if target_len == src_len:
        return np.arange(src_len, dtype=np.int64), "same_length"

    if "actions" in src_demo and "actions" in track_demo:
        src_actions = np.asarray(src_demo["actions"])
        track_actions = np.asarray(track_demo["actions"])
        for stride in range(1, max_stride_search + 1):
            idx = np.arange(0, src_len, stride, dtype=np.int64)
            if len(idx) != target_len:
                continue
            if np.allclose(src_actions[idx], track_actions, atol=1e-5, rtol=1e-5):
                return idx, f"matched_actions_stride_{stride}"

    for stride in range(1, max_stride_search + 1):
        idx = np.arange(0, src_len, stride, dtype=np.int64)
        if len(idx) == target_len:
            return idx, f"inferred_stride_{stride}"

    idx = np.rint(np.linspace(0, src_len - 1, target_len)).astype(np.int64)
    return idx, "linspace_fallback"


def setup_libero_imports(libero_root: Path):
    root = str(libero_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    return OffScreenRenderEnv, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map


def make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def get_T_base_world(sim) -> np.ndarray:
    for body_name in ("robot0_base", "robot0_link0"):
        try:
            body_id = sim.model.body_name2id(body_name)
            T_world_base = make_pose(sim.data.body_xpos[body_id], sim.data.body_xmat[body_id].reshape(3, 3))
            return np.linalg.inv(T_world_base)
        except Exception:
            pass
    return np.eye(4, dtype=np.float64)


def rotate_camera_180_for_processed_image(
    K_raw: np.ndarray,
    T_world_cam_raw: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(K_raw, dtype=np.float64, copy=True)
    K[0, 2] = (width - 1) - K_raw[0, 2]
    K[1, 2] = (height - 1) - K_raw[1, 2]

    Rz_180 = np.eye(4, dtype=np.float64)
    Rz_180[0, 0] = -1.0
    Rz_180[1, 1] = -1.0
    T_world_cam = np.asarray(T_world_cam_raw, dtype=np.float64) @ Rz_180
    return K, T_world_cam


def render_metric_camera_data(
    env,
    states: np.ndarray,
    state_indices: np.ndarray,
    camera_name: str,
    height: int,
    width: int,
    match_processed_rotation: bool,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_key = f"{camera_name}_depth"
    depths = []
    intrinsics = []
    T_world_cams = []
    T_base_cams = []

    for idx in tqdm(state_indices, desc=f"render {camera_name}", leave=False):
        obs = env.set_init_state(states[int(idx)])
        if depth_key not in obs:
            raise KeyError(f"Observation has no {depth_key}; keys include {sorted(obs.keys())[:20]}")

        depth = get_real_depth_map(env.sim, obs[depth_key]).astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        K_raw = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
        T_world_cam_raw = get_camera_extrinsic_matrix(env.sim, camera_name)
        if match_processed_rotation:
            depth = depth[::-1, ::-1].copy()
            K, T_world_cam = rotate_camera_180_for_processed_image(K_raw, T_world_cam_raw, height, width)
        else:
            K = np.asarray(K_raw, dtype=np.float64)
            T_world_cam = np.asarray(T_world_cam_raw, dtype=np.float64)

        T_base_world = get_T_base_world(env.sim)
        T_base_cam = T_base_world @ T_world_cam

        depths.append(depth.astype(np.float32))
        intrinsics.append(K.astype(np.float32))
        T_world_cams.append(T_world_cam.astype(np.float32))
        T_base_cams.append(T_base_cam.astype(np.float32))

    return (
        np.stack(depths, axis=0),
        np.stack(intrinsics, axis=0),
        np.stack(T_world_cams, axis=0),
        np.stack(T_base_cams, axis=0),
    )


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

    # When x or y lies exactly on the image boundary, x0 == x1 or y0 == y1.
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


def track_xy_to_image_pixels(track_xy: np.ndarray, height: int, width: int) -> tuple[np.ndarray, str]:
    xy = np.asarray(track_xy, dtype=np.float32).copy()
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
) -> tuple[np.ndarray, str]:
    if track2d.shape[0] != depths.shape[0]:
        raise ValueError(f"track2d length {track2d.shape[0]} != depths length {depths.shape[0]}")

    out = np.empty(track2d.shape[:-1] + (3,), dtype=np.float32)
    track2d_px, uv_mode = track_xy_to_image_pixels(track2d, depths.shape[1], depths.shape[2])
    for t in range(track2d.shape[0]):
        xy = track2d_px[t]
        z = bilinear_sample_depth(depths[t], xy)
        K = intrinsics[t]
        x_cam = (xy[:, 0] - K[0, 2]) * z / K[0, 0]
        y_cam = (xy[:, 1] - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
        base = (T_base_cams[t] @ cam[..., None])[..., 0]
        out[t] = base[:, :3]
    return out, str(uv_mode)


def filter_metric_point_traj(
    point_traj: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Filter metric trajectories with motion, teleport, and component checks."""
    stats: dict[str, float | int | bool | str] = {
        "enabled": bool(args.filter_metric_points),
        "motion_thresh": float(args.filter_motion_thresh),
        "sor_k": int(args.filter_sor_k),
        "sor_std_ratio": float(args.filter_sor_std_ratio),
        "replace_outliers": bool(args.filter_replace_outliers),
        "replace_mode": str(args.filter_replace_mode),
        "teleport_step_thresh_m": float(args.filter_teleport_step_thresh),
        "component_enabled": bool(getattr(args, "filter_component_points", True)),
        "component_k": int(getattr(args, "filter_component_k", 5)),
        "component_factor": float(getattr(args, "filter_component_factor", 6.0)),
        "component_frame_stride": int(getattr(args, "filter_component_frame_stride", 1)),
        "component_min_bad_frames": int(getattr(args, "filter_component_min_bad_frames", 1)),
        "component_min_keep_ratio": float(getattr(args, "filter_component_min_keep_ratio", 0.55)),
        "trajectory_component_enabled": bool(getattr(args, "filter_trajectory_component", True)),
    }
    point_traj = np.asarray(point_traj, dtype=np.float32)
    if not args.filter_metric_points:
        return point_traj, stats
    if point_traj.ndim != 3 or point_traj.shape[-1] != 3:
        raise ValueError(f"point_traj must be [T,N,3], got {point_traj.shape}")
    if point_traj.shape[0] < 2 or point_traj.shape[1] == 0:
        stats.update({"skipped": True, "skip_reason": "too_short_or_empty"})
        return point_traj, stats

    import torch
    from utils.motion_filter import filter_points_component_outliers, filter_points_moving_and_sor_firstframe

    torch.manual_seed(int(args.filter_seed))
    points = torch.from_numpy(point_traj).float()
    step_norm = torch.linalg.norm(points[1:] - points[:-1], dim=-1)
    max_step = step_norm.max(dim=0).values
    teleport_bad = max_step > float(args.filter_teleport_step_thresh)

    call = lambda: filter_points_moving_and_sor_firstframe(
        points,
        motion_thresh=float(args.filter_motion_thresh),
        k=int(args.filter_sor_k),
        std_ratio=float(args.filter_sor_std_ratio),
        replace_outliers=bool(args.filter_replace_outliers),
        replace_mode=str(args.filter_replace_mode),
        teleport_step_thresh=float(args.filter_teleport_step_thresh),
    )
    if args.filter_verbose:
        filtered, inlier_masks, moving_mask, motion_mag = call()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            filtered, inlier_masks, moving_mask, motion_mag = call()

    inlier0 = inlier_masks[0].bool()
    filtered, component_keep, component_stats = filter_points_component_outliers(
        filtered,
        candidate_mask=inlier0,
        enabled=bool(getattr(args, "filter_component_points", True)),
        spatial_k=int(getattr(args, "filter_component_k", 5)),
        spatial_factor=float(getattr(args, "filter_component_factor", 6.0)),
        frame_stride=int(getattr(args, "filter_component_frame_stride", 1)),
        min_bad_frames=int(getattr(args, "filter_component_min_bad_frames", 1)),
        min_keep_ratio=float(getattr(args, "filter_component_min_keep_ratio", 0.55)),
        use_trajectory_component=bool(getattr(args, "filter_trajectory_component", True)),
        replace_outliers=bool(args.filter_replace_outliers),
        replace_mode=str(args.filter_replace_mode),
    )
    if bool(getattr(args, "filter_component_points", True)):
        inlier0 = component_keep.bool()
    changed = torch.linalg.norm(filtered - points, dim=-1).max(dim=0).values > 1e-7
    stats.update(
        {
            "skipped": False,
            "num_frames": int(point_traj.shape[0]),
            "num_points": int(point_traj.shape[1]),
            "moving_count": int(moving_mask.sum().item()),
            "inlier_count": int(inlier0.sum().item()),
            "outlier_count": int((~inlier0).sum().item()),
            "teleport_bad_count": int(teleport_bad.sum().item()),
            "changed_count": int(changed.sum().item()),
            "motion_mag_mean": float(motion_mag.mean().item()),
            "motion_mag_max": float(motion_mag.max().item()),
            "max_step_before_m": float(max_step.max().item()),
        }
    )
    stats.update(component_stats)

    filtered_step_norm = torch.linalg.norm(filtered[1:] - filtered[:-1], dim=-1)
    stats["max_step_after_m"] = float(filtered_step_norm.max().item())
    return filtered.cpu().numpy().astype(np.float32), stats


def h5_compression(args: argparse.Namespace):
    return None if args.compression.lower() in {"", "none", "false", "0"} else args.compression


def write_dataset(group: h5py.Group, name: str, data: np.ndarray, overwrite: bool, compression):
    if name in group:
        if not overwrite:
            print(f"    skip existing {name}")
            return
        del group[name]
    group.create_dataset(name, data=data, compression=compression)
    print(f"    wrote {name} {data.shape} {data.dtype}")


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
    group.attrs["point_traj_spatracker_note"] = "copied from point_traj before metric patching"
    print(f"    wrote point_traj_spatracker {group['point_traj_spatracker'].shape} {group['point_traj_spatracker'].dtype}")


def update_point_traj_modes_attr(group: h5py.Group) -> None:
    modes = [k for k in ("point_traj_spatracker", "point_traj_base_metric") if k in group]
    if modes:
        group.attrs["point_traj_modes_available"] = ",".join(modes)


def patch_track_file(track_file: Path, args: argparse.Namespace) -> None:
    source_hdf5 = infer_source_hdf5(track_file, args.source_hdf5)
    if not source_hdf5.exists():
        raise FileNotFoundError(source_hdf5)

    camera_name = infer_camera_name(track_file, args)
    image_key = infer_image_key(track_file, args.image_key)
    demo_ids = get_track_demo_ids(track_file, args.demo_ids, args.max_demos)
    height, width = infer_render_size(track_file, source_hdf5, demo_ids, image_key)
    bddl_path = infer_bddl_path(source_hdf5, Path(args.libero_root))

    print(f"\n[tracks] {track_file}")
    print(f"  source: {source_hdf5}")
    print(f"  bddl:   {bddl_path}")
    print(f"  camera: {camera_name}, size: {height}x{width}, demos: {len(demo_ids)}")

    OffScreenRenderEnv, get_K, get_T_wc, get_real_depth = setup_libero_imports(Path(args.libero_root))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=height,
        camera_widths=width,
        camera_names=[camera_name],
        camera_depths=True,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    env.seed(0)
    env.reset()

    compression = h5_compression(args)
    try:
        file_mode = "r" if args.dry_run else "r+"
        with h5py.File(source_hdf5, "r") as fsrc, h5py.File(track_file, file_mode) as ftrk:
            for demo_id in demo_ids:
                if f"data/{demo_id}" not in fsrc:
                    print(f"  [WARN] source missing {demo_id}, skip")
                    continue
                src_demo = fsrc[f"data/{demo_id}"]
                track_demo = ftrk[f"data/{demo_id}"]
                state_indices, index_mode = infer_state_indices(src_demo, track_demo, args.max_stride_search)
                if args.limit_frames is not None:
                    if not args.dry_run:
                        raise ValueError("--limit_frames is only allowed with --dry_run to avoid partial HDF5 writes.")
                    state_indices = state_indices[: args.limit_frames]

                print(f"  {demo_id}: render {len(state_indices)} frames ({index_mode})")
                depths, Ks, T_world_cams, T_base_cams = render_metric_camera_data(
                    env=env,
                    states=np.asarray(src_demo["states"]),
                    state_indices=state_indices,
                    camera_name=camera_name,
                    height=height,
                    width=width,
                    match_processed_rotation=args.match_processed_rotation,
                    get_camera_intrinsic_matrix=get_K,
                    get_camera_extrinsic_matrix=get_T_wc,
                    get_real_depth_map=get_real_depth,
                )

                point_traj_base = None
                point_traj_filter_stats: dict[str, float | int | bool | str] = {}
                point_traj_lift_stats: dict[str, float] = {}
                if args.point_traj_metric and "track2d" in track_demo and track_demo["track2d"].shape[0] == len(state_indices):
                    point_traj_base, point_traj_uv_mode, point_traj_lift_stats = lift_track2d_to_base_metric(
                        track2d=np.asarray(track_demo["track2d"]),
                        depths=depths,
                        intrinsics=Ks,
                        T_base_cams=T_base_cams,
                        uv_mode="auto",
                        depth_sample_mode=str(args.depth_sample_mode),
                        depth_patch_radius=int(args.depth_patch_radius),
                        depth_patch_percentile=float(args.depth_patch_percentile),
                        depth_temporal_pixel_weight=float(args.depth_temporal_pixel_weight),
                    )
                    print(f"    lifted point_traj_base_metric {point_traj_base.shape} ({point_traj_uv_mode})")
                    if point_traj_lift_stats:
                        print(
                            "    depth sampling: "
                            f"mode={args.depth_sample_mode}, "
                            "offset mean/p95/max="
                            f"{point_traj_lift_stats.get('sample_pixel_offset_mean', 0.0):.2f}/"
                            f"{point_traj_lift_stats.get('sample_pixel_offset_p95', 0.0):.2f}/"
                            f"{point_traj_lift_stats.get('sample_pixel_offset_max', 0.0):.2f}px"
                        )
                else:
                    point_traj_uv_mode = ""

                if args.dry_run:
                    print(
                        f"    dry-run shapes: sim_depths={depths.shape}, sim_intrinsics={Ks.shape}, "
                        f"sim_T_base_cam={T_base_cams.shape}"
                    )
                    continue

                if point_traj_base is not None:
                    point_traj_base, point_traj_filter_stats = filter_metric_point_traj(point_traj_base, args)
                    if point_traj_filter_stats.get("enabled"):
                        print(
                            "    filtered point_traj_base_metric: "
                            f"inliers={point_traj_filter_stats.get('inlier_count', 0)}/"
                            f"{point_traj_filter_stats.get('num_points', point_traj_base.shape[1])}, "
                            f"teleport_bad={point_traj_filter_stats.get('teleport_bad_count', 0)}, "
                            f"max_step {point_traj_filter_stats.get('max_step_before_m', 0.0):.4f}"
                            f" -> {point_traj_filter_stats.get('max_step_after_m', 0.0):.4f} m"
                        )

                write_dataset(track_demo, "sim_depths", depths, args.overwrite, compression)
                write_dataset(track_demo, "sim_intrinsics", Ks, args.overwrite, compression)
                write_dataset(track_demo, "sim_T_world_cam", T_world_cams, args.overwrite, compression)
                write_dataset(track_demo, "sim_T_base_cam", T_base_cams, args.overwrite, compression)
                if args.save_state_indices:
                    write_dataset(track_demo, "sim_state_indices", state_indices.astype(np.int64), args.overwrite, compression)

                if args.write_standard_keys:
                    standard_overwrite = args.overwrite_standard_keys
                    write_dataset(track_demo, "depths", depths, standard_overwrite, compression)
                    write_dataset(track_demo, "intrinsics", Ks, standard_overwrite, compression)
                    write_dataset(track_demo, "T_world_cam", T_world_cams, standard_overwrite, compression)
                    write_dataset(track_demo, "T_base_cam", T_base_cams, standard_overwrite, compression)

                if point_traj_base is not None:
                    ensure_spatracker_point_traj(track_demo, compression)
                    write_dataset(track_demo, "point_traj_base_metric", point_traj_base, args.overwrite, compression)
                    track_demo.attrs["point_traj_base_metric_uv_mode"] = point_traj_uv_mode
                    track_demo.attrs["point_traj_base_metric_depth_sample_mode"] = str(args.depth_sample_mode)
                    track_demo.attrs["point_traj_base_metric_depth_patch_radius"] = int(args.depth_patch_radius)
                    track_demo.attrs["point_traj_base_metric_depth_patch_percentile"] = float(args.depth_patch_percentile)
                    track_demo.attrs["point_traj_base_metric_depth_temporal_pixel_weight"] = float(
                        args.depth_temporal_pixel_weight
                    )
                    track_demo.attrs["point_traj_base_metric_units"] = "meters"
                    track_demo.attrs["point_traj_base_metric_coordinate_frame"] = "robot0_base"
                    for key, value in point_traj_lift_stats.items():
                        track_demo.attrs[f"point_traj_base_metric_lift_{key}"] = value
                    for key, value in point_traj_filter_stats.items():
                        track_demo.attrs[f"point_traj_base_metric_filter_{key}"] = value
                    update_point_traj_modes_attr(track_demo)
                    if args.replace_point_traj:
                        if "point_traj" in track_demo and "point_traj_original" not in track_demo:
                            track_demo.copy("point_traj", "point_traj_original")
                        write_dataset(track_demo, "point_traj", point_traj_base, True, compression)
                        track_demo.attrs["point_traj_active_source"] = "point_traj_base_metric"
                        track_demo.attrs["point_traj_mode"] = "metric"
                        track_demo.attrs["point_traj_units"] = "meters"
                        track_demo.attrs["point_traj_coordinate_frame"] = "robot0_base"

                track_demo.attrs["sim_metric_source_hdf5"] = str(source_hdf5)
                track_demo.attrs["sim_metric_bddl_path"] = str(bddl_path)
                track_demo.attrs["sim_metric_camera_name"] = camera_name
                track_demo.attrs["sim_metric_depth_units"] = "meters"
                track_demo.attrs["sim_metric_coordinate_frame"] = "robot0_base"
                track_demo.attrs["sim_metric_matches_processed_rotation"] = bool(args.match_processed_rotation)
                track_demo.attrs["sim_metric_state_index_mode"] = index_mode
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.limit_frames is not None and not args.dry_run:
        raise ValueError("--limit_frames is only supported with --dry_run.")
    # EGL is the usual headless backend for MuJoCo on this machine.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    for track_file in collect_track_files(args):
        patch_track_file(track_file, args)


if __name__ == "__main__":
    main()
