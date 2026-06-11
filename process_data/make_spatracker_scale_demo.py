#!/usr/bin/env python3
"""Create a non-destructive SpaTracker-scale HDF5 copy for one demo.

This is useful when a metric-patched LIBERO file already contains
`point_traj_spatracker`, but the active `point_traj` was switched to metric
scale. The script writes a new file and never modifies the source file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np


METRIC_DATA_KEYS = (
    "point_traj_base_metric",
    "sim_depths",
    "sim_intrinsics",
    "sim_T_base_cam",
    "sim_T_world_cam",
    "T_base_cam",
    "T_world_cam",
)
METRIC_ATTR_PREFIXES = (
    "point_traj_base_metric",
    "sim_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one demo as a SpaTracker-scale HDF5 copy."
    )
    parser.add_argument("--src", required=True, help="Source tracks HDF5.")
    parser.add_argument("--out", required=True, help="Output HDF5 to write.")
    parser.add_argument("--demo_id", default="demo_0", help="Demo id under /data.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fit_metric_scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If point_traj_base_metric and sim camera keys exist, fit a similarity "
            "transform from metric robot-base coordinates to SpaTracker coordinates "
            "and write dense-scene keys depths/intrinsics/c2w in SpaTracker scale."
        ),
    )
    parser.add_argument(
        "--drop_metric_keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop metric trajectory and sim_* keys from the output copy.",
    )
    parser.add_argument(
        "--max_fit_points",
        type=int,
        default=100000,
        help="Maximum correspondence count used for the metric->SpaTracker fit.",
    )
    return parser.parse_args()


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def copy_one_demo(src_path: Path, out_path: Path, demo_id: str, overwrite: bool) -> None:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace it: {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as fin, h5py.File(out_path, "w") as fout:
        copy_attrs(fin.attrs, fout.attrs)
        for key in fin.keys():
            if key != "data":
                fin.copy(key, fout, name=key)

        if "data" not in fin:
            raise KeyError(f"{src_path} has no /data group")
        if demo_id not in fin["data"]:
            raise KeyError(f"{src_path} has no /data/{demo_id}; available={list(fin['data'].keys())}")

        data_out = fout.create_group("data")
        copy_attrs(fin["data"].attrs, data_out.attrs)
        fin["data"].copy(demo_id, data_out, name=demo_id)


def replace_dataset(group: h5py.Group, key: str, value: np.ndarray, compression: str = "gzip") -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=np.asarray(value), compression=compression)


def finite_pairs(src: np.ndarray, dst: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    src_flat = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_flat = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(src_flat).all(axis=1) & np.isfinite(dst_flat).all(axis=1)
    src_flat = src_flat[ok]
    dst_flat = dst_flat[ok]
    if src_flat.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(src_flat.shape[0], size=max_points, replace=False)
        src_flat = src_flat[idx]
        dst_flat = dst_flat[idx]
    return src_flat, dst_flat


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Fit dst ~= scale * R @ src + t with Umeyama alignment."""
    if src.shape[0] < 3:
        raise ValueError("Need at least three point correspondences for similarity fitting")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src < 1e-12:
        raise ValueError("Degenerate source points for similarity fitting")

    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, svals, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    r = u @ np.diag(sign) @ vt
    scale = float(np.sum(svals * sign) / var_src)
    trans = mu_dst - scale * (r @ mu_src)
    residual = dst - (scale * (src @ r.T) + trans)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return scale, r, trans, rmse


def robust_fit_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    scale, rot, trans, _ = fit_similarity(src, dst)
    pred = scale * (src @ rot.T) + trans
    err = np.linalg.norm(dst - pred, axis=1)
    keep = err <= np.percentile(err, 90.0)
    if keep.sum() >= 16:
        scale, rot, trans, rmse = fit_similarity(src[keep], dst[keep])
    else:
        _, _, _, rmse = fit_similarity(src, dst)
    return scale, rot, trans, rmse


def similarity_matrix(scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = (scale * rot).astype(np.float32)
    mat[:3, 3] = trans.astype(np.float32)
    return mat


def first_existing(group: h5py.Group, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in group:
            return key
    return None


def patch_spatracker_copy(group: h5py.Group, max_fit_points: int, fit_metric_scene: bool) -> None:
    if "point_traj_spatracker" in group:
        spatracker = np.asarray(group["point_traj_spatracker"], dtype=np.float32)
    elif "point_traj_original" in group:
        spatracker = np.asarray(group["point_traj_original"], dtype=np.float32)
        replace_dataset(group, "point_traj_spatracker", spatracker)
    elif "point_traj" in group:
        spatracker = np.asarray(group["point_traj"], dtype=np.float32)
        replace_dataset(group, "point_traj_spatracker", spatracker)
    else:
        raise KeyError("No point_traj_spatracker, point_traj_original, or point_traj found")

    replace_dataset(group, "point_traj", spatracker)
    group.attrs["point_traj_active_source"] = "point_traj_spatracker"
    group.attrs["point_traj_mode"] = "spatracker"
    group.attrs["point_traj_modes_available"] = "point_traj_spatracker"
    group.attrs["point_traj_units"] = "spatracker_v2_relative"
    group.attrs["point_traj_coordinate_frame"] = "spatracker_v2_or_vggt"

    if not fit_metric_scene or "point_traj_base_metric" not in group:
        return

    metric = np.asarray(group["point_traj_base_metric"], dtype=np.float32)
    src, dst = finite_pairs(metric, spatracker, max_points=max_fit_points)
    scale, rot, trans, rmse = robust_fit_similarity(src, dst)
    metric_to_spatracker = similarity_matrix(scale, rot, trans)
    replace_dataset(group, "metric_to_spatracker", metric_to_spatracker, compression="gzip")
    group.attrs["metric_to_spatracker_scale"] = scale
    group.attrs["metric_to_spatracker_rmse"] = rmse
    group.attrs["metric_to_spatracker_note"] = "Maps robot-base metric xyz into SpaTracker/VGGT coordinates."

    depth_key = first_existing(group, ("sim_depths", "depths"))
    intr_key = first_existing(group, ("sim_intrinsics", "intrinsics"))
    pose_key = first_existing(group, ("sim_T_base_cam", "T_base_cam"))
    if depth_key is not None:
        replace_dataset(group, "depths", np.asarray(group[depth_key], dtype=np.float32))
    if intr_key is not None:
        replace_dataset(group, "intrinsics", np.asarray(group[intr_key], dtype=np.float32))
    if pose_key is not None:
        metric_cam = np.asarray(group[pose_key], dtype=np.float32)
        if metric_cam.ndim == 2:
            c2w = metric_to_spatracker @ metric_cam
        else:
            c2w = np.einsum("ij,tjk->tik", metric_to_spatracker, metric_cam).astype(np.float32)
        replace_dataset(group, "c2w", c2w)
        group.attrs["c2w_note"] = "Fitted SpaTracker-scale camera-to-scene transform for visualization."


def drop_metric_keys(group: h5py.Group) -> None:
    for key in METRIC_DATA_KEYS:
        if key in group:
            del group[key]
    for attr_key in list(group.attrs.keys()):
        if any(str(attr_key).startswith(prefix) for prefix in METRIC_ATTR_PREFIXES):
            del group.attrs[attr_key]


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    copy_one_demo(src, out, str(args.demo_id), overwrite=bool(args.overwrite))

    with h5py.File(out, "r+") as fout:
        group = fout[f"data/{args.demo_id}"]
        patch_spatracker_copy(
            group,
            max_fit_points=int(args.max_fit_points),
            fit_metric_scene=bool(args.fit_metric_scene),
        )
        if args.drop_metric_keys:
            drop_metric_keys(group)
        group.attrs["source_hdf5"] = str(src)
        group.attrs["source_demo_id"] = str(args.demo_id)
        group.attrs["export_note"] = "SpaTracker-scale copy; source file was not modified."

    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
