"""
Predict future 3D point trajectories and write them back to HDF5 demos.

This is the public, metric-aware inference entry point. It supports plain
SpaTracker-scale flows and calibrated metric flows, and can optionally save
2D debug projections using the same camera calibration stored in the processed
LIBERO / ManiSkill data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from transformers import SiglipProcessor

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _predict_flow_hdf5_impl as base


TRAJ_KEYS = ("point_traj", "point_traj_metric")


def infer_task_type_from_path(path: Path) -> str:
    text = str(path).lower()
    if any(k in text for k in ("maniskill", "pickcube", "pushcube", "stackcube")):
        return "maniskill"
    if any(k in text for k in ("libero", "kitchen_scene", "living_room_scene", "study_scene")):
        return "libero"
    return "libero"


def resolve_plus_is_close(task_type: str, override: str = "auto") -> bool:
    override = str(override).lower()
    if override in ("true", "1", "yes"):
        return True
    if override in ("false", "0", "no"):
        return False
    return str(task_type).lower() == "libero"


def binarize_gripper(g: np.ndarray, plus_is_close: bool) -> np.ndarray:
    g = np.nan_to_num(np.asarray(g, dtype=np.float32))
    if g.size == 0:
        return np.zeros((0,), dtype=np.int32)
    thr = 0.5 * (float(np.nanmin(g)) + float(np.nanmax(g)))
    if plus_is_close:
        return (g > thr).astype(np.int32)
    return (g < thr).astype(np.int32)


def debounce_gripper_changes(gb: np.ndarray, debounce: int = 3) -> List[int]:
    gb = np.asarray(gb, dtype=np.int32)
    if gb.size <= 1:
        return []
    if debounce <= 1:
        return (np.where(np.diff(gb) != 0)[0] + 1).astype(int).tolist()

    cur = int(gb[0])
    changes: List[int] = []
    i = 1
    while i < gb.size:
        if int(gb[i]) == cur:
            i += 1
            continue
        new = int(gb[i])
        ok = True
        for j in range(i, min(gb.size, i + debounce)):
            if int(gb[j]) != new:
                ok = False
                break
        if ok:
            changes.append(i)
            cur = new
            i += debounce
        else:
            i += 1
    return changes


def segment_gripper_state(
    actions: Optional[np.ndarray],
    t_len: int,
    plus_is_close: bool,
    debounce: int = 3,
) -> List[Tuple[int, int]]:
    t_len = int(t_len)
    if t_len <= 0:
        return []
    if actions is None or len(actions) == 0:
        return [(0, t_len - 1)]

    a = np.asarray(actions)
    g = a[:, -1] if a.ndim == 2 else a.reshape(-1)
    usable = min(t_len, int(g.shape[0]))
    if usable <= 0:
        return [(0, t_len - 1)]

    gb = binarize_gripper(g[:usable], plus_is_close=plus_is_close)
    change_idxs = debounce_gripper_changes(gb, debounce=debounce)
    boundaries = sorted(set([0] + change_idxs + [usable]))

    segments: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s0 = int(boundaries[i])
        s1 = int(boundaries[i + 1] - 1)
        if s1 >= s0:
            segments.append((s0, s1))
    if not segments:
        segments = [(0, usable - 1)]
    if usable < t_len:
        last_start, _ = segments[-1]
        segments[-1] = (last_start, t_len - 1)
    return segments


def first_existing_key(grp: h5py.Group, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in grp:
            return key
    return None


def resolve_debug_gt_key(demo: h5py.Group, args) -> Optional[str]:
    if args.debug_gt_key != "auto":
        return args.debug_gt_key if args.debug_gt_key in demo else None

    out_key = str(args.out_key).lower()
    preferred: List[str] = []
    if "metric" in out_key:
        preferred.append("point_traj_metric")
    if "metric" not in out_key:
        preferred.append("point_traj")

    for key in preferred:
        if key in demo:
            return key
    for key in TRAJ_KEYS:
        if key in demo:
            return key
    return None


def take_time(arr: np.ndarray, idx: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3:
        return arr[int(np.clip(idx, 0, arr.shape[0] - 1))]
    return arr


def project_cam_points(K: np.ndarray, xyz_cam: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32)
    xyz_cam = np.asarray(xyz_cam, dtype=np.float32)
    z = xyz_cam[:, 2]
    valid = np.isfinite(xyz_cam).all(axis=1) & (z > 1e-6)
    uv = np.full((xyz_cam.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = K[0, 0] * xyz_cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * xyz_cam[valid, 1] / z[valid] + K[1, 2]
    return uv


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([xyz, ones], axis=1)
    out = (np.asarray(T, dtype=np.float32) @ homo.T).T
    return out[:, :3]


def project_metric_trajectory(
    demo: h5py.Group,
    traj_xyz: np.ndarray,
    frame_indices: np.ndarray,
    current_frame: int,
    camera_frame: str,
) -> Tuple[np.ndarray, str]:
    traj_xyz = np.asarray(traj_xyz, dtype=np.float32)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)

    libero_K_key = first_existing_key(demo, ("sim_intrinsics", "intrinsics", "intrs2"))
    libero_pose_key = first_existing_key(demo, ("sim_T_base_cam", "T_base_cam"))
    if libero_K_key is not None and libero_pose_key is not None:
        Ks = np.asarray(demo[libero_K_key])
        T_base_cams = np.asarray(demo[libero_pose_key])
        uv = []
        for j in range(traj_xyz.shape[0]):
            pose_idx = current_frame if camera_frame == "current" else int(frame_indices[j])
            K = take_time(Ks, pose_idx)
            T_base_cam = take_time(T_base_cams, pose_idx)
            xyz_cam = transform_points(np.linalg.inv(T_base_cam), traj_xyz[j])
            uv.append(project_cam_points(K, xyz_cam))
        return np.stack(uv, axis=0), f"{libero_K_key}+{libero_pose_key}"

    K_key = first_existing_key(demo, ("intrinsics", "intrs2", "sim_intrinsics"))
    w2c_key = first_existing_key(demo, ("w2c_traj", "w2c", "camera_extrinsics"))
    c2w_key = first_existing_key(demo, ("c2w_traj", "c2w", "extrinsics"))
    if K_key is not None and (w2c_key is not None or c2w_key is not None):
        Ks = np.asarray(demo[K_key])
        poses = np.asarray(demo[w2c_key]) if w2c_key is not None else np.asarray(demo[c2w_key])
        pose_label = w2c_key if w2c_key is not None else c2w_key
        uv = []
        for j in range(traj_xyz.shape[0]):
            pose_idx = current_frame if camera_frame == "current" else int(frame_indices[j])
            K = take_time(Ks, pose_idx)
            pose = take_time(poses, pose_idx)
            w2c = pose if w2c_key is not None else np.linalg.inv(pose)
            xyz_cam = transform_points(w2c, traj_xyz[j])
            uv.append(project_cam_points(K, xyz_cam))
        return np.stack(uv, axis=0), f"{K_key}+{pose_label}"

    raise KeyError(
        "No metric camera calibration found. Expected LIBERO keys "
        "(sim_intrinsics, sim_T_base_cam) or ManiSkill keys "
        "(intrinsics, w2c_traj/c2w_traj)."
    )


def draw_pred_gt_tracks_on_image(
    img: np.ndarray,
    pred_uv: np.ndarray,
    gt_uv: Optional[np.ndarray],
    title: str,
    out_path: Path,
    stride: int = 1,
    alpha_pred: float = 0.9,
    alpha_gt: float = 0.55,
):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.imshow(img)

    k_steps = int(pred_uv.shape[0])
    colors = cm.get_cmap("turbo", max(k_steps, 2))
    for n in range(0, pred_uv.shape[1], stride):
        if gt_uv is not None:
            pts_g = gt_uv[:, n]
            m_g = np.isfinite(pts_g).all(axis=1)
            if m_g.sum() >= 2:
                plt.plot(pts_g[m_g, 0], pts_g[m_g, 1], "--", linewidth=1.8, alpha=alpha_gt, color="white")
                plt.plot(pts_g[m_g, 0], pts_g[m_g, 1], "--", linewidth=1.0, alpha=alpha_gt, color="black")

        pts_p = pred_uv[:, n]
        m_p = np.isfinite(pts_p).all(axis=1)
        if m_p.sum() >= 2:
            color = colors(float(n % max(pred_uv.shape[1], 1)) / max(pred_uv.shape[1] - 1, 1))
            plt.plot(pts_p[m_p, 0], pts_p[m_p, 1], "-", linewidth=1.0, alpha=alpha_pred, color=color)
            first_idx = np.where(m_p)[0][0]
            last_idx = np.where(m_p)[0][-1]
            plt.scatter(pts_p[first_idx, 0], pts_p[first_idx, 1], s=10, c="white",
                        edgecolors="black", linewidths=0.3, alpha=0.85)
            plt.scatter(pts_p[last_idx, 0], pts_p[last_idx, 1], s=12, c="red",
                        edgecolors="black", linewidths=0.3, alpha=0.85)

    plt.xlim(0, w - 1)
    plt.ylim(h - 1, 0)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_debug_images(
    demo: h5py.Group,
    frames_ds,
    pred_traj: np.ndarray,
    gt_key: Optional[str],
    segments: List[Tuple[int, int]],
    instruction: str,
    demo_id: str,
    args,
):
    debug_root = Path(args.debug_tracks_dir)
    safe_instruction = re.sub(r"[^\w\-_.() ]+", "_", str(instruction)).strip()
    out_dir = debug_root / safe_instruction
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_ds = demo[gt_key] if gt_key is not None and gt_key in demo else None
    t_len = pred_traj.shape[0]
    for step in range(t_len):
        seg_end = base.find_segment_end(segments, step, t_len)
        key_idxs = base.resample_on_traj(step, seg_end, args.k_steps)
        key_idxs = np.clip(key_idxs, 0, t_len - 1)

        pred_xyz = pred_traj[step]
        gt_xyz = base.h5_take_time(gt_ds, key_idxs, dtype=np.float32) if gt_ds is not None else None

        projection = args.debug_projection
        if projection == "auto":
            projection = "metric" if gt_key != "point_traj" else "pnp"

        if projection == "metric":
            pred_uv, proj_label = project_metric_trajectory(
                demo,
                pred_xyz,
                key_idxs,
                current_frame=step,
                camera_frame=args.debug_camera_frame,
            )
            gt_uv = None
            if gt_xyz is not None:
                gt_uv, _ = project_metric_trajectory(
                    demo,
                    gt_xyz,
                    key_idxs,
                    current_frame=step,
                    camera_frame=args.debug_camera_frame,
                )
            draw_pred_gt_tracks_on_image(
                base._to_rgb_hwc(frames_ds[step]),
                pred_uv=pred_uv,
                gt_uv=gt_uv,
                title=f"metric debug: {proj_label}",
                out_path=out_dir / f"tracks_checks_{demo_id}_{step}.png",
            )
        elif projection == "pnp":
            if "intrs2" not in demo or "track2d" not in demo or "point_traj" not in demo:
                raise KeyError("PnP debug needs intrs2, track2d, and point_traj.")
            base.draw_tracks(
                base._to_rgb_hwc(frames_ds[step]),
                traj_world_pred=pred_xyz,
                traj_world_gt=gt_xyz,
                intrinsics=np.asarray(demo["intrs2"])[0],
                P0_world=np.asarray(demo["point_traj"])[0],
                track2d=np.asarray(demo["track2d"])[0][..., :2],
                instruction=instruction,
                demo_id=demo_id,
                step=step,
                k_steps=args.k_steps,
                out_dir=args.debug_tracks_dir,
            )
        else:
            raise ValueError(f"Unknown debug projection: {args.debug_projection}")


def process_hdf5_file(h5_path: Path, model, processor: SiglipProcessor, args) -> int:
    with h5py.File(str(h5_path), "r+") as f:
        if args.data_group not in f:
            return 0

        task_type = args.task_type if args.task_type != "auto" else infer_task_type_from_path(h5_path)
        plus_is_close = resolve_plus_is_close(task_type, args.plus_is_close)

        data_grp = f[args.data_group]
        demo_ids = list(data_grp.keys())
        if args.max_demos is not None:
            demo_ids = demo_ids[: args.max_demos]

        wrote = 0
        for demo_id in demo_ids:
            demo = data_grp[demo_id]
            frames_key = first_existing_key(demo, (args.frames_key, "frames_rgb"))
            if frames_key is None:
                continue

            qp_key = first_existing_key(demo, (args.query_key, "query_xy_t0"))
            if (not args.no_query_points) and qp_key is None:
                continue

            if (args.out_key in demo) and (not args.overwrite):
                continue

            frames_ds = demo[frames_key]
            if args.no_query_points:
                query_points = None
                qp_key = None
            else:
                query_points = base.normalize_query_points_for_model(
                    demo[qp_key][:],
                    frame_hw=base.infer_frame_hw(frames_ds.shape),
                )

            instruction = base.normalize_instr_key(h5_path.stem)
            pred_traj = base.infer_pred_point_traj_for_demo(
                model=model,
                processor=processor,
                frames_ds=frames_ds,
                query_points_np=query_points,
                instruction=str(instruction),
                k_steps=args.k_steps,
                num_points=args.num_points,
                ddim_steps=args.ddim_steps,
                cond_k=args.cond_k,
                cond_stride=args.cond_stride,
                window_bs=args.window_bs,
                device=args.device,
                guidance_scale=args.guidance_scale,
            )

            gt_key = resolve_debug_gt_key(demo, args)
            if args.debug_tracks_dir is not None:
                actions_np = np.asarray(demo["actions"][:]) if "actions" in demo else None
                segments = segment_gripper_state(
                    actions_np,
                    pred_traj.shape[0],
                    plus_is_close=plus_is_close,
                    debounce=3,
                )
                make_debug_images(
                    demo=demo,
                    frames_ds=frames_ds,
                    pred_traj=pred_traj,
                    gt_key=gt_key,
                    segments=segments,
                    instruction=instruction,
                    demo_id=str(demo_id),
                    args=args,
                )

            if not args.dry_run:
                base.overwrite_h5_dataset(demo, args.out_key, pred_traj, compression="gzip", chunks=True)
                demo.attrs[f"{args.out_key}_ckpt"] = str(args.ckpt)
                demo.attrs[f"{args.out_key}_ddim_steps"] = int(args.ddim_steps)
                demo.attrs[f"{args.out_key}_guidance_scale"] = float(args.guidance_scale)
                demo.attrs[f"{args.out_key}_k_steps"] = int(args.k_steps)
                demo.attrs[f"{args.out_key}_query_points"] = bool(not args.no_query_points)
                demo.attrs[f"{args.out_key}_query_key"] = "" if qp_key is None else str(qp_key)
                demo.attrs[f"{args.out_key}_debug_gt_key"] = "" if gt_key is None else str(gt_key)
                demo.attrs[f"{args.out_key}_debug_projection"] = str(args.debug_projection)
                demo.attrs[f"{args.out_key}_task_type"] = str(task_type)
                wrote += 1

        if not args.dry_run:
            f.flush()
        return wrote


def list_hdf5_files(root: Path) -> List[Path]:
    if root.is_file() and root.suffix in (".hdf5", ".h5"):
        return [root]
    return sorted(p for p in root.rglob("*.hdf5") if p.is_file())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_py", type=str, default="flow_model.py")
    parser.add_argument("--siglip_name", type=str, default="google/siglip-base-patch16-224")

    parser.add_argument("--data_group", type=str, default="data")
    parser.add_argument("--frames_key", type=str, default="frames_rgb")
    parser.add_argument("--query_key", type=str, default="query_points")
    parser.add_argument("--no_query_points", action="store_true")
    parser.add_argument("--instr_attr", type=str, default="prompt")

    parser.add_argument("--out_key", type=str, default="pre_point_traj")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--debug_tracks_dir", type=str, default=None)
    parser.add_argument(
        "--debug_gt_key",
        type=str,
        default="auto",
        help="GT trajectory key for debug images. Use point_traj_metric for metric checkpoints.",
    )
    parser.add_argument("--debug_projection", choices=("auto", "metric", "pnp"), default="auto")
    parser.add_argument("--debug_camera_frame", choices=("current", "keyframe"), default="current")

    parser.add_argument("--task_type", choices=("auto", "libero", "maniskill"), default="auto")
    parser.add_argument("--plus_is_close", choices=("auto", "true", "false"), default="auto")

    parser.add_argument("--k_steps", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--cond_k", type=int, default=4)
    parser.add_argument("--cond_stride", type=int, default=1)
    parser.add_argument("--window_bs", type=int, default=150)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--norm_stats", type=str, default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.norm_stats is not None:
        with open(args.norm_stats, "r") as rf:
            args.norm_stats = json.load(rf)

    base.set_seed(args.seed)
    base.enable_sdpa()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_py = Path(args.model_py)
    if not model_py.is_absolute() and not model_py.exists():
        cand = HERE / model_py
        if cand.exists():
            model_py = cand
    args.model_py = str(model_py)

    print("[INFO] Loading SigLIP processor:", args.siglip_name)
    processor = SiglipProcessor.from_pretrained(args.siglip_name, use_fast=True)

    print("[INFO] Loading model from ckpt:", args.ckpt)
    model, k_ckpt, n_ckpt, cfg = base.load_flow_model(
        ckpt_path=Path(args.ckpt),
        model_py=Path(args.model_py),
        device=args.device,
        k_steps_override=args.k_steps,
        num_points_override=args.num_points,
        strict=True,
    )
    args.k_steps = int(args.k_steps if args.k_steps is not None else k_ckpt)
    args.num_points = int(args.num_points if args.num_points is not None else n_ckpt)

    files = list_hdf5_files(Path(args.hdf5_root))
    if args.max_files is not None:
        files = files[: args.max_files]

    print(
        f"[INFO] device={args.device}, files={len(files)}, k_steps={args.k_steps}, "
        f"num_points={args.num_points}, out_key={args.out_key}, debug_gt_key={args.debug_gt_key}, "
        f"debug_projection={args.debug_projection}"
    )
    print(f"[INFO] model cfg keys: {list(cfg.keys())}")

    total_wrote = 0
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        wrote = process_hdf5_file(fp, model, processor, args)
        total_wrote += wrote
        if wrote > 0:
            print(f"[{i}/{len(files)}] wrote {wrote} demos into {fp}")
    print(f"[DONE] total demos written: {total_wrote}, elapsed: {time.time() - t0:.1f}s")
    if args.dry_run:
        print("[NOTE] dry_run=True: no file was modified.")


if __name__ == "__main__":
    main()
