#!/usr/bin/env python3
"""Export one demo with native SpaTracker/VGGT-scale scene geometry.

This script is intentionally non-destructive. It reads an existing processed
track HDF5 demo for RGB frames and query points, reruns VGGT + SpaTrackerV2,
and writes a new HDF5 file containing a self-consistent SpaTracker-scale scene:

  point_traj / point_traj_spatracker
  depths / intrs2 / c2w
  track2d / vis / frames_rgb

Unlike ``make_spatracker_scale_demo.py``, this does not fit a metric scene into
SpaTracker coordinates. The dense scene is reconstructed directly from the
VGGT/SpaTracker predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPA_ROOT = REPO_ROOT / "SpaTrackerV2"
if SPA_ROOT.exists() and str(SPA_ROOT) not in sys.path:
    sys.path.insert(0, str(SPA_ROOT))

from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track  # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image  # noqa: E402
from models.SpaTrackV2.models.utils import get_points_on_a_grid  # noqa: E402
from utils.motion_filter import filter_points_moving_and_sor_firstframe  # noqa: E402


DEFAULT_COPY_KEYS = (
    "actions",
    "dones",
    "robot_states",
    "wrist_frames",
    "vggt_hidden",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun VGGT + SpaTrackerV2 for one HDF5 demo and save native SpaTracker-scale geometry."
    )
    parser.add_argument("--src", required=True, help="Source HDF5 containing data/<demo_id>/frames_rgb.")
    parser.add_argument("--out", required=True, help="Output HDF5 path. The source is never modified.")
    parser.add_argument("--demo_id", default="demo_0", help="Demo group id under /data.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")

    parser.add_argument(
        "--query_key",
        default="auto",
        help="Query point dataset. auto tries query_xy_t0, grid_points_xy, then p0_uv. Use 'grid' to build a dense grid.",
    )
    parser.add_argument("--num_points", type=int, default=100, help="Number of query points to keep or sample.")
    parser.add_argument("--grid_size", type=int, default=50, help="Grid size when --query_key grid or fallback is used.")
    parser.add_argument(
        "--query_coord_mode",
        choices=("auto", "tracker", "image"),
        default="auto",
        help="tracker means coordinates are already in the 518-style preprocessed frame; image rescales from raw RGB pixels.",
    )

    parser.add_argument("--track_mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--device", default="cuda:0", help="SpaTracker device.")
    parser.add_argument("--vggt_device", default=None, help="VGGT device. Defaults to --device.")
    parser.add_argument("--vggt_amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--vggt_chunk", type=int, default=48)
    parser.add_argument("--iters_track", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=-1, help="-1 means last frame.")
    parser.add_argument("--replace_ratio", type=float, default=0.2)

    parser.add_argument("--motion_thresh", type=float, default=0.25)
    parser.add_argument("--sor_k", type=int, default=64)
    parser.add_argument("--sor_std_ratio", type=float, default=2.5)
    parser.add_argument("--no_filter_points", action="store_true", help="Disable moving-point + SOR filtering.")
    parser.add_argument(
        "--no_mask_low_conf_depth",
        action="store_true",
        help="Keep low-confidence dense depth points instead of zeroing them with SpaTracker confidence.",
    )

    parser.add_argument(
        "--copy_keys",
        nargs="*",
        default=list(DEFAULT_COPY_KEYS),
        help="Extra datasets copied from the source demo when present.",
    )
    return parser.parse_args()


def _get_amp_context(device: torch.device, amp_mode: str):
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda or amp_mode == "off":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def _pad_time(x: torch.Tensor, target_len: int, mode: str = "edge") -> torch.Tensor:
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    pad_len = target_len - x.shape[0]
    if mode == "ones":
        pad = torch.ones_like(x[:1]).expand(pad_len, *x.shape[1:])
    elif mode == "zeros":
        pad = torch.zeros_like(x[:1]).expand(pad_len, *x.shape[1:])
    else:
        pad = x[-1:].expand(pad_len, *x.shape[1:])
    return torch.cat([x, pad], dim=0)


def _canonize_vggt_chunk(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def squeeze_leading_one(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() >= 1 and x.size(0) == 1 else x

    def to_t1hw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.squeeze(0)
        if x.dim() == 4:
            if x.shape[1] in (1, 2, 3, 4) and x.shape[2] > 8 and x.shape[3] > 8:
                return x[:, :1].contiguous()
            return x[..., :1].permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            if x.shape[0] in (1, 2, 3, 4) and x.shape[1] > 8 and x.shape[2] > 8:
                return x[:1].unsqueeze(0).contiguous()
            return x.unsqueeze(1).contiguous()
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0).contiguous()
        raise RuntimeError(f"Unexpected unc_metric shape: {tuple(x.shape)}")

    out: Dict[str, torch.Tensor] = {}
    out["poses_pred"] = squeeze_leading_one(pred["poses_pred"]).contiguous()
    out["intrs"] = squeeze_leading_one(pred["intrs"]).contiguous()
    out["features"] = squeeze_leading_one(pred["features"]).contiguous()

    points_map = squeeze_leading_one(pred["points_map"]).contiguous()
    if points_map.dim() == 4:
        if points_map.shape[1] in (1, 2, 3, 4) and points_map.shape[2] > 8 and points_map.shape[3] > 8:
            points_map = points_map.permute(0, 2, 3, 1).contiguous()
    elif points_map.dim() == 3:
        if points_map.shape[0] in (1, 2, 3, 4) and points_map.shape[1] > 8 and points_map.shape[2] > 8:
            points_map = points_map.permute(1, 2, 0).unsqueeze(0).contiguous()
        else:
            points_map = points_map.unsqueeze(0).contiguous()
    else:
        raise RuntimeError(f"Unexpected points_map shape: {tuple(points_map.shape)}")
    out["points_map"] = points_map
    out["unc_metric"] = to_t1hw(pred["unc_metric"])
    return out


@torch.inference_mode()
def vggt_forward_chunked(
    vggt_front: VGGT4Track,
    video_tensor: torch.Tensor,
    vggt_dev: torch.device,
    amp_mode: str,
    chunk_len: int,
    overlap: int = 1,
) -> Dict[str, torch.Tensor]:
    total_frames = int(video_tensor.shape[0])
    outputs = {key: [] for key in ("poses_pred", "intrs", "points_map", "unc_metric", "features")}
    amp_ctx = _get_amp_context(vggt_dev, amp_mode)

    start = 0
    while start < total_frames:
        end = min(total_frames, start + int(chunk_len))
        chunk_start = start if start == 0 else max(0, start - int(overlap))
        chunk = video_tensor[chunk_start:end].to(vggt_dev, non_blocking=True)
        needed = int(chunk.shape[0])
        cut = 0 if start == 0 else int(overlap)

        with amp_ctx:
            raw = vggt_front(chunk[None] / 255.0)
        pred = _canonize_vggt_chunk(raw)

        for key in outputs:
            value = pred[key]
            if value.shape[0] != needed:
                value = _pad_time(value, needed, mode="ones" if key == "unc_metric" else "edge")
            outputs[key].append(value[cut:].contiguous())

        start = end
        del raw, pred, chunk
        if str(vggt_dev).startswith("cuda"):
            torch.cuda.empty_cache()

    return {key: torch.cat(chunks, dim=0)[:total_frames].contiguous() for key, chunks in outputs.items()}


def to_rgb_frames(raw: np.ndarray) -> np.ndarray:
    frames = np.asarray(raw)
    if frames.ndim != 4:
        raise ValueError(f"frames_rgb must be 4D, got {frames.shape}")
    if frames.shape[-1] == 3:
        out = frames
    elif frames.shape[1] == 3:
        out = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise ValueError(f"Cannot infer RGB layout from frames_rgb shape {frames.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def select_queries(
    group: h5py.Group,
    query_key: str,
    num_points: int,
    grid_size: int,
    tracker_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    coord_mode: str,
) -> Tuple[np.ndarray, str]:
    if query_key == "auto":
        for candidate in ("query_xy_t0", "grid_points_xy", "p0_uv"):
            if candidate in group:
                query_key = candidate
                break
        else:
            query_key = "grid"

    if query_key == "grid":
        queries = get_points_on_a_grid(int(grid_size), tracker_hw, device="cpu")[0].cpu().numpy()
    else:
        if query_key not in group:
            raise KeyError(f"query_key={query_key!r} not found. Available keys={sorted(group.keys())}")
        queries = np.asarray(group[query_key], dtype=np.float32)
        if queries.ndim == 3:
            queries = queries[0]
        if queries.ndim != 2 or queries.shape[1] < 2:
            raise ValueError(f"{query_key} must have shape [N,2] or [T,N,2], got {queries.shape}")
        queries = queries[:, :2].astype(np.float32)

    h_img, w_img = image_hw
    h_track, w_track = tracker_hw
    if coord_mode == "image" or (
        coord_mode == "auto"
        and np.nanmax(queries[:, 0]) <= w_img + 1.5
        and np.nanmax(queries[:, 1]) <= h_img + 1.5
    ):
        queries = queries.copy()
        queries[:, 0] *= float(w_track) / float(w_img)
        queries[:, 1] *= float(h_track) / float(h_img)

    finite = np.isfinite(queries).all(axis=1)
    queries = queries[finite]
    if queries.shape[0] == 0:
        raise ValueError("No finite query points after filtering.")

    if num_points > 0 and queries.shape[0] != num_points:
        rng = np.random.default_rng(0)
        if queries.shape[0] > num_points:
            idx = rng.choice(queries.shape[0], size=num_points, replace=False)
            queries = queries[np.sort(idx)]
        else:
            idx = rng.choice(queries.shape[0], size=num_points - queries.shape[0], replace=True)
            queries = np.concatenate([queries, queries[idx]], axis=0)

    queries[:, 0] = np.clip(queries[:, 0], 0.0, float(w_track - 1))
    queries[:, 1] = np.clip(queries[:, 1], 0.0, float(h_track - 1))
    return queries.astype(np.float32), query_key


def project_world_to_pixels(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    z = np.clip(points[:, 2:3], 1e-6, None)
    uvw = (K @ points.T).T
    return uvw[:, :2] / z


def poses_to_4x4_numpy(poses: torch.Tensor) -> np.ndarray:
    arr = poses.detach().float().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[-2:] == (3, 4):
        bottom = np.zeros(arr.shape[:-2] + (1, 4), dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        arr = np.concatenate([arr, bottom], axis=-2)
    return arr.astype(np.float32)


def extract_depth_map(point_map: torch.Tensor) -> torch.Tensor:
    """Return [T,H,W] depth from a SpaTracker point_map tensor."""
    if point_map.dim() == 4:
        if point_map.shape[1] in (3, 4) and point_map.shape[2] > 8 and point_map.shape[3] > 8:
            return point_map[:, 2]
        if point_map.shape[-1] in (3, 4) and point_map.shape[1] > 8 and point_map.shape[2] > 8:
            return point_map[..., 2]
    if point_map.dim() == 3:
        return point_map
    raise ValueError(f"Cannot extract depth from point_map shape {tuple(point_map.shape)}")


def squeeze_confidence(conf_depth: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Return [T,H,W] confidence aligned with a depth map when possible."""
    conf = conf_depth
    if conf.dim() == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]
    elif conf.dim() == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if tuple(conf.shape) != tuple(target_shape):
        raise ValueError(f"conf_depth shape {tuple(conf.shape)} does not match depth shape {target_shape}")
    return conf


def replace_dataset(group: h5py.Group, key: str, value: np.ndarray, compression: Optional[str] = "gzip") -> None:
    if key in group:
        del group[key]
    kwargs = {}
    if compression and np.asarray(value).ndim > 0:
        kwargs["compression"] = compression
    group.create_dataset(key, data=value, **kwargs)


def copy_attrs(src: h5py.Group, dst: h5py.Group) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_extra_keys(src: h5py.Group, dst: h5py.Group, keys) -> None:
    for key in keys:
        if key in src and key not in dst:
            replace_dataset(dst, key, np.asarray(src[key]), compression="gzip")


@torch.inference_mode()
def run_spatracker(
    frames_rgb: np.ndarray,
    queries_xy: np.ndarray,
    predictor: Predictor,
    vggt_front: VGGT4Track,
    device: torch.device,
    vggt_device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    video_raw = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).contiguous().float()
    video_pre = preprocess_image(video_raw)
    vggt_video = video_pre.to(vggt_device, non_blocking=True)
    preds = vggt_forward_chunked(vggt_front, vggt_video, vggt_device, args.vggt_amp, args.vggt_chunk)

    vggt_hidden = preds["features"].mean(1).detach().float().cpu().numpy().astype(np.float32)
    extrs = preds["poses_pred"].to(device).contiguous()
    intrs = preds["intrs"].to(device).contiguous()
    points_map = preds["points_map"].to(device).contiguous()
    unc_conf = preds["unc_metric"].to(device).contiguous()
    del preds, vggt_video
    if str(vggt_device).startswith("cuda"):
        torch.cuda.empty_cache()

    depth_tensor = points_map[..., 2] if points_map.shape[-1] == 3 else points_map[:, 2]
    extrinsics_np = poses_to_4x4_numpy(extrs)
    unc_metric = (unc_conf[:, 0] > 0.5).float() if unc_conf.dim() == 4 else (unc_conf > 0.5).float()

    query_xyt = np.concatenate(
        [np.zeros((queries_xy.shape[0], 1), dtype=np.float32), queries_xy.astype(np.float32)],
        axis=1,
    )

    predictor_video = video_pre.to(device, non_blocking=True)
    support_frame = int(args.support_frame)
    if support_frame < 0:
        support_frame = int(predictor_video.shape[0]) - 1

    amp_ctx = _get_amp_context(device, "bf16")
    with amp_ctx:
        (
            c2w_traj,
            intrs2,
            point_map,
            conf_depth,
            track3d_pred,
            track2d_pred,
            vis_pred,
            conf_pred,
            video_out,
        ) = predictor.forward(
            predictor_video,
            depth=depth_tensor,
            intrs=intrs,
            extrs=extrs,
            queries=query_xyt,
            fps=1,
            full_point=False,
            iters_track=int(args.iters_track),
            query_no_BA=True,
            fixed_cam=True,
            stage=1,
            unc_metric=unc_metric,
            support_frame=support_frame,
            replace_ratio=float(args.replace_ratio),
        )

    dense_depth = extract_depth_map(point_map).detach().float()
    dense_conf = squeeze_confidence(conf_depth.detach().float(), tuple(dense_depth.shape))
    if not args.no_mask_low_conf_depth:
        dense_depth = dense_depth.clone()
        dense_depth[dense_conf < 0.5] = 0.0
    depths = dense_depth.cpu().numpy().astype(np.float32)
    depth_conf_np = dense_conf.cpu().numpy().astype(np.float32)

    point_traj = (
        torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[..., :3].cpu())
        + c2w_traj[:, :3, 3][:, None, :]
    )

    if not args.no_filter_points:
        point_traj, _, _, _ = filter_points_moving_and_sor_firstframe(
            point_traj,
            motion_thresh=float(args.motion_thresh),
            k=int(args.sor_k),
            std_ratio=float(args.sor_std_ratio),
            replace_outliers=True,
        )

    point_traj_np = point_traj.detach().float().cpu().numpy().astype(np.float32)
    intrs2_np = intrs2.detach().float().cpu().numpy().astype(np.float32)
    c2w_np = c2w_traj.detach().float().cpu().numpy().astype(np.float32)

    track2d = []
    for frame_idx in range(point_traj_np.shape[0]):
        K = intrs2_np[frame_idx] if intrs2_np.ndim == 3 else intrs2_np
        track2d.append(project_world_to_pixels(K[:3, :3], point_traj_np[frame_idx]))
    track2d_np = np.stack(track2d, axis=0).astype(np.float32)

    del extrs, intrs, points_map, unc_conf, depth_tensor, unc_metric, predictor_video
    del point_map, conf_depth, track3d_pred, track2d_pred, conf_pred, video_out
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "frames_rgb": frames_rgb.astype(np.uint8),
        "depths": depths,
        "depth_conf": depth_conf_np,
        "intrs2": intrs2_np,
        "c2w": c2w_np,
        "c2w_traj": c2w_np,
        "extrinsics": extrinsics_np,
        "point_traj": point_traj_np,
        "point_traj_spatracker": point_traj_np,
        "track2d": track2d_np,
        "vis": vis_pred.detach().float().cpu().numpy().astype(np.float32),
        "grid_points_xy": queries_xy.astype(np.float32),
        "query_xy_t0": queries_xy.astype(np.float32),
        "p0_uv": track2d_np[0].astype(np.float32),
        "vggt_hidden": vggt_hidden,
    }


def main() -> None:
    args = parse_args()
    src_path = Path(args.src).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    vggt_device = torch.device(args.vggt_device or args.device)

    with h5py.File(src_path, "r") as src_file:
        src_group = src_file[f"data/{args.demo_id}"]
        frames_rgb = to_rgb_frames(np.asarray(src_group["frames_rgb"]))
        video_raw = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).contiguous().float()
        pre_shape = preprocess_image(video_raw[:1]).shape[-2:]
        queries, used_query_key = select_queries(
            src_group,
            args.query_key,
            int(args.num_points),
            int(args.grid_size),
            tracker_hw=(int(pre_shape[0]), int(pre_shape[1])),
            image_hw=(int(frames_rgb.shape[1]), int(frames_rgb.shape[2])),
            coord_mode=args.query_coord_mode,
        )

        print(f"[input] frames={frames_rgb.shape}, tracker_hw={tuple(pre_shape)}, queries={queries.shape} from {used_query_key}")
        print("[models] loading VGGT and SpaTrackerV2...")
        vggt_front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(vggt_device)
        predictor = Predictor.from_pretrained(f"Yuxihenry/SpatialTrackerV2-{args.track_mode.capitalize()}")
        predictor.eval().to(device)
        predictor.spatrack.track_num = int(args.num_points) if int(args.num_points) > 0 else int(queries.shape[0])

        tracks = run_spatracker(frames_rgb, queries, predictor, vggt_front, device, vggt_device, args)

        if out_path.exists():
            out_path.unlink()
        with h5py.File(out_path, "w") as out_file:
            dst_group = out_file.create_group(f"data/{args.demo_id}")
            copy_attrs(src_group, dst_group)
            copy_extra_keys(src_group, dst_group, args.copy_keys)
            for key, value in tracks.items():
                replace_dataset(dst_group, key, value, compression="gzip")

            dst_group.attrs["has_tracks"] = True
            dst_group.attrs["source_hdf5"] = str(src_path)
            dst_group.attrs["source_demo_id"] = str(args.demo_id)
            dst_group.attrs["export_note"] = "Native SpaTracker/VGGT-scale export; source file was not modified."
            dst_group.attrs["point_traj_active_source"] = "point_traj_spatracker"
            dst_group.attrs["point_traj_mode"] = "spatracker"
            dst_group.attrs["point_traj_modes_available"] = "point_traj_spatracker"
            dst_group.attrs["point_traj_units"] = "spatracker_v2_relative"
            dst_group.attrs["point_traj_coordinate_frame"] = "spatracker_v2_or_vggt"
            dst_group.attrs["scene_geometry_source"] = "vggt_spatracker_direct"
            dst_group.attrs["query_source_key"] = used_query_key

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
