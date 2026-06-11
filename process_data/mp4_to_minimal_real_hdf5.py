#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MP4 videos into the minimal HDF5 layout expected by process_arm_real.py.

Output layout:
  <out_dir>/<video_stem>.hdf5
    attrs:
      source_video
      task_text
      fps
      num_frames
    /data/demo_0/
      attrs:
        prompt
        source_video
      observations/
        image   uint8   [T, H, W, 3]
        state   float32 [T, state_dim]
      actions   float32 [T, action_dim]
      dones     int8    [T]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing mp4 files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save converted hdf5 files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern under input_dir.",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="image",
        help="Dataset key under observations/ used by process_arm_real.py.",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=1,
        help="Dummy robot state dimension.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=1,
        help="Dummy action dimension.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="robot gripper",
        help="Prompt stored into demo attrs for downstream real-video tracking. Override this for non-gripper targets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hdf5 outputs.",
    )
    return parser.parse_args()


def normalize_task_slug(task_text: str) -> str:
    task_text = task_text.strip().lower()
    task_text = re.sub(r"[^a-z0-9]+", "_", task_text)
    task_text = re.sub(r"_+", "_", task_text).strip("_")
    return task_text or "video"


def parse_video_metadata(video_path: Path) -> Dict[str, Optional[str]]:
    stem = video_path.stem
    task_text = stem
    success = None
    env_id = None

    task_match = re.search(r"task=(.+?)(?:--success=|$)", stem)
    if task_match:
        task_text = task_match.group(1).strip()

    success_match = re.search(r"--success=([^-\s]+)", stem)
    if success_match:
        success = success_match.group(1)

    env_match = re.search(r"--env_id([^-\s]+)", stem)
    if env_match:
        env_id = env_match.group(1)

    return {
        "task_text": task_text,
        "task_slug": normalize_task_slug(task_text),
        "success": success,
        "env_id": env_id,
    }


def derive_output_name(video_path: Path, meta: Dict[str, Optional[str]]) -> str:
    parts: List[str] = [str(meta["task_slug"] or video_path.stem)]
    if meta.get("success"):
        parts.append(f"success_{meta['success']}")
    if meta.get("env_id"):
        parts.append(f"env_id_{meta['env_id']}")
    return "__".join(parts) + ".hdf5"


def read_video_rgb(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return np.stack(frames, axis=0), fps


def make_dummy_array(length: int, dim: int, dtype: np.dtype) -> np.ndarray:
    return np.zeros((length, dim), dtype=dtype)


def make_dones(length: int) -> np.ndarray:
    dones = np.zeros((length,), dtype=np.int8)
    dones[-1] = 1
    return dones


def write_hdf5(
    out_path: Path,
    video_path: Path,
    frames_rgb: np.ndarray,
    fps: float,
    image_key: str,
    state_dim: int,
    action_dim: int,
    prompt: str,
    meta: Dict[str, Optional[str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_len = int(frames_rgb.shape[0])

    with h5py.File(out_path, "w") as f:
        f.attrs["source_video"] = str(video_path)
        f.attrs["task_text"] = str(meta.get("task_text") or "")
        f.attrs["task_slug"] = str(meta.get("task_slug") or "")
        f.attrs["fps"] = float(fps)
        f.attrs["num_frames"] = int(t_len)
        if meta.get("success") is not None:
            f.attrs["success"] = str(meta["success"])
        if meta.get("env_id") is not None:
            f.attrs["env_id"] = str(meta["env_id"])

        data_grp = f.create_group("data")
        demo_grp = data_grp.create_group("demo_0")
        demo_grp.attrs["prompt"] = str(prompt)
        demo_grp.attrs["source_video"] = str(video_path)

        obs_grp = demo_grp.create_group("observations")
        obs_grp.create_dataset(image_key, data=frames_rgb, compression="gzip")
        obs_grp.create_dataset(
            "state",
            data=make_dummy_array(t_len, state_dim, np.float32),
            compression="gzip",
        )
        demo_grp.create_dataset(
            "actions",
            data=make_dummy_array(t_len, action_dim, np.float32),
            compression="gzip",
        )
        demo_grp.create_dataset("dones", data=make_dones(t_len), compression="gzip")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    video_paths = sorted(input_dir.glob(args.pattern))
    if not video_paths:
        raise RuntimeError(f"No videos found under {input_dir} with pattern {args.pattern}")

    print(f"[INFO] Found {len(video_paths)} video(s) in {input_dir}")
    for video_path in video_paths:
        meta = parse_video_metadata(video_path)
        out_name = derive_output_name(video_path, meta)
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path}")
            continue

        print(f"[READ] {video_path}")
        frames_rgb, fps = read_video_rgb(video_path)
        prompt = str(args.prompt)

        write_hdf5(
            out_path=out_path,
            video_path=video_path,
            frames_rgb=frames_rgb,
            fps=fps,
            image_key=args.image_key,
            state_dim=int(args.state_dim),
            action_dim=int(args.action_dim),
            prompt=prompt,
            meta=meta,
        )
        print(
            f"[OK] {out_path} "
            f"frames={frames_rgb.shape[0]} size={frames_rgb.shape[2]}x{frames_rgb.shape[1]} fps={fps:.3f}"
        )


if __name__ == "__main__":
    main()
