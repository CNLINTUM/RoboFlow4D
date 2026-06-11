"""Shared visualization helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


class TrajectorySegmenter:
    def __init__(
        self,
        gripper_debounce: int = 3,
        keep_last_segment: bool = True,
        min_seg_len: int = 10,
        gamma_s: float = 1.2,
        gamma_e: float = 1.6,
        min_unique_ratio: float = 0.7,
        plus_is_close: bool = True,
        normalize_gripper: bool = False,
    ):
        self.gripper_debounce = gripper_debounce
        self.keep_last_segment = keep_last_segment
        self.min_seg_len = min_seg_len
        self.gamma_s = gamma_s
        self.gamma_e = gamma_e
        self.min_unique_ratio = min_unique_ratio
        self.plus_is_close = plus_is_close
        self.normalize_gripper = normalize_gripper

    def _load_gripper_signal(self, grp: h5py.Group) -> np.ndarray:
        if "actions" not in grp:
            raise KeyError(f"'actions' not found in group. keys={list(grp.keys())}")
        actions = np.asarray(grp["actions"][:], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must be (T,A), got {actions.shape}")
        if self.normalize_gripper:
            actions[:, -1] = (actions[:, -1] - 0.5) * 2.0
        return actions[:, -1]

    def _binarize_gripper(self, gripper: np.ndarray) -> np.ndarray:
        gripper = np.nan_to_num(np.asarray(gripper, dtype=np.float32))
        if gripper.size == 0:
            return np.zeros((0,), dtype=np.int32)
        threshold = 0.5 * (float(np.nanmin(gripper)) + float(np.nanmax(gripper)))
        if self.plus_is_close:
            return (gripper > threshold).astype(np.int32)
        return (gripper < threshold).astype(np.int32)

    def _debounce_changes(self, binary_gripper: np.ndarray, debounce: int) -> List[int]:
        if binary_gripper.size <= 1:
            return []
        if debounce <= 1:
            return (np.where(np.diff(binary_gripper) != 0)[0] + 1).astype(int).tolist()

        current = int(binary_gripper[0])
        changes: List[int] = []
        i = 1
        while i < len(binary_gripper):
            if int(binary_gripper[i]) == current:
                i += 1
                continue
            new_state = int(binary_gripper[i])
            stable = True
            for j in range(i, min(len(binary_gripper), i + debounce)):
                if int(binary_gripper[j]) != new_state:
                    stable = False
                    break
            if stable:
                changes.append(i)
                current = new_state
                i += debounce
            else:
                i += 1
        return changes

    def seg_gripper_state(self, gripper: np.ndarray, t_len: int) -> List[Tuple[int, int]]:
        if len(gripper) != int(t_len):
            raise ValueError(f"gripper length {len(gripper)} != t_len {t_len}")
        binary = self._binarize_gripper(gripper)
        change_idxs = self._debounce_changes(binary, int(self.gripper_debounce))
        boundaries = sorted(set([0] + [int(x) for x in change_idxs] + [int(t_len)]))

        segments: List[Tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end <= start:
                continue
            if i == len(boundaries) - 2 and not self.keep_last_segment and change_idxs:
                continue
            if (end - start) >= int(self.min_seg_len):
                segments.append((start, end))

        if not change_idxs and int(t_len) >= int(self.min_seg_len):
            segments = [(0, int(t_len))]
        return segments

    def resample_on_traj(self, start: int, end: int, num: int) -> np.ndarray:
        if end <= start:
            return np.array([start] * int(num), dtype=np.int64)

        u = np.linspace(0.0, 1.0, num=int(num), dtype=np.float64)
        warped = np.empty_like(u)
        left = u <= 0.5
        warped[left] = 0.5 * np.power(2.0 * u[left], float(self.gamma_s))
        warped[~left] = 1.0 - 0.5 * np.power(2.0 * (1.0 - u[~left]), float(self.gamma_e))

        idxs = np.round(start + (end - start) * warped).astype(np.int64)
        idxs[0] = start
        idxs[-1] = end
        idxs = np.maximum.accumulate(idxs)

        if np.unique(idxs).size < int(float(self.min_unique_ratio) * int(num)):
            idxs = np.round(np.linspace(start, end, num=int(num))).astype(np.int64)
            idxs[0] = start
            idxs[-1] = end
            idxs = np.maximum.accumulate(idxs)
        return idxs


def to_rgb_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr, 0, -1)[..., :3]
        else:
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 2:
                arr = np.stack([squeezed, squeezed, squeezed], axis=-1)
            else:
                raise ValueError(f"Unexpected image shape after squeeze: {arr.shape}")
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def pick_one_demo_id(data_grp: h5py.Group) -> str:
    demo_ids = list(data_grp.keys())
    if not demo_ids:
        raise RuntimeError("No demos found under /data")
    if all(re.fullmatch(r"\d+", demo_id) for demo_id in demo_ids):
        return str(min(int(demo_id) for demo_id in demo_ids))
    return sorted(demo_ids)[0]


def task_name_from_h5(h5_path: Path) -> str:
    name = h5_path.name
    name = re.sub(r"_tracks\.hdf5$", "", name)
    name = re.sub(r"\.hdf5$", "", name)
    name = re.sub(r"\.h5$", "", name)
    return name
