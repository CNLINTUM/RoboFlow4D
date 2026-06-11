#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

def _pad_or_trim(arr: np.ndarray, T: int) -> np.ndarray:
    """Convert arr from [T0,...] to [T,...] by padding with the last frame or truncating."""
    T0 = arr.shape[0]
    if T0 == T:
        return arr
    if T0 > T:
        return arr[:T]
    if T0 == 0:
        raise RuntimeError("Empty sequence")
    pad = np.repeat(arr[-1:], repeats=(T - T0), axis=0)
    return np.concatenate([arr, pad], axis=0)

def _get_T_full(fsrc: h5py.File, demo_id: str, image_key: str) -> int:
    # Use the source observation length as the full sequence length.
    ds = fsrc[f"data/{demo_id}/obs/{image_key}"]
    return int(ds.shape[0])

def patch_one_tracks(tracks_h5: Path,
                     image_key: str,
                     wrist_key: str,
                     overwrite: bool,
                     flip_hw: bool):
    wrote = 0
    with h5py.File(tracks_h5, "a") as fout:
        if "data" not in fout:
            return 0

        src_path = fout.attrs.get("source_hdf5", None)
        if not src_path:
            raise RuntimeError(f"[{tracks_h5}] missing attr source_hdf5")
        src_path = Path(str(src_path))
        if not src_path.exists():
            raise FileNotFoundError(f"[{tracks_h5}] source_hdf5 not found: {src_path}")

        with h5py.File(src_path, "r") as fsrc:
            for demo_id in fout["data"].keys():
                out_grp = fout[f"data/{demo_id}"]

                # Define full length from the source obs/image_key sequence.
                T_full = _get_T_full(fsrc, demo_id, image_key)

                # ---- robot_states (ee + gripper) ----
                ee = fsrc[f"data/{demo_id}/obs/ee_states"][:]          # [T_full, ...]
                gr = fsrc[f"data/{demo_id}/obs/gripper_states"][:]     # [T_full, ...]
                ee = _pad_or_trim(ee, T_full)
                gr = _pad_or_trim(gr, T_full)
                robot_states = np.concatenate([ee, gr], axis=-1)

                if "robot_states" in out_grp:
                    if overwrite: del out_grp["robot_states"]
                    else: pass
                if overwrite or ("robot_states" not in out_grp):
                    out_grp.create_dataset("robot_states", data=robot_states, compression="gzip")

                # ---- actions ----
                if f"data/{demo_id}/actions" in fsrc:
                    act = fsrc[f"data/{demo_id}/actions"][:]
                    # Some environments store T_full or T_full-1 actions; align to T_full.
                    act = _pad_or_trim(act, T_full)
                    if "actions" in out_grp:
                        if overwrite: del out_grp["actions"]
                    if overwrite or ("actions" not in out_grp):
                        out_grp.create_dataset("actions", data=act, compression="gzip")

                # ---- dones ----
                if f"data/{demo_id}/dones" in fsrc:
                    dn = fsrc[f"data/{demo_id}/dones"][:]
                    dn = _pad_or_trim(dn, T_full).astype(np.int8)
                    if "dones" in out_grp:
                        if overwrite: del out_grp["dones"]
                    if overwrite or ("dones" not in out_grp):
                        out_grp.create_dataset("dones", data=dn, compression="gzip")

                # ---- wrist_frames ----
                src_wrist_path = f"data/{demo_id}/obs/{wrist_key}"
                if src_wrist_path in fsrc:
                    wrist = fsrc[src_wrist_path][:]  # [T_full, H, W, C] uint8
                    wrist = _pad_or_trim(wrist, T_full)
                    if flip_hw:
                        wrist = wrist[:, ::-1, ::-1, :]
                    wrist = np.transpose(wrist, (0, 3, 1, 2))  # [T_full, C, H, W]

                    if "wrist_frames" in out_grp:
                        if overwrite: del out_grp["wrist_frames"]
                    if overwrite or ("wrist_frames" not in out_grp):
                        C, H, W = wrist.shape[1:]
                        out_grp.create_dataset(
                            "wrist_frames", data=wrist, dtype=np.uint8,
                            compression="gzip", chunks=(1, C, H, W)
                        )

                wrote += 1

        fout.attrs["patched_full_robot"] = True
        fout.attrs["patched_image_key_for_Tfull"] = image_key
        fout.attrs["patched_wrist_key"] = wrist_key
    return wrote

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_root", required=True)
    ap.add_argument("--image_key", default="agentview_rgb",
                    help="Observation key in the source HDF5 used to define T_full.")
    ap.add_argument("--wrist_key", default="eye_in_hand_rgb")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--flip_hw", action="store_true",
                    help="Apply [:, ::-1, ::-1, :] to wrist images when needed.")
    args = ap.parse_args()

    tracks_root = Path(args.tracks_root)
    tracks_files = sorted(tracks_root.glob("*/*_tracks.hdf5"))
    print(f"Found {len(tracks_files)} tracks files under {tracks_root}")

    total = 0
    for tf in tqdm(tracks_files, desc="Patching full robot/actions/dones/wrist"):
        total += patch_one_tracks(tf, args.image_key, args.wrist_key, args.overwrite, args.flip_hw)

    print(f"Done. patched demos={total}")

if __name__ == "__main__":
    main()
