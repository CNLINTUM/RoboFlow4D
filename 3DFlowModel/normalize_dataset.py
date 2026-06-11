#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse, json
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

def infer_suit(h5_path: Path) -> str:
    # Expected layout: flow_root/<suite>/<task>.../*_tracks.hdf5
    return h5_path.parent.parent.name

class Running:
    def __init__(self, eps=1e-6):
        self.sum = np.zeros(3, np.float64)
        self.sum2 = np.zeros(3, np.float64)
        self.count = 0.0
        self.eps = eps

    def update(self, x_tn3: np.ndarray, w_tn: np.ndarray | None):
        x = x_tn3.astype(np.float64, copy=False)
        if w_tn is None:
            w = np.ones(x.shape[:2], np.float64)
        else:
            w = w_tn.astype(np.float64, copy=False)
        ww = w[..., None]
        self.sum  += (x * ww).sum(axis=(0,1))
        self.sum2 += ((x*x) * ww).sum(axis=(0,1))
        self.count += w.sum()

    def finalize(self):
        c = max(self.count, self.eps)
        mean = self.sum / c
        var = self.sum2 / c - mean * mean
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32), float(self.count)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--use_vis", action="store_true", help="use vis mask if available")
    ap.add_argument("--max_h5", type=int, default=-1)
    args = ap.parse_args()

    root = Path(args.flow_root)
    h5_files = sorted(root.rglob("*_tracks.hdf5"))
    if args.max_h5 > 0:
        h5_files = h5_files[:args.max_h5]

    moms = {}

    for hp in tqdm(h5_files, desc="H5"):
        suit = infer_suit(hp)
        moms.setdefault(suit, Running())

        try:
            f = h5py.File(hp, "r")
        except Exception as e:
            print(f"[WARN] open fail {hp}: {e}")
            continue

        with f:
            if "data" not in f: 
                continue
            for demo_id, grp in f["data"].items():
                if grp.attrs.get("has_tracks", True) is False: 
                    continue
                if "point_traj" not in grp:
                    continue

                P = np.asarray(grp["point_traj"][:], dtype=np.float32)  # (T,N,3)

                W = None
                if args.use_vis and ("vis" in grp):
                    V = np.asarray(grp["vis"][:], dtype=np.float32)       # (T,N)
                    V = np.squeeze(V)
                    W = (V > 0.5).astype(np.float32)

                moms[suit].update(P, W)

    out = {"version": 1, "mode": "all_frames_all_points", "flow_root": str(root), "suits": {}}
    for s in sorted(moms.keys()):
        mean, std, cnt = moms[s].finalize()
        out["suits"][s] = {"mean_xyz": mean.tolist(), "std_xyz": std.tolist(), "count_visible_tn": cnt}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as wf:
        json.dump(out, wf, indent=2)
    print("[OK]", args.out)

if __name__ == "__main__":
    main()

# python normalize_dataset.py \
#   --flow_root /path/to/Data/LIBERO/Flow_training_all_refined \
#   --out /path/to/Data/LIBERO/norm_stats_allpoints.json


# python normalize_dataset.py \
#   --flow_root /path/to/Data/LIBERO/Flow_training_all_refined \
#   --out /path/to/Data/LIBERO/norm_stats_vis.json --use_vis


# MANISKILL
# python normalize_dataset.py \
#   --flow_root /path/to/Data/Maniskill \
#   --out /path/to/Data/Maniskill/norm_stats_allpoints.json
