#!/usr/bin/env python3
"""Scan processed HDF5 files for missing or empty demo datasets."""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

REQUIRED_KEYS = [
    'actions', 'dones', 'frames_rgb', 'grid_points_xy', 'intrs2',
    'p0_uv', 'point_traj', 'query_xy_t0', 'robot_states', 'track2d',
    'vggt_hidden', 'vis', 'wrist_frames'
]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root directory searched recursively for HDF5 files.")
    parser.add_argument("--limit_files", type=int, default=None, help="Scan only the first N HDF5 files.")
    parser.add_argument("--no_save_report", action="store_true", help="Print results without writing CSV/JSON reports.")
    parser.add_argument("--out_csv", type=Path, default=None, help="Optional CSV report path.")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


# -------- helpers --------
def iter_hdf5_files(root: Path):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".hdf5") or fn.endswith(".h5"):
                yield Path(dp) / fn

def is_empty_h5obj(obj) -> bool:
    # group: empty if no members
    if isinstance(obj, h5py.Group):
        return len(obj.keys()) == 0

    # dataset: empty if any dim==0 or size==0 (scalar () treated as non-empty)
    if isinstance(obj, h5py.Dataset):
        try:
            shape = obj.shape
            if shape is None:
                return True
            if len(shape) == 0:  # scalar
                return False
            if any(d == 0 for d in shape):
                return True
            return obj.size == 0
        except Exception:
            return True

    return True

def count_csv_items(s):
    if not isinstance(s, str) or s.strip() == "":
        return 0
    return len([x for x in s.split(",") if x.strip() != ""])

def scan_all_hdf5(root: Path, required_keys, limit_files=None):
    rows = []
    files = sorted(list(iter_hdf5_files(root)))
    if limit_files is not None:
        files = files[:limit_files]

    for fp in tqdm(files, desc="Scanning HDF5"):
        try:
            with h5py.File(fp, "r") as f:
                if "data" not in f:
                    rows.append({
                        "file": str(fp),
                        "demo_id": None,
                        "status": "BAD_FILE",
                        "error": "missing_data_group",
                        "missing_keys": "data",
                        "empty_keys": "",
                    })
                    continue

                demo_ids = list(f["data"].keys())
                if len(demo_ids) == 0:
                    rows.append({
                        "file": str(fp),
                        "demo_id": None,
                        "status": "BAD_FILE",
                        "error": "no_demos_under_data",
                        "missing_keys": "",
                        "empty_keys": "",
                    })
                    continue

                for demo_id in demo_ids:
                    grp = f[f"data/{demo_id}"]
                    missing, empty = [], []

                    for k in required_keys:
                        if k not in grp:
                            missing.append(k)
                        else:
                            if is_empty_h5obj(grp[k]):
                                empty.append(k)

                    status = "OK" if (not missing and not empty) else "BAD_DEMO"
                    rows.append({
                        "file": str(fp),
                        "demo_id": demo_id,
                        "status": status,
                        "error": "",
                        "missing_keys": ",".join(missing),
                        "empty_keys": ",".join(empty),
                    })

        except Exception as e:
            rows.append({
                "file": str(fp),
                "demo_id": None,
                "status": "BAD_FILE",
                "error": repr(e),
                "missing_keys": "",
                "empty_keys": "",
            })

    return pd.DataFrame(rows)

def main():
    args = parse_args()
    root = args.root
    df = scan_all_hdf5(root, REQUIRED_KEYS, limit_files=args.limit_files)

    print("Total rows (file-demo records):", len(df))
    print("\nStatus counts:")
    print(df["status"].value_counts())

    bad = df[df["status"] != "OK"]
    print("\nBad rows:", len(bad))
    print(bad.head(30))

    file_summary = (
        df.groupby("file")
          .agg(
              total_demos=("demo_id", lambda x: x.notna().sum()),
              bad_demos=("status", lambda s: (s == "BAD_DEMO").sum()),
              bad_file=("status", lambda s: (s == "BAD_FILE").any()),
          )
          .reset_index()
          .sort_values(["bad_file", "bad_demos", "total_demos"], ascending=[False, False, False])
    )
    print("\nFile-level summary (top 50):")
    print(file_summary.head(50))

    df["n_missing"] = df["missing_keys"].apply(count_csv_items)
    df["n_empty"] = df["empty_keys"].apply(count_csv_items)

    print("\nTop 30 demos by missing keys:")
    print(df.sort_values("n_missing", ascending=False).head(30)[
        ["file", "demo_id", "status", "n_missing", "missing_keys", "n_empty", "empty_keys", "error"]
    ])

    print("\nTop 30 demos by empty keys:")
    print(df.sort_values("n_empty", ascending=False).head(30)[
        ["file", "demo_id", "status", "n_empty", "empty_keys", "n_missing", "missing_keys", "error"]
    ])

    if not args.no_save_report:
        out_csv = args.out_csv or (root / "scan_report.csv")
        out_json = args.out_json or (root / "scan_report.json")
        df.to_csv(out_csv, index=False)
        df.to_json(out_json, orient="records", indent=2)
        print("\nSaved:", out_csv)
        print("Saved:", out_json)


if __name__ == "__main__":
    main()
