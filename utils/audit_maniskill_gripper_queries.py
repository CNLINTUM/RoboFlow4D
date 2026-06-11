#!/usr/bin/env python3
"""Audit whether ManiSkill query points lie on the saved gripper masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="ManiSkill *_tracks.hdf5 files or directories.")
    p.add_argument("--query_key", default="query_xy_t0", choices=("query_xy_t0", "grid_points_xy", "p0_uv"))
    p.add_argument("--threshold", type=float, default=0.85, help="Minimum best mask coverage to mark a demo as good.")
    p.add_argument("--report_csv", default=None)
    return p.parse_args()


def find_track_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.rglob("*_tracks.hdf5")))
        elif p.is_file():
            files.append(p)
    return sorted(dict.fromkeys(files))


def sort_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        tail = str(x).split("_")[-1]
        return (0, int(tail)) if tail.isdigit() else (1, str(x))

    return sorted([str(x) for x in ids], key=key_fn)


def load_mask_info(mask_dir: Path) -> tuple[np.ndarray | None, str, float | None]:
    mask_path = mask_dir / "mask_binary.png"
    if not mask_path.exists():
        return None, "", None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    label = ""
    logit = None
    meta_path = mask_dir / "mask.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            objects = [x for x in meta if x.get("value") == 1]
            if objects:
                label = str(objects[0].get("label", ""))
                raw_logit = objects[0].get("logit", None)
                logit = None if raw_logit is None else float(raw_logit)
        except Exception:
            pass
    return mask, label, logit


def candidate_mask_dirs(track_file: Path, demo_id: str) -> list[Path]:
    stem = track_file.stem[:-len("_tracks")] if track_file.stem.endswith("_tracks") else track_file.stem
    suffix = str(demo_id).split("_")[-1]
    return [
        track_file.parent / f"{stem}_{demo_id}",
        track_file.parent / f"{stem}_{suffix}",
        track_file.parent / f"{stem}_demo_{demo_id}",
        track_file.parent / str(demo_id),
    ]


def scale_query(query: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    xy = np.asarray(query, dtype=np.float32)[..., :2].copy()
    if mode == "518":
        xy[:, 0] *= float(width) / 518.0
        xy[:, 1] *= float(height) / 518.0
    elif mode == "normalized":
        xy[:, 0] *= max(width - 1, 1)
        xy[:, 1] *= max(height - 1, 1)
    elif mode == "auto":
        finite = xy[np.isfinite(xy)]
        if finite.size:
            mn = float(np.nanmin(finite))
            mx = float(np.nanmax(finite))
            if mx > max(height, width) * 1.2:
                xy = scale_query(query, height, width, "518")
            elif mx <= 2.0 and mn >= -0.5:
                xy = scale_query(query, height, width, "normalized")
    return xy


def mask_ratio(mask: np.ndarray, query: np.ndarray, mode: str) -> float:
    height, width = mask.shape
    xy = scale_query(query, height, width, mode)
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return 0.0
    xi = np.clip(np.rint(xy[finite, 0]).astype(np.int64), 0, width - 1)
    yi = np.clip(np.rint(xy[finite, 1]).astype(np.int64), 0, height - 1)
    return float((mask[yi, xi] > 0).mean())


def inspect_file(track_file: Path, threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with h5py.File(track_file, "r") as f:
        source_hdf5 = str(f.attrs.get("source_hdf5", ""))
        if "data" not in f:
            return rows
        for demo_id in sort_ids(f["data"].keys()):
            group = f[f"data/{demo_id}"]
            row: dict[str, object] = {
                "track_file": str(track_file),
                "demo_id": demo_id,
                "source_hdf5": source_hdf5,
                "has_metric": "point_traj_base_metric" in group,
                "has_world_metric": "point_traj_world_metric" in group,
                "has_tcp_rigid_metric": "point_traj_tcp_rigid_metric" in group,
                "status": "missing_query",
            }
            if "query_xy_t0" not in group:
                rows.append(row)
                continue
            query = np.asarray(group["query_xy_t0"], dtype=np.float32)
            if query.ndim != 2 or query.shape[-1] < 2:
                row["status"] = f"bad_query_shape:{query.shape}"
                rows.append(row)
                continue

            mask_dir = next((p for p in candidate_mask_dirs(track_file, demo_id) if (p / "mask_binary.png").exists()), None)
            if mask_dir is None:
                row["status"] = "missing_mask"
                rows.append(row)
                continue
            mask, label, logit = load_mask_info(mask_dir)
            if mask is None:
                row["status"] = "bad_mask"
                rows.append(row)
                continue

            ratios = {mode: mask_ratio(mask, query, mode) for mode in ("pixels", "518", "normalized", "auto")}
            best_mode = max(("pixels", "518", "normalized"), key=lambda x: ratios[x])
            best_ratio = ratios[best_mode]
            finite = query[..., :2][np.isfinite(query[..., :2])]
            row.update(
                {
                    "status": "ok" if best_ratio >= threshold else "low_coverage",
                    "mask_dir": str(mask_dir),
                    "mask_label": label,
                    "mask_logit": logit,
                    "num_query": int(query.shape[0]),
                    "query_min": float(np.nanmin(finite)) if finite.size else np.nan,
                    "query_max": float(np.nanmax(finite)) if finite.size else np.nan,
                    "ratio_pixels": ratios["pixels"],
                    "ratio_518": ratios["518"],
                    "ratio_normalized": ratios["normalized"],
                    "ratio_auto": ratios["auto"],
                    "best_mode": best_mode,
                    "best_ratio": best_ratio,
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    files = find_track_files(args.inputs)
    rows: list[dict[str, object]] = []
    for track_file in files:
        rows.extend(inspect_file(track_file, args.threshold))

    total = len(rows)
    good = sum(row.get("status") == "ok" for row in rows)
    low = sum(row.get("status") == "low_coverage" for row in rows)
    missing = total - good - low
    print(f"[DONE] files={len(files)} demos={total} good={good} low_coverage={low} missing_or_bad={missing}")

    by_file: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_file.setdefault(str(row["track_file"]), []).append(row)
    for track_file, file_rows in by_file.items():
        vals = [float(r["best_ratio"]) for r in file_rows if "best_ratio" in r]
        if vals:
            arr = np.asarray(vals)
            print(
                f"  {track_file}: checked={len(vals)}/{len(file_rows)} "
                f"best_ratio median/min={np.median(arr):.3f}/{np.min(arr):.3f} "
                f"low<{args.threshold}={(arr < args.threshold).sum()}"
            )
        else:
            print(f"  {track_file}: checked=0/{len(file_rows)}")

    if args.report_csv:
        keys = sorted({key for row in rows for key in row.keys()})
        out = Path(args.report_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] report_csv={out}")


if __name__ == "__main__":
    main()
