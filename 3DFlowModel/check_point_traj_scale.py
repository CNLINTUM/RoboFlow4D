#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check scale stats of point trajectories in LIBERO flow tracks hdf5.

Expected hdf5 layout:
  <task>_demo_tracks.hdf5
    /data
      /0
        point_traj (T, N, D)   # usually D=3
      /1
        point_traj ...
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py


# ---------------------------
# Running stats (vector + scalar)
# ---------------------------

@dataclass
class VecStats:
    dim: int
    count: int = 0
    nan_elems: int = 0
    inf_elems: int = 0
    sum_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sumsq: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    min_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    max_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self):
        self.sum_ = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min_ = np.full(self.dim, np.inf, dtype=np.float64)
        self.max_ = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray):
        """
        x: (..., dim)
        Drop any row that contains non-finite entries (NaN/Inf in any dim).
        """
        if x.size == 0:
            return
        x = np.asarray(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"VecStats.update got shape {x.shape}, expected last dim {self.dim}")

        self.nan_elems += int(np.isnan(x).sum())
        self.inf_elems += int(np.isinf(x).sum())

        mask = np.isfinite(x).all(axis=-1)
        if not np.any(mask):
            return
        v = x[mask].reshape(-1, self.dim).astype(np.float64)

        self.count += v.shape[0]
        self.sum_ += v.sum(axis=0)
        self.sumsq += (v * v).sum(axis=0)
        self.min_ = np.minimum(self.min_, v.min(axis=0))
        self.max_ = np.maximum(self.max_, v.max(axis=0))

    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.full(self.dim, np.nan)
        return self.sum_ / self.count

    def std(self) -> np.ndarray:
        if self.count == 0:
            return np.full(self.dim, np.nan)
        m = self.mean()
        var = self.sumsq / self.count - m * m
        var = np.maximum(var, 0.0)
        return np.sqrt(var)


@dataclass
class ScalarStats:
    count: int = 0
    nan: int = 0
    inf: int = 0
    sum_: float = 0.0
    sumsq: float = 0.0
    min_: float = math.inf
    max_: float = -math.inf

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        x = np.asarray(x)

        self.nan += int(np.isnan(x).sum())
        self.inf += int(np.isinf(x).sum())

        mask = np.isfinite(x)
        if not np.any(mask):
            return
        v = x[mask].reshape(-1).astype(np.float64)

        self.count += v.size
        self.sum_ += float(v.sum())
        self.sumsq += float((v * v).sum())
        self.min_ = min(self.min_, float(v.min()))
        self.max_ = max(self.max_, float(v.max()))

    def mean(self) -> float:
        return self.sum_ / self.count if self.count > 0 else float("nan")

    def std(self) -> float:
        if self.count == 0:
            return float("nan")
        m = self.mean()
        var = self.sumsq / self.count - m * m
        var = max(var, 0.0)
        return math.sqrt(var)


# ---------------------------
# Helpers
# ---------------------------

def infer_suite_and_task(base_dir: Path, h5_path: Path) -> Tuple[str, str]:
    rel = h5_path.relative_to(base_dir)
    parts = rel.parts
    suite = parts[0] if len(parts) >= 1 else "UNKNOWN_SUITE"
    task = parts[1] if len(parts) >= 2 else "UNKNOWN_TASK"
    return suite, task


def safe_open_h5(path: Path) -> Optional[h5py.File]:
    try:
        return h5py.File(path, "r")
    except (OSError, BlockingIOError) as e:
        print(f"[WARN] Cannot open {path}: {e}")
        return None


def choose_traj_key(grp: h5py.Group, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in grp:
            return k
    return None


def iter_time_chunks(ds: h5py.Dataset, chunk_T: int):
    T = ds.shape[0]
    for t0 in range(0, T, chunk_T):
        t1 = min(T, t0 + chunk_T)
        yield t0, t1, ds[t0:t1]


def format_arr(a: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(a).reshape(-1)]


# ---------------------------
# Per-demo result
# ---------------------------

@dataclass
class DemoResult:
    suite: str
    task: str
    h5_file: str
    demo_id: str
    dataset_path: str
    T: int
    N: int
    D: int

    valid_rows: int
    nan_elems: int
    inf_elems: int

    mean: List[float]
    std: List[float]
    min: List[float]
    max: List[float]

    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float

    delta_norm_mean: float
    delta_norm_std: float
    delta_norm_min: float
    delta_norm_max: float


def process_demo(
    suite: str,
    task: str,
    h5_path: Path,
    demo_id: str,
    grp: h5py.Group,
    traj_keys: List[str],
    chunk_T: int,
) -> Optional[Tuple[DemoResult, VecStats, ScalarStats, ScalarStats]]:
    key = choose_traj_key(grp, traj_keys)
    if key is None:
        return None

    ds = grp[key]
    if ds.ndim != 3:
        print(f"[WARN] {h5_path} /data/{demo_id}/{key} shape={ds.shape}, expected (T,N,D). Skip.")
        return None

    T, N, D = ds.shape
    vec = VecStats(dim=D)
    norm = ScalarStats()
    delta_norm = ScalarStats()

    prev_last_frame = None

    for t0, t1, chunk in iter_time_chunks(ds, chunk_T=chunk_T):
        # chunk: (dt, N, D)
        chunk = np.asarray(chunk)
        flat = chunk.reshape(-1, D)
        vec.update(flat)

        norms = np.linalg.norm(flat.astype(np.float64), axis=-1)
        norm.update(norms)

        # delta norm within chunk
        if (t1 - t0) >= 2:
            d = np.diff(chunk.astype(np.float64), axis=0)  # (dt-1, N, D)
            dflat = d.reshape(-1, D)
            dnorms = np.linalg.norm(dflat, axis=-1)
            delta_norm.update(dnorms)

        # delta norm across chunk boundary
        if prev_last_frame is not None:
            first = chunk[0].astype(np.float64)           # (N, D)
            d0 = (first - prev_last_frame).reshape(-1, D)
            d0_norm = np.linalg.norm(d0, axis=-1)
            delta_norm.update(d0_norm)

        prev_last_frame = chunk[-1].astype(np.float64)

    res = DemoResult(
        suite=suite,
        task=task,
        h5_file=str(h5_path),
        demo_id=str(demo_id),
        dataset_path=f"/data/{demo_id}/{key}",
        T=int(T),
        N=int(N),
        D=int(D),

        valid_rows=int(vec.count),
        nan_elems=int(vec.nan_elems),
        inf_elems=int(vec.inf_elems),

        mean=format_arr(vec.mean()),
        std=format_arr(vec.std()),
        min=format_arr(vec.min_),
        max=format_arr(vec.max_),

        norm_mean=float(norm.mean()),
        norm_std=float(norm.std()),
        norm_min=float(norm.min_ if norm.count > 0 else float("nan")),
        norm_max=float(norm.max_ if norm.count > 0 else float("nan")),

        delta_norm_mean=float(delta_norm.mean()),
        delta_norm_std=float(delta_norm.std()),
        delta_norm_min=float(delta_norm.min_ if delta_norm.count > 0 else float("nan")),
        delta_norm_max=float(delta_norm.max_ if delta_norm.count > 0 else float("nan")),
    )
    return res, vec, norm, delta_norm


# ---------------------------
# Suite aggregation
# ---------------------------

@dataclass
class SuiteAgg:
    suite: str
    demos: int = 0
    files: int = 0
    missing_demos: int = 0

    vec: Optional[VecStats] = None
    norm: ScalarStats = field(default_factory=ScalarStats)         # FIX
    delta_norm: ScalarStats = field(default_factory=ScalarStats)   # FIX

    def ensure_dim(self, D: int):
        if self.vec is None:
            self.vec = VecStats(dim=D)
        elif self.vec.dim != D:
            raise ValueError(f"Suite {self.suite}: inconsistent D: {self.vec.dim} vs {D}")

    def update(self, vec: VecStats, norm: ScalarStats, delta_norm: ScalarStats):
        self.ensure_dim(vec.dim)
        self.demos += 1

        # merge vec
        self.vec.count += vec.count
        self.vec.nan_elems += vec.nan_elems
        self.vec.inf_elems += vec.inf_elems
        self.vec.sum_ += vec.sum_
        self.vec.sumsq += vec.sumsq
        self.vec.min_ = np.minimum(self.vec.min_, vec.min_)
        self.vec.max_ = np.maximum(self.vec.max_, vec.max_)

        # merge scalar
        self._merge_scalar(self.norm, norm)
        self._merge_scalar(self.delta_norm, delta_norm)

    @staticmethod
    def _merge_scalar(dst: ScalarStats, src: ScalarStats):
        dst.nan += src.nan
        dst.inf += src.inf
        dst.count += src.count
        dst.sum_ += src.sum_
        dst.sumsq += src.sumsq
        dst.min_ = min(dst.min_, src.min_)
        dst.max_ = max(dst.max_, src.max_)

    def to_row(self) -> Dict:
        assert self.vec is not None
        return dict(
            suite=self.suite,
            files=self.files,
            demos=self.demos,
            missing_demos=self.missing_demos,
            D=self.vec.dim,
            valid_rows=self.vec.count,
            nan_elems=self.vec.nan_elems,
            inf_elems=self.vec.inf_elems,
            mean=format_arr(self.vec.mean()),
            std=format_arr(self.vec.std()),
            min=format_arr(self.vec.min_),
            max=format_arr(self.vec.max_),
            norm_mean=float(self.norm.mean()),
            norm_std=float(self.norm.std()),
            norm_min=float(self.norm.min_ if self.norm.count > 0 else float("nan")),
            norm_max=float(self.norm.max_ if self.norm.count > 0 else float("nan")),
            delta_norm_mean=float(self.delta_norm.mean()),
            delta_norm_std=float(self.delta_norm.std()),
            delta_norm_min=float(self.delta_norm.min_ if self.delta_norm.count > 0 else float("nan")),
            delta_norm_max=float(self.delta_norm.max_ if self.delta_norm.count > 0 else float("nan")),
        )


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*_tracks.hdf5")
    ap.add_argument("--traj_keys", type=str, nargs="+", default=["point_traj", "pint_traj"])
    ap.add_argument("--chunk_T", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="./point_traj_scale_report")
    ap.add_argument("--only_suites", type=str, nargs="*", default=None)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(base_dir.rglob(args.pattern))
    if args.only_suites:
        only = set(args.only_suites)
        h5_files = [p for p in h5_files if infer_suite_and_task(base_dir, p)[0] in only]

    print(f"[INFO] Found {len(h5_files)} hdf5 files matching {args.pattern}")

    per_demo: List[DemoResult] = []
    suite_aggs: Dict[str, SuiteAgg] = {}

    for h5_path in h5_files:
        suite, task = infer_suite_and_task(base_dir, h5_path)
        agg = suite_aggs.setdefault(suite, SuiteAgg(suite=suite))
        agg.files += 1

        f = safe_open_h5(h5_path)
        if f is None:
            continue

        with f:
            if "data" not in f:
                print(f"[WARN] {h5_path} has no /data group. Skipping.")
                continue

            data_grp = f["data"]
            for demo_id, grp in data_grp.items():
                out = process_demo(
                    suite=suite,
                    task=task,
                    h5_path=h5_path,
                    demo_id=str(demo_id),
                    grp=grp,
                    traj_keys=args.traj_keys,
                    chunk_T=args.chunk_T,
                )
                if out is None:
                    agg.missing_demos += 1
                    continue

                demo_res, vec, norm, delta_norm = out
                per_demo.append(demo_res)
                try:
                    agg.update(vec, norm, delta_norm)
                except ValueError as e:
                    print(f"[WARN] {e}. Skip aggregating this demo.")

    # Write per-demo CSV
    per_demo_csv = out_dir / "per_demo_point_traj_stats.csv"
    if per_demo:
        D0 = per_demo[0].D
        dim_fields = []
        for name in ["mean", "std", "min", "max"]:
            for d in range(D0):
                dim_fields.append(f"{name}_d{d}")

        fieldnames = [
            "suite", "task", "h5_file", "demo_id", "dataset_path", "T", "N", "D",
            "valid_rows", "nan_elems", "inf_elems",
            *dim_fields,
            "norm_mean", "norm_std", "norm_min", "norm_max",
            "delta_norm_mean", "delta_norm_std", "delta_norm_min", "delta_norm_max",
        ]

        with per_demo_csv.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for r in per_demo:
                row = {
                    "suite": r.suite,
                    "task": r.task,
                    "h5_file": r.h5_file,
                    "demo_id": r.demo_id,
                    "dataset_path": r.dataset_path,
                    "T": r.T,
                    "N": r.N,
                    "D": r.D,
                    "valid_rows": r.valid_rows,
                    "nan_elems": r.nan_elems,
                    "inf_elems": r.inf_elems,
                    "norm_mean": r.norm_mean,
                    "norm_std": r.norm_std,
                    "norm_min": r.norm_min,
                    "norm_max": r.norm_max,
                    "delta_norm_mean": r.delta_norm_mean,
                    "delta_norm_std": r.delta_norm_std,
                    "delta_norm_min": r.delta_norm_min,
                    "delta_norm_max": r.delta_norm_max,
                }
                for d in range(r.D):
                    row[f"mean_d{d}"] = r.mean[d] if d < len(r.mean) else float("nan")
                    row[f"std_d{d}"]  = r.std[d] if d < len(r.std) else float("nan")
                    row[f"min_d{d}"]  = r.min[d] if d < len(r.min) else float("nan")
                    row[f"max_d{d}"]  = r.max[d] if d < len(r.max) else float("nan")
                w.writerow(row)

    # Write per-suite CSV
    per_suite_csv = out_dir / "per_suite_point_traj_stats.csv"
    with per_suite_csv.open("w", newline="") as fp:
        fieldnames = [
            "suite", "files", "demos", "missing_demos", "D",
            "valid_rows", "nan_elems", "inf_elems",
            "mean", "std", "min", "max",
            "norm_mean", "norm_std", "norm_min", "norm_max",
            "delta_norm_mean", "delta_norm_std", "delta_norm_min", "delta_norm_max",
        ]
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for suite, agg in sorted(suite_aggs.items()):
            if agg.vec is None:
                w.writerow({
                    "suite": suite, "files": agg.files, "demos": agg.demos, "missing_demos": agg.missing_demos,
                    "D": "", "valid_rows": 0, "nan_elems": 0, "inf_elems": 0,
                    "mean": "", "std": "", "min": "", "max": "",
                    "norm_mean": "", "norm_std": "", "norm_min": "", "norm_max": "",
                    "delta_norm_mean": "", "delta_norm_std": "", "delta_norm_min": "", "delta_norm_max": "",
                })
                continue
            row = agg.to_row()
            for k in ["mean", "std", "min", "max"]:
                row[k] = json.dumps(row[k])
            w.writerow(row)

    summary = {
        "base_dir": str(base_dir),
        "pattern": args.pattern,
        "traj_keys": args.traj_keys,
        "num_files": len(h5_files),
        "num_demos_with_traj": len(per_demo),
        "suites": {k: {"files": v.files, "demos": v.demos, "missing_demos": v.missing_demos} for k, v in suite_aggs.items()},
        "outputs": {
            "per_demo_csv": str(per_demo_csv),
            "per_suite_csv": str(per_suite_csv),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(f"  per-demo : {per_demo_csv}")
    print(f"  per-suite: {per_suite_csv}")
    print(f"  summary  : {out_dir / 'summary.json'}")

    print("\n[PREVIEW] per-suite:")
    for suite, agg in sorted(suite_aggs.items()):
        if agg.vec is None:
            print(f"  - {suite}: (no valid demos) files={agg.files} missing_demos={agg.missing_demos}")
        else:
            m = agg.vec.mean()
            mx = agg.vec.max_
            mn = agg.vec.min_
            print(f"  - {suite}: files={agg.files} demos={agg.demos} "
                  f"mean={np.array2string(m, precision=4)} "
                  f"min={np.array2string(mn, precision=4)} "
                  f"max={np.array2string(mx, precision=4)} "
                  f"norm_max={agg.norm.max_:.6g} delta_norm_max={agg.delta_norm.max_:.6g}")


if __name__ == "__main__":
    main()


#   --out_dir /path/to/Data/LIBERO/Flow_training_all_refined/_point_traj_scale_report
