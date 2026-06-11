#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report or delete HDF5 demo groups that are missing required datasets.

This is useful after metric patching, where a few original demo groups may
exist in the HDF5 but have no processed tracks. By default the script only
prints a report. It deletes groups only when both --delete and --yes are set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py


DEFAULT_REQUIRED_KEYS = (
    "frames_rgb",
    "track2d",
    "point_traj",
    "point_traj_base_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and optionally delete demo groups with missing required HDF5 datasets."
    )
    parser.add_argument("--tracks", nargs="*", default=None, help="One or more *_tracks.hdf5 files.")
    parser.add_argument("--tracks_root", nargs="*", default=None, help="Directories searched recursively for *_tracks.hdf5.")
    parser.add_argument("--data_group", default="data", help="Top-level group containing demos.")
    parser.add_argument("--required_keys", nargs="+", default=list(DEFAULT_REQUIRED_KEYS))
    parser.add_argument("--demo_ids", nargs="*", default=None, help="Optional demo ids to inspect.")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--require_nonempty", action="store_true", help="Treat empty datasets as invalid too.")
    parser.add_argument("--delete", action="store_true", help="Delete invalid demo groups.")
    parser.add_argument("--yes", action="store_true", help="Required together with --delete.")
    parser.add_argument("--report_csv", default=None, help="Optional CSV path for the invalid-demo report.")
    return parser.parse_args()


def collect_track_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.tracks or []:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*_tracks.hdf5")))
        else:
            paths.append(p)
    for root in args.tracks_root or []:
        paths.extend(sorted(Path(root).rglob("*_tracks.hdf5")))

    uniq: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise ValueError("No track files found. Pass --tracks or --tracks_root.")
    return uniq


def sort_demo_ids(ids: Iterable[str]) -> list[str]:
    def key_fn(x: str):
        if x.startswith("demo_"):
            tail = x.split("_")[-1]
            if tail.isdigit():
                return (0, int(tail))
        if x.isdigit():
            return (0, int(x))
        return (1, x)

    return sorted(ids, key=key_fn)


def select_demo_ids(
    data_grp: h5py.Group,
    requested: Optional[Sequence[str]],
    max_demos: Optional[int],
) -> list[str]:
    ids = sort_demo_ids(data_grp.keys())
    if requested:
        keep = set(requested)
        ids = [x for x in ids if x in keep]
    if max_demos is not None:
        ids = ids[: int(max_demos)]
    return ids


def missing_reasons(group: h5py.Group, required_keys: Sequence[str], require_nonempty: bool) -> list[str]:
    reasons: list[str] = []
    for key in required_keys:
        if key not in group:
            reasons.append(f"missing:{key}")
            continue
        obj = group[key]
        if require_nonempty and isinstance(obj, h5py.Dataset) and obj.size == 0:
            reasons.append(f"empty:{key}")
    return reasons


def write_report_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["track_file", "demo_id", "reasons", "action"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def inspect_or_delete_file(track_file: Path, args: argparse.Namespace) -> tuple[int, int, list[dict]]:
    mode = "r+" if args.delete else "r"
    rows: list[dict] = []
    inspected = 0
    invalid = 0

    with h5py.File(track_file, mode) as f:
        if args.data_group not in f:
            rows.append(
                {
                    "track_file": str(track_file),
                    "demo_id": "",
                    "reasons": f"missing_data_group:{args.data_group}",
                    "action": "file_skipped",
                }
            )
            return inspected, 1, rows

        data_grp = f[args.data_group]
        demo_ids = select_demo_ids(data_grp, args.demo_ids, args.max_demos)
        to_delete: list[str] = []
        for demo_id in demo_ids:
            inspected += 1
            group = data_grp[demo_id]
            reasons = missing_reasons(group, args.required_keys, args.require_nonempty)
            if not reasons:
                continue

            invalid += 1
            action = "delete" if args.delete else "report"
            rows.append(
                {
                    "track_file": str(track_file),
                    "demo_id": str(demo_id),
                    "reasons": ",".join(reasons),
                    "action": action,
                }
            )
            if args.delete:
                to_delete.append(str(demo_id))

        for demo_id in to_delete:
            del data_grp[demo_id]

    return inspected, invalid, rows


def main() -> None:
    args = parse_args()
    if args.delete and not args.yes:
        raise SystemExit("Refusing to delete without --yes. Re-run with --delete --yes after checking the report.")

    track_files = collect_track_files(args)
    all_rows: list[dict] = []
    total_inspected = 0
    total_invalid = 0

    print(f"[INFO] track_files={len(track_files)}")
    print(f"[INFO] required_keys={list(args.required_keys)}")
    print(f"[INFO] mode={'delete' if args.delete else 'report'}")

    for track_file in track_files:
        inspected, invalid, rows = inspect_or_delete_file(track_file, args)
        total_inspected += inspected
        total_invalid += invalid
        all_rows.extend(rows)
        if invalid:
            print(f"[MISS] {track_file} invalid={invalid}/{inspected}")
            for row in rows[:8]:
                print(f"  {row['demo_id']}: {row['reasons']} ({row['action']})")
            if len(rows) > 8:
                print(f"  ... {len(rows) - 8} more")

    if args.report_csv:
        write_report_csv(all_rows, Path(args.report_csv))
        print(f"[OK] report_csv={args.report_csv}")

    print(
        f"[DONE] inspected_demos={total_inspected} "
        f"invalid_demos={total_invalid} action={'deleted' if args.delete else 'reported'}"
    )


if __name__ == "__main__":
    main()
