#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a side-by-side 3D HTML comparison site for two trajectory keys."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GT-vs-calibrated 3D flow comparison site.")
    parser.add_argument("--flow_root", required=True, help="Input tracks HDF5 or directory.")
    parser.add_argument("--out_dir", required=True, help="Output comparison directory.")
    parser.add_argument("--demo_id", default=None)
    parser.add_argument("--task_type", default="libero", choices=("libero", "maniskill", "real"))
    parser.add_argument("--gt_key", default="point_traj_base_metric")
    parser.add_argument("--calib_key", default="point_traj_metric")
    parser.add_argument("--gt_label", default="GT RGB-D lift")
    parser.add_argument("--calib_label", default="Calibrated SpaTracker metric")
    parser.add_argument("--calib_flow_align_key", default="")
    parser.add_argument("--calib_flow_align_mode", default="none", choices=("none", "start_translation", "endpoint_lerp", "per_frame_translation"))
    parser.add_argument("--calib_flow_align_max_offset", type=float, default=0.40)

    parser.add_argument("--depth_key", default="sim_depths")
    parser.add_argument("--scene_coord_mode", default="metric")
    parser.add_argument("--scene_stride", type=int, default=1)
    parser.add_argument("--max_scene_points", type=int, default=50000)
    parser.add_argument("--scene_point_size", type=float, default=7.0)
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--line_opacity", type=float, default=0.75)
    parser.add_argument("--flow_color_mode", default="rainbow")
    parser.add_argument(
        "--no_filter_flow_visibility",
        action="store_true",
        help="Forward to the HTML exporter to keep tracks even when vis marks them invalid.",
    )
    parser.add_argument(
        "--no_filter_flow_outliers",
        action="store_true",
        help="Forward to the HTML exporter to keep all finite tracks without visual outlier filtering.",
    )
    parser.add_argument("--k_steps", type=int, default=20)
    parser.add_argument("--max_start_frames", type=int, default=1)
    parser.add_argument("--start_stride", type=int, default=9999)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--static_scene", action="store_true")
    parser.add_argument(
        "--page_mode",
        default="all_stages",
        choices=("all_stages", "first_segment"),
        help="Embed the all-stage viewer or only the first exported segment. first_segment is much lighter.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_exporter(args: argparse.Namespace, traj_key: str, out_dir: Path, *, flow_align: bool = False) -> Path:
    script = Path(__file__).resolve().with_name("save_seg_all_starts_to_goal_3d_html.py")
    cmd = [
        sys.executable,
        str(script),
        "--flow_root",
        str(Path(args.flow_root)),
        "--out_dir",
        str(out_dir),
        "--task_type",
        str(args.task_type),
        "--traj_key",
        str(traj_key),
        "--depth_key",
        str(args.depth_key),
        "--scene_coord_mode",
        str(args.scene_coord_mode),
        "--scene_stride",
        str(args.scene_stride),
        "--max_scene_points",
        str(args.max_scene_points),
        "--scene_point_size",
        str(args.scene_point_size),
        "--point_size",
        str(args.point_size),
        "--line_opacity",
        str(args.line_opacity),
        "--flow_color_mode",
        str(args.flow_color_mode),
        "--k_steps",
        str(args.k_steps),
        "--max_start_frames",
        str(args.max_start_frames),
        "--start_stride",
        str(args.start_stride),
        "--fps",
        str(args.fps),
    ]
    if args.demo_id is not None:
        cmd.extend(["--demo_id", str(args.demo_id)])
    if args.static_scene:
        cmd.append("--static_scene")
    if args.no_filter_flow_visibility:
        cmd.append("--no_filter_flow_visibility")
    if args.no_filter_flow_outliers:
        cmd.append("--no_filter_flow_outliers")
    if args.verbose:
        cmd.append("--verbose")
    if flow_align and args.calib_flow_align_key and args.calib_flow_align_mode != "none":
        cmd.extend(
            [
                "--flow_align_key",
                str(args.calib_flow_align_key),
                "--flow_align_mode",
                str(args.calib_flow_align_mode),
                "--flow_align_max_offset",
                str(args.calib_flow_align_max_offset),
            ]
        )
    subprocess.run(cmd, check=True)
    return out_dir / "_all_tasks_3d_index.json"


def find_all_stages(index_path: Path) -> Path:
    manifests = json.loads(index_path.read_text(encoding="utf-8"))
    for manifest in manifests:
        html = manifest.get("all_stages_html_path")
        if html:
            p = Path(html)
            if p.exists():
                return p
    raise RuntimeError(f"No all_stages_html_path found in {index_path}")


def find_first_segment(index_path: Path) -> Path:
    manifests = json.loads(index_path.read_text(encoding="utf-8"))
    for manifest in manifests:
        for segment in manifest.get("segments", []):
            for item in segment.get("start_htmls", []):
                html = item.get("html_path")
                if html:
                    p = Path(html)
                    if p.exists():
                        return p
    raise RuntimeError(f"No segment html_path found in {index_path}")


def rel_link(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_index(out_dir: Path, gt_html: Path, calib_html: Path, args: argparse.Namespace) -> None:
    gt_src = rel_link(gt_html, out_dir)
    calib_src = rel_link(calib_html, out_dir)
    title = "Metric Flow Calibration Comparison"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #172033;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 14px 18px;
      border-bottom: 1px solid #d7dde8;
      background: rgba(255,255,255,0.92);
      display: flex;
      gap: 16px;
      align-items: baseline;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      letter-spacing: 0;
    }}
    .meta {{
      color: #5c677a;
      font-size: 13px;
    }}
    main {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 10px;
      min-height: 0;
    }}
    .pane {{
      min-width: 0;
      min-height: 76vh;
      border: 1px solid #d7dde8;
      background: white;
      display: flex;
      flex-direction: column;
    }}
    .pane-title {{
      padding: 9px 12px;
      font-size: 14px;
      font-weight: 700;
      border-bottom: 1px solid #d7dde8;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}
    .pane-title a {{
      color: #087b68;
      font-weight: 600;
      text-decoration: none;
      font-size: 12px;
    }}
    iframe {{
      width: 100%;
      flex: 1;
      border: 0;
      background: white;
    }}
    @media (max-width: 1100px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .pane {{
        min-height: 68vh;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">left: {args.gt_key} | right: {args.calib_key}</div>
  </header>
  <main>
    <section class="pane">
      <div class="pane-title"><span>{args.gt_label}</span><a href="{gt_src}" target="_blank">open</a></div>
      <iframe src="{gt_src}" title="{args.gt_label}"></iframe>
    </section>
    <section class="pane">
      <div class="pane-title"><span>{args.calib_label}</span><a href="{calib_src}" target="_blank">open</a></div>
      <iframe src="{calib_src}" title="{args.calib_label}"></iframe>
    </section>
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_index = run_exporter(args, args.gt_key, out_dir / "gt_rgbd_lift", flow_align=False)
    calib_index = run_exporter(args, args.calib_key, out_dir / "calibrated_spatracker", flow_align=True)
    if args.page_mode == "first_segment":
        gt_html = find_first_segment(gt_index)
        calib_html = find_first_segment(calib_index)
    else:
        gt_html = find_all_stages(gt_index)
        calib_html = find_all_stages(calib_index)
    write_index(out_dir, gt_html, calib_html, args)
    print(f"[OK] comparison site: {out_dir / 'index.html'}")
    print(f"     gt: {gt_html}")
    print(f"  calib: {calib_html}")


if __name__ == "__main__":
    main()
