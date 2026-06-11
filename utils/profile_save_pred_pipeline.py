#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile predict_flow_hdf5.py with whole-pipeline FLOPs and GPU memory polling."
    )
    parser.add_argument("--gpu-index", type=int, required=True, help="Physical GPU index to monitor with nvidia-smi.")
    parser.add_argument("--metrics-path", type=str, required=True, help="JSON path for profiling summary.")
    parser.add_argument(
        "--save-pred-script",
        type=str,
        default=str(REPO_ROOT / "3DFlowModel/predict_flow_hdf5.py"),
        help="Path to predict_flow_hdf5.py.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi polling calls.",
    )
    parser.add_argument(
        "--disable-visuals",
        action="store_true",
        help="Monkeypatch draw_tracks to a no-op so profiling focuses on compute rather than matplotlib overhead.",
    )
    parser.add_argument(
        "--disable_cfg",
        action="store_true",
        help="Force guidance_scale=1.0 in forwarded save_pred args so profiling avoids CFG batch doubling.",
    )
    parser.add_argument(
        "save_pred_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to predict_flow_hdf5.py. Prefix with -- to terminate parsing.",
    )
    args = parser.parse_args()
    if args.save_pred_args and args.save_pred_args[0] == "--":
        args.save_pred_args = args.save_pred_args[1:]
    return args


def maybe_disable_cfg(save_pred_args: list[str], disable_cfg: bool) -> list[str]:
    if not disable_cfg:
        return list(save_pred_args)

    rewritten = list(save_pred_args)
    if "--guidance_scale" in rewritten:
        idx = rewritten.index("--guidance_scale")
        if idx + 1 < len(rewritten):
            rewritten[idx + 1] = "1.0"
        else:
            rewritten.append("1.0")
    else:
        rewritten.extend(["--guidance_scale", "1.0"])
    return rewritten


def query_gpu_memory_mb(gpu_index: int) -> tuple[float | None, float | None]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True).strip()
    if not output:
        return None, None
    _index, used_mb, total_mb = [part.strip() for part in output.split(",")]
    return float(used_mb), float(total_mb)


def sum_profile_flops(profile_obj: torch.profiler.profile) -> float:
    total_flops = 0.0
    for event in profile_obj.key_averages():
        total_flops += float(getattr(event, "flops", 0) or 0)
    return total_flops


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    peak_used_mb = 0.0
    total_mb = None

    def poll_gpu_memory() -> None:
        nonlocal peak_used_mb, total_mb
        while not stop_event.is_set():
            try:
                used_mb, total_mb_candidate = query_gpu_memory_mb(args.gpu_index)
                if used_mb is not None:
                    peak_used_mb = max(peak_used_mb, used_mb)
                if total_mb_candidate is not None:
                    total_mb = total_mb_candidate
            except Exception:
                pass
            stop_event.wait(args.poll_interval)

    poll_thread = threading.Thread(target=poll_gpu_memory, daemon=True)
    poll_thread.start()

    start_time = time.time()
    return_code = 0
    measured_flops = None
    notes: list[str] = []
    forwarded_save_pred_args = maybe_disable_cfg(args.save_pred_args, args.disable_cfg)
    if args.disable_cfg:
        notes.append("disable_cfg=True: forwarded guidance_scale was forced to 1.0.")

    try:
        save_pred_path = Path(args.save_pred_script).expanduser().resolve()
        script_dir = str(save_pred_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import importlib.util

        spec = importlib.util.spec_from_file_location("save_pred_pipeline_module", str(save_pred_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import {save_pred_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["save_pred_pipeline_module"] = module
        spec.loader.exec_module(module)
        if args.disable_visuals and hasattr(module, "draw_tracks"):
            module.draw_tracks = lambda *unused_args, **unused_kwargs: None

        original_argv = sys.argv
        sys.argv = [str(save_pred_path), *forwarded_save_pred_args]
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as profile_obj:
                module.main()
            measured_flops = sum_profile_flops(profile_obj)
            if measured_flops == 0:
                notes.append("torch.profiler returned 0 FLOPs; unsupported ops may be omitted.")
        finally:
            sys.argv = original_argv
    except SystemExit as exc:
        return_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception:
        return_code = 1
        raise
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)

        elapsed = time.time() - start_time
        metrics = {
            "gpu_index": args.gpu_index,
            "processing_time_s": elapsed,
            "peak_gpu_memory_gb": peak_used_mb / 1024.0 if peak_used_mb else 0.0,
            "gpu_total_memory_gb": (total_mb / 1024.0) if total_mb else None,
            "measured_total_flops": measured_flops,
            "notes": notes,
            "save_pred_script": str(Path(args.save_pred_script).expanduser().resolve()),
            "save_pred_args_original": args.save_pred_args,
            "save_pred_args_effective": forwarded_save_pred_args,
            "disable_cfg": bool(args.disable_cfg),
            "return_code": return_code,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
