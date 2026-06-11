#!/usr/bin/env python3
# pip install lerobot Pillow tqdm numpy
import argparse
import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------------------------
# Runtime configuration. Values are populated from CLI args in main().
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path("data")
REPO_NAME = "cpu70"
ROBOT_TYPE = "panda"
FPS = 10
IMAGE_SIZE = (224, 224)
ONLY_DATASET_NAMES = {"dataset_cup"}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert local robot demos into a LeRobot dataset.")
    parser.add_argument("--datasets_root", type=Path, default=DATASETS_ROOT, help="Root folder containing dataset subfolders.")
    parser.add_argument("--repo_name", default=REPO_NAME, help="LeRobot repo_id to create.")
    parser.add_argument("--robot_type", default=ROBOT_TYPE)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--image_size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--only_dataset_names", nargs="*", default=sorted(ONLY_DATASET_NAMES))
    return parser.parse_args()


def load_replay_log(log_path: Path):
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if (not isinstance(data, list)) or len(data) == 0:
        return None
    return data


def list_indices(frames_dir: Path):
    main_idx = set()
    wrist_idx = set()
    for p in frames_dir.glob("*.png"):
        name = p.name
        if name.startswith("main_") and name.endswith(".png"):
            try:
                main_idx.add(int(name[len("main_") : -len(".png")]))
            except Exception:
                pass
        elif name.startswith("wrist_") and name.endswith(".png"):
            try:
                wrist_idx.add(int(name[len("wrist_") : -len(".png")]))
            except Exception:
                pass
    return main_idx, wrist_idx


def first_missing_in_prefix(idx_set, T):
    for i in range(T):
        if i not in idx_set:
            return i
    return None


def pad_tail_images(frames_dir: Path, num_steps: int, verbose: bool = True):
    """
    Ensure that every idx=0..num_steps-1 has both main_idx and wrist_idx.
    Only tail padding is supported: copy the last paired frame to missing tail
    indices. Gaps in the middle are not repaired.

    Returns:
      ok: whether conversion is safe
      last_paired: last paired index used as the copy source
      padded: number of padded indices
    """
    if num_steps <= 0:
        return False, None, 0

    main_idx, wrist_idx = list_indices(frames_dir)
    paired = sorted(list(main_idx & wrist_idx))

    if len(paired) == 0:
        if verbose:
            print(f"[Bad] {frames_dir}: no paired (main_i, wrist_i) images found")
        return False, None, 0

    last_paired = paired[-1]

    # Check that 0..min(last_paired, num_steps-1) is continuously paired.
    need_prefix = min(last_paired, num_steps - 1) + 1
    miss_main = first_missing_in_prefix(main_idx, need_prefix)
    miss_wrist = first_missing_in_prefix(wrist_idx, need_prefix)
    if miss_main is not None or miss_wrist is not None:
        if verbose:
            print(f"[Bad] {frames_dir}: missing frames in the middle; cannot align by tail padding only")
            if miss_main is not None:
                print(f"  missing main_{miss_main}.png within 0..{need_prefix-1}")
            if miss_wrist is not None:
                print(f"  missing wrist_{miss_wrist}.png within 0..{need_prefix-1}")
        return False, last_paired, 0

    # If replay_log is longer, pad last_paired+1..num_steps-1 from last_paired.
    padded = 0
    if num_steps - 1 > last_paired:
        src_main = frames_dir / f"main_{last_paired}.png"
        src_wrist = frames_dir / f"wrist_{last_paired}.png"
        if (not src_main.exists()) or (not src_wrist.exists()):
            if verbose:
                print(f"[Bad] {frames_dir}: copy-source image does not exist, src={last_paired}")
            return False, last_paired, 0

        for idx in range(last_paired + 1, num_steps):
            dst_main = frames_dir / f"main_{idx}.png"
            dst_wrist = frames_dir / f"wrist_{idx}.png"

            # Do not overwrite existing files.
            if not dst_main.exists():
                shutil.copy2(src_main, dst_main)
            if not dst_wrist.exists():
                shutil.copy2(src_wrist, dst_wrist)

            padded += 1

        if verbose:
            print(
                f"[PadTail] {frames_dir.name}: replay_log={num_steps}, "
                f"last_paired={last_paired}, padded_indices={padded}"
            )

    # Recheck that 0..num_steps-1 is fully populated.
    main_idx2, wrist_idx2 = list_indices(frames_dir)
    for idx in range(num_steps):
        if idx not in main_idx2 or idx not in wrist_idx2:
            if verbose:
                print(f"[Bad] {frames_dir}: main or wrist still missing at idx={idx} after tail padding")
            return False, last_paired, padded

    return True, last_paired, padded


def infer_dims_and_create_dataset():
    """
    Scan the dataset once to find the first valid trajectory and infer:
      - state dimension (joints + gripper)
      - action dimension (= state dimension)
    Then create the LeRobotDataset instance.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("This converter requires LeRobot. Install it before running conversion.") from exc

    if not DATASETS_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {DATASETS_ROOT}")

    first_state_dim = None
    first_action_dim = None

    for ds_dir in sorted(DATASETS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue

        replay_logs = sorted(ds_dir.glob("*_replay_log.json"))
        if not replay_logs:
            continue

        for log_path in replay_logs:
            traj_id = log_path.stem.replace("_replay_log", "")
            frames_dir = ds_dir / f"camera_frames_{traj_id}"
            if not frames_dir.exists():
                continue

            log_data = load_replay_log(log_path)
            if log_data is None:
                continue

            sample0 = log_data[0]
            joints = sample0["joints"]
            gripper = sample0.get("gripper", 0.0)

            state = np.array(list(joints) + [gripper], dtype=np.float32)
            action = state.copy()

            first_state_dim = state.shape[0]
            first_action_dim = action.shape[0]

            print(f"[Infer] state_dim={first_state_dim}, action_dim={first_action_dim}")
            break

        if first_state_dim is not None:
            break

    if first_state_dim is None:
        raise RuntimeError("Could not find a valid replay_log in the dataset")

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (first_state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (first_action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    return dataset, first_state_dim, first_action_dim


# ---------------------------------------------------------------------------
# Main conversion flow
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global DATASETS_ROOT, REPO_NAME, ROBOT_TYPE, FPS, IMAGE_SIZE, ONLY_DATASET_NAMES
    DATASETS_ROOT = args.datasets_root
    REPO_NAME = args.repo_name
    ROBOT_TYPE = args.robot_type
    FPS = args.fps
    IMAGE_SIZE = tuple(args.image_size)
    ONLY_DATASET_NAMES = set(args.only_dataset_names)

    dataset, state_dim, action_dim = infer_dims_and_create_dataset()

    print(f"Scanning dataset root: {DATASETS_ROOT}")
    top_level_dirs = sorted(DATASETS_ROOT.iterdir())
    print(f"Found {len(top_level_dirs)} dataset folders.")

    for ds_dir in top_level_dirs:
        if not ds_dir.is_dir():
            continue

        print(f"\n[Dataset] {ds_dir.name}")
        if ds_dir.name not in ONLY_DATASET_NAMES:
            print(f"[Skip] skipping {ds_dir.name}")
            continue

        if ds_dir.name == "dataset_cup":
            dataset_task = "Pick up the cup on the white plate, then place it inside the white box."
        elif ds_dir.name == "dataset_cube":
            dataset_task = "Pick up the small red cube on the table, then place it inside the white box."
        else:
            dataset_task = "Complete the task."

        replay_logs = sorted(ds_dir.glob("*_replay_log.json"))
        print(f"Found {len(replay_logs)} trajectories in {ds_dir.name}")

        for log_path in tqdm(replay_logs, desc=f"Processing {ds_dir.name}"):
            traj_id = log_path.stem.replace("_replay_log", "")

            try:
                traj_num = int(traj_id)
            except ValueError:
                print(f"[Skip] traj_id is not numeric: {traj_id} (from {log_path.name})")
                continue

            if not (51 <= traj_num <= 100):
                continue

            frames_dir = ds_dir / f"camera_frames_{traj_id}"
            if not frames_dir.exists():
                print(f"[Skip] {frames_dir} does not exist")
                continue


            log_data = load_replay_log(log_path)
            if log_data is None:
                print(f"[Skip] {log_path.name} is empty or malformed")
                continue

            num_steps = len(log_data)

            # Tail-pad images so idx=0..num_steps-1 is fully readable.
            ok, last_paired, padded = pad_tail_images(frames_dir, num_steps, verbose=True)
            if not ok:
                print(f"[Skip] {traj_id}: image frames cannot be aligned for conversion")
                continue

            added_frames = 0

            # Read num_steps frames; every step should now have images.
            for idx in range(num_steps):
                sample = log_data[idx]
                joints = sample["joints"]
                gripper = sample.get("gripper", 0.0)

                state = np.array(list(joints) + [gripper], dtype=np.float32)

                if idx < num_steps - 1:
                    next_sample = log_data[idx + 1]
                    next_joints = next_sample["joints"]
                    next_gripper = next_sample.get("gripper", 0.0)
                    action = np.array(list(next_joints) + [next_gripper], dtype=np.float32)
                else:
                    action = state.copy()

                main_img_path = frames_dir / f"main_{idx}.png"
                wrist_img_path = frames_dir / f"wrist_{idx}.png"

                # This should not happen after padding, but keep a guard.
                if not main_img_path.exists() or not wrist_img_path.exists():
                    print(
                        f"[Warn] image still missing after padding: main={main_img_path.name} / "
                        f"wrist={wrist_img_path.name}; skipping this step"
                    )
                    continue

                main_image = Image.open(main_img_path).convert("RGB").resize(IMAGE_SIZE)
                wrist_image = Image.open(wrist_img_path).convert("RGB").resize(IMAGE_SIZE)

                frame_data = {
                    "image": main_image,
                    "wrist_image": wrist_image,
                    "state": state.astype(np.float32),
                    "actions": action.astype(np.float32),
                    "task": dataset_task,
                }

                dataset.add_frame(frame_data)
                added_frames += 1

            if added_frames > 0:
                dataset.save_episode()
            else:
                print(f"[Skip] {traj_id}: no frames were added; episode not saved")

    print("Finalizing dataset...")
    print(f"\n✅ Conversion complete! Dataset saved as '{REPO_NAME}'")
    

if __name__ == "__main__":
    main()
