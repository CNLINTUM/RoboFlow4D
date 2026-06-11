#!/usr/bin/env python3
# pip install h5py pillow tqdm numpy
import argparse
import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os

# ---------------------------------------------------------------------------
# Runtime configuration. Values are populated from CLI args in main().
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path("data/demo")
OUTPUT_HDF5 = Path("data/demo/demo_assemble/dataset.hdf5")
ROBOT_TYPE = "panda"
FPS = 10
IMAGE_SIZE = (256, 256)
ONLY_DATASET_NAMES = {"demo_assemble"}

# Task description mapping.
TASK_DESCRIPTIONS = {
    "datasets_cup0": "Pick up the cup, then place it inside the white box.",
    "dataset_cube": "Pick up the small red cube on the table, then place it inside the white box.",
    "datasets_stack": "Stack the red cube on the blue cube.",
    "datasets_drawer": "Put the red cube into the closed top drawer, then close the drawer.",
    "datasets_assemble": "Pick up the brown cup and place it inside the black cup."
}

# pick_up_the_cup_and_place_it_inside_the_white_box

# stack_the_red_cube_on_the_blue_cube

# put_the_red_cube_into_the_closed_top_drawer_and_close_the_drawer

# pick_up_the_brown_cup_and_place_it_inside_the_black_cup

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert a folder of robot demos into a per-demo HDF5 file.")
    parser.add_argument("--datasets_root", type=Path, default=DATASETS_ROOT, help="Root folder containing dataset subfolders.")
    parser.add_argument("--output_hdf5", type=Path, default=OUTPUT_HDF5, help="Output HDF5 path.")
    parser.add_argument("--only_dataset_names", nargs="*", default=sorted(ONLY_DATASET_NAMES), help="Dataset folder names to convert.")
    parser.add_argument("--robot_type", default=ROBOT_TYPE)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--image_size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("HEIGHT", "WIDTH"))
    return parser.parse_args()


def center_crop_width(img: Image.Image, target_w: int):
    w, h = img.size
    if target_w >= w:
        return img
    left = (w - target_w) // 2
    right = left + target_w
    return img.crop((left, 0, right, h))

def analyze_data_structure():
    """Analyze the data directory layout and infer dimensions."""
    print("=== Analyzing data directory structure ===")
    
    if not DATASETS_ROOT.exists():
        print(f"Error: root directory does not exist: {DATASETS_ROOT}")
        return None
    
    dataset_info = {}
    
    for dataset_dir in DATASETS_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        if dataset_name not in ONLY_DATASET_NAMES:
            continue
        
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        # Find all trajectory directories.
        trajectory_dirs = []
        for item in dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith("frames_"):
                try:
                    # Extract the trajectory id.
                    traj_num = int(item.name.replace("frames_", ""))
                    trajectory_dirs.append((traj_num, item))
                except:
                    continue
        
        trajectory_dirs.sort(key=lambda x: x[0])
        
        if not trajectory_dirs:
            print("  Warning: no trajectory directories found")
            continue

        # Inspect the first trajectory to infer dimension information.
        first_traj_num, first_traj_dir = trajectory_dirs[0]
        trajectory_file = first_traj_dir / "trajectory.json"
        frames_dir = first_traj_dir / "frames"
        
        print(f"  Found {len(trajectory_dirs)} trajectory directories")
        print(f"  Inspecting first trajectory: frames_{first_traj_num}")

        # Check trajectory.json.
        if not trajectory_file.exists():
            print(f"  Warning: {trajectory_file} does not exist")
            continue

        # Check the frames directory.
        if not frames_dir.exists():
            print(f"  Warning: {frames_dir} does not exist")
            continue

        # Count main and wrist images.
        main_images = list(frames_dir.glob("main_*.png"))
        wrist_images = list(frames_dir.glob("wrist_*.png"))
        
        print(f"  main images: {len(main_images)}")
        print(f"  wrist images: {len(wrist_images)}")

        # Read trajectory.json to infer dimensions.
        try:
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            
            if not isinstance(trajectory_data, list) or len(trajectory_data) == 0:
                print("  Warning: trajectory.json is invalid")
                continue

            # Inspect the first frame. This dataset uses the "joints" field.
            first_frame = trajectory_data[0]
            joints = first_frame.get("joints", [])  # End-effector pose [x, y, z, rx, ry, rz].
            gripper = first_frame.get("gripper", 0.0)

            # joints are 6D (x, y, z, rx, ry, rz), plus gripper = 7D state.
            state_dim = len(joints) + 1
            action_dim = state_dim

            print(f"  state_dim: {state_dim} (joint={len(joints)} + gripper=1)")
            print(f"  joint example: {joints[:3]}... ({len(joints)} values total)")
            
            dataset_info[dataset_name] = {
                'trajectory_dirs': trajectory_dirs,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'trajectory_count': len(trajectory_dirs)
            }
            
        except Exception as e:
            print(f"  Error while reading trajectory.json: {e}")
            import traceback
            traceback.print_exc()
    
    return dataset_info


def create_hdf5_file():
    """Create the HDF5 file."""
    # Back up an existing output file.
    if OUTPUT_HDF5.exists():
        backup_path = OUTPUT_HDF5.with_suffix('.bak.hdf5')
        print(f"Backing up existing file: {OUTPUT_HDF5} -> {backup_path}")
        os.rename(OUTPUT_HDF5, backup_path)

    # Create the HDF5 file.
    h5_file = h5py.File(OUTPUT_HDF5, 'w')
    
    # Store global metadata.
    h5_file.attrs['robot_type'] = ROBOT_TYPE
    h5_file.attrs['fps'] = FPS
    h5_file.attrs['image_size'] = IMAGE_SIZE
    h5_file.attrs['dataset_format'] = 'per-demo'  # Mark as per-demo storage format.

    # Create a group for all demos.
    demos_group = h5_file.create_group('data')
    
    return h5_file


def get_image_indices(frames_dir):
    """Get image index lists."""
    main_indices = []
    wrist_indices = []
    
    for img_file in frames_dir.glob("main_*.png"):
        try:
            idx = int(img_file.stem.replace("main_", ""))
            main_indices.append(idx)
        except:
            pass
    
    for img_file in frames_dir.glob("wrist_*.png"):
        try:
            idx = int(img_file.stem.replace("wrist_", ""))
            wrist_indices.append(idx)
        except:
            pass
    
    return sorted(main_indices), sorted(wrist_indices)

def find_missing_indices(indices, expected_count):
    """Find missing indices."""
    if not indices:
        return list(range(expected_count))
    
    max_idx = max(indices)
    missing = []
    for i in range(expected_count):
        if i > max_idx:
            break
        if i not in indices:
            missing.append(i)
    return missing


def load_and_process_trajectory(traj_dir, state_dim, action_dim):
    """Load and process one trajectory."""
    trajectory_file = traj_dir / "trajectory.json"
    frames_dir = traj_dir / "frames"
    
    if not trajectory_file.exists():
        print("  Warning: trajectory.json does not exist")
        return None
    
    if not frames_dir.exists():
        print("  Warning: frames directory does not exist")
        return None
    
    # Load trajectory metadata.
    try:
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
    except Exception as e:
        print(f"  Warning: failed to read trajectory.json: {e}")
        return None
    
    if not isinstance(trajectory_data, list) or len(trajectory_data) == 0:
        print("  Warning: trajectory.json is empty or invalid")
        return None
    
    # Get image indices.
    main_indices, wrist_indices = get_image_indices(frames_dir)
    num_steps = len(trajectory_data)
    
    print(f"  trajectory: {num_steps} steps, main images: {len(main_indices)}, wrist images: {len(wrist_indices)}")

    # Check whether image frames are complete.
    missing_main = find_missing_indices(main_indices, num_steps)
    missing_wrist = find_missing_indices(wrist_indices, num_steps)
    
    if missing_main or missing_wrist:
        print(f"  Warning: missing images - main: {len(missing_main)}, wrist: {len(missing_wrist)}")
        # Only process indices that have both main and wrist images.
        available_indices = set(main_indices) & set(wrist_indices)
        available_indices = [i for i in range(num_steps) if i in available_indices]
    else:
        available_indices = list(range(num_steps))
    
    if not available_indices:
        print("  Warning: no usable image data")
        return None

    # Collect data arrays.
    images = []
    wrist_images = []
    states = []
    actions = []
    
    for idx in available_indices:
        if idx >= len(trajectory_data):
            break
            
        sample = trajectory_data[idx]
        joints = sample.get("joints", [])
        gripper = sample.get("gripper", 0.0)
        
        # Check data dimensions.
        if len(joints) + 1 != state_dim:
            print(f"  Warning: state dimension mismatch, expected {state_dim}, got {len(joints)+1}; skipping frame {idx}")
            continue

        # State: joints + gripper.
        state = np.array(list(joints) + [gripper], dtype=np.float32)
        
        if idx < num_steps - 1 and (idx + 1) in available_indices:
            next_sample = trajectory_data[idx + 1]
            next_joints = next_sample.get("joints", [])
            next_gripper = next_sample.get("gripper", 0.0)
            action = np.array(list(next_joints) + [next_gripper], dtype=np.float32)
        else:
            action = state.copy()
        
        main_img_path = frames_dir / f"main_{idx}.png"
        wrist_img_path = frames_dir / f"wrist_{idx}.png"
        
        try:
            main_img = Image.open(main_img_path).convert('RGB')
            main_img = center_crop_width(main_img, 480)
            main_img = main_img.resize(IMAGE_SIZE)

            wrist_img = Image.open(wrist_img_path).convert('RGB')
            wrist_img = center_crop_width(wrist_img, 480)
            wrist_img = wrist_img.resize(IMAGE_SIZE)

            
            images.append(np.array(main_img, dtype=np.uint8))
            wrist_images.append(np.array(wrist_img, dtype=np.uint8))
            states.append(state)
            actions.append(action)
        except Exception as e:
            print(f"  Warning: failed to load images at idx={idx}: {e}")
    
    if not images:
        print("  Warning: no usable data")
        return None
    
    return {
        'images': np.array(images, dtype=np.uint8),
        'wrist_images': np.array(wrist_images, dtype=np.uint8),
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'num_frames': len(images)
    }


def save_demo_to_hdf5(demos_group, demo_index, demo_data, task_desc, state_dim):
    """Save one demo into HDF5."""
    demo_group = demos_group.create_group(f'demo_{demo_index}')
    
    # Save image data.
    demo_group.create_dataset('observations/image', 
                             data=demo_data['images'],
                             compression='gzip')
    
    demo_group.create_dataset('observations/wrist_image', 
                             data=demo_data['wrist_images'],
                             compression='gzip')
    
    # Save state and action data.
    demo_group.create_dataset('observations/state', 
                             data=demo_data['states'],
                             compression='gzip')
    
    demo_group.create_dataset('actions', 
                             data=demo_data['actions'],
                             compression='gzip')
    
    # Save metadata.
    demo_group.attrs['num_frames'] = demo_data['num_frames']
    demo_group.attrs['task'] = task_desc
    demo_group.attrs['state_dim'] = state_dim
    demo_group.attrs['image_size'] = IMAGE_SIZE
    
    return True


# ---------------------------------------------------------------------------
# Main conversion flow
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global DATASETS_ROOT, OUTPUT_HDF5, ROBOT_TYPE, FPS, IMAGE_SIZE, ONLY_DATASET_NAMES
    DATASETS_ROOT = args.datasets_root
    OUTPUT_HDF5 = args.output_hdf5
    ROBOT_TYPE = args.robot_type
    FPS = args.fps
    IMAGE_SIZE = tuple(args.image_size)
    ONLY_DATASET_NAMES = set(args.only_dataset_names)

    # Analyze the data structure.
    dataset_info = analyze_data_structure()
    if not dataset_info:
        print("Error: no valid datasets found")
        return

    # Use the first dataset to infer dimensions.
    first_dataset = list(dataset_info.keys())[0]
    state_dim = dataset_info[first_dataset]['state_dim']
    action_dim = dataset_info[first_dataset]['action_dim']
    print("\n=== Creating HDF5 file ===")
    print(f"state_dim: {state_dim} (joints[6] + gripper[1])")
    print(f"action_dim: {action_dim}")
    
    h5_file = create_hdf5_file()
    
    # Store global dimension metadata.
    h5_file.attrs['state_dim'] = state_dim
    h5_file.attrs['action_dim'] = action_dim
    h5_file.attrs['state_description'] = 'joints[6] + gripper[1]'
    
    # Get the demos group.
    demos_group = h5_file['data']
    
    # Counters.
    total_frames = 0
    demo_counter = 0
    processed_trajectories = 0
    
    print("\n=== Processing data ===")

    # Process each dataset.
    for dataset_name, info in dataset_info.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Total trajectories: {info['trajectory_count']}")

        # Get task description.
        task_desc = TASK_DESCRIPTIONS.get(dataset_name, "Complete the task.")
        
        # Keep trajectories with ids <= 100.
        trajectory_dirs = info['trajectory_dirs']
        filtered_trajectories = [(num, dir) for num, dir in trajectory_dirs if num <= 100]
        
        print(f"Filtered trajectories (<100): {len(filtered_trajectories)}")

        # Process each trajectory.
        for traj_num, traj_dir in tqdm(filtered_trajectories, desc=f"Processing {dataset_name}"):
            print(f"\nProcessing trajectory {traj_num}...")

            # Load and process trajectory data.
            demo_data = load_and_process_trajectory(traj_dir, state_dim, action_dim)

            if demo_data is None:
                print(f"  Skipping trajectory {traj_num}: data loading failed")
                continue

            if demo_data['num_frames'] == 0:
                print(f"  Skipping trajectory {traj_num}: no valid frames")
                continue

            print(f"  trajectory {traj_num}: processed {demo_data['num_frames']} frames")

            # Save as a separate demo.
            success = save_demo_to_hdf5(demos_group, demo_counter, demo_data, task_desc, state_dim)
            
            if success:
                total_frames += demo_data['num_frames']
                demo_counter += 1
                processed_trajectories += 1
                print(f"  Saved as demo_{demo_counter-1}")
            else:
                print(f"  Save failed: trajectory {traj_num}")

    # Store demo-count metadata.
    h5_file.attrs['num_demos'] = demo_counter
    h5_file.attrs['total_frames'] = total_frames
    
    # Close the HDF5 file.
    h5_file.close()
    
    print("\nConversion finished.")
    print(f"  Output file: {OUTPUT_HDF5}")
    print(f"  Total demos: {demo_counter}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processed trajectories: {processed_trajectories}")
    print(f"  State dimension: {state_dim} (joints[6] + gripper[1])")

    # Verify the output file structure.
    print("\nVerifying HDF5 file structure...")
    with h5py.File(OUTPUT_HDF5, 'r') as f:
        print("  File attributes:")
        print(f"    Robot type: {f.attrs.get('robot_type')}")
        print(f"    FPS: {f.attrs.get('fps')}")
        print(f"    Number of demos: {f.attrs.get('num_demos')}")
        print(f"    Total frames: {f.attrs.get('total_frames')}")
        
        data_group = f['data']
        print("\n  Data group structure:")

        # List all demos.
        demo_keys = list(data_group.keys())
        print(f"  Included demos: {demo_keys[:5]}{'...' if len(demo_keys) > 5 else ''}")

        if demo_keys:
            # Inspect the first demo.
            first_demo = data_group[demo_keys[0]]
            print("\n  First demo (demo_0) structure:")
            print(f"    Frames: {first_demo.attrs.get('num_frames')}")
            print(f"    Task: {first_demo.attrs.get('task')}")
            print(f"    State dimension: {first_demo.attrs.get('state_dim')}")

            # Check dataset shapes.
            if 'observations/image' in first_demo:
                img_shape = first_demo['observations/image'].shape
                print(f"    Image shape: {img_shape}")

            if 'observations/state' in first_demo:
                state_shape = first_demo['observations/state'].shape
                print(f"    State shape: {state_shape}")

            if 'actions' in first_demo:
                action_shape = first_demo['actions'].shape
                print(f"    Action shape: {action_shape}")


# Simple readback example.
def read_hdf5_example():
    """Show how to read the converted HDF5 file."""
    print("\n=== Readback example ===")

    with h5py.File(OUTPUT_HDF5, 'r') as f:
        # Get number of demos.
        num_demos = f.attrs['num_demos']
        print(f"File contains {num_demos} demos")

        # Read the first demo.
        demo_0 = f['data/demo_0']

        # Get metadata.
        num_frames = demo_0.attrs['num_frames']
        task = demo_0.attrs['task']
        
        print("\ndemo_0 info:")
        print(f"  Frames: {num_frames}")
        print(f"  Task: {task}")

        # Read arrays.
        images = demo_0['observations/image'][:]
        wrist_images = demo_0['observations/wrist_image'][:]
        states = demo_0['observations/state'][:]
        actions = demo_0['actions'][:]
        
        print("\nData shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Wrist images: {wrist_images.shape}")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")

        # Show the first frame's state.
        if len(states) > 0:
            print(f"\nFirst-frame state (first 3 dims): {states[0][:3]}")
            print(f"First-frame gripper value: {states[0][-1]}")
        
        return images, states, actions


if __name__ == "__main__":
    main()
    
    # If the file was created successfully, show how to read it.
    if OUTPUT_HDF5.exists():
        try:
            read_hdf5_example()
        except Exception as e:
            print(f"Readback example failed: {e}")
