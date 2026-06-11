<h1 align="center">
  <img src="./assets/flow-icon.svg" alt="RoboFlow4D icon" width="42" align="center">
  RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation
</h1>

<p align="center">
  Proceedings of the International Conference on Machine Learning 2026
</p>

<p align="center">
  <strong>Sixu Lin</strong><sup>1,*</sup>,
  <strong>Junliang Chen</strong><sup>2,*</sup>,
  <strong>Huaiyuan Xu</strong><sup>2,&dagger;</sup>,
  <strong>Zhuohao Li</strong><sup>3</sup>,
  <strong>Guangming Wang</strong><sup>4</sup>,
  <strong>Yixiong Jing</strong><sup>4</sup>,
  <strong>Sheng Xu</strong><sup>1</sup>,
  <strong>Runyi Zhao</strong><sup>1</sup>,
  <strong>Brian Sheil</strong><sup>4</sup>,
  <strong>Lap-Pui Chau</strong><sup>2</sup>,
  <strong>Guiliang Liu</strong><sup>1,3,&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup>School of Data Science, The Chinese University of Hong Kong (Shenzhen)&nbsp;&nbsp;
  <sup>2</sup>The Hong Kong Polytechnic University<br>
  <sup>3</sup>Shenzhen Loop Area Institute&nbsp;&nbsp;
  <sup>4</sup>University of Cambridge
</p>

<p align="center">
  <sup>*</sup>Equal contribution&nbsp;&nbsp;
  <sup>&dagger;</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.17522"><img src="https://img.shields.io/badge/arXiv-2605.17522-b31b1b.svg" alt="arXiv"></a>
  <a href="https://simonlinsx.github.io/RoboFlow4D_Page/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg" alt="Project Page"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-b31b1b.svg" alt="Citation"></a>
</p>

<p align="center">
  <a href="./assets/pipeline.png">
    <img src="./assets/pipeline.png" alt="RoboFlow4D pipeline" width="100%">
  </a>
</p>

## Overview

RoboFlow4D is a lightweight flow world model for real-time flow-guided robotic manipulation. This repository provides the flow-model pipeline: data preprocessing, model training, inference, slow-fast flow-guided action evaluation, and visualization.

## Repository Scope

- `process_data/`: preprocessing for both simulation and real-world data.
- `3DFlowModel/`: flow model definition, training and inference.
- `action_policy/`: flow-conditioned action-policy training and evaluation.
- `visualization/`: visualization of the predicted point flow in 3D space.
- `utils/`: HDF5 inspection, trajectory filtering, metric checks, and helper scripts.

## Environment

Create the Python environment from the flow-model environment file:

```bash
conda env create -f 3DFlowModel/environment.yml
conda activate roboflow
```

After cloning the repository, initialize third-party code submodules:

```bash
git submodule update --init --recursive
```

The preprocessing scripts expect optional external repositories/checkpoints depending on the dataset:

- `SpaTrackerV2/` for 3D point trajectories.
- `Grounded-SAM-2/` for segmentation-guided query point selection.

RGB-D metric calibration is optional. It is needed when you want metric-scale point flows for motion planning, model-based control, or metric-scale training. Dataset-specific requirements are summarized in the preprocessing section.

## Data Processing

The processed HDF5 files use a compact training-facing convention:

- `point_traj`: original SpaTracker-scale 3D point flow.
- `point_traj_metric`: optional stage-aligned metric-scale point flow.

### Raw Data Sources

Start from the original dataset releases or official collection tools, then run the conversion scripts below.

- LIBERO: download demonstration HDF5 files from the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO), using `benchmark_scripts/download_libero_datasets.py` or its Hugging Face option.
- ManiSkill: follow the official [demonstration download](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) and [trajectory replay/conversion](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) documentation to obtain HDF5 demonstrations with the desired observations.
- DROID: use the official [DROID dataset](https://droid-dataset.github.io/) / TFDS release as the raw data source.
- Custom real-world videos: provide your own RGB videos. Metric calibration additionally requires synchronized depth, intrinsics, and camera-to-robot calibration.

For LIBERO preprocessing:

```bash
python process_data/process_libero_hdf5.py \
  --input_dirs /path/to/libero_hdf5_dir \
  --out_root /path/to/processed_tracks_root \
  --prompt_from_text \
  --device cuda
```

If you only need SpaTracker-scale flow training, stop here and train with `--traj_key point_traj`.

Optional: to add simulator RGB-D metric calibration and build metric trajectories:

```bash
python process_data/build_libero_metric_tracks.py \
  --tracks_root /path/to/processed_tracks_root \
  --libero_root /path/to/LIBERO \
  --out_root /path/to/clean_metric_tracks \
  --overwrite
```

This wrapper replays simulator depth, calibrates SpaTracker trajectories to metric coordinates, applies stage-level alignment, and exports the compact `point_traj` / `point_traj_metric` training keys. The lower-level scripts remain available for debugging or ablations.

For ManiSkill preprocessing:

```bash
python process_data/process_maniskill_hdf5.py \
  --input_dirs /path/to/maniskill_hdf5_dir \
  --out_root /path/to/maniskill_tracks \
  --device cuda
```

Optional: to add ManiSkill RGB-D metric calibration:

```bash
python process_data/patch_maniskill_rgbd_to_tracks.py \
  --tracks /path/to/maniskill_tracks/task_name/task_name_tracks.hdf5 \
  --rgbd_h5 /path/to/maniskill_rgbd_replay.h5 \
  --overwrite
```

For DROID datasets:

```bash
python process_data/process_droid.py \
  --droid_dir /path/to/droid_tfds \
  --out_root /path/to/droid_tracks
```

For custom real-world videos:

```bash
python process_data/mp4_to_minimal_real_hdf5.py \
  --input_dir /path/to/real_videos \
  --out_dir /path/to/real_hdf5

python process_data/process_arm_real.py \
  --input_dirs /path/to/real_hdf5 \
  --out_root /path/to/real_tracks
```

DROID, ManiSkill, and custom real-world preprocessing use gripper tracking by default. Pass `--prompt` only when tracking a non-gripper target. DROID and custom real-world preprocessing produce SpaTracker-scale `point_traj` by default. Metric calibration is optional and requires synchronized RGB-D, intrinsics, and camera-to-robot calibration.

The main dataset keys and common preprocessing commands are summarized above.

## Training

Train with the desired trajectory key. Native SpaTracker-scale training uses `--traj_key point_traj`; metric-scale training uses `--traj_key point_traj_metric`.

```bash
python 3DFlowModel/train_flow_model.py \
  --flow_root /path/to/clean_metric_tracks \
  --traj_key point_traj_metric \
  --save_dir /path/to/checkpoints
```

For multi-GPU training, launch the same script with `torchrun`.

## Inference

Run HDF5 inference with the same trajectory mode used in training:

```bash
python 3DFlowModel/predict_flow_hdf5.py \
  --hdf5_root /path/to/clean_metric_tracks \
  --ckpt /path/to/siglip_flow_futureK_best.pt \
  --out_key pre_point_traj_metric \
  --debug_tracks_dir outputs/debug_pred_tracks_metric \
  --no_query_points \
  --overwrite
```

`k_steps` and `num_points` are inferred from the checkpoint by default. The debug projection key is selected automatically from the output key and the available calibration data.

## Action Policy Learning

Train a flow-conditioned diffusion policy from tracks that already contain predicted flow:

```bash
PYTHONPATH=action_policy python action_policy/train/train_policy_dp_ema.py \
  --flow_root /path/to/libero_tracks_with_pred_flow \
  --save_dir /path/to/action_policy_ckpts
```

Evaluate the trained action policy:

```bash
PYTHONPATH=action_policy python action_policy/eval/eval_libero_dp.py \
  --suit_type libero_spatial \
  --action_ckpt /path/to/action_policy_ckpts/best_action_policy_ema.pt \
  --ckpt /path/to/siglip_flow_futureK_best.pt \
  --video_dir outputs/action_policy_rollouts
```

In this evaluator, the flow model is the slow planner and the diffusion policy is the fast controller. The planner predicts a future point-flow plan from recent RGB observations, and the controller executes short action chunks conditioned on the current observation and that flow plan.

## Visualization

Create an interactive 3D HTML demo for predicted point flow after inference:

```bash
python visualization/save_seg_all_starts_to_goal_3d_html.py \
  --flow_root /path/to/tracks_with_predictions.hdf5 \
  --out_dir outputs/html/predicted_flow \
  --traj_key pre_point_traj_metric
```

Use `--traj_key pre_point_traj` for native SpaTracker-scale predictions, or the corresponding prediction key you wrote with `--out_key` during inference.

Serve the generated HTML locally:

```bash
cd outputs/html/predicted_flow
python -m http.server 8899
```

Then open `http://localhost:8899/` in a browser.

## Acknowledgements

This code builds on several excellent open-source projects and datasets, including [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2), [VGGT](https://github.com/facebookresearch/vggt), [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://github.com/haosulab/ManiSkill), [robosuite](https://github.com/ARISE-Initiative/robosuite), [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip), [DINOv2](https://github.com/facebookresearch/dinov2), [PyTorch](https://pytorch.org/), and [diffusers](https://github.com/huggingface/diffusers).

## Citation

If you find this project useful, please cite:

```bibtex
@article{lin2026roboflow4d,
  title={RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation},
  author={Lin, Sixu and Chen, Junliang and Xu, Huaiyuan and Li, Zhuohao and Wang, Guangming and Jing, Yixiong and Xu, Sheng and Zhao, Runyi and Sheil, Brian and Chau, Lap-Pui and others},
  journal={arXiv preprint arXiv:2605.17522},
  year={2026}
}
```
