from __future__ import annotations

import os
import random
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_POLICY_ROOT = REPO_ROOT / "action_policy"
FLOW_ROOT = REPO_ROOT / "3DFlowModel"

for _path in (ACTION_POLICY_ROOT, FLOW_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

_FLOW_INFER = None


def get_flow_infer():
    global _FLOW_INFER
    if _FLOW_INFER is None:
        import _predict_flow_hdf5_impl as flow_infer

        _FLOW_INFER = flow_infer
    return _FLOW_INFER


@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    suit_type: str = "libero_spatial"
    task_ids: str = ""
    task_num: int = 10
    eval_num_init_states: int = 50
    init_state_start: int = 0
    num_envs: int = 10
    num_eval_steps: int = 0
    initial_wait_steps: int = 15
    execution_steps: int = 20
    flow_plan_chunks: int = 1

    action_ckpt: str = ""
    policy_cfg: str = str(ACTION_POLICY_ROOT / "model/model_res_dp_flow.yaml")
    ckpt: str = ""
    model_py: str = str(FLOW_ROOT / "flow_model.py")
    siglip_name: str = "google/siglip-base-patch16-224"

    k_steps: Optional[int] = None
    num_points: Optional[int] = None
    flow_ddim_steps: int = 20
    flow_train_noise_timesteps: int = 100
    flow_guidance_scale: float = 1.0
    flow_no_query_points: bool = True
    cond_kframes: int = 4

    action_ddim_steps: int = 16
    action_dim: int = 7

    capture_video: bool = True
    video_dir: str = "outputs/action_policy_rollouts"
    video_tag: str = ""
    video_max_success_per_task: int = 0
    video_max_fail_per_task: int = -1
    video_fps: int = 30


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Evaluate a LIBERO flow-conditioned diffusion policy.")
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--torch_deterministic", action=argparse.BooleanOptionalAction, default=Args.torch_deterministic)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=Args.cuda)

    p.add_argument("--suit_type", type=str, default=Args.suit_type, choices=sorted(LIBERO_MAX_STEPS.keys()))
    p.add_argument("--task_ids", type=str, default=Args.task_ids)
    p.add_argument("--task_num", type=int, default=Args.task_num)
    p.add_argument("--eval_num_init_states", type=int, default=Args.eval_num_init_states)
    p.add_argument("--init_state_start", type=int, default=Args.init_state_start)
    p.add_argument("--num_envs", type=int, default=Args.num_envs)
    p.add_argument("--num_eval_steps", type=int, default=Args.num_eval_steps)
    p.add_argument("--initial_wait_steps", type=int, default=Args.initial_wait_steps)
    p.add_argument("--execution_steps", type=int, default=Args.execution_steps)
    p.add_argument(
        "--flow_plan_chunks",
        type=int,
        default=Args.flow_plan_chunks,
        help="Number of action chunks executed before refreshing the flow plan. Values >1 enable lower-frequency flow planning.",
    )

    p.add_argument("--action_ckpt", type=str, default=Args.action_ckpt)
    p.add_argument("--policy_cfg", type=str, default=Args.policy_cfg)
    p.add_argument("--ckpt", type=str, default=Args.ckpt)
    p.add_argument("--model_py", type=str, default=Args.model_py)
    p.add_argument("--siglip_name", type=str, default=Args.siglip_name)

    p.add_argument("--k_steps", type=int, default=Args.k_steps)
    p.add_argument("--num_points", type=int, default=Args.num_points)
    p.add_argument("--flow_ddim_steps", type=int, default=Args.flow_ddim_steps)
    p.add_argument("--flow_train_noise_timesteps", type=int, default=Args.flow_train_noise_timesteps)
    p.add_argument("--flow_guidance_scale", type=float, default=Args.flow_guidance_scale)
    p.add_argument("--flow_no_query_points", action=argparse.BooleanOptionalAction, default=Args.flow_no_query_points)
    p.add_argument("--cond_kframes", type=int, default=Args.cond_kframes)

    p.add_argument("--action_ddim_steps", type=int, default=Args.action_ddim_steps)
    p.add_argument("--action_dim", type=int, default=Args.action_dim)

    p.add_argument("--capture_video", action=argparse.BooleanOptionalAction, default=Args.capture_video)
    p.add_argument("--video_dir", type=str, default=Args.video_dir)
    p.add_argument("--video_tag", type=str, default=Args.video_tag)
    p.add_argument("--video_max_success_per_task", type=int, default=Args.video_max_success_per_task)
    p.add_argument("--video_max_fail_per_task", type=int, default=Args.video_max_fail_per_task)
    p.add_argument("--video_fps", type=int, default=Args.video_fps)
    return Args(**vars(p.parse_args()))


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)


def parse_task_ids(task_ids: str, task_num: int) -> List[int]:
    if task_ids is None or str(task_ids).strip() == "":
        return list(range(int(task_num)))
    out: List[int] = []
    for item in str(task_ids).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("--task_ids did not contain any valid task ids")
    if min(out) < 0 or max(out) >= int(task_num):
        raise ValueError(f"LIBERO task ids must be in [0, {int(task_num) - 1}]")
    return out


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return root / p


def strip_module_prefix(state):
    out = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / den).astype(np.float32)


def preprocess_obs(obs, suit_type: str):
    images, wrist_images, states = [], [], []
    for item in obs:
        images.append(item["agentview_image"][::-1, ::-1])
        wrist = item["robot0_eye_in_hand_image"]
        if suit_type == "libero_10":
            wrist = wrist[::-1, ::-1]
        wrist_images.append(wrist)
        states.append(
            np.concatenate(
                [
                    item["robot0_eef_pos"],
                    quat2axisangle(item["robot0_eef_quat"]),
                    item["robot0_gripper_qpos"][:2],
                ]
            )
        )
    return {
        "state": np.stack(states).astype(np.float32),
        "full_image": np.stack(images),
        "wrist_image": np.stack(wrist_images),
    }


def process_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    action[:, 6] = np.where(action[:, 6] < 0.0, -1.0, 1.0)
    return action


def slugify(text: str, max_len: int = 96) -> str:
    text = str(text).strip().lower()
    chars, prev_sep = [], False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    slug = "".join(chars).strip("_")
    return (slug or "task")[:max_len].strip("_")


def create_env(args: Args, task_id: int, init_state_start: int, env_num: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    if args.suit_type not in LIBERO_MAX_STEPS:
        raise ValueError(f"Unsupported LIBERO suite: {args.suit_type}")

    benchmark_instance = benchmark.get_benchmark_dict()[args.suit_type]()
    task = benchmark_instance.get_task(task_id)
    instruction = task.language
    bddl_root = get_libero_path("bddl_files")
    env_args = {
        "bddl_file_name": os.path.join(bddl_root, task.problem_folder, task.bddl_file),
        "camera_heights": 256,
        "camera_widths": 256,
    }

    init_states = benchmark_instance.get_task_init_states(task_id)
    init_state_start = int(init_state_start)
    init_state_end = min(init_state_start + int(env_num), init_states.shape[0])
    if init_state_end <= init_state_start:
        return None, instruction, np.zeros((0,), dtype=np.int64)

    init_indices = np.arange(init_state_start, init_state_end, dtype=np.int64)
    envs = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(len(init_indices))])
    envs.seed(0)
    envs.set_init_state(init_states[init_indices])
    return envs, instruction, init_indices


def load_policy_cfg(args: Args):
    from omegaconf import OmegaConf

    cfg_path = resolve_path(args.policy_cfg, ACTION_POLICY_ROOT)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Policy config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    return cfg, cfg_path


def load_action_policy(args: Args, cfg, device: torch.device):
    from hydra.utils import instantiate

    if not args.action_ckpt:
        raise ValueError("Please provide --action_ckpt /path/to/best_action_policy_ema.pt")
    model = instantiate(cfg.model)
    ckpt = torch.load(args.action_ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", None)))
    if state is None:
        raise KeyError(f"No model weights found in action checkpoint: {args.action_ckpt}")
    model.load_state_dict(strip_module_prefix(state), strict=True)
    return model.to(device).eval()


def load_flow_model(args: Args, device: torch.device):
    if not args.ckpt:
        raise ValueError("Please provide --ckpt /path/to/siglip_flow_futureK_best.pt")
    if not args.flow_no_query_points:
        raise ValueError("This public LIBERO evaluator supports flow_no_query_points=True only.")
    flow_infer = get_flow_infer()
    model_py = resolve_path(args.model_py, FLOW_ROOT)
    model, k_steps, num_points, _ = flow_infer.load_flow_model(
        ckpt_path=Path(args.ckpt),
        model_py=model_py,
        device=str(device),
        k_steps_override=args.k_steps,
        num_points_override=args.num_points,
        strict=True,
    )
    args.k_steps = int(k_steps)
    args.num_points = int(num_points)
    return model.eval()


def process_img_history(img_history_batch, processor, device: torch.device) -> torch.Tensor:
    seqs = [seq for seq in img_history_batch]
    batch_size, num_frames = len(seqs), len(seqs[0])
    flat_imgs = [img for seq in seqs for img in seq]
    pixel_values = processor(images=flat_imgs, return_tensors="pt")["pixel_values"]
    return pixel_values.view(batch_size, num_frames, *pixel_values.shape[1:]).to(device)


@torch.inference_mode()
def predict_future_flow(
    flow_model,
    processor,
    img_history,
    instruction: List[str],
    args: Args,
    device: torch.device,
) -> np.ndarray:
    flow_infer = get_flow_infer()
    proc_txt = processor(text=instruction, return_tensors="pt", padding=True)
    pixel_values = process_img_history(img_history, processor, device)
    input_ids = proc_txt["input_ids"].to(device)
    attn_mask = proc_txt.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return flow_infer.predict_future_k_ddim_cfg_batch_tensor(
        model=flow_model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attn_mask=attn_mask,
        query_points=None,
        k_steps=int(args.k_steps),
        num_points=int(args.num_points),
        device=str(device),
        num_train_timesteps=int(args.flow_train_noise_timesteps),
        num_inference_steps=int(args.flow_ddim_steps),
        guidance_scale=float(args.flow_guidance_scale),
    )


@torch.no_grad()
def sample_actions_ddim(
    model,
    noise_scheduler,
    images: np.ndarray,
    wrist_images: np.ndarray,
    proprios: np.ndarray,
    flows: np.ndarray,
    instruction: List[str],
    processor,
    args: Args,
    device: torch.device,
) -> torch.Tensor:
    flows_t = torch.as_tensor(flows, dtype=torch.float32, device=device)
    images_t = torch.as_tensor(images, dtype=torch.float32, device=device)
    wrist_t = torch.as_tensor(wrist_images, dtype=torch.float32, device=device)
    proprios_t = torch.as_tensor(proprios, dtype=torch.float32, device=device)
    query_t = torch.zeros((flows_t.shape[0], flows_t.shape[2], 2), dtype=torch.float32, device=device)
    target_flows = torch.zeros_like(flows_t)

    proc_txt = processor(text=instruction, return_tensors="pt", padding=True)
    input_ids = proc_txt["input_ids"].to(device)
    attn_mask = proc_txt.get("attention_mask", torch.ones_like(input_ids)).to(device)

    batch_size = flows_t.shape[0]
    noise_scheduler.set_timesteps(int(args.action_ddim_steps), device=device)
    actions = torch.randn(batch_size, int(args.k_steps), int(args.action_dim), device=device)

    for t in noise_scheduler.timesteps:
        t_vec = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        noise_pred, _, _ = model(
            actions,
            t_vec,
            images_t,
            wrist_t,
            proprios_t,
            flows_t,
            query_t,
            target_flows,
            input_ids,
            attn_mask,
        )
        actions = noise_scheduler.step(noise_pred, t, actions, eta=0.0).prev_sample
    return actions


@torch.no_grad()
def predict_action_chunk(
    flow_model,
    action_policy,
    noise_scheduler,
    processor,
    obs_batch,
    img_history,
    instruction: str,
    args: Args,
    device: torch.device,
    use_flow_condition: bool,
    flows_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    batch_size = obs_batch["full_image"].shape[0]
    instruction_batch = [instruction] * batch_size
    if use_flow_condition:
        if flows_override is None:
            flows = predict_future_flow(flow_model, processor, img_history, instruction_batch, args, device)
        else:
            flows = np.asarray(flows_override, dtype=np.float32)
    else:
        flows = np.zeros((batch_size, int(args.k_steps), int(args.num_points), 3), dtype=np.float32)

    images = obs_batch["full_image"].transpose(0, 3, 1, 2)
    wrist_images = obs_batch["wrist_image"].transpose(0, 3, 1, 2)
    actions = sample_actions_ddim(
        action_policy,
        noise_scheduler,
        images,
        wrist_images,
        obs_batch["state"],
        flows,
        instruction_batch,
        processor,
        args,
        device,
    )
    return actions.detach().cpu().numpy(), flows


def save_rollout_video(
    args: Args,
    rollout_images: List[np.ndarray],
    task_description: str,
    episode_success: bool,
    task_id: int,
    init_id: int,
    num_success: int,
    num_envs: int,
) -> Path:
    import imageio

    rollout_dir = Path(args.video_dir) / str(args.suit_type)
    if args.video_tag:
        rollout_dir = rollout_dir / slugify(args.video_tag, max_len=80)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    result = "success" if episode_success else "fail"
    mp4_path = rollout_dir / (
        f"task{int(task_id):02d}--init{int(init_id):03d}"
        f"--{result}--success={int(num_success)}of{int(num_envs)}"
        f"--{slugify(task_description)}.mp4"
    )
    writer = imageio.get_writer(str(mp4_path), fps=int(args.video_fps))
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    return mp4_path


def evaluate_task(
    task_id: int,
    args: Args,
    flow_model,
    action_policy,
    noise_scheduler,
    processor,
    device: torch.device,
    use_flow_condition: bool,
):
    task_success_chunks = []
    task_ep_len_chunks = []
    task_init_indices = []
    saved_success_videos = 0
    saved_fail_videos = 0
    max_steps = int(args.num_eval_steps or LIBERO_MAX_STEPS[args.suit_type])
    init_stop = int(args.init_state_start + args.eval_num_init_states)

    for chunk_start in range(int(args.init_state_start), init_stop, int(args.num_envs)):
        chunk_envs = min(int(args.num_envs), init_stop - chunk_start)
        envs, instruction, init_indices = create_env(args, task_id, chunk_start, chunk_envs)
        if envs is None or len(init_indices) == 0:
            break

        print(
            f"[Task {task_id}] init states {int(init_indices[0])}-{int(init_indices[-1])} "
            f"({len(init_indices)} envs): {instruction}"
        )
        try:
            obs = envs.reset()
            num_envs = int(len(init_indices))
            ep_len = np.zeros(num_envs, dtype=np.int32)
            env_active = np.ones(num_envs, dtype=bool)
            env_success = np.zeros(num_envs, dtype=bool)
            video_frames: List[np.ndarray] = []

            dummy_action = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (num_envs, 1))
            for _ in range(int(args.initial_wait_steps)):
                obs, _, _, _ = envs.step(dummy_action)

            first = preprocess_obs(obs, args.suit_type)
            img_history = [
                [Image.fromarray(first["full_image"][env_id])] * int(args.cond_kframes)
                for env_id in range(num_envs)
            ]

            num_decisions = max_steps // int(args.execution_steps)
            flow_plan = None
            flow_plan_chunks = max(1, int(args.flow_plan_chunks))
            for decision_step in range(num_decisions):
                obs_batch = preprocess_obs(obs, args.suit_type)
                if use_flow_condition and (flow_plan is None or decision_step % flow_plan_chunks == 0):
                    instruction_batch = [instruction] * num_envs
                    flow_plan = predict_future_flow(
                        flow_model,
                        processor,
                        img_history,
                        instruction_batch,
                        args,
                        device,
                    )
                action_chunk, _ = predict_action_chunk(
                    flow_model,
                    action_policy,
                    noise_scheduler,
                    processor,
                    obs_batch,
                    img_history,
                    instruction,
                    args,
                    device,
                    use_flow_condition,
                    flows_override=flow_plan,
                )

                action_chunk = action_chunk.reshape(num_envs, -1, action_chunk.shape[-1])
                steps_this_chunk = min(int(args.execution_steps), action_chunk.shape[1])
                for action_id in range(steps_this_chunk):
                    obs, _, terminations, infos = envs.step(process_action(action_chunk[:, action_id]))
                    ep_len[env_active] += 1
                    post_obs = preprocess_obs(obs, args.suit_type)
                    if args.capture_video:
                        video_frames.append(post_obs["full_image"])
                    for env_id in range(num_envs):
                        img_history[env_id].append(Image.fromarray(post_obs["full_image"][env_id]))
                        img_history[env_id] = img_history[env_id][-int(args.cond_kframes):]

                    terminations = np.asarray(terminations, dtype=bool)
                    env_active = np.logical_and(env_active, np.logical_not(terminations))
                    if any(isinstance(info, dict) and "success" in info for info in infos):
                        this_success = np.array(
                            [bool(info.get("success", False)) if isinstance(info, dict) else False for info in infos],
                            dtype=bool,
                        )
                    else:
                        this_success = terminations
                    env_success |= this_success

                print(
                    f"[Task {task_id}] decision {decision_step + 1}/{num_decisions}: "
                    f"success={int(env_success.sum())}/{num_envs}",
                    flush=True,
                )

            succ_mask = env_success.astype(bool)
            num_success = int(succ_mask.sum())
            if args.capture_video and video_frames:
                for env_id in range(num_envs):
                    if succ_mask[env_id]:
                        should_save = args.video_max_success_per_task < 0 or saved_success_videos < args.video_max_success_per_task
                        saved_success_videos += int(should_save)
                    else:
                        should_save = args.video_max_fail_per_task < 0 or saved_fail_videos < args.video_max_fail_per_task
                        saved_fail_videos += int(should_save)
                    if should_save:
                        rollout_images = [batch[env_id] for batch in video_frames]
                        path = save_rollout_video(
                            args,
                            rollout_images,
                            instruction,
                            bool(succ_mask[env_id]),
                            task_id,
                            int(init_indices[env_id]),
                            num_success,
                            num_envs,
                        )
                        print(f"[video] {path}")

            print(
                f"[Task {task_id}] chunk {int(init_indices[0])}-{int(init_indices[-1])}: "
                f"success={num_success}/{num_envs} ({env_success.mean():.3f})"
            )
            task_success_chunks.append(env_success.copy())
            task_ep_len_chunks.append(ep_len.copy())
            task_init_indices.append(init_indices.copy())
        finally:
            envs.close()

    if not task_success_chunks:
        print(f"[Task {task_id}] no init states evaluated.")
        return None

    task_success = np.concatenate(task_success_chunks, axis=0)
    task_ep_len = np.concatenate(task_ep_len_chunks, axis=0)
    task_indices = np.concatenate(task_init_indices, axis=0)
    succ_mask = task_success.astype(bool)
    result = {
        "task_id": int(task_id),
        "num_episodes": int(len(task_success)),
        "success_rate": float(task_success.mean()),
        "init_state_start": int(task_indices.min()),
        "init_state_end": int(task_indices.max()),
    }
    if succ_mask.any():
        success_lens = task_ep_len[succ_mask]
        result.update(
            {
                "success_episode_length_mean": float(success_lens.mean()),
                "success_episode_length_median": float(np.median(success_lens)),
            }
        )
    print(f"[Task {task_id}] result: {result}")
    return result


def main() -> None:
    args = parse_args()

    from hydra.utils import instantiate
    from transformers import SiglipProcessor

    set_seed(args.seed, args.torch_deterministic)
    flow_infer = get_flow_infer()
    flow_infer.enable_sdpa()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    cfg, cfg_path = load_policy_cfg(args)
    use_flow_condition = bool(cfg.get("use_flow_condition", True))
    if args.k_steps is None:
        args.k_steps = 20
    if args.num_points is None:
        args.num_points = 100
    args.flow_plan_chunks = max(1, int(args.flow_plan_chunks))

    print(f"[INFO] device={device}")
    print(f"[INFO] policy_cfg={cfg_path}")
    print(f"[INFO] use_flow_condition={use_flow_condition}")
    print(
        f"[INFO] slow_fast: flow_plan_chunks={args.flow_plan_chunks}, "
        f"execution_steps={args.execution_steps}"
    )

    processor = SiglipProcessor.from_pretrained(args.siglip_name, use_fast=True)
    flow_model = load_flow_model(args, device) if use_flow_condition else None
    action_policy = load_action_policy(args, cfg, device)
    noise_scheduler = instantiate(cfg.noise_scheduler)

    task_ids = parse_task_ids(args.task_ids, args.task_num)
    results = []
    t0 = time.perf_counter()
    for task_id in task_ids:
        result = evaluate_task(
            task_id,
            args,
            flow_model,
            action_policy,
            noise_scheduler,
            processor,
            device,
            use_flow_condition,
        )
        if result is not None:
            results.append(result)

    if results:
        avg_success = float(np.mean([r["success_rate"] for r in results]))
        print(f"[DONE] average success rate: {avg_success:.4f}")
    else:
        print("[DONE] no tasks were evaluated.")
    print(f"[DONE] elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
