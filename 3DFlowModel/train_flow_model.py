import os, random, re, sys, math
# Avoid HDF5 file-lock errors when multiple processes read the same .hdf5 files
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from torch.utils.data import Dataset, DataLoader
from transformers import SiglipProcessor
from align_projector import AlignProjector, resample_tokens_to_length
import random
import h5py

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
import time


import wandb
wandb.util.working_set = lambda: ()  

import importlib.util
MODEL_PATH = Path(__file__).parent / "flow_model.py"
spec = importlib.util.spec_from_file_location("flow_model", MODEL_PATH)
model_siglip = importlib.util.module_from_spec(spec)
sys.modules["flow_model"] = model_siglip
spec.loader.exec_module(model_siglip)  # now we can access GenerativeFlowModel


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def enable_sdpa():
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception:
        pass


# ==========================
# Distributed (DDP) helpers
# ==========================
def _dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if _dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if _dist_avail_and_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_for_distributed(is_master: bool):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    """Initialize torch.distributed from torchrun environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif getattr(args, "local_rank", -1) != -1:
        # Legacy torch.distributed.launch
        args.rank = int(os.environ.get("RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        setup_for_distributed(True)
        return

    args.distributed = args.world_size > 1
    setup_for_distributed(args.rank == 0)

    if args.distributed:
        backend = getattr(args, "ddp_backend", None)
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        dist.barrier()

def cleanup_distributed():
    if _dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m
def second_order_smooth_loss(x, weights=None, eps=1e-3, all3_visible=False):
    """
    x: [B,K,N,3]  (x0_hat)
    weights: [B,K,N] or None  (visibility)
    return: scalar
    """
    B, K, N, _ = x.shape
    if K < 3:
        return x.new_zeros(())

    # Second-order difference: a_t = x_{t+1} - 2 x_t + x_{t-1}.
    acc = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]          # [B,K-2,N,3]

    # Charbonnier is more tolerant of real fast motion than plain L2.
    acc_mag = torch.sqrt((acc * acc).sum(dim=-1) + eps * eps)  # [B,K-2,N]

    if weights is None:
        return acc_mag.mean()

    w_mid = weights[:, 1:-1]  # Align to K-2.

    if all3_visible:
        w_mid = w_mid * weights[:, :-2] * weights[:, 2:]

    denom = w_mid.sum().clamp_min(1e-8)
    return (acc_mag * w_mid).sum() / denom

def normalize_instr_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("_", " ").strip()

    # Remove prefixes such as "kitchen scene4", "living room scene5", or "study scene1".
    s = re.sub(r"^(?:kitchen|living room|study)\s+scene\s*\d+\s+", "", s)

    # Strip common "tracks" / "track(s)" suffixes, optionally with "demo".
    s = re.sub(r"(?:\s+demo)?\s+tracks?\s*$", "", s)

    # Then handle odd suffixes such as "demo", "demo 123", or "demo demo 123".
    s = re.sub(r"(?:\s|_)+(?:demo(?:\s|_)+demo|\bdemo)\s*\d+\s*$", "", s)
    s = re.sub(r"(?:\s|_)+demo\s*$", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_task_type_from_path(path: Path) -> str:
    """Infer dataset family from an HDF5 path for gripper-stage parsing."""
    text = str(path).lower()
    if (
        "maniskill" in text
        or "pickcube" in text
        or "pushcube" in text
        or "stackcube" in text
    ):
        return "maniskill"
    if (
        "libero" in text
        or "kitchen_scene" in text
        or "living_room_scene" in text
        or "study_scene" in text
    ):
        return "libero"
    return "real"


def resolve_plus_is_close(task_type: str, override: str = "auto") -> bool:
    """Return the gripper sign convention used for stage segmentation."""
    override = str(override).lower()
    if override in {"true", "1", "yes"}:
        return True
    if override in {"false", "0", "no"}:
        return False
    task_type = str(task_type).lower()
    if task_type == "libero":
        return True
    # ManiSkill action convention is opposite to LIBERO in the processed data.
    return False

def list_demo_dirs(flow_root: Path) -> List[Path]:
    """
    Recursively find directories under flow_root that contain point_traj.npy.
    This supports both older layouts and newer suite-level layouts.
    """
    demo_dirs: List[Path] = []
    # Use the parent of each point_traj.npy file as a demo directory.
    for traj_file in sorted(flow_root.rglob("point_traj.npy")):
        demo_dirs.append(traj_file.parent)
    return demo_dirs


def load_mapping(mapping_csv: Optional[str], text_col: Optional[str], prompt_col: Optional[str]):
    if mapping_csv is None:
        return {}
    df = pd.read_csv(mapping_csv)
    cols = {c.lower(): c for c in df.columns}
    text = cols.get((text_col or "instruction").lower())
    prompt = cols.get((prompt_col or "prompt").lower())

    out = {}
    for _, row in df.iterrows():
        instr = str(row.get(text, "")).strip() if text else ""
        pr    = str(row.get(prompt, "")).strip() if prompt else ""
        key   = normalize_instr_key(instr)
        if key:
            out[key] = {"instruction": instr, "prompt": pr}
    return out


from collections import OrderedDict
from threading import Lock

class _LRUMemmapCache:
    """Small LRU cache for np.memmap arrays; it mainly uses the OS page cache."""
    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()

    def get(self, path: Path, mmap: bool = True):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            arr = np.load(path, mmap_mode='r' if mmap else None, allow_pickle=False)
            self._cache[key] = arr
            if len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
            return arr

def augment_image_sequence_rlds_style(
    images: List[Image.Image],
    image_size: Tuple[int, int] = (256, 256),
    train: bool = True,
    image_augment_kwargs: Dict[str, Any] = None,
    return_geom: bool = False,
) -> List[Image.Image]:
    """
    Apply RLDS-style image augmentation to a sequence.

    The same random crop and color jitter parameters are shared across all
    frames in the sequence.

    Args:
        images: list of K PIL images.
        image_size: output image size as (H, W).
        train: if False, only deterministic resizing is applied.
        image_augment_kwargs: configuration dictionary such as:
            {
                "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
                "random_brightness": [0.2],
                "random_contrast": [0.8, 1.2],
                "random_saturation": [0.8, 1.2],
                "random_hue": [0.05],
                "augment_order": [...]
            }
    Returns:
        The augmented image sequence as a list of PIL images.
    """
    assert len(images) > 0, "images sequence must not be empty"
    images = [img.convert("RGB") for img in images]

    if image_augment_kwargs is None:
        image_augment_kwargs = {}

    def _image_size_to_hw(size) -> Tuple[int, int]:
        if isinstance(size, int):
            return int(size), int(size)
        if isinstance(size, (tuple, list)):
            if len(size) == 1:
                return int(size[0]), int(size[0])
            return int(size[0]), int(size[1])
        raise TypeError(f"Unsupported image_size type: {type(size)}")

    src_hw = (int(images[0].height), int(images[0].width))
    out_hw = _image_size_to_hw(image_size)

    # Evaluation uses deterministic resize only.
    if not train:
        out = [
            F.resize(img, image_size, interpolation=InterpolationMode.BILINEAR)
            for img in images
        ]
        if return_geom:
            return out, {"src_hw": src_hw, "out_hw": out_hw, "crop_params": None}
        return out

    # Parse augmentation config.
    aug_order = image_augment_kwargs.get(
        "augment_order",
        [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # random_resized_crop
    rrc_cfg = image_augment_kwargs.get("random_resized_crop", None)
    # brightness: [0.2] samples a factor from [1-0.2, 1+0.2].
    rb = image_augment_kwargs.get("random_brightness", None)   # [0.2]
    # contrast: [0.8, 1.2] samples a factor from [0.8, 1.2].
    rc = image_augment_kwargs.get("random_contrast", None)
    # saturation: [0.8, 1.2]
    rsat = image_augment_kwargs.get("random_saturation", None)
    # hue: [0.05] samples a factor from [-0.05, 0.05] following torch adjust_hue semantics.
    rh = image_augment_kwargs.get("random_hue", None)

    # Sample one augmentation configuration per sequence.

    # 1) RandomResizedCrop parameters, sampled from the first frame.
    if rrc_cfg is not None:
        scale = tuple(rrc_cfg.get("scale", [0.9, 0.9]))
        ratio = tuple(rrc_cfg.get("ratio", [1.0, 1.0]))
        # torchvision get_params samples one crop region.
        i, j, h, w = RandomResizedCrop.get_params(images[0], scale=scale, ratio=ratio)
        rrc_params = (i, j, h, w)
    else:
        rrc_params = None

    # 2) brightness factor
    if rb is not None and len(rb) > 0:
        # rb = [0.2] -> [1-0.2, 1+0.2]
        delta = float(rb[0])
        brightness_factor = random.uniform(max(0.0, 1.0 - delta), 1.0 + delta)
    else:
        brightness_factor = None

    # 3) contrast factor
    if rc is not None and len(rc) == 2:
        contrast_factor = random.uniform(float(rc[0]), float(rc[1]))
    else:
        contrast_factor = None

    # 4) saturation factor
    if rsat is not None and len(rsat) == 2:
        saturation_factor = random.uniform(float(rsat[0]), float(rsat[1]))
    else:
        saturation_factor = None

    # 5) hue factor
    if rh is not None and len(rh) > 0:
        max_h = float(rh[0])
        hue_factor = random.uniform(-max_h, max_h)
    else:
        hue_factor = None

    # Apply augmentations in the requested order.
    aug_images = images
    for op in aug_order:
        if op == "random_resized_crop" and rrc_params is not None:
            i, j, h, w = rrc_params
            new_seq = []
            for img in aug_images:
                img = F.resized_crop(
                    img,
                    top=i,
                    left=j,
                    height=h,
                    width=w,
                    size=image_size,
                    interpolation=InterpolationMode.BILINEAR,
                )
                new_seq.append(img)
            aug_images = new_seq

        elif op == "random_brightness" and brightness_factor is not None:
            new_seq = []
            for img in aug_images:
                img = F.adjust_brightness(img, brightness_factor)
                new_seq.append(img)
            aug_images = new_seq

        elif op == "random_contrast" and contrast_factor is not None:
            new_seq = []
            for img in aug_images:
                img = F.adjust_contrast(img, contrast_factor)
                new_seq.append(img)
            aug_images = new_seq

        elif op == "random_saturation" and saturation_factor is not None:
            new_seq = []
            for img in aug_images:
                img = F.adjust_saturation(img, saturation_factor)
                new_seq.append(img)
            aug_images = new_seq

        elif op == "random_hue" and hue_factor is not None:
            new_seq = []
            for img in aug_images:
                img = F.adjust_hue(img, hue_factor)
                new_seq.append(img)
            aug_images = new_seq

        else:
            # Unknown op or missing parameters; skip it.
            continue

    # If random_resized_crop was not applied, resize the sequence at the end.
    if rrc_params is None:
        aug_images = [
            F.resize(img, image_size, interpolation=InterpolationMode.BILINEAR)
            for img in aug_images
        ]

    if return_geom:
        return aug_images, {"src_hw": src_hw, "out_hw": out_hw, "crop_params": rrc_params}
    return aug_images


def transform_query_points(
    query_points: np.ndarray,
    src_hw: Tuple[int, int],
    out_hw: Tuple[int, int],
    crop_params: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Map query points from original-frame pixel coordinates to augmented-frame
    pixel coordinates. Coordinates use (x, y) order.
    """
    q = np.asarray(query_points, dtype=np.float32).copy()
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    out_h, out_w = int(out_hw[0]), int(out_hw[1])

    if crop_params is not None:
        top, left, crop_h, crop_w = [int(v) for v in crop_params]
        q[:, 0] = (q[:, 0] - float(left)) * (float(out_w) / max(float(crop_w), 1.0))
        q[:, 1] = (q[:, 1] - float(top)) * (float(out_h) / max(float(crop_h), 1.0))
    else:
        q[:, 0] = q[:, 0] * (float(out_w) / max(float(src_w), 1.0))
        q[:, 1] = q[:, 1] * (float(out_h) / max(float(src_h), 1.0))

    q[:, 0] = np.clip(q[:, 0], 0.0, max(float(out_w - 1), 0.0))
    q[:, 1] = np.clip(q[:, 1], 0.0, max(float(out_h - 1), 0.0))
    return q


def normalize_query_points(query_points: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Normalize pixel coordinates to [0, 1]."""
    q = np.asarray(query_points, dtype=np.float32).copy()
    h, w = int(hw[0]), int(hw[1])
    if w > 1:
        q[:, 0] = q[:, 0] / float(w - 1)
    else:
        q[:, 0] = 0.0
    if h > 1:
        q[:, 1] = q[:, 1] / float(h - 1)
    else:
        q[:, 1] = 0.0
    q = np.clip(q, 0.0, 1.0)
    return q

def aug_query_points(
    q,                 # [N,2] float32 in pixel coords
    H=256, W=256,
    global_shift_std=0.05,   # pixels
    local_jitter_std=0.05,   # pixels
    p_outlier=0.02,
    do_shuffle=True,
    p_shuffle=0.05,
    enforce_change=False,
):
    q = q.copy()

    # (1) global + local jitter
    g = np.random.normal(scale=global_shift_std, size=(1, 2)).astype(np.float32)
    n = np.random.normal(scale=local_jitter_std, size=q.shape).astype(np.float32)
    q = q + g + n

    # (2) optional: inject a few outliers
    if p_outlier > 0:
        N = q.shape[0]
        out = (np.random.rand(N) < p_outlier)
        if out.any():
            q[out, 0] = np.random.uniform(0, W - 1, size=out.sum()).astype(np.float32)
            q[out, 1] = np.random.uniform(0, H - 1, size=out.sum()).astype(np.float32)

    # clamp
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)

    perm = None

    # (3) partial shuffle order
    if do_shuffle and p_shuffle > 0:
        N = q.shape[0]
        perm = np.arange(N)

        mask = (np.random.rand(N) < p_shuffle)
        idx = np.where(mask)[0]

        if idx.size > 1:
            shuffled = idx.copy()
            np.random.shuffle(shuffled)

            if enforce_change:
                # Try to avoid fixed points; this may be impossible for tiny subsets.
                for _ in range(10):
                    if not np.any(shuffled == idx):
                        break
                    np.random.shuffle(shuffled)
                if np.any(shuffled == idx):
                    shuffled = np.roll(idx, 1)

            perm[idx] = shuffled
            q = q[perm]

    return q, perm




class FutureKDataset(Dataset):
    """
    Load data from *_tracks.hdf5 files instead of per-directory .npy files.

    Expected layout:
        flow_root/
          libero_object/
            <task_name>_demo/
              <task_name>_demo_tracks.hdf5
              <task_name>_demo_demo_xx/  (original per-demo directory, optional)

    HDF5 layout:
        <task_name>_demo_tracks.hdf5
          / (root)
            attrs: task_text, source_hdf5, image_key, test_demo_id, ...
            /data
              /0
                attrs: prompt, has_tracks
                datasets:
                  point_traj   (T, N, 3)
                  vis          (T, N)
                  frames_rgb   (T, H, W, 3)
                  vggt_hidden  (T, L, D) or (T, D)
                  query_points     (N, 2)  # current-state query points (q_cur)
    """
    def __init__(self, flow_root: str, k_steps: int, num_points: int,
                 mapping_csv: Optional[str] = None,
                 text_col: Optional[str] = None,
                 prompt_col: Optional[str] = None,
                 image_size: int = 224, cond_kframes: int = 4,
                 cond_stride: int = 1, img_aug: bool = True,
                 query_aug: bool = True,
                 traj_key: str = "point_traj_metric",
                 task_type: str = "auto",
                 plus_is_close: str = "auto",
                 gripper_debounce: int = 3,
                 min_seg_len: int = 10,
                 keep_last_segment: bool = True):
        super().__init__()
        self.root = Path(flow_root)
        self.k = k_steps
        self.n = num_points
        self.traj_key = str(traj_key)
        self.task_type = str(task_type).lower()
        self.plus_is_close = str(plus_is_close).lower()
        self.image_size = image_size
        self.processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-224", use_fast=True
            # "google/siglip-large-patch16-384", use_fast=True
            # "google/siglip-base-patch16-384", use_fast=True
        )

        self.cond_k = cond_kframes
        self.cond_stride = cond_stride

        self.img_aug = img_aug
        # Only augment / permute query points when the model actually receives them.
        # Otherwise permuting F_win corrupts point-index supervision.
        self.query_aug = bool(query_aug)
        self.image_augment_kwargs = dict(
            random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.05],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        )

        # Per-worker HDF5 file-handle cache.
        self._file_handles: Dict[str, h5py.File] = {}
        self.samples: List[Tuple[str, str, int, int]] = []  # (h5_path, demo_id, seg_start, seg_end)

        self.min_seg_len = int(min_seg_len)
        self.gripper_debounce = int(gripper_debounce)
        self.keep_last_segment = bool(keep_last_segment)

        # Sampling parameters between keyframes.
        self.gamma_s = 1.2
        self.gamma_e = 1.6
        self.min_unique_ratio = 0.7

        # Only scan processed track files to avoid large raw demo HDF5 files.
        h5_files = sorted(self.root.rglob("*_tracks.hdf5"))
        if len(h5_files) == 0:
            raise RuntimeError(f"No *_tracks.hdf5 files found under {self.root}")

        for h5_path in h5_files:
            # Some files may be locked; skip them.
            try:
                f = h5py.File(h5_path, "r")
            except (BlockingIOError, OSError) as e:
                print(f"[WARN] Cannot open {h5_path}: {e}. Skipping this file.")
                continue

            with f:
                if "data" not in f:
                    continue
                data_grp = f["data"]
                h5_task_type = self._task_type_for_file(h5_path)
                h5_plus_is_close = resolve_plus_is_close(h5_task_type, self.plus_is_close)
                for demo_id, grp in data_grp.items():
                    # Skip demos explicitly marked as missing tracks.
                    if grp.attrs.get("has_tracks", True) is False:
                        continue
                    if self.traj_key not in grp:
                        continue

                    T_len, N_tot, _ = grp[self.traj_key].shape

                    g = self._load_gripper_signal(grp)            # (T,)
                    segments = self.seg_gripper_state(
                        g,
                        T_len,
                        plus_is_close=h5_plus_is_close,
                    )   # [(s0,s1),...]

                    anchor_stride = 2          # Use 1 for every frame; use larger values to reduce samples.
                    min_remain = 2             # Require at least two frames from current anchor to target.
                    for s0, s1 in segments:
                        seg_len = (s1 - s0 + 1)
                        if seg_len < self.min_seg_len:
                            continue

                        # Sample multiple anchors inside each segment.
                        for s in range(s0, s1, anchor_stride):   # Up to s1-1.
                            if (s1 - s + 1) < min_remain:
                                continue
                            self.samples.append((str(h5_path), str(demo_id), int(s), int(s1)))

        assert len(self.samples) > 0, "No valid windows for K-step future prediction."

        print(
            f"[DATA] traj_key={self.traj_key} task_type={self.task_type} "
            f"plus_is_close={self.plus_is_close} samples={len(self.samples)}"
        )

    def _task_type_for_file(self, h5_path: Path) -> str:
        if self.task_type != "auto":
            return self.task_type
        return infer_task_type_from_path(h5_path)

    def _h5_take_time(self, ds, idxs, dtype=np.float32):
        """
        Read a h5py dataset at time indices idxs.
        Supports non-strictly-increasing or repeated indices, which h5py cannot read directly.
        """
        idxs = np.asarray(idxs, dtype=np.int64)

        # Clip to the valid range.
        T = ds.shape[0]
        idxs = np.clip(idxs, 0, T - 1)

        # Ensure non-decreasing order; time should be monotonic.
        idxs = np.maximum.accumulate(idxs)

        # h5py only accepts strictly increasing indices; read unique indices first.
        uniq = np.unique(idxs)  # sorted & unique & increasing
        buf = np.asarray(ds[uniq], dtype=dtype)  # (len(uniq), ...)

        # Restore repeats/order by mapping each idx back into uniq.
        pos = np.searchsorted(uniq, idxs)
        out = buf[pos]
        return out


    def _load_gripper_signal(self, grp) -> np.ndarray:
        if "actions" not in grp:
            raise KeyError(f"'actions' not found in group. keys={list(grp.keys())}")
        a = np.asarray(grp["actions"][:], dtype=np.float32)  # (T,A)
        if a.ndim != 2:
            raise ValueError(f"actions must be (T,A), got {a.shape}")
        return a[:, -1]  # (T,)

    def _binarize_gripper(self, g: np.ndarray, plus_is_close: bool) -> np.ndarray:
        """
        Convert the last action dimension into a binary gripper state.
        LIBERO usually uses positive-as-close; ManiSkill can be opposite.
        The midpoint threshold also handles 0/1-style gripper commands.
        """
        g = np.nan_to_num(g.astype(np.float32))
        if g.size == 0:
            return np.zeros((0,), dtype=np.int32)
        thr = 0.5 * (float(np.nanmin(g)) + float(np.nanmax(g)))
        if bool(plus_is_close):
            return (g > thr).astype(np.int32)
        return (g < thr).astype(np.int32)

    def _debounce_changes(self, gb: np.ndarray, debounce: int) -> list[int]:
        # Debounce state changes: require the new state for several consecutive frames.
        if debounce <= 1:
            return (np.where(np.diff(gb) != 0)[0] + 1).astype(int).tolist()

        T = len(gb)
        cur = gb[0]
        changes = []
        i = 1
        while i < T:
            if gb[i] == cur:
                i += 1
                continue
            new = gb[i]
            ok = True
            for j in range(i, min(T, i + debounce)):
                if gb[j] != new:
                    ok = False
                    break
            if ok:
                changes.append(i)  # The new state starts at i.
                cur = new
                i = i + debounce
            else:
                i += 1
        return changes
    
    def seg_gripper_state(self, g: np.ndarray, T_len: int, plus_is_close: bool) -> list[tuple[int, int]]:
        """
        Segment trajectory by gripper command changes.

        IMPORTANT semantics:
        - change_idx means the NEW state starts at idx.
        - Therefore, the previous segment should end at idx-1 (inclusive).

        Returns:
            segments: list of (seg_start, seg_end_inclusive)
        """
        assert len(g) == T_len
        gb = self._binarize_gripper(g, plus_is_close=plus_is_close)

        debounce = int(getattr(self, "gripper_debounce", 3))
        change_idxs = self._debounce_changes(gb, debounce)  # NEW state starts at idx

        # Build boundaries in end-exclusive form first: [b0, b1, ..., T_len]
        boundaries = [0] + change_idxs + [T_len]  # NOTE: last is T_len (exclusive), not T_len-1
        boundaries = sorted(set(int(x) for x in boundaries))

        # Ensure strictly increasing
        b2 = [boundaries[0]]
        for b in boundaries[1:]:
            if b > b2[-1]:
                b2.append(b)
        boundaries = b2

        keep_last = bool(getattr(self, "keep_last_segment", False))

        segments: List[Tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            s0 = boundaries[i]
            s1_excl = boundaries[i + 1]  # segment is [s0, s1_excl)

            if s1_excl <= s0:
                continue

            # Optionally drop the last segment (tail)
            if (i == len(boundaries) - 2) and (not keep_last) and (len(change_idxs) > 0):
                continue

            s1_incl = s1_excl - 1
            if s1_incl < s0:
                continue

            # Filter short segments (inclusive length)
            if (s1_incl - s0 + 1) >= int(self.min_seg_len):
                segments.append((int(s0), int(s1_incl)))

        # If no change at all: whole traj is one segment
        if len(change_idxs) == 0:
            segments = [(0, T_len - 1)]

        return segments


    def resample_on_traj(self, start: int, end: int, num: int) -> np.ndarray:
        """
        Densify both endpoints with a gamma warp:
        - gamma_s controls density near the start.
        - gamma_e controls density near the end.
        gamma=1 gives uniform sampling.
        """
        if end <= start:
            return np.array([start] * num, dtype=np.int64)

        gamma_s = self.gamma_s
        gamma_e = self.gamma_e
        min_unique = self.min_unique_ratio

        u = np.linspace(0.0, 1.0, num=num, dtype=np.float64)

        # U-shaped warp: denser near both endpoints, sparser in the middle.
        w = np.empty_like(u)
        left = (u <= 0.5)
        w[left]  = 0.5 * np.power(2.0 * u[left], gamma_s)
        w[~left] = 1.0 - 0.5 * np.power(2.0 * (1.0 - u[~left]), gamma_e)

        idxs = start + (end - start) * w
        idxs = np.round(idxs).astype(np.int64)

        idxs[0] = start
        idxs[-1] = end
        idxs = np.maximum.accumulate(idxs)  # Non-decreasing; repeats are allowed.

        # Fall back to uniform sampling if too many repeats would weaken supervision.
        if np.unique(idxs).size < int(min_unique * num):
            idxs = np.round(np.linspace(start, end, num=num)).astype(np.int64)
            idxs[0] = start
            idxs[-1] = end
            idxs = np.maximum.accumulate(idxs)

        return idxs



    # Keep HDF5 handles process-local for DataLoader workers.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handles"] = None   # Let each worker reopen its own HDF5 handles.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = {}

    def _get_file(self, path: str) -> h5py.File:
        """Cache HDF5 file handles per process to avoid repeated open/close calls."""
        if self._file_handles is None:
            self._file_handles = {}
        fh = self._file_handles.get(path)
        if fh is None:
            fh = h5py.File(path, "r")
            self._file_handles[path] = fh
        return fh

    def __del__(self):
        if getattr(self, "_file_handles", None):
            for fh in self._file_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass

    # Convert stored frames to HWC uint8.
    def _to_rgb_hwc(self, frame: np.ndarray) -> np.ndarray:
        """Robustly convert any frame to HWC uint8 RGB."""
        a = np.asarray(frame)
        a = np.squeeze(a)

        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3:
            if a.shape[-1] in (3, 4):
                a = a[..., :3]
            elif a.shape[0] in (3, 4):
                a = np.moveaxis(a, 0, -1)[..., :3]
            else:
                a2 = np.squeeze(a)
                if a2.ndim == 2:
                    a = np.stack([a2, a2, a2], axis=-1)
                else:
                    raise ValueError(f"Unexpected image shape after squeeze: {a.shape}")
        else:
            raise ValueError(f"Unexpected image shape: {a.shape}")

        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        return a

    def _load_cond_frames(self, frames_ds, s: int) -> List[Image.Image]:
        """
        Load cond_k frames from frames_rgb.
        Frames are sampled backward from s with cond_stride and padded at the front if needed.
        """
        T = frames_ds.shape[0]
        idxs = list(range(
            max(0, s - (self.cond_k - 1) * self.cond_stride),
            s + 1,
            self.cond_stride,
        ))
        if len(idxs) > self.cond_k:
            idxs = idxs[-self.cond_k:]
        while len(idxs) < self.cond_k:
            idxs.insert(0, idxs[0])

        imgs: List[Image.Image] = []
        for idx in idxs:
            idx_clamped = min(idx, T - 1)
            arr = self._to_rgb_hwc(frames_ds[idx_clamped])
            imgs.append(Image.fromarray(arr))
        return imgs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # === sample: (h5, demo_id, seg_start, seg_end) ===
        sample = self.samples[i]
        h5_path_str, demo_id, seg_start, seg_end = sample
        seg_start, seg_end = int(seg_start), int(seg_end)

        f = self._get_file(h5_path_str)
        grp = f["data"][demo_id]

        P_ds = grp[self.traj_key]  # (T, N, 3)
        T_len, N_tot, _ = P_ds.shape

        seg_start = max(0, min(seg_start, T_len - 1))
        seg_end   = max(0, min(seg_end,   T_len - 1))
        if seg_end <= seg_start:
            seg_end = min(seg_start + 1, T_len - 1)

        key_idxs = self.resample_on_traj(seg_start, seg_end, num=self.k).astype(np.int64)
        key_idxs = np.clip(key_idxs, 0, T_len - 1)
        F_win = self._h5_take_time(P_ds, key_idxs, dtype=np.float32)  # (K, N_tot, 3)

        if "vis" in grp:
            vis_ds = grp["vis"]  # (T, N)
            vis_key = self._h5_take_time(vis_ds, key_idxs, dtype=np.float32)  # (K, N_tot)
            W_win = (vis_key > 0.5).astype(np.float32)  # (K, N_tot)
        else:
            W_win = np.ones((self.k, F_win.shape[1]), dtype=np.float32)

        vggt_ds = grp["vggt_hidden"]  # (T, L, D) or (T, D)
        vggt_hidden = np.asarray(vggt_ds[seg_start:seg_start + 1], dtype=np.float32)  # [1, ...]

        frames_ds = grp["frames_rgb"]  # (T, H, W, 3)
        img_seq = self._load_cond_frames(frames_ds, seg_start)

        if "track2d" not in grp:
            raise KeyError(f"'track2d' not found in group. keys={list(grp.keys())}")
        current_query_points = np.asarray(grp["track2d"][seg_start], dtype=np.float32)[..., :2]

        img_seq, geom = augment_image_sequence_rlds_style(
            img_seq,
            image_size=self.image_size,
            train=bool(self.img_aug),
            image_augment_kwargs=self.image_augment_kwargs,
            return_geom=True,
        )
        current_query_points = transform_query_points(
            current_query_points,
            src_hw=geom["src_hw"],
            out_hw=geom["out_hw"],
            crop_params=geom["crop_params"],
        )

        task_dir_name = Path(h5_path_str).parent.name
        norm_key = normalize_instr_key(task_dir_name)
        instruction = norm_key

        if self.query_aug:
            out_h, out_w = geom["out_hw"]
            current_query_points, perm = aug_query_points(current_query_points, H=out_h, W=out_w)
            if perm is not None:
                F_win = F_win[:, perm, :]
                W_win = W_win[:, perm]

        current_query_points = normalize_query_points(current_query_points, geom["out_hw"])

        return {
            "images": img_seq,  # List[PIL], len = cond_k
            "instruction": instruction,
            "query_points": torch.from_numpy(current_query_points).float(),  # (N,2) normalized q_cur
            "clean_flow": torch.from_numpy(F_win).float(),                   # (K,N,3) keyframe flow
            "vggt_hidden": torch.from_numpy(vggt_hidden).float(),            # [1,...]
            "weights": torch.from_numpy(W_win).float().squeeze(),                      # (K,N)
        }

# Data
def compute_step_flows(point_traj: np.ndarray) -> np.ndarray:
    # (T,N,3) -> (T,N,3) with ΔP_0 = 0
    return np.diff(point_traj, axis=0, prepend=point_traj[0:1])

# Number of discrete forward-diffusion timesteps. Inference can use fewer DDIM/DPM steps by sampling a subsequence.
# --------------------------
# Diffusion schedule (with sqrt caches for v-pred & SNR) 
# --------------------------
# Forward noising schedule for the diffusion model.
class SimpleDiffusionNoiseSchedule:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.10):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None):
        # x0: [B,K,N,3]; t: [B]
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod.to(x0.device)[t].view(-1, 1, 1, 1)  # [B,1,1,1]
        noisy = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return noisy, noise


@torch.no_grad()
def ddim_sample_vpred(
    model: nn.Module,
    schedule: SimpleDiffusionNoiseSchedule,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    query_points: Optional[torch.Tensor],
    shape: Tuple[int, int, int, int],
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Deployment-style validation sampler: start from Gaussian noise and run DDIM.
    The model uses v-prediction, so x0/eps are recovered analytically at each step.
    """
    x_t = torch.randn(shape, device=device, generator=generator)
    a_bar = schedule.alphas_cumprod.to(device)
    ts = torch.linspace(
        schedule.num_train_timesteps - 1,
        0,
        steps=max(int(num_inference_steps), 1),
        device=device,
    ).long()

    B = shape[0]
    for i, t in enumerate(ts):
        t_vec = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

        if guidance_scale is None or float(guidance_scale) == 1.0:
            v_hat = model(
                image_pixels=pixel_values,
                instruction_input_ids=input_ids,
                instruction_attention_mask=attn_mask,
                query_points=query_points,
                noisy_flow=x_t,
                timestep=t_vec,
                drop_condition_mask=torch.zeros(B, dtype=torch.bool, device=device),
            )
        else:
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_vec, t_vec], dim=0)
            pv_in = torch.cat([pixel_values, pixel_values], dim=0)
            ids_in = torch.cat([input_ids, input_ids], dim=0)
            att_in = torch.cat([attn_mask, attn_mask], dim=0)
            qp_in = torch.cat([query_points, query_points], dim=0) if query_points is not None else None
            drop_mask = torch.cat([
                torch.zeros(B, dtype=torch.bool, device=device),
                torch.ones(B, dtype=torch.bool, device=device),
            ], dim=0)

            v_out = model(
                image_pixels=pv_in,
                instruction_input_ids=ids_in,
                instruction_attention_mask=att_in,
                query_points=qp_in,
                noisy_flow=x_in,
                timestep=t_in,
                drop_condition_mask=drop_mask,
            )
            v_cond, v_uncond = v_out.chunk(2, dim=0)
            v_hat = v_uncond + float(guidance_scale) * (v_cond - v_uncond)

        s = torch.sqrt(a_bar[t])
        c = torch.sqrt(1.0 - a_bar[t])
        x0 = s * x_t - c * v_hat
        eps = c * x_t + s * v_hat

        if i < len(ts) - 1:
            t_next = ts[i + 1]
            s_next = torch.sqrt(a_bar[t_next])
            c_next = torch.sqrt(1.0 - a_bar[t_next])
            x_t = s_next * x0 + c_next * eps
        else:
            x_t = x0

    return x_t


def flow_mse_epe3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = pred - target
    mse_map = diff.pow(2).mean(dim=-1)
    epe_map = diff.pow(2).sum(dim=-1).sqrt()

    if weights is not None:
        w = weights.to(dtype=mse_map.dtype)
        denom = w.sum(dim=(1, 2)).clamp_min(1e-8)
        mse_b = (mse_map * w).sum(dim=(1, 2)) / denom
        epe_b = (epe_map * w).sum(dim=(1, 2)) / denom
    else:
        mse_b = mse_map.view(mse_map.size(0), -1).mean(dim=1)
        epe_b = epe_map.view(epe_map.size(0), -1).mean(dim=1)

    return mse_b.mean(), epe_b.mean()


def collate_fn(batch, processor: SiglipProcessor):
    txts = [b["instruction"] for b in batch]
    proc_txt = processor(text=txts, return_tensors="pt", padding=True)
    input_ids = proc_txt["input_ids"]
    attn_mask = proc_txt.data.get("attention_mask", None)
    if attn_mask is None:
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
        attn_mask = (input_ids != pad_id).long() if pad_id is not None else torch.ones_like(input_ids)
    
    vggt_hidden = torch.stack([b["vggt_hidden"] for b in batch], dim=0)  # [B, L_vggt, D_vggt]

    if "images" in batch[0]:
        seqs = [b["images"] for b in batch]   # List[List[PIL]]
        B, Kc = len(seqs), len(seqs[0])
        flat_imgs = [im for seq in seqs for im in seq]
        proc_img = processor(images=flat_imgs, return_tensors="pt")
        pv = proc_img["pixel_values"]                # [B*Kc, 3, H, W]
        pixel_values = pv.view(B, Kc, *pv.shape[1:])
    else:
        imgs = [b["image"] for b in batch]
        proc = processor(images=imgs, return_tensors="pt")
        pixel_values = proc["pixel_values"]          # [B, 3, H, W]

    query_points = torch.stack([b["query_points"] for b in batch], dim=0)  # [B, N, 2] normalized q_cur
    clean_flow   = torch.stack([b["clean_flow"]   for b in batch], dim=0)  # [B, K, N, 3]
    weights      = torch.stack([b["weights"]      for b in batch], dim=0)  # [B, K, N]

    return pixel_values, input_ids, attn_mask, query_points, clean_flow, weights, vggt_hidden

# --------------------------
# Train
# --------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_root", type=str, required=True)
    parser.add_argument(
        "--traj_key",
        type=str,
        default="point_traj_metric",
        choices=("point_traj", "point_traj_metric"),
        help="HDF5 trajectory key used for flow supervision.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        choices=("auto", "libero", "maniskill"),
        help="Dataset family used for gripper-stage detection. auto infers LIBERO/ManiSkill from the HDF5 path.",
    )
    parser.add_argument(
        "--plus_is_close",
        type=str,
        default="auto",
        choices=("auto", "true", "false"),
        help="Override gripper sign convention. auto uses LIBERO=true, ManiSkill=false.",
    )
    parser.add_argument("--mapping_csv", type=str, default=None, help="Optional CSV for instructions/prompts")
    parser.add_argument("--text_col", type=str, default="instruction")
    parser.add_argument("--prompt_col", type=str, default="prompt")

    parser.add_argument("--k_steps", type=int, default=20, help="Number of future flow keyframes")
    parser.add_argument("--num_points", type=int, default=100, help="Number of flow points")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Disable clipping with a non-positive value")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--final_lr", type=float, default=5e-6)
    parser.add_argument("--anneal_lr", action="store_true", help="Enable learning rate annealing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./ckpts_improved")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Optional checkpoint to initialize / resume from. Leave unset to train from scratch.")

    # New training knobs
    parser.add_argument("--p_uncond", type=float, default=0.2, help="CFG training: probability to drop condition per-sample")
    parser.add_argument("--snr_gamma", type=float, default=0.0, help="SNR weighting exponent (0.0 disables)")  # 0.5

    parser.add_argument("--train_noise_timesteps", type=int, default=100, help="Adding noise steps during training")
    parser.add_argument("--val_sampling", action="store_true",
                        help="Run deployment-style validation by DDIM denoising from pure noise")
    parser.add_argument("--val_sampling_steps", type=int, default=20,
                        help="DDIM steps for --val_sampling")
    parser.add_argument("--val_sampling_guidance_scale", type=float, default=2.0,
                        help="CFG guidance scale for --val_sampling; set 1.0 to disable CFG")
    parser.add_argument("--val_sampling_max_batches", type=int, default=0,
                        help="Max val batches for --val_sampling; 0 means all validation batches")
    parser.add_argument("--val_sampling_seed", type=int, default=0,
                        help="Fixed seed for deployment-style validation noise")

    parser.add_argument("--cond_kframes", type=int, default=4, help="History frames for conditioning image")
    parser.add_argument("--cond_stride",  type=int, default=1, help="History frame stride for conditioning image")

    parser.add_argument("--vggt_feature", action="store_true", help="Enable vggt feature alignment")

    parser.add_argument("--align_weight", type=float, default=0.5, help="Total loss = flow loss + align_weight * align_loss")
    parser.add_argument("--align_lr", type=float, default=1e-4, help="Learning rate for AlignProjector; defaults to --lr when unset.")

    parser.add_argument("--motion_module", action="store_true", help="Enable motion_module")
    parser.add_argument("--query_points", action="store_true", help="Enable initial query_points")

    # add smoothness
    parser.add_argument("--smooth2_weight", type=float, default=1e-3,
                        help="Second-order smoothness loss weight. Set 0 to disable.")
    parser.add_argument("--smooth2_eps", type=float, default=1e-3,
                        help="Charbonnier epsilon for the smoothness loss.")

    # --- [W&B] args ---
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="3DFlowPrediction", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_dir", type=str, default=None, help="W&B local dir for logs")


    # --- [DDP] args ---
    parser.add_argument("--local_rank", type=int, default=-1, help="For torchrun/legacy launch compatibility")
    parser.add_argument("--ddp_backend", type=str, default=None, help="DDP backend (default: nccl if cuda else gloo)")
    parser.add_argument("--ddp_find_unused", action=argparse.BooleanOptionalAction, default=True,
                    help="DDP find_unused_parameters (default: True)")
    parser.add_argument("--sync_bn", action="store_true", help="Convert BatchNorm to SyncBatchNorm")

    args = parser.parse_args()
    align_weight = args.align_weight
    align_lr = args.align_lr

    args.wandb_run = f"Flow_k{args.k_steps}_n{args.num_points}_traj{args.traj_key}_task{args.task_type}_lr{args.lr}_final_lr{args.final_lr}_bs{args.batch_size}_align_weight{args.align_weight}_p_uncond{args.p_uncond}_snr_gamma{args.snr_gamma}_cond_kframes{args.cond_kframes}_train_noise_timesteps{args.train_noise_timesteps}_motion_module{args.motion_module}_query_points{args.query_points}" \


    init_distributed_mode(args)

    # Seed for deterministic dataset split (same across ranks)
    set_seed(args.seed)
    enable_sdpa()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda", args.local_rank) if use_cuda else torch.device("cpu")
    device_type = device.type


    ds = FutureKDataset(args.flow_root, args.k_steps, args.num_points,
                        mapping_csv=args.mapping_csv, text_col=args.text_col, prompt_col=args.prompt_col,
                        image_size=256, cond_kframes=args.cond_kframes, cond_stride=args.cond_stride,
                        query_aug=args.query_points,
                        traj_key=args.traj_key,
                        task_type=args.task_type,
                        plus_is_close=args.plus_is_close)

    n_train = int(len(ds) * 0.98); n_val = len(ds) - n_train
    train_set, val_set = torch.utils.data.random_split(ds, [n_train, n_val])


    # Different RNG streams per rank (data aug / dropout etc.)
    if getattr(args, "distributed", False):
        set_seed(args.seed + args.rank)

    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224", use_fast=True)
    
    collate = lambda b: collate_fn(b, processor)

    # --- DataLoader (supports DDP) ---
    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_set,   shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate,
    )

    # Model: DiT backbone; model outputs v
    def count_params(module: torch.nn.Module): # count total parameters & trainable parameters
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def human_m(x):  # Display parameter counts in millions.
        return x / 1e6


    cfg = {"model_dim": 768, "num_layers": 10, "num_heads": 12}
    model = model_siglip.GenerativeFlowModel(k_steps=args.k_steps, num_points=args.num_points, **cfg).to(device)
    if args.load_path:
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[INFO] Initialized flow model from {args.load_path}")
    else:
        print("[INFO] Training flow model from scratch (no --load_path provided).")

    tot, trn = count_params(model)
    print(f"Total parameters:     {human_m(tot):8.2f} M")
    print(f"Trainable parameters: {human_m(trn):8.2f} M")
    print(f"Frozen ratio:         {100.0 * (1.0 - trn / max(tot,1)):6.2f}%")


    for name, sub in unwrap_model(model).named_children():
        t, r = count_params(sub)
        print(f"[{name:>12}] total: {human_m(t):8.2f} M | trainable: {human_m(r):8.2f} M | frozen: {100.0 * (1.0 - r / max(t,1)):6.2f}%")

    # --- [DDP] wrap model ---
    if getattr(args, "distributed", False):
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=bool(args.ddp_find_unused),
        )

    # --- [W&B] init ---
    wb = None
    if args.wandb and is_main_process():
        assert wandb is not None, "wandb not installed. pip install wandb"
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config=vars(args),
        )
        wandb.summary["params_trainable"] = int(human_m(tot))
    print(model)
    schedule = SimpleDiffusionNoiseSchedule(num_train_timesteps=args.train_noise_timesteps) # 1000

    from torch.amp import GradScaler, autocast
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-2)
    align_tok = None  # Initialize after the first batch reveals feature dimensions.

    fp16_on = bool(args.fp16 and device.type == "cuda")
    scaler = GradScaler(enabled=fp16_on)

    best_val = 1e9
    print("Start training: future-K prediction (v-pred + SNR-weighted + CFG)")
    total_updates = len(train_loader) * args.epochs // args.accum_steps
    num_updates = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if getattr(args, "distributed", False) and 'train_sampler' in locals() and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_main_div = 0.0     # Main v-pred loss only.
        total_align_div = 0.0    # Alignment loss only.
        total_smooth_div = 0.0
        total_all_div = 0.0      # Total loss, including auxiliary terms.
        micro = 0
        epoch_updates = 0
        optim.zero_grad(set_to_none=True)
        for micro, (pixel_values, input_ids, attn_mask, query_points, clean_flow, weights, vggt_hidden) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            input_ids    = input_ids.to(device, non_blocking=True)
            attn_mask    = attn_mask.to(device, non_blocking=True)
            query_points = query_points.to(device, non_blocking=True)   # [B,K,N,3]
            clean_flow   = clean_flow.to(device, non_blocking=True)     # [B,K,N,3]
            weights      = weights.to(device, non_blocking=True)        # [B,K,N]
            vggt_hidden  = vggt_hidden.to(device, non_blocking=True)    # [B, L_vggt, D_vggt]  torch.Size([B, 1369, 2048])

            B = clean_flow.size(0)
            t = torch.randint(0, schedule.num_train_timesteps, (B,), device=device).long()
            noisy_flow, noise = schedule.add_noise(clean_flow, t)

            s = schedule.sqrt_alphas_cumprod.to(device)[t].view(B,1,1,1)
            c = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t].view(B,1,1,1)
            v_target = s * noise - c * clean_flow  # [B,K,N,3]

            drop_mask = (torch.rand(B, device=device) < args.p_uncond)

            align_loss = torch.zeros([], device=device)   # Scalar tensor that can enter the graph.
            if args.vggt_feature:
                features_3d = unwrap_model(model).get_features_3d(pixel_values)  # torch.Size([4, 1024])

                # Initialize the auxiliary alignment head after the first batch reveals feature sizes.
                if align_tok is None:
                    llm_dim  = features_3d.shape[-1]
                    vggt_dim = vggt_hidden.shape[-1] // 2
                    align_tok = AlignProjector(
                        llm_dim=llm_dim,
                        vggt_dim=vggt_dim,
                        use_vlm_norm=True,
                    ).to(device)
                    if getattr(args, "distributed", False):
                        align_tok = DDP(
                            align_tok,
                            device_ids=[args.local_rank] if device.type == "cuda" else None,
                            output_device=args.local_rank if device.type == "cuda" else None,
                            broadcast_buffers=False,
                            find_unused_parameters=False,
                        )
                    optim.add_param_group({"params": align_tok.parameters(), "lr": align_lr})
                
                align_loss = align_tok(features_3d, vggt_hidden)

            with autocast(device_type=device_type, dtype=torch.float16, enabled=fp16_on):
                pred_v = model(
                    image_pixels=pixel_values,
                    instruction_input_ids=input_ids,
                    instruction_attention_mask=attn_mask,
                    query_points=(query_points if args.query_points else None),
                    noisy_flow=noisy_flow, timestep=t,
                    drop_condition_mask=drop_mask
                )  # [B,K,N,3]

                err = (pred_v - v_target).pow(2).mean(dim=-1)        # [B,K,N]
                if weights is not None:
                    err = err * weights.to(err.dtype)
                    denom = (weights.sum(dim=(1,2)) + 1e-8)          # [B]
                else:
                    denom = torch.full((B,), fill_value=clean_flow.size(1)*clean_flow.size(2), device=device)
                loss_b = err.sum(dim=(1,2)) / denom                  # [B]

                # Optional SNR reweighting.
                if args.snr_gamma > 0:
                    ab = schedule.alphas_cumprod.to(device)[t]       # [B]
                    snr = (ab / (1 - ab)).clamp(1e-3, 50.0)
                    w_snr = snr.pow(args.snr_gamma)
                    loss_main = (w_snr * loss_b).mean()
                else:
                    loss_main = loss_b.mean()

                smooth_loss = torch.zeros([], device=device)

                if args.smooth2_weight > 0:
                    x0_hat = s * noisy_flow - c * pred_v   # [B,K,N,3]
                    if weights is not None:
                        w = weights.to(x0_hat.dtype)
                    else:
                        w = None

                    smooth_loss = second_order_smooth_loss(
                        x0_hat.float(),                     # Compute in float32 for stability.
                        None if w is None else w.float(),
                        eps=args.smooth2_eps,
                        all3_visible=True,
                    )

                loss_total = loss_main + align_weight * align_loss + args.smooth2_weight * smooth_loss

                # Divide before gradient accumulation.
                loss_divided_main  = loss_main  / max(args.accum_steps, 1)  # Main-loss statistic.
                loss_divided_align = (align_weight * align_loss) / max(args.accum_steps, 1)  # NEW
                loss_divided_smooth = (args.smooth2_weight * smooth_loss) / max(args.accum_steps, 1)  # NEW
                loss_divided_total = loss_total / max(args.accum_steps, 1)  # Used for backward/step.

            # Backward pass.
            # DDP + grad-accum: avoid gradient all-reduce on non-sync micro-steps
            do_sync = True
            if getattr(args, "distributed", False) and hasattr(model, "no_sync"):
                is_last_batch = (micro == len(train_loader))
                do_sync = (micro % args.accum_steps == 0) or is_last_batch

            if getattr(args, "distributed", False) and hasattr(model, "no_sync") and (not do_sync):
                # no_sync on the main model; align_tok (if wrapped) is tiny so optional
                if align_tok is not None and isinstance(align_tok, DDP):
                    with model.no_sync(), align_tok.no_sync():
                        scaler.scale(loss_divided_total).backward()
                else:
                    with model.no_sync():
                        scaler.scale(loss_divided_total).backward()
            else:
                scaler.scale(loss_divided_total).backward()


            # Accumulate values that have already been divided by accum_steps.
            total_main_div  += float(loss_divided_main.detach().cpu())
            total_align_div += float(loss_divided_align.detach().cpu())
            total_smooth_div += float(loss_divided_smooth.detach().cpu())
            total_all_div   += float(loss_divided_total.detach().cpu())

            print(f"[Epoch {epoch}] mb {micro}/{len(train_loader)} | upd {num_updates}/{total_updates} "
                f"| main {loss_main.item():.4f} | align {align_loss.item()} | smooth {smooth_loss.item()}"
                f"| total {loss_total.item():.4f}", end="\r")


            # Take a real optimizer step every accum_steps micro-batches.
            if micro % args.accum_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if align_tok is not None:
                        torch.nn.utils.clip_grad_norm_(align_tok.parameters(), args.max_grad_norm)

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                num_updates += 1
                epoch_updates += 1

            print(f"[Epoch {epoch}] mb {micro}/{len(train_loader)} | upd {num_updates}/{total_updates} "
                f"| main {loss_main.item():.4f} | align {align_loss.item()} | smooth {smooth_loss.item()}"
                f"| total {loss_total.item():.4f}", end="\r")

        # Flush a tail batch that did not fill accum_steps.
        if micro % max(args.accum_steps, 1) != 0:
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if align_tok is not None:
                    torch.nn.utils.clip_grad_norm_(align_tok.parameters(), args.max_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            num_updates += 1
            epoch_updates += 1

        if args.anneal_lr:
            frac = 1.0 - (num_updates / (total_updates - 1))
            lrnow = args.final_lr + frac * (args.lr - args.final_lr)
            optim.param_groups[0]['lr'] = lrnow  
        # ===== Epoch logs =====
        # Reduce train stats across ranks for cleaner logging
        if getattr(args, "distributed", False):
            _t = torch.tensor(
                [total_main_div, total_align_div, total_smooth_div, total_all_div, float(epoch_updates)],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(_t, op=dist.ReduceOp.SUM)
            total_main_div, total_align_div, total_smooth_div, total_all_div, epoch_updates_f = _t.tolist()
            epoch_updates_global = max(int(epoch_updates_f), 1)
        else:
            epoch_updates_global = max(int(epoch_updates), 1)

        train_main_loss   = (total_main_div   / epoch_updates_global) * max(args.accum_steps, 1)
        train_align_loss  = (total_align_div  / epoch_updates_global) * max(args.accum_steps, 1)
        train_smooth_loss = (total_smooth_div / epoch_updates_global) * max(args.accum_steps, 1)
        train_total_loss  = (total_all_div    / epoch_updates_global) * max(args.accum_steps, 1)
        print(f"\n[Epoch {epoch}] train main {train_main_loss:.4f} | align {train_align_loss:.4f} | smooth {train_smooth_loss:.4f} | total {train_total_loss:.4f}")

        # Evaluate one-step denoising, not full trajectory generation.
        model.eval()
        v_main_total = 0.0
        v_align_total = 0.0
        v_all_total = 0.0
        v_mse_x0_total = 0.0
        v_epe3d_total  = 0.0
        v_steps = 0
        with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float16, enabled=fp16_on):
            for pixel_values, input_ids, attn_mask, query_points, clean_flow, weights, vggt_hidden in val_loader:
                pixel_values = pixel_values.to(device, non_blocking=True)
                input_ids    = input_ids.to(device, non_blocking=True)
                attn_mask    = attn_mask.to(device, non_blocking=True)
                query_points = query_points.to(device, non_blocking=True)
                clean_flow   = clean_flow.to(device, non_blocking=True)
                weights      = weights.to(device, non_blocking=True)
                vggt_hidden  = vggt_hidden.to(device, non_blocking=True)

                B = clean_flow.size(0)
                t = torch.randint(0, schedule.num_train_timesteps, (B,), device=device).long()
                noisy_flow, noise = schedule.add_noise(clean_flow, t)

                s = schedule.sqrt_alphas_cumprod.to(device)[t].view(B, 1, 1, 1)
                c = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t].view(B, 1, 1, 1)
                # v-parameterization is usually more stable than predicting noise directly.
                v_target = s * noise - c * clean_flow

                drop_mask = torch.zeros(B, dtype=torch.bool, device=device)

                if args.query_points:
                    pred_v = model(
                        image_pixels=pixel_values,
                        instruction_input_ids=input_ids,
                        instruction_attention_mask=attn_mask,
                        query_points=query_points,
                        noisy_flow=noisy_flow, timestep=t,
                        drop_condition_mask=drop_mask
                    )
                else:
                    pred_v = model(
                        image_pixels=pixel_values,
                        instruction_input_ids=input_ids,
                        instruction_attention_mask=attn_mask,
                        query_points=None,
                        noisy_flow=noisy_flow, timestep=t,
                        drop_condition_mask=drop_mask
                    )

                err = (pred_v - v_target).pow(2).mean(dim=-1)
                denom = torch.full((B,), fill_value=clean_flow.size(1)*clean_flow.size(2), device=device)
                loss_b = err.sum(dim=(1,2)) / denom

                if args.snr_gamma > 0:
                    ab = schedule.alphas_cumprod.to(device)[t]
                    snr = (ab / (1 - ab)).clamp(1e-3, 50.0)
                    w_snr = snr.pow(args.snr_gamma)
                    v_loss_main = (w_snr * loss_b).mean()
                else:
                    v_loss_main = loss_b.mean()

                # v_align_loss
                v_align_loss = torch.zeros([], device=device)
                if args.vggt_feature and align_tok is not None:
                    features_3d = unwrap_model(model).get_features_3d(pixel_values)
                    v_align_loss = align_tok(features_3d, vggt_hidden)   # Scalar.

                v_loss_total = v_loss_main + align_weight * v_align_loss

                v_main_total  += float(v_loss_main.detach().cpu())
                v_align_total += float(v_align_loss.detach().cpu())
                v_all_total   += float(v_loss_total.detach().cpu())
                v_steps += 1

                # Closed-form reconstruction for v-parameterization: x0_hat = s * x_t - c * v_pred.
                x0_hat = s * noisy_flow - c * pred_v                    # [B,K,N,3]
                diff_x0 = x0_hat - clean_flow                           # [B,K,N,3]

                # MSE(x0): average xyz first, then compute a visibility-weighted mean.
                mse_x0_map = diff_x0.pow(2).mean(dim=-1)                # [B,K,N]
                if weights is not None:
                    denom_w = (weights.sum(dim=(1,2)) + 1e-8)           # [B]
                    mse_x0_b = (mse_x0_map * weights).sum(dim=(1,2)) / denom_w
                else:
                    denom = clean_flow.size(1) * clean_flow.size(2)
                    mse_x0_b = mse_x0_map.view(B, -1).mean(dim=1)

                # EPE3D(x0): vector L2 distance with the same weighted averaging.

                epe3d_map = diff_x0.pow(2).sum(dim=-1).sqrt()           # [B,K,N]
                if weights is not None:
                    epe3d_b = (epe3d_map * weights).sum(dim=(1,2)) / denom_w
                else:
                    epe3d_b = epe3d_map.view(B, -1).mean(dim=1)

                v_mse_x0_total += float(mse_x0_b.mean().detach().cpu())
                v_epe3d_total  += float(epe3d_b.mean().detach().cpu())

        if getattr(args, "distributed", False):
            _v = torch.tensor(
                [v_main_total, v_align_total, v_all_total, v_mse_x0_total, v_epe3d_total, float(v_steps)],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(_v, op=dist.ReduceOp.SUM)
            v_main_total, v_align_total, v_all_total, v_mse_x0_total, v_epe3d_total, v_steps_f = _v.tolist()
            v_steps = int(v_steps_f)

        val_main_loss  = v_main_total  / max(v_steps, 1)
        val_align_loss = v_align_total / max(v_steps, 1)
        val_total_loss = v_all_total   / max(v_steps, 1)

        val_mse_x0  = v_mse_x0_total / max(v_steps, 1)
        val_epe3d   = v_epe3d_total  / max(v_steps, 1)
        print(f"[Epoch {epoch}] val   main {val_main_loss:.4f} | align {val_align_loss:.4f} | total {val_total_loss:.4f} | x0_MSE {val_mse_x0:.6f} | x0_EPE3D {val_epe3d:.6f}")

        # ----------------- Deployment-style Val -----------------
        # This one matches inference more closely: no clean_flow is fed into the denoising path.
        # It starts from Gaussian noise and runs DDIM, then compares the final x0 to GT.
        val_sampling_mse = None
        val_sampling_epe3d = None
        if args.val_sampling:
            sample_mse_total = 0.0
            sample_epe_total = 0.0
            sample_steps = 0
            gen_device = device if device.type == "cuda" else torch.device("cpu")
            sample_gen = torch.Generator(device=gen_device)
            sample_gen.manual_seed(int(args.val_sampling_seed))

            with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float16, enabled=fp16_on):
                for sample_i, (pixel_values, input_ids, attn_mask, query_points, clean_flow, weights, vggt_hidden) in enumerate(val_loader, start=1):
                    if args.val_sampling_max_batches > 0 and sample_i > args.val_sampling_max_batches:
                        break

                    pixel_values = pixel_values.to(device, non_blocking=True)
                    input_ids    = input_ids.to(device, non_blocking=True)
                    attn_mask    = attn_mask.to(device, non_blocking=True)
                    query_points = query_points.to(device, non_blocking=True)
                    clean_flow   = clean_flow.to(device, non_blocking=True)
                    weights      = weights.to(device, non_blocking=True)

                    pred_flow = ddim_sample_vpred(
                        model=model,
                        schedule=schedule,
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attn_mask=attn_mask,
                        query_points=(query_points if args.query_points else None),
                        shape=tuple(clean_flow.shape),
                        device=device,
                        num_inference_steps=args.val_sampling_steps,
                        guidance_scale=args.val_sampling_guidance_scale,
                        generator=sample_gen,
                    )
                    sample_mse, sample_epe = flow_mse_epe3d(pred_flow, clean_flow, weights)
                    sample_mse_total += float(sample_mse.detach().cpu())
                    sample_epe_total += float(sample_epe.detach().cpu())
                    sample_steps += 1

            if getattr(args, "distributed", False):
                _sv = torch.tensor(
                    [sample_mse_total, sample_epe_total, float(sample_steps)],
                    device=device, dtype=torch.float64
                )
                dist.all_reduce(_sv, op=dist.ReduceOp.SUM)
                sample_mse_total, sample_epe_total, sample_steps_f = _sv.tolist()
                sample_steps = int(sample_steps_f)

            val_sampling_mse = sample_mse_total / max(sample_steps, 1)
            val_sampling_epe3d = sample_epe_total / max(sample_steps, 1)
            print(
                f"[Epoch {epoch}] val sample DDIM{args.val_sampling_steps} "
                f"cfg {args.val_sampling_guidance_scale:.2f} | "
                f"x0_MSE {val_sampling_mse:.6f} | x0_EPE3D {val_sampling_epe3d:.6f}"
            )


        # --- [W&B] eval log ---
        if wb is not None:
            log_payload = {
                "train/loss_main":  float(train_main_loss),
                "train/align_loss": float(train_align_loss),
                "train/smooth_loss": float(train_smooth_loss),
                "train/loss_total": float(train_total_loss),
                "val/loss_main":    float(val_main_loss),
                "val/align_loss":   float(val_align_loss),
                "val/loss_total":   float(val_total_loss),
                "val/x0_mse":    float(val_mse_x0),
                "val/x0_epe3d":  float(val_epe3d),
                "epoch": int(epoch),
                # Useful diagnostics.
                "optim/lr_group0":  float(optim.param_groups[0]["lr"]),
                "optim/lr_align":   float(align_lr if args.vggt_feature else 0.0),
                "scaler/scale":     float(scaler.get_scale()),
            }
            if args.val_sampling:
                log_payload.update({
                    "val_sampling/x0_mse": float(val_sampling_mse),
                    "val_sampling/x0_epe3d": float(val_sampling_epe3d),
                    "val_sampling/ddim_steps": int(args.val_sampling_steps),
                    "val_sampling/guidance_scale": float(args.val_sampling_guidance_scale),
                })
            wandb.log(log_payload, commit=True)
        if args.val_sampling and val_sampling_epe3d is not None:
            best_metric = float(val_sampling_epe3d)
            best_metric_name = "val_sampling/x0_epe3d"
        else:
            best_metric = float(val_total_loss)
            best_metric_name = "val/loss_total"

        if is_main_process() and (best_metric < best_val):
            best_val = best_metric
            save_dir = args.save_dir + args.wandb_run
            os.makedirs(save_dir, exist_ok=True)
            save_path = Path(save_dir) / f"siglip_flow_futureK_best.pt"
            torch.save({"epoch": epoch,
                        "model": unwrap_model(model).state_dict(),
                        "align_tok": (unwrap_model(align_tok).state_dict() if align_tok is not None else None),
                        "val_loss": val_total_loss,
                        "val_sampling_mse": val_sampling_mse,
                        "val_sampling_epe3d": val_sampling_epe3d,
                        "best_metric": best_metric,
                        "best_metric_name": best_metric_name,
                        "k_steps": args.k_steps,
                        "num_points": args.num_points,
                        "train_noise_timesteps": args.train_noise_timesteps,
                        "cfg": cfg}, save_path)
            print(f"  -> saved: {save_path} ({best_metric_name}={best_metric:.6f})")

    print("Done.")
    cleanup_distributed()

if __name__ == "__main__":
    main()
