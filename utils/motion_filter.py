import torch


def _robust_upper_threshold(values: torch.Tensor, factor: float) -> torch.Tensor:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return torch.tensor(float("inf"))
    median = values.median()
    mad = (values - median).abs().median()
    if float(mad.item()) > 1e-12:
        return median + float(factor) * 1.4826 * mad
    return values.mean() + float(factor) * (values.std(unbiased=False) + 1e-12)


def _largest_component_mask(dist: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a local mask for the largest connected component in a distance graph."""
    n = int(dist.shape[0])
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=dist.device)

    adj = (dist <= float(threshold)).detach().cpu().numpy()
    seen = [False] * n
    best: list[int] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            neighbors = adj[cur].nonzero()[0].tolist()
            for nb in neighbors:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        if len(comp) > len(best):
            best = comp

    keep = torch.zeros(n, dtype=torch.bool, device=dist.device)
    if best:
        keep[torch.as_tensor(best, dtype=torch.long, device=dist.device)] = True
    return keep


@torch.no_grad()
def filter_points_component_outliers(
    points: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    enabled: bool = True,
    spatial_k: int = 5,
    spatial_factor: float = 6.0,
    frame_stride: int = 1,
    min_bad_frames: int = 1,
    min_keep_ratio: float = 0.55,
    use_trajectory_component: bool = True,
    replace_outliers: bool = True,
    replace_mode: str = "nearest",
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | bool]]:
    """
    Remove whole trajectories that form small disconnected islands.

    This catches points that drift as a small coherent group, which ordinary
    first-frame SOR and max-step filters can miss.
    """
    assert points.ndim == 3 and points.shape[-1] == 3
    device = points.device
    T, N, _ = points.shape
    stats: dict[str, float | int | bool] = {
        "component_enabled": bool(enabled),
        "component_bad_count": 0,
        "component_frame_bad_count": 0,
        "component_traj_bad_count": 0,
        "component_fallback_count": 0,
        "component_min_keep_ratio": float(min_keep_ratio),
    }
    finite = torch.isfinite(points).all(dim=(0, 2))
    if candidate_mask is None:
        candidate = finite.clone()
    else:
        candidate = candidate_mask.bool().to(device) & finite
    if (not enabled) or int(candidate.sum().item()) < max(4, int(spatial_k) + 2):
        return points, candidate, stats

    candidate_idxs = candidate.nonzero(as_tuple=False).squeeze(-1)
    min_keep = max(3, int(torch.ceil(torch.tensor(float(min_keep_ratio) * candidate_idxs.numel())).item()))
    bad_votes = torch.zeros(N, dtype=torch.int64, device=device)
    fallback_count = 0

    frame_ids = list(range(0, T, max(1, int(frame_stride))))
    if (T - 1) not in frame_ids:
        frame_ids.append(T - 1)
    k = int(max(1, min(int(spatial_k), int(candidate_idxs.numel()) - 1)))

    for t in frame_ids:
        P = points[int(t), candidate_idxs]
        dist = torch.cdist(P, P)
        dist.fill_diagonal_(float("inf"))
        nn = torch.topk(dist, k=k, largest=False, dim=1).values
        nn_score = nn.mean(dim=1)
        threshold = _robust_upper_threshold(nn_score, spatial_factor)
        finite_threshold = float(threshold.item())
        if not torch.isfinite(threshold) or finite_threshold <= 0:
            fallback_count += 1
            continue

        graph_dist = dist.clone()
        graph_dist.fill_diagonal_(0.0)
        local_keep = _largest_component_mask(graph_dist, finite_threshold)
        if int(local_keep.sum().item()) < min_keep:
            fallback_count += 1
            continue

        local_bad = ~local_keep
        if local_bad.any():
            bad_votes[candidate_idxs[local_bad]] += 1

    traj_bad = torch.zeros(N, dtype=torch.bool, device=device)
    if use_trajectory_component and int(candidate_idxs.numel()) >= max(4, int(spatial_k) + 2):
        traj = points[:, candidate_idxs, :].permute(1, 0, 2).reshape(candidate_idxs.numel(), -1)
        dist = torch.cdist(traj, traj) / max(float(T) ** 0.5, 1.0)
        dist.fill_diagonal_(float("inf"))
        nn = torch.topk(dist, k=k, largest=False, dim=1).values
        threshold = _robust_upper_threshold(nn.mean(dim=1), spatial_factor)
        finite_threshold = float(threshold.item())
        if torch.isfinite(threshold) and finite_threshold > 0:
            graph_dist = dist.clone()
            graph_dist.fill_diagonal_(0.0)
            local_keep = _largest_component_mask(graph_dist, finite_threshold)
            if int(local_keep.sum().item()) >= min_keep:
                traj_bad[candidate_idxs[~local_keep]] = True
            else:
                fallback_count += 1
        else:
            fallback_count += 1

    frame_component_bad = bad_votes >= max(1, int(min_bad_frames))
    component_bad = frame_component_bad | traj_bad
    keep_mask = candidate & (~component_bad)
    if int(keep_mask.sum().item()) < min_keep and bool(traj_bad.any().item()):
        # Frame-wise components can be too strict when the object briefly splits
        # into several visible parts. Keep the more stable full-trajectory test.
        component_bad = traj_bad
        keep_mask = candidate & (~component_bad)
    if int(keep_mask.sum().item()) < min_keep:
        stats["component_fallback_count"] = int(fallback_count + 1)
        return points, candidate, stats

    stats.update(
        {
            "component_bad_count": int(component_bad.sum().item()),
            "component_frame_bad_count": int(frame_component_bad.sum().item()),
            "component_traj_bad_count": int(traj_bad.sum().item()),
            "component_fallback_count": int(fallback_count),
        }
    )
    if (not replace_outliers) or int(component_bad.sum().item()) == 0:
        return points, keep_mask, stats

    inlier_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
    outlier_idx = component_bad.nonzero(as_tuple=False).squeeze(-1)
    if inlier_idx.numel() == 0 or outlier_idx.numel() == 0:
        return points, keep_mask, stats

    if replace_mode == "random":
        src = inlier_idx[torch.randint(0, inlier_idx.numel(), (outlier_idx.numel(),), device=device)]
    elif replace_mode == "nearest":
        frame0 = points[0]
        out_pts = frame0[outlier_idx]
        in_pts = frame0[inlier_idx]
        best_d2 = torch.full((out_pts.shape[0],), float("inf"), device=device)
        best_j = torch.zeros((out_pts.shape[0],), dtype=torch.long, device=device)
        for st in range(0, in_pts.shape[0], chunk):
            ed = min(st + chunk, in_pts.shape[0])
            d2 = ((out_pts[:, None, :] - in_pts[None, st:ed, :]) ** 2).sum(dim=-1)
            d2_min, argmin = torch.min(d2, dim=1)
            better = d2_min < best_d2
            best_d2[better] = d2_min[better]
            best_j[better] = st + argmin[better]
        src = inlier_idx[best_j]
    else:
        raise ValueError(f"Unknown replace_mode={replace_mode}")

    new_points = points.clone()
    new_points[:, outlier_idx, :] = points[:, src, :]
    return new_points, keep_mask, stats

@torch.no_grad()
def filter_points_moving_and_sor_firstframe(
    points: torch.Tensor,
    motion_thresh: float,
    k: int = 16,
    std_ratio: float = 2.0,
    replace_outliers: bool = True,
    replace_mode: str = "random",
    chunk: int = 2048,
    noise_std: float = 0.0,
    fallback_to_moving_if_no_inlier: bool = True,

    teleport_step_thresh: float = 0.3,
    teleport_min_steps: int = 1,
    teleport_use_max: bool = True,
    teleport_ratio_thresh: float = 0.01,
):
    """
    Filter whole point tracks using motion, first-frame SOR, and teleport checks.

    Returns:
        new_points: [T,N,3]
        inlier_masks: [T,N] bool
        moving_mask: [N] bool
        motion_mag: [N]
    """
    assert points.ndim == 3 and points.shape[-1] == 3
    device = points.device
    T, N, _ = points.shape

    v = points[1:] - points[:-1]                              # [T-1,N,3]
    step_norm = torch.linalg.norm(v, dim=-1)                  # [T-1,N]
    motion_mag = step_norm.sum(dim=0)                         # [N]
    moving_mask = motion_mag > motion_thresh                  # [N]

    max_step = step_norm.max(dim=0).values                    # [N]
    exceed_steps = (step_norm > teleport_step_thresh).sum(dim=0)  # [N]

    if teleport_use_max:
        teleport_bad = (max_step > teleport_step_thresh) & (exceed_steps >= teleport_min_steps)
    else:
        ratio = exceed_steps.float() / max(float(T - 1), 1.0)
        teleport_bad = (ratio > teleport_ratio_thresh)

    frame0 = points[0]                                        # [N,3]
    cand_idx = moving_mask.nonzero(as_tuple=False).squeeze(-1) # [M]
    sor_inlier0 = torch.zeros(N, dtype=torch.bool, device=device)

    M = cand_idx.numel()
    if M <= k:
        sor_inlier0[cand_idx] = True
    else:
        P = frame0[cand_idx]  # [M,3]
        mean_knn = torch.empty(M, device=device, dtype=P.dtype)

        for st in range(0, M, chunk):
            ed = min(st + chunk, M)
            Pq = P[st:ed]                  # [c,3]
            d = torch.cdist(Pq, P)         # [c,M]

            rows = torch.arange(ed - st, device=device)
            cols = torch.arange(st, ed, device=device)
            d[rows, cols] = float("inf")

            knn = torch.topk(d, k=k, dim=1, largest=False).values  # [c,k]
            mean_knn[st:ed] = knn.mean(dim=1)

        mu = mean_knn.mean()
        sigma = mean_knn.std(unbiased=False) + 1e-12
        thr = mu + std_ratio * sigma
        inlier_local = mean_knn <= thr
        sor_inlier0[cand_idx[inlier_local]] = True

    inlier0_mask = moving_mask & sor_inlier0 & (~teleport_bad)     # [N]
    outlier0_mask = ~inlier0_mask                                  # [N]

    inlier_masks = inlier0_mask.unsqueeze(0).expand(T, N).clone()  # [T,N]

    if not replace_outliers:
        return points, inlier_masks, moving_mask, motion_mag

    inlier_idx = inlier0_mask.nonzero(as_tuple=False).squeeze(-1)
    outlier_idx = outlier0_mask.nonzero(as_tuple=False).squeeze(-1)

    if inlier_idx.numel() == 0:
        if fallback_to_moving_if_no_inlier and moving_mask.any():
            inlier0_mask = moving_mask & (~teleport_bad)
            inlier_masks = inlier0_mask.unsqueeze(0).expand(T, N).clone()
            inlier_idx = inlier0_mask.nonzero(as_tuple=False).squeeze(-1)
            outlier_idx = (~inlier0_mask).nonzero(as_tuple=False).squeeze(-1)

            if inlier_idx.numel() == 0:
                return points, inlier_masks, moving_mask, motion_mag
        else:
            return points, inlier_masks, moving_mask, motion_mag

    if outlier_idx.numel() == 0:
        return points, inlier_masks, moving_mask, motion_mag

    if replace_mode == "random":
        r = torch.randint(0, inlier_idx.numel(), (outlier_idx.numel(),), device=device)
        src = inlier_idx[r]

    elif replace_mode == "nearest":
        out_pts = frame0[outlier_idx]   # [K,3]
        in_pts  = frame0[inlier_idx]    # [M,3]

        best_d2 = torch.full((out_pts.shape[0],), float("inf"), device=device)
        best_j  = torch.zeros((out_pts.shape[0],), dtype=torch.long, device=device)

        for st in range(0, in_pts.shape[0], chunk):
            ed = min(st + chunk, in_pts.shape[0])
            in_chunk = in_pts[st:ed]   # [c,3]
            d2 = ((out_pts[:, None, :] - in_chunk[None, :, :]) ** 2).sum(dim=-1)  # [K,c]
            d2_min, argmin = torch.min(d2, dim=1)
            better = d2_min < best_d2
            best_d2[better] = d2_min[better]
            best_j[better]  = st + argmin[better]

        src = inlier_idx[best_j]
    else:
        raise ValueError(f"Unknown replace_mode={replace_mode}")

    new_points = points.clone()
    new_points[:, outlier_idx, :] = points[:, src, :]

    if noise_std > 0:
        new_points[:, outlier_idx, :] += noise_std * torch.randn_like(new_points[:, outlier_idx, :])


    # Added inside filter_points_moving_and_sor_firstframe.
    print(f"Detailed filtering stats:")
    print(f"  - Total points: {N}")
    print(f"  - moving_mask (motion>{motion_thresh}): {moving_mask.sum().item()}")
    print(f"  - sor_inlier0 (after SOR): {sor_inlier0.sum().item()}")
    print(f"  - teleport_bad: {teleport_bad.sum().item()}")
    print(f"  - inlier0_mask (moving & sor & ~teleport): {inlier0_mask.sum().item()}")
    print(f"  - outlier0_mask: {outlier0_mask.sum().item()}")

    # Check the intersection.
    moving_and_sor = moving_mask & sor_inlier0
    print(f"  - moving & sor: {moving_and_sor.sum().item()}")
    moving_and_not_teleport = moving_mask & (~teleport_bad)
    print(f"  - moving & ~teleport: {moving_and_not_teleport.sum().item()}")

    return new_points, inlier_masks, moving_mask, motion_mag
