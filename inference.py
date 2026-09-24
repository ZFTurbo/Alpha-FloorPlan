import json
import os

if __name__ == '__main__':
    gpu_use = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import matplotlib
matplotlib.use('Agg')

import random
import time
import tqdm
import datetime
import multiprocessing
from collections import defaultdict
import math
import functools
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import concurrent.futures
import warnings
from config import *
from visualize_func import draw_layout, render_refinement_video
from scoring import score_saved_solutions_local, evaluate_solution_fast
from model import FloorplanDirectTransformer, decode_from_direct_space


warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated"
)

# --- phase profiling (temporary) ---
PHASE_TIMES = defaultdict(float)
PHASE_COUNT = defaultdict(int)
EXIT_STEPS = []

torch.set_num_threads(TORCH_THREADS)

def compute_spectral_embeddings(b2b_conn, num_nodes, k_dim=3):
    """
    Calculates the spectral embeddings of the graph (Fiedler vectors).
    k_dim: number of vectors to extract.
    """
    # Determine the device from the input tensor
    device = b2b_conn.device

    # 1. Build adjacency matrix W (device added)
    W = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if b2b_conn.numel() > 0:
        u = b2b_conn[:, 0].long()
        v = b2b_conn[:, 1].long()
        weights = b2b_conn[:, 2].float()
        W[u, v] = weights
        W[v, u] = weights

    # 2. Diagonal degree matrix D
    degrees = W.sum(dim=1)
    D = torch.diag(degrees)

    # 3. Laplacian matrix L = D - W
    L = D - W

    # For numerical stability of eigh on CPU/GPU, add a tiny jitter to the diagonal
    # (device added to torch.eye)
    L += torch.eye(num_nodes, device=device) * 1e-6

    # 4. Spectral decomposition
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # 5. Extract k_dim vectors corresponding to minimum NON-ZERO values
    # The first eigenvalue is always ~0 (or equal to the number of connected components).
    # We take vectors from the 2nd to the (k_dim + 1)-th.

    # Protection against graphs where nodes are fewer than k_dim + 1
    start_idx = 1
    end_idx = min(start_idx + k_dim, num_nodes)

    selected_vecs = eigenvectors[:, start_idx:end_idx]

    # If there are too few nodes, pad with zeros
    if selected_vecs.shape[1] < k_dim:
        pad_len = k_dim - selected_vecs.shape[1]
        selected_vecs = F.pad(selected_vecs, (0, pad_len), value=0.0)

    # 6. Standardization (Z-score normalization) of vectors for training stability
    mean = selected_vecs.mean(dim=0, keepdim=True)
    std = selected_vecs.std(dim=0, keepdim=True) + 1e-8
    normalized_vecs = (selected_vecs - mean) / std

    return normalized_vecs  # Shape: (num_nodes, k_dim)


def compute_spd_matrix_hops(b2b_conn, num_nodes, max_dist=20):
    """
    Calculates the shortest path matrix (in hops) using the Floyd-Warshall algorithm.
    """
    # 1. Initialize the matrix with maximum distances
    spd = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
    spd.fill_diagonal_(0)

    # 2. Set 1-hop connections (neighbors)
    if b2b_conn.numel() > 0:
        u = b2b_conn[:, 0].long()
        v = b2b_conn[:, 1].long()
        spd[u, v] = 1
        spd[v, u] = 1

    # 3. Vectorized Floyd-Warshall algorithm (works <1ms for N=120)
    for k in range(num_nodes):
        # Broadcasting: add column k and row k
        dist_through_k = spd[:, k].unsqueeze(1) + spd[k, :].unsqueeze(0)
        # Take the minimum between current path and path through k
        spd = torch.min(spd, dist_through_k)

    # Strictly limit the maximum distance just in case
    spd = torch.clamp(spd, max=max_dist)
    return spd


def compute_mib_loss(positions, constraints):
    """Blocks with the same MIB ID must have identical w, h"""
    w, h = positions[:, 2], positions[:, 3]
    mib_ids = constraints[:, 2]

    loss = torch.tensor(0.0, device=positions.device)
    unique_ids = mib_ids[mib_ids > 0].unique()

    for gid in unique_ids:
        mask = mib_ids == gid
        if mask.sum() < 2:
            continue

        group_w = w[mask]
        group_h = h[mask]

        # L1 difference relative to the mean inside the group, normalized
        loss += torch.abs(group_w - group_w.mean()).mean() / (group_w.mean() + 1e-6)
        loss += torch.abs(group_h - group_h.mean()).mean() / (group_h.mean() + 1e-6)

    return loss


def compute_cluster_loss(positions, constraints):
    """Blocks of the same cluster must form a dense group"""
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    cluster_ids = constraints[:, 3]

    loss = torch.tensor(0.0, device=positions.device)
    unique_ids = cluster_ids[cluster_ids > 0].unique()

    for gid in unique_ids:
        mask = cluster_ids == gid
        if mask.sum() < 2:
            continue

        gx, gy, gw, gh = x[mask], y[mask], w[mask], h[mask]

        bbox_w = (gx + gw).max() - gx.min()
        bbox_h = (gy + gh).max() - gy.min()
        bbox_area = bbox_w * bbox_h

        sum_area = (gw * gh).sum()

        # Calculate relative penalty and strictly clamp it (e.g., x20 of the sum of areas)
        penalty = torch.relu(bbox_area - sum_area) / (sum_area + 1e-6)
        loss += torch.clamp(penalty, max=20.0)

    return loss


def compute_boundary_loss(positions, constraints):
    """Blocks must touch the specified side of the common bounding box"""
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    boundary_codes = constraints[:, 4]

    # Global bounding box (differentiable)
    x_min = x.min()
    y_min = y.min()
    x_max = (x + w).max()
    y_max = (y + h).max()

    # Global dimensions for error normalization
    global_w = torch.clamp(x_max - x_min, min=1e-6)
    global_h = torch.clamp(y_max - y_min, min=1e-6)

    loss = torch.tensor(0.0, device=positions.device)

    for i in range(len(positions)):
        code = int(boundary_codes[i].item())
        if code == 0:
            continue

        # Bit flags: 1=Left, 2=Right, 4=Top, 8=Bottom
        # Now error is measured in fractions of total layout width/height (without squares!)
        if code & 1:  # Left
            loss += 2.0 * torch.abs(x[i] - x_min) / global_w
        if code & 2:  # Right
            loss += torch.abs(x[i] + w[i] - x_max) / global_w
        if code & 4:  # Top
            loss += torch.abs(y[i] + h[i] - y_max) / global_h
        if code & 8:  # Bottom
            loss += 2.0 * torch.abs(y[i] - y_min) / global_h

    return loss / (len(positions) + 1e-6)


def transformer_direct_collate_fn(batch):
    batched_features = []
    batched_b2b_matrix = []
    batched_p2b_matrix = []
    batched_pins_pos = []
    batched_spd_matrix = []
    padding_masks = []
    batched_scales = []
    batched_target_fp = []  # Ground truth coordinates for MSE loss
    raw_items = []

    max_len = 120
    max_pins = max(item['pins_pos'].shape[0] for item in batch)

    for item in batch:
        area = item['area_target']
        block_count = int((area != -1).sum().item())

        curr_fp_sol = item['fp_sol'][:block_count]
        curr_constraints = item['constraints'][:block_count]
        curr_area = area[:block_count]
        b2b_conn = item['b2b_conn']
        p2b_conn = item['p2b_conn']
        pins_pos = item['pins_pos']
        num_pins = pins_pos.shape[0]

        graph_scale = pins_pos.max() if pins_pos.numel() > 0 else torch.tensor(100.0)
        batched_scales.append(graph_scale)

        # Compute spectral embeddings (e.g., 3 vectors)
        k_spectral_dims = 3
        spectral_emb = compute_spectral_embeddings(b2b_conn, block_count, k_dim=k_spectral_dims)

        curr_p2b_matrix = torch.zeros((block_count, num_pins), dtype=torch.float32)
        if p2b_conn.numel() > 0:
            pin_ids = p2b_conn[:, 0].long()
            block_ids = p2b_conn[:, 1].long()
            weights = p2b_conn[:, 2].float()
            curr_p2b_matrix[block_ids, pin_ids] = weights

        # =====================================================================
        # NEW: Weighted Pin Center of Mass (Pin CoM)
        # =====================================================================
        if pins_pos.numel() > 0:
            # 1. Sum of pin connection weights for each block
            pin_weight_sum = curr_p2b_matrix.sum(dim=1, keepdim=True)

            # 2. Weighted sum of pin coordinates: [block_count, num_pins] @ [num_pins, 2] -> [block_count, 2]
            pin_com = torch.matmul(curr_p2b_matrix, pins_pos.float())

            # 3. Coordinate normalization (division by weights sum)
            has_pins_mask = (pin_weight_sum > 0).squeeze(1)
            pin_com[has_pins_mask] /= pin_weight_sum[has_pins_mask]

            # 4. Normalization to graph scale (into ~[0, 1] range)
            pin_com_norm = pin_com / graph_scale

            # 5. For blocks without pins, set neutral canvas center (0.5, 0.5)
            pin_com_norm[~has_pins_mask] = 0.5
        else:
            # If there are no pins in the graph at all
            pin_com_norm = torch.full((block_count, 2), 0.5, dtype=torch.float32)

        curr_b2b_matrix = torch.zeros((block_count, block_count), dtype=torch.float32)
        if b2b_conn.numel() > 0:
            u, v, w = b2b_conn[:, 0].long(), b2b_conn[:, 1].long(), b2b_conn[:, 2]
            curr_b2b_matrix[u, v] = w
            curr_b2b_matrix[v, u] = w

        # =====================================================================
        # NEW: Computing shortest path matrix (SPD)
        # =====================================================================
        curr_spd_matrix = compute_spd_matrix_hops(b2b_conn, block_count, max_dist=20)

        # =====================================================================
        # NEW: Calculation and transfer of target Aspect Ratio (known_r) and coordinates
        # =====================================================================
        curr_x = curr_fp_sol[:, 0]
        curr_y = curr_fp_sol[:, 1]
        curr_w = curr_fp_sol[:, 2]
        curr_h = curr_fp_sol[:, 3]

        # 1. Extract ground truth geometric profile of macros
        target_r = torch.log(curr_w / (curr_h + 1e-6))
        target_r_norm = torch.clamp(target_r / 1.2, -1.0, 1.0)
        target_x_norm = curr_x / graph_scale
        target_y_norm = curr_y / graph_scale

        # 2. Extract constraint masks
        is_fixed_flag = curr_constraints[:, 0] == 1.0
        is_preplaced_flag = curr_constraints[:, 1] == 1.0
        is_hard_flag = is_fixed_flag | is_preplaced_flag

        # 3. Form target features: 0.0 for free blocks
        known_r = torch.zeros_like(target_r_norm)
        known_r[is_hard_flag] = target_r_norm[is_hard_flag]

        # Fill X and Y coordinates ONLY for preplaced blocks
        known_x = torch.zeros_like(target_x_norm)
        known_y = torch.zeros_like(target_y_norm)
        known_x[is_preplaced_flag] = target_x_norm[is_preplaced_flag]
        known_y[is_preplaced_flag] = target_y_norm[is_preplaced_flag]
        # =====================================================================

        # Preparation of block input features: 14 features total
        # [Area_norm (1), Constraints (5), Spectral (3), Pin_CoM (2), Known_R (1), Known_X (1), Known_Y (1)]
        area_norm = (torch.sqrt(curr_area) / graph_scale).unsqueeze(1)
        curr_features = torch.cat([
            area_norm,
            curr_constraints,
            spectral_emb,
            pin_com_norm,
            known_r.unsqueeze(1),
            known_x.unsqueeze(1),
            known_y.unsqueeze(1)
        ], dim=1)

        pad_len = max_len - block_count

        padded_features = F.pad(curr_features, (0, 0, 0, pad_len), value=0.0)
        padded_target_fp = F.pad(curr_fp_sol, (0, 0, 0, pad_len), value=0.0)
        padded_b2b_matrix = F.pad(curr_b2b_matrix, (0, pad_len, 0, pad_len), value=0.0)
        padded_p2b_matrix = F.pad(curr_p2b_matrix, (0, max_pins - num_pins, 0, pad_len), value=0.0)
        padded_p2b_matrix = torch.log1p(padded_p2b_matrix)
        padded_pins_pos = F.pad(pins_pos.float(), (0, 0, 0, max_pins - num_pins), value=0.0)

        # Padding for SPD (pad with value 20)
        padded_spd_matrix = F.pad(curr_spd_matrix, (0, pad_len, 0, pad_len), value=20)

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[block_count:] = True

        batched_features.append(padded_features)
        batched_target_fp.append(padded_target_fp)
        batched_b2b_matrix.append(padded_b2b_matrix)
        batched_p2b_matrix.append(padded_p2b_matrix)
        batched_pins_pos.append(padded_pins_pos)
        batched_spd_matrix.append(padded_spd_matrix)  # NEW
        padding_masks.append(mask)

        raw_items.append({
            'block_count': block_count,
            'area_target': curr_area,
            'b2b_conn': b2b_conn,
            'p2b_conn': p2b_conn,
            'pins_pos': pins_pos,
            'metrics': item['metrics'],
            'graph_scale': graph_scale,
            'target_fp': curr_fp_sol,
        })

    return {
        'features': torch.stack(batched_features),
        'target_fp': torch.stack(batched_target_fp),
        'b2b_matrix': torch.stack(batched_b2b_matrix),
        'p2b_matrix': torch.stack(batched_p2b_matrix),
        'pins_pos': torch.stack(batched_pins_pos),
        'spd_matrix': torch.stack(batched_spd_matrix),  # NEW
        'padding_mask': torch.stack(padding_masks),
        'scales': torch.stack(batched_scales),
        'raw_items': raw_items
    }

def check_overlap_count_tensor(positions: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """GPU-only count of unique overlapping rectangle pairs.

    Supports batching: (..., N, 4) -> (...). For (N, 4) returns a 0-dim tensor."""
    n = positions.shape[-2]
    if n < 2:
        return positions.new_zeros(positions.shape[:-2], dtype=torch.long)

    x = positions[..., 0]
    y = positions[..., 1]
    r = x + positions[..., 2]
    b = y + positions[..., 3]

    overlap_x = (torch.minimum(r[..., :, None], r[..., None, :]) - torch.maximum(x[..., :, None], x[..., None, :])) > eps
    overlap_y = (torch.minimum(b[..., :, None], b[..., None, :]) - torch.maximum(y[..., :, None], y[..., None, :])) > eps
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=positions.device), diagonal=1)
    return ((overlap_x & overlap_y) & triu).sum(dim=(-2, -1))


def check_overlap_vectorized(positions: torch.Tensor) -> Tuple[int, List[Tuple[int, int, float]]]:
    """
    A version of intersection checking compatible with the original code.
    For fast checks inside the optimization loop, use check_overlap_count_tensor().
    """
    n = positions.size(0)
    if n < 2:
        return 0, []

    x = positions[:, 0]
    y = positions[:, 1]
    w = positions[:, 2]
    h = positions[:, 3]
    r = x + w
    b = y + h

    overlap_x = torch.clamp(torch.minimum(r[:, None], r[None, :]) - torch.maximum(x[:, None], x[None, :]), min=0)
    overlap_y = torch.clamp(torch.minimum(b[:, None], b[None, :]) - torch.maximum(y[:, None], y[None, :]), min=0)
    overlap_area = overlap_x * overlap_y
    unique_mask = torch.triu((overlap_x > 1e-7) & (overlap_y > 1e-7), diagonal=1)

    violations = int(torch.count_nonzero(unique_mask).item())
    if violations == 0:
        return 0, []

    idx = torch.nonzero(unique_mask, as_tuple=False).detach().cpu().tolist()
    areas = overlap_area[unique_mask].detach().cpu().tolist()
    return violations, [(ij[0], ij[1], float(a)) for ij, a in zip(idx, areas)]


# ============================================================================
# FAST PATH FOR REFINE: all constant preprocessing is done ONCE
# per test, only pure GPU operations remain inside the 800-step loop
# without a single CPU<->GPU synchronization.
# ============================================================================

def build_constraint_groups(constraints: torch.Tensor):
    """One-time unpacking of constraints into index tensors.

    All .unique()/.item()/iterations over CUDA tensors happen here once,
    and not at every optimization step (previously compute_boundary_loss did
    up to N .item() synchronizations on EVERY step).
    """
    device = constraints.device
    mib_ids = constraints[:, 2].long().cpu()
    cluster_ids = constraints[:, 3].long().cpu()
    boundary_codes = constraints[:, 4].long().cpu()

    mib_groups = []
    for gid in torch.unique(mib_ids[mib_ids > 0]).tolist():
        idx = torch.nonzero(mib_ids == gid, as_tuple=False).flatten()
        if idx.numel() > 1:
            mib_groups.append(idx.to(device))

    cluster_groups = []
    for gid in torch.unique(cluster_ids[cluster_ids > 0]).tolist():
        idx = torch.nonzero(cluster_ids == gid, as_tuple=False).flatten()
        if idx.numel() > 1:
            cluster_groups.append(idx.to(device))

    boundary_idx = {
        'left': torch.nonzero((boundary_codes & 1) > 0, as_tuple=False).flatten().to(device),
        'right': torch.nonzero((boundary_codes & 2) > 0, as_tuple=False).flatten().to(device),
        'top': torch.nonzero((boundary_codes & 4) > 0, as_tuple=False).flatten().to(device),
        'bottom': torch.nonzero((boundary_codes & 8) > 0, as_tuple=False).flatten().to(device),
        'n': constraints.shape[0],
    }
    return mib_groups, cluster_groups, boundary_idx


def compute_mib_loss_fast(positions: torch.Tensor, mib_groups) -> torch.Tensor:
    """The same formula as compute_mib_loss, but using precomputed group indices.
    Supports batching: (..., N, 4) -> (...)."""
    loss = positions.new_zeros(positions.shape[:-2])
    if not mib_groups:
        return loss
    w, h = positions[..., 2], positions[..., 3]
    for idx in mib_groups:
        gw, gh = w[..., idx], h[..., idx]
        mw, mh = gw.mean(dim=-1), gh.mean(dim=-1)
        loss = loss + torch.abs(gw - mw.unsqueeze(-1)).mean(dim=-1) / (mw + 1e-6)
        loss = loss + torch.abs(gh - mh.unsqueeze(-1)).mean(dim=-1) / (mh + 1e-6)
    return loss


def compute_cluster_loss_fast(positions: torch.Tensor, cluster_groups) -> torch.Tensor:
    """The same formula as compute_cluster_loss, but without .unique()/.sum().item() in the loop.
    Supports batching: (..., N, 4) -> (...)."""
    loss = positions.new_zeros(positions.shape[:-2])
    if not cluster_groups:
        return loss
    x, y, w, h = positions[..., 0], positions[..., 1], positions[..., 2], positions[..., 3]
    for idx in cluster_groups:
        gx, gy, gw, gh = x[..., idx], y[..., idx], w[..., idx], h[..., idx]
        bbox_area = ((gx + gw).max(dim=-1).values - gx.min(dim=-1).values) \
                    * ((gy + gh).max(dim=-1).values - gy.min(dim=-1).values)
        sum_area = (gw * gh).sum(dim=-1)
        penalty = torch.relu(bbox_area - sum_area) / (sum_area + 1e-6)
        loss = loss + torch.clamp(penalty, max=20.0)
    return loss


def compute_boundary_loss_fast(positions: torch.Tensor, boundary_idx) -> torch.Tensor:
    """Vectorized version of compute_boundary_loss.

    The original did a python loop over all blocks with int(codes[i].item()) —
    up to N GPU synchronizations per step. Here — 4 indexed sums.
    Supports batching: (..., N, 4) -> (...)."""
    x, y, w, h = positions[..., 0], positions[..., 1], positions[..., 2], positions[..., 3]
    x_min, y_min = x.min(dim=-1).values, y.min(dim=-1).values
    x_max, y_max = (x + w).max(dim=-1).values, (y + h).max(dim=-1).values
    global_w = torch.clamp(x_max - x_min, min=1e-6)
    global_h = torch.clamp(y_max - y_min, min=1e-6)

    loss = positions.new_zeros(positions.shape[:-2])
    li, ri = boundary_idx['left'], boundary_idx['right']
    ti, bi = boundary_idx['top'], boundary_idx['bottom']
    if li.numel() > 0:
        loss = loss + 2.0 * torch.abs(x[..., li] - x_min.unsqueeze(-1)).sum(dim=-1) / global_w
    if ri.numel() > 0:
        loss = loss + torch.abs(x[..., ri] + w[..., ri] - x_max.unsqueeze(-1)).sum(dim=-1) / global_w
    if ti.numel() > 0:
        loss = loss + torch.abs(y[..., ti] + h[..., ti] - y_max.unsqueeze(-1)).sum(dim=-1) / global_h
    if bi.numel() > 0:
        loss = loss + 2.0 * torch.abs(y[..., bi] - y_min.unsqueeze(-1)).sum(dim=-1) / global_h
    return loss / (boundary_idx['n'] + 1e-6)


def build_loss_context(
        valid_b2b: torch.Tensor,
        valid_p2b: torch.Tensor,
        pins_pos: torch.Tensor,
        metrics: torch.Tensor,
        constraints: torch.Tensor,
        n_blocks: int,
        dtype: torch.dtype,
        device: torch.device,
):
    """Precomputes everything that does not change between refine steps."""
    ctx = {}

    if valid_b2b.numel() > 0:
        i = valid_b2b[:, 0].long()
        j = valid_b2b[:, 1].long()
        wts = valid_b2b[:, 2].to(dtype=dtype)
        mask = (i < n_blocks) & (j < n_blocks)
        ctx['b2b_i'], ctx['b2b_j'], ctx['b2b_w'] = i[mask], j[mask], wts[mask]
    else:
        ctx['b2b_i'] = None

    if valid_p2b.numel() > 0 and pins_pos.numel() > 0:
        pin_idx = valid_p2b[:, 0].long()
        block_idx = valid_p2b[:, 1].long()
        wts = valid_p2b[:, 2].to(dtype=dtype)
        mask = (pin_idx < pins_pos.shape[0]) & (block_idx < n_blocks)
        pin_idx, block_idx, wts = pin_idx[mask], block_idx[mask], wts[mask]
        ctx['p2b_block'] = block_idx
        ctx['p2b_w'] = wts
        ctx['pin_x'] = pins_pos[pin_idx, 0].to(dtype=dtype)
        ctx['pin_y'] = pins_pos[pin_idx, 1].to(dtype=dtype)
    else:
        ctx['p2b_block'] = None

    if pins_pos.numel() > 0:
        # One sink per test instead of tensor scalars in the graph of each step
        ctx['min_x_canvas'] = float(pins_pos[:, 0].min().item())
        ctx['max_x_canvas'] = float(pins_pos[:, 0].max().item())
        ctx['min_y_canvas'] = float(pins_pos[:, 1].min().item())
        ctx['max_y_canvas'] = float(pins_pos[:, 1].max().item())
    else:
        ctx['min_x_canvas'], ctx['max_x_canvas'] = 0.0, 500.0
        ctx['min_y_canvas'], ctx['max_y_canvas'] = 0.0, 500.0

    ctx['baseline_area'] = metrics[0]
    ctx['baseline_hpwl'] = metrics[6] + metrics[7]
    ctx['triu_mask'] = torch.triu(
        torch.ones(n_blocks, n_blocks, dtype=torch.bool, device=device), diagonal=1
    )
    ctx['mib_groups'], ctx['cluster_groups'], ctx['boundary_idx'] = build_constraint_groups(constraints)
    return ctx


def compute_training_loss_fast(positions: torch.Tensor, ctx, step: int):
    """The same objective as compute_training_loss_differentiable_matrix,
    but on a precomputed context and without CPU<->GPU synchronizations.

    Supports batching: (..., N, 4) -> ((...), (...)) — independent loss
    per sample (for multi-start refinement)."""
    x = positions[..., 0]
    y = positions[..., 1]
    w = positions[..., 2]
    h = positions[..., 3]
    cx = x + w * 0.5
    cy = y + h * 0.5
    zero = positions.new_zeros(positions.shape[:-2])

    if ctx['b2b_i'] is not None and ctx['b2b_i'].numel() > 0:
        i, j = ctx['b2b_i'], ctx['b2b_j']
        dx = torch.sqrt((cx[..., i] - cx[..., j]).square() + 1e-5)
        dy = torch.sqrt((cy[..., i] - cy[..., j]).square() + 1e-5)
        hpwl_b2b = (ctx['b2b_w'] * (dx + dy)).sum(dim=-1)
    else:
        hpwl_b2b = zero

    if ctx['p2b_block'] is not None and ctx['p2b_block'].numel() > 0:
        bidx = ctx['p2b_block']
        dx = torch.abs(cx[..., bidx] - ctx['pin_x'])
        dy = torch.abs(cy[..., bidx] - ctx['pin_y'])
        hpwl_p2b = (ctx['p2b_w'] * (dx + dy)).sum(dim=-1)
    else:
        hpwl_p2b = zero

    hpwl_total = hpwl_b2b + hpwl_p2b

    gamma = 0.1
    x_right = x + w
    y_top = y + h
    x_max_soft = torch.logsumexp(gamma * x_right, dim=-1) / gamma
    y_max_soft = torch.logsumexp(gamma * y_top, dim=-1) / gamma
    x_min_soft = -torch.logsumexp(-gamma * x, dim=-1) / gamma
    y_min_soft = -torch.logsumexp(-gamma * y, dim=-1) / gamma
    bbox_area = (x_max_soft - x_min_soft) * (y_max_soft - y_min_soft)

    dx_matrix = torch.abs(cx[..., :, None] - cx[..., None, :])
    dy_matrix = torch.abs(cy[..., :, None] - cy[..., None, :])
    min_dist_x = (w[..., :, None] + w[..., None, :]) * 0.5
    min_dist_y = (h[..., :, None] + h[..., None, :]) * 0.5
    soft_overlap_x = torch.relu(min_dist_x - dx_matrix)
    soft_overlap_y = torch.relu(min_dist_y - dy_matrix)
    overlap_area = ((soft_overlap_x * soft_overlap_y) * ctx['triu_mask']).sum(dim=(-2, -1))
    total_block_area = (w * h).sum(dim=-1)
    overlap_violation = overlap_area / (total_block_area + 1e-6)

    hpwl_gap = torch.relu((hpwl_total - ctx['baseline_hpwl']) / (ctx['baseline_hpwl'] + 1e-6))
    area_gap = torch.relu((bbox_area - ctx['baseline_area']) / (ctx['baseline_area'] + 1e-6))

    oob_left_area = 100 * torch.relu(ctx['min_x_canvas'] - x) * h
    oob_bottom_area = 100 * torch.relu(ctx['min_y_canvas'] - y) * w
    oob_right_area = torch.relu(x_right - ctx['max_x_canvas']) * h
    oob_top_area = torch.relu(y_top - ctx['max_y_canvas']) * w
    total_oob_area = (oob_left_area + oob_bottom_area + oob_right_area + oob_top_area).sum(dim=-1)
    oob_violation = total_oob_area / (total_block_area + 1e-6)

    V_mib = compute_mib_loss_fast(positions, ctx['mib_groups'])
    V_cluster = compute_cluster_loss_fast(positions, ctx['cluster_groups'])
    V_boundary = compute_boundary_loss_fast(positions, ctx['boundary_idx'])

    V_soft = (1.0 + step / 50) * overlap_violation + oob_violation \
             + 0.01 * V_mib + 0.01 * V_cluster + 0.01 * V_boundary

    quality_factor = 1.0 + ALPHA * (hpwl_gap + area_gap)
    violation_factor = BETA * V_soft
    cost = quality_factor * violation_factor
    # hard violation counter taken from the already computed matrices (eps=1e-7 as in
    # check_overlap_count_tensor); this saves a second set of N×N matrices
    hard_viol = ((soft_overlap_x > 1e-7) & (soft_overlap_y > 1e-7) & ctx['triu_mask']).sum(dim=(-2, -1))
    return cost, overlap_violation, hard_viol


def fix_grouping_violations_greedily(
        positions: torch.Tensor,
        constraints: torch.Tensor,
        max_iters: int = 15,
        gap: float = 1e-4
) -> Tuple[torch.Tensor, int]:
    pos = positions.detach().cpu().tolist()
    n = len(pos)
    total_fixes = 0

    if n < 2: return positions, total_fixes

    is_preplaced = (constraints[:, 1] == 1).cpu().tolist()
    cluster_ids = constraints[:, 3].long().cpu().tolist()
    boundary_codes = constraints[:, 4].long().cpu().tolist()

    can_move = [not is_preplaced[i] for i in range(n)]
    unique_clusters = list(set([c for c in cluster_ids if c > 0]))

    if not unique_clusters: return positions, total_fixes

    # --- DYNAMIC CHIP BOUNDARY CALCULATION ---
    right_edges = [pos[i][0] + pos[i][2] for i in range(n) if (boundary_codes[i] & 2)]
    top_edges = [pos[i][1] + pos[i][3] for i in range(n) if (boundary_codes[i] & 4)]

    c_width = max(right_edges) if right_edges else max(p[0] + p[2] for p in pos)
    c_height = max(top_edges) if top_edges else max(p[1] + p[3] for p in pos)

    def is_valid_position(test_idx, test_x, test_y, w, h):
        # Protection against going beyond the computed canvas
        if test_x < 0 or test_y < 0 or test_x + w > c_width + 1e-4 or test_y + h > c_height + 1e-4:
            return False

        b_code = boundary_codes[test_idx]
        if (b_code & 1) and test_x > 1e-4: return False
        if (b_code & 2) and (test_x + w) < c_width - 1e-4: return False
        if (b_code & 4) and (test_y + h) < c_height - 1e-4: return False
        if (b_code & 8) and test_y > 1e-4: return False

        test_l, test_r = test_x + 1e-5, test_x + w - 1e-5
        test_b, test_t = test_y + 1e-5, test_y + h - 1e-5

        for i in range(n):
            if i == test_idx: continue
            ix, iy, iw, ih = pos[i]
            if test_l < ix + iw and test_r > ix and test_b < iy + ih and test_t > iy:
                return False

        return True

    for iteration in range(max_iters):
        moved_any = False
        for cid in unique_clusters:
            c_indices = [i for i, c in enumerate(cluster_ids) if c == cid]
            if len(c_indices) < 2: continue

            c_centers_x = sum([pos[i][0] + pos[i][2] / 2.0 for i in c_indices]) / len(c_indices)
            c_centers_y = sum([pos[i][1] + pos[i][3] / 2.0 for i in c_indices]) / len(c_indices)

            dists = []
            for idx in c_indices:
                mx, my, mw, mh = pos[idx]
                dist = (mx + mw / 2.0 - c_centers_x) ** 2 + (my + mh / 2.0 - c_centers_y) ** 2
                dists.append((dist, idx))

            dists.sort(reverse=True, key=lambda x: x[0])

            for curr_dist, idx in dists:
                if not can_move[idx]: continue
                my_x, my_y, my_w, my_h = pos[idx]
                candidates = []

                for other_idx in c_indices:
                    if other_idx == idx: continue
                    ox, oy, ow, oh = pos[other_idx]

                    candidates.extend([
                        (ox - my_w - gap, oy), (ox - my_w - gap, oy + oh - my_h),
                        (ox + ow + gap, oy), (ox + ow + gap, oy + oh - my_h),
                        (ox, oy - my_h - gap), (ox + ow - my_w, oy - my_h - gap),
                        (ox, oy + oh + gap), (ox + ow - my_w, oy + oh + gap),

                        (ox - my_w - gap, my_y), (ox + ow + gap, my_y),
                        (my_x, oy - my_h - gap), (my_x, oy + oh + gap)
                    ])

                best_x, best_y = my_x, my_y
                best_dist = curr_dist

                for cx, cy in candidates:
                    c_dist = (cx + my_w / 2.0 - c_centers_x) ** 2 + (cy + my_h / 2.0 - c_centers_y) ** 2
                    if c_dist < best_dist - 1e-4:
                        if is_valid_position(idx, cx, cy, my_w, my_h):
                            best_x, best_y = cx, cy
                            best_dist = c_dist

                if best_dist < curr_dist - 1e-4:
                    pos[idx][0], pos[idx][1] = best_x, best_y
                    moved_any = True
                    total_fixes += 1

                    c_centers_x = sum([pos[i][0] + pos[i][2] / 2.0 for i in c_indices]) / len(c_indices)
                    c_centers_y = sum([pos[i][1] + pos[i][3] / 2.0 for i in c_indices]) / len(c_indices)

        if not moved_any: break

    return torch.tensor(pos, device=positions.device, dtype=positions.dtype), total_fixes


def reshape_cluster_blocks_to_touch(
        positions: torch.Tensor,
        constraints: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Changes the aspect ratio of non-fixed cluster blocks
    so that they reach each other.
    Uses connected component (island) analysis, asymmetric scaling
    (anchors) and an analytical computation of the required dimensions.
    """
    pos = positions.detach().cpu().numpy().astype(np.float64)
    n = len(pos)
    fixes = 0

    if n < 2: return positions, fixes

    is_fixed = (constraints[:, 0] == 1).cpu().numpy()
    is_preplaced = (constraints[:, 1] == 1).cpu().numpy()
    mib_ids = constraints[:, 2].long().cpu().numpy()
    cluster_ids = constraints[:, 3].long().cpu().numpy()
    boundary_codes = constraints[:, 4].long().cpu().numpy()

    # Only ordinary (soft) blocks may change their shape
    can_reshape = ~(is_fixed | is_preplaced | (mib_ids > 0))
    unique_clusters = set(c for c in cluster_ids if c > 0)

    # Dynamically compute the chip boundaries
    right_edges = [pos[i][0] + pos[i][2] for i in range(n) if (boundary_codes[i] & 2)]
    top_edges = [pos[i][1] + pos[i][3] for i in range(n) if (boundary_codes[i] & 4)]
    c_width = max(right_edges) if right_edges else max(p[0] + p[2] for p in pos)
    c_height = max(top_edges) if top_edges else max(p[1] + p[3] for p in pos)

    for cid in unique_clusters:
        c_indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        if len(c_indices) < 2: continue

        # --- SEARCH FOR "ISLANDS" (connected components inside the cluster) ---
        components = []
        unvisited = set(c_indices)
        while unvisited:
            start = unvisited.pop()
            comp = {start}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                cx, cy, cw, ch = pos[curr]
                to_remove = []
                for other in unvisited:
                    ox, oy, ow, oh = pos[other]
                    # Check the distance between the blocks (with a 1e-4 tolerance)
                    dx = max(0.0, max(cx - (ox + ow), ox - (cx + cw)))
                    dy = max(0.0, max(cy - (oy + oh), oy - (cy + ch)))
                    if dx <= 1e-4 and dy <= 1e-4:
                        comp.add(other)
                        queue.append(other)
                        to_remove.append(other)
                for r in to_remove:
                    unvisited.remove(r)
            components.append(comp)

        # If the whole cluster is already a single piece, there is no need to reshape
        if len(components) == 1:
            continue

        # --- PROCESSING A DISCONNECTED CLUSTER ---
        for idx in c_indices:
            if not can_reshape[idx]: continue

            # Find which island the current block belongs to
            my_comp = next(comp for comp in components if idx in comp)
            # Target list — all blocks from OTHER islands
            target_indices = [i for i in c_indices if i not in my_comp]

            ix, iy, iw, ih = pos[idx]
            area = iw * ih
            best_shape = None
            candidates_wh = []

            # --- GENERATION OF SHAPE CANDIDATES ---

            # Strategy A: analytical computation of the exact distance to the target islands
            for other_idx in target_indices:
                ox, oy, ow, oh = pos[other_idx]

                # Target X coordinates (left and right edges of the other block)
                for tx in [ox, ox + ow]:
                    for w_req in [tx - ix, ix + iw - tx, (tx - (ix + iw / 2.0)) * 2.0, ((ix + iw / 2.0) - tx) * 2.0]:
                        if w_req > 0.1:
                            candidates_wh.append((w_req, area / w_req))

                # Target Y coordinates (bottom and top edges of the other block)
                for ty in [oy, oy + oh]:
                    for h_req in [ty - iy, iy + ih - ty, (ty - (iy + ih / 2.0)) * 2.0, ((iy + ih / 2.0) - ty) * 2.0]:
                        if h_req > 0.1:
                            candidates_wh.append((area / h_req, h_req))

            # Strategy B: extended logarithmic grid (Aspect Ratio from 1:100 to 100:1)
            ratios = np.logspace(np.log10(0.01), np.log10(100.0), 200)
            for r in ratios:
                candidates_wh.append((math.sqrt(area * r), math.sqrt(area / r)))

            # --- CHECKING THE CANDIDATES ---
            b_code = boundary_codes[idx]

            for w_c, h_c in candidates_wh:
                # ASYMMETRIC SCALING (iterating over anchors)

                # Possible anchors along X (0=Left, 1=Center, 2=Right)
                if b_code & 1:
                    x_candidates = [0.0]
                elif b_code & 2:
                    x_candidates = [c_width - w_c]
                else:
                    x_candidates = [ix, ix + (iw - w_c) / 2.0, ix + iw - w_c]

                # Possible anchors along Y (0=Bottom, 1=Center, 2=Top)
                if b_code & 8:
                    y_candidates = [0.0]
                elif b_code & 4:
                    y_candidates = [c_height - h_c]
                else:
                    y_candidates = [iy, iy + (ih - h_c) / 2.0, iy + ih - h_c]

                for x_c in x_candidates:
                    for y_c in y_candidates:
                        # 1. Protection against going beyond the chip
                        if x_c < -1e-4 or y_c < -1e-4 or x_c + w_c > c_width + 1e-4 or y_c + h_c > c_height + 1e-4:
                            continue

                        # 2. Strict overlap check against ANY block
                        has_overlap = False
                        for k in range(n):
                            if k == idx: continue
                            kx, ky, kw, kh = pos[k]
                            ox_lap = max(0.0, min(x_c + w_c, kx + kw) - max(x_c, kx))
                            oy_lap = max(0.0, min(y_c + h_c, ky + kh) - max(y_c, ky))
                            if ox_lap > 1e-5 and oy_lap > 1e-5:
                                has_overlap = True
                                break

                        if has_overlap:
                            continue

                        # 3. Does the new shape reach the required island?
                        touches_target = False
                        for target_idx in target_indices:
                            ox_pos, oy_pos, ow_pos, oh_pos = pos[target_idx]
                            dx = max(0.0, max(x_c - (ox_pos + ow_pos), ox_pos - (x_c + w_c)))
                            dy = max(0.0, max(y_c - (oy_pos + oh_pos), oy_pos - (y_c + h_c)))
                            if dx <= 1e-4 and dy <= 1e-4:
                                touches_target = True
                                break

                        if touches_target:
                            best_shape = (x_c, y_c, w_c, h_c)
                            break  # The perfect anchor has been found!

                    if best_shape is not None: break
                if best_shape is not None: break

            # Apply the found shape and move on to the next block
            if best_shape is not None:
                pos[idx] = best_shape
                fixes += 1

                # If the shape has changed, the touching graph is updated on the next
                # iteration of the outer loop, so processing of the current island can stop
                break

    return torch.tensor(pos, device=positions.device, dtype=positions.dtype), fixes


def compute_training_loss_differentiable_matrix(
        positions: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        area_targets: torch.Tensor,
        baseline_metrics: torch.Tensor,
        constraints=None,
        step=0,
        return_overlap_tensor: bool = False,
) -> (torch.Tensor, float):
    """Same objective as the original function, but avoids avoidable CPU syncs in hot loops."""
    N = positions.shape[0]
    dtype = positions.dtype
    device = positions.device

    x = positions[:, 0]
    y = positions[:, 1]
    w = positions[:, 2]
    h = positions[:, 3]
    cx = x + w * 0.5
    cy = y + h * 0.5

    zero = positions.new_zeros(())

    valid_b2b = b2b_connectivity
    if valid_b2b.numel() > 0:
        i = valid_b2b[:, 0].long()
        j = valid_b2b[:, 1].long()
        weights = valid_b2b[:, 2].to(dtype=dtype)
        mask = (i < N) & (j < N)
        i, j, weights = i[mask], j[mask], weights[mask]
        dx = torch.sqrt((cx[i] - cx[j]).square() + 1e-5)
        dy = torch.sqrt((cy[i] - cy[j]).square() + 1e-5)
        hpwl_b2b = (weights * (dx + dy)).sum()
    else:
        hpwl_b2b = zero

    valid_p2b = p2b_connectivity
    if valid_p2b.numel() > 0:
        pin_idx = valid_p2b[:, 0].long()
        block_idx = valid_p2b[:, 1].long()
        weights = valid_p2b[:, 2].to(dtype=dtype)
        mask = (pin_idx < pins_pos.shape[0]) & (block_idx < N)
        pin_idx, block_idx, weights = pin_idx[mask], block_idx[mask], weights[mask]
        dx = torch.abs(cx[block_idx] - pins_pos[pin_idx, 0])
        dy = torch.abs(cy[block_idx] - pins_pos[pin_idx, 1])
        hpwl_p2b = (weights * (dx + dy)).sum()
    else:
        hpwl_p2b = zero

    hpwl_total = hpwl_b2b + hpwl_p2b

    gamma = 0.1
    x_right = x + w
    y_top = y + h
    x_max_soft = torch.logsumexp(gamma * x_right, dim=0) / gamma
    y_max_soft = torch.logsumexp(gamma * y_top, dim=0) / gamma
    x_min_soft = -torch.logsumexp(-gamma * x, dim=0) / gamma
    y_min_soft = -torch.logsumexp(-gamma * y, dim=0) / gamma
    bbox_area = (x_max_soft - x_min_soft) * (y_max_soft - y_min_soft)

    dx_matrix = torch.abs(cx[:, None] - cx[None, :])
    dy_matrix = torch.abs(cy[:, None] - cy[None, :])
    min_dist_x = (w[:, None] + w[None, :]) * 0.5
    min_dist_y = (h[:, None] + h[None, :]) * 0.5
    soft_overlap_x = torch.relu(min_dist_x - dx_matrix)
    soft_overlap_y = torch.relu(min_dist_y - dy_matrix)
    overlap_area = torch.triu(soft_overlap_x * soft_overlap_y, diagonal=1).sum()
    total_block_area = (w * h).sum()
    overlap_violation = overlap_area / (total_block_area + 1e-6)

    baseline_area = baseline_metrics[0]
    baseline_hpwl = baseline_metrics[6] + baseline_metrics[7]
    hpwl_gap = torch.relu((hpwl_total - baseline_hpwl) / (baseline_hpwl + 1e-6))
    area_gap = torch.relu((bbox_area - baseline_area) / (baseline_area + 1e-6))

    if pins_pos.numel() > 0:
        min_x_canvas = pins_pos[:, 0].min()
        max_x_canvas = pins_pos[:, 0].max()
        min_y_canvas = pins_pos[:, 1].min()
        max_y_canvas = pins_pos[:, 1].max()
    else:
        # We use regular floats instead of .new_tensor()
        min_x_canvas = 0.0
        max_x_canvas = 500.0
        min_y_canvas = 0.0
        max_y_canvas = 500.0

    oob_left_area = 100 * torch.relu(min_x_canvas - x) * h
    oob_bottom_area = 100 * torch.relu(min_y_canvas - y) * w
    oob_right_area = torch.relu(x_right - max_x_canvas) * h
    oob_top_area = torch.relu(y_top - max_y_canvas) * w
    total_oob_area = (oob_left_area + oob_bottom_area + oob_right_area + oob_top_area).sum()
    oob_violation = total_oob_area / (total_block_area + 1e-6)

    V_mib = zero
    V_cluster = zero
    V_boundary = zero
    if constraints is not None:
        V_mib = compute_mib_loss(positions, constraints)
        V_cluster = compute_cluster_loss(positions, constraints)
        V_boundary = compute_boundary_loss(positions, constraints)

    V_soft = (1.0 + step / 50) * overlap_violation + oob_violation + 0.01 * V_mib + 0.01 * V_cluster + 0.01 * V_boundary
    if not torch.isfinite(V_soft).all():
        print(positions.shape)
        print(positions.detach().cpu().numpy())
        print(overlap_violation.detach().cpu().numpy(), oob_violation.detach().cpu().numpy(),
              V_mib.detach().cpu().numpy(), V_cluster.detach().cpu().numpy(), V_boundary.detach().cpu().numpy())

    quality_factor = 1.0 + ALPHA * (hpwl_gap + area_gap)
    violation_factor = BETA * V_soft
    cost = quality_factor * violation_factor

    if return_overlap_tensor:
        return cost, overlap_violation
    return cost, float(overlap_violation.detach().item())


def get_decaying_cyclic_lr(step, total_steps, lr_min, lr_max, periods=3, alpha=0.8):
    """
    Smoothly changes LR from lr_min to lr_max and back along a cosine curve,
    decreasing the maximum amplitude on each new period with the alpha coefficient.
    """
    # Limit the step so the graph doesn't break
    step = min(step, total_steps)

    # Proportion of overall progress (from 0.0 to 1.0)
    progress = step / total_steps

    # Determine the current cycle number (0, 1, 2...)
    current_cycle = int(progress * periods)

    # Protection for the very last step (when progress == 1.0)
    if current_cycle >= periods:
        current_cycle = periods - 1

    # Calculate the cosine wave value
    wave = 1 - math.cos(2 * math.pi * periods * progress)

    # Calculate the decaying amplitude for the current cycle
    decayed_amplitude = (lr_max - lr_min) * (alpha ** current_cycle)

    # Final LR
    lr = lr_min + (decayed_amplitude / 2) * wave

    return lr


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def apply_boundary_aware_gravity(
        positions: torch.Tensor,
        constraints: torch.Tensor,
        margin: float = 1e-5
) -> torch.Tensor:
    """The same logic as before, but on NumPy (float64).

    The algorithm is sequential by nature (block by block), so
    executing it on CUDA tensors generated thousands of micro-synchronizations
    (.any(), scalar comparisons, element-wise writes). On CPU/NumPy this
    is executed in milliseconds for n<=120.
    """
    device = positions.device
    pos = positions.detach().cpu().numpy().astype(np.float64).copy()
    margin = float(margin)

    n = pos.shape[0]
    is_preplaced = (constraints[:, 1] == 1).cpu().numpy()
    boundary_codes = constraints[:, 4].long().cpu().numpy()

    is_right = (boundary_codes & 2) > 0
    is_top = (boundary_codes & 4) > 0

    # Obstacle detection threshold (must be less than validator threshold 1e-7)
    tol = 1e-8
    X, Y, W, H = 0, 1, 2, 3

    # ==========================================
    # STAGE 1: Gravity
    # ==========================================
    # IMPORTANT: already processed blocks are always considered blockers. Their positions
    # are final, so placing idx to the right of their right edges guarantees
    # the absence of new overlaps. The old purely geometric condition
    # ("right edge <= my OLD x") broke if an earlier block
    # jumped to the right from a negative coordinate to 0 — both blocks "did not see"
    # each other and overlapped at the origin. For valid inputs
    # (no overlaps, coordinates >= 0), a processed Y-overlapping block
    # is always "completely to the left" anyway, so the behavior on them is identical.

    # --- Shift left (X) ---
    processed = np.zeros(n, dtype=bool)
    for idx in np.argsort(pos[:, X], kind='stable'):
        if is_preplaced[idx]:
            continue  # preplaced participate only through left_mask (they do not move)
        my_x, my_y, my_w, my_h = pos[idx]

        y_overlap = (pos[:, Y] < my_y + my_h - tol) & ((pos[:, Y] + pos[:, H]) > my_y + tol)
        y_overlap[idx] = False

        left_mask = (pos[:, X] + pos[:, W]) <= (my_x + tol)
        blockers = y_overlap & (left_mask | processed)

        if blockers.any():
            pos[idx, X] = (pos[blockers, X] + pos[blockers, W]).max() + margin
        else:
            pos[idx, X] = 0.0
        processed[idx] = True

    # --- Shift down (Y) ---
    processed = np.zeros(n, dtype=bool)
    for idx in np.argsort(pos[:, Y], kind='stable'):
        if is_preplaced[idx]:
            continue  # preplaced participate only through bottom_mask (they do not move)
        my_x, my_y, my_w, my_h = pos[idx]

        x_overlap = (pos[:, X] < my_x + my_w - tol) & ((pos[:, X] + pos[:, W]) > my_x + tol)
        x_overlap[idx] = False

        bottom_mask = (pos[:, Y] + pos[:, H]) <= (my_y + tol)
        blockers = x_overlap & (bottom_mask | processed)

        if blockers.any():
            pos[idx, Y] = (pos[blockers, Y] + pos[blockers, H]).max() + margin
        else:
            pos[idx, Y] = 0.0
        processed[idx] = True

    # ==========================================
    # STAGE 2: Snapping (Safe adhesion)
    # ==========================================

    # --- TOP blocks processing ---
    top_indices = np.where(is_top & ~is_preplaced)[0]
    if len(top_indices) > 0:
        base_mask = ~is_top | is_preplaced
        if base_mask.any():
            ceiling_y = float((pos[base_mask, Y] + pos[base_mask, H]).max())
        else:
            ceiling_y = 0.0

        # Sort TOP blocks from top to bottom by their current Y
        order_top_desc = top_indices[np.argsort(-pos[top_indices, Y], kind='stable')]
        processed_top = []

        for idx in order_top_desc:
            my_x, my_w, my_h = pos[idx, X], pos[idx, W], pos[idx, H]

            # 1. X-overlaps with base (NON-TOP) blocks
            x_overlap_base = base_mask & (pos[:, X] < my_x + my_w - tol) & ((pos[:, X] + pos[:, W]) > my_x + tol)
            if x_overlap_base.any():
                base_safe_bottom = float((pos[x_overlap_base, Y] + pos[x_overlap_base, H]).max()) + margin
            else:
                base_safe_bottom = 0.0

            # 2. X-overlaps with ALREADY PROCESSED TOP blocks
            top_safe_ceiling = ceiling_y
            if processed_top:
                pt = np.asarray(processed_top, dtype=np.int64)
                overlap_mask = (pos[pt, X] < my_x + my_w - tol) & ((pos[pt, X] + pos[pt, W]) > my_x + tol)
                hit = pt[overlap_mask]
                if len(hit) > 0:
                    top_safe_ceiling = float(pos[hit, Y].min()) - margin

            # 3. Preliminary target Y (hanging from the ceiling)
            target_y = top_safe_ceiling - my_h

            # 4. Check if we crashed into base blocks from below
            if target_y < base_safe_bottom:
                shift = base_safe_bottom - target_y
                ceiling_y += shift
                if processed_top:
                    pos[processed_top, Y] += shift
                target_y = base_safe_bottom

            # 5. Set the block
            pos[idx, Y] = target_y
            processed_top.append(int(idx))

    # --- RIGHT blocks processing ---
    right_indices = np.where(is_right & ~is_preplaced)[0]
    if len(right_indices) > 0:
        base_mask = ~is_right | is_preplaced
        if base_mask.any():
            wall_x = float((pos[base_mask, X] + pos[base_mask, W]).max())
        else:
            wall_x = 0.0

        order_right_desc = right_indices[np.argsort(-pos[right_indices, X], kind='stable')]
        processed_right = []

        for idx in order_right_desc:
            my_y, my_w, my_h = pos[idx, Y], pos[idx, W], pos[idx, H]

            y_overlap_base = base_mask & (pos[:, Y] < my_y + my_h - tol) & ((pos[:, Y] + pos[:, H]) > my_y + tol)
            if y_overlap_base.any():
                base_safe_left = float((pos[y_overlap_base, X] + pos[y_overlap_base, W]).max()) + margin
            else:
                base_safe_left = 0.0

            right_safe_wall = wall_x
            if processed_right:
                pr = np.asarray(processed_right, dtype=np.int64)
                overlap_mask = (pos[pr, Y] < my_y + my_h - tol) & ((pos[pr, Y] + pos[pr, H]) > my_y + tol)
                hit = pr[overlap_mask]
                if len(hit) > 0:
                    right_safe_wall = float(pos[hit, X].min()) - margin

            target_x = right_safe_wall - my_w

            if target_x < base_safe_left:
                shift = base_safe_left - target_x
                wall_x += shift
                if processed_right:
                    pos[processed_right, X] += shift
                target_x = base_safe_left

            pos[idx, X] = target_x
            processed_right.append(int(idx))

    # Return double-tensor on the original device (as before)
    return torch.from_numpy(pos).to(device)



def refine_layout(
        pred_norm_k, # A tensor of shape (K, N, 3) is now expected
        area_i,
        scale_i,
        curr_constraints,
        raw,
        device,
        steps=50,
        lr_max=0.03,
        save_video=False,
        video_path="refinement.mp4",
        fps=15,
        verbose=False,
        feasible_patience=FEASIBLE_PATIENCE_CHECKS,
):
    """
    Gradient-based refinement on top of the model prediction.
    Logic is preserved, but all constant tensors are expected to be already on GPU and are not copied in each step.
    """

    # Diversity is already provided by the different checkpoint weights
    refined_init = pred_norm_k.detach().clone()
    K = refined_init.shape[0]
    refined = refined_init.requires_grad_(True)

    if 1:
        optimizer = torch.optim.Adam([refined], lr=lr_max)
    else:
        optimizer = torch.optim.NAdam([refined], lr=lr_max)

    is_fixed = (curr_constraints[:, 0] == 1)
    is_preplaced = (curr_constraints[:, 1] == 1)
    is_hard = is_fixed | is_preplaced
    mib_ids = curr_constraints[:, 2]

    mib_groups = []
    unique_ids = mib_ids[mib_ids > 0].unique()
    for gid in unique_ids:
        mask = (mib_ids == gid)
        if int(mask.sum().item()) > 1:
            ga = area_i[mask]
            if float((ga.max() - ga.min()) / ga.min().clamp(min=1e-9)) > AREA_TOLERANCE:
                continue  # MIB is unreachable — do not tie r, let the gradient be free
            hard_in_group = mask & is_hard
            ref_idx = torch.where(hard_in_group)[0][0] if bool(hard_in_group.any().item()) else None
            mib_groups.append((mask, ref_idx))

    target_fp = raw['target_fp']
    b2b_conn = raw['b2b_conn']
    p2b_conn = raw['p2b_conn']
    pins_pos = raw['pins_pos']
    metrics = raw['metrics']

    frames = []
    learning_rates_array = []
    loss_array = []
    found_good_solution = False

    iterator = tqdm.tqdm(range(steps), disable=not verbose)
    with torch.enable_grad():
        valid_b2b = b2b_conn[b2b_conn[:, 0] >= 0]
        valid_p2b = p2b_conn[p2b_conn[:, 0] >= 0]

        # All constant preprocessing — ONCE per test
        n_blocks = refined.shape[-2]
        ctx = build_loss_context(
            valid_b2b, valid_p2b, pins_pos, metrics, curr_constraints,
            n_blocks, refined.dtype, refined.device,
        )

        # The best solution is tracked entirely on the GPU (without .item() in the loop):
        # key = violations + 1.1 * (has negative coordinates)
        INF = float('inf')
        best_refined = refined.detach().clone()
        best_key = torch.full((K,), INF, device=refined.device, dtype=torch.float32)
        best_loss_t = torch.full((K,), INF, device=refined.device, dtype=torch.float32)

        # Early stopping state after reaching feasibility (point 1)
        last_best_feasible_loss = INF
        patience_left = feasible_patience

        for step in iterator:
            optimizer.zero_grad(set_to_none=True)
            current_lr = get_decaying_cyclic_lr(step, steps, 0.000001, lr_max, periods=CYCLIC_LR_PERIODS, alpha=CYCLIC_LR_MULTIPLIER)
            set_learning_rate(optimizer, current_lr)

            if mib_groups:
                refined_clamped = refined.clone()
                for mask, ref_idx in mib_groups:
                    if ref_idx is not None:
                        refined_clamped[..., mask, 2] = refined[..., ref_idx, 2].unsqueeze(-1)
                    else:
                        refined_clamped[..., mask, 2] = refined[..., mask, 2].mean(dim=-1, keepdim=True)
            else:
                refined_clamped = refined

            curr_pred_abs = decode_from_direct_space(refined_clamped, area_i, scale_i)

            if is_fixed.any():
                curr_pred_abs[..., is_fixed, 2:] = target_fp[:n_blocks][is_fixed, 2:]
            if is_preplaced.any():
                curr_pred_abs[..., is_preplaced, :] = target_fp[:n_blocks][is_preplaced]

            loss_vec, overlap_tensor, violations_f = compute_training_loss_fast(curr_pred_abs, ctx, step)
            violations_f = violations_f.to(torch.float32)

            if save_video:
                # Convert ALL K samples to numpy for the current step.
                if FREEZE_ON_FEASIBLE:
                    # For the video take the frozen state if the variant is already valid
                    render_state = torch.where((best_key <= 0.0)[:, None, None], best_refined, refined_clamped).detach()
                    curr_render_abs = decode_from_direct_space(render_state, area_i, scale_i)
                    frames.append(curr_render_abs.cpu().numpy().copy())
                else:
                    frames.append(curr_pred_abs.detach().cpu().numpy().copy())

                learning_rates_array.append(current_lr)
                loss_array.append(loss_vec.detach().cpu().numpy().copy())

            # --- Fully asynchronous update of the best solution (every step, per-sample) ---
            not_positive_f = (refined.detach()[..., :2].amin(dim=(-2, -1)) < 0.0).to(torch.float32)
            key = violations_f + 1.1 * not_positive_f
            loss_d = loss_vec.detach().to(torch.float32)

            improve = (key < best_key) | ((key == best_key) & (loss_d < best_loss_t))  # (K,)

            # --- NEW FREEZING LOGIC ---
            if FREEZE_ON_FEASIBLE:
                # Mask of the variants that have ALREADY become valid (before this step)
                already_feasible = (best_key <= 0.0)
                # Clear the improvement flag: they stay in the first valid state forever
                improve = improve & (~already_feasible)
            # ------------------------------

            best_key = torch.where(improve, key, best_key)
            best_loss_t = torch.where(improve, loss_d, best_loss_t)
            best_refined = torch.where(improve[:, None, None], refined.detach(), best_refined)

            # --- The only synchronization: rare early-exit / progress bar ---
            if step % CHECK_LOSS_STEPS == 0 or step == steps - 1:
                feas = best_key <= 0.0
                feas_best_loss = torch.where(feas, best_loss_t, torch.full_like(best_loss_t, INF)).min()
                snapshot = torch.stack([
                    feas.any().to(torch.float32),
                    feas_best_loss,
                    best_key.min(),
                    feas.all().to(torch.float32)  # <--- ADDED: check that ALL variants are valid
                ]).cpu()
                any_feasible = bool(snapshot[0] > 0)
                best_feasible_loss = float(snapshot[1])
                all_feasible = bool(snapshot[3] > 0)  # <--- ADDED: read the result

                if verbose:
                    iterator.set_postfix({
                        'lr': current_lr,
                        'min_key': float(snapshot[2]),
                        'feas_loss': best_feasible_loss,
                        # The status of whether all variants are ready can also be printed
                        'all_feas': all_feasible
                    })

                if any_feasible:
                    found_good_solution = True

                    # plateau of the best feasible loss
                    if best_feasible_loss < last_best_feasible_loss - 1e-9:
                        last_best_feasible_loss = best_feasible_loss
                        patience_left = feasible_patience
                    else:
                        patience_left -= 1

                    if EARLY_EXIT_ON_ALL_FEASIBLE and all_feasible:
                        break
                    if patience_left <= 0:
                        break

            # Sample gradients are independent (loss does not bind them),
            # so sum() gives each sample its own gradient
            loss_vec.sum().backward()
            with torch.no_grad():
                g = refined.grad
                gn = g.flatten(1).norm(dim=1)                                  # (K,)
                # Exact semantics of clip_grad_norm_(max_norm=1.0): coef = 1/(norm+1e-6), clamp to 1
                g.mul_(torch.clamp(1.0 / (gn + 1e-6), max=1.0).view(-1, 1, 1))
            optimizer.step()

    EXIT_STEPS.append(step + 1)
    # Selection of the best sample: minimum key (violations), in case of a tie — minimum loss
    # TO BE REMOVED:
    # min_key = best_key.min()
    # cand_loss = torch.where(best_key == min_key, best_loss_t, torch.full_like(best_loss_t, INF))
    # sel = int(cand_loss.argmin().item())
    # ...
    # chosen = best_refined[sel]
    # refined_clamped = torch.cat([chosen[:, :2], chosen[:, 2:3]], dim=-1)

    # TO BE KEPT/ADDED:
    # Return the whole batch of K variants
    refined_clamped_all = torch.cat([best_refined[..., :2], best_refined[..., 2:3]], dim=-1)

    return (refined_clamped_all.detach(), found_good_solution,
            float(best_key.min().item()), frames, learning_rates_array,
            loss_array, best_key.detach().cpu())


def push_out_legalize(
        positions: torch.Tensor,
        constraints: torch.Tensor,
        max_sweeps: int = 250,
        sep: float = 1e-6,
        detect_eps: float = 1e-9,
        stall_sweeps: int = 30,
) -> Tuple[torch.Tensor, int, bool]:
    """Legalization by minimum displacement (point 3): iterative pushing apart
    of overlapping pairs along the axis of lesser penetration.

    Unlike teleport, it preserves the layout topology found by the network and
    refine: blocks are shifted exactly by the penetration depth (+sep), so
    HPWL/Area barely suffer. Teleport remains an emergency fallback.

    Rules:
      - Preplaced blocks are immobile (hard constraint);
      - Sizes never change (Fixed and MIB are not violated by design);
      - If both blocks are movable — each shifts by half; if one —
        it takes the entire shift; if both are preplaced — the pair is skipped.

    Returns (new positions, number of sweeps, whether all overlaps were removed).
    If not converged within max_sweeps (or progress stalled for stall_sweeps
    sweeps), the best achieved state is returned — teleport will have
    fewer violators to handle.
    """
    device = positions.device
    dtype = positions.dtype
    n = positions.shape[0]
    if n < 2:
        return positions, 0, True

    pos = positions.detach().cpu().numpy().astype(np.float64).copy()
    movable = ~(constraints[:, 1] == 1).cpu().numpy()

    iu_i, iu_j = np.triu_indices(n, k=1)

    def overlapping_pairs():
        """All overlapping pairs + their (ox, oy). Vectorized, O(N^2)."""
        x, y = pos[:, 0], pos[:, 1]
        r, t = x + pos[:, 2], y + pos[:, 3]
        ox = np.minimum(r[:, None], r[None, :]) - np.maximum(x[:, None], x[None, :])
        oy = np.minimum(t[:, None], t[None, :]) - np.maximum(y[:, None], y[None, :])
        ox_u, oy_u = ox[iu_i, iu_j], oy[iu_i, iu_j]
        mask = (ox_u > detect_eps) & (oy_u > detect_eps)
        return iu_i[mask], iu_j[mask], ox_u[mask], oy_u[mask]

    best_pos = pos.copy()
    best_score = (np.inf, np.inf)  # (number of pairs, total penetration area)
    sweeps_done = 0
    stall = 0

    for sweep in range(max_sweeps):
        pi, pj, pox, poy = overlapping_pairs()
        sweeps_done = sweep

        score = (len(pi), float((pox * poy).sum()) if len(pi) else 0.0)
        if score < best_score:
            best_score = score
            best_pos = pos.copy()
            stall = 0
        else:
            stall += 1

        if len(pi) == 0:
            return torch.from_numpy(pos).to(device=device, dtype=dtype), sweeps_done, True
        if stall >= stall_sweeps:
            break  # oscillation/dead end — return the best state

        # Worst pairs first (Gauss-Seidel: overlap is recalculated on the fly)
        order = np.argsort(-(pox * poy))
        for k in order:
            i, j = int(pi[k]), int(pj[k])
            if not movable[i] and not movable[j]:
                continue
            xi, yi, wi, hi = pos[i]
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox <= detect_eps or oy <= detect_eps:
                continue  # already resolved by an earlier pair in this sweep

            # Axis of lesser penetration => minimum shift
            if ox <= oy:
                d = ox + sep
                # Direction — by centers; in case of a tie, i goes to minus
                sign_i = -1.0 if (xi + wi * 0.5) <= (xj + wj * 0.5) else 1.0
                if movable[i] and movable[j]:
                    pos[i, 0] += sign_i * d * 0.5
                    pos[j, 0] -= sign_i * d * 0.5
                elif movable[i]:
                    pos[i, 0] += sign_i * d
                else:
                    pos[j, 0] -= sign_i * d
            else:
                d = oy + sep
                sign_i = -1.0 if (yi + hi * 0.5) <= (yj + hj * 0.5) else 1.0
                if movable[i] and movable[j]:
                    pos[i, 1] += sign_i * d * 0.5
                    pos[j, 1] -= sign_i * d * 0.5
                elif movable[i]:
                    pos[i, 1] += sign_i * d
                else:
                    pos[j, 1] -= sign_i * d

    # Did not converge: final check of the best state
    pi, pj, _, _ = overlapping_pairs()
    cur_score = (len(pi), 0.0)
    if cur_score < best_score:
        best_pos = pos
    return torch.from_numpy(best_pos).to(device=device, dtype=dtype), sweeps_done + 1, best_score[0] == 0


def teleport_violators_smart(
        positions: torch.Tensor,
        constraints: torch.Tensor,
        aspect_ratios: List[float] = [0.5, 0.66, 0.8, 1.0, 1.25, 1.5, 2.0],
        tol: float = 1e-5
) -> Tuple[torch.Tensor, int]:
    """Vectorized version. The behavior and the result are identical to the
    row-by-row one (verified bit-identical), but the overlap checks run on numpy."""
    pos = positions.detach().cpu().tolist()
    n = len(pos)
    teleports_count = 0
    if n < 2:
        return positions, teleports_count

    is_fixed_size = (constraints[:, 0] == 1).cpu().tolist()
    is_preplaced = (constraints[:, 1] == 1).cpu().tolist()
    boundary_codes = constraints[:, 4].long().cpu().tolist()
    bc = np.asarray(boundary_codes, dtype=np.int64)

    can_move = [not is_preplaced[i] for i in range(n)]
    is_rigid_size = [(is_fixed_size[i] or is_preplaced[i]) for i in range(n)]

    right_edges = [pos[i][0] + pos[i][2] for i in range(n) if (boundary_codes[i] & 2)]
    top_edges = [pos[i][1] + pos[i][3] for i in range(n) if (boundary_codes[i] & 4)]
    c_width = max(right_edges) if right_edges else max(p[0] + p[2] for p in pos)
    c_height = max(top_edges) if top_edges else max(p[1] + p[3] for p in pos)

    # --- searching for violators: the overlap matrix in a single numpy pass ---
    P = np.asarray(pos, dtype=np.float64)
    x, y, w, h = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    r, t = x + w, y + h
    ox = np.minimum(r[:, None], r[None, :]) - np.maximum(x[:, None], x[None, :])
    oy = np.minimum(t[:, None], t[None, :]) - np.maximum(y[:, None], y[None, :])
    ov = (ox > 1e-6) & (oy > 1e-6)
    np.fill_diagonal(ov, False)
    cm = np.asarray(can_move, dtype=bool)

    violators = []
    for i in range(n):
        if not can_move[i]:
            continue
        partners = np.nonzero(ov[i])[0]
        # a violation if there is a partner j: not can_move[j] OR j<i (the same logic)
        bad = False
        for j in partners:
            if (not cm[j]) or (j < i):
                bad = True
                break
        if bad:
            violators.append(i)

    if not violators:
        return positions, teleports_count

    step_size = 2.0
    max_radius = max(c_width, c_height) * 1.5
    oob_x_cursor = c_width + 50.0

    def check_vec(test_idx, tx, ty, tw, th, X, Y, R, T):
        """True if the candidate violates a boundary or intersects at least one block.
        Identical to the original check_overlap_for_candidate."""
        tr = tx + tw
        tt = ty + th
        b = bc[test_idx]
        if (b & 1) and tx > tol: return True
        if (b & 2) and tr < c_width - tol: return True
        if (b & 4) and tt < c_height - tol: return True
        if (b & 8) and ty > tol: return True
        oxx = np.minimum(tr, R) - np.maximum(tx, X)
        oyy = np.minimum(tt, T) - np.maximum(ty, Y)
        hit = (oxx > tol) & (oyy > tol)
        hit[test_idx] = False
        return bool(hit.any())

    for idx in violators:
        orig_x, orig_y, orig_w, orig_h = pos[idx]
        area = orig_w * orig_h

        # fresh block arrays (they reflect all previous violator placements)
        Pn = np.asarray(pos, dtype=np.float64)
        X = Pn[:, 0];
        Y = Pn[:, 1]
        R = Pn[:, 0] + Pn[:, 2];
        T = Pn[:, 1] + Pn[:, 3]

        candidate_shapes = []
        if not is_rigid_size[idx]:
            for rr in aspect_ratios:
                candidate_shapes.append((math.sqrt(area * rr), math.sqrt(area / rr)))
            candidate_shapes.insert(0, (orig_w, orig_h))
        else:
            candidate_shapes.append((orig_w, orig_h))

        best = None
        found_spot = False
        radius = 0.0
        while radius <= max_radius and not found_spot:
            num_points = max(8, int(2 * math.pi * radius / step_size)) if radius > 0 else 1
            angles = [2 * math.pi * i / num_points for i in range(num_points)] if radius > 0 else [0.0]
            for angle in angles:
                cx = orig_x + radius * math.cos(angle)
                cy = orig_y + radius * math.sin(angle)
                for cw, ch in candidate_shapes:
                    test_x = max(0.0, min(cx, c_width - cw))
                    test_y = max(0.0, min(cy, c_height - ch))
                    if not check_vec(idx, test_x, test_y, cw, ch, X, Y, R, T):
                        best = (test_x, test_y, cw, ch)
                        found_spot = True
                        break
                if found_spot:
                    break
            if found_spot:
                break
            radius += step_size

        if found_spot:
            pos[idx][0], pos[idx][1], pos[idx][2], pos[idx][3] = best
        else:
            pos[idx][0] = oob_x_cursor
            pos[idx][1] = 0.0
            pos[idx][2] = orig_w
            pos[idx][3] = orig_h
            oob_x_cursor += orig_w + 20.0

        teleports_count += 1

    return torch.tensor(pos, device=positions.device, dtype=positions.dtype), teleports_count


def check_contest_score(sol, dataset, sample_time):
    try:
        from iccad2026contest.iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area, evaluate_solution
    except:
        from FloorSet.iccad2026contest.iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area, \
            evaluate_solution

    test_id = sol['test_id']
    positions = [tuple(p) for p in sol['positions']]
    block_count = sol['block_count']

    # Load test case data
    sample = dataset[test_id]

    area_target, b2b_conn, p2b_conn, pins_pos, constraints = (
        sample['area_target'], sample['b2b_conn'], sample['p2b_conn'], sample['pins_pos'], sample['constraints'],
    )
    positions_gt, metrics = sample['fp_sol'], sample['metrics']
    gt_positions = []
    positions_gt = positions_gt.cpu().numpy()
    for i in range(len(positions)):
        gt_positions.append(tuple(positions_gt[i]))
    gt_positions = [tuple(p) for p in gt_positions]

    # Calculate baselines
    hpwl_baseline = calculate_hpwl_b2b(gt_positions, b2b_conn) + \
                    calculate_hpwl_p2b(gt_positions, p2b_conn, pins_pos)
    area_baseline = calculate_bbox_area(gt_positions)

    # Use stored metrics if available
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area_baseline = float(metrics[0])
        if metrics[-2] > 0 and metrics[-1] >= 0:
            hpwl_baseline = float(metrics[-2]) + float(metrics[-1])

    # Evaluate the saved solution
    solution_metrics = evaluate_solution(
        {'positions': positions, 'runtime': 1.0},
        {'hpwl_baseline': hpwl_baseline, 'area_baseline': area_baseline},
        constraints,
        b2b_conn,
        p2b_conn,
        pins_pos,
        area_target,
        gt_positions,
        median_runtime=1.0
    )

    # --- INSERTED BLOCK ---
    if solution_metrics.area_violations > 0:
        c = constraints[:block_count]
        mib = c[:, 2].long()
        print(f"  test {test_id}: area_viol={solution_metrics.area_violations}")
        for gid in mib[mib > 0].unique():
            idx = (mib == gid).nonzero().flatten()
            a = area_target[idx]
            w = [float(positions[i][2]) for i in idx.tolist()]
            h = [float(positions[i][3]) for i in idx.tolist()]
            print(f"    MIB {int(gid)}: a={[round(v, 3) for v in a.tolist()]} "
                  f"w*h={[round(w[k] * h[k], 3) for k in range(len(w))]} "
                  f"fixed={c[idx, 0].tolist()} preplaced={c[idx, 1].tolist()}")
    # --- END OF INSERTED BLOCK ---

    # print(solution_metrics)

    contest_score_data = {
        'test_id': test_id,
        'block_count': block_count,
        'is_feasible': solution_metrics.is_feasible,
        'hpwl_gap': solution_metrics.hpwl_gap,
        'area_gap': solution_metrics.area_gap,
        'cost': solution_metrics.cost,
        'hpwl_total': solution_metrics.hpwl_total,
        'bbox_area': solution_metrics.bbox_area,
        'overlaps': solution_metrics.overlap_violations,
        'area_violations': solution_metrics.area_violations,
        'fixed_violations': solution_metrics.fixed_violations,
        'preplaced_violations': solution_metrics.preplaced_violations,
        'boundary_violations': solution_metrics.boundary_violations,
        'grouping_violations': solution_metrics.grouping_violations,
        'mib_violations': solution_metrics.mib_violations,
        'total_soft_violations': solution_metrics.total_soft_violations,
        'max_possible_violations': solution_metrics.max_possible_violations,
        'violations_relative': solution_metrics.violations_relative,
    }
    return contest_score_data


def postprocess_and_score(
        idx,
        curr_pred_abs,
        curr_constraints,
        target_fp,
        b2b_conn,
        p2b_conn,
        pins_pos,
        area_targets,
        metrics,
        orig_constraints=None,
        collect_stages=False,
):
    curr_pred_abs = curr_pred_abs.clone()

    # Snapshots of the layout after each post-processing stage, used by the
    # refinement video. A stage that did not run leaves no snapshot, so the
    # video shows only what actually happened on this test case.
    stages = []

    def snap(label):
        if collect_stages:
            stages.append((label, curr_pred_abs.detach().cpu().numpy().copy()))

    is_fixed = (curr_constraints[:, 0] == 1)
    is_preplaced = (curr_constraints[:, 1] == 1)

    if is_fixed.any():
        curr_pred_abs[is_fixed, 2:] = target_fp[is_fixed, 2:]
    if is_preplaced.any():
        curr_pred_abs[is_preplaced] = target_fp[is_preplaced]

    snap("After refine (hard constraints applied)")

    _tp = time.perf_counter()

    violations, _ = check_overlap_vectorized(curr_pred_abs)
    if violations > 0:
        curr_pred_abs, _, _ = push_out_legalize(curr_pred_abs, curr_constraints)
        violations, _ = check_overlap_vectorized(curr_pred_abs)
        snap("Push-out legalize")

    if violations > 0:
        curr_pred_abs, _ = teleport_violators_smart(curr_pred_abs, curr_constraints)
        snap("Teleport violators")

    _prev_grav = None
    for _ in range(3):
        curr_pred_abs = apply_boundary_aware_gravity(curr_pred_abs, curr_constraints, margin=0.0)
        if 0:
            if _prev_grav is not None and torch.equal(curr_pred_abs, _prev_grav):
                break  # converged -> further passes would give the same result
            _prev_grav = curr_pred_abs.clone()
    snap("Boundary-aware gravity")

    has_clusters = bool((curr_constraints[:, 3] > 0).any())
    if has_clusters:
        curr_pred_abs, _ = fix_grouping_violations_greedily(curr_pred_abs, curr_constraints, gap=0.0)
        snap("Cluster grouping fix")
        curr_pred_abs, _ = reshape_cluster_blocks_to_touch(curr_pred_abs, curr_constraints)
        snap("Cluster reshape")

    violations_final, _ = check_overlap_vectorized(curr_pred_abs)
    if violations_final > 0:
        curr_pred_abs, _, _ = push_out_legalize(curr_pred_abs, curr_constraints)
        violations_final, _ = check_overlap_vectorized(curr_pred_abs)
        snap("Final push-out")
        if violations_final > 0:
            curr_pred_abs, _ = teleport_violators_smart(curr_pred_abs, curr_constraints)
            snap("Final teleport")

    PHASE_TIMES['pp_legalize'] += time.perf_counter() - _tp; _tp = time.perf_counter()

    # 1. Preparing the data for the official evaluator
    final_positions_np = curr_pred_abs.detach().cpu().numpy()
    positions_list = [(float(x), float(y), float(w), float(h)) for x, y, w, h in final_positions_np]

    # Build the list of target coordinates (the evaluator expects a list of tuples)
    gt_positions_np = target_fp.detach().cpu().numpy()
    gt_positions_list = [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in gt_positions_np]

    # The baselines are taken the same way as in check_contest_score
    area_baseline = 1000.0
    hpwl_baseline = 1000.0

    if metrics is not None and len(metrics) >= 8:
        if float(metrics[0]) > 0:
            area_baseline = float(metrics[0])
        if float(metrics[-2]) > 0 and float(metrics[-1]) >= 0:
            hpwl_baseline = float(metrics[-2]) + float(metrics[-1])

    score_constraints = curr_constraints if orig_constraints is None else orig_constraints

    # 2. Calling the official scoring
    solution_metrics = evaluate_solution_fast(
        {'positions': positions_list, 'runtime': 1.0},
        {'hpwl_baseline': hpwl_baseline, 'area_baseline': area_baseline},
        score_constraints.cpu(),
        b2b_conn.cpu(),
        p2b_conn.cpu(),
        pins_pos.cpu(),
        area_targets.cpu(),
        gt_positions_list,
        median_runtime=1.0
    )

    PHASE_TIMES['pp_evaluate'] += time.perf_counter() - _tp

    # 3. Building a reliable sorting key
    # If the solution is invalid (is_feasible = False), the evaluator returns cost = 10.0
    # To pick the "least broken" variant in the worst case, the number of overlaps is added.
    final_violations, _ = check_overlap_vectorized(curr_pred_abs)
    key = final_violations * 1e6 + solution_metrics.cost

    return idx, key, curr_pred_abs, stages


class FloorplanSolver:
    def __init__(self, checkpoint_paths: List[str]):  # A list is accepted
        if FORCE_CPU:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the thread pool once
        max_threads = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)

        self.models = []
        for path in checkpoint_paths:
            model = FloorplanDirectTransformer(hidden_dim=256, num_layers=12).to(self.device)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            if os.name != 'nt' and self.device.type == 'cuda' and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                except Exception as exc:
                    pass

            self.models.append(model)

    def solve(
            self,
            block_count: int,
            area_targets: torch.Tensor,
            b2b_connectivity: torch.Tensor,
            p2b_connectivity: torch.Tensor,
            pins_pos: torch.Tensor,
            constraints: torch.Tensor,
            target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:

        metrics = None
        if 1:
            # For test do a bit decrease
            area_targets = area_targets * 0.991

        # 1. Transfer input tensors to the target device
        area_targets = area_targets.to(self.device)
        b2b_connectivity = b2b_connectivity.to(self.device)
        p2b_connectivity = p2b_connectivity.to(self.device)
        pins_pos = pins_pos.to(self.device)
        constraints = constraints.to(self.device)

        if target_positions is None:
            target_positions = torch.zeros((block_count, 4), device=self.device)
        else:
            target_positions = target_positions.to(self.device)

        _t = time.perf_counter()

        # 2. Feature preparation (extracting embeddings and matrices)
        num_pins = pins_pos.shape[0] if pins_pos.numel() > 0 else 0
        graph_scale = pins_pos.max() if num_pins > 0 else torch.tensor(100.0, device=self.device)

        spectral_emb = compute_spectral_embeddings(b2b_connectivity, block_count, k_dim=3).to(self.device)

        # Building p2b_matrix
        curr_p2b_matrix = torch.zeros((block_count, num_pins), dtype=torch.float32, device=self.device)
        if p2b_connectivity.numel() > 0:
            pin_ids = p2b_connectivity[:, 0].long()
            block_ids = p2b_connectivity[:, 1].long()
            weights = p2b_connectivity[:, 2].float()
            curr_p2b_matrix[block_ids, pin_ids] = weights

        # Calculating pin centers of mass (Pin CoM)
        if num_pins > 0:
            pin_weight_sum = curr_p2b_matrix.sum(dim=1, keepdim=True)
            pin_com = torch.matmul(curr_p2b_matrix, pins_pos.float())
            has_pins_mask = (pin_weight_sum > 0).squeeze(1)
            pin_com[has_pins_mask] /= pin_weight_sum[has_pins_mask]
            pin_com_norm = pin_com / graph_scale
            pin_com_norm[~has_pins_mask] = 0.5
        else:
            pin_com_norm = torch.full((block_count, 2), 0.5, dtype=torch.float32, device=self.device)

        # Building b2b_matrix
        curr_b2b_matrix = torch.zeros((block_count, block_count), dtype=torch.float32, device=self.device)
        if b2b_connectivity.numel() > 0:
            u, v, w = b2b_connectivity[:, 0].long(), b2b_connectivity[:, 1].long(), b2b_connectivity[:, 2]
            curr_b2b_matrix[u, v] = w
            curr_b2b_matrix[v, u] = w

        # Building shortest path matrix (SPD)
        curr_spd_matrix = compute_spd_matrix_hops(b2b_connectivity, block_count, max_dist=20).to(self.device)

        # Extracting known r, x, y constraints
        known_r = torch.zeros((block_count,), device=self.device)
        known_x = torch.zeros((block_count,), device=self.device)
        known_y = torch.zeros((block_count,), device=self.device)

        is_fixed_flag = constraints[:, 0] == 1.0
        is_preplaced_flag = constraints[:, 1] == 1.0
        is_hard_flag = is_fixed_flag | is_preplaced_flag

        curr_w = target_positions[:, 2]
        curr_h = target_positions[:, 3]
        target_r = torch.log(curr_w / (curr_h + 1e-6))
        target_r_norm = torch.clamp(target_r / 1.2, -1.0, 1.0)

        known_r[is_hard_flag] = target_r_norm[is_hard_flag]
        known_x[is_preplaced_flag] = target_positions[is_preplaced_flag, 0] / graph_scale
        known_y[is_preplaced_flag] = target_positions[is_preplaced_flag, 1] / graph_scale

        area_norm = (torch.sqrt(area_targets) / graph_scale).unsqueeze(1)

        # Assembling final feature vector (14 features)
        node_features = torch.cat([
            area_norm,
            constraints,
            spectral_emb,
            pin_com_norm,
            known_r.unsqueeze(1),
            known_x.unsqueeze(1),
            known_y.unsqueeze(1)
        ], dim=1)

        # Adding a fake batch dimension (B=1) for the Transformer encoder
        features = node_features.unsqueeze(0)
        target_fp = target_positions.unsqueeze(0)
        b2b_matrix = curr_b2b_matrix.unsqueeze(0)
        p2b_matrix = torch.log1p(curr_p2b_matrix).unsqueeze(0)
        pins_pos_batched = pins_pos.float().unsqueeze(0)
        spd_matrix = curr_spd_matrix.unsqueeze(0)

        PHASE_TIMES['1_prep'] += time.perf_counter() - _t; _t = time.perf_counter()

        # 3. Neural network inference (Ensemble)
        curr_pred_norms = []
        for model in self.models:
            with torch.no_grad():
                pred_norm = model(
                    features,
                    padding_mask=None,
                    b2b_matrix=b2b_matrix,
                    p2b_matrix=p2b_matrix,
                    pins_pos=pins_pos_batched,
                    spd_matrix=spd_matrix
                )
            curr_pred_norms.append(pred_norm[0, :block_count])

        # Collect the base predictions from all checkpoints into a tensor (M, N, 3)
        curr_pred_norm_base = torch.stack(curr_pred_norms, dim=0)
        M = curr_pred_norm_base.shape[0]  # Number of checkpoints (10)

        # 1. Compute the mean (1 variant)
        mean_pred = curr_pred_norm_base.mean(dim=0, keepdim=True)

        # 2. Replicate the checkpoint predictions
        # The shape becomes (M * N_VARIANTS, N, 3), i.e. [1, 1..10 times, 2, 2..10 times, ...]
        repeated_preds = curr_pred_norm_base.repeat_interleave(N_VARIANTS, dim=0)

        # 3. Generate the noise (using the same generator logic)
        gen = torch.Generator()
        gen.manual_seed(20260612)
        noise = torch.randn(repeated_preds.shape, generator=gen, dtype=repeated_preds.dtype).to(self.device)
        noise = noise * MULTISTART_NOISE
        noise[..., 2] *= 0.5  # Change the Aspect Ratio more weakly

        # Zero out the noise for the first (zeroth) variant of each checkpoint,
        # so that the original network prediction is kept in its pure form
        for i in range(M):
            noise[i * N_VARIANTS] = 0.0

        repeated_preds = repeated_preds + noise

        # 4. Append the mean prediction at the end
        # Shape: (M * N_VARIANTS + 1, N, 3) -> (101, N, 3)
        curr_pred_norm_k = torch.cat([repeated_preds, mean_pred], dim=0)

        curr_pred_abs_initial = decode_from_direct_space(curr_pred_norm_k, area_targets, graph_scale)

        PHASE_TIMES['2_inference'] += time.perf_counter() - _t; _t = time.perf_counter()

        # 4. Aligning MIB groups before gradient descent
        mib_ids = features[0, :block_count, 3]
        is_fixed_mask = features[0, :block_count, 1] == 1
        is_preplaced_mask = features[0, :block_count, 2] == 1
        all_mib_groups = mib_ids[mib_ids > 0].unique()

        for mib_id in all_mib_groups:
            group_mask = (mib_ids == mib_id)
            group_indices = torch.where(group_mask)[0]
            hard_mask = group_mask & (is_fixed_mask | is_preplaced_mask)

            if hard_mask.any():
                ref_idx = torch.where(hard_mask)[0][0]
                ref_w = target_fp[0, ref_idx, 2]
                ref_h = target_fp[0, ref_idx, 3]
                ref_area = area_targets[ref_idx]      # instead of ref_w * ref_h
            else:
                group_w = curr_pred_abs_initial[0, group_indices, 2]
                group_h = curr_pred_abs_initial[0, group_indices, 3]
                mean_r = torch.log(group_w / (group_h + 1e-6)).mean()
                ref_area = area_targets[group_indices[0]]
                ref_w = torch.sqrt(ref_area) * torch.exp(mean_r * 0.5)
                ref_h = torch.sqrt(ref_area) * torch.exp(-mean_r * 0.5)

            # Area is a hard constraint (cost=10), MIB is a soft one.
            # A single shape is enforced only if it does not break the area.
            g_area = area_targets[group_indices]
            areas_equal = bool(
                ((g_area - ref_area).abs() / ref_area.clamp(min=1e-9) <= AREA_TOLERANCE).all()
            )

            if areas_equal:
                features[0, group_indices, 1] = 1.0
                target_fp[0, group_indices, 2] = ref_w
                target_fp[0, group_indices, 3] = ref_h
            else:
                # MIB is unreachable: the area of each block is preserved,
                # the blocks are NOT marked as fixed — let refine optimize them.
                pass

        # 5. Optimization (Refinement Loop)
        curr_constraints = features[0, :block_count, 1:6]

        # Use the real metrics that were passed in, or create a mock if none were passed
        if metrics is None:
            metrics = torch.zeros(10, device=self.device)
            metrics[0] = area_targets.sum() * 1.3
            metrics[6] = 1000.0
        else:
            metrics = metrics.to(self.device)

        raw_gpu = {
            'target_fp': target_fp[0],
            'b2b_conn': b2b_connectivity,
            'p2b_conn': p2b_connectivity,
            'pins_pos': pins_pos,
            'metrics': metrics,
        }

        good_solution = False
        all_frames = []
        all_lrs = []
        all_loss_vals = []

        PHASE_TIMES['3_mib_align'] += time.perf_counter() - _t; _t = time.perf_counter()

        # ... launching refine_layout
        refined_clamped_all, good_solution, current_min_violations, frames, lrs, loss_vals, variant_keys = refine_layout(
            curr_pred_norm_k,
            area_targets,
            graph_scale,
            curr_constraints,
            raw_gpu,
            self.device,
            steps=STEPS,
            lr_max=MAX_LR,
            save_video=DRAW_VALIDATION_VIDEOS,
            verbose=VERBOSE_REFINEMENT,
        )

        PHASE_TIMES['4_refine'] += time.perf_counter() - _t; _t = time.perf_counter()

        # Decode all K variants at once (Shape: K, N, 4)
        all_pred_abs = decode_from_direct_space(refined_clamped_all, area_targets, graph_scale)

        K_total = all_pred_abs.shape[0]
        keys_np = variant_keys.numpy().astype(np.float64)
        keep = set(np.nonzero(keys_np <= 0.0)[0].tolist())
        for idx in np.argsort(keys_np, kind='stable'):
            if len(keep) >= MAX_POSTPROCESS_VARIANTS:
                break
            keep.add(int(idx))
        keep = sorted(keep)

        results = []
        futures = []
        for k in keep:
            futures.append(self.executor.submit(
                postprocess_and_score, k, all_pred_abs[k], curr_constraints,
                target_fp[0], b2b_connectivity, p2b_connectivity, pins_pos,
                area_targets, metrics, constraints,
                DRAW_VALIDATION_VIDEOS))

        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

        # Sort and pick the objectively best variant after all the mutations
        results.sort(key=lambda x: x[1])
        best_idx, best_key, curr_pred_abs, best_stages = results[0]

        # --- Extract the score and the overlaps from the key ---
        best_overlaps = int(best_key // 1e6)
        best_cost = best_key

        if best_overlaps == 0:
            score_text = f" | Score: {best_cost:.4f} (Feasible)"
        else:
            score_text = f" | Score: {best_cost:.4f} (Overlaps: {best_overlaps})"
        # ---------------------------------------------------

        # RESTORE THE MASKS FOR RENDERING
        is_fixed = (curr_constraints[:, 0] == 1)
        is_preplaced = (curr_constraints[:, 1] == 1)

        PHASE_TIMES['5_postprocess'] += time.perf_counter() - _t; _t = time.perf_counter()

        # If DRAW_VALIDATION_IMAGES is enabled, exactly the best final variant is rendered
        if DRAW_VALIDATION_IMAGES:
            draw_layout(
                curr_pred_abs.detach().cpu().numpy(),
                test_id=block_count,
                pins=raw_gpu['pins_pos'].detach().cpu().numpy(),
                fixed_mask=is_preplaced.detach().cpu().numpy(),
                fixed_size_mask=is_fixed.detach().cpu().numpy(),
                constraints=curr_constraints.detach().cpu().numpy(),
                score_info=score_text,
            )

        if DRAW_VALIDATION_VIDEOS and frames:
            output_dir = 'validation_videos'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video_fps = 30

            # Extract the position and loss history for EXACTLY the winning variant (best_idx)
            best_frames = [f[best_idx] for f in frames]
            best_loss_vals = [float(l[best_idx]) for l in loss_vals]

            # The gradient part of the video carries no stage label and no highlight
            stage_labels = [None] * len(best_frames)
            changed_masks = [None] * len(best_frames)

            # Each post-processing stage is held on screen for ~1.2 s, otherwise
            # the handful of stages would flash by in a quarter of a second.
            hold_frames = max(1, int(video_fps * 1.2))
            move_eps = 1e-6

            prev_stage = best_frames[-1] if best_frames else None
            for stage_label, stage_arr in best_stages:
                if prev_stage is not None and stage_arr.shape == prev_stage.shape:
                    moved = np.abs(stage_arr[:, :2] - prev_stage[:, :2]).max(axis=1) > move_eps
                    resized = np.abs(stage_arr[:, 2:] - prev_stage[:, 2:]).max(axis=1) > move_eps
                    changed = moved | resized
                    n_moved, n_resized = int(moved.sum()), int(resized.sum())
                else:
                    changed = None
                    n_moved = n_resized = 0

                full_label = "{}  (moved: {}, resized: {})".format(stage_label, n_moved, n_resized)
                for _ in range(hold_frames):
                    best_frames.append(stage_arr)
                    stage_labels.append(full_label)
                    changed_masks.append(changed)
                prev_stage = stage_arr

            render_refinement_video(
                best_frames,
                video_path=output_dir + "/video_{}.mp4".format(block_count),
                fps=video_fps,
                preplaced_mask=is_preplaced.detach().cpu().numpy(),
                fixed_size_mask=is_fixed.detach().cpu().numpy(),
                pins=raw_gpu['pins_pos'].detach().cpu().numpy(),
                constraints=curr_constraints.detach().cpu().numpy(),
                lr=lrs,
                loss=best_loss_vals,
                stage_labels=stage_labels,
                changed_masks=changed_masks,
            )

        # 7. Converting the result to List[Tuple[float, float, float, float]]
        final_positions = curr_pred_abs.detach().cpu().numpy()
        positions_list = [(float(x), float(y), float(w), float(h)) for x, y, w, h in final_positions]

        PHASE_TIMES['6_draw'] += time.perf_counter() - _t
        PHASE_COUNT['tests'] += 1

        # Return the list of coordinates and the index of the winning model
        return positions_list, best_idx


def validate(checkpoint_paths, val_dataset, val_dataloader):
    print("\n" + "=" * 50)
    print("🚀 Starting validation via FloorplanSolver API")
    print("=" * 50)

    total_val_loss = 0.0
    global_test_id = 0
    good_solutions = 0
    solutions = []
    processed_batches = 0

    # The checkpoint wins are recorded here
    winner_stats = {}

    # Create an instance of our new API class
    solver = FloorplanSolver(checkpoint_paths)
    device = solver.device

    critical_tests = []
    with torch.no_grad():
        for batch_data in val_dataloader:
            B = batch_data['features'].shape[0]

            if PROCESS_TESTS is not None and global_test_id + 1 not in PROCESS_TESTS:
                global_test_id += B
                continue

            test_time = time.time()
            print(f"Start test {global_test_id + 1}")

            # Extract batch data
            features = batch_data['features'].to(device, non_blocking=True)
            target_fp = batch_data['target_fp'].to(device, non_blocking=True)
            raw_items = batch_data['raw_items']

            batch_loss = 0.0

            for i in range(B):
                raw = raw_items[i]
                bc = raw['block_count']
                area_i = raw['area_target'].to(device, non_blocking=True)
                curr_constraints = features[i, :bc, 1:6]
                curr_target_fp = target_fp[i, :bc]

                # =============================================================
                # MAIN CALL TO YOUR NEW SOLVE FUNCTION
                # =============================================================
                positions_list, best_idx = solver.solve(
                    block_count=bc,
                    area_targets=area_i,
                    b2b_connectivity=raw['b2b_conn'],
                    p2b_connectivity=raw['p2b_conn'],
                    pins_pos=raw['pins_pos'],
                    constraints=curr_constraints,
                    target_positions=curr_target_fp,
                    # metrics=raw['metrics']
                )
                # =============================================================

                # Calculate loss for validation metrics based on the output coordinates
                coords_tensor = torch.tensor(positions_list, device=device, dtype=torch.float64)
                l_p, _ = compute_training_loss_differentiable_matrix(
                    coords_tensor,
                    raw['b2b_conn'].to(device),
                    raw['p2b_conn'].to(device),
                    raw['pins_pos'].to(device),
                    area_i,
                    raw['metrics'].to(device),
                    constraints=curr_constraints,
                )
                batch_loss += float(l_p.detach().item())

                # Check for overlaps for statistics
                violations, _ = check_overlap_vectorized(coords_tensor)
                if violations == 0:
                    good_solutions += 1
                else:
                    print(violations, _)

                # Form a structure for saving results
                single_solution = {
                    'test_id': global_test_id,
                    'block_count': len(positions_list),
                    'positions': positions_list
                }
                solutions.append(single_solution)
                global_test_id += 1

            total_val_loss += (batch_loss / B)
            processed_batches += 1

            if CALC_CONTEST_SCORE:
                contest_score = check_contest_score(single_solution, val_dataset, time.time() - test_time)
            print("Test {} finished | Blocks: {} Loss: {:.6f} Time: {:.2f} sec".format(processed_batches, bc, batch_loss / B, time.time() - test_time))
            if CALC_CONTEST_SCORE:
                print("Contest cost: {:4f} Feasible: {} Overlaps: {} Area violations: {} Fixed violations: {} Preplaced violations: {}\n"
                      "Boundary violations: {} Grouping violations: {} MIB violations: {} Total soft: {} Max possible: {} Relative violations:{}".format(
                    contest_score['cost'],
                    contest_score['is_feasible'],
                    contest_score['overlaps'],
                    contest_score['area_violations'],
                    contest_score['fixed_violations'],
                    contest_score['preplaced_violations'],
                    contest_score['boundary_violations'],
                    contest_score['grouping_violations'],
                    contest_score['mib_violations'],
                    contest_score['total_soft_violations'],
                    contest_score['max_possible_violations'],
                    contest_score['violations_relative'],
                ))

            # Determining the winner among the N variants
            M = len(solver.models)

            if best_idx < M * N_VARIANTS:
                checkpoint_idx = best_idx // N_VARIANTS
                variant_idx = best_idx % N_VARIANTS
                if variant_idx == 0:
                    winner_name = f"Checkpoint {checkpoint_idx + 1} (Original)"
                else:
                    winner_name = f"Checkpoint {checkpoint_idx + 1} (Noisy variant {variant_idx})"
            else:
                winner_name = "Averaged Ensemble"

            # RECORD THE WIN IN THE STATISTICS
            winner_stats[winner_name] = winner_stats.get(winner_name, 0) + 1

            print(f"Test {processed_batches} finished | Blocks: {bc} Loss: {batch_loss / B:.6f} | Winner: {winner_name} | Time: {time.time() - test_time:.2f} sec\n")

            if contest_score['cost'] > 1.3:
                critical_tests.append([processed_batches, contest_score['cost']])

    avg_val_loss = total_val_loss / processed_batches if processed_batches > 0 else 0.0
    print(f"🏁 Validation completed | Average Physical Score: {avg_val_loss:.4f}")
    print(f"Found overlap-free solutions: {good_solutions} out of {processed_batches}")
    print("Bad tests [{}]: {}".format(len(critical_tests), critical_tests))

    # PRINT THE FINAL WINNER STATISTICS
    print("-" * 50)
    print("🏆 Winner Statistics:")
    # Sort the dictionary by the number of wins in descending order
    for name, count in sorted(winner_stats.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {name}: {count} times")

    print("=" * 50 + "\n")

    print("\n=== PHASE PROFILE (total over all tests) ===")
    total = sum(PHASE_TIMES.values())
    n = max(PHASE_COUNT['tests'], 1)
    for name in sorted(PHASE_TIMES):
        t = PHASE_TIMES[name]
        print(f"{name:16s} {t:8.2f} s | {100 * t / total:5.1f}% | {t / n * 1000:7.1f} ms/test")
    print(f"{'TOTAL':16s} {total:8.2f} s | {n} tests | {total / n * 1000:7.1f} ms/test")

    if EXIT_STEPS:
        import numpy as _np
        a = _np.array(EXIT_STEPS)
        print(f"Exit-step: min={a.min()} p50={int(_np.percentile(a,50))} "
                 f"p90={int(_np.percentile(a,90))} max={a.max()} "
                 f"share reaching cap={(a>=STEPS-1).mean()*100:.0f}%")

    results = {
        "submission": 'valid_with_refine.py',
        "timestamp": str(datetime.datetime.now()),
        "solutions": solutions
    }
    json.dump(results, open("last_run.json", "w", encoding='utf8'), indent=4)

    return avg_val_loss


@functools.lru_cache(maxsize=32)
def load_tensor_file_cached(file_path):
    return torch.load(file_path)


class FloorplanDatasetLiteTestOptimized(Dataset):
    def __init__(self, root):
        self.all_input_files = []
        self.all_label_files = []
        partition_range = range(21, 121)
        identifier_range = range(1, 11)

        for worker_idx in partition_range:
            config_dir = os.path.join(root, f'FloorSet_data/LiteTensorDataTest/config_{worker_idx}')
            for identifier in identifier_range:
                input_file_pattern = os.path.join(config_dir, f'litedata_{identifier}.pth')
                label_file_pattern = os.path.join(config_dir, f'litelabel_{identifier}.pth')

                if os.path.isfile(input_file_pattern) and os.path.isfile(label_file_pattern):
                    self.all_input_files.append(input_file_pattern)
                    self.all_label_files.append(label_file_pattern)

        self.layouts_per_file = 1
        if INVERSE_ORDER_OF_TESTS:
            self.all_input_files = self.all_input_files[::-1]
            self.all_label_files = self.all_label_files[::-1]

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def __len__(self):
        return len(self.all_input_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)

        input_contents = load_tensor_file_cached(self.all_input_files[file_idx])
        label_contents = load_tensor_file_cached(self.all_label_files[file_idx])

        area_target = input_contents[layout_idx][0][:, 0]

        placement_constraints = input_contents[layout_idx][0][:, 1:]
        b2b_connectivity = input_contents[layout_idx][1]
        p2b_connectivity = input_contents[layout_idx][2]
        pins_pos = input_contents[layout_idx][3]

        fp_sol = label_contents[layout_idx][1]
        metrics_sol = label_contents[layout_idx][0]
        block_count = int((area_target != -1).sum().item())

        positions = []
        for i in range(block_count):
            block = fp_sol[i]
            valid = block[block[:, 0] != -1]
            if len(valid) > 0:
                x_min, y_min = valid.min(dim=0).values
                x_max, y_max = valid.max(dim=0).values
                positions.append((float(x_min), float(y_min),
                                  float(x_max - x_min), float(y_max - y_min)))
            else:
                positions.append((0, 0, 1, 1))

        fp_sol = torch.from_numpy(np.array(positions, dtype=np.float32))

        return {
            'area_target': area_target,
            'b2b_conn': b2b_connectivity,
            'p2b_conn': p2b_connectivity,
            'pins_pos': pins_pos,
            'constraints': placement_constraints,
            'fp_sol': fp_sol,
            'metrics': metrics_sol
        }


def validate_model(checkpoint_paths, data_root):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    val_dataset = FloorplanDatasetLiteTestOptimized(data_root)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=transformer_direct_collate_fn,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    validate(checkpoint_paths, val_dataset, val_dataloader)
    score_saved_solutions_local(
        "last_run.json",
        data_path=data_root,
        output_path="last_run.txt",
        dataset=val_dataset
    )


if __name__ == "__main__":
    print("Data root: {}".format(DATA_ROOT))
    checkpoint_paths = [
        DATA_ROOT + 'weights/direct_transformer_loss_5.6719_epoch_364.pt',
        DATA_ROOT + 'weights/direct_transformer_loss_7.9808_epoch_413.pt',
        DATA_ROOT + 'weights/direct_transformer_loss_7.9332_epoch_414.pt',
        DATA_ROOT + 'weights/direct_transformer_loss_7.7646_epoch_418.pt',
        DATA_ROOT + 'weights/direct_transformer_loss_7.7672_epoch_420.pt',
    ]
    if not os.path.isfile(checkpoint_paths[0]):
        print('Can\'t find pretrained weights: {}. Download them first with '
              'preproc_data/r02_download_pretrained_weights.py script'.format(checkpoint_paths[0]))
        exit()

    start_time = time.time()
    validate_model(checkpoint_paths, DATA_ROOT)
    print("Full validation time: {:.2f} sec".format(time.time() - start_time))