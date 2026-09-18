import os

if __name__ == '__main__':
    gpu_use = "1"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import time
import glob
import math
import functools
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from model import FloorplanDirectTransformer, decode_from_direct_space
from config import *

torch.set_num_threads(TORCH_NUM_THREADS)


def compute_spectral_embeddings(b2b_conn, num_nodes, k_dim=3):
    """
    Calculates the spectral embeddings of the graph (Fiedler vectors).
    k_dim: number of vectors to extract.
    """
    # 1. Build adjacency matrix W
    W = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
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

    # For numerical stability of eigh on CPU, add a tiny jitter to the diagonal
    L += torch.eye(num_nodes) * 1e-6

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


def compute_rigid_loss(positions, target_fp, constraints):
    is_fixed = (constraints[:, 0] == 1)
    is_preplaced = (constraints[:, 1] == 1)

    loss = torch.tensor(0.0, device=positions.device)

    # For Fixed blocks, strictly penalize for changing w and h (indices 2 and 3)
    if is_fixed.any():
        loss += F.l1_loss(positions[is_fixed, 2:], target_fp[is_fixed, 2:]) * 10.0

    # For Preplaced blocks, strictly penalize for changing all parameters (x, y, w, h)
    if is_preplaced.any():
        loss += F.l1_loss(positions[is_preplaced], target_fp[is_preplaced]) * 10.0

    return loss


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

    num_groups = 0
    for gid in unique_ids:
        mask = cluster_ids == gid
        if mask.sum() < 2:
            continue

        gx, gy, gw, gh = x[mask], y[mask], w[mask], h[mask]

        bbox_w = (gx + gw).max() - gx.min()
        bbox_h = (gy + gh).max() - gy.min()
        bbox_area = bbox_w * bbox_h
        sum_area = (gw * gh).sum()

        penalty = torch.relu(bbox_area - sum_area) / (sum_area + 1e-6)
        loss = loss + torch.clamp(penalty, max=20.0)
        num_groups += 1

    if num_groups == 0:
        return loss

    return loss / num_groups


def compute_boundary_loss(positions, constraints):
    """Blocks must touch the specified side of the common bounding box"""
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]

    # Bit flags require an integer type; in constraints the code is stored as float
    codes = constraints[:, 4].long()

    x_min = x.min()
    y_min = y.min()
    x_max = (x + w).max()
    y_max = (y + h).max()

    global_w = torch.clamp(x_max - x_min, min=1e-6)
    global_h = torch.clamp(y_max - y_min, min=1e-6)

    # Bit flags: 1=Left, 2=Right, 4=Top, 8=Bottom
    has_left   = (codes & 1).ne(0).to(x.dtype)
    has_right  = (codes & 2).ne(0).to(x.dtype)
    has_top    = (codes & 4).ne(0).to(x.dtype)
    has_bottom = (codes & 8).ne(0).to(x.dtype)

    loss = (
        2.0 * has_left   * torch.abs(x - x_min) / global_w
        +     has_right  * torch.abs(x + w - x_max) / global_w
        +     has_top    * torch.abs(y + h - y_max) / global_h
        + 2.0 * has_bottom * torch.abs(y - y_min) / global_h
    ).sum()

    denom = torch.clamp((codes != 0).sum().to(x.dtype), min=1.0)

    return loss / denom


def compute_pairwise_distance_loss_weighted(
        pred_abs,
        target_abs,
        b2b_connectivity,
):
    pred_cx = pred_abs[:, 0] + pred_abs[:, 2] / 2.0
    pred_cy = pred_abs[:, 1] + pred_abs[:, 3] / 2.0

    target_cx = target_abs[:, 0] + target_abs[:, 2] / 2.0
    target_cy = target_abs[:, 1] + target_abs[:, 3] / 2.0

    pred_centers = torch.stack([pred_cx, pred_cy], dim=-1)
    target_centers = torch.stack([target_cx, target_cy], dim=-1)

    # pred_dist = torch.cdist(pred_centers, pred_centers)
    diff_xy = pred_centers.unsqueeze(1) - pred_centers.unsqueeze(0)
    pred_dist = torch.sqrt((diff_xy ** 2).sum(-1) + 1e-6)
    target_dist = torch.cdist(target_centers, target_centers)

    N = pred_dist.shape[0]

    weight_matrix = torch.ones(
        (N, N),
        device=pred_abs.device
    ) * 0.1

    if b2b_connectivity.numel() > 0:
        valid = b2b_connectivity[b2b_connectivity[:, 0] >= 0]
        if valid.numel() > 0:
            i = valid[:, 0].long()
            j = valid[:, 1].long()
            w = valid[:, 2]

            edge_mask = (i < N) & (j < N)
            i, j, w = i[edge_mask], j[edge_mask], w[edge_mask]

            weight_matrix[i, j] += w * 10.0
            weight_matrix[j, i] += w * 10.0

    mask = torch.triu(
        torch.ones(N, N, device=pred_abs.device),
        diagonal=1
    ).bool()

    diff = F.smooth_l1_loss(
        pred_dist,
        target_dist,
        reduction='none'
    )

    loss = (
        diff[mask] *
        weight_matrix[mask]
    ).mean()

    return loss


def compute_training_loss_differentiable_matrix(
        positions: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        area_targets: torch.Tensor,
        baseline_metrics: torch.Tensor,
        target_fp: torch.Tensor,
        constraints=None,
) -> torch.Tensor:
    N = positions.shape[0]

    # Unpacking positions: [x, y, w, h]
    x = positions[:, 0]
    y = positions[:, 1]
    w = positions[:, 2]
    h = positions[:, 3]

    # Calculating centers
    cx = x + w / 2.0
    cy = y + h / 2.0

    # =========================================================================
    # 1. HPWL B2B - VECTORIZED
    # =========================================================================
    valid_b2b = b2b_connectivity[b2b_connectivity[:, 0] >= 0]
    if valid_b2b.numel() > 0:
        i = valid_b2b[:, 0].long()
        j = valid_b2b[:, 1].long()
        weights = valid_b2b[:, 2]

        # Filter in case of garbage indices
        mask = (i < N) & (j < N)
        i, j, weights = i[mask], j[mask], weights[mask]

        # dx = torch.abs(cx[i] - cx[j])
        # dy = torch.abs(cy[i] - cy[j])
        dx = torch.sqrt((cx[i] - cx[j])**2 + 1e-5)
        dy = torch.sqrt((cy[i] - cy[j]) ** 2 + 1e-5)
        hpwl_b2b = (weights * (dx + dy)).sum()
    else:
        hpwl_b2b = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

    # =========================================================================
    # 1.1 HPWL P2B - VECTORIZED
    # =========================================================================
    valid_p2b = p2b_connectivity[p2b_connectivity[:, 0] >= 0]
    if valid_p2b.numel() > 0:
        pin_idx = valid_p2b[:, 0].long()
        block_idx = valid_p2b[:, 1].long()
        weights = valid_p2b[:, 2]

        mask = (pin_idx < pins_pos.shape[0]) & (block_idx < N)
        pin_idx, block_idx, weights = pin_idx[mask], block_idx[mask], weights[mask]

        pin_x = pins_pos[pin_idx, 0]
        pin_y = pins_pos[pin_idx, 1]

        dx = torch.abs(cx[block_idx] - pin_x)
        dy = torch.abs(cy[block_idx] - pin_y)
        hpwl_p2b = (weights * (dx + dy)).sum()
    else:
        hpwl_p2b = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

    hpwl_total = hpwl_b2b + hpwl_p2b

    # =========================================================================
    # 2. Bounding Box Area (Soft Approximation via Log-Sum-Exp)
    # =========================================================================
    # Scale parameter (temperature).
    # Since you decode coordinates into absolute values (which can be large, e.g. 100-1000),
    # gamma should be kept small to avoid overflow and overly "sharp" gradients.
    # For normalized coordinates [-1, 1], gamma = 10.0 is usually chosen
    # For absolute coordinates, it is better to start with gamma = 0.1 or compute it dynamically.
    gamma = 0.1

    x_right = x + w
    y_top = y + h

    # Soft Maximum (for right and top edge)
    x_max_soft = torch.logsumexp(gamma * x_right, dim=0) / gamma
    y_max_soft = torch.logsumexp(gamma * y_top, dim=0) / gamma

    # Soft Minimum (for left and bottom edge)
    # Mathematical trick: min(x) is equivalent to -max(-x)
    x_min_soft = -torch.logsumexp(-gamma * x, dim=0) / gamma
    y_min_soft = -torch.logsumexp(-gamma * y, dim=0) / gamma

    # Differentiable area where ALL blocks contribute
    bbox_area = (x_max_soft - x_min_soft) * (y_max_soft - y_min_soft)

    # =========================================================================
    # 3. Overlap Violation (Distance-based repulsion)
    # =========================================================================
    # 1. Pairwise absolute distances between centers of all blocks
    dx_matrix = torch.abs(cx.unsqueeze(1) - cx.unsqueeze(0))
    dy_matrix = torch.abs(cy.unsqueeze(1) - cy.unsqueeze(0))

    # 2. Minimum required distance between centers (when blocks exactly touch edges)
    min_dist_x = (w.unsqueeze(1) + w.unsqueeze(0)) / 2.0
    min_dist_y = (h.unsqueeze(1) + h.unsqueeze(0)) / 2.0

    # 3. Penetration depth
    # If > 0, blocks physically overlap.
    # If < 0, there is a gap between them.
    penetration_x = min_dist_x - dx_matrix
    penetration_y = min_dist_y - dy_matrix

    # 4. Apply Softplus instead of ReLU.
    # The beta parameter is responsible for the stiffness of the "force field".
    # At beta=1.0, if the gap is 2 units (penetration = -2),
    # softplus will output a penalty ~0.12. This will provide a soft repulsive gradient.
    soft_overlap_x = F.softplus(penetration_x, beta=5.0)
    soft_overlap_y = F.softplus(penetration_y, beta=5.0)

    # 5. Final "smoothed" intersection matrix
    overlap_matrix = soft_overlap_x * soft_overlap_y

    # Take only upper triangle of matrix (without diagonal)
    overlap_area = torch.triu(overlap_matrix, diagonal=1).sum()

    total_block_area = (w * h).sum()

    # Final penalty
    overlap_violation = overlap_area / (total_block_area + 1e-6)

    if 0:
        # =========================================================================
        # 4. Area Tolerance Violation - VECTORIZED
        # =========================================================================
        actual_areas = w * h
        valid_mask = area_targets > 0
        area_errors = torch.zeros_like(area_targets)

        # Compute only for valid blocks
        area_errors[valid_mask] = torch.abs(actual_areas[valid_mask] - area_targets[valid_mask]) / area_targets[valid_mask]

        area_excess = torch.relu(area_errors - AREA_TOLERANCE)
        area_violation = area_excess.sum() / (valid_mask.sum() + 1e-6)

    # =========================================================================
    # 5. Compute Gaps vs Baseline
    # =========================================================================
    baseline_area = baseline_metrics[0]
    baseline_hpwl = baseline_metrics[6] + baseline_metrics[7]

    hpwl_gap = torch.relu((hpwl_total - baseline_hpwl) / (baseline_hpwl + 1e-6))
    area_gap = torch.relu((bbox_area - baseline_area) / (baseline_area + 1e-6))

    # =========================================================================
    # 4.5. NEW: Out-of-Bounds (OOB) Violation
    # =========================================================================
    # Canvas boundaries are taken from min and max pin coordinates for each axis
    if pins_pos.numel() > 0:
        min_x_canvas = pins_pos[:, 0].min()
        max_x_canvas = pins_pos[:, 0].max()
        min_y_canvas = pins_pos[:, 1].min()
        max_y_canvas = pins_pos[:, 1].max()
    else:
        # Fallback, if there are no pins
        min_x_canvas = torch.tensor(0.0, device=positions.device)
        max_x_canvas = torch.tensor(350.0, device=positions.device)
        min_y_canvas = torch.tensor(0.0, device=positions.device)
        max_y_canvas = torch.tensor(350.0, device=positions.device)

    # Area that "fell out" beyond the pins rectangle (left, bottom, right, top)
    oob_left_area = torch.relu(min_x_canvas - x) * h
    oob_bottom_area = torch.relu(min_y_canvas - y) * w
    oob_right_area = torch.relu((x + w) - max_x_canvas) * h
    oob_top_area = torch.relu((y + h) - max_y_canvas) * w

    total_oob_area = (oob_left_area + oob_bottom_area + oob_right_area + oob_top_area).sum()

    # Penalty in the same proportions as overlap (from 0.0 to 1.0+)
    # Add 5.0 multiplier so the network "fears" going out of bounds more than overlapping blocks
    oob_violation = 1.0 * (total_oob_area / (total_block_area + 1e-6))

    # =========================================================================
    # NEW: Soft constraints
    # =========================================================================
    # V_mib = torch.tensor(0.0, device=positions.device)
    V_cluster = torch.tensor(0.0, device=positions.device)
    V_boundary = torch.tensor(0.0, device=positions.device)

    if constraints is not None:
        # V_mib = compute_mib_loss(positions, constraints)
        V_cluster = compute_cluster_loss(positions, constraints)
        V_boundary = compute_boundary_loss(positions, constraints)

    # V_rigid = torch.tensor(0.0, device=positions.device)
    # if constraints is not None and target_fp is not None:
    #     V_rigid = compute_rigid_loss(positions, target_fp, constraints)

    # Replacing the old V_soft with the full one
    # V_soft = 10.0 * overlap_violation + 1.0 * oob_violation + V_mib + V_cluster + V_boundary + 0.01 * V_rigid
    V_soft = 10.0 * overlap_violation + 1.0 * oob_violation + V_cluster + V_boundary
    if not torch.isfinite(V_soft).all():
        print(positions.shape)
        print(positions.detach().cpu().numpy())
        print(overlap_violation.detach().cpu().numpy(), oob_violation.detach().cpu().numpy(), V_cluster.detach().cpu().numpy(), V_boundary.detach().cpu().numpy())
    # print(overlap_violation, area_violation, 0.0001 * V_mib, 0.0001 * V_cluster, 0.0001 * V_boundary)
    # V_soft = torch.clamp(V_soft, max=10.0)

    # =========================================================================
    # 7. Contest Cost Formula
    # =========================================================================
    quality_factor = 1.0 + ALPHA * (hpwl_gap + area_gap)
    # violation_factor = torch.exp(BETA * V_soft)
    violation_factor = 1.0 + BETA * V_soft

    cost = quality_factor * violation_factor

    return cost



# --- 2. OPTIMIZED DATA LOADING ---

@functools.lru_cache(maxsize=1000)
def load_tensor_file_cached(file_path):
    return torch.load(file_path)


class FloorplanDatasetLiteOptimized(Dataset):
    def __init__(self, root_dirs):
        # Support both a single string (old format) and a list of paths
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.all_files = []
        for root in root_dirs:
            for worker_idx in range(100):
                # Search for all .th files. The 'layouts*.th' pattern captures both
                # the original 'layouts_0.th' and the new 'layouts_0_aug_xy.th'
                search_pattern = os.path.join(root, f"worker_{worker_idx}", "layouts*.th")
                self.all_files.extend(glob.glob(search_pattern))

        self.layouts_per_file = 112
        print(f"[Dataset] Paths connected: {len(root_dirs)}")
        print(f"[Dataset] Files found: {len(self.all_files)}. Total graphs: {self.__len__()}")

    def __len__(self):
        return len(self.all_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        file_contents = load_tensor_file_cached(self.all_files[file_idx])

        # FIXED: Index 0 for area_target and placement_constraints
        area_target = file_contents[0][layout_idx][:, 0]
        placement_constraints = file_contents[0][layout_idx][:, 1:]

        b2b_connectivity = file_contents[1][layout_idx]
        p2b_connectivity = file_contents[2][layout_idx]
        pins_pos = file_contents[3][layout_idx]
        tree_sol = file_contents[4][layout_idx]

        # For some reason here is different order of dimensions. We need to move
        fp_sol = file_contents[5][layout_idx] # w h x y
        fp_sol1 = torch.clone(fp_sol)
        fp_sol1[:, 0] = fp_sol[:, 2]
        fp_sol1[:, 1] = fp_sol[:, 3]
        fp_sol1[:, 2] = fp_sol[:, 0]
        fp_sol1[:, 3] = fp_sol[:, 1]
        fp_sol = fp_sol1

        metrics_sol = file_contents[6][layout_idx]

        # READING THE PRECOMPUTED DATA
        spectral_emb = file_contents[7][layout_idx]
        spd_matrix = file_contents[8][layout_idx]

        return {
            'area_target': area_target,
            'b2b_conn': b2b_connectivity,
            'p2b_conn': p2b_connectivity,
            'pins_pos': pins_pos,
            'constraints': placement_constraints,
            'tree_sol': tree_sol,
            'fp_sol': fp_sol,
            'metrics': metrics_sol,
            'spectral_emb': spectral_emb,  # <- New
            'spd_matrix': spd_matrix  # <- New
        }


def transformer_direct_collate_fn(batch):
    B = len(batch)
    max_len = 120

    # Safe lookup of the maximum number of pins (returns 0 if there are none)
    max_pins = max((item['pins_pos'].shape[0] for item in batch), default=0)

    # =========================================================================
    # 1. MEMORY PREALLOCATION
    # =========================================================================
    batched_features = torch.zeros((B, max_len, 14), dtype=torch.float32)
    batched_target_fp = torch.zeros((B, max_len, 4), dtype=torch.float32)
    batched_b2b_matrix = torch.zeros((B, max_len, max_len), dtype=torch.float32)
    batched_p2b_matrix = torch.zeros((B, max_len, max_pins), dtype=torch.float32)
    batched_pins_pos = torch.zeros((B, max_pins, 2), dtype=torch.float32)

    # SPD padding defaults to 20
    batched_spd_matrix = torch.full((B, max_len, max_len), 20, dtype=torch.long)

    # Padding mask: True by default (everything is padding)
    padding_mask = torch.ones((B, max_len), dtype=torch.bool)
    batched_scales = torch.zeros(B, dtype=torch.float32)

    raw_items = []

    # =========================================================================
    # 2. FILLING WITH DATA (with on-the-fly computations)
    # =========================================================================
    for i, item in enumerate(batch):
        area = item['area_target']
        block_count = int((area != -1).sum().item())

        # Mark real blocks as False in the mask
        padding_mask[i, :block_count] = False

        curr_fp_sol = item['fp_sol'][:block_count]
        curr_constraints = item['constraints'][:block_count]
        curr_area = area[:block_count]
        b2b_conn = item['b2b_conn']
        p2b_conn = item['p2b_conn']
        pins_pos = item['pins_pos']
        num_pins = pins_pos.shape[0]

        graph_scale = pins_pos.max() if pins_pos.numel() > 0 else torch.tensor(100.0)
        batched_scales[i] = graph_scale

        # Copy the coordinates
        batched_target_fp[i, :block_count] = curr_fp_sol

        # --- HEAVY ON-THE-FLY COMPUTATIONS ---
        # spectral_emb = compute_spectral_embeddings(b2b_conn, block_count, k_dim=3)
        # curr_spd_matrix = compute_spd_matrix_hops(b2b_conn, block_count, max_dist=20)

        # REPLACED WITH:
        spectral_emb = item['spectral_emb']
        curr_spd_matrix = item['spd_matrix']

        batched_spd_matrix[i, :block_count, :block_count] = curr_spd_matrix

        # --- B2B matrix ---
        curr_b2b_matrix = torch.zeros((block_count, block_count), dtype=torch.float32)
        if b2b_conn.numel() > 0:
            u, v, w = b2b_conn[:, 0].long(), b2b_conn[:, 1].long(), b2b_conn[:, 2]
            curr_b2b_matrix[u, v] = w
            curr_b2b_matrix[v, u] = w

        batched_b2b_matrix[i, :block_count, :block_count] = curr_b2b_matrix

        # --- P2B matrix and pins ---
        curr_p2b_matrix = torch.zeros((block_count, num_pins), dtype=torch.float32)
        if p2b_conn.numel() > 0:
            pin_ids = p2b_conn[:, 0].long()
            block_ids = p2b_conn[:, 1].long()
            weights = p2b_conn[:, 2].float()
            curr_p2b_matrix[block_ids, pin_ids] = weights

        batched_p2b_matrix[i, :block_count, :num_pins] = torch.log1p(curr_p2b_matrix)

        if num_pins > 0:
            batched_pins_pos[i, :num_pins] = pins_pos.float()

        # --- Center of Mass (Pin CoM) ---
        if pins_pos.numel() > 0:
            pin_weight_sum = curr_p2b_matrix.sum(dim=1, keepdim=True)
            pin_com = torch.matmul(curr_p2b_matrix, pins_pos.float())
            has_pins_mask = (pin_weight_sum > 0).squeeze(1)
            pin_com[has_pins_mask] /= pin_weight_sum[has_pins_mask]
            pin_com_norm = pin_com / graph_scale
            pin_com_norm[~has_pins_mask] = 0.5
        else:
            pin_com_norm = torch.full((block_count, 2), 0.5, dtype=torch.float32)

        # --- Geometric targets ---
        curr_x = curr_fp_sol[:, 0]
        curr_y = curr_fp_sol[:, 1]
        curr_w = curr_fp_sol[:, 2]
        curr_h = curr_fp_sol[:, 3]

        target_r = torch.log(curr_w / (curr_h + 1e-6))
        target_r_norm = torch.clamp(target_r / R_SCALE, -1.0, 1.0)
        target_x_norm = curr_x / graph_scale
        target_y_norm = curr_y / graph_scale

        is_fixed_flag = curr_constraints[:, 0] == 1.0
        is_preplaced_flag = curr_constraints[:, 1] == 1.0
        is_hard_flag = is_fixed_flag | is_preplaced_flag

        known_r = torch.zeros_like(target_r_norm)
        known_r[is_hard_flag] = target_r_norm[is_hard_flag]

        known_x = torch.zeros_like(target_x_norm)
        known_y = torch.zeros_like(target_y_norm)
        known_x[is_preplaced_flag] = target_x_norm[is_preplaced_flag]
        known_y[is_preplaced_flag] = target_y_norm[is_preplaced_flag]

        # --- Feature assembly ---
        area_norm = (torch.sqrt(curr_area) / graph_scale).unsqueeze(1)
        curr_features = torch.cat([
            area_norm,
            curr_constraints,
            spectral_emb,  # Freshly computed spectral features
            pin_com_norm,
            known_r.unsqueeze(1),
            known_x.unsqueeze(1),
            known_y.unsqueeze(1)
        ], dim=1)

        # Put the features into the proper batch slot
        batched_features[i, :block_count] = curr_features

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
        'features': batched_features,
        'target_fp': batched_target_fp,
        'b2b_matrix': batched_b2b_matrix,
        'p2b_matrix': batched_p2b_matrix,
        'pins_pos': batched_pins_pos,
        'spd_matrix': batched_spd_matrix,
        'padding_mask': padding_mask,
        'scales': batched_scales,
        'raw_items': raw_items
    }


def train_direct_batched():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-5

    model = FloorplanDirectTransformer(hidden_dim=256, num_layers=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    data_root = DATA_ROOT
    dataset_orig = data_root + "FloorSet_data/floorset_lite_precomputed/"
    dataset_augm = data_root + "FloorSet_data/floorset_lite_augm_precomputed/"
    if os.path.isdir(dataset_augm):
        print("Use augmented data for training")
        dataset = FloorplanDatasetLiteOptimized([dataset_orig, dataset_augm])
    else:
        print("Augmented data wasn't found. Skip it for training!")
        dataset = FloorplanDatasetLiteOptimized([dataset_orig])

    lambda_phys = LAMBDA_PHYS_LOSS

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=transformer_direct_collate_fn,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # ---------------------------------------------------------
    # AUTOMATIC CHECKPOINT LOADING BLOCK
    # ---------------------------------------------------------
    start_epoch = 0
    checkpoint_dir = os.path.join(data_root, "checkpoints")

    if os.path.exists(checkpoint_dir):
        # Find all epoch checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "direct_transformer_*.pt"))

        if checkpoint_files:
            # Sort files by epoch number in filename
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]

            print(f"\n[Loading] Found latest checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

            # 1. Extract weights dictionary
            state_dict = checkpoint['model_state_dict']

            # 3. Load adapted weights into the model
            model.load_state_dict(state_dict, strict=False)

            if 0:
                # Freezing the old weights
                for name, param in model.named_parameters():
                    if name in state_dict:
                        # The parameter came from the old checkpoint - freeze it
                        param.requires_grad = False
                    else:
                        # This is a new parameter (new transformer layers) - keep it trainable
                        param.requires_grad = True
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = True

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            print(f"[Success] Weights restored. Resuming training from epoch {start_epoch}\n")
        else:
            print("\n[Info] Checkpoint folder is empty. Starting training from scratch.\n")
    else:
        print("\n[Info] Checkpoint folder not found. Starting training from scratch.\n")

    model.train()
    for epoch in range(start_epoch, 10000):

        # First 2 epochs, train only via MSE to copy dataset.
        # Then gradually enable physical loss.
        current_lambda_phys = 0.0 if epoch < 2 else lambda_phys * min(0.5, (epoch - 2) / 10.0)
        losses_arr = np.zeros(len(dataloader))
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        print('Start epoch: {} Lambda: {} Learning rate: {}'.format(epoch, current_lambda_phys, learning_rate))

        for step, batch_data in bar:
            features = batch_data['features'].to(device)
            target_fp = batch_data['target_fp'].to(device)
            b2b_matrix = batch_data['b2b_matrix'].to(device)
            p2b_matrix = batch_data['p2b_matrix'].to(device)
            pins_pos = batch_data['pins_pos'].to(device)
            padding_mask = batch_data['padding_mask'].to(device)
            scales = batch_data['scales'].to(device)
            spd_matrix = batch_data['spd_matrix'].to(device)
            raw_items = batch_data['raw_items']

            B = features.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                # 1. Forward pass: model outputs [x_norm, y_norm, r_norm]
                pred_norm = model(
                    features,
                    padding_mask=padding_mask,
                    b2b_matrix=b2b_matrix,
                    p2b_matrix=p2b_matrix,
                    pins_pos=pins_pos,
                    spd_matrix=spd_matrix,
                )

                loss_mse = torch.tensor(0.0, device=device)
                loss_phys = torch.tensor(0.0, device=device)

                phys_sub_batch_size = min(10, B)

                if USE_BATCHED_MSE:
                    # 1. Reshape the graph scales for broadcasting
                    # scales originally has shape (B,), we make it (B, 1)
                    scales_batched = scales.unsqueeze(1)

                    # 2. Recover area_targets directly from the input features.
                    # In the collate_fn, feature index 0 is area_norm = sqrt(area) / scale.
                    # Recover the original area: area = (area_norm * scale)^2
                    area_targets_batched = (features[..., 0] * scales_batched) ** 2

                    # 3. Decode the whole batch at once (B, max_len, 3) -> (B, max_len, 4)
                    pred_abs_batched = decode_from_direct_space(pred_norm, area_targets_batched, scales_batched)

                    # 4. Compute the raw per-component MSE (without reduction)
                    mse_raw = F.mse_loss(pred_abs_batched, target_fp, reduction='none')  # Shape: (B, max_len, 4)

                    # 5. Apply the mask to zero out the errors on padding elements
                    # padding_mask == True for padding, so we take the logical negation ~
                    valid_mask = (~padding_mask).unsqueeze(-1)  # Shape: (B, max_len, 1)
                    mse_masked = mse_raw * valid_mask

                    # 6. Count the valid coordinates for each graph (number of blocks * 4 coordinates)
                    valid_elements_count = (~padding_mask).sum(dim=1) * 4.0  # Shape: (B,)
                    valid_elements_count = valid_elements_count.clamp(min=1.0)

                    # 7. Sum the error inside each graph and divide by its number of valid elements.
                    # This exactly reproduces the behaviour of F.mse_loss(...) on the [i, :bc] slices
                    mse_per_item = mse_masked.sum(dim=(1, 2)) / valid_elements_count

                    # Final loss (replaces the original loss_mse = loss_mse / B)
                    loss_mse = mse_per_item.mean()

                if USE_PER_ITEM_LOSSES:
                    for i in range(B):
                        raw = raw_items[i]
                        bc = raw['block_count']
                        scale_i = raw['graph_scale']
                        area_i = raw['area_target'].to(device)
                        b2b_c = raw['b2b_conn'].to(device, non_blocking=True)

                        # Decode from [x, y, r] to absolute [x, y, w, h]
                        curr_pred_norm = pred_norm[i, :bc]
                        curr_pred_abs = decode_from_direct_space(curr_pred_norm, area_i, scale_i)

                        curr_target_fp = target_fp[i, :bc]
                        if not USE_BATCHED_MSE:
                            loss_mse += F.mse_loss(curr_pred_abs, curr_target_fp)

                        if USE_PAIRWISE_LOSS:
                            loss_pairwise = compute_pairwise_distance_loss_weighted(
                                curr_pred_abs,
                                curr_target_fp,
                                b2b_connectivity=b2b_c,
                            )
                            # print("!!!", loss_mse.item(), 10 * loss_pairwise.item())
                            loss_mse += 10 * loss_pairwise

                        if USE_PHYS_LOSS:
                            if current_lambda_phys > 0.0 and i < phys_sub_batch_size:
                                b2b_c = raw['b2b_conn'].to(device, non_blocking=True)
                                p2b_c = raw['p2b_conn'].to(device, non_blocking=True)
                                pins = raw['pins_pos'].to(device, non_blocking=True)
                                mets = raw['metrics'].to(device, non_blocking=True)
                                curr_constraints = features[i, :bc, 1:]
                                curr_target_fp = target_fp[i, :bc]  # Extract target for the current graph

                                l_p = compute_training_loss_differentiable_matrix(
                                    curr_pred_abs, b2b_c, p2b_c, pins, area_i, mets, curr_target_fp, constraints=curr_constraints
                                )
                                loss_phys += l_p

                    loss_mse = loss_mse / B
                if current_lambda_phys > 0.0:
                    loss_phys = loss_phys / phys_sub_batch_size

                loss = loss_mse + current_lambda_phys * loss_phys

            # BACKWARD PASS
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses_arr[step] = loss.item()
            bar.set_postfix({
                'loss mse': "{:.2f}".format(loss_mse.item()),
                'loss phys': "{:.2f}".format(loss_phys.item()),
                'loss avg': "{:.4f}".format(losses_arr[:step + 1].mean()),
            })
            # print(f"Epoch {epoch} | Batch {step} | Loss MSE: {loss_mse.item():.2f} | Loss Phys: {loss_phys.item():.4f} | Loss Avg: {losses_arr[:step+1].mean():.4f}")

        loss_avg = losses_arr.mean()
        print('Loss for epoch {}: {:.4f}'.format(epoch, loss_avg))

        # ---------------------------------------------------------
        # START CHECKPOINT SAVING BLOCK (AT END OF EPOCH)
        # ---------------------------------------------------------
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"direct_transformer_loss_{loss_avg:.4f}_epoch_{epoch}.pt")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss_mse': loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
            'loss_phys': loss_phys.item() if isinstance(loss_phys, torch.Tensor) else loss_phys
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"\n=== Epoch {epoch} completed. Checkpoint successfully saved to: {checkpoint_path} ===\n")


if __name__ == "__main__":
    train_direct_batched()