import os
import math
import numpy as np
from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

import warnings

warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated"
)

ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3
M_PENALTY = 10.0
AREA_TOLERANCE = 0.01
EARLY_EXIT_ON_ALL_FEASIBLE = True  # Set True to exit when all N variants are valid
FREEZE_ON_FEASIBLE = True  # If True, the variant is fixed upon finding the first valid solution
MAX_POSTPROCESS_VARIANTS = 1

TORCH_THREADS = 1
FORCE_CPU = True

# CPU<->GPU synchronization interval inside the refine loop (early exit).
# The best solution is now tracked on the GPU EVERY step without synchronization,
# so a large interval does not degrade quality — it only speeds up the process.
CHECK_LOSS_STEPS = 20
FEASIBLE_PATIENCE_CHECKS = 800
N_VARIANTS = 1 # Number of starting points per checkpoint
MULTISTART_NOISE = 0.0003
STEPS = 400
MAX_LR = 0.0025
CYCLIC_LR_MULTIPLIER = 0.8
CYCLIC_LR_PERIODS = 5

R_SCALE = 1.2
torch.set_num_threads(TORCH_THREADS)

def compute_spectral_embeddings(b2b_conn, num_nodes, k_dim=3):
    """
    Calculates the spectral embeddings of the graph (Fiedler vectors).
    k_dim: number of vectors to extract.
    """
    # Determine device from input tensor
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
    # (device added in torch.eye)
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


def decode_from_direct_space(x_y_r_norm, area_targets, graph_scale):
    # x_y_r_norm[..., :2] is already in (0, 1) thanks to sigmoid
    x = x_y_r_norm[..., 0] * graph_scale
    y = x_y_r_norm[..., 1] * graph_scale

    # r_norm is in (-1, 1) thanks to tanh
    r = x_y_r_norm[..., 2] * R_SCALE

    w = torch.sqrt(area_targets) * torch.exp(r / 2.0)
    h = torch.sqrt(area_targets) * torch.exp(-r / 2.0)
    return torch.stack([x, y, w, h], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, src_key_padding_mask, attn_bias):
        # Classic Pre-Norm architecture
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FloorplanDirectTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.num_heads = num_heads

        # Input size 14:
        # 1 (Area) + 5 (Constraints) + 3 (Spectral) + 2 (Pin CoM) + 1 (Known R) + 2 (Known X, Y)
        self.feature_proj = nn.Linear(14, hidden_dim)

        # 21 classes: distances from 0 to 20
        self.spatial_embedding = nn.Embedding(21, num_heads)

        self.pin_spatial_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, num_heads)
        )
        self.edge_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        # Predict 3 parameters: [x_norm, y_norm, r_norm]
        self.head = nn.Linear(hidden_dim, 3)

        # Initialize with zeros, so model outputs center initially (if sigmoid is used) or zeros
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, node_features, padding_mask=None, b2b_matrix=None, p2b_matrix=None, pins_pos=None, spd_matrix=None):
        x = self.feature_proj(node_features)

        if p2b_matrix is not None and pins_pos is not None:
            pin_embeddings = self.pin_spatial_proj(pins_pos)
            pin_anchors = torch.bmm(p2b_matrix, pin_embeddings)
            x = x + pin_anchors

        attn_bias = None
        if spd_matrix is not None:
            B, K, _ = spd_matrix.shape

            # 1. Get topological Spatial Bias from shortest paths matrix
            # shape: (B, K, K, num_heads) -> (B, num_heads, K, K)
            spatial_bias = self.spatial_embedding(spd_matrix)
            spatial_bias = spatial_bias.permute(0, 3, 1, 2)

            # 2. Add info about direct connections strength (from b2b_matrix)
            if b2b_matrix is not None:
                eye = torch.eye(K, device=b2b_matrix.device).unsqueeze(0)
                b2b_matrix_adj = b2b_matrix + eye
                b2b_matrix_adj = torch.log1p(b2b_matrix_adj)
                b2b_features = b2b_matrix_adj.unsqueeze(-1)

                edge_bias = self.edge_proj(b2b_features)
                edge_bias = edge_bias.permute(0, 3, 1, 2)
                edge_bias = edge_bias * self.edge_scale

                # Sum topology (SPD) and connection strength
                spatial_bias = spatial_bias + edge_bias

            # Reshape to meet MultiheadAttention requirements: (B * num_heads, K, K)
            attn_bias = spatial_bias.reshape(B * self.num_heads, K, K)

        for block in self.blocks:
            x = block(x, padding_mask, attn_bias)

        x = self.final_norm(x)
        out = self.head(x)

        # Separate coordinates and r
        xy_norm = torch.sigmoid(out[..., :2])  # x, y in range (0, 1)
        r_norm = torch.tanh(out[..., 2:3])  # r_norm in range (-1, 1)

        # =====================================================================
        # ARCHITECTURAL FIXATION OF MIB GROUPS (Training)
        # =====================================================================
        is_fixed = node_features[..., 1] == 1.0
        is_preplaced = node_features[..., 2] == 1.0
        is_hard = is_fixed | is_preplaced
        mib_ids = node_features[..., 3]  # Index 3 in node_features is the MIB group ID

        B, N, _ = r_norm.shape
        updated_r_norm = r_norm.clone()

        for b in range(B):
            curr_mib = mib_ids[b]
            curr_hard = is_hard[b]
            unique_ids = curr_mib[curr_mib > 0].unique()

            for gid in unique_ids:
                mask = (curr_mib == gid)
                if mask.sum() > 1:
                    hard_in_group = mask & curr_hard
                    if hard_in_group.any():
                        # SCENARIO A: There is a hard block. Take r_norm from the first hard element.
                        ref_idx = torch.where(hard_in_group)[0][0]
                        updated_r_norm[b, mask] = r_norm[b, ref_idx]
                    else:
                        # SCENARIO B: No hard blocks. Average the entire group.
                        updated_r_norm[b, mask] = r_norm[b, mask].mean()

        r_norm = updated_r_norm
        # =====================================================================

        return torch.cat([xy_norm, r_norm], dim=-1)


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
    # hard violation counter from already computed matrices (eps=1e-7 as in
    # check_overlap_count_tensor); saves second set of NxN matrices
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
    Changes proportions (Aspect Ratio) of non-fixed cluster blocks,
    so they can reach each other.
    Uses connected components analysis (islands), asymmetric scaling
    (anchors) and analytical calculation of required sizes.
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

    # Only normal (soft) blocks can change shape
    can_reshape = ~(is_fixed | is_preplaced | (mib_ids > 0))
    unique_clusters = set(c for c in cluster_ids if c > 0)

    # Dynamically calculate chip boundaries
    right_edges = [pos[i][0] + pos[i][2] for i in range(n) if (boundary_codes[i] & 2)]
    top_edges = [pos[i][1] + pos[i][3] for i in range(n) if (boundary_codes[i] & 4)]
    c_width = max(right_edges) if right_edges else max(p[0] + p[2] for p in pos)
    c_height = max(top_edges) if top_edges else max(p[1] + p[3] for p in pos)

    for cid in unique_clusters:
        c_indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        if len(c_indices) < 2: continue

        # --- SEARCH FOR "ISLANDS" (Connected components within the cluster) ---
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
                    # Check distance between blocks (with 1e-4 tolerance)
                    dx = max(0.0, max(cx - (ox + ow), ox - (cx + cw)))
                    dy = max(0.0, max(cy - (oy + oh), oy - (cy + ch)))
                    if dx <= 1e-4 and dy <= 1e-4:
                        comp.add(other)
                        queue.append(other)
                        to_remove.append(other)
                for r in to_remove:
                    unvisited.remove(r)
            components.append(comp)

        # If the entire cluster is already a single entity - no need to change shape
        if len(components) == 1:
            continue

        # --- PROCESSING BROKEN CLUSTER ---
        for idx in c_indices:
            if not can_reshape[idx]: continue

            # Find out which island the current block belongs to
            my_comp = next(comp for comp in components if idx in comp)
            # List of targets - all blocks from OTHER islands
            target_indices = [i for i in c_indices if i not in my_comp]

            ix, iy, iw, ih = pos[idx]
            area = iw * ih
            best_shape = None
            candidates_wh = []

            # --- GENERATION OF SHAPE CANDIDATES ---

            # Strategy A: Analytical calculation of exact distance to target islands
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

            # Strategy B: Extended logarithmic grid (Aspect Ratio from 1:100 to 100:1)
            ratios = np.logspace(np.log10(0.01), np.log10(100.0), 16)
            for r in ratios:
                candidates_wh.append((math.sqrt(area * r), math.sqrt(area / r)))

            # --- CHECK CANDIDATES ---
            b_code = boundary_codes[idx]

            for w_c, h_c in candidates_wh:
                # ASYMMETRIC SCALING (Iterating Anchors)

                # Possible X anchors (0=Left, 1=Center, 2=Right)
                if b_code & 1:
                    x_candidates = [0.0]
                elif b_code & 2:
                    x_candidates = [c_width - w_c]
                else:
                    x_candidates = [ix, ix + (iw - w_c) / 2.0, ix + iw - w_c]

                # Possible Y anchors (0=Bottom, 1=Center, 2=Top)
                if b_code & 8:
                    y_candidates = [0.0]
                elif b_code & 4:
                    y_candidates = [c_height - h_c]
                else:
                    y_candidates = [iy, iy + (ih - h_c) / 2.0, iy + ih - h_c]

                for x_c in x_candidates:
                    for y_c in y_candidates:
                        # 1. Protection against leaving the chip
                        if x_c < -1e-4 or y_c < -1e-4 or x_c + w_c > c_width + 1e-4 or y_c + h_c > c_height + 1e-4:
                            continue

                        # 2. Strict check for overlaps with ANY blocks
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

                        # 3. Does the new shape reach the desired island?
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
                            break  # Perfect anchor found!

                    if best_shape is not None: break
                if best_shape is not None: break

            # Apply found shape and move to next block
            if best_shape is not None:
                pos[idx] = best_shape
                fixes += 1

                # If shape changed, we update touch graph on next iteration
                # of outer loop, so we can interrupt processing of current island
                break

    return torch.tensor(pos, device=positions.device, dtype=positions.dtype), fixes


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
        pred_norm_k,
        area_i,
        scale_i,
        curr_constraints,
        raw,
        device,
        steps=50,
        lr_max=0.03,
        feasible_patience=FEASIBLE_PATIENCE_CHECKS,
):
    """
    Gradient-based refinement on top of the model prediction.
    Logic is preserved, but all constant tensors are expected to be already on GPU and are not copied in each step.
    """

    # Diversity is already ensured by different checkpoint weights
    refined_init = pred_norm_k.detach().clone()
    K = refined_init.shape[0]
    refined = refined_init.requires_grad_(True)

    optimizer = torch.optim.Adam([refined], lr=lr_max)

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
                continue  # MIB недостижим — не связываем r, даём градиенту свободу
            hard_in_group = mask & is_hard
            ref_idx = torch.where(hard_in_group)[0][0] if bool(hard_in_group.any().item()) else None
            mib_groups.append((mask, ref_idx))

    target_fp = raw['target_fp']
    b2b_conn = raw['b2b_conn']
    p2b_conn = raw['p2b_conn']
    pins_pos = raw['pins_pos']
    metrics = raw['metrics']

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

        for step in range(steps):
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

            # --- Fully asynchronous update of the best solution (every step, per-sample) ---
            not_positive_f = (refined.detach()[..., :2].amin(dim=(-2, -1)) < 0.0).to(torch.float32)
            key = violations_f + 1.1 * not_positive_f
            loss_d = loss_vec.detach().to(torch.float32)

            improve = (key < best_key) | ((key == best_key) & (loss_d < best_loss_t))  # (K,)

            # --- NEW FREEZE LOGIC ---
            if FREEZE_ON_FEASIBLE:
                # Mask of variants that have ALREADY achieved validity (before this step)
                already_feasible = (best_key <= 0.0)
                # Reset improvement flag: they will forever remain in the first valid state
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
                    feas.all().to(torch.float32)  # <--- ADDED: check that ALL variants are legal
                ]).cpu()
                any_feasible = bool(snapshot[0] > 0)
                best_feasible_loss = float(snapshot[1])
                all_feasible = bool(snapshot[3] > 0)  # <--- ADDED: read result

                if any_feasible:
                    # plateau of the best feasible-loss
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

    # Selection of the best sample: minimum key (violations), in case of a tie — minimum loss
    # DELETE:
    # min_key = best_key.min()
    # cand_loss = torch.where(best_key == min_key, best_loss_t, torch.full_like(best_loss_t, INF))
    # sel = int(cand_loss.argmin().item())
    # ...
    # chosen = best_refined[sel]
    # refined_clamped = torch.cat([chosen[:, :2], chosen[:, 2:3]], dim=-1)

    # KEEP/ADD:
    # Return the entire batch of K variants
    refined_clamped_all = torch.cat([best_refined[..., :2], best_refined[..., 2:3]], dim=-1)

    return refined_clamped_all.detach(), best_key.detach().cpu()


def legalize_remaining_overlaps_fast(
        positions: torch.Tensor,
        constraints: torch.Tensor,
        tol: float = 1e-6,
        preserve_boundary: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Resolve residual overlaps with a bounded nearest-edge search.

    Gradient refinement normally leaves only a few conflicting movable blocks.
    Candidate coordinates are induced by existing rectangle edges, so this is
    O(N^3) in the worst case and cannot enter the unbounded radial search used
    by the emergency legalizer.
    """
    pos = positions.detach().cpu().numpy().astype(np.float64).copy()
    boundary = constraints[:, 4].long().detach().cpu().numpy()
    immovable = (constraints[:, 1] == 1).detach().cpu().numpy()
    if preserve_boundary:
        immovable |= boundary > 0
    n = len(pos)
    moved = 0

    def overlaps_for(i):
        x, y, w, h = pos[i]
        ox = np.minimum(x + w, pos[:, 0] + pos[:, 2]) - np.maximum(x, pos[:, 0])
        oy = np.minimum(y + h, pos[:, 1] + pos[:, 3]) - np.maximum(y, pos[:, 1])
        hit = (ox > tol) & (oy > tol)
        hit[i] = False
        return hit

    # Each move is legal against every other block, so the number of
    # conflicting movable blocks decreases monotonically.
    for _ in range(n):
        conflict = np.array([overlaps_for(i).sum() if not immovable[i] else 0
                             for i in range(n)])
        if conflict.max(initial=0) == 0:
            break
        i = int(conflict.argmax())
        ox, oy, w, h = pos[i]
        canvas_left = float(pos[:, 0].min())
        canvas_bottom = float(pos[:, 1].min())
        canvas_right = float((pos[:, 0] + pos[:, 2]).max())
        canvas_top = float((pos[:, 1] + pos[:, 3]).max())

        candidates = [(ox, oy)]
        for j in range(n):
            if j == i:
                continue
            x, y, jw, jh = pos[j]
            for cy in (oy, y, y + jh - h):
                candidates.append((x - w - tol, cy))
                candidates.append((x + jw + tol, cy))
            for cx in (ox, x, x + jw - w):
                candidates.append((cx, y - h - tol))
                candidates.append((cx, y + jh + tol))

        code = int(boundary[i])
        if code & 1:
            candidates.extend((canvas_left, cy) for _, cy in candidates[:])
        if code & 2:
            candidates.extend((canvas_right - w, cy) for _, cy in candidates[:])
        if code & 8:
            candidates.extend((cx, canvas_bottom) for cx, _ in candidates[:])
        if code & 4:
            candidates.extend((cx, canvas_top - h) for cx, _ in candidates[:])

        best = None
        best_key = None
        other = np.arange(n) != i
        for cx, cy in candidates:
            if not np.isfinite(cx + cy):
                continue
            rx = np.minimum(cx + w, pos[other, 0] + pos[other, 2]) - np.maximum(cx, pos[other, 0])
            ry = np.minimum(cy + h, pos[other, 1] + pos[other, 3]) - np.maximum(cy, pos[other, 1])
            if np.any((rx > tol) & (ry > tol)):
                continue
            new_left = min(canvas_left, cx)
            new_bottom = min(canvas_bottom, cy)
            new_right = max(canvas_right, cx + w)
            new_top = max(canvas_top, cy + h)
            expansion = ((new_right - new_left) * (new_top - new_bottom)
                         - (canvas_right - canvas_left) * (canvas_top - canvas_bottom))
            key = abs(cx - ox) + abs(cy - oy) + max(0.0, expansion)
            if best_key is None or key < best_key:
                best_key = key
                best = (cx, cy)

        if best is None:
            best = (canvas_right + tol, canvas_bottom)
        pos[i, 0], pos[i, 1] = best
        moved += 1

    return torch.from_numpy(pos).to(device=positions.device, dtype=positions.dtype), moved


def postprocess_and_score(
        idx,
        curr_pred_abs,
        curr_constraints,
        target_fp,
        b2b_conn,
        p2b_conn,
        pins_pos,
        area_targets,
        metrics
):
    curr_pred_abs = curr_pred_abs.clone()

    is_fixed = (curr_constraints[:, 0] == 1)
    is_preplaced = (curr_constraints[:, 1] == 1)

    if is_fixed.any():
        curr_pred_abs[is_fixed, 2:] = target_fp[is_fixed, 2:]
    if is_preplaced.any():
        curr_pred_abs[is_preplaced] = target_fp[is_preplaced]

    violations, _ = check_overlap_vectorized(curr_pred_abs)
    if violations > 0:
        curr_pred_abs, _ = legalize_remaining_overlaps_fast(curr_pred_abs, curr_constraints)

    for _ in range(3):
        curr_pred_abs = apply_boundary_aware_gravity(curr_pred_abs, curr_constraints, margin=0.0)

    has_clusters = bool((curr_constraints[:, 3] > 0).any())
    if has_clusters:
        curr_pred_abs, _ = fix_grouping_violations_greedily(curr_pred_abs, curr_constraints, gap=0.0)
        curr_pred_abs, _ = reshape_cluster_blocks_to_touch(curr_pred_abs, curr_constraints)

    violations_final, _ = check_overlap_vectorized(curr_pred_abs)
    if violations_final > 0:
        curr_pred_abs, _ = legalize_remaining_overlaps_fast(curr_pred_abs, curr_constraints)

    final_violations, _ = check_overlap_vectorized(curr_pred_abs)
    return idx, final_violations, curr_pred_abs


try:
    from iccad2026_evaluate import FloorplanOptimizer
except:
    from FloorSet.iccad2026contest.iccad2026_evaluate import FloorplanOptimizer


class ContestOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

        DATA_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'
        checkpoint_paths = [
            DATA_ROOT + 'weights/direct_transformer_loss_5.6719_epoch_364.pt',
        ]
        if FORCE_CPU:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Leave a small numerical margin inside the official 1% area tolerance.
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

        # 2. Feature preparation (extracting embeddings and matrices)
        num_pins = pins_pos.shape[0] if pins_pos.numel() > 0 else 0
        graph_scale = (
            pins_pos.max().clamp_min(1.0)
            if num_pins > 0
            else torch.tensor(100.0, device=self.device)
        )

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

        # 3. Neural network inference (Ensemble)
        curr_pred_norms = []
        with torch.inference_mode():
            for model in self.models:
                pred_norm = model(
                    features,
                    padding_mask=None,
                    b2b_matrix=b2b_matrix,
                    p2b_matrix=p2b_matrix,
                    pins_pos=pins_pos_batched,
                    spd_matrix=spd_matrix
                )
                curr_pred_norms.append(pred_norm[0, :block_count])

        # Collect base predictions from all checkpoints into tensor (M, N, 3)
        curr_pred_norm_base = torch.stack(curr_pred_norms, dim=0)
        M = curr_pred_norm_base.shape[0]  # Количество чекпоинтов (10)

        # 1. Calculate mean (1 variant)
        mean_pred = curr_pred_norm_base.mean(dim=0, keepdim=True)

        # 2. Duplicate checkpoint predictions
        # Shape becomes (M * N_VARIANTS, N, 3), i.e. [1, 1..10 times, 2, 2..10 times, ...]
        repeated_preds = curr_pred_norm_base.repeat_interleave(N_VARIANTS, dim=0)

        # 3. Generate noise (use same generator logic)
        gen = torch.Generator()
        gen.manual_seed(20260612)
        noise = torch.randn(repeated_preds.shape, generator=gen, dtype=repeated_preds.dtype).to(self.device)
        noise = noise * MULTISTART_NOISE
        noise[..., 2] *= 0.5  # Change Aspect Ratio less

        # Zero noise for first (zero) variant of each checkpoint,
        # to keep original network prediction clean
        for i in range(M):
            noise[i * N_VARIANTS] = 0.0

        repeated_preds = repeated_preds + noise

        # 4. Append mean prediction to the end
        # Shape: (M * N_VARIANTS + 1, N, 3) -> (101, N, 3)
        curr_pred_norm_k = (
            repeated_preds
            if M == 1
            else torch.cat([repeated_preds, mean_pred], dim=0)
        )

        curr_pred_abs_initial = decode_from_direct_space(curr_pred_norm_k, area_targets, graph_scale)

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
                ref_area = area_targets[ref_idx]  # вместо ref_w * ref_h
            else:
                group_w = curr_pred_abs_initial[0, group_indices, 2]
                group_h = curr_pred_abs_initial[0, group_indices, 3]
                mean_r = torch.log(group_w / (group_h + 1e-6)).mean()
                ref_area = area_targets[group_indices[0]]
                ref_w = torch.sqrt(ref_area) * torch.exp(mean_r * 0.5)
                ref_h = torch.sqrt(ref_area) * torch.exp(-mean_r * 0.5)

            # Площадь — жёсткое ограничение (cost=10), MIB — мягкое.
            # Единую форму навязываем только если она не ломает площадь.
            g_area = area_targets[group_indices]
            areas_equal = bool(
                ((g_area - ref_area).abs() / ref_area.clamp(min=1e-9) <= AREA_TOLERANCE).all()
            )

            if areas_equal:
                features[0, group_indices, 1] = 1.0
                target_fp[0, group_indices, 2] = ref_w
                target_fp[0, group_indices, 3] = ref_h
            else:
                # MIB недостижим: сохраняем площадь каждого блока,
                # блоки НЕ помечаем как fixed — пусть refine их оптимизирует.
                pass

        # 5. Optimization (Refinement Loop)
        curr_constraints = features[0, :block_count, 1:6]

        # Use passed real metrics or create mock if not passed
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

        refined_clamped_all, variant_keys = refine_layout(
            curr_pred_norm_k,
            area_targets,
            graph_scale,
            curr_constraints,
            raw_gpu,
            self.device,
            steps=STEPS,
            lr_max=MAX_LR,
        )

        # Decode all K variants simultaneously (Shape: K, N, 4)
        all_pred_abs = decode_from_direct_space(refined_clamped_all, area_targets, graph_scale)

        keys_np = variant_keys.numpy().astype(np.float64)
        keep = set(np.nonzero(keys_np <= 0.0)[0].tolist())
        for idx in np.argsort(keys_np, kind='stable'):
            if len(keep) >= MAX_POSTPROCESS_VARIANTS:
                break
            keep.add(int(idx))
        keep = sorted(keep)

        results = [postprocess_and_score(
            k, all_pred_abs[k], curr_constraints, target_fp[0],
            b2b_connectivity, p2b_connectivity, pins_pos, area_targets, metrics
        ) for k in keep]

        # Sort and select objectively best variant after all mutations
        results.sort(key=lambda x: x[1])
        best_idx, best_key, curr_pred_abs = results[0]

        # 7. Converting the result to List[Tuple[float, float, float, float]]
        final_positions = curr_pred_abs.detach().cpu().numpy()
        positions_list = [(float(x), float(y), float(w), float(h)) for x, y, w, h in final_positions]

        return positions_list
