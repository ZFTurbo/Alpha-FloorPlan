from config import *
import json
import torch
import math
import tqdm
import datetime
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field


try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from shapely.affinity import rotate, translate
    from shapely.geometry import LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("WARNING: shapely is not installed. Soft constraint violations "
          "(fixed, preplaced, grouping) will not be computed. "
          "Install it with: pip install shapely>=2.0.0")


def check_preplaced_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon], threshold: float = 0.001) -> int:
    """
    Check for violations of the preplaced constraint by evaluating intersection areas.

    Args:
        indices (torch.Tensor): Indices to check for preplaced constraints.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.
        threshold (float): The threshold for intersection area comparison.

    Returns:
        int: The count of violations found.
    """
    viol_count = sum(
        polygon1.intersection(polygon2).area + threshold <= polygon1.area
        for index in indices
        for polygon1, polygon2 in [(pred_sol[index], target_sol[index])]
    )
    return viol_count


def normalize_polygon(polygon: Polygon) -> Polygon:
    """
    Normalize a polygon by translating its bounding box to the origin and rotating it to align with axes.

    Args:
        polygon (Polygon): The input Shapely polygon to be normalized.

    Returns:
        Polygon: The normalized polygon.
    """
    bbox = polygon.minimum_rotated_rectangle
    bbox_coords = list(bbox.exterior.coords)[:-1]
    min_x = min(coord[0] for coord in bbox_coords)
    min_y = min(coord[1] for coord in bbox_coords)
    translated_polygon = translate(polygon, xoff=-min_x, yoff=-min_y)

    # Get the oriented bounding box again after translation
    bbox = translated_polygon.minimum_rotated_rectangle
    bbox_coords = list(bbox.exterior.coords)[:-1]
    angle = np.arctan2(bbox_coords[1][1] - bbox_coords[0][1], bbox_coords[1][0] - bbox_coords[0][0])

    aligned_polygon = rotate(translated_polygon, -np.degrees(angle), origin='centroid')
    return aligned_polygon


def polygons_have_same_shape(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-3) -> bool:
    """
    Determine if two polygons have the same shape, disregarding location.

    Args:
        poly1 (Polygon): The first polygon for comparison.
        poly2 (Polygon): The second polygon for comparison.
        tolerance (float): The tolerance for area and equality comparison.

    Returns:
        bool: True if polygons have the same shape, False otherwise.
    """
    if not np.isclose(poly1.area, poly2.area, atol=tolerance):
        return False

    norm_poly1 = normalize_polygon(poly1)
    norm_poly2 = normalize_polygon(poly2)

    return norm_poly1.equals_exact(norm_poly2, tolerance)


def check_fixed_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon]) -> int:
    """
    Check for violations of the fixed constraint by comparing predicted and target polygons.

    Args:
        indices (torch.Tensor): Indices to check for fixed constraints.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.

    Returns:
        int: The count of violations found.
    """
    viol_count = sum(
        not polygons_have_same_shape(pred_sol[index], target_sol[index])
        for index in indices
    )
    return viol_count


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================
def calculate_hpwl_b2b(positions, b2b_connectivity):
    """Block-to-block HPWL. Vectorized; the result is identical to the row-by-row version.

    The original semantics are preserved: an edge is counted when edge[0] != -1 and
    i < N and j < N. Negative j values (other than -1, filtered by edge[0]) are
    indexed by numpy like Python positions[j] (wrap-around), which matches the original.
    """
    if b2b_connectivity is None or len(b2b_connectivity) == 0:
        return 0.0
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    cx = pos[:, 0] + pos[:, 2] * 0.5
    cy = pos[:, 1] + pos[:, 3] * 0.5

    b = b2b_connectivity
    if torch.is_tensor(b):
        b = b.detach().cpu().numpy()
    b = np.asarray(b, dtype=np.float64)
    if b.shape[0] == 0:
        return 0.0

    i = b[:, 0].astype(np.int64)
    j = b[:, 1].astype(np.int64)
    w = b[:, 2]
    mask = (b[:, 0] != -1) & (i < n) & (j < n)
    if not mask.any():
        return 0.0
    i, j, w = i[mask], j[mask], w[mask]
    d = np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j])
    return float((w * d).sum())


def calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos):
    """Pin-to-block HPWL. Vectorized; the result is identical to the row-by-row version."""
    if p2b_connectivity is None or len(p2b_connectivity) == 0:
        return 0.0
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    cx = pos[:, 0] + pos[:, 2] * 0.5
    cy = pos[:, 1] + pos[:, 3] * 0.5

    pins = pins_pos
    if torch.is_tensor(pins):
        pins = pins.detach().cpu().numpy()
    pins = np.asarray(pins, dtype=np.float64)
    npins = pins.shape[0]

    p = p2b_connectivity
    if torch.is_tensor(p):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float64)
    if p.shape[0] == 0:
        return 0.0

    pin = p[:, 0].astype(np.int64)
    blk = p[:, 1].astype(np.int64)
    w = p[:, 2]
    mask = (p[:, 0] != -1) & (blk < n) & (pin < npins)
    if not mask.any():
        return 0.0
    pin, blk, w = pin[mask], blk[mask], w[mask]
    d = np.abs(pins[pin, 0] - cx[blk]) + np.abs(pins[pin, 1] - cy[blk])
    return float((w * d).sum())


def calculate_bbox_area(positions: List[Tuple[float, float, float, float]]) -> float:
    """Calculate bounding box area of all blocks."""
    if not positions:
        return 0.0

    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)

    return (x_max - x_min) * (y_max - y_min)


def check_overlap(positions):
    """Number of overlapping pairs (edge touching is OK). Threshold 1e-6.

    Note: the threshold here is 1e-6, as in the original check_overlap. Do NOT confuse
    it with check_overlap_vectorized, where 1e-7 is used — that is a different
    function for a different place.
    """
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        return 0
    x = pos[:, 0]
    y = pos[:, 1]
    r = x + pos[:, 2]
    b = y + pos[:, 3]
    ox = np.minimum(r[:, None], r[None, :]) - np.maximum(x[:, None], x[None, :])
    oy = np.minimum(b[:, None], b[None, :]) - np.maximum(y[:, None], y[None, :])
    both = (ox > 1e-6) & (oy > 1e-6)
    return int(np.count_nonzero(np.triu(both, 1)))


def check_area_tolerance(positions, target_areas, tolerance=AREA_TOLERANCE, skip_indices=None):
    """Number of soft blocks outside the area tolerance. Identical to the row-by-row version."""
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]

    ta = target_areas
    if torch.is_tensor(ta):
        ta = ta.detach().cpu().numpy()
    ta = np.asarray(ta, dtype=np.float64)
    m = ta.shape[0]

    actual = pos[:, 2] * pos[:, 3]
    k = min(n, m)
    taf = np.full(n, -1.0, dtype=np.float64)
    taf[:k] = ta[:k]
    valid = np.zeros(n, dtype=bool)
    valid[:k] = True  # i < len(target_areas)

    skip = np.zeros(n, dtype=bool)
    if skip_indices:
        si = np.fromiter((s for s in skip_indices if 0 <= s < n), dtype=np.int64)
        if si.size:
            skip[si] = True

    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(actual - taf) / taf
    cond = valid & (~skip) & (taf != -1) & (taf > 0) & (diff > tolerance)
    return int(np.count_nonzero(cond))


def check_dimension_hard_constraints(
        positions: List[Tuple[float, float, float, float]],
        target_positions: Optional[List[Tuple[float, float, float, float]]],
        target_constraints: Optional[torch.Tensor],
        block_count: int,
        tolerance: float = 1e-4
) -> int:
    """
    Check that fixed-shape and preplaced blocks have immutable dimensions.

    Per PDF hard constraints:
      - Fixed-shape blocks: (w, h) must match input specification exactly.
      - Preplaced blocks: (x, y, w, h) must all match input specification.
      - "Any solution that deviates from the fixed dimensions is classified
        as infeasible."

    Returns the number of blocks that violate these requirements.
    """
    if target_positions is None or target_constraints is None:
        return 0
    if len(target_constraints) < block_count:
        return 0

    ncols = target_constraints.shape[1]
    violations = 0

    for i in range(min(block_count, len(positions), len(target_positions))):
        is_fixed = ncols > 0 and target_constraints[i, 0] != 0
        is_preplaced = ncols > 1 and target_constraints[i, 1] != 0

        if not (is_fixed or is_preplaced):
            continue

        px, py, pw, ph = positions[i]
        tx, ty, tw, th = target_positions[i]

        if abs(pw - tw) > tolerance or abs(ph - th) > tolerance:
            violations += 1
            continue

        if is_preplaced:
            if abs(px - tx) > tolerance or abs(py - ty) > tolerance:
                violations += 1

    return violations


def compute_cost(
        hpwl_gap: float,
        area_gap: float,
        violations_relative: float,
        runtime_factor: float,
        is_feasible: bool
) -> float:
    """
    Compute the official contest cost.

    Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_rel) × max(0.7, R^γ)
         = M (10.0) if infeasible

    Infeasible means ANY hard constraint is violated:
        - block overlaps
        - soft-block area outside 1% tolerance
        - fixed-shape dimensions deviate from input
        - preplaced position or dimensions deviate from input

    Note: HPWL_gap and Area_gap are clamped to zero from below (max(0, gap));
    beating the baseline gives no additional score reduction.

    Note: feasible cost is capped at M−ε (9.999999) so that any feasible
    solution always scores strictly better than an infeasible one (M=10.0).
    """
    if not is_feasible:
        return M_PENALTY

    quality_factor = 1 + ALPHA * (max(0, hpwl_gap) + max(0, area_gap))
    violation_factor = math.exp(BETA * violations_relative)
    runtime_adjustment = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA))

    # Cap feasible cost strictly below M so that any feasible solution
    # (no overlaps, area tolerance met, fixed/preplaced respected) always
    # scores better than an infeasible one (cost = M).
    return min(quality_factor * violation_factor * runtime_adjustment,
               M_PENALTY - 1e-6)


def compute_total_score(costs: List[float], block_counts: List[int]) -> float:
    """
    Compute exponentially weighted average score.

    Total Score = Σ Cost[i] · e^{n_i/12} / Σ e^{n_j/12}

    where n_i is the block count for test case i. The /12 scaling ensures
    every size from 21 to 120 carries non-zero weight while still strongly
    favouring larger instances:
        exp(n)    → n=120 alone ≈ 63%, cases below n=116 ≈ 1% total
        exp(n/4)  → n=120 alone ≈ 22%, cases below n=111 ≈ 8% total
        exp(n/12) → n=120 alone ≈ 34%, cases below n=111 ≈ 28% total
                    n=21-25 bucket: 0.01% (first option with full-range coverage)
    """
    if not costs:
        return 0.0
    if not block_counts or all(n == 0 for n in block_counts):
        return sum(costs) / len(costs)

    max_n = max(block_counts)
    weights = [math.exp((n - max_n) / 12) for n in block_counts]
    total_weight = sum(weights)
    return sum(c * w for c, w in zip(costs, weights)) / total_weight


class FloorplanDatasetLiteTest(Dataset):
    def __init__(self, root):
        self.all_input_files = []
        self.all_label_files = []
        partition_range = range(21, 121)  # number of partitions in prime
        identifier_range = range(1, 11)  # Identifiers from 1 to 10

        for worker_idx in partition_range:
            config_dir = os.path.join(root, f'FloorSet_data/LiteTensorDataTest/config_{worker_idx}')
            # Collect data files within the specified identifier range
            for identifier in identifier_range:
                input_file_pattern = os.path.join(config_dir, f'litedata_{identifier}.pth')
                label_file_pattern = os.path.join(config_dir, f'litelabel_{identifier}.pth')
                if os.path.isfile(input_file_pattern):
                    self.all_input_files.append(input_file_pattern)
                if os.path.isfile(label_file_pattern):
                    self.all_label_files.append(label_file_pattern)

        self.layouts_per_file = 1
        self.cached_file_idx = -1

    def __len__(self):
        return len(self.all_input_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_input_file_contents = torch.load(self.all_input_files[file_idx])
            self.cached_label_file_contents = torch.load(self.all_label_files[file_idx])
            self.cached_file_idx = file_idx

        area_target = self.cached_input_file_contents[layout_idx][0][:, 0]
        placement_constraints = self.cached_input_file_contents[layout_idx][0][:, 1:]
        b2b_connectivity = self.cached_input_file_contents[layout_idx][1]
        p2b_connectivity = self.cached_input_file_contents[layout_idx][2]
        pins_pos = self.cached_input_file_contents[layout_idx][3]

        fp_sol = self.cached_label_file_contents[layout_idx][1]
        metrics_sol = self.cached_label_file_contents[layout_idx][0]

        input_data = (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
        label_data = (fp_sol, metrics_sol)
        sample = {'input': input_data, 'label': label_data}
        return sample


# =============================================================================
# SCORE SAVED SOLUTIONS
# =============================================================================
def score_saved_solutions_local(
        solutions_path: str,
        data_path: str = "../",
        output_path: Optional[str] = None,
        dataset=None
) -> Dict:
    """
    Re-score saved solutions without re-running the optimizer.

    Args:
        solutions_path: Path to solutions JSON (from --save-solutions)
        data_path: Path to FloorSet data
        output_path: Output file path (optional)

    Returns:
        Dict with scores
    """
    print(f"\nScoring saved solutions: {solutions_path}")
    print("=" * 60)

    # Load solutions
    with open(solutions_path) as f:
        data = json.load(f)

    solutions = data.get('solutions', [])
    print(f"Loaded {len(solutions)} solutions")

    # Load test dataset
    if dataset is None:
        dataset = FloorplanDatasetLiteTest(data_path)

    results = []

    for sol in tqdm.tqdm(solutions, desc="Scoring"):
        test_id = sol['test_id']
        positions = [tuple(p) for p in sol['positions']]
        block_count = sol['block_count']

        sample = dataset[test_id]
        if 'input' in sample:  # old test format
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = sample['input']
            polygons, metrics = sample['label']
            gt_positions = []
            for i in range(block_count):
                valid = polygons[i][polygons[i][:, 0] != -1]
                if len(valid) > 0:
                    x_min, y_min = valid.min(dim=0).values
                    x_max, y_max = valid.max(dim=0).values
                    gt_positions.append((float(x_min), float(y_min),
                                         float(x_max - x_min), float(y_max - y_min)))
                else:
                    gt_positions.append((0, 0, 1, 1))
        else:  # dict format (train / optimized)
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = (
                sample['area_target'], sample['b2b_conn'], sample['p2b_conn'],
                sample['pins_pos'], sample['constraints'])
            metrics = sample['metrics']
            gt_positions = [tuple(map(float, p))
                            for p in sample['fp_sol'].cpu().numpy()[:block_count]]

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
        solution_metrics = evaluate_solution_fast(
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

        results.append({
            'test_id': test_id,
            'block_count': block_count,
            'is_feasible': solution_metrics.is_feasible,
            'hpwl_gap': solution_metrics.hpwl_gap,
            'area_gap': solution_metrics.area_gap,
            'cost': solution_metrics.cost,
            'hpwl_total': solution_metrics.hpwl_total,
            'bbox_area': solution_metrics.bbox_area,
            'overlaps': solution_metrics.overlap_violations,
            'area_violations': solution_metrics.area_violations
        })

    # Compute total score
    costs = [r['cost'] for r in results]
    blocks = [r['block_count'] for r in results]
    total_score = compute_total_score(costs, blocks)

    # Print summary
    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"\nTotal Score: {total_score:.4f}")
    print(f"Tests: {len(results)}")
    print(f"Feasible: {sum(1 for r in results if r['is_feasible'])}")
    print(f"Avg Cost: {sum(costs) / len(costs):.4f}")

    output = {
        'source': solutions_path,
        'timestamp': datetime.datetime.now().isoformat(),
        'total_score': total_score,
        'results': results
    }

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output



# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class SolutionMetrics_fast:
    """Container for all evaluation metrics.

    Hard constraints (any violation → infeasible, cost = M = 10):
        overlap_violations   — number of overlapping block pairs
        area_violations      — soft blocks exceeding 1% area tolerance
        dimension_violations — fixed-shape or preplaced blocks whose
                               dimensions (or location) deviate from input

    Soft constraints (violations feed into exponential penalty):
        boundary_violations  — blocks not touching required bbox edge/corner
        grouping_violations  — Σ(connected_components - 1) per group
        mib_violations       — Σ(distinct_shapes - 1) per MIB group

    Informational only (not used in scoring):
        fixed_violations     — Shapely-based shape mismatch count (debugging)
        preplaced_violations — Shapely-based placement mismatch count (debugging)
    """
    is_feasible: bool
    overlap_violations: int
    area_violations: int
    dimension_violations: int
    hpwl_b2b: float
    hpwl_p2b: float
    hpwl_total: float
    hpwl_baseline: float
    hpwl_gap: float
    bbox_area: float
    bbox_area_baseline: float
    area_gap: float
    fixed_violations: int       # informational only — hard constraint
    preplaced_violations: int   # informational only — hard constraint
    boundary_violations: int
    grouping_violations: int
    mib_violations: int
    total_soft_violations: int
    max_possible_violations: int
    violations_relative: float
    runtime_seconds: float
    cost: float


def evaluate_solution_fast(
        solution: Dict,
        baseline_metrics: Dict,
        target_constraints: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        target_areas: torch.Tensor,
        target_positions: Optional[List] = None,
        median_runtime: float = 1.0
) -> SolutionMetrics_fast:
    """Evaluate a solution and compute all metrics."""
    positions = solution['positions']
    runtime = solution.get('runtime', 1.0)
    block_count = len(positions)

    # Calculate HPWL
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
    hpwl_total = hpwl_b2b + hpwl_p2b

    hpwl_baseline = baseline_metrics.get('hpwl_baseline', hpwl_total)
    hpwl_gap = (hpwl_total - hpwl_baseline) / max(hpwl_baseline, 1e-6)

    # Calculate area
    bbox_area = calculate_bbox_area(positions)
    area_baseline = baseline_metrics.get('area_baseline', bbox_area)
    area_gap = (bbox_area - area_baseline) / max(area_baseline, 1e-6)

    # Check hard constraints (feasibility).
    # Three conditions must all hold for a feasible solution:
    #   1. No block overlaps (intersection area = 0 for all pairs).
    #   2. Soft-block areas within 1% of target (|w*h - a| / a ≤ 0.01).
    #   3. Fixed-shape / preplaced dimensions (and locations for preplaced)
    #      match the input specification exactly (tolerance 1e-4).
    overlap_violations = check_overlap(positions)

    # Identify fixed/preplaced blocks so they are excluded from the 1%
    # area-tolerance check (they have a stricter exact-dimension check).
    fixed_or_preplaced = set()
    if target_constraints is not None and len(target_constraints) >= block_count:
        ncols_hc = target_constraints.shape[1]
        for i in range(block_count):
            if (ncols_hc > 0 and target_constraints[i, 0] != 0) or \
                    (ncols_hc > 1 and target_constraints[i, 1] != 0):
                fixed_or_preplaced.add(i)

    area_violations = check_area_tolerance(
        positions, target_areas, skip_indices=fixed_or_preplaced)
    dimension_violations = check_dimension_hard_constraints(
        positions, target_positions, target_constraints, block_count)
    is_feasible = (overlap_violations == 0 and area_violations == 0
                   and dimension_violations == 0)

    # =========================================================================
    # Soft constraint violations
    #
    # Fixed-shape and preplaced are HARD constraints: any deviation from
    # the specified dimensions (or location for preplaced) makes the
    # solution infeasible (cost = M = 10).  They are enforced above via
    # check_dimension_hard_constraints().
    #
    # The remaining soft constraints are boundary, grouping, and MIB:
    #
    #   Violations_relative = (V_boundary + V_grouping + V_mib) / N_soft
    #
    # N_soft is the maximum possible soft violations (normalization
    # constant so that Violations_relative ∈ [0, 1]):
    #
    #   N_soft = |B_boundary|
    #            + Σ_p (|G_p| - 1)       ← max grouping violations
    #            + Σ_q (|M_q| - 1)       ← max MIB violations
    #
    # Per-block constraints (boundary) contribute 1 each because each
    # block either satisfies (0) or violates (1).
    #
    # Per-group constraints contribute (group_size - 1) each because:
    #   • Grouping worst case: all blocks isolated → c_p = |G_p|,
    #     violation = c_p - 1 = |G_p| - 1
    #   • MIB worst case: all blocks have different shapes → s_q = |M_q|,
    #     violation = s_q - 1 = |M_q| - 1
    # =========================================================================
    fixed_violations = 0
    preplaced_violations = 0
    boundary_violations = 0
    grouping_violations = 0
    mib_violations = 0
    n_soft = 0  # maximum possible violations (PDF: N_soft)

    if target_constraints is not None and len(target_constraints) >= block_count:
        constraints_block = target_constraints[:block_count]
        ncols = constraints_block.shape[1]

        # Constraint tensor columns: [fixed, preplaced, mib_id, cluster_id, boundary_code]
        fixed_const = constraints_block[:, 0] if ncols > 0 else torch.zeros(block_count)
        preplaced_const = constraints_block[:, 1] if ncols > 1 else torch.zeros(block_count)
        mib_const = constraints_block[:, 2] if ncols > 2 else torch.zeros(block_count)
        clust_const = constraints_block[:, 3] if ncols > 3 else torch.zeros(block_count)
        bound_const = constraints_block[:, 4] if ncols > 4 else torch.zeros(block_count)

        # Count blocks that carry each per-block constraint
        n_fixed = int((fixed_const != 0).sum().item())
        n_preplaced = int((preplaced_const != 0).sum().item())
        n_boundary = int((bound_const != 0).sum().item())

        # -----------------------------------------------------------------
        # N_soft: maximum possible soft violations (normalization constant
        # so that Violations_relative ∈ [0, 1]).
        #
        # Fixed-shape and preplaced are hard constraints (dimension
        # deviations make the solution infeasible), so they are excluded
        # from the soft-violation denominator.  Only boundary (per-block),
        # grouping (per-group), and MIB (per-group) count here.
        # -----------------------------------------------------------------
        n_soft = n_boundary

        # Per-group (MIB): worst case is |M_q| distinct shapes → |M_q|-1
        n_mib_groups = int(mib_const.max().item()) if mib_const.numel() > 0 else 0
        for g in range(1, n_mib_groups + 1):
            group_size = int((mib_const == g).sum().item())
            n_soft += max(0, group_size - 1)

        # Per-group (grouping): worst case is |G_p| isolated blocks → |G_p|-1
        n_clust_groups = int(clust_const.max().item()) if clust_const.numel() > 0 else 0
        for g in range(1, n_clust_groups + 1):
            group_size = int((clust_const == g).sum().item())
            n_soft += max(0, group_size - 1)

        # -----------------------------------------------------------------
        # Actual violation counts (PDF Eq. 3 numerator)
        # -----------------------------------------------------------------

        if SHAPELY_AVAILABLE:
            pred_polys = [box(x, y, x + w, y + h) for x, y, w, h in positions]
            target_polys = None
            if target_positions is not None:
                target_polys = [box(tx, ty, tx + tw, ty + th)
                                for tx, ty, tw, th in target_positions]

            # Fixed-shape and preplaced violations are reported for
            # informational/debugging purposes only.  They do NOT
            # contribute to the soft-violation score — deviations from
            # specified dimensions or locations are hard constraints
            # (checked by check_dimension_hard_constraints above) and
            # make the solution infeasible (cost = M).
            if n_fixed > 0 and target_polys is not None:
                fixed_violations = check_fixed_const(
                    torch.nonzero(fixed_const).flatten(), pred_polys, target_polys)

            if n_preplaced > 0 and target_polys is not None:
                preplaced_violations = check_preplaced_const(
                    torch.nonzero(preplaced_const).flatten(), pred_polys, target_polys)

            # V_grouping = Σ(c_p - 1): connected components minus 1 per group.
            # Blocks that share an edge form a connected component.
            # Perfectly abutted group → c_p=1 → violation=0.
            for g in range(1, n_clust_groups + 1):
                group_indices = torch.where(clust_const == g)[0].tolist()
                group_polys = [pred_polys[i] for i in group_indices]
                union_result = unary_union(group_polys)
                if union_result.geom_type == 'MultiPolygon':
                    grouping_violations += len(union_result.geoms) - 1

        # V_mib = Σ(s_q - 1): distinct (w,h) pairs minus 1 per group.
        # All blocks in a group should share identical dimensions.
        # Perfectly uniform group → s_q=1 → violation=0.
        for g in range(1, n_mib_groups + 1):
            group_indices = torch.where(mib_const == g)[0].tolist()
            distinct_shapes = set()
            for i in group_indices:
                bw, bh = round(positions[i][2], 4), round(positions[i][3], 4)
                distinct_shapes.add((bw, bh))
            mib_violations += len(distinct_shapes) - 1

        # V_boundary: 1 per block that doesn't touch its required bbox edge/corner.
        # Encoding is a bitmask: 1=left, 2=right, 4=top, 8=bottom.
        # Corners are sums, e.g. 5=top-left (4+1), 10=bottom-right (8+2).
        if n_boundary > 0:
            x_min_bb = min(p[0] for p in positions)
            y_min_bb = min(p[1] for p in positions)
            x_max_bb = max(p[0] + p[2] for p in positions)
            y_max_bb = max(p[1] + p[3] for p in positions)
            eps = 1e-6

            for i in range(block_count):
                code = int(bound_const[i].item())
                if code == 0:
                    continue
                bx, by, bw, bh = positions[i]
                touches = {
                    1: abs(bx - x_min_bb) < eps,  # left edge
                    2: abs(bx + bw - x_max_bb) < eps,  # right edge
                    4: abs(by + bh - y_max_bb) < eps,  # top edge
                    8: abs(by - y_min_bb) < eps,  # bottom edge
                }
                if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                    boundary_violations += 1

    # Fixed-shape and preplaced constraints are hard constraints (enforced
    # via check_dimension_hard_constraints above).  Only boundary, grouping,
    # and MIB remain as soft constraints whose violations feed into the
    # exponential penalty term.
    total_soft_violations = (boundary_violations + grouping_violations
                             + mib_violations)
    violations_relative = total_soft_violations / max(n_soft, 1)

    # Compute cost
    runtime_factor = runtime / max(median_runtime, 0.01)
    cost = compute_cost(hpwl_gap, area_gap, violations_relative, runtime_factor, is_feasible)

    return SolutionMetrics_fast(
        is_feasible=is_feasible,
        overlap_violations=overlap_violations,
        area_violations=area_violations,
        dimension_violations=dimension_violations,
        hpwl_b2b=hpwl_b2b,
        hpwl_p2b=hpwl_p2b,
        hpwl_total=hpwl_total,
        hpwl_baseline=hpwl_baseline,
        hpwl_gap=hpwl_gap,
        bbox_area=bbox_area,
        bbox_area_baseline=area_baseline,
        area_gap=area_gap,
        fixed_violations=fixed_violations,
        preplaced_violations=preplaced_violations,
        boundary_violations=boundary_violations,
        grouping_violations=grouping_violations,
        mib_violations=mib_violations,
        total_soft_violations=total_soft_violations,
        max_possible_violations=n_soft,
        violations_relative=violations_relative,
        runtime_seconds=runtime,
        cost=cost
    )

