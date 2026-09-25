# Alpha FloorPlan

Solution for Problem C of ICCAD Contest 2026: The FloorSet Challenge.

## Problem description

Fixed-outline SoC floorplanning: arrange `k` rectangular blocks (21–120 per test case)
on a 2D canvas so that wirelength and bounding-box area are minimized while all
placement constraints hold. The origin `(0, 0)` is the lower-left corner of the canvas.

**Input**

* Area target `a_i` for each block — blocks are soft, their aspect ratio is free.
* `r` terminals (pins) at fixed coordinates on the canvas, used for external interfacing.
* Weighted block-to-block connectivity `W_int` and block-to-terminal connectivity `W_ext`.
* Per-block placement constraints (fixed shape, preplaced, MIB group, cluster, boundary).

**Output** — for every block, the lower-left corner `(x_i, y_i)` and the dimensions `(w_i, h_i)`,
so that block `b_i` occupies the region `[x_i, x_i + w_i] x [y_i, y_i + h_i]`.

### Hard constraints

Violating any of them makes the solution infeasible and costs a flat `M = 10`
for that test case:

* soft-block area within 1% of target: `|w*h - a| / a <= 0.01`;
* strictly overlap-free placement (touching edges are allowed);
* fixed-shape blocks keep their input `(w, h)` exactly;
* preplaced blocks keep their input `(x, y, w, h)` exactly.

### Soft constraints

Penalized but not disqualifying:

* **grouping** — blocks of a group should be abutted into a single connected component;
* **MIB** — blocks of a group should share identical dimensions;
* **boundary** — the block should touch the specified edge (or corner) of the bounding box.

Violations are counted per block (boundary) or per group (grouping: connected
components − 1, MIB: distinct shapes − 1) and normalized:

    Violations_rel = (V_grouping + V_boundary + V_mib) / N_soft

    N_soft = |B_boundary| + sum_p (|G_p| - 1) + sum_q (|M_q| - 1)

so that `Violations_rel` lies in `[0, 1]`.

### Cost per test case

    Cost = min( (1 + a * (HPWL_gap + Area_gap)) * exp(b * Violations_rel) * max(0.7, R^g),
                M - 1e-6 )

    Cost = M = 10     if the solution is infeasible

with `a = 0.5`, `b = 2.0`, `g = 0.3`.

* `HPWL_gap` — relative gap of the achieved half-perimeter wirelength against the
  optimal baseline shipped with the dataset.
* `Area_gap` — relative gap of the achieved bounding-box area against the baseline.
* `R` — your runtime divided by the median runtime of all submissions for that test
  case. Speed-up is capped at −30%, slowness is uncapped.

HPWL sums weighted Manhattan distances between block centroids (`c_x = x + w/2`,
`c_y = y + h/2`) and between block centroids and terminals.

### Total score

Weighted average over 100 hidden test cases, one per block count from 21 to 120:

    Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)

Large instances dominate the score. Lower is better; a solution matching the baseline
everywhere at median runtime scores about 1.0.

### Dataset

FloorSet-Lite — rectangular blocks with a fixed rectangular outline, optimal-by-construction
layouts:

| Split | Size | Availability |
|---|---|---|
| Training | 1M samples, 21–120 blocks | public |
| Validation | 100 samples, one per size | public |
| Test | 100 samples, one per size | hidden |

Full rules and the official evaluator: [contest repository](https://github.com/IntelLabs/FloorSet/tree/main/iccad2026contest).

## Solution overview

The solver combines a graph transformer that predicts a full layout in one forward pass
with a short gradient refinement of that prediction and a deterministic legalization pass.
No search over discrete representations (B*-tree, sequence pair) is used at any stage.

### 1. Geometry parameterization

Each block is described by three numbers instead of four:

    x = x_norm * scale          x_norm in (0, 1)   via sigmoid
    y = y_norm * scale          y_norm in (0, 1)   via sigmoid
    w = sqrt(area) * exp(r/2)   r = r_norm * R_SCALE, r_norm in (-1, 1) via tanh
    h = sqrt(area) * exp(-r/2)

Because `w * h = area` identically, the 1% area tolerance — one of the hard constraints —
is satisfied by construction and never has to be optimized for. The network only has to
choose a position and an aspect ratio. Area targets are additionally scaled by `0.991`
before decoding, which leaves a safety margin against rounding in the official checker.

### 2. Model

A transformer encoder over blocks as tokens:

* **14 input features per block** — normalized `sqrt(area)`, 5 placement constraints,
  3 spectral graph embeddings (Fiedler vectors of the b2b Laplacian), the weighted
  center of mass of the pins the block connects to, and the known aspect ratio /
  coordinates for fixed and preplaced blocks.
* **Attention bias** built from the shortest-path-distance matrix (hops, capped at 20,
  as a learned embedding per head) plus a learned projection of the b2b edge weights —
  the graph topology enters attention directly rather than through message passing.
* **Pin anchors** — pin coordinates are projected and added to the block embeddings
  through the p2b incidence matrix.
* 12 layers, hidden dim 256, 8 heads, pre-norm blocks.
* MIB groups are enforced architecturally: after the head, `r_norm` inside a group is
  replaced by the value of its first hard block, or by the group mean if there is none.

### 3. Training

Supervised on the optimal-by-construction layouts, with three terms:

* MSE between decoded `(x, y, w, h)` and the ground-truth layout;
* a connectivity-weighted pairwise distance loss — differences between the matrices of
  pairwise centroid distances, weighted by b2b edge weights, which teaches relative
  arrangement rather than absolute coordinates;
* a differentiable approximation of the contest cost (soft bounding box via log-sum-exp,
  softplus overlap penalty, out-of-bounds penalty, cluster and boundary terms), ramped
  in gradually after the first epochs.

Spectral embeddings and SPD matrices are precomputed offline, and the training set is
doubled by an x/y-transposed augmentation with the boundary flags remapped accordingly.

### 4. Inference

1. **Ensemble.** 5 checkpoints run on the same instance; their predictions plus the
   ensemble mean form K starting points. Optional Gaussian multi-start noise adds more.
2. **Gradient refinement.** All K variants are optimized jointly as one batch for up to
   600 Adam steps on a decaying cyclic learning rate. The objective is the fast,
   synchronization-free version of the contest cost, with the overlap term ramped up
   over the steps. Fixed and preplaced blocks are overwritten with their input values on
   every step, so the corresponding hard constraints cannot drift.
3. **Feasibility tracking.** The best state per variant is tracked entirely on the GPU
   (violations first, cost second), a variant is frozen as soon as it becomes feasible,
   and the loop exits early once all variants are feasible.

### 5. Legalization and selection

The refined layouts still contain small overlaps, so each candidate goes through:

* **push-out legalization** — overlapping pairs are separated along the axis of smaller
  penetration, which preserves the topology found by the network and barely affects
  HPWL and area;
* **teleport** — an emergency fallback that relocates the remaining violators by a
  spiral search, optionally reshaping them;
* **boundary-aware gravity** — blocks are pulled toward the origin and blocks with
  boundary constraints are snapped to their walls, which compacts the bounding box;
* **cluster repair** — greedy movement toward the cluster centroid and, if a cluster
  is still fragmented, analytic reshaping of soft blocks so the islands touch.

Every candidate is then scored with the official cost function, and the best one wins.
The sorting key is `overlaps * 1e6 + cost`, so a feasible solution always beats an
infeasible one and, among infeasible ones, the least broken is chosen.

## Data downloading

1. Download data from official repository: [train (~6.6 GB)](https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz?download=true) and [test (~69 MB)](https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorDataTest.tar.gz?download=true). And unzip data in folder `./FloorSet_data/`. Or use script:

```bash
python preproc_data/r01_download_contest_data.py
```

## Inference (contest version)

1. To run inference first you need to download pretrained checkpoints. Download manually [from HF](https://huggingface.co/datasets/ZFTurbo/ICCAD-Contest-2026-The-FloorSet-Challenge-Weights/tree/main) and put in `./weights/` folder or use script:

```bash
python preproc_data/r02_download_pretrained_weights.py
```

2. Download and put [FloorSet](https://github.com/IntelLabs/FloorSet/) in `./FloorSet/` folder.

3. Run contest inference:

```bash
python FloorSet/iccad2026contest/iccad2026_evaluate.py --evaluate op_src.py --save-solutions
```

Results should be around:
```
Total Score: 1.1674
Tests: 100
Feasible: 100
Avg Cost: 1.1915
Avg Runtime: 1.90s
```

## Inference local (better score)

1. To run inference first you need to download pretrained checkpoints. Download manually [from HF](https://huggingface.co/datasets/ZFTurbo/ICCAD-Contest-2026-The-FloorSet-Challenge-Weights/tree/main) and put in `./weights/` folder or use script:

```bash
python preproc_data/r02_download_pretrained_weights.py
```

2. Run inference:

```bash
python inference.py
```

**Note**: inference parameters are available in `config.py`.

```text
STEPS = 600 MAX_LR = 0.004
Total Score: 1.1003
Tests: 100
Feasible: 100
Avg Cost: 1.1090
Full validation time: 481.05 sec

STEPS = 800 MAX_LR = 0.003
Total Score: 1.0829
Tests: 100
Feasible: 100
Avg Cost: 1.1006
Full validation time: 560.16 sec
```


## Data preprocessing for training

1. **[Optional]** Create augmented data - it will increase train dataset 2 times:

```bash
python preproc_data/r03_create_augmented_train_data.py --source <path to train dataset> --output <path to augmented dataset>
```

2. Preprocess data for training (calculate additional variables):
```bash
python preproc_data/r04_preprocess_data_for_training.py --source <path to train dataset> --target <path to preproc dataset>
```
if you have augmented data you must run code for this data too.

## Training

```bash
python train.py
```

**Note**: Train parameters are available in `config.py`.

## Visualization

https://github.com/user-attachments/assets/eda25fcd-2741-46aa-8040-c1ca97974598

<video src="https://github.com/user-attachments/assets/8d048987-bcfe-45b1-81eb-8e16fdba2b22" width="1200" controls></video>

