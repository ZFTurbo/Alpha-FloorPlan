import os
import glob
import argparse
import multiprocessing
import torch
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# The computation functions are copied from the training code unchanged (they run on CPU)
# =============================================================================
def compute_spectral_embeddings(b2b_conn, num_nodes, k_dim=3):
    if num_nodes == 0:
        return torch.zeros((0, k_dim), dtype=torch.float32)

    W = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if b2b_conn.numel() > 0:
        u = b2b_conn[:, 0].long()
        v = b2b_conn[:, 1].long()
        weights = b2b_conn[:, 2].float()
        W[u, v] = weights
        W[v, u] = weights

    degrees = W.sum(dim=1)
    D = torch.diag(degrees)
    L = D - W
    L += torch.eye(num_nodes) * 1e-6

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    start_idx = 1
    end_idx = min(start_idx + k_dim, num_nodes)
    selected_vecs = eigenvectors[:, start_idx:end_idx]

    if selected_vecs.shape[1] < k_dim:
        pad_len = k_dim - selected_vecs.shape[1]
        selected_vecs = F.pad(selected_vecs, (0, pad_len), value=0.0)

    mean = selected_vecs.mean(dim=0, keepdim=True)
    std = selected_vecs.std(dim=0, keepdim=True) + 1e-8
    normalized_vecs = (selected_vecs - mean) / std

    return normalized_vecs


def compute_spd_matrix_hops(b2b_conn, num_nodes, max_dist=20):
    if num_nodes == 0:
        return torch.zeros((0, 0), dtype=torch.long)

    spd = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.long)
    spd.fill_diagonal_(0)

    if b2b_conn.numel() > 0:
        u = b2b_conn[:, 0].long()
        v = b2b_conn[:, 1].long()
        spd[u, v] = 1
        spd[v, u] = 1

    for k in range(num_nodes):
        dist_through_k = spd[:, k].unsqueeze(1) + spd[k, :].unsqueeze(0)
        spd = torch.min(spd, dist_through_k)

    spd = torch.clamp(spd, max=max_dist)
    return spd


# =============================================================================
# File processing logic
# =============================================================================
def process_single_file(args):
    src_path, dst_path = args

    # If the file already exists, skip it (convenient for resuming after an interruption)
    if os.path.exists(dst_path):
        return True

    # Create the required subfolder (for example, worker_0)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Load the original file
    file_contents = torch.load(src_path, weights_only=False)

    # Convert to a list so that new elements can be appended easily
    file_contents = list(file_contents)

    layouts_per_file = len(file_contents[0])

    all_spectral_embs = []
    all_spd_matrices = []

    for layout_idx in range(layouts_per_file):
        # Extract the data needed to determine the number of nodes
        area_target = file_contents[0][layout_idx][:, 0]
        block_count = int((area_target != -1).sum().item())
        b2b_conn = file_contents[1][layout_idx]

        # Compute the heavy matrices
        spectral_emb = compute_spectral_embeddings(b2b_conn, block_count, k_dim=3)
        spd_matrix = compute_spd_matrix_hops(b2b_conn, block_count, max_dist=20)

        all_spectral_embs.append(spectral_emb)
        all_spd_matrices.append(spd_matrix)

    # Append them under indices 7 and 8
    file_contents.append(all_spectral_embs)
    file_contents.append(all_spd_matrices)

    # Save into the new directory.
    # Write to a temporary name first and rename afterwards, so that an
    # interrupted run never leaves a truncated file that looks complete.
    tmp_path = dst_path + ".tmp"
    torch.save(file_contents, tmp_path)
    os.replace(tmp_path, dst_path)
    return True


def main(source_dir, target_dir, num_workers, pattern):
    print(f"Looking for {pattern} files in {source_dir}...")

    # Search recursively (this captures worker_X/layouts_Y.th)
    search_pattern = source_dir + "/**/" + pattern
    print('Search pattern: {}'.format(search_pattern))
    src_files = glob.glob(search_pattern, recursive=True)

    if not src_files:
        print("No files found! Check the paths.")
        return

    print(f"Files found for processing: {len(src_files)}")

    # Build the task list (src_path, dst_path)
    tasks = []
    for src_path in src_files:
        # Compute the relative path to preserve the folder structure
        rel_path = os.path.relpath(src_path, source_dir)
        dst_path = os.path.join(target_dir, rel_path)
        tasks.append((src_path, dst_path))

    # Start multiprocessing
    print(f"Starting the pool with {num_workers} processes...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # tqdm shows a nice progress bar
        results = list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks)))

    success_count = sum(1 for r in results if r)
    print(f"Finished! Successfully processed files: {success_count} out of {len(tasks)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute spectral embeddings and SPD matrices for the floorset dataset."
    )
    parser.add_argument(
        "--source",
        default="../FloorSet_data/floorset_lite",
        help="Directory holding the source .th files.",
    )
    parser.add_argument(
        "--target",
        default="../FloorSet_data/floorset_lite_precomputed",
        help="Directory the precomputed files are written to (folder structure is preserved).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        # default=max(1, multiprocessing.cpu_count() // 4),
        default=1,
        help="Number of worker processes (defaults to a quarter of the CPU cores).",
    )
    parser.add_argument(
        "--pattern",
        default="*.th",
        help="Filename pattern to search for, e.g. 'layouts*.th'.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()

    args = parse_args()
    main(args.source, args.target, args.workers, args.pattern)