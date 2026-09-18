import os
import glob
import argparse
import torch
import multiprocessing
from collections import Counter
from tqdm import tqdm

# Transformation map for the Boundary flags under the x <-> y reflection.
# Original -> Transposed
BOUNDARY_MAP = {
    0: 0,
    1: 8,  # Left (1) -> Bottom (8)
    2: 4,  # Right (2) -> Top (4)
    4: 2,  # Top (4) -> Right (2)
    8: 1,  # Bottom (8) -> Left (1)
    3: 12,  # Left-Right (3) -> Bottom-Top (12)
    12: 3,  # Top-Bottom (12) -> Right-Left (3)
    5: 10,  # Top-Left (5) -> Bottom-Right (10)
    6: 6,  # Top-Right (6) -> Top-Right (6) - invariant
    9: 9,  # Bottom-Left (9) -> Bottom-Left (9) - invariant
    10: 5  # Bottom-Right (10) -> Top-Left (5)
}


def remap_boundaries(boundary_tensor):
    """
    Applies the mapping to a tensor of Boundary flags.

    Returns the remapped tensor and a list of codes that are absent from
    BOUNDARY_MAP. Such codes would silently become 0 (no constraint), so the
    caller is expected to report them.
    """
    mapped = torch.zeros_like(boundary_tensor)
    known = torch.zeros_like(boundary_tensor, dtype=torch.bool)

    for old_val, new_val in BOUNDARY_MAP.items():
        match = boundary_tensor == old_val
        mapped[match] = new_val
        known |= match

    unmapped = []
    if not bool(known.all()):
        unmapped = boundary_tensor[~known].unique().tolist()

    return mapped, unmapped


def process_file(task):
    input_path, output_path = task

    # Load the original tensor
    try:
        file_contents = torch.load(input_path, weights_only=False)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return input_path, []

    num_layouts = len(file_contents[0])
    print("0", file_contents[0].cpu().numpy())
    print("1", file_contents[1].cpu().numpy())
    print("2", file_contents[2].cpu().numpy())
    print("3", file_contents[3].cpu().numpy())
    print("4", file_contents[4].cpu().numpy())
    print("5", file_contents[5].cpu().numpy())
    print("6", file_contents[6].cpu().numpy())

    new_file_contents = [[] for _ in range(7)]
    unmapped_codes = []

    for layout_idx in range(num_layouts):
        # 0: area_target and placement_constraints
        feat = file_contents[0][layout_idx].clone()
        # Index 5 corresponds to the Boundary flag (0: area, 1: fixed, 2: preplaced, 3: mib, 4: cluster, 5: boundary)
        feat[:, 5], layout_unmapped = remap_boundaries(feat[:, 5])
        unmapped_codes.extend(layout_unmapped)
        new_file_contents[0].append(feat)

        # 1: b2b_connectivity (unchanged)
        new_file_contents[1].append(file_contents[1][layout_idx].clone())

        # 2: p2b_connectivity (unchanged)
        new_file_contents[2].append(file_contents[2][layout_idx].clone())

        # 3: pins_pos (x <-> y)
        pins = file_contents[3][layout_idx].clone()
        if pins.numel() > 0:
            pins = pins[:, [1, 0]]  # Swap x and y
        new_file_contents[3].append(pins)

        # 4: tree_sol (unchanged)
        new_file_contents[4].append(file_contents[4][layout_idx].clone())

        # 5: fp_sol (w, h, x, y) -> (h, w, y, x)
        fp = file_contents[5][layout_idx].clone()
        if fp.numel() > 0:
            fp = fp[:, [1, 0, 3, 2]]  # Swap width/height and coordinates
        new_file_contents[5].append(fp)

        # 6: metrics_sol (unchanged)
        new_file_contents[6].append(file_contents[6][layout_idx].clone())

    # Save the augmented file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_file_contents, output_path)

    return input_path, unmapped_codes


def build_dataset_augmented(input_dir, output_dir, num_workers=12):
    """
    Scans the directory for dataset files and generates mirrored copies in parallel.
    """
    print(f"Scanning directory: {input_dir}")

    # Find all files matching layouts*.th in every worker_ subfolder
    search_pattern = os.path.join(input_dir, 'floorset_lite', '**', 'layouts*.th')
    print(search_pattern)
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        print("No files found. Check the path.")
        return

    print(f"Files found for augmentation: {len(all_files)}")

    tasks = []
    for input_path in all_files:
        # Compute the relative path to preserve the folder structure (worker_X/...)
        rel_path = os.path.relpath(input_path, input_dir)

        # Rename the file so it is clear that it has been augmented
        dir_name, file_name = os.path.split(rel_path)
        new_file_name = file_name.replace('.th', '_aug_xy.th')
        output_path = os.path.join(output_dir, dir_name, new_file_name)

        tasks.append((input_path, output_path))

    # Start multiprocessing
    print(f"Starting the worker pool ({num_workers} threads)...")

    unmapped_counter = Counter()
    files_with_unmapped = []

    with multiprocessing.Pool(num_workers) as pool:
        results = tqdm(
            pool.imap_unordered(process_file, tasks),
            total=len(tasks),
            desc="Augmentation"
        )
        for input_path, unmapped_codes in results:
            if unmapped_codes:
                unmapped_counter.update(unmapped_codes)
                files_with_unmapped.append(input_path)

    # Report boundary codes that are missing from BOUNDARY_MAP.
    # They were written out as 0, i.e. the constraint was dropped.
    if unmapped_counter:
        print("\n[WARNING] Boundary codes missing from BOUNDARY_MAP were reset to 0:")
        for code, count in sorted(unmapped_counter.items()):
            print(f"  code {code}: seen in {count} layout(s)")
        print(f"  affected files: {len(files_with_unmapped)}")
        for path in files_with_unmapped[:10]:
            print(f"    {path}")
        if len(files_with_unmapped) > 10:
            print(f"    ... and {len(files_with_unmapped) - 10} more")
    else:
        print("\n[OK] All boundary codes were covered by BOUNDARY_MAP.")

    print("X<->Y dataset generation finished!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an x<->y mirrored copy of the floorset dataset."
    )
    parser.add_argument(
        "--source",
        default="../FloorSet_data/",
        help="Root directory holding the floorset_lite folder.",
    )
    parser.add_argument(
        "--output",
        default="../FloorSet_data/floorset_lite_augm/",
        help="Directory the augmented files are written to.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    build_dataset_augmented(args.source, args.output, num_workers=args.workers)