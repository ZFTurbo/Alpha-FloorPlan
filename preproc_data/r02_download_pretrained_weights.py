"""
Download the model weights used by the inference script.

The script lives in preproc_data/ and by default writes to ../weights/,
i.e. a sibling folder of preproc_data.

Usage:
    python download_weights.py
    python download_weights.py --dest /data/weights
    python download_weights.py --verify
"""

import os
import sys
import time
import shutil
import argparse
import urllib.error
import urllib.request

from tqdm import tqdm

# Hugging Face repository holding the checkpoints
REPO_BASE = (
    "https://huggingface.co/datasets/ZFTurbo/"
    "ICCAD-Contest-2026-The-FloorSet-Challenge-Weights/resolve/main/"
)

WEIGHTS = [
    "direct_transformer_loss_5.6719_epoch_364.pt",
    "direct_transformer_loss_7.7646_epoch_418.pt",
    "direct_transformer_loss_7.7672_epoch_420.pt",
    "direct_transformer_loss_7.9332_epoch_414.pt",
    "direct_transformer_loss_7.9808_epoch_413.pt",
]

CHUNK_SIZE = 1024 * 1024  # 1 MB
USER_AGENT = "floorset-weights-downloader/1.0"


def human_size(num_bytes):
    """Format a byte count for logging."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def get_remote_size(url):
    """Return the size the server reports for the file, or None if unknown."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def download(url, dst_path, retries=3):
    """
    Download url into dst_path with a progress bar.

    Supports resuming: if a partial file is already present, the download
    continues from where it stopped via a Range request. A file whose size
    already matches the server is left untouched.
    """
    remote_size = get_remote_size(url)

    if os.path.exists(dst_path):
        local_size = os.path.getsize(dst_path)
        if remote_size is not None and local_size == remote_size:
            print(f"  Already downloaded ({human_size(local_size)}), skipping.")
            return True
        if remote_size is not None and local_size > remote_size:
            print("  Local file is larger than the remote one, re-downloading from scratch.")
            os.remove(dst_path)

    for attempt in range(1, retries + 1):
        resume_from = os.path.getsize(dst_path) if os.path.exists(dst_path) else 0

        headers = {"User-Agent": USER_AGENT}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            print(f"  Resuming from {human_size(resume_from)}...")

        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request) as response:
                # If the server ignored the Range header, start over
                if resume_from > 0 and response.status != 206:
                    print("  The server ignored the resume request, starting over.")
                    resume_from = 0

                total = remote_size
                if total is None:
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        total = int(length) + resume_from

                mode = "ab" if resume_from > 0 else "wb"
                with open(dst_path, mode) as out_file, tqdm(
                    total=total,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="  downloading",
                ) as bar:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        bar.update(len(chunk))

            final_size = os.path.getsize(dst_path)
            if remote_size is not None and final_size != remote_size:
                raise IOError(
                    f"size mismatch: got {final_size} bytes, expected {remote_size}"
                )

            return True

        except (urllib.error.URLError, IOError, TimeoutError) as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                delay = 5 * attempt
                print(f"  Retrying in {delay} s...")
                time.sleep(delay)
            else:
                print("  Giving up on this file.")
                return False

    return False


def verify_checkpoint(path):
    """
    Try to load the checkpoint to make sure the file is not truncated.

    torch is imported lazily so that the script stays usable without it
    when --verify is not requested.
    """
    try:
        import torch
    except ImportError:
        print("  torch is not installed, skipping verification.")
        return True

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  Verification FAILED: {e}")
        return False

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if epoch is not None:
        print(f"  Verified, epoch {epoch}.")
    else:
        print("  Verified.")
    return True


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dest = os.path.normpath(os.path.join(script_dir, "..", "weights"))

    parser = argparse.ArgumentParser(
        description="Download the model checkpoints used for inference."
    )
    parser.add_argument(
        "--dest",
        default=default_dest,
        help="Destination directory (default: ../weights relative to this script).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load each checkpoint with torch after downloading to confirm it is intact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    free_space = shutil.disk_usage(dest_dir).free
    print(f"Destination: {dest_dir}")
    print(f"Free space:  {human_size(free_space)}")
    print(f"Checkpoints: {len(WEIGHTS)}\n")

    failed = []

    for idx, file_name in enumerate(WEIGHTS, start=1):
        url = REPO_BASE + file_name + "?download=true"
        dst_path = os.path.join(dest_dir, file_name)

        print(f"[{idx}/{len(WEIGHTS)}] {file_name}")

        if not download(url, dst_path):
            failed.append(file_name)
            continue

        if args.verify and not verify_checkpoint(dst_path):
            failed.append(file_name)
            continue

        print()

    if failed:
        print(f"Finished with errors. Failed files: {len(failed)}")
        for file_name in failed:
            print(f"  {file_name}")
        return 1

    print("All done.")
    print(f"Weights are in: {dest_dir}")
    return 0


if __name__ == "__main__":
    main()