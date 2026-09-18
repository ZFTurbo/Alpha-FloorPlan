"""
Download the FloorSet Lite dataset and extract it next to the project.

The script lives in preproc_data/ and by default writes to ../FloorSet/,
i.e. a sibling folder of preproc_data.

Usage:
    python download_floorset.py
    python download_floorset.py --dest /data/FloorSet_data --keep-archives
    python download_floorset.py --only test
"""

import os
import sys
import time
import shutil
import tarfile
import argparse
import urllib.error
import urllib.request

from tqdm import tqdm

# Official FloorSet repository on Hugging Face
DATASETS = {
    "train": {
        # "url": "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData.tar.gz?download=true",
        "url": "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz?download=true",
        "archive": "LiteTensorData.tar.gz",
        "approx_size": "~6.6 GB",
    },
    "test": {
        "url": "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorDataTest.tar.gz?download=true",
        "archive": "LiteTensorDataTest.tar.gz",
        "approx_size": "~69 MB",
    },
}

CHUNK_SIZE = 1024 * 1024  # 1 MB
USER_AGENT = "floorset-downloader/1.0"


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
            print(f"  Local file is larger than the remote one, re-downloading from scratch.")
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


def extract(archive_path, dest_dir):
    """Extract a .tar.gz archive into dest_dir."""
    print(f"  Extracting into {dest_dir} ...")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        # filter='data' (Python 3.12+) blocks absolute paths and links pointing
        # outside the destination; fall back silently on older interpreters.
        extract_kwargs = {}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"

        for member in tqdm(members, desc="  extracting", unit="file"):
            tar.extract(member, path=dest_dir, **extract_kwargs)

    print(f"  Done: {len(members)} entries.")


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dest = os.path.normpath(os.path.join(script_dir, "..", "FloorSet_data"))

    parser = argparse.ArgumentParser(
        description="Download and extract the FloorSet Lite dataset."
    )
    parser.add_argument(
        "--dest",
        default=default_dest,
        help="Destination directory (default: ../FloorSet_data relative to this script).",
    )
    parser.add_argument(
        "--only",
        choices=sorted(DATASETS.keys()),
        default=None,
        help="Fetch only one split instead of both.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the .tar.gz files after extraction (they are deleted by default).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download the archives, do not extract them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dest_dir = os.path.abspath(args.dest)
    archive_dir = os.path.join(dest_dir, "_archives")
    os.makedirs(archive_dir, exist_ok=True)

    splits = [args.only] if args.only else list(DATASETS.keys())

    free_space = shutil.disk_usage(dest_dir).free
    print(f"Destination: {dest_dir}")
    print(f"Free space:  {human_size(free_space)}")
    print(f"Splits:      {', '.join(splits)}\n")

    failed = []

    for split in splits:
        info = DATASETS[split]
        archive_path = os.path.join(archive_dir, info["archive"])

        print(f"[{split}] {info['archive']} ({info['approx_size']})")

        if not download(info["url"], archive_path):
            failed.append(split)
            continue

        if args.no_extract:
            print(f"  Archive saved to {archive_path}")
            continue

        try:
            extract(archive_path, dest_dir)
        except (tarfile.TarError, OSError) as e:
            print(f"  Extraction failed: {e}")
            print("  The archive may be corrupted - delete it and run the script again.")
            failed.append(split)
            continue

        if not args.keep_archives:
            os.remove(archive_path)
            print("  Archive removed.")

        print()

    if not args.keep_archives and not args.no_extract and not os.listdir(archive_dir):
        os.rmdir(archive_dir)

    if failed:
        print(f"Finished with errors. Failed splits: {', '.join(failed)}")
        return 1

    print("All done.")
    print(f"Data is in: {dest_dir}")
    return 0


if __name__ == "__main__":
    main()