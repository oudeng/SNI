#!/usr/bin/env python3
"""Create a clean release copy of the repository, excluding build artifacts.

Copies the project tree to a new directory or zip file, excluding:
  - .git/
  - __MACOSX/
  - .DS_Store
  - __pycache__/
  - *.pyc
  - .claude/
  - *_old*
  - results*/

Usage:
    # Copy to a directory
    python scripts/clean_release.py --dest ../SNI_release

    # Create a zip file
    python scripts/clean_release.py --zip ../SNI_release.zip

    # From a custom source
    python scripts/clean_release.py --src /path/to/project --zip release.zip
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import zipfile
from pathlib import Path
from typing import List


# Patterns to exclude (checked against relative paths)
EXCLUDE_DIRS = {
    ".git",
    "__MACOSX",
    "__pycache__",
    ".claude",
}

EXCLUDE_DIR_PATTERNS = [
    "results*",
    "*_old*",
]

EXCLUDE_FILES = {
    ".DS_Store",
}

EXCLUDE_FILE_PATTERNS = [
    "*.pyc",
    "*_old*",
]


def _should_exclude_dir(name: str) -> bool:
    """Check if a directory name should be excluded."""
    if name in EXCLUDE_DIRS:
        return True
    for pat in EXCLUDE_DIR_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _should_exclude_file(name: str) -> bool:
    """Check if a file name should be excluded."""
    if name in EXCLUDE_FILES:
        return True
    for pat in EXCLUDE_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _collect_files(src: Path) -> List[Path]:
    """Collect all files from src that should be included in the release.

    Returns a list of paths relative to src.
    """
    included: List[Path] = []

    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)

        # Check if any parent directory should be excluded
        skip = False
        for part in rel.parts[:-1] if item.is_file() else rel.parts:
            if _should_exclude_dir(part):
                skip = True
                break
        if skip:
            continue

        if item.is_dir():
            # Check directory itself
            if _should_exclude_dir(item.name):
                continue
            # Directories are implicitly created when copying files
            continue

        if item.is_file():
            if _should_exclude_file(item.name):
                continue
            included.append(rel)

    return included


def copy_to_dest(src: Path, dest: Path) -> int:
    """Copy the clean tree from src to dest.

    Returns the number of files copied.
    """
    if dest.exists():
        print(f"[WARN] Destination already exists: {dest}")
        print(f"       Files will be merged/overwritten.")

    files = _collect_files(src)
    count = 0

    for rel in files:
        src_file = src / rel
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        count += 1

    return count


def create_zip(src: Path, zip_path: Path) -> int:
    """Create a zip archive of the clean tree.

    Returns the number of files added.
    """
    files = _collect_files(src)
    count = 0

    # Use the zip file's top-level directory name from the zip filename
    # e.g., SNI_release.zip -> files stored as SNI_release/...
    top_dir = zip_path.stem

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in files:
            src_file = src / rel
            arcname = f"{top_dir}/{rel}"
            zf.write(src_file, arcname)
            count += 1

    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a clean release copy of the project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src",
        type=str,
        default=".",
        help="Source directory (default: current directory).",
    )

    output = ap.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory for the clean copy.",
    )
    output.add_argument(
        "--zip",
        type=str,
        default=None,
        help="Path for the output zip file.",
    )

    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"[INFO] Source: {src}")
    print(f"[INFO] Excluding directories matching: {EXCLUDE_DIRS | set(EXCLUDE_DIR_PATTERNS)}")
    print(f"[INFO] Excluding files matching: {EXCLUDE_FILES | set(EXCLUDE_FILE_PATTERNS)}")

    # Preview
    files = _collect_files(src)
    print(f"[INFO] {len(files)} files to include")

    if args.dest:
        dest = Path(args.dest).resolve()
        print(f"[INFO] Copying to: {dest}")
        count = copy_to_dest(src, dest)
        print(f"[DONE] Copied {count} files to {dest}")
    elif args.zip:
        zip_path = Path(args.zip).resolve()
        print(f"[INFO] Creating zip: {zip_path}")
        count = create_zip(src, zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[DONE] Created {zip_path} ({count} files, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
