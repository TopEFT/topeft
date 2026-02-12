#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replace_root_urls.py

Scans all .cfg files in a given directory (default: current working directory)
and replaces every occurrence of:
    root://hactar01.crc.nd.edu/
with:
    root://skynet013.crc.nd.edu/
"""

import argparse
import sys
from pathlib import Path


def replace_in_file(file_path: Path, old: str, new: str) -> None:
    """
    Read the contents of file_path, replace all occurrences of `old` with `new`,
    and write back the modified contents in place.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Skipping non-text file: {file_path}", file=sys.stderr)
        return

    if old not in text:
        # Nothing to replace
        return

    updated = text.replace(old, new)
    file_path.write_text(updated, encoding="utf-8")
    print(f"Updated: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace all occurrences of "
            "'root://hactar01.crc.nd.edu/' with 'root://skynet013.crc.nd.edu/' "
            "in every .cfg file under the specified directory."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Path to the directory containing .cfg files (default: current directory).",
    )
    args = parser.parse_args()
    base_dir = args.directory

    if not base_dir.is_dir():
        print(f"Error: {base_dir!s} is not a directory.", file=sys.stderr)
        sys.exit(1)

    old_url = "root://hactar01.crc.nd.edu/"
    new_url = "root://skynet013.crc.nd.edu/"

    # Iterate over all .cfg files in base_dir (non-recursive)
    for cfg_file in sorted(base_dir.glob("*.cfg")):
        replace_in_file(cfg_file, old_url, new_url)


if __name__ == "__main__":
    main()
