#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toggle_redirector.py

Scan all .cfg files in a given directory (default: current working directory)
and toggle between two redirector lines by commenting/uncommenting:

  root redirector:   root://skynet013.crc.nd.edu/
  file redirector:   file:///cms/cephfs/data/

Usage examples:

    # Inside your "cfgs" directory, toggle everything to use ROOT:
    ./toggle_redirector.py --use root

    # Or, specify a different directory:
    ./toggle_redirector.py --dir /path/to/cfgs --use file
"""

import argparse
import sys
from pathlib import Path
from typing import Literal
from topeft.modules.paths import topeft_path

# Constants: the two redirector substrings we care about
ROOT_REDIRECTOR = "root://skynet013.crc.nd.edu/"
FILE_REDIRECTOR = "file:///cms/cephfs/data/"


def comment_line(line: str) -> str:
    """
    Given a single line of text, ensure that the first non-whitespace character
    is '#' (i.e. the line is “commented out”). If it already starts with '#'
    (after indentation), leave it as-is.
    """
    # Find index of first non-whitespace character
    stripped = line.lstrip()
    if not stripped:
        # empty or all‐whitespace line → return as-is
        return line

    indent_len = len(line) - len(stripped)
    first_char = line[indent_len]

    if first_char == "#":
        # already commented
        return line
    else:
        # insert '#' at that position
        return line[:indent_len] + "#" + line[indent_len:]


def uncomment_line(line: str) -> str:
    """
    Given a single line of text, remove the first '#' if (and only if) it is
    the first non-whitespace character. Otherwise, return the line unchanged.
    """
    stripped = line.lstrip()
    if not stripped:
        return line

    indent_len = len(line) - len(stripped)
    first_char = line[indent_len]

    if first_char == "#":
        # drop that single '#'
        return line[:indent_len] + line[indent_len + 1 :]
    else:
        # already uncommented
        return line


def process_file(
    file_path: Path, use: Literal["root", "file"]
) -> bool:
    """
    Read all lines from file_path, toggle comment/uncomment on any line
    containing either ROOT_REDIRECTOR or FILE_REDIRECTOR, based on `use`.
    Returns True if the file was modified (and written back); False otherwise.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Skipping non-text file: {file_path}", file=sys.stderr)
        return False

    lines = text.splitlines(keepends=True)
    modified = False
    new_lines = []

    for line in lines:
        # We’ll check if the line contains either redirector substring.
        if ROOT_REDIRECTOR in line:
            if use == "root":
                # Ensure this line is uncommented
                new_line = uncomment_line(line)
            else:
                # use == "file": ensure this line is commented out
                new_line = comment_line(line)

            if new_line != line:
                modified = True
            new_lines.append(new_line)

        elif FILE_REDIRECTOR in line:
            if use == "file":
                # Ensure this line is uncommented
                new_line = uncomment_line(line)
            else:
                # use == "root": ensure this line is commented out
                new_line = comment_line(line)

            if new_line != line:
                modified = True
            new_lines.append(new_line)

        else:
            # Lines that don’t mention either redirector are left untouched
            new_lines.append(line)

    if modified:
        # Write back the entire file in one go
        file_path.write_text("".join(new_lines), encoding="utf-8")
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Toggle between 'root' and 'file' redirectors in all .cfg files "
            "by commenting/uncommenting the appropriate lines."
        )
    )
    parser.add_argument(
        "--use",
        "-u",
        choices=["root", "file"],
        required=True,
        help=(
            "Which redirector to activate for every .cfg: "
            "'root' → uncomment lines containing root://… and comment file://…  "
            "'file' → uncomment lines containing file://… and comment root://…"
        ),
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=topeft_path("input_samples/cfgs"),
        help="Directory containing all .cfg files (default: current working directory).",
    )
    args = parser.parse_args()
    base_dir = args.dir
    choice = args.use

    if not base_dir.is_dir():
        base_dir_orig = args.dir
        base_dir = Path(str(base_dir).replace("topeft/topeft", "topeft"))
        if not base_dir.is_dir():
            print(f"Error: neither '{base_dir_orig}' nor '{base_dir}' are a directory.", file=sys.stderr)
            sys.exit(1)

    cfg_files = sorted(base_dir.glob("*.cfg"))
    if not cfg_files:
        print(f"No .cfg files found in '{base_dir}'.", file=sys.stderr)
        sys.exit(0)

    any_updated = False
    for cfg in cfg_files:
        if process_file(cfg, choice):
            print(f"Updated: {cfg.name}")
            any_updated = True

    if not any_updated:
        print("No files needed changes. All .cfg redirectors are already set.")


if __name__ == "__main__":
    main()
