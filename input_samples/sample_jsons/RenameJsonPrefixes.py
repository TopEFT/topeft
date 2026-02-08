#!/usr/bin/env python3
from pathlib import Path
import argparse
from typing import List, Optional, Tuple

def RemoveAnyPrefix(Stem: str, PrefixList: List[str]) -> Tuple[str, Optional[str]]:
    for Prefix in PrefixList:
        if Stem.startswith(Prefix):
            return Stem[len(Prefix):], Prefix
    return Stem, None

def RemoveAnySuffix(Stem: str, SuffixList: List[str]) -> Tuple[str, Optional[str]]:
    for Suffix in SuffixList:
        if Stem.endswith(Suffix):
            return Stem[:-len(Suffix)], Suffix
    return Stem, None

def CleanJsonFileName(FileName: str, PrefixList: List[str], SuffixList: List[str]) -> Tuple[str, bool]:
    PathObj = Path(FileName)
    Suffix = PathObj.suffix  # expect ".json"
    Stem = PathObj.stem

    Changed = False

    # Remove (possibly multiple) prefixes
    while True:
        NewStem, Matched = RemoveAnyPrefix(Stem, PrefixList)
        if Matched is None:
            break
        Stem = NewStem
        Changed = True

    # Remove (possibly multiple) suffixes
    while True:
        NewStem, Matched = RemoveAnySuffix(Stem, SuffixList)
        if Matched is None:
            break
        Stem = NewStem
        Changed = True

    NewName = Stem + Suffix
    return NewName, (Changed and NewName != FileName)

def RenameFiles(RootPath: Path, PrefixList: List[str], SuffixList: List[str], DryRun: bool) -> None:
    RenamedCount = 0
    SkippedCount = 0

    for FilePath in RootPath.rglob("*.json"):
        OriginalName = FilePath.name
        NewName, DidChange = CleanJsonFileName(OriginalName, PrefixList, SuffixList)
        if not DidChange:
            continue

        NewPath = FilePath.with_name(NewName)

        if NewPath.exists():
            print(f"SKIP (target exists): {FilePath} -> {NewPath}")
            SkippedCount += 1
            continue

        if DryRun:
            print(f"DRYRUN: {FilePath} -> {NewPath}")
        else:
            FilePath.rename(NewPath)
            print(f"RENAMED: {FilePath} -> {NewPath}")

        RenamedCount += 1

    print(f"\nDone. Renamed={RenamedCount}, Skipped={SkippedCount}")

def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(description="Clean year tags from json filenames (prefixes and suffixes).")
    Parser.add_argument("--RootPath", type=Path, default=Path("."), help="Root directory to scan (default: .)")
    Parser.add_argument("--Apply", action="store_true", help="Actually rename files (default is dry-run).")
    return Parser.parse_args()

def Main() -> None:
    Args = ParseArgs()

    # Longer first (important)
    PrefixList = ["2023BPix_", "2022EE_", "2023_", "2022_"]
    SuffixList = ["_2023BPix", "_2022EE", "_2023", "_2022"]

    DryRun = not Args.Apply
    RenameFiles(Args.RootPath, PrefixList, SuffixList, DryRun)

if __name__ == "__main__":
    Main()