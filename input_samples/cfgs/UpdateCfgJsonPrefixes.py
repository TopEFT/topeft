#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
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

    while True:
        NewStem, Matched = RemoveAnyPrefix(Stem, PrefixList)
        if Matched is None:
            break
        Stem = NewStem
        Changed = True

    while True:
        NewStem, Matched = RemoveAnySuffix(Stem, SuffixList)
        if Matched is None:
            break
        Stem = NewStem
        Changed = True

    NewName = Stem + Suffix
    return NewName, (Changed and NewName != FileName)

def RewritePathBasename(PathText: str, PrefixList: List[str], SuffixList: List[str]) -> Tuple[str, bool]:
    PathObj = Path(PathText)
    OldName = PathObj.name
    NewName, DidChange = CleanJsonFileName(OldName, PrefixList, SuffixList)
    if not DidChange:
        return PathText, False
    NewText = str(PathObj.with_name(NewName))
    return NewText, (NewText != PathText)

def RewriteCfgContent(Content: str, PrefixList: List[str], SuffixList: List[str]) -> Tuple[str, int]:
    Lines = Content.splitlines(True)
    ChangedCount = 0
    OutLines: List[str] = []

    for Line in Lines:
        OriginalLine = Line
        NewLine = Line

        HasNewline = NewLine.endswith("\n")
        BaseNoNl = NewLine[:-1] if HasNewline else NewLine

        LeadingSpacesLen = len(BaseNoNl) - len(BaseNoNl.lstrip(" "))
        LeadingSpaces = BaseNoNl[:LeadingSpacesLen]
        AfterLeading = BaseNoNl[LeadingSpacesLen:]

        if not AfterLeading.strip():
            OutLines.append(Line)
            continue

        if AfterLeading.startswith("#"):
            AfterHash = AfterLeading[1:]
            HashSpacesLen = len(AfterHash) - len(AfterHash.lstrip(" "))
            HashSpaces = AfterHash[:HashSpacesLen]
            CommentContent = AfterHash[HashSpacesLen:]

            TrailingSpaces = CommentContent[len(CommentContent.rstrip(" ")):]
            PathCandidate = CommentContent.rstrip(" ")

            if PathCandidate.endswith(".json"):
                Rewritten, DidChange = RewritePathBasename(PathCandidate, PrefixList, SuffixList)
                if DidChange:
                    BaseNoNl = LeadingSpaces + "#" + HashSpaces + Rewritten + TrailingSpaces
                    NewLine = BaseNoNl + ("\n" if HasNewline else "")
        else:
            TrailingSpaces = AfterLeading[len(AfterLeading.rstrip(" ")):]
            PathCandidate = AfterLeading.rstrip(" ")

            if PathCandidate.endswith(".json"):
                Rewritten, DidChange = RewritePathBasename(PathCandidate, PrefixList, SuffixList)
                if DidChange:
                    BaseNoNl = LeadingSpaces + Rewritten + TrailingSpaces
                    NewLine = BaseNoNl + ("\n" if HasNewline else "")

        OutLines.append(NewLine)
        if NewLine != OriginalLine:
            ChangedCount += 1

    return "".join(OutLines), ChangedCount

def ExtractJsonReferences(Content: str) -> List[Tuple[int, bool, str]]:
    Refs: List[Tuple[int, bool, str]] = []
    Lines = Content.splitlines(False)

    for Index, Line in enumerate(Lines, start=1):
        Stripped = Line.strip()
        if not Stripped:
            continue

        IsCommented = False
        Working = Line.lstrip(" ")

        if Working.startswith("#"):
            IsCommented = True
            Working = Working[1:].lstrip(" ")

        if not Working.endswith(".json"):
            continue

        if "://" in Working and not Working.startswith("file://"):
            continue

        if Working.startswith("file://"):
            Working = Working[len("file://"):]

        Refs.append((Index, IsCommented, Working))

    return Refs

def ResolveRefPath(CfgPath: Path, RefText: str) -> Path:
    RefPath = Path(RefText)
    if RefPath.is_absolute():
        return RefPath
    Candidate = (CfgPath.parent / RefPath)
    try:
        return Candidate.resolve()
    except Exception:
        return Candidate.absolute()

def VerifyJsonReferences(CfgPath: Path, ContentToCheck: str) -> Tuple[int, int, int]:
    Refs = ExtractJsonReferences(ContentToCheck)
    MissingActive = 0
    MissingCommented = 0

    for LineNo, IsCommented, RefText in Refs:
        AbsPath = ResolveRefPath(CfgPath, RefText)
        Exists = AbsPath.exists()

        if not Exists:
            Tag = "WARN" if IsCommented else "ERROR"
            Kind = "COMMENTED" if IsCommented else "ACTIVE"
            print(f"{Tag}: missing {Kind} ref in {CfgPath} @ line {LineNo}: {RefText}")
            print(f"      -> {AbsPath}")
            if IsCommented:
                MissingCommented += 1
            else:
                MissingActive += 1

    return len(Refs), MissingActive, MissingCommented

def ShouldProcessCfg(CfgPath: Path, CfgPrefixes: List[str]) -> bool:
    if not CfgPrefixes:
        return True
    Name = CfgPath.name
    for Prefix in CfgPrefixes:
        if Name.startswith(Prefix):
            return True
    return False

def ProcessCfgFiles(RootPath: Path, PrefixList: List[str], SuffixList: List[str], DryRun: bool, BackupExt: str, CfgPrefixes: List[str]) -> int:
    TotalFilesChanged = 0
    TotalLinesChanged = 0

    TotalRefsChecked = 0
    TotalMissingActive = 0
    TotalMissingCommented = 0

    ConsideredCfgs = 0

    for CfgPath in sorted(RootPath.rglob("*.cfg")):
        if not ShouldProcessCfg(CfgPath, CfgPrefixes):
            continue

        ConsideredCfgs += 1
        Content = CfgPath.read_text(encoding="utf-8", errors="replace")
        NewContent, ChangedLines = RewriteCfgContent(Content, PrefixList, SuffixList)

        # Always verify every referenced json (post-rewrite), even if no changes
        RefsChecked, MissingActive, MissingCommented = VerifyJsonReferences(CfgPath, NewContent)
        TotalRefsChecked += RefsChecked
        TotalMissingActive += MissingActive
        TotalMissingCommented += MissingCommented

        if ChangedLines == 0:
            continue

        TotalFilesChanged += 1
        TotalLinesChanged += ChangedLines

        if DryRun:
            print(f"DRYRUN: {CfgPath} (would change {ChangedLines} line(s))")
        else:
            if BackupExt:
                BackupPath = CfgPath.with_suffix(CfgPath.suffix + BackupExt)
                BackupPath.write_text(Content, encoding="utf-8")
            CfgPath.write_text(NewContent, encoding="utf-8")
            print(f"UPDATED: {CfgPath} (changed {ChangedLines} line(s))")

    print(
        f"\nDone."
        f" ConsideredCfgs={ConsideredCfgs}."
        f" FilesChanged={TotalFilesChanged}, LinesChanged={TotalLinesChanged}."
        f" JsonRefsChecked={TotalRefsChecked}, MissingActive={TotalMissingActive}, MissingCommented={TotalMissingCommented}"
    )

    return 2 if TotalMissingActive > 0 else 0

def ParseArgs() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(
        description="Update .cfg files to match cleaned json names (prefix+suffix) and optionally verify only cfgs matching given prefixes."
    )
    Parser.add_argument("--RootPath", type=Path, default=Path("."), help="Root directory to scan for *.cfg (default: .)")
    Parser.add_argument("--Apply", action="store_true", help="Actually write changes (default is dry-run).")
    Parser.add_argument("--BackupExt", default=".bak", help="Backup extension when applying (default: .bak). Use '' to disable.")
    Parser.add_argument(
        "--CfgPrefix",
        action="append",
        default=[],
        help="If set, process only cfg files whose basename starts with this prefix. Can be repeated."
    )
    return Parser.parse_args()

def Main() -> None:
    Args = ParseArgs()

    PrefixList = ["2023BPix_", "2022EE_", "2023_", "2022_"]  # longer first
    SuffixList = ["_2023BPix", "_2022EE", "_2023", "_2022"]  # longer first

    DryRun = not Args.Apply
    BackupExt = Args.BackupExt if Args.Apply else ""

    ExitCode = ProcessCfgFiles(
        Args.RootPath,
        PrefixList,
        SuffixList,
        DryRun,
        BackupExt,
        Args.CfgPrefix
    )
    sys.exit(ExitCode)

if __name__ == "__main__":
    Main()
