from __future__ import annotations

from pathlib import Path
import re
import subprocess


CFG_ROOT = Path(__file__).parents[1] / "input_samples" / "cfgs"
EXPECTED_REDIRECTOR = "root://cmsxrootd.crc.nd.edu/"
XROOTD_REDIRECTOR_PATTERN = re.compile(r"root://[A-Za-z0-9.-]+/")


def test_nd_sample_cfg_redirectors_use_the_crc_redirector():
    repository_root = CFG_ROOT.parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "--", "input_samples/cfgs"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    cfg_paths = sorted(
        repository_root / relative_path
        for relative_path in tracked
        if relative_path.endswith(".cfg")
    )
    assert cfg_paths
    occurrences = []
    offenders = []
    for cfg_path in cfg_paths:
        for line_number, line in enumerate(
            cfg_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in XROOTD_REDIRECTOR_PATTERN.finditer(line):
                token = match.group(0)
                occurrences.append((cfg_path, line_number, token))
                if token != EXPECTED_REDIRECTOR:
                    offenders.append((cfg_path, line_number, token))

    assert occurrences, "No maintained cfg XRootD redirectors were found"
    assert not offenders, "Unexpected ND cfg XRootD redirectors: " + "; ".join(
        f"{path.relative_to(CFG_ROOT)}:{line_number}: {token}"
        for path, line_number, token in offenders
    )
