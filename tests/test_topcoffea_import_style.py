"""Ensure top-level topcoffea imports follow the namespace style."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BANNED_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*from\s+topcoffea\."), "use 'import topcoffea' and attribute access"),
    (re.compile(r"^\s*import\s+topcoffea\.modules"), "import the top-level package instead of submodules"),
    (
        re.compile(r"importlib\.import_module\(\s*[\"']topcoffea\."),
        "load through 'topcoffea.import_module' and attribute access",
    ),
)


def _iter_source_files() -> Iterator[Path]:
    for pattern in ("*.py", "*.ipynb"):
        yield from _REPO_ROOT.rglob(pattern)


def _is_vendor_file(path: Path) -> bool:
    relative = path.relative_to(_REPO_ROOT)
    return relative.parts and relative.parts[0] == "topcoffea"


def _scan_text_lines(path: Path, lines: Iterable[str]) -> List[str]:
    violations: List[str] = []
    for lineno, line in enumerate(lines, 1):
        for pattern, guidance in _BANNED_PATTERNS:
            if pattern.search(line):
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)}:{lineno}: {guidance} -> {line.rstrip()}"
                )
    return violations


def _scan_ipynb(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    lines: List[str] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines.extend(cell.get("source", []))
    return _scan_text_lines(path, lines)


def _scan_py(path: Path) -> List[str]:
    return _scan_text_lines(path, path.read_text().splitlines())


def test_topcoffea_import_style() -> None:
    violations: List[str] = []
    for path in _iter_source_files():
        if _is_vendor_file(path):
            continue
        if path.suffix == ".ipynb":
            violations.extend(_scan_ipynb(path))
        else:
            violations.extend(_scan_py(path))
    assert not violations, "\n".join(violations)
