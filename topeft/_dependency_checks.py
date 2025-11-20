"""Dependency guards for ensuring aligned topcoffea checkouts."""

from __future__ import annotations

"""Dependency guards to keep sibling topcoffea checkouts aligned."""

import os
from pathlib import Path
from typing import Optional, Sequence

EXPECTED_TOPCOFFEA_REFS: tuple[str, ...] = tuple(
    ref.strip()
    for ref in os.environ.get("TOPEFT_TOPCOFFEA_EXPECTED", "ch_update_calcoffea").split(",")
    if ref.strip()
)


def _topcoffea_repo_root(topcoffea_pkg: object) -> Optional[Path]:
    module_file = getattr(topcoffea_pkg, "__file__", None)
    candidates = []
    if module_file:
        try:
            candidates.append(Path(module_file).resolve())
        except TypeError:  # pragma: no cover - defensive
            pass

    for entry in getattr(topcoffea_pkg, "__path__", []):
        try:
            candidates.append(Path(entry).resolve())
        except TypeError:  # pragma: no cover - defensive
            continue

    for module_path in candidates:
        try:
            return module_path.parent.parent
        except AttributeError:  # pragma: no cover - defensive
            continue
    return None


def _branch_from_head(repo_root: Path) -> Optional[str]:
    head_path = repo_root / ".git" / "HEAD"
    try:
        contents = head_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    if contents.startswith("ref:"):
        ref_name = contents.split("ref:", 1)[1].strip()
        return Path(ref_name).name
    return None


def _normalise_expected(expected: Optional[Sequence[str]]) -> tuple[str, ...]:
    refs = tuple(ref.strip() for ref in (expected or EXPECTED_TOPCOFFEA_REFS) if ref.strip())
    return refs or ("ch_update_calcoffea",)


def ensure_topcoffea_branch(expected_refs: Optional[Sequence[str]] = None) -> None:
    """Raise ``RuntimeError`` when the topcoffea checkout is off the baseline."""

    if os.environ.get("TOPEFT_SKIP_TOPCOFFEA_BRANCH_CHECK"):
        return

    try:
        import topcoffea  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "topcoffea is not installed. Clone https://github.com/TopEFT/topcoffea, "
            "switch to the ch_update_calcoffea branch, and install it with 'pip install -e .'"
        ) from exc

    repo_root = _topcoffea_repo_root(topcoffea)
    if repo_root is None:
        return

    branch = os.environ.get("TOPCOFFEA_BRANCH")
    if not branch:
        branch = _branch_from_head(repo_root)

    if not branch:
        # No git metadata available; assume tests or a vendored copy.
        return

    expected = _normalise_expected(expected_refs)
    if branch in expected:
        return

    message = (
        "topcoffea checkout at {repo} is on '{branch}' but the workflow expects one of "
        "{expected}. Run 'git -C {repo} switch ch_update_calcoffea' (or checkout the "
        "matching tag) before running topeft, or export TOPCOFFEA_BRANCH to override.".format(
            repo=repo_root,
            branch=branch,
            expected=", ".join(expected),
        )
    )
    raise RuntimeError(message)


__all__ = ["ensure_topcoffea_branch"]
