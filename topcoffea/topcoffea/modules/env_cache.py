"""Utilities for naming and locating cached environment tarballs.

The naming scheme historically lived in multiple helpers.  Keeping the
canonical implementation here avoids the format drifting between
repositories that share the cache directory.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

__all__ = [
    "ENV_CACHE_PREFIX",
    "ENV_SPEC_LABEL",
    "ENV_EDIT_LABEL",
    "ENV_TARBALL_SUFFIX",
    "validate_component",
    "build_env_tarball_stem",
    "build_env_tarball_path",
    "cache_glob_pattern",
]

ENV_CACHE_PREFIX = "env"
"""Leading prefix for packaged environment tarballs."""

ENV_SPEC_LABEL = "spec"
"""Segment label for the package specification hash."""

ENV_EDIT_LABEL = "edit"
"""Segment label for the editable source state hash."""

ENV_TARBALL_SUFFIX = ".tar.gz"
"""Filename suffix for packaged environments."""

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_component(value: str, *, label: str) -> str:
    """Ensure ``value`` is safe to embed in a filename.

    Parameters
    ----------
    value:
        String that will become part of the tarball name.
    label:
        Human readable description used in error messages.

    Returns
    -------
    str
        The original value, if validation succeeds.

    Raises
    ------
    ValueError
        If ``value`` contains characters that would make the resulting
        filename ambiguous or unsafe to use in glob patterns.
    """

    if not value:
        raise ValueError(f"Empty {label} is not a valid cache component")
    if not _SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"Invalid characters in {label!s}: {value!r}. "
            "Allowed characters are alphanumerics, dash, underscore, and period."
        )
    return value


def build_env_tarball_stem(
    package_digest: str,
    editable_digest: str,
    *,
    prefix: str = ENV_CACHE_PREFIX,
    spec_label: str = ENV_SPEC_LABEL,
    edit_label: str = ENV_EDIT_LABEL,
) -> str:
    """Return the stem for an environment cache tarball.

    The returned value does **not** include an extension.  All components
    are validated to guard against unsafe strings making their way into
    glob expressions or filesystem paths.
    """

    validated_components: Iterable[str] = (
        validate_component(prefix, label="cache prefix"),
        validate_component(spec_label, label="spec label"),
        validate_component(package_digest, label="package digest"),
        validate_component(edit_label, label="editable label"),
        validate_component(editable_digest, label="editable digest"),
    )
    return "_".join(validated_components)


def build_env_tarball_path(
    cache_dir: Path,
    package_digest: str,
    editable_digest: str,
    *,
    suffix: str = ENV_TARBALL_SUFFIX,
    **stem_kwargs,
) -> Path:
    """Build the :class:`~pathlib.Path` to the cached environment tarball."""

    stem = build_env_tarball_stem(package_digest, editable_digest, **stem_kwargs)
    return Path(cache_dir).joinpath(stem).with_suffix(suffix)


def cache_glob_pattern(cache_dir: Path, *, suffix: str = ENV_TARBALL_SUFFIX) -> str:
    """Return a glob that matches cache tarballs in ``cache_dir``."""

    validated_components = (
        validate_component(ENV_CACHE_PREFIX, label="cache prefix"),
        validate_component(ENV_SPEC_LABEL, label="spec label"),
        "*",
        validate_component(ENV_EDIT_LABEL, label="editable label"),
        "*",
    )
    pattern = "_".join(validated_components) + suffix
    return str(Path(cache_dir).joinpath(pattern))
