"""Utility helpers for building cached remote execution environments.

This module mirrors the layout used in the ttbarEFT workflows while keeping the
TopCoffea-specific conveniences for tracking editable installs.  The main entry
point is :func:`get_environment`, which prepares a ``poncho`` environment
archive ready to be shipped to remote workers.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

ENV_DIR_CACHE = Path.cwd() / "topeft-envs"
PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

DEFAULT_MODULES: Dict[str, object] = {
    "conda": {
        "channels": ["conda-forge"],
        "packages": [
            f"python={PY_VERSION}",
            "pip",
            "conda-pack",
            "ndcctools>=7.14.11",
            "xrootd",
            "setuptools>=72",
        ],
    },
    "pip": [
        "topeft",
        "topcoffea",
        "coffea==2025.7.3",
        "awkward==2.8.7",
    ],
}

PIP_LOCAL_TO_WATCH: Dict[str, Sequence[str]] = {
    "topcoffea": ("topcoffea", "setup.py"),
    "topeft": ("topeft", "setup.py"),
}


class UnstagedChanges(RuntimeError):
    """Raised when editable packages contain unstaged modifications."""


def _ensure_executable(executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(f"Required executable '{executable}' was not found on PATH")


def _normalise_iterable(values: Iterable[str]) -> list[str]:
    # Preserve order while dropping duplicates
    seen = set()
    normalised: list[str] = []
    for value in values:
        if value not in seen:
            normalised.append(value)
            seen.add(value)
    return normalised


def _spec_with_overrides(
    extra_conda: Optional[Iterable[str]] = None,
    extra_pip: Optional[Iterable[str]] = None,
    extra_pip_local: Optional[Mapping[str, Sequence[str]]] = None,
) -> tuple[Dict[str, object], Dict[str, Sequence[str]]]:
    spec = deepcopy(DEFAULT_MODULES)
    pip_local_watch = dict(PIP_LOCAL_TO_WATCH)

    if extra_conda:
        conda_packages = spec["conda"]["packages"]  # type: ignore[index]
        spec["conda"]["packages"] = _normalise_iterable(list(conda_packages) + list(extra_conda))  # type: ignore[index]

    if extra_pip:
        spec["pip"] = _normalise_iterable(list(spec["pip"]) + list(extra_pip))  # type: ignore[index]

    if extra_pip_local:
        # Merge the watch configuration in a ttbarEFT-compatible fashion
        for pkg, paths in extra_pip_local.items():
            spec["pip"] = _normalise_iterable(list(spec["pip"]) + [pkg])  # type: ignore[index]
            pip_local_watch[pkg] = tuple(paths)

    return spec, pip_local_watch


def _current_versions_conda(conda_env_path: Optional[Path] = None) -> Dict[str, str]:
    if not conda_env_path:
        conda_env_path = Path(os.environ["CONDA_PREFIX"])

    proc = subprocess.run(
        ["conda", "list", "--export", "--json"],
        check=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    raw_pkgs = json.loads(proc.stdout.decode())

    pkgs: Dict[str, str] = {}
    for pkg in raw_pkgs:
        name = pkg["name"]
        version = f"{pkg['version']}={pkg['build_string']}"
        pkgs[name] = f"{name}={version}"

    return pkgs


def _check_current_env(spec: Dict[str, object]) -> Dict[str, object]:
    with tempfile.NamedTemporaryFile() as temp:
        subprocess.check_call(
            ["conda", "env", "export", "--json"],
            stdout=temp,
            stdin=subprocess.DEVNULL,
        )
        with open(temp.name, "r", encoding="utf-8") as spec_file:
            current_spec = json.load(spec_file)

    current_spec["pinning"] = {"conda": _current_versions_conda()}

    if "dependencies" in current_spec:
        conda_deps = {
            re.sub("[!~=<>].*$", "", dep): dep
            for dep in current_spec["dependencies"]
            if not isinstance(dep, dict)
        }
        pip_deps = {
            re.sub("[!~=<>].*$", pip_dep): pip_dep
            for dependency in current_spec["dependencies"]
            if isinstance(dependency, dict) and "pip" in dependency
            for pip_dep in dependency["pip"]
        }

        conda_packages = spec["conda"]["packages"]  # type: ignore[index]
        for index, package in enumerate(conda_packages):
            if not re.search("[!~=<>].*$", package) and package in conda_deps:
                conda_packages[index] = conda_deps[package]

        pip_packages = spec["pip"]  # type: ignore[index]
        for index, package in enumerate(pip_packages):
            if not re.search("[!~=<>].*$", package) and package in pip_deps:
                pip_packages[index] = pip_deps[package]

    return spec


def _find_local_pip() -> Dict[str, str]:
    edit_raw = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--editable"],
        stdin=subprocess.DEVNULL,
    ).decode()

    edit_raw = edit_raw.split("\n")[2:]
    path_of: Dict[str, str] = {}
    for line in edit_raw:
        if not line:
            continue
        pkg, _version, location = line.split()
        path_of[pkg] = location
    return path_of


def _commits_local_pip(
    paths: Mapping[str, str],
    watch_paths: Mapping[str, Sequence[str]],
) -> Dict[str, str]:
    commits: Dict[str, str] = {}
    for pkg, path in paths.items():
        try:
            to_watch: list[str] = []
            if pkg in watch_paths:
                to_watch = [f":(top){watch_path}" for watch_path in watch_paths[pkg]]

            try:
                commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=path,
                        stdin=subprocess.DEVNULL,
                    )
                    .decode()
                    .rstrip()
                )
            except FileNotFoundError as error:
                raise FileNotFoundError("Could not find the git executable in PATH") from error

            changed = True
            cmd = ["git", "status", "--porcelain", "--untracked-files=no"]
            try:
                changed = subprocess.check_output(
                    cmd + to_watch,
                    cwd=path,
                    stdin=subprocess.DEVNULL,
                ).decode().rstrip()
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not apply git paths-to-watch filters. Trying without them...",
                )
                changed = subprocess.check_output(
                    cmd,
                    cwd=path,
                    stdin=subprocess.DEVNULL,
                ).decode().rstrip()

            if changed:
                logger.warning("Found unstaged changes in %s:\n%s", path, changed)
                commits[pkg] = "HEAD"
            else:
                commits[pkg] = commit
        except Exception as error:  # noqa: BLE001 - we want resilience
            logger.warning("Could not get current commit of '%s': %s", path, error)
            commits[pkg] = "HEAD"
    return commits


def _compute_commit(paths: Iterable[str], commits: Mapping[str, str]) -> str:
    path_commits = [commits[path] for path in paths if path in commits]
    if not path_commits:
        return "fixed"
    if "HEAD" in path_commits:
        return "HEAD"
    return hashlib.sha256("".join(path_commits).encode()).hexdigest()[0:8]


def _clean_cache(cache_size: int, *current_files: str) -> None:
    envs = sorted(
        glob.glob(os.path.join(ENV_DIR_CACHE, "env_*.tar.gz")),
        key=lambda file: -os.stat(file).st_mtime,
    )
    keep = set(current_files)
    for file_path in envs[cache_size:]:
        if file_path not in keep:
            logger.info("Trimming cached environment file %s", file_path)
            os.remove(file_path)


def _create_env(env_name: str, spec: Dict[str, object], force: bool = False) -> str:
    if force and Path(env_name).exists():
        logger.info("Forcing rebuilding of %s", env_name)
        Path(env_name).unlink()
    elif Path(env_name).exists():
        logger.info("Found cached environment %s", env_name)
        return env_name

    logger.info("Checking current conda environment")
    spec = _check_current_env(spec)

    with tempfile.NamedTemporaryFile() as temp:
        packages_json = json.dumps(spec, sort_keys=True)
        logger.info("Base environment specification: %s", packages_json)
        temp.write(packages_json.encode())
        temp.flush()

        logger.info("Creating environment %s", env_name)
        try:
            subprocess.check_output(
                ["poncho_package_create", temp.name, env_name],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as error:  # pragma: no cover - passthrough for visibility
            logger.error("poncho package creation failed with code %s", error.returncode)
            logger.error("%s", error.output.decode())
            raise

    return env_name


def get_environment(
    extra_conda: Optional[Iterable[str]] = None,
    extra_pip: Optional[Iterable[str]] = None,
    extra_pip_local: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    force: bool = False,
    unstaged: str = "rebuild",
    cache_size: int = 3,
) -> str:
    """Build (or reuse) the tarball describing the execution environment."""

    _ensure_executable("conda")
    _ensure_executable("poncho_package_create")

    ENV_DIR_CACHE.mkdir(parents=True, exist_ok=True)

    spec, pip_local_watch = _spec_with_overrides(extra_conda, extra_pip, extra_pip_local)

    packages_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[0:8]

    pip_paths = _find_local_pip()
    pip_commits = _commits_local_pip(pip_paths, pip_local_watch)
    pip_check = _compute_commit(pip_paths.keys(), pip_commits)

    env_name = str(
        (ENV_DIR_CACHE / f"env_spec_{packages_hash}_edit_{pip_check}").with_suffix(".tar.gz")
    )
    _clean_cache(cache_size, env_name)

    if pip_check == "HEAD":
        changed = [pkg for pkg, commit in pip_commits.items() if commit == "HEAD"]
        if unstaged == "fail":
            raise UnstagedChanges(
                f"Editable packages with unstaged changes: {', '.join(sorted(changed))}",
            )
        if unstaged == "rebuild":
            force = True
            logger.warning(
                "Rebuilding environment because unstaged changes in %s",
                ", ".join(sorted(changed)),
            )

    return _create_env(env_name, spec, force)


if __name__ == "__main__":
    print(get_environment())
