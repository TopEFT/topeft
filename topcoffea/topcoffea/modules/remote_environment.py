#! /usr/bin/env python
import json
import hashlib
import subprocess
import sys
import tempfile
import logging
import glob
import os
import re
from pathlib import Path

from typing import Dict, List, Optional

from .env_cache import build_env_tarball_path, cache_glob_pattern

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

env_dir_cache = Path.cwd().joinpath(Path('topeft-envs'))

py_version = "{}.{}.{}".format(
    sys.version_info[0], sys.version_info[1], sys.version_info[2]
)  # 3.8 or 3.9, or etc.

default_modules = {
    "conda": {
        "channels": ["conda-forge"],
        "packages": [
            f"python={py_version}",
            "awkward=2.8.7",
            "coffea=2025.7.3",
            "numpy",
            "ndcctools",
            "pip",
            "conda",
            "conda-pack",
            "dill",
            "xrootd",
            "setuptools==80.9.0",
        ],
    },
    "pip": ["topcoffea"],
}

pip_local_to_watch = {
    "topcoffea": ["topcoffea", "setup.py"],
    "topeft": [
        "topeft",
        "setup.py",
        "pyproject.toml",
        "poetry.lock",
        "requirements.txt",
        "setup.cfg",
        "environment.yml",
    ],
}

# Backwards-compatibility aliases retained for callers expecting uppercase names.
DEFAULT_MODULES = default_modules
PIP_LOCAL_TO_WATCH = pip_local_to_watch


def _check_current_env(spec: Dict):
    with tempfile.NamedTemporaryFile() as f:
        # export current conda enviornment
        subprocess.check_call(['conda', 'env', 'export', '--json'], stdout=f)
        spec_file = open(f.name, "r")
        current_spec = json.load(spec_file)
        if 'dependencies' in current_spec:
            # get current conda packages
            conda_deps = {
                re.sub("[!~=<>].*$", "", x): x
                for x in current_spec["dependencies"]
                if not isinstance(x, dict)
            }
            # get current pip packages
            pip_deps = {
                re.sub("[!~=<>].*$", "", y): y
                for y in [
                    x
                    for x in current_spec["dependencies"]
                    if isinstance(x, dict) and "pip" in x
                    for x in x["pip"]
                ]
            }

            # replace any conda packages
            for i in range(len(spec['conda']['packages'])):
                # ignore packages where a version is already specified
                package = spec['conda']['packages'][i]
                if not re.search("[!~=<>].*$", package):
                    if package in conda_deps:
                        spec['conda']['packages'][i] = conda_deps[package]

            # replace any pip packages
            for i in range(len(spec['pip'])):
                # ignore packages where a version is already specified
                package = spec['pip'][i]
                if not re.search("[!~=<>].*$", package):
                    if package in pip_deps:
                        spec['pip'][i] = pip_deps[package]
    return spec


def _create_env(env_name: str, spec: Dict, force: bool = False):
    if force:
        logger.info("Forcing rebuilding of {}".format(env_name))
        Path(env_name).unlink(missing_ok=True)
    elif Path(env_name).exists():
        logger.info("Found in cache {}".format(env_name))
        return env_name

    with tempfile.NamedTemporaryFile() as f:
        logger.info("Checking current conda environment")
        spec = _check_current_env(spec)
        packages_json = json.dumps(spec)
        logger.info("base env specification:{}".format(packages_json))
        f.write(packages_json.encode())
        f.flush()
        logger.info("Creating environment {}".format(env_name))
        subprocess.check_call(['poncho_package_create', f.name, env_name])
        return env_name


def _find_local_pip():
    edit_raw = subprocess.check_output([sys.executable, '-m' 'pip', 'list', '--editable']).decode()

    # drop first two lines, which are just a header
    edit_raw = edit_raw.split('\n')[2:]
    path_of = {}
    for line in edit_raw:
        if not line:
            # skip empty lines
            continue
        # we are only interested in the path information of the package, which
        # is in the last column
        (pkg, version, location) = line.split()
        path_of[pkg] = location
    return path_of


def _commits_local_pip(paths, watches: Optional[Dict[str, List[str]]] = None):
    commits = {}
    watch_paths = watches or pip_local_to_watch
    for (pkg, path) in paths.items():
        try:
            to_watch = []
            pkg_watch_paths = watch_paths.get(pkg, None)
            if pkg_watch_paths:
                to_watch = [":(top){}".format(d) for d in pkg_watch_paths]

            try:
                commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode().rstrip()
            except FileNotFoundError:
                raise FileNotFoundError("Could not find the git executable in PATH")

            changed = True
            cmd = ['git', 'status', '--porcelain', '--untracked-files=no']
            try:
                changed = subprocess.check_output(cmd + to_watch, cwd=path).decode().rstrip()
            except subprocess.CalledProcessError:
                logger.warning("Could not apply git paths-to-watch filters. Trying without them...")
                changed = subprocess.check_output(cmd, cwd=path).decode().rstrip()

            if changed:
                logger.warning(
                    "Found unstaged changes in {}:\n{}".format(path, changed)
                )
                commits[pkg] = 'HEAD'
            else:
                commits[pkg] = commit
        except Exception as e:
            # on error, e.g., not a git repository, assume that current state
            # should be installed
            logger.warning(f"Could not get current commit of '{path}': {e}")
            commits[pkg] = "HEAD"
    return commits


def _compute_commit(paths, commits):
    if not commits:
        return "fixed"
    # list commits according to paths ordering
    values = [commits[p] for p in paths]
    if 'HEAD' in values:
        # if commit is HEAD, then return that, as we always rebuild the
        # environment in that case.
        return 'HEAD'
    return hashlib.sha256(''.join(values).encode()).hexdigest()[0:8]


def _clean_cache(cache_size, *current_files):
    envs = sorted(
        glob.glob(cache_glob_pattern(env_dir_cache)),
        key=lambda f: -os.stat(f).st_mtime,
    )
    for f in envs[cache_size:]:
        if f not in current_files:
            logger.info("Trimming cached environment file {}".format(f))
            os.remove(f)


def get_environment(
    extra_conda: Optional[List[str]] = None,
    extra_pip: Optional[List[str]] = None,
    extra_pip_local: Optional[Dict[str, List[str]]] = None,
    force: bool = False,
    unstaged: str = "rebuild",
    cache_size: int = 3,
):
    # ensure cache directory exists
    Path(env_dir_cache).mkdir(parents=True, exist_ok=True)

    spec = dict(default_modules)
    spec_pip_local_to_watch = dict(pip_local_to_watch)
    if extra_conda:
        spec["conda"]["packages"].extend(extra_conda)
    if extra_pip:
        spec["pip"].extend(extra_pip)
    if extra_pip_local:
        spec["pip"].extend(extra_pip_local)
        spec_pip_local_to_watch.update(extra_pip_local)

    packages_hash = hashlib.sha256(json.dumps(spec).encode()).hexdigest()[0:8]
    pip_paths = _find_local_pip()
    pip_commits = _commits_local_pip(pip_paths, spec_pip_local_to_watch)
    pip_check = _compute_commit(pip_paths, pip_commits)

    env_name = str(build_env_tarball_path(env_dir_cache, packages_hash, pip_check))
    _clean_cache(cache_size, env_name)

    if pip_check == 'HEAD':
        changed = [p for p in pip_commits if pip_commits[p] == 'HEAD']
        if unstaged == 'fail':
            raise UnstagedChanges(changed)
        if unstaged == 'rebuild':
            force = True
            logger.warning("Rebuilding environment because unstaged changes in {}".format(', '.join([Path(p).name for p in changed])))

    return _create_env(env_name, spec, force)


class UnstagedChanges(Exception):
    pass


if __name__ == '__main__':
    print(get_environment())
