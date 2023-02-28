#! /usr/bin/env python
import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import logging
import glob
import os
import re
import string
from pathlib import Path

import coffea

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

env_dir_cache = Path.cwd().joinpath(Path('topeft-envs'))

py_version = "{}.{}.{}".format(
        sys.version_info[0], sys.version_info[1], sys.version_info[2]
        )  # 3.8 or 3.9, or etc.

coffea_version = coffea.__version__


# Define packages to install from different conda channels.
# This is defines as string so that we can easily checksum the contents.
packages_json_template = string.Template('''
{
    "conda": {
        "channels": [
            "conda-forge"
        ],
        "packages": [
            "python=$py_version",
            "pip",
            "conda",
            "conda-pack",
            "dill",
            "xrootd"
        ]
    },
    "pip": [
        "coffea==$coffea_version",
        "topcoffea"
        ]
}''')

pip_local_to_watch = { "topcoffea": ["topcoffea", "setup.py"] }

packages_json = packages_json_template.substitute(py_version=py_version,coffea_version=coffea_version)

def _check_current_env():
    spec = json.loads(packages_json)
    with tempfile.NamedTemporaryFile() as f:
        # export current conda enviornment
        subprocess.check_call(['conda', 'env', 'export', '--json'], stdout=f)
        spec_file = open(f.name,  'r')
        current_spec = json.load(spec_file)
        if 'dependencies' in current_spec:
	    # get current conda packages
            conda_deps = {re.sub("[!~=<>].*$", "", x):x  for x in current_spec['dependencies'] if not isinstance(x, dict)}
	    # get current pip packages
            pip_deps = {re.sub("[!~=<>].*$", "", y):y for y in  [x for x in current_spec['dependencies'] if isinstance(x, dict) and 'pip' in x for x in x['pip']]}


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

def _create_env(env_name, force=False):
    if force:
        logger.info("Forcing rebuilding of {}".format(env_name))
        Path(env_name).unlink(missing_ok=True)
    elif Path(env_name).exists():
        logger.info("Found in cache {}".format(env_name))
        return env_name

    with tempfile.NamedTemporaryFile() as f:
        logger.info("Checking current conda environment")
        spec = _check_current_env()
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


def _commits_local_pip(paths):
    commits = {}
    for (pkg, path) in paths.items():
        try:
            to_watch = []
            paths = pip_local_to_watch.get(pkg, None)
            if paths:
                to_watch = [":(top){}".format(d) for d in paths]

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
                logger.warning("Found unstaged changes in {}:\n{}".format(path,changed))
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
    envs = sorted(glob.glob(os.path.join(env_dir_cache, 'env_*.tar.gz')), key=lambda f: -os.stat(f).st_mtime)
    for f in envs[cache_size:]:
        if not f in current_files:
            logger.info("Trimming cached environment file {}".format(f))
            os.remove(f)


def get_environment(force=False, unstaged='rebuild', cache_size=3):
    # ensure cache directory exists
    Path(env_dir_cache).mkdir(parents=True, exist_ok=True)

    packages_hash = hashlib.sha256(packages_json.encode()).hexdigest()[0:8]
    pip_paths = _find_local_pip()
    pip_commits = _commits_local_pip(pip_paths)
    pip_check = _compute_commit(pip_paths, pip_commits)

    env_name = str(Path(env_dir_cache).joinpath("env_spec_{}_edit_{}".format(packages_hash, pip_check)).with_suffix(".tar.gz"))
    _clean_cache(cache_size, env_name)

    if pip_check == 'HEAD':
        changed = [p for p in pip_commits if pip_commits[p] == 'HEAD']
        if unstaged == 'fail':
            raise UnstagedChanges(changed)
        if unstaged == 'rebuild':
            force = True
            logger.warning("Rebuilding environment because unstaged changes in {}".format(', '.join([Path(p).name for p in changed])))

    return _create_env(env_name, force)



class UnstagedChanges(Exception):
    pass

if __name__ == '__main__':
    print(get_environment())

