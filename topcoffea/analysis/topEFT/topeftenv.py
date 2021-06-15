#! /usr/bin/env python
import json
import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import logging
import glob
import os

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

env_dir_cache = 'envs'

# Define packages to install from different conda channels.
# This is defines as json so that we can easily checksum the contents.
packages_json = '''
{
    "base": {
        "conda": {
            "defaults" : ["python=3.8.3", "conda"],
            "conda-forge" : ["conda-pack", "dill", "xrootd", "coffea"]
        }
    },
    "user": {
        "conda": [
        ],
        "pip": [
        ]
    }
}
'''
packages = json.loads(packages_json)


def _run_conda_command(environment, command, *args):
    all_args = ['conda', command, '--yes', '--prefix={}'.format(str(environment))] + list(args)

    try:
        subprocess.check_output(all_args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.warning("Error executing: {}".format(' '.join(all_args)))
        print(e.output)
        sys.exit(1)

def _install_conda_packages(env_path, channel, pkgs):
    logger.debug("Installing {} into {} via conda".format(','.join(pkgs), env_path))
    return _run_conda_command(
            env_path,
            'install',
            '-c',
            channel,
            *pkgs)

def _install_pip_requirements(env_path, pip_path):
    req_file = pathlib.Path(pip_path).joinpath('requirements.txt')
    if req_file.is_file():
        logger.debug("Installing {} into {} via pip".format(pip_path, env_path))
        _run_conda_command(
                env_path,
                'run',
                'sh', '-c', 'pip install . && pip uninstall `python setup.py --name`')

def _create_base_env(packages_hash, pip_paths, force=False):
    pathlib.Path(env_dir_cache).mkdir(parents=True, exist_ok=True)
    base_env_path=pathlib.Path(env_dir_cache).joinpath("base_env_{}".format(packages_hash))
    output=base_env_path.with_suffix(".tar.gz")

    logger.debug("Looking for base environment {}...".format(output))

    if force:
        logger.debug("Forcing rebuilding of {}".format(output))
        pathlib.Path(output).unlink(missing_ok=True)

    if pathlib.Path(output).exists():
        logger.debug("Found in cache {}".format(output))
        return str(output)

    logger.debug("Creating environment {}".format(base_env_path))
    _run_conda_command(base_env_path, 'create')

    for (channel, pkgs) in packages['base']['conda'].items():
        _install_conda_packages(base_env_path, channel, pkgs)

    for pip_path in pip_paths:
        _install_pip_requirements(base_env_path, pip_path)

    logger.debug("Generating {} environment file".format(output))
    try:
        subprocess.check_output(['conda-pack', '--prefix', base_env_path, '--output', str(output)], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)

    shutil.rmtree(base_env_path)

    return str(output)


def _find_local_pip():
    edit_raw = subprocess.check_output([sys.executable, '-m' 'pip', 'list', '--editable']).decode()

    # drop first two lines, which are just a header
    edit_raw = edit_raw.split('\n')[2:]

    paths = []
    for line in edit_raw:
        if not line:
            # skip empty lines
            continue
        # we are only interested in the path information of the package, which
        # is in the last column
        path = line.split()[-1]
        paths.append(path)
    return sorted(paths)

def _commits_local_pip(paths):
    commits = {}
    for path in paths:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path).decode().rstrip()
            changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=path).decode().rstrip()
            if changed:
                logger.warning("Found unstaged changes in '{}'".format(path))
                commits[path] = 'HEAD'
            else:
                commits[path] = commit
        except Exception as e:
            # on error, e.g., not a git repository, assume that current state
            # should be installed
            logger.warning("Could not get current commit of '{}'.".format(path))
            commits[path] = 'HEAD'
    return commits

def _compute_commit(paths, commits):
    # list commits according to paths ordering
    values = [commits[p] for p in paths]
    if 'HEAD' in values:
        # if commit is HEAD, then return that, as we always rebuild the
        # environment in that case.
        return 'HEAD'
    return hashlib.sha256(''.join(values).encode()).hexdigest()[0:8]

def _clean_cache(cache_size, *current_files):
    base_envs = sorted(glob.glob(os.path.join(env_dir_cache, 'base_env_*')), key=lambda f: -os.stat(f).st_mtime)
    full_envs = sorted(glob.glob(os.path.join(env_dir_cache, 'full_env_*')), key=lambda f: -os.stat(f).st_mtime)

    for f in base_envs[cache_size:]:
        if not f in current_files:
            logger.debug("Trimming cached environment file {}".format(f))
            os.remove(f)
    for f in full_envs[cache_size:]:
        if not f in current_files:
            logger.debug("Trimming cached environment file {}".format(f))
            os.remove(f)

def _install_local_pip(base_env_tarball, env_dir, pip_path):
    logger.debug("Installing {} from editable pip".format(pip_path))

    with tempfile.NamedTemporaryFile(mode='w') as pip_recipe:
        pip_recipe.write("""
#! /bin/bash
# remove if conda installed:
pip_name=$(cd {path} && python setup.py --name)
python_package_run -e {base_env_tarball} -u {env_dir} -- conda remove --yes --force "$pip_name"

# install from pip local path
set -e
python_package_run -e {base_env_tarball} -u {env_dir} -- pip install {path}
    """.format(base_env_tarball=base_env_tarball, env_dir=env_dir, path=pip_path))
        pip_recipe.flush()
        try:
            subprocess.check_output(['/bin/bash', pip_recipe.name], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output)
            sys.exit(1)

def get_environment(force=False, unstaged='rebuild', cache_size=3):
    pathlib.Path(env_dir_cache).mkdir(parents=True, exist_ok=True)

    packages_hash = hashlib.sha256(packages_json.encode()).hexdigest()[0:8]
    pip_paths = _find_local_pip()
    pip_commits = _commits_local_pip(pip_paths)
    pip_check = _compute_commit(pip_paths, pip_commits)

    base_env_tarball = _create_base_env(packages_hash, pip_paths, force)
    full_env_tarball = pathlib.Path(env_dir_cache).joinpath("full_env_{}_{}".format(packages_hash, pip_check)).with_suffix(".tar.gz")

    _clean_cache(cache_size, base_env_tarball, full_env_tarball)

    if pip_check == 'HEAD':
        changed = [p for p in pip_commits if pip_commits[p] == 'HEAD']
        if unstaged == 'fail':
            raise UnstagedChanges(changed)
        if unstaged == 'rebuild':
            force = True
            logger.warning("Rebuilding environment because unstaged changes in {}".format(', '.join([pathlib.Path(p).name for p in changed])))

    if force:
        logger.warning("Forcing rebuild of {}".format(full_env_tarball))
        pathlib.Path(full_env_tarball).unlink(missing_ok=True)

    if pathlib.Path(full_env_tarball).exists():
        logger.debug("Found in cache {}".format(full_env_tarball))
        return str(full_env_tarball)

    with tempfile.TemporaryDirectory() as tmp_env:
        for path in pip_paths:
            _install_local_pip(base_env_tarball, tmp_env, path)

        logger.debug("Generating {} environment file".format(full_env_tarball))
        try:
            subprocess.check_output(['conda-pack', '--prefix', tmp_env, '--output', str(full_env_tarball)], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output)
            sys.exit(1)
    return str(full_env_tarball)


class UnstagedChanges(Exception):
    pass

if __name__ == '__main__':
    print(get_environment())

