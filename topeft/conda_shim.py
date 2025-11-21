import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_target():
    requested = os.environ.get("CONDA_EXE")
    if requested:
        return requested

    found = shutil.which("conda")
    if found:
        return found
    return None


def _which_micromamba():
    return shutil.which("micromamba")


def _is_self(target: str) -> bool:
    try:
        return Path(target).resolve() == Path(sys.argv[0]).resolve()
    except FileNotFoundError:
        return False


def _call_conda_module(argv):
    if importlib.util.find_spec("conda") is None:
        return None

    return subprocess.call([sys.executable, "-m", "conda", *argv])


def main() -> int:
    target = _resolve_target()
    if target:
        if not _is_self(target):
            return subprocess.call([target, *sys.argv[1:]])

        sibling_real = Path(sys.argv[0]).with_name("conda_real")
        if sibling_real.exists():
            return subprocess.call([str(sibling_real), *sys.argv[1:]])

        module_result = _call_conda_module(sys.argv[1:])
        if module_result is not None:
            return module_result

    micromamba = _which_micromamba()
    if micromamba:
        return subprocess.call([micromamba, *sys.argv[1:]])

    sys.stderr.write(
        "No conda-compatible executable found. Install conda or micromamba to run this command.\n"
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
