#!/usr/bin/env bash
set -Eeuo pipefail

# Trap errors to provide clear diagnostics to Condor logs.
trap 'echo "[condor_plotter_entry] ERROR: Script failed at line ${LINENO}." >&2' ERR

print_usage() {
    cat <<USAGE
Usage: $(basename "$0") [--ceph-root PATH] [--conda-prefix PATH] [PLOTTER ARGS...]

Optional arguments:
  --ceph-root PATH      Root directory containing the analysis repository (defaults to \$CEPH_ROOT environment variable).
  --conda-prefix PATH   Prefix path of the conda installation (defaults to \$CONDA_PREFIX environment variable).
  -h, --help            Show this help message and exit.

All remaining arguments are forwarded to run_plotter.sh.
USAGE
}

CEPH_ROOT_DEFAULT="${CEPH_ROOT:-}"
CONDA_PREFIX_DEFAULT="${CONDA_PREFIX:-}"

CEPH_ROOT_VALUE="${CEPH_ROOT_DEFAULT}"
CONDA_PREFIX_VALUE="${CONDA_PREFIX_DEFAULT}"

# Collect positional arguments for run_plotter.sh
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ceph-root)
            shift
            if [[ $# -eq 0 ]]; then
                echo "[condor_plotter_entry] ERROR: --ceph-root requires a value." >&2
                exit 1
            fi
            CEPH_ROOT_VALUE="$1"
            ;;
        --conda-prefix)
            shift
            if [[ $# -eq 0 ]]; then
                echo "[condor_plotter_entry] ERROR: --conda-prefix requires a value." >&2
                exit 1
            fi
            CONDA_PREFIX_VALUE="$1"
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            ARGS+=("$@")
            break
            ;;
        -*)
            # Allow dash-leading arguments to pass through to run_plotter.sh.
            ARGS+=("$1")
            ;;
        *)
            ARGS+=("$1")
            ;;
    esac
    shift || true
done

if [[ -z "$CEPH_ROOT_VALUE" ]]; then
    echo "[condor_plotter_entry] ERROR: No CEPH root provided. Use --ceph-root or set the CEPH_ROOT environment variable." >&2
    exit 1
fi

if [[ -z "$CONDA_PREFIX_VALUE" ]]; then
    echo "[condor_plotter_entry] ERROR: No conda prefix provided. Use --conda-prefix or set the CONDA_PREFIX environment variable." >&2
    exit 1
fi

if [[ ! -d "$CEPH_ROOT_VALUE" ]]; then
    echo "[condor_plotter_entry] ERROR: CEPH root '$CEPH_ROOT_VALUE' does not exist or is not a directory." >&2
    exit 1
fi

CONDA_SH_PATH="$CONDA_PREFIX_VALUE/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH_PATH" ]]; then
    echo "[condor_plotter_entry] ERROR: Unable to find conda.sh at '$CONDA_SH_PATH'." >&2
    exit 1
fi

export CEPH_ROOT="$CEPH_ROOT_VALUE"
export CONDA_PREFIX="$CONDA_PREFIX_VALUE"

unset PYTHONPATH

echo "[condor_plotter_entry] Changing directory to CEPH root: $CEPH_ROOT" >&2
cd "$CEPH_ROOT"

# shellcheck disable=SC1090
source "$CONDA_SH_PATH"

if ! conda activate clib-env; then
    echo "[condor_plotter_entry] ERROR: Failed to activate conda environment 'clib-env'." >&2
    exit 1
fi

echo "[condor_plotter_entry] Activated conda environment 'clib-env'." >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[condor_plotter_entry] Running run_plotter.sh ${ARGS[*]:-}" >&2
exec ./run_plotter.sh "${ARGS[@]}"
