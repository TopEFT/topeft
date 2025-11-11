#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

REPO_ROOT="/users/apiccine/work/correction-lib/topeft"
ENTRY_DIR="${REPO_ROOT}/analysis/topeft_run2"

cd "${REPO_ROOT}"

activate_with_conda() {
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        eval "$(conda shell.bash hook)"
        conda activate clib-env
        return 0
    fi
    return 1
}

if ! activate_with_conda; then
    ACTIVATE_SCRIPT="${REPO_ROOT}/clib-env/bin/activate"
    if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
        # shellcheck disable=SC1091
        source "${ACTIVATE_SCRIPT}"
    else
        echo "[condor_plotter_entry] ERROR: Unable to activate clib-env; conda not found and '${ACTIVATE_SCRIPT}' is missing." >&2
        exit 1
    fi
fi

cd "${ENTRY_DIR}"

exec ./run_plotter.sh "$@"
