#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

main() {
    unset PYTHONPATH

    local entry_dir="${TOPEFT_ENTRY_DIR:-}"

    if [[ -z "${entry_dir}" && -n "${_CONDOR_JOB_IWD:-}" ]]; then
        entry_dir="${_CONDOR_JOB_IWD}"
    fi

    if [[ -z "${entry_dir}" && -n "${PWD:-}" ]]; then
        entry_dir="${PWD}"
    fi

    if [[ -z "${entry_dir}" ]]; then
        local script_dir
        script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        entry_dir="${script_dir}"
    fi

    echo "[condor_plotter_entry] Using entry directory: ${entry_dir}" >&2

    cd "${entry_dir}" || {
        echo "[condor_plotter_entry] ERROR: Failed to cd into '${entry_dir}'." >&2
        return 1
    }

    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        eval "$(conda shell.bash hook)"
        if [[ -n "${TOPEFT_CONDA_PREFIX:-}" ]]; then
            conda activate "${TOPEFT_CONDA_PREFIX}"
        else
            conda activate clib-env
        fi
    elif [[ -n "${TOPEFT_CONDA_PREFIX:-}" && -f "${TOPEFT_CONDA_PREFIX}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${TOPEFT_CONDA_PREFIX}/bin/activate"
    fi

    ./run_plotter.sh "$@"
}

if main "$@"; then
    exit_code=0
else
    exit_code=$?
fi
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit "${exit_code}"
else
    return "${exit_code}"
fi
