#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

main() {
    unset PYTHONPATH

    local SCRIPT_DIR
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local REPO_ROOT="${TOPEFT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
    local ENTRY_DIR="${TOPEFT_ENTRY_DIR:-${SCRIPT_DIR}}"

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

    activate_with_prefix() {
        local prefix="$1"
        local activate_script="${prefix}/bin/activate"

        if [[ -f "${activate_script}" ]]; then
            # shellcheck disable=SC1091
            source "${activate_script}"
            return 0
        fi

        if [[ -f "${prefix}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "${prefix}/etc/profile.d/conda.sh"
            conda activate clib-env
            return 0
        fi

        return 1
    }

    if [[ -n "${TOPEFT_CONDA_PREFIX:-}" ]]; then
        if ! activate_with_prefix "${TOPEFT_CONDA_PREFIX}"; then
            echo "[condor_plotter_entry] ERROR: Unable to activate clib-env from TOPEFT_CONDA_PREFIX='${TOPEFT_CONDA_PREFIX}'." >&2
            return 1
        fi
    elif ! activate_with_conda; then
        local DEFAULT_PREFIX="${REPO_ROOT}/clib-env"
        if ! activate_with_prefix "${DEFAULT_PREFIX}"; then
            echo "[condor_plotter_entry] ERROR: Unable to activate clib-env; conda not found and neither TOPEFT_CONDA_PREFIX nor '${DEFAULT_PREFIX}' is usable." >&2
            return 1
        fi
    fi

    cd "${ENTRY_DIR}"

    "${ENTRY_DIR}/run_plotter.sh" "$@"
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
