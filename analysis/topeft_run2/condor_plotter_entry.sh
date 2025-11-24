#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

main() {
    unset PYTHONPATH

    local entry_dir="${TOPEFT_ENTRY_DIR:-}"

    if [[ $# -gt 0 && "$1" == TOPEFT_ENTRY_DIR=* ]]; then
        if [[ -z "${entry_dir}" ]]; then
            entry_dir="${1#TOPEFT_ENTRY_DIR=}"
        fi
        shift
    fi

    if [[ -z "${entry_dir}" ]]; then
        local script_dir
        script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

        if [[ -n "${PWD:-}" ]]; then
            entry_dir="${PWD}"
        fi

        if [[ -z "${entry_dir}" && -n "${script_dir}" ]]; then
            entry_dir="${script_dir}"
        fi
    fi

    if [[ -z "${entry_dir}" ]]; then
        echo "[condor_plotter_entry] ERROR: TOPEFT_ENTRY_DIR is not set and no fallback was found." >&2
        return 1
    fi

    echo "[condor_plotter_entry] Using entry directory: ${entry_dir}" >&2

    cd "${entry_dir}" || {
        echo "[condor_plotter_entry] ERROR: Failed to cd into '${entry_dir}'." >&2
        return 1
    }

    activate_with_conda() {
        if command -v conda >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            eval "$(conda shell.bash hook)"
            conda activate "${1}"
            return 0
        fi
        return 1
    }

    activate_from_prefix() {
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
            conda activate "${prefix}"
            return 0
        fi

        return 1
    }

    if [[ -n "${TOPEFT_CONDA_PREFIX:-}" ]]; then
        if ! activate_with_conda "${TOPEFT_CONDA_PREFIX}" && ! activate_from_prefix "${TOPEFT_CONDA_PREFIX}"; then
            echo "[condor_plotter_entry] ERROR: Unable to activate environment from TOPEFT_CONDA_PREFIX='${TOPEFT_CONDA_PREFIX}'." >&2
            return 1
        fi
    elif ! activate_with_conda "clib-env"; then
        local -a fallback_prefixes=()

        if [[ -n "${TOPEFT_REPO_ROOT:-}" ]]; then
            fallback_prefixes+=("${TOPEFT_REPO_ROOT}/clib-env")
        fi

        fallback_prefixes+=("${entry_dir}/clib-env")

        local activated=0
        for prefix in "${fallback_prefixes[@]}"; do
            if activate_from_prefix "${prefix}"; then
                activated=1
                break
            fi
        done

        if (( ! activated )); then
            echo "[condor_plotter_entry] ERROR: Unable to activate clib-env; conda not found and no usable prefix discovered." >&2
            return 1
        fi
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
