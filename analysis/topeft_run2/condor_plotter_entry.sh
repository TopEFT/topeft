#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

main() {
    unset PYTHONPATH

    local entry_dir="${TOPEFT_ENTRY_DIR:-}"

    local -a condor_ulimit_sources=()
    if [[ -n "${_CONDOR_JOB_IWD:-}" ]]; then
        condor_ulimit_sources+=("_CONDOR_JOB_IWD")
    fi
    if [[ -n "${TOPEFT_CONDOR_ULIMIT:-}" && "${TOPEFT_CONDOR_ULIMIT}" != "0" ]]; then
        condor_ulimit_sources+=("TOPEFT_CONDOR_ULIMIT")
    fi

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

    if (( ${#condor_ulimit_sources[@]} > 0 )); then
        echo "[condor_plotter_entry] Condor ulimit safeguards enabled (sources: ${condor_ulimit_sources[*]})." >&2
        echo "[condor_plotter_entry] Limits before adjustment:" >&2
        ulimit -a >&2

        local core_msg=""
        if core_msg=$(ulimit -S -c 0 2>&1); then
            echo "[condor_plotter_entry] Set core file size (soft) to 0." >&2
        else
            echo "[condor_plotter_entry] Could not adjust core file size (${core_msg}). Current: $(ulimit -c)." >&2
        fi

        adjust_memory_limit() {
            local flag="$1" desc="$2"
            local soft hard
            soft=$(ulimit -S "${flag}")
            hard=$(ulimit -H "${flag}")

            if [[ "${hard}" == "unlimited" ]]; then
                echo "[condor_plotter_entry] ${desc}: leaving unchanged (hard limit is unlimited)." >&2
                return
            fi

            if [[ "${soft}" =~ ^[0-9]+$ && "${hard}" =~ ^[0-9]+$ && ${soft} -le ${hard} ]]; then
                echo "[condor_plotter_entry] ${desc}: soft limit already ${soft} (hard ${hard})." >&2
                return
            fi

            local set_msg=""
            if set_msg=$(ulimit -S "${flag}" "${hard}" 2>&1); then
                echo "[condor_plotter_entry] ${desc}: set soft limit to hard cap ${hard}." >&2
            else
                echo "[condor_plotter_entry] ${desc}: unable to set limit (${set_msg}). Current soft ${soft}, hard ${hard}." >&2
            fi
        }

        adjust_memory_limit -v "Virtual memory"
        adjust_memory_limit -m "Resident set size"

        echo "[condor_plotter_entry] Limits before run_plotter.sh:" >&2
        ulimit -a >&2
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
