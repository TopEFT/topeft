#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PLOTTER="${SCRIPT_DIR}/run_plotter.sh"
ENTRY_SCRIPT="${SCRIPT_DIR}/condor_plotter_entry.sh"
DEFAULT_CEPH_ROOT="/users/apiccine/work/correction-lib/topeft"

show_help() {
    cat <<'USAGE'
Usage: submit_plotter_condor.sh [condor options] [run_plotter arguments]

Submit run_plotter.sh jobs to HTCondor with minimal boilerplate. The script
validates plotting options by invoking run_plotter.sh --dry-run locally before
creating a submission file that spools the worker wrapper.

Condor options (provide these anywhere before an optional "--" delimiter):
  --queue N        Number of job instances to submit (default: 1)
  --log-dir PATH   Directory for Condor log, output, and error files
                   (default: ./condor_logs)
  --ceph-root DIR  Location of the topeft repository on CephFS
                   (default: /users/apiccine/work/correction-lib/topeft)
  --request-cpus N Request this many CPU cores (must be a positive integer)
  --request-memory SIZE
                   Request this amount of memory (HTCondor size expression)
  --conda-prefix DIR
                   Location of the clib-env Conda environment on the worker
                   nodes. When provided, TOPEFT_CONDA_PREFIX is exported to
                   the entry script so it can activate the environment without
                   relying on a global conda command.
  --dry-run        Print the generated submission file instead of calling condor_submit
  -h, --help       Show this help message and exit

All other arguments are forwarded directly to run_plotter.sh. The legacy "--"
delimiter remains supported when you need to force everything that follows to
be treated as plotting arguments; in that case, Condor flags like --dry-run
must be placed before the delimiter to avoid being mis-parsed as plotting
options.
In particular, plotting switches such as --channel-output understand the same
merged/split/both values as the Python CLI along with the new -njets variants
that preserve the per-njet bins defined in cr_sr_plots_metadata.yml.
USAGE
}

main() {
    if [[ ! -x "${RUN_PLOTTER}" ]]; then
        echo "Error: run_plotter.sh not found next to this helper." >&2
        return 1
    fi

    if [[ ! -x "${ENTRY_SCRIPT}" ]]; then
        echo "Error: condor_plotter_entry.sh not found next to this helper." >&2
        return 1
    fi

    local queue_count=1
    local log_dir=""
    local ceph_root="${DEFAULT_CEPH_ROOT}"
    local conda_prefix=""
    local dry_run=0
    local request_cpus=""
    local request_memory=""
    local -a plotter_args=()
    local parsing_condor=1

    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            parsing_condor=0
            shift
            continue
        fi

        if (( parsing_condor )); then
            case "$1" in
                --queue)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --queue requires a value." >&2
                        return 1
                    fi
                    queue_count="$2"
                    shift 2
                    continue
                    ;;
                --log-dir)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --log-dir requires a value." >&2
                        return 1
                    fi
                    log_dir="$2"
                    shift 2
                    continue
                    ;;
                --ceph-root)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --ceph-root requires a value." >&2
                        return 1
                    fi
                    ceph_root="$2"
                    shift 2
                    continue
                    ;;
                --conda-prefix)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --conda-prefix requires a value." >&2
                        return 1
                    fi
                    conda_prefix="$2"
                    shift 2
                    continue
                    ;;
                --request-cpus)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --request-cpus requires a value." >&2
                        return 1
                    fi
                    request_cpus="$2"
                    shift 2
                    continue
                    ;;
                --request-memory)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --request-memory requires a value." >&2
                        return 1
                    fi
                    request_memory="$2"
                    shift 2
                    continue
                    ;;
                --dry-run)
                    dry_run=1
                    shift
                    continue
                    ;;
                -h|--help)
                    show_help
                    return 0
                    ;;
            esac
        fi

        plotter_args+=("$1")
        shift
    done

    for arg in "${plotter_args[@]}"; do
        if [[ "${arg}" == "--dry-run" ]]; then
            echo "Error: --dry-run is a Condor helper flag; place Condor options before" >&2
            echo "the plotting arguments (or omit the -- delimiter) so it is parsed" >&2
            echo "correctly." >&2
            return 1
        fi
    done

    if (( ${#plotter_args[@]} == 0 )); then
        echo "Error: run_plotter.sh arguments are required. Use --help for details." >&2
        return 1
    fi

    local -a allowed_channel_outputs=(
        merged
        split
        both
        merged-njets
        split-njets
        both-njets
    )
    local idx=0
    while (( idx < ${#plotter_args[@]} )); do
        if [[ "${plotter_args[idx]}" == "--channel-output" ]]; then
            if (( idx + 1 >= ${#plotter_args[@]} )); then
                echo "Error: --channel-output requires a value." >&2
                return 1
            fi
            local candidate="${plotter_args[idx + 1],,}"
            local valid=0
            for mode in "${allowed_channel_outputs[@]}"; do
                if [[ "${candidate}" == "${mode}" ]]; then
                    valid=1
                    break
                fi
            done
            if (( ! valid )); then
                echo "Error: --channel-output expects one of: ${allowed_channel_outputs[*]}" >&2
                return 1
            fi
            ((idx+=2))
            continue
        fi
        ((idx++))
    done

    if ! [[ "${queue_count}" =~ ^[0-9]+$ ]] || (( queue_count < 1 )); then
        echo "Error: --queue expects a positive integer." >&2
        return 1
    fi

    if [[ -n "${request_cpus}" ]]; then
        if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus < 1 )); then
            echo "Error: --request-cpus expects a positive integer." >&2
            return 1
        fi
    fi

    if [[ -n "${request_memory}" ]]; then
        if [[ "${request_memory}" =~ ^[[:space:]]*$ ]]; then
            echo "Error: --request-memory expects a non-empty string." >&2
            return 1
        fi
    fi

    ceph_root=$(python3 - "${ceph_root}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)

    if [[ -n "${conda_prefix}" ]]; then
        conda_prefix=$(python3 - "${conda_prefix}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)
    fi

    local analysis_dir="${ceph_root}/analysis/topeft_run2"
    if [[ ! -d "${analysis_dir}" ]]; then
        echo "Error: '${analysis_dir}' does not exist; check --ceph-root." >&2
        return 1
    fi

    if [[ -z "${log_dir}" ]]; then
        log_dir="${analysis_dir}/condor_logs"
    fi

    log_dir=$(python3 - "${log_dir}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)
    mkdir -p "${log_dir}"

    local entry_dir="${analysis_dir}"

    local validation_output=""
    if ! validation_output=$("${RUN_PLOTTER}" "${plotter_args[@]}" --dry-run 2>&1); then
        echo "Error: run_plotter.sh validation failed:" >&2
        printf '%s\n' "${validation_output}" >&2
        return 1
    fi

    local submit_dir="${SCRIPT_DIR}/condor_submissions"
    mkdir -p "${submit_dir}"

    local submit_file="${submit_dir}/plotter_job_$(date +%Y%m%d_%H%M%S)_${RANDOM}.sub"

    local repo_root
    repo_root=$(python3 - "${analysis_dir}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.join(sys.argv[1], os.pardir)))
PY
)

    local -a environment_entries=(
        "TOPEFT_REPO_ROOT=${repo_root}"
        "TOPEFT_ENTRY_DIR=${entry_dir}"
    )

    if [[ -n "${conda_prefix}" ]]; then
        environment_entries+=("TOPEFT_CONDA_PREFIX=${conda_prefix}")
    fi

    local environment_string=""
    local entry
    for entry in "${environment_entries[@]}"; do
        if [[ -n "${environment_string}" ]]; then
            environment_string+=";"
        fi
        environment_string+="${entry}"
    done

    local arg_string=""
    printf -v arg_string ' %q' "${plotter_args[@]}"
    arg_string="${arg_string# }"

    local arguments_line="${arg_string}"

    local executable_path="${entry_dir}/condor_plotter_entry.sh"
    local should_transfer_files="NO"

    {
cat <<EOF
universe                = vanilla
executable              = ${executable_path}
arguments               = ${arguments_line}
initialdir              = ${entry_dir}
log                     = ${log_dir}/plotter.\$(Cluster).\$(Process).log
output                  = ${log_dir}/plotter.\$(Cluster).\$(Process).out
error                   = ${log_dir}/plotter.\$(Cluster).\$(Process).err
getenv                  = True
should_transfer_files   = ${should_transfer_files}
environment              = "${environment_string}"
EOF
    if [[ -n "${request_cpus}" ]]; then
        printf 'request_cpus            = %s\n' "${request_cpus}"
    fi
    if [[ -n "${request_memory}" ]]; then
        printf 'request_memory          = %s\n' "${request_memory}"
    fi
    cat <<EOF
queue ${queue_count}
EOF
    } > "${submit_file}"

    if (( dry_run )); then
        echo "run_plotter.sh validation output:" >&2
        printf '%s\n' "${validation_output}" >&2
        echo "Submit file: ${submit_file}" >&2
        echo "entry_dir: ${entry_dir}" >&2
        echo "initialdir: ${entry_dir}" >&2
        echo "Executable path: ${executable_path}" >&2
        echo "should_transfer_files: ${should_transfer_files}" >&2
        echo "--- ${submit_file} ---"
        cat "${submit_file}"
        return 0
    fi

    (
        cd "${SCRIPT_DIR}"
        condor_submit "${submit_file}"
    )

    return 0
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
