#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PLOTTER="${SCRIPT_DIR}/run_plotter.sh"
ENTRY_SCRIPT="${SCRIPT_DIR}/condor_plotter_entry.sh"
DEFAULT_CEPH_ROOT="/users/apiccine/work/correction-lib/topeft"

show_help() {
    cat <<'USAGE'
Usage: submit_plotter_condor.sh [options] -- [run_plotter arguments]

Submit run_plotter.sh jobs to HTCondor with minimal boilerplate. The script
validates plotting options by invoking run_plotter.sh --dry-run locally before
creating a temporary submission file.

Condor options:
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

All other arguments are forwarded directly to run_plotter.sh.
USAGE
}

if [[ ! -x "${RUN_PLOTTER}" ]]; then
    echo "Error: run_plotter.sh not found next to this helper." >&2
    exit 1
fi

if [[ ! -x "${ENTRY_SCRIPT}" ]]; then
    echo "Error: condor_plotter_entry.sh not found next to this helper." >&2
    exit 1
fi

queue_count=1
log_dir=""
ceph_root="${DEFAULT_CEPH_ROOT}"
conda_prefix=""
dry_run=0
request_cpus=""
request_memory=""
plotter_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)
            if [[ $# -lt 2 ]]; then
                echo "Error: --queue requires a value." >&2
                exit 1
            fi
            queue_count="$2"
            shift 2
            ;;
        --log-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: --log-dir requires a value." >&2
                exit 1
            fi
            log_dir="$2"
            shift 2
            ;;
        --ceph-root)
            if [[ $# -lt 2 ]]; then
                echo "Error: --ceph-root requires a value." >&2
                exit 1
            fi
            ceph_root="$2"
            shift 2
            ;;
        --conda-prefix)
            if [[ $# -lt 2 ]]; then
                echo "Error: --conda-prefix requires a value." >&2
                exit 1
            fi
            conda_prefix="$2"
            shift 2
            ;;
        --request-cpus)
            if [[ $# -lt 2 ]]; then
                echo "Error: --request-cpus requires a value." >&2
                exit 1
            fi
            request_cpus="$2"
            shift 2
            ;;
        --request-memory)
            if [[ $# -lt 2 ]]; then
                echo "Error: --request-memory requires a value." >&2
                exit 1
            fi
            request_memory="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            plotter_args+=("$@")
            break
            ;;
        *)
            plotter_args+=("$1")
            shift
            ;;
    esac
done

if (( ${#plotter_args[@]} == 0 )); then
    echo "Error: run_plotter.sh arguments are required. Use --help for details." >&2
    exit 1
fi

if ! [[ "${queue_count}" =~ ^[0-9]+$ ]] || (( queue_count < 1 )); then
    echo "Error: --queue expects a positive integer." >&2
    exit 1
fi

if [[ -n "${request_cpus}" ]]; then
    if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus < 1 )); then
        echo "Error: --request-cpus expects a positive integer." >&2
        exit 1
    fi
fi

if [[ -n "${request_memory}" ]]; then
    if [[ "${request_memory}" =~ ^[[:space:]]*$ ]]; then
        echo "Error: --request-memory expects a non-empty string." >&2
        exit 1
    fi
fi

if [[ -z "${log_dir}" ]]; then
    log_dir="${PWD}/condor_logs"
fi

log_dir=$(python3 - "${log_dir}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)
mkdir -p "${log_dir}"

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

analysis_dir="${ceph_root}/analysis/topeft_run2"
if [[ ! -d "${analysis_dir}" ]]; then
    echo "Error: '${analysis_dir}' does not exist; check --ceph-root." >&2
    exit 1
fi

entry_on_ceph="${analysis_dir}/condor_plotter_entry.sh"
if [[ ! -f "${entry_on_ceph}" ]]; then
    echo "Error: '${entry_on_ceph}' was not found." >&2
    exit 1
fi

if ! validation_output=$("${RUN_PLOTTER}" "${plotter_args[@]}" --dry-run 2>&1); then
    echo "Error: run_plotter.sh validation failed:" >&2
    printf '%s\n' "${validation_output}" >&2
    exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "${tmp_dir}"' EXIT

submit_file="${tmp_dir}/plotter_job.sub"
staged_entry="${tmp_dir}/$(basename "${ENTRY_SCRIPT}")"

cp "${ENTRY_SCRIPT}" "${staged_entry}"
chmod +x "${staged_entry}"

repo_root=$(python3 - "${analysis_dir}" <<'PY'
import os
import sys
print(os.path.abspath(os.path.join(sys.argv[1], os.pardir)))
PY
)

environment_entries=(
    "TOPEFT_REPO_ROOT=${repo_root}"
    "TOPEFT_ENTRY_DIR=${analysis_dir}"
)

if [[ -n "${conda_prefix}" ]]; then
    environment_entries+=("TOPEFT_CONDA_PREFIX=${conda_prefix}")
fi

environment_string=""
for entry in "${environment_entries[@]}"; do
    if [[ -n "${environment_string}" ]]; then
        environment_string+=";"
    fi
    environment_string+="${entry}"
done

printf -v arg_string ' %q' "${plotter_args[@]}"
arg_string="${arg_string# }"

{
cat <<EOF
universe                = vanilla
executable              = "${staged_entry}"
arguments               = ${arg_string}
initialdir              = ${analysis_dir}
log                     = ${log_dir}/plotter.\$(Cluster).\$(Process).log
output                  = ${log_dir}/plotter.\$(Cluster).\$(Process).out
error                   = ${log_dir}/plotter.\$(Cluster).\$(Process).err
getenv                  = True
should_transfer_files   = YES
transfer_executable     = True
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
    echo "--- ${submit_file} ---"
    cat "${submit_file}"
    exit 0
fi

condor_submit "${submit_file}"
