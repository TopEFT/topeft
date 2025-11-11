#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PLOTTER="${SCRIPT_DIR}/run_plotter.sh"
DEFAULT_CEPH_ROOT="/users/apiccine/work/correction-lib/topeft"

show_help() {
    cat <<'USAGE'
Usage: submit_plotter_condor.sh [condor options] [run_plotter arguments]

Submit analysis/topeft_run2/run_plotter.sh to the Glados Condor scheduler.
The script validates the plotting arguments using run_plotter.sh --dry-run
before creating a lightweight Condor sandbox.

Condor-specific options:
  --queue N           Number of jobs to queue (default: 1)
  --log-dir PATH      Directory where Condor log/stdout/stderr are written
  --ceph-root PATH    Location of the topeft repository on CephFS
                      (default: /users/apiccine/work/correction-lib/topeft)
  --dry-run           Print the generated job files instead of submitting
  -h, --help          Show this help message and exit

All other options are forwarded directly to run_plotter.sh. The required
run_plotter flags (-f/--input, -o/--output-dir, -y/--year) must be supplied.
USAGE
}

if [[ ! -x "${RUN_PLOTTER}" ]]; then
    echo "Error: run_plotter.sh was not found next to this helper." >&2
    exit 1
fi

queue_count=1
log_dir=""
ceph_root="${DEFAULT_CEPH_ROOT}"
condor_dry_run=0
plotter_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)
            if [[ $# -lt 2 ]]; then
                echo "Error: --queue requires a value" >&2
                exit 1
            fi
            queue_count="$2"
            shift 2
            ;;
        --log-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: --log-dir requires a value" >&2
                exit 1
            fi
            log_dir="$2"
            shift 2
            ;;
        --ceph-root)
            if [[ $# -lt 2 ]]; then
                echo "Error: --ceph-root requires a value" >&2
                exit 1
            fi
            ceph_root="$2"
            shift 2
            ;;
        --dry-run)
            condor_dry_run=1
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
    echo "Error: run_plotter arguments must be provided." >&2
    echo "       Use --help for usage information." >&2
    exit 1
fi

if ! [[ "${queue_count}" =~ ^[0-9]+$ ]] || (( queue_count < 1 )); then
    echo "Error: --queue expects a positive integer value." >&2
    exit 1
fi

original_cwd=$(pwd)

if [[ -z "${log_dir}" ]]; then
    log_dir="${original_cwd}/condor_logs"
fi

log_dir=$(python3 - "${log_dir}" <<'PY'
import os
import sys
path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
)
mkdir -p "${log_dir}"

ceph_root=$(python3 - "${ceph_root}" <<'PY'
import os
import sys
path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
)

if [[ ! -d "${ceph_root}" ]]; then
    echo "Error: The provided CephFS root '${ceph_root}' does not exist." >&2
    exit 1
fi

validation_output=""
if ! validation_output=$("${RUN_PLOTTER}" "${plotter_args[@]}" --dry-run 2>&1); then
    echo "Validation failed while invoking run_plotter.sh:" >&2
    echo "${validation_output}" >&2
    exit 1
fi

command_line=$(awk '/^[[:space:]]+/ { cmd=$0 } END { print cmd }' <<<"${validation_output}")
if [[ -z "${command_line}" ]]; then
    echo "Error: Unable to determine plotter command from run_plotter.sh output." >&2
    echo "Full output:" >&2
    printf '%s\n' "${validation_output}" >&2
    exit 1
fi

printf -v ceph_root_quoted '%q' "${ceph_root}"
printf -v plotter_arg_string ' %q' "${plotter_args[@]}"

work_dir=$(mktemp -d)
cleanup() {
    rm -rf "${work_dir}"
}
trap cleanup EXIT

job_script="${work_dir}/plotter_job.sh"
cat >"${job_script}" <<JOB
#!/usr/bin/env bash
set -euo pipefail

cd ${ceph_root_quoted}
exec ./analysis/topeft_run2/run_plotter.sh${plotter_arg_string}
JOB
chmod +x "${job_script}"

submission_file="${work_dir}/plotter_job.sub"
cat >"${submission_file}" <<SUB
universe                = vanilla
executable              = "${job_script}"
arguments               =
log                     = "${log_dir}/plotter.\$(Cluster).\$(Process).log"
output                  = "${log_dir}/plotter.\$(Cluster).\$(Process).out"
error                   = "${log_dir}/plotter.\$(Cluster).\$(Process).err"
initialdir              = "${ceph_root}"
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
queue ${queue_count}
SUB

if (( condor_dry_run )); then
    echo "Dry-run requested; generated job script and submission file:" >&2
    echo "--- ${job_script} ---"
    cat "${job_script}"
    echo "--- ${submission_file} ---"
    cat "${submission_file}"
    exit 0
fi

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "Error: condor_submit command not found in PATH." >&2
    exit 1
fi

submission_output=""
if ! submission_output=$(condor_submit "${submission_file}" 2>&1); then
    echo "condor_submit failed:" >&2
    echo "${submission_output}" >&2
    exit 1
fi

echo "${submission_output}" >&2
cluster_id=$(awk '/submitted to cluster/ { print $NF }' <<<"${submission_output}" | tail -n1)
cluster_id="${cluster_id%.}"

if [[ -z "${cluster_id}" ]]; then
    echo "Warning: Unable to determine Condor cluster ID from submission output." >&2
    exit 0
fi

echo "${cluster_id}"
