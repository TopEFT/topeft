#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PLOTTER="${SCRIPT_DIR}/run_plotter.sh"
DEFAULT_CEPH_ROOT="/users/apiccine/work/correction-lib/topeft"
DEFAULT_CONDA_PREFIX="${CONDA_PREFIX:-}"

show_help() {
    cat <<'USAGE'
Usage: submit_plotter_condor.sh [condor options] [run_plotter arguments]

Submit analysis/topeft_run2/run_plotter.sh to the Glados Condor scheduler.
The script validates the plotting arguments using run_plotter.sh --dry-run
before creating a lightweight Condor sandbox.

Condor-specific options:
  --queue N                 Number of jobs to queue (default: 1)
  --request-cpus N          Number of CPU cores to request from Condor
  --request-memory VALUE    Memory request (e.g. 4GB, 8192MB)
  --log-dir PATH            Directory where Condor log/stdout/stderr are written
  --sandbox PATH            Additional file or directory to transfer with the job
  --ceph-root PATH          Location of the topeft repository on CephFS
                            (default: /users/apiccine/work/correction-lib/topeft)
  --conda-prefix PATH       Prefix path of the conda installation accessible to workers
                            (default: value of the CONDA_PREFIX environment variable)
  --dry-run                 Print the generated job files instead of submitting
  -h, --help                Show this help message and exit

All other options are forwarded directly to run_plotter.sh. The required
run_plotter flags (-f/--input, -o/--output-dir, -y/--year) must be supplied.
USAGE
}

if [[ ! -x "${RUN_PLOTTER}" ]]; then
    echo "Error: run_plotter.sh was not found next to this helper." >&2
    exit 1
fi

queue_count=1
request_cpus=""
request_memory=""
log_dir=""
sandbox_paths=()
ceph_root="${DEFAULT_CEPH_ROOT}"
conda_prefix="${DEFAULT_CONDA_PREFIX}"
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
        --request-cpus)
            if [[ $# -lt 2 ]]; then
                echo "Error: --request-cpus requires a value" >&2
                exit 1
            fi
            request_cpus="$2"
            shift 2
            ;;
        --request-memory)
            if [[ $# -lt 2 ]]; then
                echo "Error: --request-memory requires a value" >&2
                exit 1
            fi
            request_memory="$2"
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
        --sandbox)
            if [[ $# -lt 2 ]]; then
                echo "Error: --sandbox requires a value" >&2
                exit 1
            fi
            sandbox_paths+=("$2")
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
        --conda-prefix)
            if [[ $# -lt 2 ]]; then
                echo "Error: --conda-prefix requires a value" >&2
                exit 1
            fi
            conda_prefix="$2"
            shift 2
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

if [[ -z "${conda_prefix}" ]]; then
    echo "Error: --conda-prefix must be provided or CONDA_PREFIX must be set in the submit environment." >&2
    exit 1
fi

conda_prefix=$(python3 - "${conda_prefix}" <<'PY'
import os
import sys
path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
)

sandbox_inputs=()
declare -a sandbox_relative_paths=()
declare -A sandbox_dest_seen=()
for sandbox_path in "${sandbox_paths[@]}"; do
    mapfile -t sandbox_info < <(python3 - "${sandbox_path}" "${original_cwd}" <<'PY'
import os
import sys

path = sys.argv[1]
base = sys.argv[2]

if path.startswith('~'):
    expanded = os.path.expanduser(path)
elif os.path.isabs(path):
    expanded = path
else:
    expanded = os.path.join(base, path)

resolved = os.path.abspath(expanded)
if not os.path.exists(resolved):
    print(f"Error: sandbox path '{path}' does not exist.", file=sys.stderr)
    sys.exit(1)

if os.path.isabs(path):
    relative = os.path.basename(path)
else:
    relative = os.path.normpath(path)
    if relative.startswith('..') or '..' in relative.split('/'):  # prevent escaping the repo root
        print(
            f"Error: --sandbox path '{path}' must not reference parent directories (.. components are unsupported).",
            file=sys.stderr,
        )
        sys.exit(1)
    if relative in ('', '.'):
        relative = os.path.basename(resolved)

relative = relative.strip('/')
if not relative:
    relative = os.path.basename(resolved)

print(resolved)
print(relative)
PY
    )

    resolved_sandbox="${sandbox_info[0]}"
    relative_target="${sandbox_info[1]}"

    if [[ -z "${relative_target}" ]]; then
        echo "Error: Unable to derive sandbox destination for '${sandbox_path}'." >&2
        exit 1
    fi

    if [[ -n "${sandbox_dest_seen["${relative_target}"]+x}" ]]; then
        echo "Error: Multiple --sandbox entries map to '${relative_target}'." >&2
        exit 1
    fi
    sandbox_dest_seen["${relative_target}"]=1

    sandbox_inputs+=("${resolved_sandbox}")
    sandbox_relative_paths+=("${relative_target}")
done

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
printf -v conda_prefix_quoted '%q' "${conda_prefix}"
printf -v plotter_arg_string ' %q' "${plotter_args[@]}"

entry_helper="${ceph_root}/analysis/topeft_run2/condor_plotter_entry.sh"
printf -v entry_helper_quoted '%q' "${entry_helper}"

work_dir=$(mktemp -d)
cleanup() {
    rm -rf "${work_dir}"
}
trap cleanup EXIT

sandbox_archive=""
sandbox_archive_name=""
if (( ${#sandbox_inputs[@]} )); then
    sandbox_staging="${work_dir}/sandbox_stage"
    mkdir -p "${sandbox_staging}"

    for idx in "${!sandbox_inputs[@]}"; do
        source_path="${sandbox_inputs[$idx]}"
        relative_target="${sandbox_relative_paths[$idx]}"
        destination_path="${sandbox_staging}/${relative_target}"

        if [[ -d "${source_path}" ]]; then
            mkdir -p "${destination_path}"
            cp -a "${source_path}/." "${destination_path}/"
        else
            mkdir -p "$(dirname -- "${destination_path}")"
            cp -a "${source_path}" "${destination_path}"
        fi
    done

    sandbox_archive="${work_dir}/plotter_sandbox.tar.gz"
    (
        cd "${sandbox_staging}"
        tar -czf "${sandbox_archive}" .
    )
    sandbox_archive_name=$(basename -- "${sandbox_archive}")
fi

sandbox_extract_block=$'sandbox_cleanup_active=0\n'
sandbox_extract_block+=$'cleanup_sandbox() {\n    return 0\n}\n'
if [[ -n "${sandbox_archive_name}" ]]; then
    printf -v sandbox_archive_name_quoted '%q' "${sandbox_archive_name}"
    sandbox_extract_format=$(cat <<'EOS'
sandbox_archive=%s
sandbox_archive_path="\${scratch_dir}/\${sandbox_archive}"
sandbox_extract_dir="\${scratch_dir}/plotter_sandbox_extract"
sandbox_backup_root="\${scratch_dir}/plotter_sandbox_backup"
declare -a sandbox_backups=()
declare -a sandbox_new_files=()
declare -a sandbox_new_dirs=()

cleanup_sandbox() {
    local entry dest backup
    for entry in "\${sandbox_new_files[@]}"; do
        if [[ -e "\${entry}" || -L "\${entry}" ]]; then
            rm -rf "\${entry}"
        fi
    done
    for entry in "\${sandbox_backups[@]}"; do
        dest="\${entry%%%%|*}"
        backup="\${entry#*|}"
        if [[ -e "\${dest}" || -L "\${dest}" ]]; then
            rm -rf "\${dest}"
        fi
        mkdir -p "$(dirname -- "\${dest}")"
        mv "\${backup}" "\${dest}"
    done
    for (( sandbox_dir_index=\${#sandbox_new_dirs[@]}-1; sandbox_dir_index>=0; sandbox_dir_index-- )); do
        dest="\${sandbox_new_dirs[sandbox_dir_index]}"
        if [[ -d "\${dest}" ]]; then
            rmdir "\${dest}" 2>/dev/null || true
        fi
    done
    rm -rf "\${sandbox_backup_root}" "\${sandbox_extract_dir}"
}

if [[ -f "\${sandbox_archive_path}" ]]; then
    sandbox_cleanup_active=1
    rm -rf "\${sandbox_extract_dir}" "\${sandbox_backup_root}"
    mkdir -p "\${sandbox_extract_dir}" "\${sandbox_backup_root}"
    if tar -xzf "\${sandbox_archive_path}" -C "\${sandbox_extract_dir}"; then
        while IFS= read -r -d '' dir_entry; do
            rel="\${dir_entry#./}"
            src="\${sandbox_extract_dir}/\${rel}"
            dest="${ceph_root}/\${rel}"
            if [[ -d "\${dest}" ]]; then
                continue
            fi
            if [[ -e "\${dest}" || -L "\${dest}" ]]; then
                backup="\${sandbox_backup_root}/\${rel}"
                mkdir -p "$(dirname -- "\${backup}")"
                mv "\${dest}" "\${backup}"
                sandbox_backups+=("\${dest}|\${backup}")
            fi
            mkdir -p "\${dest}"
            sandbox_new_dirs+=("\${dest}")
        done < <(cd "\${sandbox_extract_dir}" && find . -mindepth 1 -type d -print0 | sort -z)

        while IFS= read -r -d '' file_entry; do
            rel="\${file_entry#./}"
            src="\${sandbox_extract_dir}/\${rel}"
            dest="${ceph_root}/\${rel}"
            mkdir -p "$(dirname -- "\${dest}")"
            if [[ -e "\${dest}" || -L "\${dest}" ]]; then
                backup="\${sandbox_backup_root}/\${rel}"
                mkdir -p "$(dirname -- "\${backup}")"
                mv "\${dest}" "\${backup}"
                sandbox_backups+=("\${dest}|\${backup}")
            else
                sandbox_new_files+=("\${dest}")
            fi
            cp -a "\${src}" "\${dest}"
        done < <(cd "\${sandbox_extract_dir}" && find . -mindepth 1 ! -type d -print0 | sort -z)

        trap 'cleanup_sandbox' EXIT INT TERM
    else
        echo "Error: Failed to extract sandbox archive '\${sandbox_archive}'." >&2
        exit 1
    fi
else
    echo "Warning: sandbox archive '\${sandbox_archive}' was not transferred." >&2
fi

EOS
    )
    printf -v sandbox_extract_rendered "${sandbox_extract_format}" "${sandbox_archive_name_quoted}"
    sandbox_extract_block+="${sandbox_extract_rendered}"
    sandbox_extract_block+=$'\n'
fi

job_script="${work_dir}/plotter_job.sh"
cat >"${job_script}" <<JOB
#!/usr/bin/env bash
set -euo pipefail

ceph_root=${ceph_root_quoted}
entry_helper=${entry_helper_quoted}
conda_prefix=${conda_prefix_quoted}
scratch_dir=\${_CONDOR_SCRATCH_DIR:-\$(pwd)}
${sandbox_extract_block}
if [[ ! -x "${entry_helper}" ]]; then
    echo "Error: Condor entry helper '${entry_helper}' is missing or not executable." >&2
    exit 1
fi

if [[ ! -d "\${scratch_dir}" ]]; then
    echo "Error: Condor scratch directory '\${scratch_dir}' is missing." >&2
    exit 1
fi

if ! cd "\${scratch_dir}"; then
    echo "Error: Failed to enter Condor scratch directory '\${scratch_dir}'." >&2
    exit 1
fi

set +e
"${entry_helper}" --ceph-root "\${ceph_root}" --conda-prefix "\${conda_prefix}"${plotter_arg_string}
status=\$?
set -e

if (( sandbox_cleanup_active )); then
    trap - EXIT INT TERM
    cleanup_sandbox
fi

exit "\${status}"
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
SUB

transfer_inputs=()
if [[ -n "${sandbox_archive}" ]]; then
    transfer_inputs+=("${sandbox_archive}")
fi

transfer_input_files=$(python3 - "${transfer_inputs[@]}" <<'PY'
import sys

def quote(token: str) -> str:
    if any(ch in token for ch in (' ', ',', '"')):
        return '"' + token.replace('"', '\\"') + '"'
    return token

print(','.join(quote(arg) for arg in sys.argv[1:]))
PY
)

if [[ -n "${request_cpus}" ]]; then
    echo "request_cpus           = ${request_cpus}" >>"${submission_file}"
fi
if [[ -n "${request_memory}" ]]; then
    echo "request_memory         = ${request_memory}" >>"${submission_file}"
fi
if [[ -n "${transfer_input_files}" ]]; then
    echo "transfer_input_files   = ${transfer_input_files}" >>"${submission_file}"
fi

cat >>"${submission_file}" <<SUB
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
