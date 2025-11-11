#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PLOTTER="${SCRIPT_DIR}/run_plotter.sh"
PLOTTER_SCRIPT="${SCRIPT_DIR}/make_cr_and_sr_plots.py"

show_help() {
    cat <<'USAGE'
Usage: submit_plotter_condor.sh [condor options] -- [run_plotter arguments]

Submit make_cr_and_sr_plots.py to the Glados Condor scheduler using the same
command line produced by run_plotter.sh.

Condor-specific options:
  --queue N                 Number of jobs to queue (default: 1)
  --request-cpus N          Number of CPU cores to request
  --request-memory VALUE    Memory request (e.g. 4GB, 8192MB)
  --log-dir PATH            Directory for Condor log/stdout/stderr files
  --sandbox PATH            Additional file or directory to include in job sandbox
  --dry-run                 Print the generated submission file and exit
  -h, --help                Show this help message and exit

All other arguments are forwarded to run_plotter.sh. At least the required
run_plotter arguments (-f/--input, -o/--output-dir, -y/--year) must be provided.
USAGE
}

if [[ ! -x "${RUN_PLOTTER}" ]]; then
    echo "Error: run_plotter.sh was not found next to this helper." >&2
    exit 1
fi
if [[ ! -f "${PLOTTER_SCRIPT}" ]]; then
    echo "Error: make_cr_and_sr_plots.py was not found next to this helper." >&2
    exit 1
fi

queue_count=1
request_cpus=""
request_memory=""
log_dir=""
sandbox_path=""
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
            sandbox_path="$2"
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
    echo "Error: run_plotter arguments must be provided after Condor options." >&2
    echo "       Use --help for usage information." >&2
    exit 1
fi

# Normalize the log directory before changing directories later in the script.
if [[ -n "${log_dir}" ]]; then
    log_dir=$(python3 - "${log_dir}" <<'PY'
import os
import sys

path = os.path.expanduser(sys.argv[1])
print(os.path.abspath(path))
PY
    )
fi

# Validate inputs and capture the command run_plotter.sh would execute.
validation_output=""
if ! validation_output=$("${RUN_PLOTTER}" "${plotter_args[@]}" --dry-run 2>&1); then
    echo "Validation failed while invoking run_plotter.sh:" >&2
    echo "${validation_output}" >&2
    exit 1
fi

command_line=$(awk '/^  / { cmd=$0 } END { print cmd }' <<< "${validation_output}")
if [[ -z "${command_line}" ]]; then
    echo "Error: Unable to determine plotter command from run_plotter.sh output." >&2
    echo "Full output:" >&2
    printf '%s\n' "${validation_output}" >&2
    exit 1
fi

# Convert the printed command into an array.
read -r -a plotter_cmd <<< "${command_line}"
if (( ${#plotter_cmd[@]} < 2 )); then
    echo "Error: Parsed plotter command is unexpectedly short." >&2
    exit 1
fi

# Extract the requested output directory from the command.
original_output_dir=""
for ((i=0; i<${#plotter_cmd[@]}; ++i)); do
    if [[ "${plotter_cmd[$i]}" == "-o" && $((i+1)) -lt ${#plotter_cmd[@]} ]]; then
        original_output_dir="${plotter_cmd[$((i+1))]}"
        break
    fi
done
if [[ -z "${original_output_dir}" ]]; then
    echo "Error: Unable to locate output directory in plotter command." >&2
    exit 1
fi

output_basename=$(basename -- "${original_output_dir}")
job_output_dir="job_outputs/${output_basename}"
condor_output_dir="payload/${job_output_dir}"

# Rewrite command to operate within the job sandbox.
plotter_cmd[1]="./make_cr_and_sr_plots.py"
for ((i=0; i<${#plotter_cmd[@]}; ++i)); do
    if [[ "${plotter_cmd[$i]}" == "-o" && $((i+1)) -lt ${#plotter_cmd[@]} ]]; then
        plotter_cmd[$((i+1))]="${job_output_dir}"
        break
    fi
done

# Build a staging directory with the required payload.
temp_root=$(mktemp -d)
cleanup() {
    rm -rf "${temp_root}"
}
trap cleanup EXIT

payload_dir="${temp_root}/payload"
mkdir -p "${payload_dir}/job_outputs"
cp "${RUN_PLOTTER}" "${payload_dir}/run_plotter.sh"
cp "${PLOTTER_SCRIPT}" "${payload_dir}/make_cr_and_sr_plots.py"

if [[ -n "${sandbox_path}" ]]; then
    if [[ ! -e "${sandbox_path}" ]]; then
        echo "Error: sandbox path '${sandbox_path}' does not exist." >&2
        exit 1
    fi
    sandbox_dest="${payload_dir}/sandbox"
    mkdir -p "${sandbox_dest}"
    if [[ -d "${sandbox_path}" ]]; then
        (cd "${sandbox_path}" && tar -cf - .) | (cd "${sandbox_dest}" && tar -xf -)
    else
        cp "${sandbox_path}" "${sandbox_dest}/"
    fi
fi

metadata_file="${payload_dir}/job_metadata.txt"
{
    echo "Original command: ${command_line}";
    echo "Adjusted command: ${plotter_cmd[*]}";
    echo "Original output directory: ${original_output_dir}";
    echo "Job output directory: ${job_output_dir}";
} > "${metadata_file}"

wrapper_script="${temp_root}/run_condor_plotter_job.sh"
cat <<'WRAPPER' > "${wrapper_script}"
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PAYLOAD_TARBALL [COMMAND ...]" >&2
    exit 1
fi
payload_tar="$1"
shift
if [[ ! -f "${payload_tar}" ]]; then
    echo "Error: payload tarball '${payload_tar}' not found in working directory." >&2
    exit 1
fi
tar -xf "${payload_tar}"
cd payload
"$@"
WRAPPER
chmod +x "${wrapper_script}"

payload_tar="${temp_root}/plotter_payload.tar.gz"
(
    cd "${temp_root}"
    tar -czf "${payload_tar}" payload
)
payload_tar_name=$(basename -- "${payload_tar}")

# Prepare Condor submission file.
submit_file="${temp_root}/submit_plotter.sub"
mkdir -p "${log_dir:-.}"

executable="$(basename -- "${wrapper_script}")"
arguments=()
arguments+=("${payload_tar_name}")
for arg in "${plotter_cmd[@]}"; do
    arguments+=("${arg}")
done

# Build arguments string with Condor-safe escaping.
build_arguments_string() {
    python3 - "$@" <<'PY'
import sys
def escape(token: str) -> str:
    return '"' + token.replace('\\', '\\\\').replace('"', '\\"') + '"'

print(' '.join(escape(arg) for arg in sys.argv[1:]))
PY
}

arguments_string=$(build_arguments_string "${arguments[@]}")

transfer_output_remaps="${condor_output_dir}=${original_output_dir}"

timestamp=$(date +%Y%m%d_%H%M%S)

{
    echo "universe        = vanilla"
    echo "executable      = ${executable}"
    echo "arguments       = ${arguments_string}"
    echo "requirements    = (TARGET.OpSysAndVer == \"CentOS7\")"
    echo "should_transfer_files = YES"
    echo "when_to_transfer_output = ON_EXIT"
    echo "transfer_input_files = ${executable},${payload_tar_name}"
    echo "transfer_output_files = ${condor_output_dir}"
    echo "transfer_output_remaps = ${transfer_output_remaps}"
    if [[ -n "${request_cpus}" ]]; then
        echo "request_cpus    = ${request_cpus}"
    fi
    if [[ -n "${request_memory}" ]]; then
        echo "request_memory  = ${request_memory}"
    fi
    if [[ -n "${log_dir}" ]]; then
        mkdir -p "${log_dir}"
        echo "log             = ${log_dir}/plotter_${timestamp}.log"
        echo "output          = ${log_dir}/plotter_${timestamp}.out"
        echo "error           = ${log_dir}/plotter_${timestamp}.err"
    else
        echo "log             = plotter.log"
        echo "output          = plotter.out"
        echo "error           = plotter.err"
    fi
    echo "queue ${queue_count}"
} > "${submit_file}"

if (( condor_dry_run )); then
    echo "-- Condor submission file (${submit_file}) --"
    cat "${submit_file}"
    echo "-- End submission file --"
    exit 0
fi

submit_output=""
if ! submit_output=$(cd "${temp_root}" && condor_submit "${submit_file}" 2>&1); then
    echo "condor_submit failed:" >&2
    echo "${submit_output}" >&2
    exit 1
fi

echo "${submit_output}"
