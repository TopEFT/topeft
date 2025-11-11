#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: condor_plotter_entry.sh PAYLOAD_TARBALL [run_plotter arguments]

Prepare the Condor job sandbox, activate the requested Python environment, and
invoke ./run_plotter.sh with the forwarded arguments.

Environment overrides (set via the Condor "environment" setting):
  PYTHON_BIN            Absolute path to the preferred Python interpreter.
  CONDA_PREFIX          Path to a conda environment to activate.
  VIRTUAL_ENV           Path to a virtualenv to activate.
  PYTHON_VENV_PATH      Alternate variable for a virtualenv/venv path.
  PYTHON_ENV_ACTIVATE   Custom activation script to source before running.
  GLADOS_SETUP_SCRIPT   Setup script (e.g. cmsenv) to source on Glados.
  GLADOS_MODULES        Whitespace-separated list of environment modules to load.
  GLADOS_MODULEPATH     Additional MODULEPATH entry to prepend before loading modules.

The script writes status messages to stdout/stderr so Condor's standard log
collection captures the job history.
USAGE
}

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
    printf '[%s] %s\n' "${timestamp}" "$*"
}

fatal() {
    log "ERROR: $*" >&2
    exit 1
}

ensure_module_command() {
    if type module >/dev/null 2>&1; then
        return 0
    fi
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
    if type module >/dev/null 2>&1; then
        return 0
    fi
    if command -v modulecmd >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

load_modules() {
    local module_list="$1"
    if [[ -z "${module_list}" ]]; then
        return 0
    fi
    if ! ensure_module_command; then
        fatal "Environment modules requested via GLADOS_MODULES but the 'module' command is unavailable."
    fi
    local module_name
    for module_name in ${module_list}; do
        if type module >/dev/null 2>&1; then
            log "Loading module '${module_name}'"
            if ! module load "${module_name}"; then
                fatal "Failed to load module '${module_name}'."
            fi
        else
            log "Loading module '${module_name}' via modulecmd"
            if ! eval "$(modulecmd bash load "${module_name}")"; then
                fatal "modulecmd failed while loading '${module_name}'."
            fi
        fi
    done
}

activate_virtualenv() {
    local venv_path="$1"
    if [[ -z "${venv_path}" ]]; then
        return 0
    fi
    if [[ ! -d "${venv_path}" ]]; then
        fatal "Requested Python virtual environment '${venv_path}' does not exist."
    fi
    if [[ ! -f "${venv_path}/bin/activate" ]]; then
        fatal "Virtual environment at '${venv_path}' is missing the activate script."
    fi
    log "Activating virtual environment '${venv_path}'"
    # shellcheck disable=SC1090
    source "${venv_path}/bin/activate"
}

activate_conda() {
    local conda_prefix="$1"
    if [[ -z "${conda_prefix}" ]]; then
        return 1
    fi
    if [[ ! -d "${conda_prefix}" ]]; then
        fatal "CONDA_PREFIX='${conda_prefix}' does not exist on the worker node."
    fi
    if [[ ! -f "${conda_prefix}/bin/activate" ]]; then
        fatal "Conda environment at '${conda_prefix}' is missing bin/activate."
    fi
    log "Activating conda environment '${conda_prefix}'"
    # shellcheck disable=SC1090
    source "${conda_prefix}/bin/activate"
    return 0
}

source_optional_script() {
    local script_path="$1"
    local description="$2"
    if [[ -z "${script_path}" ]]; then
        return 0
    fi
    if [[ ! -f "${script_path}" ]]; then
        fatal "${description} '${script_path}' was not found."
    fi
    log "Sourcing ${description} '${script_path}'"
    # shellcheck disable=SC1090
    source "${script_path}"
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    *)
        ;;
esac

payload_tar="$1"
shift

if [[ ! -f "${payload_tar}" ]]; then
    fatal "Payload tarball '${payload_tar}' not found in the working directory."
fi

if [[ -d payload ]]; then
    log "Removing existing payload directory"
    rm -rf payload
fi

log "Extracting payload from '${payload_tar}'"
if ! tar -xf "${payload_tar}"; then
    fatal "Failed to extract payload tarball '${payload_tar}'."
fi

if [[ ! -d payload ]]; then
    fatal "Expected a 'payload' directory after extracting '${payload_tar}', but it was not found."
fi

cd payload

source_optional_script "${GLADOS_SETUP_SCRIPT:-}" "Glados setup script"

if [[ -n "${GLADOS_MODULEPATH:-}" ]]; then
    log "Prepending MODULEPATH with '${GLADOS_MODULEPATH}'"
    export MODULEPATH="${GLADOS_MODULEPATH}:${MODULEPATH:-}"
fi

load_modules "${GLADOS_MODULES:-}"

if [[ -n "${PYTHON_ENV_ACTIVATE:-}" ]]; then
    source_optional_script "${PYTHON_ENV_ACTIVATE}" "Python activation script"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    activate_conda "${CONDA_PREFIX}"
elif [[ -n "${PYTHON_VENV_PATH:-}" ]]; then
    activate_virtualenv "${PYTHON_VENV_PATH}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    activate_virtualenv "${VIRTUAL_ENV}"
fi

python_candidate="${PYTHON_BIN:-}"
if [[ -n "${python_candidate}" ]]; then
    if ! command -v "${python_candidate}" >/dev/null 2>&1; then
        fatal "PYTHON_BIN='${python_candidate}' is not executable on the worker node."
    fi
else
    if command -v python3 >/dev/null 2>&1; then
        python_candidate="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        python_candidate="$(command -v python)"
    else
        fatal "No python interpreter found on PATH; set PYTHON_BIN or provide a virtual environment."
    fi
    export PYTHON_BIN="${python_candidate}"
fi

log "Using Python interpreter '${PYTHON_BIN}'"

if [[ ! -x ./run_plotter.sh ]]; then
    fatal "run_plotter.sh was not found in the extracted payload."
fi

if (( $# == 0 )); then
    log "Launching run_plotter.sh (no arguments)"
else
    printf -v launch_args '%q ' "$@"
    launch_args=${launch_args% }
    log "Launching run_plotter.sh ${launch_args}"
fi
set +e
./run_plotter.sh "$@"
status=$?
set -e
log "run_plotter.sh completed with exit code ${status}"
exit ${status}
