#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLOTTER_SCRIPT="${SCRIPT_DIR}/make_cr_and_sr_plots.py"
PYTHON_BIN=${PYTHON_BIN:-${PYTHON:-python}}

show_help() {
    cat <<'USAGE'
Usage: run_plotter.sh -f PICKLE -o OUTPUT_DIR [options]

Wrapper around make_cr_and_sr_plots.py with filename-based region detection.
Consult the "CR/SR plotting CLI quickstart" section of analysis/topeft_run2/README.md
for more workflow examples.

Required arguments:
  -f, --input PATH          Input histogram pickle (e.g. histos/plotsCR_Run2.pkl.gz)
  -o, --output-dir PATH     Directory where plots will be written

Optional arguments:
  -n, --name NAME           Name for the output directory (forwarded with -n)
  -y, --year YEAR [YEAR ...]
                           One or more year tokens forwarded to the plotter
  -t, --timestamp           Append a timestamp to the output directory name
  -s, --skip-syst           Skip systematic error bands
  -u, --unit-norm           Enable unit-normalized plotting
      --variables VAR [VAR...]  Limit plotting to the listed histogram variables
      --workers N          Number of worker processes for parallel plotting (default: 1; start with 2-4; higher values use more memory)
  -v, --verbose            Forward --verbose to enable detailed diagnostics
      --quiet              Forward --quiet to suppress per-variable chatter (default)
      --cr | --sr           Override the auto-detected region
      --blind | --unblind   Force blinding or unblinding regardless of region
      --dry-run             Print the resolved command without executing it
  -h, --help                Show this help message and exit

All other tokens following "--" are forwarded verbatim to make_cr_and_sr_plots.py.
USAGE
}

if [[ ! -f "${PLOTTER_SCRIPT}" ]]; then
    echo "Error: unable to locate make_cr_and_sr_plots.py next to this wrapper." >&2
    exit 1
fi

input_path=""
output_dir=""
output_name=""
declare -a years=()
timestamp_tag=0
skip_syst=0
unit_norm=0
region_override=""
blind_override=""
declare -a variables=()
workers=1
dry_run=0
verbosity=""

# Collect positional passthrough arguments after '--'.
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--input)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                exit 1
            fi
            input_path="$2"
            shift 2
            ;;
        -o|--output-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                exit 1
            fi
            output_dir="$2"
            shift 2
            ;;
        -n|--name)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                exit 1
            fi
            output_name="$2"
            shift 2
            ;;
        -y|--year)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: Missing value for -y/--year" >&2
                exit 1
            fi
            added_year=0
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --)
                        break
                        ;;
                    --*)
                        break
                        ;;
                    -*)
                        break
                        ;;
                    *)
                        years+=("$1")
                        added_year=1
                        shift
                        ;;
                esac
            done
            if [[ ${added_year} -eq 0 ]]; then
                echo "Error: -y/--year requires at least one year token" >&2
                exit 1
            fi
            ;;
        -t|--timestamp)
            timestamp_tag=1
            shift
            ;;
        -s|--skip-syst)
            skip_syst=1
            shift
            ;;
        -u|--unit-norm)
            unit_norm=1
            shift
            ;;
        --cr)
            region_override="CR"
            shift
            ;;
        --sr)
            region_override="SR"
            shift
            ;;
        --blind)
            blind_override="blind"
            shift
            ;;
        --unblind)
            blind_override="unblind"
            shift
            ;;
        --variables)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --variables requires at least one argument" >&2
                exit 1
            fi
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --*)
                        break
                        ;;
                    -*)
                        break
                        ;;
                    *)
                        variables+=("$1")
                        shift
                        ;;
                esac
            done
            ;;
        --workers)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for --workers" >&2
                exit 1
            fi
            workers="$2"
            shift 2
            ;;
        -v|--verbose)
            verbosity="verbose"
            shift
            ;;
        -q|--quiet)
            verbosity="quiet"
            shift
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
            extra_args=("$@")
            break
            ;;
        *)
            echo "Error: Unrecognized argument '$1'" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [[ -z "${input_path}" ]]; then
    echo "Error: An input pickle must be provided with -f/--input." >&2
    exit 1
fi

if [[ -z "${output_dir}" ]]; then
    echo "Error: An output directory must be provided with -o/--output-dir." >&2
    exit 1
fi

detect_region() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        printf '\n0\n'
        return
    fi
    local filename
    filename=$(basename -- "$path")
    local uppercase="${filename^^}"
    local matches=()
    local region
    for region in CR SR; do
        local matched=0
        if [[ ${uppercase} =~ (^|[^A-Z0-9])${region} ]]; then
            matched=1
        else
            local trimmed_leading_digits="${uppercase##[0-9]*}"
            if [[ ${trimmed_leading_digits} =~ ^${region} ]]; then
                matched=1
            fi
        fi
        if (( matched )); then
            matches+=("${region}")
        fi
    done
    if (( ${#matches[@]} == 1 )); then
        printf '%s\n0\n' "${matches[0]}"
    elif (( ${#matches[@]} > 1 )); then
        printf '\n1\n'
    else
        printf '\n0\n'
    fi
}

IFS=$'\n' read -r detected_region detection_ambiguous < <(detect_region "${input_path}")

resolved_region="${region_override}"
if [[ -n "${resolved_region}" ]]; then
    echo "Region override requested: ${resolved_region}"
else
    if [[ -n "${detected_region}" ]]; then
        echo "Auto-detected region '${detected_region}' from input filename."
        resolved_region="${detected_region}"
    else
        echo "No region token detected in input filename; defaulting to 'CR'."
        resolved_region="CR"
    fi
    if [[ "${detection_ambiguous}" == "1" && -z "${region_override}" ]]; then
        echo "Warning: Detected both 'CR' and 'SR' tokens in the input filename. Defaulting to 'CR'." >&2
        resolved_region="CR"
    fi
fi

declare -r resolved_region

resolved_unblind=0
blinding_source=""
case "${blind_override}" in
    blind)
        resolved_unblind=0
        blinding_source="command-line --blind override"
        ;;
    unblind)
        resolved_unblind=1
        blinding_source="command-line --unblind override"
        ;;
    "")
        if [[ "${resolved_region}" == "CR" ]]; then
            resolved_unblind=1
            blinding_source="default for CR region"
        else
            resolved_unblind=0
            blinding_source="default for SR region"
        fi
        ;;
esac

echo "Resolved plotting region: ${resolved_region}"
if [[ ${resolved_unblind} -eq 1 ]]; then
    echo "Resolved blinding mode: unblinded (${blinding_source})"
else
    echo "Resolved blinding mode: blinded (${blinding_source})"
fi

if (( ${#years[@]} > 0 )); then
    echo "Selected years: ${years[*]}"
fi

if (( ${#variables[@]} > 0 )); then
    echo "Selected variables: ${variables[*]}"
fi

if [[ -n "${workers}" && "${workers}" != "1" ]]; then
    echo "Worker processes: ${workers}"
fi

case "${verbosity}" in
    verbose)
        echo "Verbose diagnostics enabled."
        ;;
    quiet)
        echo "Quiet mode enforced."
        ;;
esac

mkdir -p "${output_dir}"

cmd=("${PYTHON_BIN}" "${PLOTTER_SCRIPT}" "-f" "${input_path}" "-o" "${output_dir}")

if [[ -n "${output_name}" ]]; then
    cmd+=("-n" "${output_name}")
fi
if (( ${#years[@]} > 0 )); then
    cmd+=("-y")
    cmd+=("${years[@]}")
fi
if (( timestamp_tag )); then
    cmd+=("-t")
fi
if (( skip_syst )); then
    cmd+=("-s")
fi
if (( unit_norm )); then
    cmd+=("-u")
fi
if [[ "${resolved_region}" == "CR" ]]; then
    cmd+=("--cr")
else
    cmd+=("--sr")
fi
if (( resolved_unblind )); then
    cmd+=("--unblind")
else
    cmd+=("--blind")
fi
if (( ${#variables[@]} > 0 )); then
    cmd+=("--variables")
    cmd+=("${variables[@]}")
fi
cmd+=("--workers" "${workers}")
case "${verbosity}" in
    verbose)
        cmd+=("--verbose")
        ;;
    quiet)
        cmd+=("--quiet")
        ;;
esac
if (( ${#extra_args[@]} > 0 )); then
    cmd+=("--")
    cmd+=("${extra_args[@]}")
fi

echo "Executing make_cr_and_sr_plots.py with command:"
printf '  %q' "${cmd[@]}"
echo

if (( dry_run )); then
    echo "Dry-run requested; skipping execution."
else
    "${cmd[0]}" "${cmd[@]:1}"
fi
