#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLOTTER_SCRIPT="${SCRIPT_DIR}/make_cr_and_sr_plots.py"
PYTHON_BIN=${PYTHON_BIN:-${PYTHON:-python}}

show_help() {
    cat <<'USAGE'
Usage: run_plotter.sh -f PICKLE [-f PICKLE ...] -o OUTPUT_DIR -y YEAR [YEAR ...] [options]
       run_plotter.sh --campaign-manifest TSV --campaign-phase all|njets|other -o NEW_ROOT --dry-run|--execute

Wrapper around make_cr_and_sr_plots.py with filename-based region detection.
Consult the "CR/SR plotting CLI quickstart" section of analysis/topeft_run2/README.md
for more workflow examples.

Required arguments:
  -f, --input PATH          Input histogram pickle (must exist and be readable; repeat for multiple inputs)
  -o, --output-dir PATH     Directory where plots will be written
  -y, --year YEAR [YEAR ...]
                           One or more year tokens forwarded to the plotter.
                           Aggregate aliases expand as follows:
                             run2 → UL16 UL16APV UL17 UL18
                             run3 → 2022 2022EE 2023 2023BPix
                           The flag is required; omitting it raises an error.

Optional arguments:
  -n, --name NAME           Name for the output directory (forwarded with -n)
  -t, --timestamp           Append a timestamp to the output directory name
  -s, --skip-syst           Skip systematic error bands
  -u, --unit-norm           Enable unit-normalized plotting
      --log-y              Use a logarithmic y-axis for the stacked panel
      --variable VAR          Limit plotting to a single histogram variable (repeat to add more)
      --variables VAR [VAR...]
                           Limit plotting to the listed histogram variables
      --rebin-plot-vars SPEC
                           Request plot/report-time rebinning, e.g. j0pt:2,l1conept=2
      --workers N          Number of worker processes for parallel plotting (default: 1; start with 2-4; higher values use more memory)
      --channel-output MODE  Forward merged/split channel selection (merged, split, both, merged-njets,
                           split-njets, both-njets). The -njets variants behave like their counterparts
                           but keep the per-njet bins defined in cr_sr_plots_metadata.yml (default: merged)
      --no-negative-weight-report
                           Disable the default negative MC contribution CSV/Markdown report
  -v, --verbose            Forward --verbose to enable detailed diagnostics
      --quiet              Forward --quiet to suppress per-variable chatter (default)
      --cr | --sr           Override the auto-detected region
      --blind | --unblind   Force blinding or unblinding regardless of region
      --dry-run             Print the resolved command without executing it
  -h, --help                Show this help message and exit

Unrecognised options are forwarded directly to make_cr_and_sr_plots.py. The
historical "--" delimiter is no longer necessary; any leftover tokens are passed
through automatically. Negative MC contribution reports are enabled by default;
use --no-negative-weight-report when only figures are needed.

Example:
  ./run_plotter.sh -f histos/plotsCR_2017.pkl.gz -o ~/www/cr_plots -y 2017 \
    --variables j0pt l1conept --rebin-plot-vars j0pt:2,l1conept=2

Campaign mode reads the SR baseline TSV as data, validates every row, and runs
one principal input per guarded child. Dry-run prints each maintained child
command without creating the output root. Execute requires an absent root.
USAGE
}

trim_whitespace() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

declare -A YEAR_CANONICAL_MAP=(
    [2016]=2016
    [ul16]=2016
    [2016apv]=2016APV
    [ul16apv]=2016APV
    [2017]=2017
    [ul17]=2017
    [2018]=2018
    [ul18]=2018
    [2022]=2022
    [2022ee]=2022EE
    [2023]=2023
    [2023bpix]=2023BPix
)

declare -A YEAR_AGGREGATE_ALIASES=(
    [run2]="UL16 UL16APV UL17 UL18"
    [run3]="2022 2022EE 2023 2023BPix"
)

normalize_year_tokens() {
    local -a raw_tokens=("$@")
    local -a normalized=()
    declare -A seen=()

    local raw token trimmed lowered expansion_tokens canonical candidate
    for raw in "${raw_tokens[@]}"; do
        IFS=',' read -r -a token_parts <<<"$raw"
        for token in "${token_parts[@]}"; do
            trimmed=$(trim_whitespace "$token")
            if [[ -z "$trimmed" ]]; then
                continue
            fi
            lowered="${trimmed,,}"
            if [[ -n "${YEAR_AGGREGATE_ALIASES[$lowered]+x}" ]]; then
                read -r -a expansion_tokens <<<"${YEAR_AGGREGATE_ALIASES[$lowered]}"
            else
                expansion_tokens=("$trimmed")
            fi
            for candidate in "${expansion_tokens[@]}"; do
                canonical="${YEAR_CANONICAL_MAP[${candidate,,}]:-$candidate}"
                if [[ -z "${seen[$canonical]+x}" ]]; then
                    normalized+=("$canonical")
                    seen[$canonical]=1
                fi
            done
        done
    done

    printf '%s\n' "${normalized[@]}"
}

main() {
    if [[ ! -f "${PLOTTER_SCRIPT}" ]]; then
        echo "Error: unable to locate make_cr_and_sr_plots.py next to this wrapper." >&2
        return 1
    fi

    local -a input_paths=()
    local output_dir=""
    local output_name=""
    local -a years=()
    local timestamp_tag=0
    local skip_syst=0
    local unit_norm=0
    local log_y=0
    local region_override=""
    local blind_override=""
    local -a variables=()
    local workers=1
    local dry_run=0
    local verbosity=""
    local channel_output=""

    # Collect passthrough arguments for make_cr_and_sr_plots.py.
    local -a extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
        -f|--input)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                return 1
            fi
            input_paths+=("$2")
            shift 2
            ;;
        -o|--output-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                return 1
            fi
            output_dir="$2"
            shift 2
            ;;
        -n|--name)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for $1" >&2
                return 1
            fi
            output_name="$2"
            shift 2
            ;;
        -y|--year)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: Missing value for -y/--year" >&2
                return 1
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
                return 1
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
        --log-y)
            log_y=1
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
        --variable|--variables)
            local flag="$1"
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: ${flag} requires at least one argument" >&2
                return 1
            fi
            if [[ "${flag}" == "--variable" ]]; then
                variables+=("$1")
                shift
                continue
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
                return 1
            fi
            workers="$2"
            shift 2
            ;;
        --channel-output)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for --channel-output" >&2
                return 1
            fi
            channel_output="${2,,}"
            case "${channel_output}" in
                merged|split|both|merged-njets|split-njets|both-njets)
                    ;;
                *)
                    echo "Error: --channel-output expects one of: merged, split, both, merged-njets, split-njets, both-njets" >&2
                    return 1
                    ;;
            esac
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
            return 0
            ;;
        *)
            if [[ "$1" == "--" ]]; then
                # Backward compatibility: ignore the legacy delimiter and
                # continue consuming passthrough arguments without adding it to
                # the forwarded list.
                shift
                continue
            fi
            extra_args+=("$1")
            shift
            ;;
        esac
    done

    if (( ${#years[@]} > 0 )); then
        mapfile -t years < <(normalize_year_tokens "${years[@]}")
    fi

    if (( ${#years[@]} == 0 )); then
        echo "Error: At least one year must be provided with -y/--year." >&2
        return 1
    fi

    if (( ${#input_paths[@]} == 0 )); then
        echo "Error: At least one input pickle must be provided with -f/--input." >&2
        return 1
    fi

    local input_path
    for input_path in "${input_paths[@]}"; do
        if [[ ! -r "${input_path}" ]]; then
            echo "Error: Input pickle '${input_path}' is missing or unreadable." >&2
            return 1
        fi
    done

    if [[ -z "${output_dir}" ]]; then
        echo "Error: An output directory must be provided with -o/--output-dir." >&2
        return 1
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

    local resolved_region="${region_override}"
    if [[ -n "${resolved_region}" ]]; then
        echo "Region override requested: ${resolved_region}"
    elif (( ${#input_paths[@]} == 1 )); then
        local detected_region=""
        local detection_ambiguous=0
        IFS=$'\n' read -r detected_region detection_ambiguous < <(detect_region "${input_paths[0]}")

        if [[ -n "${detected_region}" ]]; then
            echo "Auto-detected region '${detected_region}' from input filename."
            resolved_region="${detected_region}"
        else
            echo "No region token detected in input filename; defaulting to 'CR'."
            resolved_region="CR"
        fi
        if [[ "${detection_ambiguous}" == "1" ]]; then
            echo "Warning: Detected both 'CR' and 'SR' tokens in the input filename. Defaulting to 'CR'." >&2
            resolved_region="CR"
        fi
    else
        local detected_region=""
        local detection_ambiguous=0
        local classified_input_count=0
        local input_path
        local first_detected_region=""

        for input_path in "${input_paths[@]}"; do
            IFS=$'\n' read -r detected_region detection_ambiguous < <(detect_region "${input_path}")
            if [[ "${detection_ambiguous}" == "1" ]]; then
                echo "Error: Input filename '${input_path}' contains both CR and SR region tokens; use --cr or --sr to override." >&2
                return 1
            fi
            if [[ -n "${detected_region}" ]]; then
                ((classified_input_count += 1))
                if [[ -z "${first_detected_region}" ]]; then
                    first_detected_region="${detected_region}"
                elif [[ "${detected_region}" != "${first_detected_region}" ]]; then
                    echo "Error: Input filenames resolve to conflicting regions; use --cr or --sr to override." >&2
                    return 1
                fi
            fi
        done

        if (( classified_input_count == 0 )); then
            echo "No region token detected in input filenames; defaulting to 'CR'."
            resolved_region="CR"
        elif (( classified_input_count != ${#input_paths[@]} )); then
            echo "Error: Some input filenames have a region token and others do not; use --cr or --sr to override." >&2
            return 1
        else
            echo "Auto-detected region '${first_detected_region}' from all input filenames."
            resolved_region="${first_detected_region}"
        fi
    fi

    declare -r resolved_region

    local resolved_unblind=0
    local blinding_source=""
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
    if [[ -n "${channel_output}" ]]; then
        echo "Channel output selection: ${channel_output}"
    fi

    if (( log_y )); then
        echo "Stacked panel will use a logarithmic y-axis."
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

    local -a cmd=("${PYTHON_BIN}" "${PLOTTER_SCRIPT}")
    for input_path in "${input_paths[@]}"; do
        cmd+=("-f" "${input_path}")
    done
    cmd+=("-o" "${output_dir}")

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
    if (( log_y )); then
        cmd+=("--log-y")
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
    if [[ -n "${channel_output}" ]]; then
        cmd+=("--channel-output" "${channel_output}")
    fi
    case "${verbosity}" in
        verbose)
            cmd+=("--verbose")
            ;;
        quiet)
            cmd+=("--quiet")
            ;;
    esac
    cmd+=("${extra_args[@]}")

    echo "Executing make_cr_and_sr_plots.py with command:"
    printf '  %q' "${cmd[@]}"
    echo

    if (( dry_run )); then
        echo "Dry-run requested; skipping execution."
    else
        mkdir -p "${output_dir}"
        "${cmd[0]}" "${cmd[@]:1}"
        return $?
    fi

    return 0
}

campaign_main() {
    local manifest="" phase="" output_root="" action=""
    while (( $# )); do
        case "$1" in
            --campaign-manifest|--campaign-phase|-o|--output-dir)
                if (( $# < 2 )); then echo "Missing campaign value for $1" >&2; return 2; fi
                case "$1" in
                    --campaign-manifest) manifest=$2 ;;
                    --campaign-phase) phase=$2 ;;
                    *) output_root=$2 ;;
                esac
                shift 2 ;;
            --dry-run|--execute)
                if [[ -n $action ]]; then echo "Choose one campaign action" >&2; return 2; fi
                action=$1; shift ;;
            *) echo "Unknown campaign argument: $1" >&2; return 2 ;;
        esac
    done
    if [[ ! -f $manifest || ! -r $manifest || ! $manifest = /* ||
          ! $output_root = /* || ! $phase =~ ^(all|njets|other)$ || -z $action ]]; then
        echo "Campaign requires readable absolute TSV, phase, absolute root and explicit action" >&2
        return 2
    fi
    local repo_root histos_root
    repo_root=$(cd "$SCRIPT_DIR/../.." && pwd -P) || return 2
    histos_root=$repo_root/histos
    if [[ $output_root != "$histos_root/"* || $output_root == *'/../'* ||
          $output_root == */.. || $output_root == */. ||
          -e $output_root || -L $output_root || ! -d $histos_root ]]; then
        echo "Campaign root must be absent beneath $histos_root (symlinks count as occupied)" >&2
        return 2
    fi
    local header expected_header
    IFS= read -r header < "$manifest"
    expected_header=$'unit\tkind\trun\tview\towner\tyears\tvariables\texpected_figures\tpkl\thistorical_pkl_sha256\tsidecar\thistorical_sidecar_sha256'
    if [[ $header != "$expected_header" ]] || ! LC_ALL=C awk -F '\t' '
        NR > 1 && (NF != 12 || $0 ~ /\r$/) { exit 1 }
        END { if (NR < 2) exit 1 }
    ' "$manifest"; then
        echo "Invalid campaign TSV header or row width" >&2; return 2
    fi
    local -a rows selected=()
    mapfile -t rows < "$manifest"
    declare -A seen_units=() seen_outputs=() seen_inputs=() seen_sidecars=()
    local row unit kind run view owner years variables expected_figures pkl pkl_hash sidecar sidecar_hash
    local output_rel year variable key count_njets=0 count_other=0 figures_njets=0 figures_other=0
    local -a year_args variable_args
    for row in "${rows[@]:1}"; do
        IFS=$'\t' read -r unit kind run view owner years variables expected_figures pkl pkl_hash sidecar sidecar_hash <<< "$row"
        output_rel="$run/$view/$owner/${variables//,/_}"
        if [[ ! $unit =~ ^[A-Za-z0-9_.+-]+$ || $unit == . || $unit == .. ||
              ! $kind =~ ^(njets|other)$ ||
              ! $run =~ ^(run2|run3)$ || ! $view =~ ^[A-Za-z0-9_.+-]+$ ||
              $view == . || $view == .. ||
              ! $owner =~ ^[A-Za-z0-9_.+-]+$ || $owner == . || $owner == .. ||
              ! $years =~ ^[A-Za-z0-9,]+$ || $years == ,* || $years == *, || $years == *,,* ||
              ! $variables =~ ^[A-Za-z0-9_,]+$ || ! $expected_figures =~ ^[1-9][0-9]*$ ||
              $variables == ,* || $variables == *, || $variables == *,,* ||
              ! $pkl = /* || ! $sidecar = /* ||
              $sidecar != "$pkl.metadata.json" ||
              ! $pkl_hash =~ ^[0-9a-f]{64}$ || ! $sidecar_hash =~ ^[0-9a-f]{64}$ ||
              -n ${seen_units[$unit]+x} || -n ${seen_outputs[$output_rel]+x} ]]; then
            echo "Invalid or duplicate campaign row: $unit" >&2; return 2
        fi
        seen_units[$unit]=1; seen_outputs[$output_rel]=1
        IFS=, read -r -a year_args <<< "$years"
        IFS=, read -r -a variable_args <<< "$variables"
        for year in "${year_args[@]}"; do
            if [[ ! $year =~ ^(2016APV|2016|2017|2018|2022|2022EE|2023|2023BPix)$ ]]; then
                echo "Invalid year in $unit" >&2; return 2
            fi
            if [[ $run == run2 && $year == 20[23]* ]] || [[ $run == run3 && $year == 201* ]]; then
                echo "Year/run mismatch in $unit" >&2; return 2
            fi
        done
        for variable in "${variable_args[@]}"; do
            if [[ ! $variable =~ ^(njets|lj0pt|ptz|ptz_wtau|ptll|lt)$ ]]; then
                echo "Unsupported SR campaign variable in $unit" >&2; return 2
            fi
        done
        if [[ $kind == njets && $variables != njets ]] ||
           [[ $kind == other && ",$variables," == *,njets,* ]]; then
            echo "Kind/variable mismatch in $unit" >&2; return 2
        fi
        if [[ -n ${seen_inputs[$pkl]+x} && ${seen_inputs[$pkl]} != "$pkl_hash" ]] ||
           [[ -n ${seen_sidecars[$sidecar]+x} && ${seen_sidecars[$sidecar]} != "$sidecar_hash" ]]; then
            echo "Conflicting digest for campaign input in $unit" >&2; return 2
        fi
        if [[ $phase == all || $phase == "$kind" ]]; then
            seen_inputs[$pkl]=$pkl_hash; seen_sidecars[$sidecar]=$sidecar_hash
            selected+=("$row")
            if [[ $kind == njets ]]; then
                ((count_njets += 1)); ((figures_njets += expected_figures))
            else
                ((count_other += 1)); ((figures_other += expected_figures))
            fi
        fi
    done
    if (( ${#selected[@]} == 0 )); then echo "No rows selected" >&2; return 2; fi
    printf 'Campaign TSV: %s\nPhase: %s\nRows: njets=%d other=%d\nPlanned figures: njets=%d other=%d\n' \
        "$manifest" "$phase" "$count_njets" "$count_other" "$figures_njets" "$figures_other"
    if [[ $action == --execute ]]; then
        if [[ $(git -C "$repo_root" branch --show-current) != te/datacard_sr_reco ]]; then
            echo "Execute requires the selected TOPEFT feature branch" >&2; return 2
        fi
        local free_bytes free_inodes
        free_bytes=$(df -B1 --output=avail "$histos_root" | tail -n 1)
        free_inodes=$(df --output=iavail "$histos_root" | tail -n 1)
        if [[ ! $free_bytes =~ ^[0-9]+$ || ! $free_inodes =~ ^[0-9]+$ ]] ||
           (( free_bytes == 0 || free_inodes == 0 )); then
            echo "Output capacity is unavailable or exhausted" >&2; return 2
        fi
        printf 'Output capacity: bytes=%s inodes=%s\n' "$free_bytes" "$free_inodes"
        for pkl in "${!seen_inputs[@]}"; do
            [[ -r $pkl ]] && printf '%s  %s\n' "${seen_inputs[$pkl]}" "$pkl" | sha256sum -c - || return 2
        done
        for sidecar in "${!seen_sidecars[@]}"; do
            [[ -r $sidecar ]] && printf '%s  %s\n' "${seen_sidecars[$sidecar]}" "$sidecar" | sha256sum -c - || return 2
        done
        mkdir -m 700 -- "$output_root" || return 2
        sha256sum "$manifest" "$PLOTTER_SCRIPT" "$SCRIPT_DIR/run_plotter.sh" \
            "$repo_root/topeft/modules/node_memory_guard.py" \
            "$repo_root/topeft/params/cr_sr_plots_metadata.yml" > "$output_root/source_sha256.txt" || return 2
        git -C "$repo_root" rev-parse HEAD > "$output_root/source_head.txt" || return 2
    fi
    local output_parent output_name guard_path log_path ledger_path status_check
    local -a child guard statuses
    for kind in njets other; do
        [[ $phase == all || $phase == "$kind" ]] || continue
        for row in "${selected[@]}"; do
            IFS=$'\t' read -r unit key run view owner years variables expected_figures pkl pkl_hash sidecar sidecar_hash <<< "$row"
            [[ $key == "$kind" ]] || continue
            IFS=, read -r -a year_args <<< "$years"
            IFS=, read -r -a variable_args <<< "$variables"
            output_parent="$output_root/$run/$view/$owner"
            output_name="${variables//,/_}"
            guard_path="$output_root/${unit}.guard.jsonl"
            log_path="$output_root/${unit}.log"
            ledger_path="$output_root/units.tsv"
            child=("$SCRIPT_DIR/run_plotter.sh" -f "$pkl" -o "$output_parent" -n "$output_name"
                   -y "${year_args[@]}" --sr --unblind --variables "${variable_args[@]}"
                   --channel-output merged --binning processing --workers 1
                   --output-formats png --year-coverage-policy warn
                   --merge-report "$output_root/${unit}.composition.json")
            printf 'Unit: %s input=%s output=%s\n' "$unit" "$pkl" "$output_parent/$output_name"
            if [[ $action == --dry-run ]]; then
                "${child[@]}" --dry-run || return $?
                continue
            fi
            if [[ -e $output_parent/$output_name || -L $output_parent/$output_name ||
                  -e $guard_path || -L $guard_path || -e $log_path || -L $log_path ]]; then
                echo "Occupied campaign unit: $unit" >&2; return 2
            fi
            mkdir -p -- "$output_parent" || return 2
            guard=("$PYTHON_BIN" -m topeft.modules.node_memory_guard
                   --hard-limit-bytes 245000000000 --sampling-interval-seconds 0.5
                   --evidence-path "$guard_path" -- "${child[@]}")
            printf '%s\tstart\t%s\t%s\n' "$unit" "$pkl" "$output_parent/$output_name" >> "$ledger_path"
            printf 'Guard command:'; printf ' %q' "${guard[@]}"; printf '\n'
            if PYTHONUNBUFFERED=1 "${guard[@]}" 2>&1 | tee "$log_path"; then
                statuses=("${PIPESTATUS[@]}")
            else
                statuses=("${PIPESTATUS[@]}")
            fi
            status_check=0
            if [[ -f $guard_path ]]; then
                "$PYTHON_BIN" -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; results=[row for row in rows if row.get("event")=="result"]; assert len(results)==1 and rows[-1] is results[0] and results[0].get("termination_kind")=="normal" and results[0].get("returncode")==0' "$guard_path" || status_check=1
            else
                status_check=1
            fi
            printf '%s\tresult\tguard=%s\ttee=%s\tterminal=%s\n' \
                "$unit" "${statuses[0]}" "${statuses[1]}" "$status_check" >> "$ledger_path"
            if (( ${statuses[0]} != 0 || ${statuses[1]} != 0 || status_check != 0 )); then
                printf 'campaign\tstopped\t%s\n' "$unit" >> "$ledger_path"
                echo "Campaign stopped after failed or ambiguous unit $unit" >&2; return 1
            fi
        done
    done
    if [[ $action == --execute ]]; then
        printf 'campaign\tcomplete\tunits=%d\n' "${#selected[@]}" >> "$ledger_path"
    fi
}

campaign_requested=0
for argument in "$@"; do
    if [[ $argument == --campaign-manifest ]]; then campaign_requested=1; break; fi
done
if (( campaign_requested )); then
    if campaign_main "$@"; then exit_code=0; else exit_code=$?; fi
elif main "$@"; then
    exit_code=0
else
    exit_code=$?
fi
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exit "${exit_code}"
else
    return "${exit_code}"
fi
