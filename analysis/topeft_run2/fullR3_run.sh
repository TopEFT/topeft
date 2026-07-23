#!/usr/bin/env bash

# PrintUsage: display script usage information
PrintUsage() {
  echo "Usage: $0 [-y YEAR [YEAR ...]] [-t TAG] --cr | --sr [--hist-vars HIST [HIST ...]] [--sample-json JSON | --cfg-override CFG] [run_analysis options]"
  echo
  echo "Options:"
  echo "  -y YEAR    Year identifier (repeat or list multiple years)"
  echo "             Bundles: run2 -> UL16 UL16APV UL17 UL18;"
  echo "                      run3 -> 2022 2022EE 2023 2023BPix"
  echo "  -t TAG     Git tag or commit identifier"
  echo "  --cr       Generate control-region histograms"
  echo "  --sr       Generate signal-region histograms"
  echo "  --defer-np Defer nonprompt post-processing (adds --np-postprocess=defer)"
  echo "  --hist-vars HIST [HIST ...]"
  echo "             Override the histogram list while preserving --cr/--sr region behavior"
  echo "  --sample-json JSON"
  echo "             Use one sample JSON instead of the default CFG bundle"
  echo "  --cfg-override CFG"
  echo "             Use one CFG file instead of the default CFG bundle"
  echo "  --ttgamma-sample-role-policy POLICY"
  echo "             Forward ttgamma sample-role policy to run_analysis.py"
  echo "             Supported values: split, run2_nlo_inclusive"
  echo "  -p, --outpath PATH"
  echo "             Override the run_analysis.py output directory"
  echo "  --dry-run  Print the resolved run_analysis.py command and exit"
  echo "  -h, --help Show this help message"
  echo
  echo "Any additional options after those listed above are passed directly"
  echo "to run_analysis.py, allowing access to its full set of arguments."
}

main() {
  # Early exit when no arguments are provided
  if [[ $# -eq 0 ]]; then
    PrintUsage
    return 0
  fi

  # Default values
  local DEFAULT_YEAR="2022"
  local DEFAULT_TAG="fec79a60_PNet"
  local FLAG_CR=false
  local FLAG_SR=false
  local FLAG_DEFER_NP=false
  local FLAG_DRY_RUN=false
  local HIST_VARS_PROVIDED=false
  local -a EXTRA_ARGS=()
  local -a HIST_VARS=()
  local -a YEARS=()
  local -a EXPANDED_YEARS=()
  local -a RESOLVED_YEARS=()
  local USER_CHUNK_OVERRIDE=false
  local USER_OUTPATH_OVERRIDE=false
  local USER_OUTPATH_OPTION_COUNT=0
  local TTGAMMA_SAMPLE_ROLE_POLICY="split"
  local DEFAULT_OUTPATH="/groups/klannon/$USER/"
  local RESOLVED_OUTPATH="$DEFAULT_OUTPATH"
  local TAG=""
  local SAMPLE_JSON=""
  local CFG_OVERRIDE=""

  # Parse command-line arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--year)
        shift
        if [[ $# -eq 0 || "$1" == -* ]]; then
          echo "Error: -y|--year requires at least one argument"
          return 1
        fi
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -*)
              break
              ;;
            *)
              YEARS+=("$1")
              shift
              ;;
          esac
        done
        ;;
      -t|--tag)
        TAG="$2"
        shift 2
        ;;
      --cr)
        FLAG_CR=true
        shift
        ;;
      --sr)
        FLAG_SR=true
        shift
        ;;
      --defer-np)
        FLAG_DEFER_NP=true
        shift
        ;;
      --hist-vars)
        shift
        if [[ $# -eq 0 || "$1" == -* ]]; then
          echo "Error: --hist-vars requires at least one histogram name"
          return 1
        fi
        HIST_VARS_PROVIDED=true
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -*)
              break
              ;;
            *)
              HIST_VARS+=("$1")
              shift
              ;;
          esac
        done
        ;;
      --dry-run)
        FLAG_DRY_RUN=true
        shift
        ;;
      --sample-json)
        if [[ $# -lt 2 || "$2" == -* ]]; then
          echo "Error: --sample-json requires a JSON path"
          return 1
        fi
        SAMPLE_JSON="$2"
        shift 2
        ;;
      --cfg-override)
        if [[ $# -lt 2 || "$2" == -* ]]; then
          echo "Error: --cfg-override requires a CFG path"
          return 1
        fi
        CFG_OVERRIDE="$2"
        shift 2
        ;;
      --ttgamma-sample-role-policy)
        if [[ $# -lt 2 || "$2" == -* ]]; then
          echo "Error: --ttgamma-sample-role-policy requires a policy value"
          return 1
        fi
        case "$2" in
          split|run2_nlo_inclusive)
            TTGAMMA_SAMPLE_ROLE_POLICY="$2"
            ;;
          *)
            echo "Error: unsupported --ttgamma-sample-role-policy value: $2" >&2
            echo "Supported values: split, run2_nlo_inclusive" >&2
            return 1
            ;;
        esac
        shift 2
        ;;
      --ttgamma-sample-role-policy=*)
        TTGAMMA_SAMPLE_ROLE_POLICY="${1#--ttgamma-sample-role-policy=}"
        case "$TTGAMMA_SAMPLE_ROLE_POLICY" in
          split|run2_nlo_inclusive) ;;
          *)
            echo "Error: unsupported --ttgamma-sample-role-policy value: $TTGAMMA_SAMPLE_ROLE_POLICY" >&2
            echo "Supported values: split, run2_nlo_inclusive" >&2
            return 1
            ;;
        esac
        shift
        ;;
      -h|--help)
        PrintUsage
        return 0
        ;;
      *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done

  # Detect if a user-specified chunk size was provided
  local ARG
  local EXTRA_INDEX
  for ((EXTRA_INDEX=0; EXTRA_INDEX<${#EXTRA_ARGS[@]}; EXTRA_INDEX++)); do
    ARG="${EXTRA_ARGS[$EXTRA_INDEX]}"
    case "$ARG" in
      -s|--chunksize|--chunksize=*)
        USER_CHUNK_OVERRIDE=true
        ;;
    esac

    case "$ARG" in
      -p|--outpath)
        USER_OUTPATH_OPTION_COUNT=$((USER_OUTPATH_OPTION_COUNT + 1))
        if (( EXTRA_INDEX + 1 >= ${#EXTRA_ARGS[@]} )) || [[ "${EXTRA_ARGS[$((EXTRA_INDEX + 1))]}" == -* ]]; then
          echo "Error: $ARG requires an output path"
          return 1
        fi
        RESOLVED_OUTPATH="${EXTRA_ARGS[$((EXTRA_INDEX + 1))]}"
        ;;
      --outpath=*)
        USER_OUTPATH_OPTION_COUNT=$((USER_OUTPATH_OPTION_COUNT + 1))
        RESOLVED_OUTPATH="${ARG#--outpath=}"
        if [[ -z "$RESOLVED_OUTPATH" ]]; then
          echo "Error: --outpath requires an output path"
          return 1
        fi
        ;;
    esac
  done

  if (( USER_OUTPATH_OPTION_COUNT > 1 )); then
    echo "Error: provide only one output path option (-p or --outpath)." >&2
    return 1
  fi

  if (( USER_OUTPATH_OPTION_COUNT == 1 )); then
    USER_OUTPATH_OVERRIDE=true
  fi

  # Ensure exactly one mode is chosen
  if [[ "$FLAG_CR" == "false" && "$FLAG_SR" == "false" ]] || [[ "$FLAG_CR" == "true" && "$FLAG_SR" == "true" ]]; then
    echo "Error: You must specify exactly one of --cr or --sr."
    echo
    PrintUsage
    return 1
  fi

  # Apply defaults with warnings if not provided
  if [[ ${#YEARS[@]} -eq 0 ]]; then
    echo "Warning: YEAR not provided, using default YEAR=$DEFAULT_YEAR"
    YEARS=("$DEFAULT_YEAR")
  fi

  local YEAR
  for YEAR in "${YEARS[@]}"; do
    case "${YEAR,,}" in
      run2)
        EXPANDED_YEARS+=(UL16 UL16APV UL17 UL18)
        ;;
      run3)
        EXPANDED_YEARS+=(2022 2022EE 2023 2023BPix)
        ;;
      *)
        EXPANDED_YEARS+=("$YEAR")
        ;;
    esac
  done

  declare -A YEAR_SEEN=()
  for YEAR in "${EXPANDED_YEARS[@]}"; do
    if [[ -z "${YEAR_SEEN[$YEAR]}" ]]; then
      RESOLVED_YEARS+=("$YEAR")
      YEAR_SEEN[$YEAR]=1
    fi
  done

  if [[ ${#RESOLVED_YEARS[@]} -eq 0 ]]; then
    echo "Error: No years resolved from the provided arguments." >&2
    return 1
  fi

  if [[ -n "$SAMPLE_JSON" && -n "$CFG_OVERRIDE" ]]; then
    echo "Error: use only one of --sample-json or --cfg-override." >&2
    return 1
  fi

  if [[ -n "$SAMPLE_JSON" && ! -f "$SAMPLE_JSON" ]]; then
    echo "Error: sample JSON not found: $SAMPLE_JSON" >&2
    return 1
  fi

  if [[ -n "$CFG_OVERRIDE" && ! -f "$CFG_OVERRIDE" ]]; then
    echo "Error: CFG override not found: $CFG_OVERRIDE" >&2
    return 1
  fi

  if [[ -z "$TAG" ]]; then
    echo "Warning: TAG not provided, using default TAG=$DEFAULT_TAG"
    TAG="$DEFAULT_TAG"
  fi

  # Define output name based on mode
  local YEAR_LABEL
  YEAR_LABEL=$(IFS=-; echo "${RESOLVED_YEARS[*]}")

  local OUT_NAME
  local REGION_LABEL
  if [[ "$FLAG_CR" == "true" ]]; then
    OUT_NAME="${YEAR_LABEL}CRs_${TAG}"
    REGION_LABEL="CR"
  else
    OUT_NAME="${YEAR_LABEL}SRs_${TAG}"
    REGION_LABEL="SR"
  fi

  echo "OUT_NAME: $OUT_NAME"

  # Build the configuration file list
  local CFGS_PATH="../../input_samples/cfgs"
  local -a CFGS_LIST=()

  declare -A RUN2_YEAR_MAP=(
    [2016]=2016
    [UL16]=2016
    [2016APV]=2016APV
    [UL16APV]=2016APV
    [2017]=2017
    [UL17]=2017
    [2018]=2018
    [UL18]=2018
  )

  local -a RUN2_CFGS_SR=(
    "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
    "${CFGS_PATH}/data_samples_NDSkim.cfg"
  )

  local -a RUN2_CFGS_CR=(
    "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_cr_NDSkim.cfg"
    "${CFGS_PATH}/data_samples_NDSkim.cfg"
  )

  declare -A SEEN_CFGS=()
  local RUN2_BUNDLE_ADDED=false

  is_run2_year() {
    local year="$1"
    [[ -n "${RUN2_YEAR_MAP[$year]+x}" ]]
  }

  add_cfg_once() {
    local cfg_file="$1"
    if [[ ! -f "$cfg_file" ]]; then
      echo "Error: Required cfg file not found: $cfg_file" >&2
      return 1
    fi
    if [[ -n "${SEEN_CFGS[$cfg_file]}" ]]; then
      return 0
    fi
    CFGS_LIST+=("$cfg_file")
    SEEN_CFGS[$cfg_file]=1
    return 0
  }

  get_run2_cfgs_for_region() {
    local region="$1"
    case "$region" in
      CR)
        printf '%s\n' "${RUN2_CFGS_CR[@]}"
        ;;
      SR)
        printf '%s\n' "${RUN2_CFGS_SR[@]}"
        ;;
      *)
        echo "Error: Unsupported region for Run 2 cfg resolution: $region" >&2
        return 1
        ;;
    esac
  }

  get_run3_cfgs_for_region_and_year() {
    local region="$1"
    local year="$2"
    case "$region" in
      CR)
        printf '%s\n' \
          "${CFGS_PATH}/NDSkim_${year}_background_samples_cr.cfg" \
          "${CFGS_PATH}/NDSkim_${year}_data_samples.cfg" \
          "${CFGS_PATH}/NDSkim_${year}_mc_signal_samples.cfg"
        ;;
      SR)
        printf '%s\n' \
          "${CFGS_PATH}/NDSkim_${year}_background_samples.cfg" \
          "${CFGS_PATH}/NDSkim_${year}_data_samples.cfg" \
          "${CFGS_PATH}/NDSkim_${year}_mc_signal_samples_sr.cfg"
        ;;
      *)
        echo "Error: Unsupported region for Run 3 cfg resolution: $region" >&2
        return 1
        ;;
    esac
  }

  add_run2_cfg_bundle_once() {
    local region="$1"
    local cfg
    if [[ "$RUN2_BUNDLE_ADDED" == "true" ]]; then
      return 0
    fi
    while IFS= read -r cfg; do
      add_cfg_once "$cfg" || return 1
    done < <(get_run2_cfgs_for_region "$region")
    RUN2_BUNDLE_ADDED=true
  }

  add_run3_cfg_bundle_for_year() {
    local region="$1"
    local year="$2"
    local cfg
    while IFS= read -r cfg; do
      add_cfg_once "$cfg" || return 1
    done < <(get_run3_cfgs_for_region_and_year "$region" "$year")
  }

  local INPUT_OVERRIDE_LABEL=""
  if [[ -n "$SAMPLE_JSON" ]]; then
    CFGS_LIST=("$SAMPLE_JSON")
    INPUT_OVERRIDE_LABEL="sample JSON: $SAMPLE_JSON"
  elif [[ -n "$CFG_OVERRIDE" ]]; then
    CFGS_LIST=("$CFG_OVERRIDE")
    INPUT_OVERRIDE_LABEL="CFG: $CFG_OVERRIDE"
  else
    for YEAR in "${RESOLVED_YEARS[@]}"; do
      if is_run2_year "$YEAR"; then
        add_run2_cfg_bundle_once "$REGION_LABEL" || return 1
      else
        add_run3_cfg_bundle_for_year "$REGION_LABEL" "$YEAR" || return 1
      fi
    done
  fi
  local CFGS
  CFGS=$(IFS=,; echo "${CFGS_LIST[*]}")

  echo "Resolved years: ${RESOLVED_YEARS[*]}"
  if [[ -n "$INPUT_OVERRIDE_LABEL" ]]; then
    echo "Input override: $INPUT_OVERRIDE_LABEL"
  fi
  echo "Resolved CFGS: $CFGS"

  local -a HIST_LIST_ARGS=()
  if [[ "$HIST_VARS_PROVIDED" == "true" ]]; then
    HIST_LIST_ARGS=(--hist-list "${HIST_VARS[@]}")
  elif [[ "$FLAG_CR" == "true" ]]; then
    HIST_LIST_ARGS=(--hist-list cr)
  else
    HIST_LIST_ARGS=(--hist-list ana)
  fi

  echo "Resolved region: $REGION_LABEL"
  echo "Resolved histogram list: ${HIST_LIST_ARGS[*]:1}"
  echo "Resolved output path: $RESOLVED_OUTPATH"
  echo "Resolved ttgamma sample-role policy: $TTGAMMA_SAMPLE_ROLE_POLICY"

  # Define options based on mode
  local -a OPTIONS
  if [[ "$FLAG_CR" == "true" ]]; then
    OPTIONS=(
      "${HIST_LIST_ARGS[@]}"
      --skip-sr
    )
    if [[ "$USER_CHUNK_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-s 100000)
    fi
    if [[ "$USER_OUTPATH_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-p "$RESOLVED_OUTPATH")
    fi
    OPTIONS+=(
      #--split-lep-flavor
      -o "$OUT_NAME"
      -x work_queue
    )
  else
    OPTIONS=(
      "${HIST_LIST_ARGS[@]}"
      --skip-cr
      --do-systs
      --do-np
    )
    if [[ "$USER_OUTPATH_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-p "$RESOLVED_OUTPATH")
    fi
    if [[ "$USER_CHUNK_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-s 100000)
    fi
    OPTIONS+=(-o "$OUT_NAME")
  fi

  # Build and run the command
  local -a RUN_CMD=(python run_analysis.py "$CFGS")
  RUN_CMD+=(--years "${RESOLVED_YEARS[@]}")
  RUN_CMD+=("${OPTIONS[@]}")
  RUN_CMD+=(--sample-universe-wrapper fullR3_run.sh)
  RUN_CMD+=(--ttgamma-sample-role-policy "$TTGAMMA_SAMPLE_ROLE_POLICY")
  if [[ "$FLAG_DEFER_NP" == "true" ]]; then
    RUN_CMD+=(--np-postprocess=defer)
  fi
  RUN_CMD+=("${EXTRA_ARGS[@]}")

  printf "\nRunning the following command:\n%s\n\n" "${RUN_CMD[*]}"

  if [[ "$FLAG_DRY_RUN" == "true" ]]; then
    return 0
  fi

  time "${RUN_CMD[@]}"
}

main "$@"
exit_code=$?
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exit "${exit_code}"
else
  return "${exit_code}"
fi
