#!/usr/bin/env bash

# PrintUsage: display script usage information
PrintUsage() {
  echo "Usage: $0 [-y YEAR [YEAR ...]] [-t TAG] --cr | --sr [run_analysis options]"
  echo
  echo "Options:"
  echo "  -y YEAR    Year identifier (repeat or list multiple years)"
  echo "             Bundles: run2 -> UL16 UL16APV UL17 UL18;"
  echo "                      run3 -> 2022 2022EE 2023 2023BPix"
  echo "  -t TAG     Git tag or commit identifier"
  echo "  --cr       Generate control-region histograms"
  echo "  --sr       Generate signal-region histograms"
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
  local -a EXTRA_ARGS=()
  local -a YEARS=()
  local -a EXPANDED_YEARS=()
  local -a RESOLVED_YEARS=()
  local USER_CHUNK_OVERRIDE=false
  local TAG=""

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
  for ARG in "${EXTRA_ARGS[@]}"; do
    case "$ARG" in
      -s|--chunksize|--chunksize=*)
        USER_CHUNK_OVERRIDE=true
        break
        ;;
    esac
  done

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

  if [[ -z "$TAG" ]]; then
    echo "Warning: TAG not provided, using default TAG=$DEFAULT_TAG"
    TAG="$DEFAULT_TAG"
  fi

  # Define output name based on mode
  local YEAR_LABEL
  YEAR_LABEL=$(IFS=-; echo "${RESOLVED_YEARS[*]}")

  local OUT_NAME
  if [[ "$FLAG_CR" == "true" ]]; then
    OUT_NAME="${YEAR_LABEL}CRs_${TAG}"
  else
    OUT_NAME="${YEAR_LABEL}SRs_${TAG}"
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

  local RUN2_CFGS_SR=(
    "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
    "${CFGS_PATH}/data_samples_NDSkim.cfg"
  )

  local RUN2_CFGS_CR=(
    "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
    "${CFGS_PATH}/mc_background_samples_cr_NDSkim.cfg"
    "${CFGS_PATH}/data_samples_NDSkim.cfg"
  )

  declare -A SEEN_CFGS=()
  local RUN2_BUNDLE_ADDED=false

  add_cfg() {
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

  local CFG YEAR_CFGS
  for YEAR in "${RESOLVED_YEARS[@]}"; do
    if [[ -n "${RUN2_YEAR_MAP[$YEAR]}" ]]; then
      if [[ "$RUN2_BUNDLE_ADDED" == "false" ]]; then
        if [[ "$FLAG_CR" == "true" ]]; then
          for CFG in "${RUN2_CFGS_CR[@]}"; do
            add_cfg "$CFG" || return 1
          done
        else
          for CFG in "${RUN2_CFGS_SR[@]}"; do
            add_cfg "$CFG" || return 1
          done
        fi
        RUN2_BUNDLE_ADDED=true
      fi
    else
      YEAR_CFGS=(
        "${CFGS_PATH}/NDSkim_${YEAR}_signal_samples.cfg"
        "${CFGS_PATH}/NDSkim_${YEAR}_background_samples.cfg"
        "${CFGS_PATH}/NDSkim_${YEAR}_data_samples.cfg"
      )

      for CFG in "${YEAR_CFGS[@]}"; do
        add_cfg "$CFG" || return 1
      done
    fi
  done
  local CFGS
  CFGS=$(IFS=,; echo "${CFGS_LIST[*]}")

  echo "Resolved years: ${RESOLVED_YEARS[*]}"
  echo "Resolved CFGS: $CFGS"

  # Define options based on mode
  local -a OPTIONS
  if [[ "$FLAG_CR" == "true" ]]; then
    OPTIONS=(
      --hist-list cr
      --skip-sr
    )
    if [[ "$USER_CHUNK_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-s 50000)
    fi
    OPTIONS+=(
      #--split-lep-flavor
      -p "/scratch365/$USER/"
      -o "$OUT_NAME"
      -x work_queue
    )
  else
    OPTIONS=(
      --hist-list ana
      --skip-cr
      --do-systs
    )
    if [[ "$USER_CHUNK_OVERRIDE" == "false" ]]; then
      OPTIONS+=(-s 50000)
    fi
    OPTIONS+=(-o "$OUT_NAME")
  fi

  # Build and run the command
  local -a RUN_CMD=(python run_analysis.py "$CFGS")
  RUN_CMD+=(--years "${RESOLVED_YEARS[@]}")
  RUN_CMD+=("${OPTIONS[@]}")
  RUN_CMD+=("${EXTRA_ARGS[@]}")

  printf "\nRunning the following command:\n%s\n\n" "${RUN_CMD[*]}"

  time "${RUN_CMD[@]}"
}

main "$@"
exit_code=$?
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exit "${exit_code}"
else
  return "${exit_code}"
fi
