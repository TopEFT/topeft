#!/usr/bin/env bash

# PrintUsage: display script usage information
PrintUsage() {
  echo "Usage: $0 [-y YEAR [YEAR ...]] [-t TAG] --cr | --sr [run_analysis options]"
  echo
  echo "Options:"
  echo "  -y YEAR    Year identifier (repeat or list multiple years)"
  echo "  -t TAG     Git tag or commit identifier"
  echo "  --cr       Generate control-region histograms"
  echo "  --sr       Generate signal-region histograms"
  echo "  -h, --help Show this help message"
  echo
  echo "Any additional options after those listed above are passed directly"
  echo "to run_analysis.py, allowing access to its full set of arguments."
}

# Default values
DEFAULT_YEAR="2022"
DEFAULT_TAG="fec79a60_PNet"
FLAG_CR=false
FLAG_SR=false
EXTRA_ARGS=()
YEARS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--year)
      shift
      if [[ $# -eq 0 || "$1" == -* ]]; then
        echo "Error: -y|--year requires at least one argument"
        exit 1
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
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Ensure exactly one mode is chosen
if [[ "$FLAG_CR" == "false" && "$FLAG_SR" == "false" ]] || [[ "$FLAG_CR" == "true" && "$FLAG_SR" == "true" ]]; then
  echo "Error: You must specify exactly one of --cr or --sr."
  echo
  PrintUsage
  exit 1
fi

# Apply defaults with warnings if not provided
if [[ ${#YEARS[@]} -eq 0 ]]; then
  echo "Warning: YEAR not provided, using default YEAR=$DEFAULT_YEAR"
  YEARS=("$DEFAULT_YEAR")
fi

if [[ -z "$TAG" ]]; then
  echo "Warning: TAG not provided, using default TAG=$DEFAULT_TAG"
  TAG="$DEFAULT_TAG"
fi

# Define output name based on mode
YEAR_LABEL=$(IFS=-; echo "${YEARS[*]}")

if [[ "$FLAG_CR" == "true" ]]; then
  OUT_NAME="${YEAR_LABEL}CRs_${TAG}"
else
  OUT_NAME="${YEAR_LABEL}SRs_${TAG}"
fi

echo "OUT_NAME: $OUT_NAME"

# Build the configuration file list
CFGS_PATH="../../input_samples/cfgs"
CFGS_LIST=()

declare -A RUN2_YEAR_MAP=(
  [2016]=1
  [2016APV]=1
  [2017]=1
  [2018]=1
)

RUN2_CFGS_SR=(
  "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
  "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
  "${CFGS_PATH}/data_samples_NDSkim.cfg"
)

RUN2_CFGS_CR=(
  "${CFGS_PATH}/mc_signal_samples_NDSkim.cfg"
  "${CFGS_PATH}/mc_background_samples_NDSkim.cfg"
  "${CFGS_PATH}/mc_background_samples_cr_NDSkim.cfg"
  "${CFGS_PATH}/data_samples_NDSkim.cfg"
)

declare -A SEEN_CFGS=()
RUN2_BUNDLE_ADDED=false

add_cfg() {
  local cfg_file="$1"
  if [[ ! -f "$cfg_file" ]]; then
    echo "Warning: Missing cfg file: $cfg_file" >&2
    return
  fi
  if [[ -n "${SEEN_CFGS[$cfg_file]}" ]]; then
    return
  fi
  CFGS_LIST+=("$cfg_file")
  SEEN_CFGS[$cfg_file]=1
}

for YEAR in "${YEARS[@]}"; do
  if [[ -n "${RUN2_YEAR_MAP[$YEAR]}" ]]; then
    if [[ "$RUN2_BUNDLE_ADDED" == "false" ]]; then
      if [[ "$FLAG_CR" == "true" ]]; then
        for CFG in "${RUN2_CFGS_CR[@]}"; do
          add_cfg "$CFG"
        done
      else
        for CFG in "${RUN2_CFGS_SR[@]}"; do
          add_cfg "$CFG"
        done
      fi
      RUN2_BUNDLE_ADDED=true
    fi
    continue
  fi

  YEAR_CFGS=(
    "${CFGS_PATH}/NDSkim_${YEAR}_signal_samples.cfg"
    "${CFGS_PATH}/NDSkim_${YEAR}_background_samples.cfg"
    "${CFGS_PATH}/NDSkim_${YEAR}_data_samples.cfg"
  )

  for CFG in "${YEAR_CFGS[@]}"; do
    add_cfg "$CFG"
  done
done
CFGS=$(IFS=,; echo "${CFGS_LIST[*]}")

echo "Resolved CFGS: $CFGS"

# Define options based on mode
if [[ "$FLAG_CR" == "true" ]]; then
  OPTIONS=(
    --hist-list cr
    --skip-sr
    -s 50000
    --split-lep-flavor
    -p "/scratch365/$USER/"
    -o "$OUT_NAME"
    -x work_queue
  )
else
  OPTIONS=(
    --hist-list ana
    --skip-cr
    --do-systs
    -s 50000
    -o "$OUT_NAME"
  )
fi

# Build and run the command
RUN_CMD=(python run_analysis.py "$CFGS")
RUN_CMD+=("${OPTIONS[@]}")
RUN_CMD+=("${EXTRA_ARGS[@]}")

printf "\nRunning the following command:\n%s\n\n" "${RUN_CMD[*]}"

time "${RUN_CMD[@]}"
