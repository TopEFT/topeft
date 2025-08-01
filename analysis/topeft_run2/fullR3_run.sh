#!/usr/bin/env bash

# PrintUsage: display script usage information
PrintUsage() {
  echo "Usage: $0 [-y YEAR] [-c COMMIT] --cr | --sr"
  echo
  echo "Options:"
  echo "  -y YEAR    Year identifier (e.g., 2022, 2022EE, 2023, 2023BPix)"
  echo "  -c COMMIT  Git commit tag or identifier"
  echo "  --cr       Generate control-region histograms"
  echo "  --sr       Generate signal-region histograms"
  echo "  -h, --help Show this help message"
}

# Default values
DEFAULT_YEAR="2022"
DEFAULT_COMMIT="fec79a60_PNet"
FLAG_CR=false
FLAG_SR=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y)
      YEAR="$2"
      shift 2
      ;;
    -c)
      COMMIT="$2"
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
      echo "Error: Unknown option '$1'"
      PrintUsage
      exit 1
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
if [[ -z "$YEAR" ]]; then
  echo "Warning: YEAR not provided, using default YEAR=$DEFAULT_YEAR"
  YEAR="$DEFAULT_YEAR"
fi

if [[ -z "$COMMIT" ]]; then
  echo "Warning: COMMIT not provided, using default COMMIT=$DEFAULT_COMMIT"
  COMMIT="$DEFAULT_COMMIT"
fi

# Define output name based on mode
if [[ "$FLAG_CR" == "true" ]]; then
  OUT_NAME="${YEAR}CRs_${COMMIT}"
else
  OUT_NAME="${YEAR}SRs_${COMMIT}"
fi

echo "OUT_NAME: $OUT_NAME"

# Build the configuration file list
CFGS_PATH="../../input_samples/cfgs"
CFGS="${CFGS_PATH}/NDSkim_${YEAR}_background_samples.cfg,${CFGS_PATH}/NDSkim_${YEAR}_data_samples.cfg,${CFGS_PATH}/NDSkim_${YEAR}_signal_samples.cfg"

# Define options based on mode
if [[ "$FLAG_CR" == "true" ]]; then
  OPTIONS="--hist-list cr --skip-sr -s 50000 --split-lep-flavor -p /scratch365/$USER/ -o $OUT_NAME -x work_queue --do-np"
else
  OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME"
fi

# Build and run the command
RUN_COMMAND="time python run_analysis.py $CFGS $OPTIONS"

printf "\nRunning the following command:\n$RUN_COMMAND\n\n"

eval $RUN_COMMAND
