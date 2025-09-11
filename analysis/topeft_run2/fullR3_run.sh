#!/usr/bin/env bash

# PrintUsage: display script usage information
PrintUsage() {
  echo "Usage: $0 [-y YEAR] [-t TAG] [-s CHUNKSIZE] [-c NCHUNKS] [-x EXECUTOR] --cr | --sr [run_analysis options]"
  echo
  echo "Options:"
  echo "  -y YEAR       Year identifier (e.g., 2022, 2022EE, 2023, 2023BPix)"
  echo "  -t TAG        Git tag or commit identifier"
  echo "  -s CHUNKSIZE  Number of events per chunk (default: 100000)"
  echo "  -c NCHUNKS    Number of chunks to run (default: all)"
  echo "  -x EXECUTOR   Executor manager (default: work_queue)"
  echo "  --cr          Generate control-region histograms"
  echo "  --sr          Generate signal-region histograms"
  echo "  -h, --help    Show this help message"
  echo
  echo "Any additional options after those listed above are passed directly"
  echo "to run_analysis.py, allowing access to its full set of arguments."
}

# RequireValue: ensure flags that take a value actually receive one
RequireValue() {
  # $1 = flag (e.g., -y), $2 = next token
  [[ -n "$2" && "$2" != -* ]] || { echo "Error: $1 requires a value"; exit 1; }
}

# Default values
DEFAULT_YEAR="2022"
DEFAULT_TAG="fec79a60_PNet"
DEFAULT_CHUNKSIZE=100000
DEFAULT_NCHUNKS=""
DEFAULT_EXEC_MAN="work_queue"

FLAG_CR=false
FLAG_SR=false
CHUNKSIZE="$DEFAULT_CHUNKSIZE"
NCHUNKS="$DEFAULT_NCHUNKS"
EXEC_MAN="$DEFAULT_EXEC_MAN"

EXTRA_ARGS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y)
      RequireValue "$1" "$2"
      YEAR="$2"
      shift 2
      ;;
    -t)
      RequireValue "$1" "$2"
      TAG="$2"
      shift 2
      ;;
    -s)
      RequireValue "$1" "$2"
      CHUNKSIZE="$2"
      shift 2
      ;;
    -c)
      RequireValue "$1" "$2"
      NCHUNKS="$2"
      shift 2
      ;;
    -x)
      RequireValue "$1" "$2"
      EXEC_MAN="$2"
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
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      # Pass through unknown flags/positionals to the underlying script
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Ensure exactly one mode is chosen
if { [[ "$FLAG_CR" == false ]] && [[ "$FLAG_SR" == false ]] ; } || \
   { [[ "$FLAG_CR" == true  ]] && [[ "$FLAG_SR" == true  ]] ; }; then
  echo "Error: You must specify exactly one of --cr or --sr."
  echo
  PrintUsage
  exit 1
fi

# Apply defaults with warnings if not provided
if [[ -z "${YEAR:-}" ]]; then
  echo "Warning: YEAR not provided, using default YEAR=$DEFAULT_YEAR"
  YEAR="$DEFAULT_YEAR"
fi

if [[ -z "${TAG:-}" ]]; then
  echo "Warning: TAG not provided, using default TAG=$DEFAULT_TAG"
  TAG="$DEFAULT_TAG"
fi

# Define output name based on mode
if [[ "$FLAG_CR" == true ]]; then
  OUT_NAME="${YEAR}CRs_${TAG}"
else
  OUT_NAME="${YEAR}SRs_${TAG}"
fi

echo "OUT_NAME: $OUT_NAME"
echo "EXECUTOR: $EXEC_MAN"

# Build the configuration file list (comma-separated, passed as a single arg)
CFGS_PATH="../../input_samples/cfgs"
CFGS="${CFGS_PATH}/NDSkim_${YEAR}_background_samples.cfg,${CFGS_PATH}/NDSkim_${YEAR}_data_samples.cfg,${CFGS_PATH}/NDSkim_${YEAR}_signal_samples.cfg"

# Define options based on mode (use arrays for safe quoting)
if [[ "$FLAG_CR" == true ]]; then
  OPTIONS=(--hist-list cr --skip-sr --split-lep-flavor -p "/scratch365/$USER/" -o "$OUT_NAME" --do-np)
else
  OPTIONS=(--hist-list ana --skip-cr --do-systs --do-np -o "$OUT_NAME")
fi

# Build the command
RUN_CMD=(python run_analysis.py "$CFGS" "${OPTIONS[@]}" -s "$CHUNKSIZE" -x "$EXEC_MAN")
if [[ -n "$NCHUNKS" ]]; then
  RUN_CMD+=(-c "$NCHUNKS")
fi
RUN_CMD+=("${EXTRA_ARGS[@]}")

printf "\nRunning the following command:\n%s\n\n" "${RUN_CMD[*]}"

# Execute
time "${RUN_CMD[@]}"