#!/usr/bin/env bash
set -euo pipefail

# This example mirrors the quickstart command from the repository README.
# It downloads a small ROOT file if missing and launches run_analysis.py with
# explicit CLI flags only (no YAML metadata overrides).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

INPUT_JSON="../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json"
EXAMPLE_ROOT="ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root"

if [[ ! -f "${EXAMPLE_ROOT}" ]]; then
  echo "Downloading example ROOT file into ${PROJECT_ROOT}" >&2
  wget -nc "http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/${EXAMPLE_ROOT}"
fi

python run_analysis.py "${INPUT_JSON}" \
  --executor futures \
  --nworkers 1 \
  --chunksize 128000 \
  --outpath example_outputs_cli \
  --outname cli_only_example
