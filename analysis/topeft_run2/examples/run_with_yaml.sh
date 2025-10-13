#!/usr/bin/env bash
set -euo pipefail

# Demonstrates running run_analysis.py with a YAML metadata file.
# The YAML file now owns the configuration; CLI flags are ignored whenever a
# YAML path is provided, so adjust the options directly in the file.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

YAML_OPTIONS="${SCRIPT_DIR}/yaml_metadata_example.yaml"
EXAMPLE_ROOT="ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root"

if [[ ! -f "${EXAMPLE_ROOT}" ]]; then
  echo "Downloading example ROOT file into ${PROJECT_ROOT}" >&2
  wget -nc "http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/${EXAMPLE_ROOT}"
fi

python run_analysis.py --options "${YAML_OPTIONS}"
