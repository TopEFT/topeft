#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

if ! python -c "import topcoffea" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Missing topcoffea dependency.
Install the sibling checkout with `pip install -e ../topcoffea` (or any other
source that provides `import topcoffea`) before running this smoke test.
EOF
  exit 1
fi

PKL_PATH=${1:-"${PROJECT_ROOT}/example_outputs_taskvine/plotsTopEFT.pkl.gz"}
OUTDIR=$(mktemp -d -t cr_sr_plots_XXXX)

if [[ ! -f "${PKL_PATH}" ]]; then
  echo "Expected a TaskVine/topcoffea 5-tuple histogram pickle at ${PKL_PATH}." >&2
  echo "Pass the pickle path explicitly if it lives elsewhere." >&2
  exit 1
fi

echo "Using histogram pickle: ${PKL_PATH}" >&2
python make_cr_and_sr_plots.py \
  -f "${PKL_PATH}" \
  -o "${OUTDIR}" \
  -n taskvine_smoke \
  -y 2018 \
  --skip-syst

echo "Plots written to ${OUTDIR}" >&2
