#!/usr/bin/env bash
# Public control bootstrap. Never use this wrapper as a batch payload.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
python_bin=${PYTHON_BIN:-${PYTHON:-python}}
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$script_dir/plot_campaign_manifest.py" "$@"
