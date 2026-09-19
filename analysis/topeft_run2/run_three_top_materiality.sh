#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_three_top_materiality.sh --output-dir PATH --campaign-tag TAG [--env-file PATH] [--dry-run]
EOF
}

output_dir=""
campaign_tag=""
env_file=""
dry_run=false

while (($#)); do
  case "$1" in
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --campaign-tag)
      campaign_tag="${2:-}"
      shift 2
      ;;
    --env-file)
      env_file="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$output_dir" || -z "$campaign_tag" ]]; then
  echo "ERROR: --output-dir and --campaign-tag are required" >&2
  usage >&2
  exit 2
fi
if [[ ! "$campaign_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: --campaign-tag must use only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cfg_path="$script_dir/../../input_samples/cfgs/three_top_materiality_ul18_ndskim.cfg"
runner="$script_dir/fullR3_run_three_top.sh"

block_ids=(a b c d e)
category_sets=(
  "2l 2lss_1tau 2los_1tau 4l"
  "3l_m_offZ"
  "3l_p_offZ"
  "3l_onZ_tau"
  "3l_fwd"
)
histogram_sets=(
  "njets lj0pt ptz ptz_wtau lt"
  "njets lj0pt ptll lt"
  "njets lj0pt ptll lt"
  "njets lj0pt ptz lt"
  "njets lj0pt ptz lt"
)

for index in "${!block_ids[@]}"; do
  block_id="${block_ids[$index]}"
  outname="${campaign_tag}_ul18_three_top_block_${block_id}"
  read -r -a categories <<< "${category_sets[$index]}"
  read -r -a histograms <<< "${histogram_sets[$index]}"
  command=(
    "$runner"
    -y 2018
    --sr
    --hist-vars "${histograms[@]}"
    --cfg-override "$cfg_path"
    --sample-universe-wrapper fullR3_run_three_top.sh
    -p "$output_dir"
    -t "$outname"
    --all-analysis
    --category-groups "${categories[@]}"
    --no-sumw2
    -x work_queue
  )
  if [[ -n "$env_file" ]]; then
    command+=(--env-file "$env_file")
  fi
  if [[ "$dry_run" == true ]]; then
    command+=(--dry-run)
  fi
  printf 'THREE_TOP_MATERIALITY_BLOCK_COMMAND\t%s\t' "$block_id"
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
done
