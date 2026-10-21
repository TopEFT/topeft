#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: run_excluded_mc.sh --run {run2|run3} --block {a|d|all} --campaign-tag TAG
  --output-root ABS_PATH --wq-staging-base ABS_PATH --wq-quota-root ABS_PATH
  --wq-free-warning-bytes N --wq-min-free-bytes N --wq-poll-seconds N
  --env-file ABS_PATH [--snapshot] [--port PORT | --port-a PORT --port-d PORT]
  (--dry-run | --execute)

--block all is Run3 only and executes a, then d after a returns success.
--port is for one block; --port-a and --port-d bind the two-unit plan.
Dry-run accepts synthetic paths and does not start production. --snapshot uses
the explicit archive after native integrity validation in the entrypoint.
Execute requires the environment archive.
USAGE
}

run="" block="" campaign_tag="" output_root="" port="" port_a="" port_d=""
wq_staging_base="" wq_quota_root="" wq_free_warning_bytes=""
wq_min_free_bytes="" wq_poll_seconds="" env_file=""
manager_name_assertion="" dry_run=false execute=false snapshot=false

while (($#)); do
  case "$1" in
    --run|--block|--campaign-tag|--output-root|--port|--port-a|--port-d|--wq-staging-base|--wq-quota-root|--wq-free-warning-bytes|--wq-min-free-bytes|--wq-poll-seconds|--env-file|--manager-name)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then usage; exit 2; fi
      case "$1" in
        --run) run="$2" ;; --block) block="$2" ;; --campaign-tag) campaign_tag="$2" ;;
        --output-root) output_root="$2" ;; --port) port="$2" ;;
        --port-a) port_a="$2" ;; --port-d) port_d="$2" ;;
        --wq-staging-base) wq_staging_base="$2" ;; --wq-quota-root) wq_quota_root="$2" ;;
        --wq-free-warning-bytes) wq_free_warning_bytes="$2" ;;
        --wq-min-free-bytes) wq_min_free_bytes="$2" ;; --wq-poll-seconds) wq_poll_seconds="$2" ;;
        --env-file) env_file="$2" ;;
        --manager-name) manager_name_assertion="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --execute) execute=true; shift ;;
    --snapshot) snapshot=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$dry_run" != "$execute" ]] || { echo 'Choose exactly one of --dry-run and --execute.' >&2; exit 2; }
[[ "$run" == run2 || "$run" == run3 ]] || { echo 'Run must be run2 or run3.' >&2; exit 2; }
[[ "$block" == a || "$block" == d || "$block" == all ]] || { echo 'Block must be a, d, or all.' >&2; exit 2; }
[[ "$block" != all || "$run" == run3 ]] || { echo 'The all-block campaign is Run3 only.' >&2; exit 2; }
[[ "$campaign_tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid campaign tag.' >&2; exit 2; }
[[ "$snapshot" != true || -n "$env_file" ]] || { echo '--snapshot requires an explicit --env-file.' >&2; exit 2; }
for path in "$output_root" "$wq_staging_base" "$wq_quota_root" "$env_file"; do
  [[ "$path" == /* ]] || { echo 'Output, staging, quota, and environment paths must be absolute.' >&2; exit 2; }
done
for amount in "$wq_free_warning_bytes" "$wq_min_free_bytes"; do
  [[ "$amount" =~ ^[0-9]+$ ]] && ((10#$amount > 0)) || { echo 'Work Queue thresholds must be positive integers.' >&2; exit 2; }
done
[[ "$wq_poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'Invalid Work Queue poll interval.' >&2; exit 2; }
if [[ "$block" == all ]]; then
  [[ -z "$port" && -n "$port_a" && -n "$port_d" && "$port_a" != "$port_d" && -z "$manager_name_assertion" ]] || { echo 'All-block execution requires distinct --port-a and --port-d, without --port or --manager-name.' >&2; exit 2; }
  blocks=(a d)
else
  [[ -z "$port_a" && -z "$port_d" && -n "$port" ]] || { echo 'Single-block execution requires --port only.' >&2; exit 2; }
  blocks=("$block")
fi
for unit_port in "$port" "$port_a" "$port_d"; do
  [[ -z "$unit_port" ]] && continue
  [[ "$unit_port" =~ ^[0-9]+$ ]] && ((10#$unit_port >= 1 && 10#$unit_port <= 65535)) || { echo 'Port must be 1 through 65535.' >&2; exit 2; }
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
cfg_path="$repo_dir/input_samples/cfgs/excluded_mc_${run}_sr.cfg"
runner="$script_dir/fullR3_run_excluded_mc.sh"
entrypoint="$script_dir/run_analysis_excluded_mc.py"
[[ -f "$cfg_path" && -f "$runner" && -f "$entrypoint" ]] || { echo 'Required dedicated CFG, wrapper, or entrypoint is absent.' >&2; exit 2; }
manager_selector="^${campaign_tag}-${run}-block-($(IFS='|'; echo "${blocks[*]}"))\$"
printf 'EXCLUDED_MC_MANAGER_SELECTOR %s\n' "$manager_selector"
printf 'EXCLUDED_MC_RUN2_STATUS deferred_TTToHadronic_unavailable\n'
printf 'EXCLUDED_MC_ENV_FILE %s\n' "$env_file"
if [[ "$snapshot" == true ]]; then
  printf 'EXCLUDED_MC_ENV_MODE explicit_snapshot\n'
else
  printf 'EXCLUDED_MC_ENV_MODE current_source\n'
fi
printf 'EXCLUDED_MC_STAGING_BASE %s\n' "$wq_staging_base"
printf 'EXCLUDED_MC_QUOTA_ROOT %s\n' "$wq_quota_root"

declare -a manager_names=() output_pkls=() output_dirs=() log_dirs=() unit_ports=()
declare -A seen_managers=() seen_outputs=()
for unit_block in "${blocks[@]}"; do
  manager_name="${campaign_tag}-${run}-block-${unit_block}"
  [[ "$manager_name" =~ ^[A-Za-z0-9_-]+$ && "${manager_name,,}" == *excludedmc* ]] || { echo 'Invalid derived manager name.' >&2; exit 2; }
  user_lower="${USER:-}"
  [[ -z "$user_lower" || "${manager_name,,}" != *"${user_lower,,}"* ]] || { echo 'Manager name includes user name.' >&2; exit 2; }
  if [[ -n "$manager_name_assertion" && "$manager_name_assertion" != "$manager_name" ]]; then
    echo 'The supplied manager name disagrees with the derived campaign identity.' >&2; exit 2
  fi
  outname="excluded_mc_${run}_block_${unit_block}_njets"
  output_dir="$output_root/$run"
  output_pkl="$output_dir/$outname.pkl.gz"
  [[ -z "${seen_managers[$manager_name]+present}" && -z "${seen_outputs[$output_pkl]+present}" ]] || { echo 'Duplicate manager or output in campaign plan.' >&2; exit 2; }
  seen_managers[$manager_name]=yes
  seen_outputs[$output_pkl]=yes
  log_dir="$output_dir/work_queue_logs/$outname/source_native"
  manager_names+=("$manager_name") output_pkls+=("$output_pkl")
  output_dirs+=("$output_dir") log_dirs+=("$log_dir")
  if [[ "$block" == all ]]; then
    [[ "$unit_block" == a ]] && unit_ports+=("$port_a") || unit_ports+=("$port_d")
  else
    unit_ports+=("$port")
  fi
done

if [[ "$execute" == true ]]; then
  [[ -f "$env_file" ]] || { echo 'Environment archive is absent.' >&2; exit 2; }
  for index in "${!blocks[@]}"; do
    for target in "${output_pkls[index]}" "${output_pkls[index]}.metadata.json"; do
      [[ ! -e "$target" && ! -L "$target" ]] || { echo "Refusing collision: $target" >&2; exit 2; }
    done
  done
fi

for index in "${!blocks[@]}"; do
  unit_block="${blocks[index]}"
  manager_name="${manager_names[index]}"
  output_dir="${output_dirs[index]}"
  output_pkl="${output_pkls[index]}"
  log_dir="${log_dirs[index]}"
  unit_port="${unit_ports[index]}"
  case "$unit_block" in
    a) category_groups=(2l 2los_1tau); hist_vars=(njets lj0pt ptz lt) ;;
    d) category_groups=(3l_onZ_tau); hist_vars=(njets lj0pt ptz lt) ;;
  esac
  if [[ "$run" == run2 ]]; then
    hist_vars=(njets)
  fi
  outname="excluded_mc_${run}_block_${unit_block}_njets"
  command=("$runner" -y "$run" --sr --hist-vars "${hist_vars[@]}" --output-name "$outname"
    -s 50000 --cfg-override "$cfg_path"
    --sample-universe-wrapper fullR3_run_excluded_mc.sh
    --ttgamma-sample-role-policy split -p "$output_dir" -t "$campaign_tag"
    --all-analysis --category-groups "${category_groups[@]}"
    --wq-manager-name "$manager_name" --wq-processing-cores 1
    --wq-accumulating-memory-mb 32000 --wq-accumulating-cores 2
    --np-postprocess=skip -x work_queue --port "$unit_port" --wq-guard
    --wq-filepath "$wq_staging_base" --wq-invocation-id "${campaign_tag}_${run}_block_${unit_block}"
    --wq-quota-root "$wq_quota_root" --wq-free-warning-bytes "$wq_free_warning_bytes"
    --wq-min-free-bytes "$wq_min_free_bytes" --wq-poll-seconds "$wq_poll_seconds"
    --wq-log-dir "$log_dir" --env-file "$env_file")
  for hist_var in "${hist_vars[@]}"; do
    command+=(--sumw2-family "$hist_var")
  done
  if [[ "$run" == run3 ]]; then
    command+=(--do-systs)
  fi
  if [[ "$snapshot" == true ]]; then
    command+=(--snapshot)
  fi
  printf 'EXCLUDED_MC_UNIT %s/%s\n' "$run" "$unit_block"
  printf 'EXCLUDED_MC_MANAGER %s\n' "$manager_name"
  printf 'EXCLUDED_MC_OUTPUT %s\n' "$output_pkl"
  printf 'EXCLUDED_MC_LOG_DIR %s\n' "$log_dir"
  printf 'EXCLUDED_MC_PORT %s\n' "$unit_port"
  printf 'EXCLUDED_MC_CATEGORIES %s\n' "${category_groups[*]}"
  printf 'EXCLUDED_MC_ARGV '; printf '%q ' "${command[@]}"; printf '\n'
  if [[ "$dry_run" == true ]]; then
    (cd "$script_dir" && "${command[@]}" --dry-run)
    continue
  fi
  for target in "$output_pkl" "$output_pkl.metadata.json"; do
    [[ ! -e "$target" && ! -L "$target" ]] || { echo "Refusing fresh collision: $target" >&2; exit 1; }
  done
  mkdir -p "$output_dir"
  # Like run_cr.sh, the foreground child inherits both terminal streams.
  if (cd "$script_dir" && "${command[@]}"); then
    child_rc=0
  else
    child_rc=$?
  fi
  [[ "$child_rc" == 0 ]] || { echo "Unit $unit_block failed with rc $child_rc; later units were not started." >&2; exit "$child_rc"; }
done
if [[ "$dry_run" == true ]]; then
  printf 'EXCLUDED_MC_CAMPAIGN_STATUS planned_only units=%s\n' "${#blocks[@]}"
else
  printf 'EXCLUDED_MC_CAMPAIGN_STATUS complete units=%s\n' "${#blocks[@]}"
fi
