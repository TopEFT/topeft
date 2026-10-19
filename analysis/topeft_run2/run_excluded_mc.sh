#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: run_excluded_mc.sh --run {run2|run3} --block {a|d|all} --campaign-tag TAG
  --output-root ABS_PATH --wq-staging-base ABS_PATH --wq-quota-root ABS_PATH
  --wq-free-warning-bytes N --wq-min-free-bytes N --wq-poll-seconds N
  --env-file ABS_PATH [--snapshot] [--port PORT | --port-a PORT --port-d PORT]
  [--expected-source-head SHA] (--dry-run | --execute)

--block all is Run3 only and executes a, then d after validated completion.
--port is for one block; --port-a and --port-d bind the two-unit plan.
Dry-run accepts synthetic paths and does not start production. --snapshot uses
the explicit archive after native integrity validation in the entrypoint.
Execute requires an existing output root, environment archive, and exact
committed source HEAD.
USAGE
}

run="" block="" campaign_tag="" output_root="" port="" port_a="" port_d=""
wq_staging_base="" wq_quota_root="" wq_free_warning_bytes=""
wq_min_free_bytes="" wq_poll_seconds="" env_file="" expected_source_head=""
manager_name_assertion="" dry_run=false execute=false snapshot=false

while (($#)); do
  case "$1" in
    --run|--block|--campaign-tag|--output-root|--port|--port-a|--port-d|--wq-staging-base|--wq-quota-root|--wq-free-warning-bytes|--wq-min-free-bytes|--wq-poll-seconds|--env-file|--expected-source-head|--manager-name)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then usage; exit 2; fi
      case "$1" in
        --run) run="$2" ;; --block) block="$2" ;; --campaign-tag) campaign_tag="$2" ;;
        --output-root) output_root="$2" ;; --port) port="$2" ;;
        --port-a) port_a="$2" ;; --port-d) port_d="$2" ;;
        --wq-staging-base) wq_staging_base="$2" ;; --wq-quota-root) wq_quota_root="$2" ;;
        --wq-free-warning-bytes) wq_free_warning_bytes="$2" ;;
        --wq-min-free-bytes) wq_min_free_bytes="$2" ;; --wq-poll-seconds) wq_poll_seconds="$2" ;;
        --env-file) env_file="$2" ;; --expected-source-head) expected_source_head="$2" ;;
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
printf 'EXCLUDED_MC_SOURCE_HEAD_BINDING %s\n' "${expected_source_head:-unbound_dry_run}"

declare -a manager_names=() output_pkls=() output_dirs=() log_roots=() log_dirs=() unit_logs=() unit_ports=()
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
  log_root="$output_dir/work_queue_logs/$outname"
  unit_log="$output_dir/$outname.runner.log"
  manager_names+=("$manager_name") output_pkls+=("$output_pkl")
  output_dirs+=("$output_dir") log_roots+=("$log_root") log_dirs+=("$log_dir") unit_logs+=("$unit_log")
  if [[ "$block" == all ]]; then
    [[ "$unit_block" == a ]] && unit_ports+=("$port_a") || unit_ports+=("$port_d")
  else
    unit_ports+=("$port")
  fi
done

if [[ "$execute" == true ]]; then
  [[ "$expected_source_head" =~ ^[0-9a-f]{40}$ ]] || { echo 'Execute requires --expected-source-head SHA.' >&2; exit 2; }
  observed_head=$(git -C "$repo_dir" rev-parse HEAD)
  [[ "$observed_head" == "$expected_source_head" ]] || { echo 'Source HEAD mismatch.' >&2; exit 2; }
  git -C "$repo_dir" diff --quiet HEAD -- "${cfg_path#"$repo_dir/"}" analysis/topeft_run2/run_excluded_mc.sh analysis/topeft_run2/fullR3_run_excluded_mc.sh analysis/topeft_run2/run_analysis_excluded_mc.py || { echo 'Dedicated producer has uncommitted changes.' >&2; exit 2; }
  [[ -d "$output_root" && ! -L "$output_root" && ! -L "$output_root/$run" && -f "$env_file" && ! -L "$env_file" ]] || { echo 'Output root or environment archive is absent/unsafe.' >&2; exit 2; }
  exec {namespace_lock_fd}< "$output_root"
  flock -n "$namespace_lock_fd" || { echo 'Campaign output namespace is in use.' >&2; exit 2; }
  env_sha256=$(sha256sum "$env_file"); env_sha256="${env_sha256%% *}"
  cfg_sha256=$(sha256sum "$cfg_path"); cfg_sha256="${cfg_sha256%% *}"
  for index in "${!blocks[@]}"; do
    for target in "${output_pkls[index]}" "${output_pkls[index]}.metadata.json" "${log_roots[index]}" "${log_dirs[index]}" "${unit_logs[index]}"; do
      [[ ! -e "$target" && ! -L "$target" ]] || { echo "Refusing collision: $target" >&2; exit 2; }
    done
  done
fi

child_pid=""
stop_child() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=""
  fi
}
on_interrupt() {
  printf 'EXCLUDED_MC_INTERRUPTED signal=%s active_child=%s\n' "$1" "${child_pid:-none}" >&2
  stop_child
  exit 130
}
trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM
trap 'on_interrupt HUP' HUP

verify_completed_unit() {
  local index="$1" pkl="${output_pkls[index]}" sidecar="${output_pkls[index]}.metadata.json"
  [[ -f "$pkl" && ! -L "$pkl" && -f "$sidecar" && ! -L "$sidecar" ]] || return 1
  local pkl_size pkl_sha current_env_sha current_cfg_sha
  pkl_size=$(stat -c '%s' "$pkl")
  pkl_sha=$(sha256sum "$pkl"); pkl_sha="${pkl_sha%% *}"
  current_env_sha=$(sha256sum "$env_file"); current_env_sha="${current_env_sha%% *}"
  current_cfg_sha=$(sha256sum "$cfg_path"); current_cfg_sha="${current_cfg_sha%% *}"
  [[ "$current_env_sha" == "$env_sha256" && "$current_cfg_sha" == "$cfg_sha256" ]] || return 1
  [[ "$(git -C "$repo_dir" rev-parse HEAD)" == "$expected_source_head" ]] || return 1
  jq -e --arg basename "${pkl##*/}" --arg sha "$pkl_sha" --argjson size "$pkl_size" \
    --arg cfg_basename "${cfg_path##*/}" --arg cfg_sha "$cfg_sha256" '
    .metadata_schema_version == 2 and
    .artifact.pkl_basename == $basename and .artifact.pkl_sha256 == $sha and
    .artifact.pkl_size_bytes == $size and .artifact.artifact_kind == "processor_output" and
    .artifact.merged == false and .artifact.nominal_container_schema_version == 2 and
    .artifact.nominal_container_layout == "split_sibling_v1" and
    (.lineage.inputs == []) and (.sumw2_content_manifest != null) and
    (.histogram_applicability != null) and
    .production_sample_contract.wrapper_identity == "fullR3_run_excluded_mc.sh" and
    .production_sample_contract.contract_version == 1 and
    .production_sample_contract.compatibility_validated == true and
    (.production_sample_contract.contract_identity_sha256 | type) == "string" and
    .production_sample_contract.cfg_identities == [{"basename":$cfg_basename,"content_sha256":$cfg_sha,"input_kind":"cfg"}] and
    .sumw2_storage_provenance.requested_mode == "full_custom" and
    .sumw2_storage_provenance.resolved_mode == "full_custom" and
    .sumw2_storage_provenance.runtime_histogram_families == ["njets"] and
    (.sumw2_storage_provenance.resolved_datasets | length) > 0 and
    (.sumw2_storage_provenance.resolved_targets | length) == (.sumw2_storage_provenance.resolved_datasets | length) and
    .requested_data_driven_products.products.nonprompt.enabled == false and
    .requested_data_driven_products.products.flips.enabled == false and
    .resolved_data_driven_contract.products.nonprompt.enabled == false and
    .resolved_data_driven_contract.products.flips.enabled == false
  ' "$sidecar" >/dev/null
}

for index in "${!blocks[@]}"; do
  unit_block="${blocks[index]}"
  manager_name="${manager_names[index]}"
  output_dir="${output_dirs[index]}"
  output_pkl="${output_pkls[index]}"
  log_dir="${log_dirs[index]}"
  unit_log="${unit_logs[index]}"
  unit_port="${unit_ports[index]}"
  case "$unit_block" in a) category_groups=(2l 2los_1tau) ;; d) category_groups=(3l_onZ_tau) ;; esac
  outname="excluded_mc_${run}_block_${unit_block}_njets"
  command=("$runner" -y "$run" --sr --hist-vars njets --output-name "$outname"
    -s 50000 --cfg-override "$cfg_path"
    --sample-universe-wrapper fullR3_run_excluded_mc.sh
    --ttgamma-sample-role-policy split -p "$output_dir" -t "$campaign_tag"
    --all-analysis --category-groups "${category_groups[@]}" --sumw2-family njets
    --wq-manager-name "$manager_name" --wq-processing-cores 1
    --wq-accumulating-memory-mb 32000 --wq-accumulating-cores 2
    --np-postprocess=skip -x work_queue --port "$unit_port" --wq-guard
    --wq-filepath "$wq_staging_base" --wq-invocation-id "${campaign_tag}_${run}_block_${unit_block}"
    --wq-quota-root "$wq_quota_root" --wq-free-warning-bytes "$wq_free_warning_bytes"
    --wq-min-free-bytes "$wq_min_free_bytes" --wq-poll-seconds "$wq_poll_seconds"
    --wq-log-dir "$log_dir" --env-file "$env_file")
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
  for target in "$output_pkl" "$output_pkl.metadata.json" "${log_roots[index]}" "$log_dir" "$unit_log"; do
    [[ ! -e "$target" && ! -L "$target" ]] || { echo "Refusing fresh collision: $target" >&2; exit 1; }
  done
  command -v ss >/dev/null || { echo 'Port inspection command ss is absent.' >&2; exit 1; }
  if ss -H -ltn "sport = :$unit_port" | grep -q .; then
    echo "Port is already listening: $unit_port" >&2; exit 1
  fi
  mkdir -p "$output_dir"
  (set -C; : > "$unit_log") || { echo "Runner log collision: $unit_log" >&2; exit 1; }
  printf 'source_head=%s env_sha256=%s cfg_sha256=%s manager=%s argv=' "$observed_head" "$env_sha256" "$cfg_sha256" "$manager_name" >> "$unit_log"
  printf '%q ' "${command[@]}" >> "$unit_log"
  printf '\nstart_utc=%s\n' "$(date -u +%FT%TZ)" >> "$unit_log"
  (cd "$script_dir" && exec setsid --wait "${command[@]}") >> "$unit_log" 2>&1 &
  child_pid=$!
  child_rc=0
  wait "$child_pid" || child_rc=$?
  child_pid=""
  printf 'end_utc=%s rc=%s\n' "$(date -u +%FT%TZ)" "$child_rc" >> "$unit_log"
  [[ "$child_rc" == 0 ]] || { echo "Unit $unit_block failed with rc $child_rc; later units were not started." >&2; exit 1; }
  verify_completed_unit "$index" || { echo "Unit $unit_block has no validated terminal artifact pair; later units were not started." >&2; exit 1; }
  printf 'validated_pkl_sha256=%s validated_sidecar_sha256=%s status=complete\n' "$(sha256sum "$output_pkl" | cut -d ' ' -f 1)" "$(sha256sum "$output_pkl.metadata.json" | cut -d ' ' -f 1)" >> "$unit_log"
done
if [[ "$dry_run" == true ]]; then
  printf 'EXCLUDED_MC_CAMPAIGN_STATUS planned_only units=%s\n' "${#blocks[@]}"
else
  printf 'EXCLUDED_MC_CAMPAIGN_STATUS complete units=%s\n' "${#blocks[@]}"
fi
