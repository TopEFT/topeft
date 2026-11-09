#!/usr/bin/env bash
set -euo pipefail

# Direct invocations cannot enable the private preparation seam via the environment.
if (( ${#BASH_SOURCE[@]} == 1 )); then
  production_prepare_only=false
fi

original_arguments=("$@")

print_usage() {
  cat <<'EOF'
Usage: ./run_cr.sh --production-profile PROFILE --output-dir PATH --campaign-tag TAG
  [--dry-run] [--env-file PATH] [--resume] [--allow-environment-migration]
  [--chunksize POSITIVE_INTEGER]
  [--selected-fragment ID] [--plan JSON] [--phase-telemetry]
  [--pkl-min-free-bytes POSITIVE_INTEGER --pkl-min-free-inodes POSITIVE_INTEGER
   --pkl-safety-reserve-bytes POSITIVE_INTEGER --pkl-safety-reserve-inodes POSITIVE_INTEGER]
  [--wq-manager-name EXACT_NAME | --wq-manager-prefix PREFIX]
  [--run2-wq-manager-prefix PREFIX --run3-wq-manager-prefix PREFIX]
  [--wq-port PORT (0..65535; 0 selects a free manager port)]
  [--wq-staging-base PATH]
  [--activate-pending-plan JSON --expected-plan-revision ID]
  [Work Queue guard and resource options]

Component profiles:
  run2_prod, run3_prod, run2_prod_CR, run3_prod_CR
  run2_prod_2los_1tau, run3_prod_2los_1tau
  run2_stat_diagnostic, run3_stat_diagnostic
  run2_stat_diagnostic_CR, run3_stat_diagnostic_CR
  run2_sr_ext, run3_sr_ext

Combined profiles:
  run2_run3_prod, run2_run3_prod_CR
  run2_run3_prod_2los_1tau
  run2_run3_stat_diagnostic, run2_run3_stat_diagnostic_CR

Combined profiles orchestrate separate Run 2 and Run 3 component states.
The full scientific plan is persisted even when one unit is selected.
Plan revision preserves completed or verified source-ready unit coverage.
EOF
}

production_profile=""
profile_dry_run=false
profile_output_dir=""
profile_campaign_tag=""
profile_env_file=""
profile_resume=false
profile_allow_environment_migration=false
profile_phase_telemetry=false
profile_chunksize=""
profile_pkl_min_free_bytes=""
profile_pkl_min_free_inodes=""
profile_pkl_safety_reserve_bytes=""
profile_pkl_safety_reserve_inodes=""
profile_wq_staging_base=""
profile_wq_quota_root=""
profile_wq_warning_bytes=""
profile_wq_min_free_bytes=""
profile_wq_poll_seconds=""
profile_wq_treereduction=""
profile_wq_processing_cores=""
profile_wq_accumulating_memory_mb=""
profile_wq_accumulating_cores=""
profile_wq_manager_name=""
profile_wq_port=""
profile_wq_port_provided=false
profile_wq_manager_name_provided=false
profile_wq_manager_prefix=""
profile_wq_manager_prefix_provided=false
profile_run2_wq_manager_prefix=""
profile_run2_wq_manager_prefix_provided=false
profile_run3_wq_manager_prefix=""
profile_run3_wq_manager_prefix_provided=false
profile_selected_fragment=""
profile_plan=""
profile_activate_pending_plan=""
profile_expected_plan_revision=""

while (( $# > 0 )); do
  case "$1" in
    --production-profile)
      if (( $# < 2 )); then
        echo "ERROR: --production-profile requires a value." >&2
        exit 1
      fi
      production_profile="$2"
      shift 2
      ;;
    --phase-telemetry)
      profile_phase_telemetry=true
      shift
      ;;
    --dry-run)
      profile_dry_run=true
      shift
      ;;
    --output-dir)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "ERROR: --output-dir requires a non-empty path." >&2
        exit 1
      fi
      profile_output_dir="$2"
      shift 2
      ;;
    --campaign-tag)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "ERROR: --campaign-tag requires a non-empty value." >&2
        exit 1
      fi
      profile_campaign_tag="$2"
      shift 2
      ;;
    --env-file)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "ERROR: --env-file requires a non-empty path." >&2
        exit 1
      fi
      profile_env_file="$2"
      shift 2
      ;;
    --wq-manager-name)
      if (( $# < 2 )) || [[ -z "$2" ]] || [[ "${profile_wq_manager_name_provided}" == "true" ]]; then
        echo "ERROR: --wq-manager-name requires one non-empty exact name." >&2
        exit 1
      fi
      profile_wq_manager_name="$2"
      profile_wq_manager_name_provided=true
      shift 2
      ;;
    --wq-port)
      if (( $# < 2 )) || [[ "${profile_wq_port_provided}" == "true" ]] \
        || [[ ! "$2" =~ ^(0|[1-9][0-9]{0,4})$ ]] \
        || (( 10#$2 > 65535 )); then
        echo "ERROR: --wq-port requires one integer in [0, 65535]." >&2
        exit 1
      fi
      profile_wq_port="$2"
      profile_wq_port_provided=true
      shift 2
      ;;
    --wq-manager-prefix)
      if (( $# < 2 )) || [[ -z "$2" ]] || [[ "${profile_wq_manager_prefix_provided}" == "true" ]]; then
        echo "ERROR: --wq-manager-prefix requires one non-empty prefix." >&2
        exit 1
      fi
      profile_wq_manager_prefix="$2"
      profile_wq_manager_prefix_provided=true
      shift 2
      ;;
    --run2-wq-manager-prefix)
      if (( $# < 2 )) || [[ -z "$2" ]] || [[ "${profile_run2_wq_manager_prefix_provided}" == "true" ]]; then
        echo "ERROR: --run2-wq-manager-prefix requires one non-empty prefix." >&2
        exit 1
      fi
      profile_run2_wq_manager_prefix="$2"
      profile_run2_wq_manager_prefix_provided=true
      shift 2
      ;;
    --run3-wq-manager-prefix)
      if (( $# < 2 )) || [[ -z "$2" ]] || [[ "${profile_run3_wq_manager_prefix_provided}" == "true" ]]; then
        echo "ERROR: --run3-wq-manager-prefix requires one non-empty prefix." >&2
        exit 1
      fi
      profile_run3_wq_manager_prefix="$2"
      profile_run3_wq_manager_prefix_provided=true
      shift 2
      ;;
    --chunksize|--pkl-min-free-bytes|--pkl-min-free-inodes|--pkl-safety-reserve-bytes|--pkl-safety-reserve-inodes|--wq-staging-base|--wq-quota-root|--wq-free-warning-bytes|--wq-min-free-bytes|--wq-poll-seconds|--wq-treereduction|--wq-processing-cores|--wq-accumulating-memory-mb|--wq-accumulating-cores)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 1
      fi
      case "$1" in
        --chunksize) profile_chunksize="$2" ;;
        --pkl-min-free-bytes) profile_pkl_min_free_bytes="$2" ;;
        --pkl-min-free-inodes) profile_pkl_min_free_inodes="$2" ;;
        --pkl-safety-reserve-bytes) profile_pkl_safety_reserve_bytes="$2" ;;
        --pkl-safety-reserve-inodes) profile_pkl_safety_reserve_inodes="$2" ;;
        --wq-staging-base) profile_wq_staging_base="$2" ;;
        --wq-quota-root) profile_wq_quota_root="$2" ;;
        --wq-free-warning-bytes) profile_wq_warning_bytes="$2" ;;
        --wq-min-free-bytes) profile_wq_min_free_bytes="$2" ;;
        --wq-poll-seconds) profile_wq_poll_seconds="$2" ;;
        --wq-treereduction) profile_wq_treereduction="$2" ;;
        --wq-processing-cores) profile_wq_processing_cores="$2" ;;
        --wq-accumulating-memory-mb) profile_wq_accumulating_memory_mb="$2" ;;
        --wq-accumulating-cores) profile_wq_accumulating_cores="$2" ;;
      esac
      if [[ "$1" == "--wq-processing-cores" && "$2" != "1" ]]; then
        echo "ERROR: --wq-processing-cores currently accepts only 1." >&2
        exit 1
      fi
      shift 2
      ;;
    --wq-warning-bytes|--wq-stop-bytes|--wq-reserve-bytes)
      echo "ERROR: retired Work Queue staging-budget option $1; use --wq-free-warning-bytes and --wq-min-free-bytes." >&2
      exit 1
      ;;
    --resume)
      profile_resume=true
      shift
      ;;
    --allow-environment-migration)
      profile_allow_environment_migration=true
      shift
      ;;
    --selected-fragment)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: --selected-fragment requires an exact fragment ID." >&2
        exit 1
      fi
      profile_selected_fragment="$2"
      shift 2
      ;;
    --plan|--activate-pending-plan|--expected-plan-revision)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 1
      fi
      case "$1" in
        --plan) profile_plan="$2" ;;
        --activate-pending-plan) profile_activate_pending_plan="$2" ;;
        --expected-plan-revision) profile_expected_plan_revision="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported run_cr.sh option '$1'." >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${production_profile}" ]]; then
  echo "ERROR: --production-profile is required; no production was scheduled." >&2
  print_usage >&2
  exit 1
fi

if [[ "${profile_wq_manager_name_provided}" == "true" \
  && "${profile_wq_manager_prefix_provided}" == "true" ]]; then
  echo "ERROR: --wq-manager-name and --wq-manager-prefix are mutually exclusive." >&2
  exit 1
fi
if [[ "${profile_run2_wq_manager_prefix_provided}" != "${profile_run3_wq_manager_prefix_provided}" ]]; then
  echo "ERROR: combined profiles require both --run2-wq-manager-prefix and --run3-wq-manager-prefix." >&2
  exit 1
fi
if [[ "${profile_run2_wq_manager_prefix_provided}" == "true" \
  && ( "${profile_wq_manager_name_provided}" == "true" || "${profile_wq_manager_prefix_provided}" == "true" ) ]]; then
  echo "ERROR: paired manager prefixes cannot be combined with --wq-manager-name or --wq-manager-prefix." >&2
  exit 1
fi
if [[ "${profile_wq_manager_prefix_provided}" == "true" ]]; then
  if [[ ! "${profile_wq_manager_prefix}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: --wq-manager-prefix requires a prefix of letters, digits, dots, underscores or hyphens." >&2
    exit 1
  fi
  if [[ "${profile_dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    echo "ERROR: fresh --wq-manager-prefix requires initialized campaign state; --dry-run does not initialize state." >&2
    exit 1
  fi
fi
if [[ "${profile_run2_wq_manager_prefix_provided}" == "true" ]]; then
  for paired_prefix in "${profile_run2_wq_manager_prefix}" "${profile_run3_wq_manager_prefix}"; do
    if [[ ! "${paired_prefix}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "ERROR: paired manager prefixes require letters, digits, dots, underscores or hyphens." >&2
      exit 1
    fi
  done
  if [[ "${profile_dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    echo "ERROR: fresh paired manager prefixes require initialized campaign state; --dry-run does not initialize state." >&2
    exit 1
  fi
fi

if [[ "${production_profile}" =~ ^run2_run3_(prod_2los_1tau|prod|stat_diagnostic)(_CR)?$ ]]; then
  combined_profile=true
  profile_mode="${BASH_REMATCH[1]}"
  profile_region="SR"
  [[ -z "${BASH_REMATCH[2]}" ]] || profile_region="CR"
  if [[ "${profile_mode}" == "prod_2los_1tau" && "${profile_region}" == "CR" ]]; then
    echo "ERROR: unsupported production profile '${production_profile}'." >&2
    exit 1
  fi
elif [[ "${production_profile}" =~ ^(run2|run3)_(prod_2los_1tau|prod|stat_diagnostic|sr_ext)(_CR)?$ ]]; then
  combined_profile=false
  profile_run_scope="${BASH_REMATCH[1]}"
  profile_mode="${BASH_REMATCH[2]}"
  profile_region="SR"
  [[ -z "${BASH_REMATCH[3]}" ]] || profile_region="CR"
  if [[ ( "${profile_mode}" == "sr_ext" || "${profile_mode}" == "prod_2los_1tau" ) && "${profile_region}" == "CR" ]]; then
    echo "ERROR: unsupported production profile '${production_profile}'." >&2
    exit 1
  fi
else
  echo "ERROR: unsupported production profile '${production_profile}'." >&2
  exit 1
fi
if [[ "${combined_profile}" == "false" \
  && "${profile_run2_wq_manager_prefix_provided}" == "true" ]]; then
  echo "ERROR: paired manager prefixes require a combined profile." >&2
  exit 1
fi

is_stat_diagnostic_profile() {
  [[ "${profile_mode}" == "stat_diagnostic" ]]
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../.." && pwd -P)
cd -- "${script_dir}"

validation_backend="${CAMPAIGN_VALIDATION_BACKEND:-}"
validation_root="${CAMPAIGN_VALIDATION_ROOT:-}"
validation_scenario="${CAMPAIGN_VALIDATION_SCENARIO:-success}"
native_wq_log_dir=""
if [[ -n "${validation_backend}" ]]; then
  diagnostics_root=$(cd -- "${repository_root}/../reports/diagnostics" && pwd -P)
  case "${validation_root}" in /tmp/*|"${diagnostics_root}"/*) ;; *) echo "ERROR: CAMPAIGN_VALIDATION_ROOT must be in an isolated validation namespace." >&2; exit 1 ;; esac
  case "${validation_backend}" in /tmp/*|"${diagnostics_root}"/*) ;; *) echo "ERROR: CAMPAIGN_VALIDATION_BACKEND must be in an isolated validation namespace." >&2; exit 1 ;; esac
  [[ -x "${validation_backend}" ]] || { echo "ERROR: validation backend is not executable." >&2; exit 1; }
fi

###############################################################################
# Global configuration
###############################################################################

output_dir=""
chunk_size="100000"
if [[ -n "${profile_chunksize}" ]]; then
  if [[ ! "${profile_chunksize}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --chunksize requires a positive integer." >&2
    exit 1
  fi
  chunk_size="${profile_chunksize}"
fi
if [[ -n "${profile_wq_staging_base}" ]]; then
  for guard_value in "${profile_wq_warning_bytes}" \
    "${profile_wq_min_free_bytes}" "${profile_wq_poll_seconds}"; do
    [[ -n "${guard_value}" ]] || { echo "ERROR: guarded staging requires free-space warning/minimum and poll values." >&2; exit 1; }
  done
elif [[ -n "${profile_wq_quota_root}${profile_wq_warning_bytes}${profile_wq_min_free_bytes}${profile_wq_poll_seconds}${profile_wq_treereduction}${profile_wq_processing_cores}${profile_wq_accumulating_memory_mb}${profile_wq_accumulating_cores}" ]]; then
  echo "ERROR: Work Queue guard options require --wq-staging-base." >&2
  exit 1
fi
if [[ -n "${profile_wq_accumulating_memory_mb}" || -n "${profile_wq_accumulating_cores}" ]]; then
  if [[ ! "${profile_wq_accumulating_memory_mb}" =~ ^[1-9][0-9]*$ \
    || ! "${profile_wq_accumulating_cores}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: accumulating memory and core minima require positive integer values together." >&2
    exit 1
  fi
fi
if [[ -n "${profile_plan}" && "${profile_resume}" == "true" ]]; then
  echo "ERROR: --plan is for a fresh campaign; use explicit pending-plan activation." >&2
  exit 1
fi
if [[ "${profile_allow_environment_migration}" == "true" \
  && ( "${profile_resume}" != "true" || -z "${profile_env_file}" \
       || -n "${profile_activate_pending_plan}" ) ]]; then
  echo "ERROR: --allow-environment-migration requires --resume and --env-file, without plan activation." >&2
  exit 1
fi
production_migration_args=()
if [[ "${profile_allow_environment_migration}" == "true" ]]; then
  production_migration_args=(--allow-environment-migration)
fi
if [[ -n "${profile_activate_pending_plan}" || -n "${profile_expected_plan_revision}" ]]; then
  if [[ -z "${profile_activate_pending_plan}" || -z "${profile_expected_plan_revision}" \
    || "${profile_resume}" != "true" || "${profile_dry_run}" == "true" \
    || -n "${profile_selected_fragment}" ]]; then
    echo "ERROR: pending-plan activation requires --resume, plan and expected revision, with no dry-run or selection." >&2
    exit 1
  fi
fi
if [[ "${profile_dry_run}" == false && -z "${profile_activate_pending_plan}" \
  && -z "${CAMPAIGN_VALIDATION_BACKEND:-}" \
  && -z "${profile_wq_staging_base}" ]]; then
  echo "ERROR: Work Queue production requires guarded manager staging; implicit /tmp staging is forbidden." >&2
  exit 1
fi
if [[ "${profile_dry_run}" == false && -z "${CAMPAIGN_VALIDATION_BACKEND:-}" \
  && -n "${profile_wq_staging_base}" ]]; then
  if [[ ! -d "${profile_wq_staging_base}" || "${profile_wq_staging_base}" != /* ]]; then
    echo "ERROR: staging base must be an existing absolute directory." >&2
    exit 1
  fi
  staging_real_path=$(realpath -e -- "${profile_wq_staging_base}") || exit 1
  if [[ "${staging_real_path}" == /tmp || "${staging_real_path}" == /tmp/* \
    || "${staging_real_path}" == /var/tmp || "${staging_real_path}" == /var/tmp/* ]]; then
    echo "ERROR: root temporary staging is forbidden for Work Queue production." >&2
    exit 1
  fi
fi

# Nominal TOP-23-002-like ttgamma sample-role strategy.
#
# Run 2:
#   TTGJets NLO                          -> production-like ttgamma
#   TTGamma_Dilept / TTGamma_SingleLept -> decay-like ttgamma
#   inclusive ttbar                     -> veto selected
#                                          external-conversion-like leptons
#
# Run 3:
#   TTG-1Jets_PTG-* -> inclusive ttgamma treatment
#   inclusive ttbar -> veto selected external-conversion-like leptons
#
# The diagnostic Run 2 NLO-only policy is intentionally not used here.
ttgamma_sample_role_policy="split"

campaign_tag=""
production_env_file=""
production_env_file_sha256=""
production_environment_fingerprint=""
production_environment_mode=""
production_initial_environment_mode=""
production_environment_snapshot=false
production_topcoffea_git_commit=""
production_topcoffea_source_fingerprint=""
production_sumw2_options_path=""
production_sumw2_temporary_options=""
production_state_path=""
production_git_commit=""

run_cr=false
run_sr=true
dry_run="${profile_dry_run}"
production_region="SR"
production_np_mode="separate"
production_component="run3"

if [[ -z "${profile_output_dir}" || -z "${profile_campaign_tag}" ]]; then
  cat >&2 <<EOF
ERROR: ${production_profile} requires explicit --output-dir and --campaign-tag.

No production or campaign-state mutation was performed.
EOF
  exit 1
fi
output_dir="${profile_output_dir}"
campaign_tag="${profile_campaign_tag}"
if [[ -n "${profile_pkl_min_free_bytes}${profile_pkl_min_free_inodes}${profile_pkl_safety_reserve_bytes}${profile_pkl_safety_reserve_inodes}" ]]; then
  for capacity_value in "${profile_pkl_min_free_bytes}" "${profile_pkl_min_free_inodes}" "${profile_pkl_safety_reserve_bytes}" "${profile_pkl_safety_reserve_inodes}"; do
    if [[ ! "${capacity_value}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: persistent output safety requires all four positive capacity thresholds." >&2
      exit 1
    fi
  done
  if (( profile_pkl_min_free_bytes <= profile_pkl_safety_reserve_bytes \
        || profile_pkl_min_free_inodes < profile_pkl_safety_reserve_inodes + 3 )); then
    echo "ERROR: persistent output admission must include an artifact budget and three pending-entry inodes beyond the safety reserve." >&2
    exit 1
  fi
fi
unset TOPEFT_PKL_NO_CLOBBER TOPEFT_PKL_PENDING_ID \
  TOPEFT_PKL_MIN_FREE_BYTES TOPEFT_PKL_MIN_FREE_INODES \
  TOPEFT_PKL_SAFETY_RESERVE_BYTES TOPEFT_PKL_SAFETY_RESERVE_INODES

for production_state_value in "${output_dir}" "${campaign_tag}"; do
  if [[ "${production_state_value}" == *$'\t'* || "${production_state_value}" == *$'\n'* ]]; then
    echo "ERROR: ${production_profile} state fields must not contain tab or newline characters." >&2
    exit 1
  fi
done

if [[ "${output_dir}" != /* ]]; then
  echo "ERROR: ${production_profile} --output-dir must be an absolute path: ${output_dir}" >&2
  exit 1
fi

if [[ "${profile_resume}" == "false" && -e "${output_dir}" ]]; then
  echo "ERROR: ${production_profile} output directory already exists; refusing overwrite: ${output_dir}" >&2
  exit 1
fi

if [[ "${profile_resume}" == "true" && ! -d "${output_dir}" ]]; then
  echo "ERROR: ${production_profile} --resume requires an existing campaign output directory: ${output_dir}" >&2
  exit 1
fi

# Re-enter the maintained component frontend in an isolated shell. Its normal
# initialization owns the full plan, environment validation and writer lock.
# FD 3 returns native plan IDs; ordinary frontend diagnostics remain visible.
prepare_component_campaign() (
  production_prepare_only=true
  source "${script_dir}/run_cr.sh" "$@" 3>&1 1>&2
)

# Publish a derived snapshot only while both native component writers are idle.
# The subshell closes every acquired descriptor, including on partial acquisition.
# These remain two atomic file replacements, not a transactional pair.
publish_combined_summary() (
  local scope root_variable profile_variable summary_lock_path summary_lock_fd
  local summary_lock_status
  for scope in run2 run3; do
    root_variable="${scope}_root"
    profile_variable="${scope}_profile"
    summary_lock_path="${!root_variable}/.${!profile_variable}_campaign.lock"
    if [[ -L "${summary_lock_path}" ]]; then
      echo "ERROR: campaign writer lock path is a symbolic link." >&2
      exit 1
    fi
  done
  # Fixed Run2 -> Run3 order for every cooperating frontend summary writer.
  for scope in run2 run3; do
    root_variable="${scope}_root"
    profile_variable="${scope}_profile"
    summary_lock_path="${!root_variable}/.${!profile_variable}_campaign.lock"
    exec {summary_lock_fd}>> "${summary_lock_path}" || exit "$?"
    if flock -n -E 75 "${summary_lock_fd}"; then
      :
    else
      summary_lock_status=$?
      if (( summary_lock_status == 75 )); then
        echo "WARNING: derived combined summary deferred because of component-lock contention (${scope})." >&2 || exit "$?"
        exit 0
      fi
      echo "ERROR: component summary lock acquisition failed (${scope}, exit ${summary_lock_status})." >&2
      exit "${summary_lock_status}"
    fi
  done
  # Guard explicitly: callers may run this helper in a conditional shell context.
  python "${script_dir}/campaign_manifest.py" combined-summary \
    "${production_profile}" "${campaign_tag}" "${output_dir}" \
    "${run2_state}" "${run3_state}" "${run2_exit}" "${run3_exit}" || exit "$?"
)

if [[ "${combined_profile}" == "true" ]]; then
  if [[ "${profile_wq_manager_name_provided}" == "true" ]]; then
    echo "ERROR: --wq-manager-name is unsupported for combined profiles." >&2
    exit 1
  fi
  if [[ "${profile_wq_manager_prefix_provided}" == "true" ]]; then
    echo "ERROR: --wq-manager-prefix requires one component campaign; use paired Run 2 and Run 3 prefixes for combined profiles." >&2
    exit 1
  fi
  if [[ -n "${profile_plan}${profile_activate_pending_plan}${profile_expected_plan_revision}" ]]; then
    echo "ERROR: plan revisions belong to component campaigns." >&2
    exit 1
  fi
  combined_suffix=""
  [[ "${profile_region}" != "CR" ]] || combined_suffix="_CR"
  component_args=()
  run2_manager_args=()
  run3_manager_args=()
  if [[ "${profile_run2_wq_manager_prefix_provided}" == "true" ]]; then
    run2_manager_args=(--wq-manager-prefix "${profile_run2_wq_manager_prefix}")
    run3_manager_args=(--wq-manager-prefix "${profile_run3_wq_manager_prefix}")
  fi
  remaining_args=("${original_arguments[@]}")
  while (( ${#remaining_args[@]} > 0 )); do
    case "${remaining_args[0]}" in
      --production-profile|--output-dir|--campaign-tag)
        remaining_args=("${remaining_args[@]:2}")
        ;;
      --run2-wq-manager-prefix|--run3-wq-manager-prefix)
        remaining_args=("${remaining_args[@]:2}")
        ;;
      --selected-fragment)
        remaining_args=("${remaining_args[@]:2}")
        ;;
      --resume)
        remaining_args=("${remaining_args[@]:1}")
        ;;
      --dry-run|--phase-telemetry|--allow-environment-migration)
        component_args+=("${remaining_args[0]}")
        remaining_args=("${remaining_args[@]:1}")
        ;;
      *)
        component_args+=("${remaining_args[0]}" "${remaining_args[1]}")
        remaining_args=("${remaining_args[@]:2}")
        ;;
    esac
  done
  if [[ "${profile_dry_run}" == "false" && "${profile_resume}" == "false" ]]; then
    mkdir -- "${output_dir}"
  fi
  run2_profile="run2_${profile_mode}${combined_suffix}"
  run3_profile="run3_${profile_mode}${combined_suffix}"
  run2_root="${output_dir}/run2${combined_suffix}"
  run3_root="${output_dir}/run3${combined_suffix}"
  run2_state="${run2_root}/.${run2_profile}_campaign_state_v9.json"
  run3_state="${run3_root}/.${run3_profile}_campaign_state_v9.json"
  # Validate a present component through its native preparation path; create
  # only an absent component. A directory without state is never silent fresh.
  for scope in run2 run3; do
    root_variable="${scope}_root"
    profile_variable="${scope}_profile"
    prepare_resume_args=()
    if [[ -e "${!root_variable}" || -L "${!root_variable}" ]]; then
      if [[ -L "${!root_variable}" ]]; then
        echo "ERROR: component campaign directory is a symbolic link." >&2
        exit 1
      fi
      prepare_resume_args=(--resume)
    fi
    prepare_manager_args=()
    if [[ "${scope}" == "run2" ]]; then
      prepare_manager_args=("${run2_manager_args[@]}")
    else
      prepare_manager_args=("${run3_manager_args[@]}")
    fi
    ids=$(prepare_component_campaign --production-profile "${!profile_variable}" \
      --output-dir "${!root_variable}" --campaign-tag "${campaign_tag}_${scope}" \
      "${component_args[@]}" "${prepare_manager_args[@]}" "${prepare_resume_args[@]}")
    printf -v "${scope}_ids" '%s' "${ids}"
  done
  child_resume_args=(--resume)
  if [[ "${profile_dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    child_resume_args=()
  fi
  if [[ -n "${profile_selected_fragment}" ]]; then
    selected_owners=0
    selected_scope=""
    for scope in run2 run3; do
      ids_variable="${scope}_ids"
      while IFS= read -r unit_id; do
        if [[ "${unit_id}" == "${profile_selected_fragment}" ]]; then
          selected_owners=$((selected_owners + 1))
          selected_scope="${scope}"
        fi
      done <<< "${!ids_variable}"
    done
    if (( selected_owners != 1 )); then
      echo "ERROR: selected fragment ${profile_selected_fragment} requires exactly one component owner (found ${selected_owners})." >&2
      exit 1
    fi
    selected_profile_variable="${selected_scope}_profile"
    selected_root_variable="${selected_scope}_root"
    selected_manager_args=()
    if [[ "${selected_scope}" == "run2" ]]; then
      selected_manager_args=("${run2_manager_args[@]}")
    else
      selected_manager_args=("${run3_manager_args[@]}")
    fi
    if [[ "${profile_dry_run}" == "false" ]]; then
      (
        sibling_scope=run2
        [[ "${selected_scope}" != "run2" ]] || sibling_scope=run3
        sibling_root_variable="${sibling_scope}_root"
        sibling_profile_variable="${sibling_scope}_profile"
        sibling_state_variable="${sibling_scope}_state"
        sibling_lock="${!sibling_root_variable}/.${!sibling_profile_variable}_campaign.lock"
        if [[ -L "${sibling_lock}" ]]; then
          echo "ERROR: campaign writer lock path is a symbolic link." >&2
          exit 1
        fi
        # Point-in-time admission only: the completed subshell closes the sibling
        # descriptor before the selected child can inherit it. No reconciliation.
        exec {sibling_lock_fd}>> "${sibling_lock}" || exit "$?"
        flock -n "${sibling_lock_fd}" || { echo "ERROR: campaign writer lock is held." >&2; exit 1; }
        sibling_classification=$(python "${script_dir}/campaign_manifest.py" classification "${!sibling_state_variable}") || exit "$?"
        if [[ "${sibling_classification}" == "blocked_ambiguous" ]]; then
          echo "ERROR: nonselected component has a running attempt; selected child was not started." >&2
          exit 1
        fi
      )
    fi
    if "$0" --production-profile "${!selected_profile_variable}" \
      --output-dir "${!selected_root_variable}" --campaign-tag "${campaign_tag}_${selected_scope}" \
      "${component_args[@]}" "${selected_manager_args[@]}" "${child_resume_args[@]}" \
      --selected-fragment "${profile_selected_fragment}"; then
      selected_exit=0
    else
      selected_exit=$?
    fi
    if [[ "${profile_dry_run}" == "false" ]]; then
      run2_exit=0
      run3_exit=0
      printf -v "${selected_scope}_exit" '%s' "${selected_exit}"
      publish_combined_summary
    fi
    exit "${selected_exit}"
  fi
  if "$0" --production-profile "${run2_profile}" --output-dir "${run2_root}" \
    --campaign-tag "${campaign_tag}_run2" "${component_args[@]}" "${run2_manager_args[@]}" "${child_resume_args[@]}"; then
    run2_exit=0
  else
    run2_exit=$?
  fi
  if [[ "${profile_dry_run}" == "false" ]]; then
    run2_state="${run2_root}/.${run2_profile}_campaign_state_v9.json"
    if [[ ! -f "${run2_state}" ]]; then
      echo "ERROR: Run 2 component has no state for safe combined continuation." >&2
      exit 1
    fi
    run2_classification=$(python "${script_dir}/campaign_manifest.py" classification "${run2_state}")
    if [[ "${run2_classification}" == "blocked_ambiguous" ]]; then
      echo "ERROR: Run 2 component has a running attempt; Run 3 was not started." >&2
      exit 1
    fi
  fi
  if "$0" --production-profile "${run3_profile}" --output-dir "${run3_root}" \
    --campaign-tag "${campaign_tag}_run3" "${component_args[@]}" "${run3_manager_args[@]}" "${child_resume_args[@]}"; then
    run3_exit=0
  else
    run3_exit=$?
  fi
  if [[ "${profile_dry_run}" == "false" ]]; then
    publish_combined_summary
  fi
  (( run2_exit == 0 && run3_exit == 0 ))
  exit $?
fi

if [[ -n "${profile_activate_pending_plan}" ]]; then
  if [[ ! -f "${output_dir}/.${production_profile}_campaign_state_v9.json" ]]; then
    echo "ERROR: pending-plan activation requires existing campaign state." >&2
    exit 1
  fi
  campaign_lock_path="${output_dir}/.${production_profile}_campaign.lock"
  if [[ -L "${campaign_lock_path}" ]]; then
    echo "ERROR: campaign writer lock path must not be a symbolic link: ${campaign_lock_path}" >&2
    exit 1
  fi
  exec {campaign_lock_fd}>> "${campaign_lock_path}"
  flock -n "${campaign_lock_fd}" || { echo "ERROR: campaign writer lock is held." >&2; exit 1; }
  CAMPAIGN_LOCK_FD="${campaign_lock_fd}" python "${script_dir}/campaign_manifest.py" activate \
    "${output_dir}/.${production_profile}_campaign_state_v9.json" \
    "${profile_activate_pending_plan}" "${profile_expected_plan_revision}"
  exit 0
fi

cr_pkl_base_tag="${campaign_tag}"
sr_pkl_base_tag="${campaign_tag}"

production_region="${profile_region}"
production_component="${profile_run_scope}"
[[ "${profile_region}" != "CR" ]] || { run_cr=true; run_sr=false; }

# Resolve and print commands without launching production.
dry_run="${profile_dry_run}"

# Shared CR/SR production switches.
do_systs=true
do_np=true
if is_stat_diagnostic_profile; then
  do_systs=false
fi

# Enable only when lepton-flavour-split outputs are explicitly required.
split_lep_flavor=false

###############################################################################
# CR configuration
###############################################################################

# Each entry is one independent histogram chunk.
cr_non_tau_var_sets=(
  "fwd0pt fwd0eta j0pt j0eta lj0pt njets nbtagsm"
  "lt met ptz l0conept l0eta l1conept l1eta"
  "nbtagsl invmass ljptsum npvsGood"
)

cr_tau_var_sets=(
  "fwd0pt fwd0eta j0pt j0eta lj0pt njets nbtagsm"
  "lt met ptz l0conept l0eta l1conept l1eta"
  "nbtagsl invmass ljptsum npvsGood ptz_wtau tau0Fpt tau0Tpt"
)

# CR outputs intentionally combine periods into Run 2, 2022, and 2023 groups.
#
# Run 2 is commented because its CR production is already complete.
cr_year_sets=(
  # "2016APV 2016 2017 2018"
  "2022 2022EE"
  "2023 2023BPix"
)

# Current category names used by the analysis helpers.
#
# The aggregate 2los_1tau group is included for the 2los tau request. If the
# branch gains explicit 2los_1tau_Ftau / 2los_1tau_Ttau category groups, add
# them only after confirming the exact names and intended histogram coverage.
cr_category_sets=(
  "2l_CR 2l_CRflip 2los_CRZ 2los_CRtt 3l_CR"
  "1l_1tau_CRtt 1l_1tau_CRDY 2los_1tau"
)

# Parallel to cr_category_sets. Each category family selects its intentional
# histogram chunks instead of forming an invalid shared Cartesian product.
cr_category_var_set_names=(
  "cr_non_tau_var_sets"
  "cr_tau_var_sets"
)

###############################################################################
# SR configuration
###############################################################################

# ptz is the Z-candidate family and ptll is the closest-SFOS dilepton-pT family
# for 3l off-Z low/high. Request each public family only from category blocks
# that can fill it.
sr_with_ptz_wtau_var_sets=(
  "njets lj0pt ptz ptz_wtau lt"
)

sr_offz_var_sets=(
  "njets lj0pt ptll lt"
)

sr_onz_tau_var_sets=(
  "njets lj0pt ptz lt"
)

sr_fwd_var_sets=(
  "njets lj0pt ptz lt"
)

sr_tau5j_var_sets=(
  "njets lj0pt ptz"
)

# The complete Run-3 plan keeps the two off-Z charge families in separate
# source blocks so failures and output identities remain unambiguous.
sr_category_sets=(
  "2l 2lss_1tau 2los_1tau 4l"
  "3l_m_offZ"
  "3l_p_offZ"
  "3l_onZ_tau"
  "3l_fwd"
)

sr_category_var_set_names=(
  "sr_with_ptz_wtau_var_sets"
  "sr_offz_var_sets"
  "sr_offz_var_sets"
  "sr_onz_tau_var_sets"
  "sr_fwd_var_sets"
)

if [[ "${profile_mode}" == "prod_2los_1tau" ]]; then
  sr_category_sets=("2los_1tau")
  sr_category_var_set_names=("sr_tau5j_var_sets")
fi

sr_year_sets=()
sr_year_category_set_names=()
sr_year_category_var_set_names=()
cr_year_sets=()
if [[ "${profile_run_scope}" == "run2" ]]; then
  profile_year_expr="run2"
  cr_year_sets=("2016APV 2016 2017 2018")
else
  profile_year_expr="2022 2022EE 2023 2023BPix"
  cr_year_sets=("2022 2022EE" "2023 2023BPix")
fi
if [[ "${profile_region}" == "SR" && "${profile_mode}" != "sr_ext" ]]; then
  sr_year_sets=("${profile_year_expr}")
  sr_year_category_set_names=("sr_category_sets")
  sr_year_category_var_set_names=("sr_category_var_set_names")
fi

production_state_filename=".${production_profile}_campaign_state_v9.json"
production_block_ids=()
production_plan_year_exprs=()
production_plan_category_sets=()
production_plan_var_sets=()

if [[ "${profile_mode}" == "sr_ext" ]]; then
  : # The scientific default comes from sr_reco_default_plan.json.
elif [[ "${profile_region}" == "SR" ]]; then
  profile_block_suffixes=(a b c d e)
  for profile_index in "${!sr_category_sets[@]}"; do
    profile_var_set_name="${sr_category_var_set_names[profile_index]}"
    declare -n profile_var_set_ref="${profile_var_set_name}"
    production_block_ids+=("${production_profile}_${profile_block_suffixes[profile_index]}")
    production_plan_year_exprs+=("${profile_year_expr}")
    production_plan_category_sets+=("${sr_category_sets[profile_index]}")
    production_plan_var_sets+=("${profile_var_set_ref[0]}")
    unset -n profile_var_set_ref
  done
else
  profile_block_index=0
  for profile_year_expr in "${cr_year_sets[@]}"; do
    for profile_category_index in "${!cr_category_sets[@]}"; do
      if (( profile_category_index == 0 )); then
        profile_var_set_name=cr_non_tau_var_sets
      else
        profile_var_set_name=cr_tau_var_sets
      fi
      declare -n profile_var_set_ref="${profile_var_set_name}"
      for profile_var_set in "${profile_var_set_ref[@]}"; do
        profile_block_index=$((profile_block_index + 1))
        production_block_ids+=("${production_profile,,}_block${profile_block_index}")
        production_plan_year_exprs+=("${profile_year_expr}")
        production_plan_category_sets+=("${cr_category_sets[profile_category_index]}")
        production_plan_var_sets+=("${profile_var_set}")
      done
      unset -n profile_var_set_ref
    done
  done
fi

###############################################################################
# Execution accounting
###############################################################################

declare -a block_summary_statuses=()
declare -a block_summary_modes=()
declare -a block_summary_years=()
declare -a block_summary_categories=()
declare -a block_summary_variables=()
declare -a block_summary_output_tags=()
declare -a block_summary_exit_codes=()
declare -a block_summary_durations=()

run_success_count=0
run_failure_count=0
run_skipped_count=0

###############################################################################
# Helpers
###############################################################################

join_by() {
  local delimiter="$1"
  shift

  local IFS="${delimiter}"
  echo "$*"
}

assert_boolean() {
  local value="$1"
  local option_name="$2"

  case "${value}" in
    true|false) ;;
    *)
      echo "ERROR: ${option_name} must be true or false, got '${value}'." >&2
      exit 1
      ;;
  esac
}

assert_parallel_array_lengths() {
  local label="$1"
  local left_count="$2"
  local right_count="$3"

  if (( left_count != right_count )); then
    cat >&2 <<EOF
ERROR: inconsistent ${label} configuration.

Number of category entries:
  ${left_count}

Number of variable-set mapping entries:
  ${right_count}
EOF
    exit 1
  fi
}

assert_array_defined() {
  local array_name="$1"

  if ! declare -p "${array_name}" >/dev/null 2>&1; then
    echo "ERROR: mapped array '${array_name}' is not defined." >&2
    exit 1
  fi
}

production_output_name() {
  local year_expr="$1"
  local pkl_tag="$2"
  local years=()
  local year_label

  if [[ "${year_expr}" == "run2" ]]; then
    year_label="UL16-UL16APV-UL17-UL18"
  else
    read -r -a years <<< "${year_expr}"
    year_label=$(join_by - "${years[@]}")
  fi
  printf '%s%ss_%s' "${year_label}" "${production_region}" "${pkl_tag}"
}

write_production_plan() {
  local plan_path="$1"
  local index
  local year_expr
  local category_set
  local var_set
  local cats=()
  local vars=()
  local cat_tag
  local var_tag
  local pkl_tag
  local output_name

  : > "${plan_path}"
  for index in "${!production_block_ids[@]}"; do
    year_expr="${production_plan_year_exprs[index]}"
    category_set="${production_plan_category_sets[index]}"
    var_set="${production_plan_var_sets[index]}"
    read -r -a cats <<< "${category_set}"
    read -r -a vars <<< "${var_set}"
    cat_tag=$(join_by - "${cats[@]}")
    var_tag=$(join_by - "${vars[@]}")
    if [[ "${profile_run_scope}" == "run2" || "${profile_region}" == "CR" ]]; then
      pkl_tag="${campaign_tag}-block$((index + 1))"
    else
      pkl_tag="${campaign_tag}_${cat_tag}_${var_tag}"
    fi
    output_name=$(production_output_name "${year_expr}" "${pkl_tag}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${production_block_ids[index]}" \
      "${year_expr}" \
      "${category_set}" \
      "${var_set}" \
      "${pkl_tag}" \
      "${output_name}" \
      "${output_dir}/${output_name}.pkl.gz" \
      "${output_dir}/${output_name}_np.pkl.gz" \
      >> "${plan_path}"
  done
}

production_state_tool() {
  CAMPAIGN_LOCK_FD="${campaign_lock_fd:-}" python ./campaign_manifest.py "$@"
}

resolve_production_environment() {
  local requested_env_file="${profile_env_file}"
  local frozen_env_file=""
  local canonical_requested_env_file=""
  local validation_status=""
  local validation_args=()

  production_state_path="${output_dir}/${production_state_filename}"

  if [[ "${profile_resume}" == "true" ]]; then
    if [[ ! -f "${production_state_path}" ]]; then
      echo "ERROR: ${production_profile} --resume requires campaign state: ${production_state_path}" >&2
      exit 1
    fi
    frozen_env_file=$(production_state_tool env_file \
      "${production_state_path}" "${production_profile}" "${campaign_tag}" "${output_dir}")
    production_initial_environment_mode=$(production_state_tool environment_mode \
      "${production_state_path}" "${production_profile}" "${campaign_tag}" "${output_dir}")
    production_environment_mode="${production_initial_environment_mode}"
    if [[ -n "${requested_env_file}" ]]; then
      if [[ "${requested_env_file}" != /* ]]; then
        echo "ERROR: ${production_profile} --env-file must be an absolute path: ${requested_env_file}" >&2
        exit 1
      fi
      canonical_requested_env_file=$(readlink -f -- "${requested_env_file}" || true)
      if [[ -z "${canonical_requested_env_file}" ]]; then
        echo "ERROR: resume --env-file cannot be resolved." >&2
        exit 1
      fi
      production_environment_mode=explicit_snapshot
      production_environment_snapshot=true
      validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
    else
      requested_env_file="${frozen_env_file}"
      if [[ "${production_environment_mode}" == "explicit_snapshot" ]]; then
        validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
        production_environment_snapshot=true
      elif [[ "${production_environment_mode}" == "current_resolved" ]]; then
        validation_args=(--validate-env-file --env-file "${requested_env_file}")
      else
        echo "ERROR: campaign state has an unsupported environment mode." >&2
        exit 1
      fi
    fi
  elif [[ -n "${requested_env_file}" ]]; then
    if [[ "${requested_env_file}" != /* ]]; then
      echo "ERROR: ${production_profile} --env-file must be an absolute path: ${requested_env_file}" >&2
      exit 1
    fi
    canonical_requested_env_file=$(readlink -f -- "${requested_env_file}" || true)
    production_environment_mode=explicit_snapshot
    production_initial_environment_mode=explicit_snapshot
    production_environment_snapshot=true
    validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
  else
    production_environment_mode=current_resolved
    production_initial_environment_mode=current_resolved
    validation_args=(--prepare-env-only)
  fi

  if [[ -n "${validation_backend}" ]]; then
    if [[ "${production_environment_mode}" == "current_resolved" ]]; then
      if ! production_environment_validation=$("${validation_backend}" prepare_environment "${validation_scenario}"); then
        echo "ERROR: ${production_profile} could not resolve its current environment archive; no campaign state was changed." >&2
        exit 1
      fi
    elif ! production_environment_validation=$("${validation_backend}" validate_environment "${validation_scenario}" "${requested_env_file}"); then
      echo "ERROR: ${production_profile} validation backend rejected the environment; no campaign state was changed." >&2
      exit 1
    fi
  elif ! production_environment_validation=$(python ./run_analysis.py "${validation_args[@]}"); then
    echo "ERROR: ${production_profile} could not validate its environment archive; no campaign state was changed." >&2
    exit 1
  fi
  printf '%s\n' "${production_environment_validation}"

  production_env_file=$(awk -F': ' '/^env_file: / {print $2; exit}' <<< "${production_environment_validation}")
  production_env_file_sha256=$(awk -F': ' '/^env_file_sha256: / {print $2; exit}' <<< "${production_environment_validation}")
  production_environment_fingerprint=$(awk -F': ' '/^environment_fingerprint: / {print $2; exit}' <<< "${production_environment_validation}")
  production_topcoffea_git_commit=$(awk -F': ' '/^topcoffea_git_commit: / {print $2; exit}' <<< "${production_environment_validation}")
  production_topcoffea_source_fingerprint=$(awk -F': ' '/^topcoffea_relevant_source_fingerprint: / {print $2; exit}' <<< "${production_environment_validation}")
  validation_status=$(awk -F': ' '/^environment_validation_status: / {print $2; exit}' <<< "${production_environment_validation}")

  if [[ "${production_env_file}" != /* ]] \
    || [[ ! -f "${production_env_file}" || ! -r "${production_env_file}" || ! -s "${production_env_file}" ]] \
    || [[ -z "${production_env_file_sha256}" ]] \
    || [[ -z "${production_environment_fingerprint}" ]] \
    || [[ "${validation_status}" != "valid" ]]; then
    echo "ERROR: run_analysis.py did not return a complete valid environment identity." >&2
    exit 1
  fi

  if [[ "${production_initial_environment_mode}" == "current_resolved" ]] \
    && { [[ -z "${production_topcoffea_git_commit}" ]] \
      || [[ -z "${production_topcoffea_source_fingerprint}" ]]; }; then
    echo "ERROR: current environment resolution did not return complete TOPCOFFEA provenance." >&2
    exit 1
  fi

  if [[ -n "${canonical_requested_env_file}" && "${production_env_file}" != "${canonical_requested_env_file}" ]]; then
    echo "ERROR: --env-file validation returned a different archive path." >&2
    exit 1
  fi
}

production_block_id() {
  local year_expr="$1"
  local category_set="$2"
  local var_set="$3"
  local index

  for index in "${!production_block_ids[@]}"; do
    if [[ "${year_expr}" == "${production_plan_year_exprs[index]}" ]] \
      && [[ "${category_set}" == "${production_plan_category_sets[index]}" ]] \
      && [[ "${var_set}" == "${production_plan_var_sets[index]}" ]]; then
      printf '%s' "${production_block_ids[index]}"
      return 0
    fi
  done

  echo "ERROR: unable to resolve ${production_profile} block identity for '${year_expr}' / '${category_set}'." >&2
  return 1
}

production_assert_live_plan() {
  local expected_count
  [[ "${profile_mode}" != "sr_ext" ]] || return 0
  if [[ "${profile_mode}" == "prod_2los_1tau" ]]; then
    expected_count=1
  elif [[ "${profile_region}" == "SR" ]]; then
    expected_count=5
  elif [[ "${profile_run_scope}" == "run2" ]]; then
    expected_count=6
  else
    expected_count=12
  fi
  for observed_count in \
    "${#production_block_ids[@]}" "${#production_plan_year_exprs[@]}" \
    "${#production_plan_category_sets[@]}" "${#production_plan_var_sets[@]}"; do
    if (( observed_count != expected_count )); then
      echo "ERROR: profile scientific default packing is inconsistent." >&2
      exit 1
    fi
  done
}

prepare_production_campaign() {
  local plan_directory
  local default_plan_path="-"
  local supplied_plan="${profile_plan:--}"

  production_assert_live_plan
  production_git_commit=$(git -C "${repository_root}" rev-parse HEAD 2>/dev/null || true)
  production_state_path="${output_dir}/${production_state_filename}"
  if [[ "${dry_run}" == "true" ]]; then
    plan_directory="${TMPDIR:-/tmp}"
  else
    if [[ "${profile_resume}" == "false" && "${profile_wq_manager_name_provided}" == "false" ]]; then
      mkdir -- "${output_dir}"
    fi
    if [[ "${profile_resume}" == "false" && "${profile_wq_manager_name_provided}" == "true" ]]; then
      plan_directory="${TMPDIR:-/tmp}"
    else
      campaign_lock_path="${output_dir}/.${production_profile}_campaign.lock"
      if [[ -L "${campaign_lock_path}" ]]; then
        echo "ERROR: campaign writer lock path is a symbolic link." >&2
        exit 1
      fi
      exec {campaign_lock_fd}>> "${campaign_lock_path}"
      flock -n "${campaign_lock_fd}" || { echo "ERROR: campaign writer lock is held." >&2; exit 1; }
      plan_directory="${output_dir}"
    fi
  fi
  production_plan_file=$(mktemp "${plan_directory}/.${production_profile}_plan.XXXXXX")
  if [[ "${profile_mode}" == "sr_ext" ]]; then
    default_plan_path="${script_dir}/sr_reco_default_plan.json"
  else
    production_default_plan_file=$(mktemp "${plan_directory}/.${production_profile}_default.XXXXXX")
    write_production_plan "${production_default_plan_file}"
    default_plan_path="${production_default_plan_file}"
  fi
  if [[ "${profile_resume}" == "true" ]]; then
    if [[ ! -f "${production_state_path}" ]]; then
      echo "ERROR: resume requires schema-9 campaign state: ${production_state_path}" >&2
      exit 1
    fi
    production_state_tool validate \
      "${production_state_path}" "${production_plan_file}" \
      "${production_profile}" "${default_plan_path}" \
      "${campaign_tag}" "${output_dir}" "${production_git_commit}" \
      "${production_env_file}" "${production_env_file_sha256}" \
      "${production_environment_fingerprint}" "${production_environment_mode}" \
      "${production_topcoffea_git_commit}" "${production_topcoffea_source_fingerprint}" \
      "${ttgamma_sample_role_policy}" "${do_systs}" "${do_np}" \
      "${production_region}" "${production_np_mode}" "${production_migration_args[@]}"
    python ./campaign_manifest.py render-active "${production_state_path}" "${production_plan_file}"
  else
    python ./campaign_manifest.py render-new \
      "${default_plan_path}" "${supplied_plan}" \
      "${production_profile}" "${campaign_tag}" \
      "${output_dir}" "${production_plan_file}"
  fi
  production_block_ids=()
  production_plan_year_exprs=()
  production_plan_category_sets=()
  production_plan_var_sets=()
  while IFS=$'\t' read -r plan_id plan_years plan_categories plan_vars _; do
    production_block_ids+=("${plan_id}")
    production_plan_year_exprs+=("${plan_years}")
    production_plan_category_sets+=("${plan_categories}")
    production_plan_var_sets+=("${plan_vars}")
  done < "${production_plan_file}"
  if [[ -n "${profile_selected_fragment}" ]]; then
    selected_found=false
    for plan_id in "${production_block_ids[@]}"; do
      [[ "${plan_id}" != "${profile_selected_fragment}" ]] || selected_found=true
    done
    if [[ "${selected_found}" != "true" ]]; then
      echo "ERROR: unknown selected fragment ${profile_selected_fragment}." >&2
      exit 1
    fi
  fi
  validate_wq_manager_name_scope
  # Preparation validates existing state under the same native lock but never
  # reconciles or changes attempt provenance before execution is selected.
  if [[ "${production_prepare_only:-false}" == "true" && "${profile_resume}" == "true" ]]; then
    return
  fi
  if [[ "${dry_run}" == "false" ]]; then
    if [[ "${profile_resume}" == "true" ]]; then
      if [[ -n "${profile_wq_staging_base}" ]]; then
        CAMPAIGN_LOCK_FD="${campaign_lock_fd}" python ./wq_staging_guard.py check-recovery \
          "${production_state_path}" "${profile_selected_fragment}" "${profile_wq_staging_base}"
      fi
      production_state_tool reconcile "${production_state_path}" "${profile_selected_fragment}"
      if [[ -n "${profile_wq_staging_base}" ]]; then
        CAMPAIGN_LOCK_FD="${campaign_lock_fd}" python ./wq_staging_guard.py recover \
          "${production_state_path}" "${profile_selected_fragment}" "${profile_wq_staging_base}"
      fi
      production_state_tool record-provenance \
        "${production_state_path}" "${production_git_commit}" \
        "${production_env_file}" "${production_env_file_sha256}" \
        "${production_environment_fingerprint}" "${production_environment_mode}" \
        "${production_topcoffea_git_commit}" "${production_topcoffea_source_fingerprint}" "${production_migration_args[@]}"
    else
      if [[ "${profile_wq_manager_name_provided}" == "true" ]]; then
        mkdir -- "${output_dir}"
        campaign_lock_path="${output_dir}/.${production_profile}_campaign.lock"
        exec {campaign_lock_fd}>> "${campaign_lock_path}"
        flock -n "${campaign_lock_fd}" || { echo "ERROR: campaign writer lock is held." >&2; exit 1; }
      fi
      production_state_tool initialize \
        "${production_state_path}" "${production_plan_file}" \
        "${production_profile}" "${default_plan_path}" \
        "${campaign_tag}" "${output_dir}" "${production_git_commit}" \
        "${production_env_file}" "${production_env_file_sha256}" \
        "${production_environment_fingerprint}" "${production_environment_mode}" \
        "${production_topcoffea_git_commit}" "${production_topcoffea_source_fingerprint}" \
        "${ttgamma_sample_role_policy}" "${do_systs}" "${do_np}" \
        "${production_region}" "${production_np_mode}"
    fi
  fi
}

validate_wq_manager_name_scope() {
  [[ "${profile_wq_manager_name_provided}" == "true" ]] || return 0
  local plan_index
  local source_status
  local status_line
  local source_manager_count=0
  for plan_index in "${!production_block_ids[@]}"; do
    if [[ -n "${profile_selected_fragment}" \
      && "${production_block_ids[plan_index]}" != "${profile_selected_fragment}" ]]; then
      continue
    fi
    if [[ "${profile_resume}" == "true" ]]; then
      status_line=$(production_state_tool status "${production_state_path}" "${production_block_ids[plan_index]}")
      IFS=$'\t' read -r _ source_status _ <<< "${status_line}"
      case "${source_status}" in
        ready) continue ;;
        planned|failed) ;;
        running)
          echo "ERROR: --wq-manager-name cannot resolve a running source before recovery." >&2
          exit 1
          ;;
        *)
          echo "ERROR: --wq-manager-name cannot resolve source status '${source_status}'." >&2
          exit 1
          ;;
      esac
    fi
    source_manager_count=$((source_manager_count + 1))
    if (( source_manager_count > 1 )); then
      echo "ERROR: --wq-manager-name requires exactly one source manager; resolved multiple." >&2
      exit 1
    fi
  done
  if (( source_manager_count != 1 )); then
    echo "ERROR: --wq-manager-name requires exactly one source manager; resolved zero." >&2
    exit 1
  fi
}

prepare_production_sumw2_options() {
  if ! is_stat_diagnostic_profile; then
    return
  fi
  if [[ "${dry_run}" == "true" ]]; then
    production_sumw2_temporary_options=$(mktemp "${TMPDIR:-/tmp}/stat_diagnostic_sumw2.XXXXXX.yml")
    production_sumw2_options_path="${production_sumw2_temporary_options}"
    printf 'sumw2_storage:\n  mode: full_diagnostics\n' > "${production_sumw2_options_path}"
  else
    production_sumw2_options_path="${output_dir}/sumw2_full_diagnostics.yml"
    if [[ "${profile_resume}" == "false" ]]; then
      printf 'sumw2_storage:\n  mode: full_diagnostics\n' > "${production_sumw2_options_path}"
    fi
  fi
  if [[ ! -f "${production_sumw2_options_path}" ]] \
    || [[ "$(<"${production_sumw2_options_path}")" != $'sumw2_storage:\n  mode: full_diagnostics' ]]; then
    echo "ERROR: stat_diagnostic sumw2 options differ from full_diagnostics." >&2
    exit 1
  fi
}

cleanup_production_plan() {
  if [[ -n "${production_plan_file:-}" && -f "${production_plan_file}" ]]; then
    rm -f -- "${production_plan_file}"
  fi
  if [[ -n "${production_default_plan_file:-}" && -f "${production_default_plan_file}" ]]; then
    rm -f -- "${production_default_plan_file}"
  fi
  if [[ -n "${production_sumw2_temporary_options:-}" && -f "${production_sumw2_temporary_options}" ]]; then
    rm -f -- "${production_sumw2_temporary_options}"
  fi
}

assert_supported_year_expr() {
  local year_expr="$1"
  local year
  local years_in_expr=()

  read -r -a years_in_expr <<< "${year_expr}"

  if (( ${#years_in_expr[@]} == 0 )); then
    echo "ERROR: empty year expression." >&2
    exit 1
  fi

  for year in "${years_in_expr[@]}"; do
    case "${year}" in
      run2|2016APV|2016|2017|2018|2022|2022EE|2023|2023BPix) ;;
      *)
        cat >&2 <<EOF
ERROR: unsupported year token '${year}' in year expression '${year_expr}'.

Allowed year tokens:
  run2 2016APV 2016 2017 2018 2022 2022EE 2023 2023BPix
EOF
        exit 1
        ;;
    esac
  done
}

print_command() {
  local -a cmd=("$@")

  echo "Executing:"
  printf ' %q' "${cmd[@]}"
  echo
}

print_var_sets() {
  local label="$1"
  shift

  local var_set
  local index=0

  echo "${label} variable chunks:"
  for var_set in "$@"; do
    index=$((index + 1))
    echo "  ${index}: ${var_set}"
  done
}

format_duration() {
  local duration_seconds="$1"
  local hours
  local minutes
  local seconds

  hours=$((duration_seconds / 3600))
  minutes=$(((duration_seconds % 3600) / 60))
  seconds=$((duration_seconds % 60))

  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

record_block_result() {
  local status="$1"
  local mode="$2"
  local year_expr="$3"
  local category_set="$4"
  local var_set="$5"
  local output_tag="$6"
  local exit_code="$7"
  local duration_seconds="$8"

  block_summary_statuses+=("${status}")
  block_summary_modes+=("${mode}")
  block_summary_years+=("${year_expr}")
  block_summary_categories+=("${category_set}")
  block_summary_variables+=("${var_set}")
  block_summary_output_tags+=("${output_tag}")
  block_summary_exit_codes+=("${exit_code}")
  block_summary_durations+=("${duration_seconds}")

  case "${status}" in
    SUCCESS)
      run_success_count=$((run_success_count + 1))
      ;;
    FAILED)
      run_failure_count=$((run_failure_count + 1))
      ;;
    SKIPPED)
      run_skipped_count=$((run_skipped_count + 1))
      ;;
    DRY_RUN)
      ;;
    *)
      echo "ERROR: unknown block status '${status}'." >&2
      exit 1
      ;;
  esac
}

print_run_summary() {
  local total_count
  local attempted_count
  local index
  local exit_code
  local signal_number
  local signal_suffix
  local duration_text

  total_count=${#block_summary_statuses[@]}
  attempted_count=$((run_success_count + run_failure_count))

  echo
  echo "========================================"
  echo "run_cr.sh execution summary"
  echo "campaign_tag: ${campaign_tag}"
  echo "output_dir: ${output_dir}"
  echo "configured block invocations: ${total_count}"
  echo "attempted: ${attempted_count}"
  echo "successful: ${run_success_count}"
  echo "failed: ${run_failure_count}"
  echo "skipped as completed: ${run_skipped_count}"
  echo "----------------------------------------"

  if (( total_count == 0 )); then
    echo "No CR/SR block invocations were scheduled."
  else
    for index in "${!block_summary_statuses[@]}"; do
      exit_code="${block_summary_exit_codes[index]}"
      signal_suffix=""
      if [[ "${block_summary_statuses[index]}" == "FAILED" ]] && (( exit_code > 128 )); then
        signal_number=$((exit_code - 128))
        signal_suffix=" signal=${signal_number}"
      fi
      duration_text=$(format_duration "${block_summary_durations[index]}")

      printf '[%02d] %s mode=%s exit=%s%s duration=%s\n' \
        "$((index + 1))" \
        "${block_summary_statuses[index]}" \
        "${block_summary_modes[index]}" \
        "${exit_code}" \
        "${signal_suffix}" \
        "${duration_text}"
      echo "     years: ${block_summary_years[index]}"
      echo "     categories: ${block_summary_categories[index]}"
      echo "     variables: ${block_summary_variables[index]}"
      echo "     output_tag: ${block_summary_output_tags[index]}"
    done
  fi

  echo "========================================"
}

build_common_command_options() {
  local -n cmd_ref="$1"

  cmd_ref+=(
    --ttgamma-sample-role-policy "${ttgamma_sample_role_policy}"
    --sample-universe-wrapper "run_cr.sh -> fullR3_run.sh"
  )

  if [[ "${profile_phase_telemetry}" == "true" ]]; then
    cmd_ref+=(--phase-telemetry)
  fi

  # cmd_ref+=(--analysis-mode taufitter)

  if [[ "${do_systs}" == "true" ]]; then
    cmd_ref+=(--do-systs)
  fi

  if is_stat_diagnostic_profile; then
    cmd_ref+=(--record-raw-count)
  fi

  if [[ "${do_np}" == "true" ]]; then
    # The independent data-driven stage starts after the source receipt is verified.
    cmd_ref+=(--do-np --defer-np)
  fi

  cmd_ref+=(
    -p "${production_pkl_root}"
    --all-analysis
  )

  cmd_ref+=(--env-file "${production_env_file}")

  if [[ "${production_environment_snapshot}" == "true" ]]; then
    cmd_ref+=(
      --snapshot
    )
  fi
  if [[ -n "${production_sumw2_options_path}" ]]; then
    cmd_ref+=(--options "${production_sumw2_options_path}")
  fi
  cmd_ref+=(-x work_queue)
  if [[ "${profile_wq_port_provided}" == "true" ]]; then
    cmd_ref+=(--port "${profile_wq_port}")
  fi

  if [[ "${split_lep_flavor}" == "true" ]]; then
    cmd_ref+=(--split-lep-flavor)
  fi

  if [[ "${dry_run}" == "true" ]]; then
    cmd_ref+=(--dry-run)
  fi
}

append_pkl_safety_options() {
  local -n pkl_cmd_ref="$1"
  pkl_cmd_ref+=(--pkl-no-clobber)
  if [[ -n "${profile_pkl_min_free_bytes}" ]]; then
    pkl_cmd_ref+=(
      --pkl-min-free-bytes "${profile_pkl_min_free_bytes}"
      --pkl-min-free-inodes "${profile_pkl_min_free_inodes}"
      --pkl-safety-reserve-bytes "${profile_pkl_safety_reserve_bytes}"
      --pkl-safety-reserve-inodes "${profile_pkl_safety_reserve_inodes}"
    )
  fi
}

append_guard_options() {
  local -n guarded_cmd_ref="$1"
  local block_id="$2"
  [[ -n "${profile_wq_staging_base}" ]] || return 0
  guarded_cmd_ref+=(
    --wq-guard
    --wq-filepath "${profile_wq_staging_base}"
    --wq-invocation-id "${campaign_tag}_${block_id}_source_attempt_${source_attempt_number}"
    --wq-free-warning-bytes "${profile_wq_warning_bytes}"
    --wq-min-free-bytes "${profile_wq_min_free_bytes}"
    --wq-poll-seconds "${profile_wq_poll_seconds}"
  )
  if [[ -n "${profile_wq_quota_root}" ]]; then
    guarded_cmd_ref+=(--wq-quota-root "${profile_wq_quota_root}")
  fi
  if [[ -n "${profile_wq_treereduction}" ]]; then
    guarded_cmd_ref+=(--wq-treereduction "${profile_wq_treereduction}")
  fi
  if [[ -n "${profile_wq_processing_cores}" ]]; then
    guarded_cmd_ref+=(--wq-processing-cores "${profile_wq_processing_cores}")
  fi
  if [[ -n "${profile_wq_accumulating_memory_mb}" ]]; then
    guarded_cmd_ref+=(
      --wq-accumulating-memory-mb "${profile_wq_accumulating_memory_mb}"
      --wq-accumulating-cores "${profile_wq_accumulating_cores}"
    )
  fi
}

complete_guarded_staging() {
  local block_id="$1"
  [[ -n "${profile_wq_staging_base}" && "${dry_run}" == "false" ]] || return 0
  [[ -z "${validation_backend}" ]] || return 0
  python ./wq_staging_guard.py cleanup \
    "${profile_wq_staging_base}/${campaign_tag}_${block_id}_source_attempt_${source_attempt_number}"
}

native_wq_log_names=(debug.log tr.log stats.log tasks.log)

set_native_wq_log_dir() {
  local block_id="$1"
  local stage="$2"
  native_wq_log_dir="${output_dir}/work_queue_logs/${production_component}/${block_id}/${stage}_native"
}

assert_native_wq_logs_clean() {
  local block_id="$1"
  local name
  mkdir -p -- "${native_wq_log_dir}"
  # A telemetry-only remainder still belongs to the previous manager attempt.
  for name in "${native_wq_log_names[@]}" phase_telemetry.jsonl; do
    if [[ -e "${native_wq_log_dir}/${name}" ]]; then
      archive_native_wq_logs "${block_id}" source
      break
    fi
  done
  for name in "${native_wq_log_names[@]}" phase_telemetry.jsonl; do
    if [[ -e "${native_wq_log_dir}/${name}" ]]; then
      echo "ERROR: owned Work Queue log already exists for this fragment and stage; refusing a new invocation: ${native_wq_log_dir}/${name}" >&2
      return 1
    fi
  done
}

run_production_source_child() {
  local block_id="$1"
  local source_path="$2"
  local nonprompt_path="$3"
  shift 3
  local command=("$@")

  if [[ -n "${validation_backend}" ]]; then
    "${validation_backend}" run_block "${validation_scenario}" \
      "${block_id}" "${production_env_file}" source \
      "${source_path}" "${nonprompt_path}" "${native_wq_log_dir}" \
      "${command[@]}"
  else
    "${command[@]}"
  fi
}

run_production_nonprompt_child() {
  local block_id="$1"
  local source_path="$2"
  local nonprompt_path="$3"
  shift 3
  local command=("$@")

  if [[ -n "${validation_backend}" ]]; then
    "${validation_backend}" run_nonprompt "${validation_scenario}" \
      "${block_id}" "${source_path}" "${nonprompt_path}" "${command[@]}"
  else
    "${command[@]}"
  fi
}

archive_native_wq_logs() {
  local block_id="$1"
  local stage="$2"
  local archive_base="${output_dir}/work_queue_logs/${production_component}/${block_id}/${stage}"
  local archive_dir="${archive_base}"
  local retry_index=1
  local name source_path destination_path source_size destination_size

  while [[ -e "${archive_dir}" ]]; do
    archive_dir="${archive_base}_resume_${retry_index}"
    retry_index=$((retry_index + 1))
  done
  mkdir -p -- "$(dirname -- "${archive_dir}")"
  mkdir -- "${archive_dir}"

  for name in "${native_wq_log_names[@]}"; do
    source_path="${native_wq_log_dir}/${name}"
    destination_path="${archive_dir}/${name}"
    if [[ ! -e "${source_path}" ]]; then
      continue
    fi
    if [[ ! -f "${source_path}" ]]; then
      echo "ERROR: native Work Queue source is not a regular file: ${source_path}" >&2
      return 1
    fi
    if ! cp -- "${source_path}" "${destination_path}"; then
      echo "ERROR: failed to copy native Work Queue log; originals were preserved: ${source_path}" >&2
      return 1
    fi
    source_size=$(stat -c %s -- "${source_path}")
    destination_size=$(stat -c %s -- "${destination_path}")
    if [[ ! -f "${destination_path}" || "${source_size}" != "${destination_size}" ]]; then
      echo "ERROR: native Work Queue log copy verification failed; originals were preserved: ${source_path}" >&2
      return 1
    fi
  done

  if [[ -n "${validation_backend}" ]] \
    && { [[ "${validation_scenario}" == "archive_copy_failure" ]] \
      || { [[ "${validation_scenario}" == "run3_archive_copy_failure" ]] \
        && [[ "${production_component}" == "run3" ]]; }; }; then
    echo "ERROR: simulated native Work Queue archive verification failure; originals were preserved." >&2
    return 1
  fi

  # Preserve the attempt-owned inode under a stable native name. The producer
  # basename must remain absent so a later invocation cannot append to it.
  # Optional telemetry never copies bulk data or changes scientific status.
  source_path="${native_wq_log_dir}/phase_telemetry.jsonl"
  if [[ -e "${source_path}" || -L "${source_path}" ]]; then
    if ! python -S - "${source_path}" "${archive_dir}" <<'PY_TELEMETRY'
import os
from pathlib import Path
import stat
import sys

source = Path(sys.argv[1])
archive = Path(sys.argv[2])
target = source.with_name("phase_telemetry." + archive.name + ".jsonl")
link = archive / "phase_telemetry.jsonl"
before = source.lstat()
if not stat.S_ISREG(before.st_mode):
    raise ValueError("telemetry source is not a regular file")
# The campaign writer lock and newly created attempt archive serialize this
# name. Refuse any existing target, including a dangling link, before rename.
# os.rename is atomic on one filesystem and raises EXDEV without copying.
if os.path.lexists(target):
    raise FileExistsError("attempt telemetry target already exists: " + str(target))
os.rename(source, target)
after = target.lstat()
if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
    raise ValueError("telemetry rename identity changed")
os.symlink(os.path.relpath(target, archive), link)
observed = link.stat()
if (observed.st_dev, observed.st_ino, observed.st_size) != (after.st_dev, after.st_ino, after.st_size):
    raise ValueError("telemetry link target identity changed")
# A metadata check alone does not demonstrate that the target can be opened.
with link.open("rb"):
    pass
PY_TELEMETRY
    then
      echo "WARNING: optional phase telemetry archive failed; original retained at ${source_path} or ${native_wq_log_dir}/phase_telemetry.$(basename -- "${archive_dir}").jsonl." >&2
    fi
  fi
  production_state_tool archive "${production_state_path}" "${block_id}" "${stage}" "${archive_dir}"
  for name in "${native_wq_log_names[@]}"; do
    source_path="${native_wq_log_dir}/${name}"
    [[ ! -e "${source_path}" ]] || rm -- "${source_path}"
  done

}

write_failure_snapshot() {
  local block_id="$1"
  local stage="$2"
  local snapshot_dir="${output_dir}/failure_diagnostics"
  mkdir -p -- "${snapshot_dir}"
  production_state_tool failure_snapshot \
    "${production_state_path}" "${block_id}" "${stage}" \
    "${snapshot_dir}/${block_id}_${stage}.tsv"
}

run_campaign_unit() {
  local year_expr="$1"
  local var_set="$2"
  shift 2
  local cats=("${@}")
  local years=()
  local vars=()
  local block_id
  local output_tag
  local source_path
  local data_driven_path
  local legacy_source_path
  local legacy_data_driven_path
  local prior_source_path
  local prior_data_driven_path
  local status
  local source_status
  local data_driven_status
  local source_reusable=false
  local source_exit_code=0
  local data_driven_exit_code=0
  local source_pending_path=""
  local data_driven_pending_path=""
  local start_epoch
  local duration_seconds
  local region_option

  assert_supported_year_expr "${year_expr}"
  read -r -a years <<< "${year_expr}"
  read -r -a vars <<< "${var_set}"
  block_id=$(production_block_id "${year_expr}" "${cats[*]}" "${var_set}")
  IFS=$'\t' read -r _ _ _ _ output_tag _ source_path data_driven_path < <(
    awk -F $'\t' -v block_id="${block_id}" '$1 == block_id {print; exit}' "${production_plan_file}"
  )
  if [[ -z "${source_path}" || -z "${data_driven_path}" ]]; then
    echo "ERROR: unit ${block_id} has no expected output paths." >&2
    exit 1
  fi
  legacy_source_path="${source_path}"
  legacy_data_driven_path="${data_driven_path}"
  set_native_wq_log_dir "${block_id}" source

  if [[ "${dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    status=planned
    source_status=planned
    data_driven_status=blocked
  else
    IFS=$'\t' read -r status source_status data_driven_status < <(
      production_state_tool status "${production_state_path}" "${block_id}"
    )
  fi
  if [[ "${status}" == "success" ]]; then
    echo "Skipping verified completed unit: ${block_id}"
    record_block_result SKIPPED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" 0 0
    return 0
  fi
  case "${source_status}" in
    ready)
      source_reusable=true
      ;;
    planned|failed)
      if [[ -e "${source_path}" || -L "${source_path}" \
        || -e "${source_path}.metadata.json" || -L "${source_path}.metadata.json" \
        || -e "${data_driven_path}" || -L "${data_driven_path}" \
        || -e "${data_driven_path}.metadata.json" || -L "${data_driven_path}.metadata.json" ]]; then
        echo "ERROR: recovery_evidence_insufficient: unverified output exists for ${block_id}." >&2
        exit 1
      fi
      if [[ "${dry_run}" == "false" ]]; then
        prior_source_path=$(production_state_tool effective-path "${production_state_path}" "${block_id}" source)
        if [[ "${prior_source_path}" != "${legacy_source_path}" ]] \
          && { [[ -e "${prior_source_path}" || -L "${prior_source_path}" \
            || -e "${prior_source_path}.metadata.json" || -L "${prior_source_path}.metadata.json" ]]; }; then
          echo "ERROR: recovery_evidence_insufficient: prior source attempt has output evidence." >&2
          exit 1
        fi
      fi
      ;;
    running)
      echo "ERROR: recovery_evidence_insufficient: source writer may still own ${block_id}." >&2
      exit 1
      ;;
    *)
      echo "ERROR: invalid source stage status for ${block_id}." >&2
      exit 1
      ;;
  esac
  if [[ "${data_driven_status}" == "running" ]]; then
    echo "ERROR: recovery_evidence_insufficient: data-driven writer may still own ${block_id}." >&2
    exit 1
  fi
  if [[ "${data_driven_status}" != "success" ]] \
    && { [[ -e "${data_driven_path}" || -L "${data_driven_path}" \
      || -e "${data_driven_path}.metadata.json" || -L "${data_driven_path}.metadata.json" ]]; }; then
    echo "ERROR: recovery_evidence_insufficient: data-driven output lacks a verified receipt." >&2
    exit 1
  fi
  if [[ "${dry_run}" == "false" && "${data_driven_status}" != "success" ]]; then
    prior_data_driven_path=$(production_state_tool effective-path "${production_state_path}" "${block_id}" data_driven)
    if [[ "${prior_data_driven_path}" != "${legacy_data_driven_path}" ]] \
      && { [[ -e "${prior_data_driven_path}" || -L "${prior_data_driven_path}" \
        || -e "${prior_data_driven_path}.metadata.json" || -L "${prior_data_driven_path}.metadata.json" ]]; }; then
      echo "ERROR: recovery_evidence_insufficient: prior data-driven attempt has output evidence." >&2
      exit 1
    fi
  fi
  if [[ "${source_reusable}" == "true" ]]; then
    source_path=$(production_state_tool effective-path "${production_state_path}" "${block_id}" source)
  fi
  production_pkl_root="$(dirname -- "${source_path}")"
  if [[ "${source_reusable}" == "false" && "${source_path}" != "${legacy_source_path}" ]] \
    && { [[ -e "${source_path}" || -L "${source_path}" \
      || -e "${source_path}.metadata.json" || -L "${source_path}.metadata.json" ]]; }; then
    echo "ERROR: unverified new source target or sidecar already exists: ${source_path}" >&2
    exit 1
  fi
  if [[ "${data_driven_status}" != "success" && "${data_driven_path}" != "${legacy_data_driven_path}" ]] \
    && { [[ -e "${data_driven_path}" || -L "${data_driven_path}" \
      || -e "${data_driven_path}.metadata.json" || -L "${data_driven_path}.metadata.json" ]]; }; then
    echo "ERROR: unverified new data-driven target or sidecar already exists: ${data_driven_path}" >&2
    exit 1
  fi

  if [[ "${profile_region}" == "CR" ]]; then
    region_option=--cr
  else
    region_option=--sr
  fi
  source_attempt_number=1
  if [[ "${profile_wq_manager_prefix_provided}" == "true" \
    && "${source_reusable}" == "false" ]]; then
    source_attempt_number=$(production_state_tool next-source-attempt "${production_state_path}" "${block_id}")
  elif [[ "${dry_run}" == "false" ]]; then
    source_attempt_number=$(production_state_tool next-attempt "${production_state_path}" "${block_id}" source)
  fi
  source_cmd=(
    ./fullR3_run.sh -y "${years[@]}" -t "${output_tag}" -s "${chunk_size}"
    "${region_option}" --hist-vars "${vars[@]}" --category-groups "${cats[@]}"
  )
  build_common_command_options source_cmd
  if [[ "${profile_wq_manager_name_provided}" == "true" ]]; then
    source_cmd+=(--wq-manager-name "${profile_wq_manager_name}")
  elif [[ "${profile_wq_manager_prefix_provided}" == "true" \
    && "${source_reusable}" == "false" ]]; then
    source_cmd+=(--wq-manager-name "${profile_wq_manager_prefix}${block_id}-a${source_attempt_number}")
  fi
  append_guard_options source_cmd "${block_id}"
  source_cmd+=(--wq-log-dir "${native_wq_log_dir}")
  data_driven_cmd=(
    python ./run_data_driven.py --input-pkl "${source_path}" --output-pkl "${data_driven_path}"
  )
  if [[ "${profile_phase_telemetry}" == "true" ]]; then
    data_driven_cmd+=(--phase-telemetry)
  fi
  append_pkl_safety_options source_cmd
  append_pkl_safety_options data_driven_cmd

  echo "Unit: ${block_id}"
  echo "Years: ${year_expr}; region: ${profile_region}"
  echo "Categories: ${cats[*]}; variables: ${vars[*]}"
  echo "Source status: ${source_status}; data-driven status: ${data_driven_status}"
  start_epoch=$(date +%s)
  if [[ "${source_reusable}" == "false" ]]; then
    if [[ "${dry_run}" == "false" ]]; then
      assert_native_wq_logs_clean "${block_id}"
      production_state_tool mark "${production_state_path}" "${block_id}" \
        source running none source_child_started -- "${source_cmd[@]}"
      source_pending_path=$(production_state_tool pending-path "${production_state_path}" "${block_id}" source)
      source_cmd+=(--pkl-pending-id "$(production_state_tool pending-id "${production_state_path}" "${block_id}" source)")
    fi
    print_command "${source_cmd[@]}"
    if [[ "${dry_run}" == "true" ]]; then
      if "${source_cmd[@]}"; then source_exit_code=0; else source_exit_code=$?; fi
    elif run_production_source_child "${block_id}" "${source_path}" \
      "${data_driven_path}" "${source_cmd[@]}"; then
      source_exit_code=0
    else
      source_exit_code=$?
    fi
  else
    echo "Reusing verified source: ${source_path}"
  fi
  duration_seconds=$(($(date +%s) - start_epoch))
  if [[ "${dry_run}" == "true" ]]; then
    echo "Data-driven command (not executed in dry-run):"
    print_command "${data_driven_cmd[@]}"
    if (( source_exit_code == 0 )); then
      record_block_result DRY_RUN "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" 0 "${duration_seconds}"
    else
      record_block_result FAILED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" "${source_exit_code}" "${duration_seconds}"
    fi
    return 0
  fi

  if [[ "${source_reusable}" == "false" ]]; then
    if (( source_exit_code != 0 )) \
      || [[ ! -s "${source_path}" || ! -s "${source_path}.metadata.json" ]] \
      || [[ -e "${data_driven_path}" || -e "${data_driven_path}.metadata.json" ]]; then
      if [[ -n "${source_pending_path}" ]] \
        && { [[ -e "${source_pending_path%/*}" || -L "${source_pending_path%/*}" ]]; }; then
        echo "ERROR: source publication left attempt-owned pending evidence; writer remains running for exclusive resume reconciliation." >&2
        record_block_result FAILED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" "${source_exit_code}" "${duration_seconds}"
        return 0
      fi
      production_state_tool mark "${production_state_path}" "${block_id}" \
        source failed "${source_exit_code}" source_failed_or_invalid_output "${duration_seconds}"
      archive_native_wq_logs "${block_id}" source
      write_failure_snapshot "${block_id}" source
      record_block_result FAILED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" "${source_exit_code}" "${duration_seconds}"
      return 0
    fi
    production_state_tool mark "${production_state_path}" "${block_id}" \
      source ready 0 source_child_completed "${duration_seconds}"
    archive_native_wq_logs "${block_id}" source
    if ! complete_guarded_staging "${block_id}"; then
      echo "WARNING: guarded staging cleanup requires review; source receipt remains valid." >&2
    fi
  fi
  production_state_tool mark "${production_state_path}" "${block_id}" \
    data_driven running none data_driven_child_started "${data_driven_cmd[@]}"
  data_driven_pending_path=$(production_state_tool pending-path "${production_state_path}" "${block_id}" data_driven)
  data_driven_cmd+=(--pkl-pending-id "$(production_state_tool pending-id "${production_state_path}" "${block_id}" data_driven)")
  print_command "${data_driven_cmd[@]}"
  if run_production_nonprompt_child "${block_id}" "${source_path}" \
    "${data_driven_path}" "${data_driven_cmd[@]}"; then
    data_driven_exit_code=0
  else
    data_driven_exit_code=$?
  fi
  duration_seconds=$(($(date +%s) - start_epoch))
  if (( data_driven_exit_code == 0 )) \
    && [[ -s "${data_driven_path}" && -s "${data_driven_path}.metadata.json" ]]; then
    production_state_tool mark "${production_state_path}" "${block_id}" \
      data_driven success 0 data_driven_child_completed "${duration_seconds}"
    record_block_result SUCCESS "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" 0 "${duration_seconds}"
  else
    if [[ -n "${data_driven_pending_path}" ]] \
      && { [[ -e "${data_driven_pending_path%/*}" || -L "${data_driven_pending_path%/*}" ]]; }; then
      echo "ERROR: data-driven publication left attempt-owned pending evidence; writer remains running for exclusive resume reconciliation." >&2
      record_block_result FAILED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" "${data_driven_exit_code}" "${duration_seconds}"
      return 0
    fi
    production_state_tool mark "${production_state_path}" "${block_id}" \
      data_driven failed "${data_driven_exit_code}" data_driven_child_failed_or_invalid_output "${duration_seconds}"
    write_failure_snapshot "${block_id}" data_driven
    record_block_result FAILED "${profile_region}" "${year_expr}" "${cats[*]}" "${vars[*]}" "${output_tag}" "${data_driven_exit_code}" "${duration_seconds}"
  fi
}

###############################################################################
# Preflight summary
###############################################################################

assert_boolean "${run_cr}" "run_cr"
assert_boolean "${run_sr}" "run_sr"
assert_boolean "${dry_run}" "dry_run"
assert_boolean "${do_systs}" "do_systs"
assert_boolean "${do_np}" "do_np"
assert_boolean "${split_lep_flavor}" "split_lep_flavor"
assert_boolean "${profile_resume}" "profile_resume"

assert_parallel_array_lengths \
  "CR category-to-variable mapping" \
  "${#cr_category_sets[@]}" \
  "${#cr_category_var_set_names[@]}"

assert_parallel_array_lengths \
  "run3_prod SR category-to-variable mapping" \
  "${#sr_category_sets[@]}" \
  "${#sr_category_var_set_names[@]}"

assert_parallel_array_lengths \
  "SR year-to-category mapping" \
  "${#sr_year_sets[@]}" \
  "${#sr_year_category_set_names[@]}"

assert_parallel_array_lengths \
  "SR year-to-variable-map mapping" \
  "${#sr_year_sets[@]}" \
  "${#sr_year_category_var_set_names[@]}"

for var_set_name in \
  "${cr_category_var_set_names[@]}" \
  "${sr_category_var_set_names[@]}"; do
  assert_array_defined "${var_set_name}"
done

for category_array_name in "${sr_year_category_set_names[@]}"; do
  assert_array_defined "${category_array_name}"
done

for mapping_array_name in "${sr_year_category_var_set_names[@]}"; do
  assert_array_defined "${mapping_array_name}"
done

resolve_production_environment

echo "========================================"
echo "run_cr.sh configuration"
echo "production_profile: ${production_profile}"
echo "campaign_tag: ${campaign_tag}"
echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
echo "output_dir: ${output_dir}"
echo "chunk_size: ${chunk_size}"
echo "run_cr: ${run_cr}"
echo "run_sr: ${run_sr}"
echo "dry_run: ${dry_run}"
echo "do_systs: ${do_systs}"
echo "do_np: ${do_np}"
echo "split_lep_flavor: ${split_lep_flavor}"
echo "resume: ${profile_resume}"
echo "allow_environment_migration: ${profile_allow_environment_migration}"
echo "env_file: ${production_env_file}"
echo "env_file_sha256: ${production_env_file_sha256}"
echo "environment_mode: ${production_environment_mode}"
echo "environment_policy: ${production_environment_mode}"
if is_stat_diagnostic_profile; then
  echo "sumw2_storage_mode: full_diagnostics"
else
  echo "sumw2_storage_mode: production (implicit)"
fi
echo "campaign_state: ${output_dir}/${production_state_filename}"
print_var_sets "CR non-tau" "${cr_non_tau_var_sets[@]}"
print_var_sets "CR tau" "${cr_tau_var_sets[@]}"
print_var_sets "SR with ptz_wtau" "${sr_with_ptz_wtau_var_sets[@]}"
print_var_sets "SR off-Z" "${sr_offz_var_sets[@]}"
print_var_sets "SR on-Z/tau" "${sr_onz_tau_var_sets[@]}"
print_var_sets "SR forward" "${sr_fwd_var_sets[@]}"

echo "run3_prod Run-3 SR category blocks:"
printf '  %s\n' "${sr_category_sets[@]}"

echo "========================================"
echo

case "${ttgamma_sample_role_policy}" in
  split) ;;
  *)
    cat >&2 <<EOF
ERROR: this production helper is intended to run the nominal split policy.

Current ttgamma_sample_role_policy:
  ${ttgamma_sample_role_policy}

Expected:
  split
EOF
    exit 1
    ;;
esac

trap cleanup_production_plan EXIT
prepare_production_campaign
prepare_production_sumw2_options
if [[ "${production_prepare_only:-false}" == "true" ]]; then
  printf '%s\n' "${production_block_ids[@]}" >&3
  exit 0
fi

###############################################################################
# Execute the selected unit from the complete state-owned plan
###############################################################################

for plan_index in "${!production_block_ids[@]}"; do
  current_fragment="${production_block_ids[plan_index]}"
  if [[ -n "${profile_selected_fragment}" \
    && "${current_fragment}" != "${profile_selected_fragment}" ]]; then
    continue
  fi
  read -r -a cats <<< "${production_plan_category_sets[plan_index]}"
  run_campaign_unit "${production_plan_year_exprs[plan_index]}" \
    "${production_plan_var_sets[plan_index]}" "${cats[@]}"
  if [[ -n "${profile_wq_staging_base}" ]] && (( run_failure_count > 0 )); then
    if [[ "${dry_run}" == "true" ]]; then
      echo "Guarded campaign stopped after a failed dry-run unit." >&2
      break
    fi
    current_status_line=$(production_state_tool status "${production_state_path}" "${current_fragment}")
    current_source_status=$(cut -f2 <<< "${current_status_line}")
    if [[ "${current_source_status}" == "failed" ]]; then
      echo "Guarded campaign stopped after a source failure." >&2
      break
    fi
  fi
done

###############################################################################
# Final status
###############################################################################

print_run_summary

echo "campaign_tag: ${campaign_tag}"
echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
echo "output_dir: ${output_dir}"

if [[ "${dry_run}" == "true" ]]; then
  echo "dry_run_complete: ${#block_summary_statuses[@]} unit commands resolved; no campaign state, scheduler, or production action was created"
fi

final_process_exit_code=0
if [[ "${dry_run}" == "false" ]]; then
  IFS=$'\t' read -r final_campaign_classification final_process_exit_code < <(
    production_state_tool finalize "${production_state_path}" "${profile_selected_fragment}"
  )
  production_state_tool report \
    "${production_state_path}" \
    "${output_dir}/campaign_summary.tsv" \
    "${output_dir}/campaign_summary.md"
  echo "final_campaign_classification: ${final_campaign_classification}"
  echo "campaign_summary_tsv: ${output_dir}/campaign_summary.tsv"
  echo "campaign_summary_md: ${output_dir}/campaign_summary.md"
elif (( run_failure_count > 0 )); then
  final_process_exit_code=1
fi

if (( final_process_exit_code != 0 )); then
  echo "run_cr.sh finished with ${run_failure_count} failed production block(s)." >&2
  exit "${final_process_exit_code}"
fi
echo "run_cr.sh completed successfully"
exit 0
