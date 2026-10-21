#!/usr/bin/env bash
set -euo pipefail

matrix_profile=""
matrix_dry_run=false
matrix_output_dir=""
matrix_campaign_tag=""
matrix_env_file=""
matrix_resume=false
matrix_reviewed_topeft_manager_head=""
matrix_reviewed_topcoffea_manager_head=""
matrix_wq_staging_base=""
matrix_guard_args=()
declare -A matrix_seen_guard=()
matrix_parse_errors=()

matrix_parse_args() {
  local args=("$@")
  local index=0
  while (( index < ${#args[@]} )); do
    case "${args[index]}" in
      --production-profile)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--production-profile requires a value")
          index=$((index + 1))
        else
          matrix_profile="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --dry-run) matrix_dry_run=true; index=$((index + 1)) ;;
      --output-dir)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--output-dir requires a value")
          index=$((index + 1))
        else
          matrix_output_dir="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --campaign-tag)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--campaign-tag requires a value")
          index=$((index + 1))
        else
          matrix_campaign_tag="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --env-file)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--env-file requires a value")
          index=$((index + 1))
        else
          matrix_env_file="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --reviewed-topeft-manager-head)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--reviewed-topeft-manager-head requires a value")
          index=$((index + 1))
        else
          matrix_reviewed_topeft_manager_head="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --reviewed-topcoffea-manager-head)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("--reviewed-topcoffea-manager-head requires a value")
          index=$((index + 1))
        else
          matrix_reviewed_topcoffea_manager_head="${args[index + 1]}"
          index=$((index + 2))
        fi
        ;;
      --resume) matrix_resume=true; index=$((index + 1)) ;;
      --wq-staging-base|--wq-quota-root|--wq-free-warning-bytes|--wq-min-free-bytes|--wq-poll-seconds|--wq-treereduction|--wq-processing-cores|--wq-accumulating-memory-mb|--wq-accumulating-cores)
        if (( index + 1 >= ${#args[@]} )) || [[ "${args[index + 1]}" == -* ]]; then
          matrix_parse_errors+=("${args[index]} requires a value")
          index=$((index + 1))
        else
          if [[ "${args[index]}" == "--wq-processing-cores" && "${args[index + 1]}" != "1" ]]; then
            matrix_parse_errors+=("--wq-processing-cores currently accepts only 1")
          fi
          [[ "${args[index]}" != "--wq-staging-base" ]] || matrix_wq_staging_base="${args[index + 1]}"
          matrix_seen_guard["${args[index]}"]=true
          matrix_guard_args+=("${args[index]}" "${args[index + 1]}")
          index=$((index + 2))
        fi
        ;;
      -h|--help) index=$((index + 1)) ;;
      --wq-warning-bytes|--wq-stop-bytes|--wq-reserve-bytes)
        matrix_parse_errors+=("retired Work Queue staging-budget option '${args[index]}'; use --wq-free-warning-bytes and --wq-min-free-bytes")
        index=$((index + 1))
        ;;
      *)
        matrix_parse_errors+=("unsupported run_cr.sh option '${args[index]}'")
        index=$((index + 1))
        ;;
    esac
  done
}

srplot009_component_classification() {
  local state_path="$1"
  python - "${state_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("blocked_global")
    raise SystemExit(0)
try:
    state = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("blocked_state_contradiction")
    raise SystemExit(0)
statuses = [block.get("status") for block in state.get("blocks", [])]
if any(status in {"source_running", "nonprompt_running"} for status in statuses):
    print("blocked_ambiguous")
elif any(status in {"planned", "source_ready"} for status in statuses):
    print("blocked_incomplete")
elif any(status in {"source_failed", "nonprompt_failed"} for status in statuses):
    print("complete_with_known_failures")
elif statuses and all(status == "success" for status in statuses):
    print("success")
else:
    print("blocked_state_contradiction")
PY
}

srplot009_shared_campaign_blocker() {
  local component_classification="$1"
  local native_log_dir
  local native_log_name

  if [[ "${component_classification}" == "blocked_ambiguous" ]]; then
    echo "possible running child remains unresolved"
    return 0
  fi

  native_log_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  if [[ -n "${SRPLOT009_VALIDATION_BACKEND:-}" ]]; then
    native_log_dir="${SRPLOT009_VALIDATION_ROOT}/native_logs"
  fi
  for native_log_name in debug.log tr.log stats.log tasks.log; do
    if [[ -e "${native_log_dir}/${native_log_name}" ]]; then
      echo "generic Work Queue logs remain unsafe for another invocation"
      return 0
    fi
  done

  return 1
}

srplot009_write_combined_summary() {
  local combined_profile="$1"
  local combined_tag="$2"
  local output_root="$3"
  local run2_state="$4"
  local run3_state="$5"
  local run2_classification="$6"
  local run3_classification="$7"
  local final_exit_code="$8"
  local shared_blocker_component="${9:-none}"
  python - "${combined_profile}" "${combined_tag}" "${output_root}" \
    "${run2_state}" "${run3_state}" "${run2_classification}" \
    "${run3_classification}" "${final_exit_code}" \
    "${shared_blocker_component}" <<'PY'
import json
import os
import sys
from pathlib import Path

profile, tag, root_text, run2_text, run3_text, run2_class, run3_class, exit_text, shared_component = sys.argv[1:]
root = Path(root_text)

if profile == "run2_run3_full":
    configured_components = [
        ("run2", "run2_full", [f"run2_full_{suffix}" for suffix in "abcde"]),
        ("run3", "run3_full", [f"run3_full_{suffix}" for suffix in "abcde"]),
    ]
elif profile == "run2_run3_full_CR":
    configured_components = [
        ("run2", "run2_full_CR", [f"run2_full_CR_block{index}" for index in range(1, 7)]),
        ("run3", "run3_full_CR", [f"run3_full_CR_block{index}" for index in range(1, 13)]),
    ]
else:
    raise SystemExit(f"unsupported combined profile {profile!r}")

def load_component(label, text, observed_classification):
    path = Path(text)
    if not path.is_file():
        return {"label": label, "classification": observed_classification, "state": None}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"label": label, "classification": "blocked_state_contradiction", "state": None}
    statuses = [block.get("status") for block in state.get("blocks", [])]
    if any(status in {"source_running", "nonprompt_running"} for status in statuses):
        classification = "blocked_ambiguous"
    elif any(status in {"planned", "source_ready"} for status in statuses):
        classification = "blocked_incomplete"
    elif any(status in {"source_failed", "nonprompt_failed"} for status in statuses):
        classification = "complete_with_known_failures"
    elif statuses and all(status == "success" for status in statuses):
        classification = "success"
    else:
        classification = "blocked_state_contradiction"
    return {"label": label, "classification": classification, "state": state}

components = [load_component("run2", run2_text, run2_class), load_component("run3", run3_text, run3_class)]
classifications = [component["classification"] for component in components]

headers = [
    "component", "component_status", "profile", "campaign_tag", "era", "block_id",
    "categories", "variables", "source_status", "source_exit", "nonprompt_status",
    "nonprompt_exit", "final_block_status", "expected_nominal_path", "expected_np_path",
    "aggregate_classification", "aggregate_reason",
]
rows = []
configured = attempted = successful = failed = attempted_incomplete = not_attempted = 0
shared_index = {"run2": 0, "run3": 1}.get(shared_component)
for component_index, (component_label, expected_profile, expected_block_ids) in enumerate(configured_components):
    component = components[component_index]
    state = component["state"]
    observed_blocks = {} if state is None else {
        block.get("id"): block for block in state.get("blocks", [])
    }
    for block_id in expected_block_ids:
        configured += 1
        block = observed_blocks.get(block_id)
        status = None if block is None else block.get("status")
        if status == "success":
            aggregate_classification = "success"
            aggregate_reason = "success"
            attempted += 1
            successful += 1
        elif status in {"source_failed", "nonprompt_failed"}:
            aggregate_classification = "known_failure"
            aggregate_reason = "known_failure"
            attempted += 1
            failed += 1
        elif status in {"source_running", "source_ready", "nonprompt_running"}:
            aggregate_classification = "attempted_incomplete"
            aggregate_reason = "attempted_incomplete"
            attempted += 1
            attempted_incomplete += 1
        else:
            aggregate_classification = "not_attempted"
            if shared_index is not None and component_index >= shared_index:
                aggregate_reason = "not_attempted_shared_campaign_blocked"
            else:
                aggregate_reason = "not_attempted_component_blocked"
            not_attempted += 1
        block = block or {}
        rows.append([
            component_label, component["classification"],
            state.get("production_profile") if state else expected_profile,
            state.get("campaign_tag") if state else f"{tag}_{component_label}",
            " ".join(block.get("years", [])), block_id,
            " ".join(block.get("category_groups", [])), " ".join(block.get("histograms", [])),
            block.get("source_status"), block.get("source_exit_code"), block.get("nonprompt_status"),
            block.get("nonprompt_exit_code"), status, block.get("expected_nominal_path"), block.get("expected_np_path"),
            aggregate_classification, aggregate_reason,
        ])

if shared_index is not None:
    final_classification = "shared_campaign_blocked"
elif attempted_incomplete or not_attempted or any(
    value.startswith("blocked_") or value == "not_attempted"
    for value in classifications
):
    final_classification = "completed_with_component_blockers"
elif failed:
    final_classification = "complete_with_known_failures"
else:
    final_classification = "success"

if successful + failed + attempted_incomplete + not_attempted != configured:
    raise SystemExit("combined campaign accounting invariant failed")
if attempted != successful + failed + attempted_incomplete:
    raise SystemExit("combined campaign attempted-count invariant failed")

def clean(value):
    return "" if value is None else str(value).replace("\t", " ").replace("\n", " ")

def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

tsv = "\t".join(headers) + "\n" + "".join("\t".join(clean(value) for value in row) + "\n" for row in rows)
atomic_text(root / "campaign_summary.tsv", tsv)
markdown = [
    f"# Combined campaign summary: {profile}", "", f"- campaign_tag: `{tag}`",
    f"- run2_component_status: `{components[0]['classification']}`",
    f"- run3_component_status: `{components[1]['classification']}`",
    f"- configured_count: {configured}", f"- attempted_count: {attempted}",
    f"- successful_count: {successful}", f"- known_failed_count: {failed}",
    f"- attempted_incomplete_count: {attempted_incomplete}",
    f"- not_attempted_count: {not_attempted}", f"- final_classification: `{final_classification}`",
    f"- final_process_exit_code: {exit_text}", "", "Per-block details are serialized in `campaign_summary.tsv`.", "",
]
atomic_text(root / "campaign_summary.md", "\n".join(markdown))
print(final_classification)
PY
}

if (( $# == 0 )) || { (( $# == 1 )) && [[ "$1" == "--dry-run" ]]; }; then
  default_dry_run=()
  [[ "${1:-}" == "--dry-run" ]] && default_dry_run=(--dry-run)
  set -- \
    --production-profile run2_full \
    --output-dir /groups/klannon/apiccine/run2_srplot009_current_branch \
    --campaign-tag current-branch-srplot009 \
    "${default_dry_run[@]}"
fi

matrix_parse_args "$@"
case "${matrix_profile}" in
  run2_full|run3_full|run2_full_CR|run3_full_CR|run2_run3_full|run2_run3_full_CR|t0_sr_statonly|t0_cr_statonly)
    if (( ${#matrix_parse_errors[@]} > 0 )); then
      printf 'ERROR: %s.\n' "${matrix_parse_errors[@]}" >&2
      exit 1
    fi
    if [[ -z "${matrix_output_dir}" || -z "${matrix_campaign_tag}" ]]; then
      echo "ERROR: ${matrix_profile} requires explicit --output-dir and --campaign-tag." >&2
      exit 1
    fi
    if [[ -n "${matrix_env_file}" && "${matrix_env_file}" != /* ]]; then
      echo "ERROR: ${matrix_profile} --env-file must be an absolute path: ${matrix_env_file}" >&2
      exit 1
    fi
    ;;
esac

case "${matrix_profile}" in
  run2_run3_full|run2_run3_full_CR)
    if [[ "${matrix_dry_run}" == "false" ]]; then
      for required_guard in --wq-staging-base --wq-quota-root --wq-free-warning-bytes \
        --wq-min-free-bytes --wq-poll-seconds; do
        if [[ -z "${matrix_seen_guard[${required_guard}]:-}" ]]; then
          echo "ERROR: combined Work Queue production requires ${required_guard} before output creation." >&2
          exit 1
        fi
      done
      if [[ -z "${matrix_wq_staging_base}" ]]; then
        echo "ERROR: Work Queue production requires guarded manager staging; implicit /tmp staging is forbidden." >&2
        exit 1
      fi
      if [[ "${matrix_wq_staging_base}" != /* || ! -d "${matrix_wq_staging_base}" ]]; then
        echo "ERROR: staging base must be an existing absolute directory." >&2
        exit 1
      fi
      matrix_staging_real=$(realpath -e -- "${matrix_wq_staging_base}") || exit 1
      if [[ "${matrix_staging_real}" == /tmp || "${matrix_staging_real}" == /tmp/* \
        || "${matrix_staging_real}" == /var/tmp || "${matrix_staging_real}" == /var/tmp/* \
        || "$(stat -c %d -- "${matrix_staging_real}")" == "$(stat -c %d -- /)" ]]; then
        echo "ERROR: /tmp or root-backed manager staging is forbidden for Work Queue production." >&2
        exit 1
      fi
    fi
    if [[ "${matrix_resume}" == "false" && -e "${matrix_output_dir}" ]]; then
      echo "ERROR: combined output namespace already exists: ${matrix_output_dir}" >&2
      exit 1
    fi
    if [[ "${matrix_resume}" == "true" && ! -d "${matrix_output_dir}" ]]; then
      echo "ERROR: ${matrix_profile} --resume requires an existing combined output namespace: ${matrix_output_dir}" >&2
      exit 1
    fi
    combined_suffix=""
    first_profile=run2_full
    second_profile=run3_full
    if [[ "${matrix_profile}" == "run2_run3_full_CR" ]]; then
      combined_suffix=_CR
      first_profile=run2_full_CR
      second_profile=run3_full_CR
    fi
    if [[ "${matrix_dry_run}" == "false" && "${matrix_resume}" == "false" ]]; then
      mkdir -- "${matrix_output_dir}"
    fi
    component_common=()
    [[ -n "${matrix_env_file}" ]] && component_common+=(--env-file "${matrix_env_file}")
    [[ -n "${matrix_reviewed_topeft_manager_head}" ]] \
      && component_common+=(--reviewed-topeft-manager-head "${matrix_reviewed_topeft_manager_head}")
    [[ -n "${matrix_reviewed_topcoffea_manager_head}" ]] \
      && component_common+=(--reviewed-topcoffea-manager-head "${matrix_reviewed_topcoffea_manager_head}")
    [[ "${matrix_dry_run}" == "true" ]] && component_common+=(--dry-run)
    [[ "${matrix_resume}" == "true" ]] && component_common+=(--resume)
    component_common+=("${matrix_guard_args[@]}")
    if "$0" --production-profile "${first_profile}" \
      --output-dir "${matrix_output_dir}/run2${combined_suffix}" \
      --campaign-tag "${matrix_campaign_tag}_run2" "${component_common[@]}"; then
      first_exit_code=0
    else
      first_exit_code=$?
    fi
    if [[ "${matrix_dry_run}" == "true" ]]; then
      (( first_exit_code == 0 )) || exit "${first_exit_code}"
    else
      first_state="${matrix_output_dir}/run2${combined_suffix}/.${first_profile}_campaign_state.json"
      first_classification=$(srplot009_component_classification "${first_state}")
      if first_shared_blocker=$(srplot009_shared_campaign_blocker "${first_classification}"); then
        second_state="${matrix_output_dir}/run3${combined_suffix}/.${second_profile}_campaign_state.json"
        srplot009_write_combined_summary \
          "${matrix_profile}" "${matrix_campaign_tag}" "${matrix_output_dir}" \
          "${first_state}" "${second_state}" "${first_classification}" \
          not_attempted 1 run2
        echo "ERROR: shared campaign safety blocker after Run 2 (${first_shared_blocker}); Run 3 was not started." >&2
        exit 1
      fi
      if [[ "${first_classification}" != "success" ]]; then
        echo "Run-2 component ended with local status ${first_classification}; entering independent Run 3." >&2
      fi
    fi
    if "$0" --production-profile "${second_profile}" \
      --output-dir "${matrix_output_dir}/run3${combined_suffix}" \
      --campaign-tag "${matrix_campaign_tag}_run3" "${component_common[@]}"; then
      second_exit_code=0
    else
      second_exit_code=$?
    fi
    if [[ "${matrix_dry_run}" == "true" ]]; then
      exit "${second_exit_code}"
    fi
    second_state="${matrix_output_dir}/run3${combined_suffix}/.${second_profile}_campaign_state.json"
    second_classification=$(srplot009_component_classification "${second_state}")
    shared_blocker_component=none
    if second_shared_blocker=$(srplot009_shared_campaign_blocker "${second_classification}"); then
      shared_blocker_component=run3
      echo "ERROR: shared campaign safety blocker in Run 3 (${second_shared_blocker})." >&2
    fi
    combined_exit_code=0
    if [[ "${first_classification}" != "success" || "${second_classification}" != "success" ]]; then
      combined_exit_code=1
    fi
    srplot009_write_combined_summary \
      "${matrix_profile}" "${matrix_campaign_tag}" "${matrix_output_dir}" \
      "${first_state}" "${second_state}" "${first_classification}" \
      "${second_classification}" "${combined_exit_code}" \
      "${shared_blocker_component}"
    exit "${combined_exit_code}"
    ;;
esac

print_usage() {
  cat <<'EOF'
Usage: ./run_cr.sh [--dry-run]
       ./run_cr.sh --production-profile PROFILE [--dry-run] \
  [--output-dir PATH] [--campaign-tag TAG] [--env-file PATH] [--resume] \
  [--reviewed-topeft-manager-head SHA] \
  [--reviewed-topcoffea-manager-head SHA] [--selected-fragment ID] \
  [--chunksize N] [--wq-processing-cores 1] \
  [--wq-accumulating-memory-mb MB --wq-accumulating-cores CORES] \
  [--sr-plan JSON] [--activate-pending-plan JSON --expected-plan-revision ID]

Public PROFILE values:
  run2_full, run3_full, run2_run3_full
  run2_sr_ext, run3_sr_ext
  run2_full_CR, run3_full_CR, run2_run3_full_CR
  t0_sr_statonly, t0_cr_statonly

With no arguments, run_cr.sh remains a backward-compatible alias for the fixed
five-block run2_full campaign. Combined profiles run Run 2 then Run 3 in
separate child namespaces. A safely classified Run-2-local failure or blocker
does not suppress the independent Run-3 component; a shared unsafe state does.

For maintained profiles, omitting --env-file resolves the current worker
environment, while --env-file PATH selects that absolute archive explicitly as
a frozen snapshot. Work Queue has no profile-level worker count. Ordinary full
profiles use the implicit production sumw2 policy; T0 stat-only profiles use
explicit full_diagnostics storage. Explicit component and combined profiles
require a fresh absolute output directory and campaign tag.
t0_sr_statonly covers the maintained Run-2 and early-Run-3 SR mapping with
nominal weights and raw counts. t0_cr_statonly reuses the complete Run-2/Run-3
CR mapping with nominal weights, raw counts, and separate native
nonprompt/charge-flip postprocessing.

run3_full is the canonical complete Run-3 SR source-production profile.
run2_sr_ext and run3_sr_ext use a complete eight-variable SR plan. --sr-plan
supplies a JSON plan for a fresh campaign. --selected-fragment filters one
ordinary unit without reducing the persisted plan. For a quiescent campaign,
--resume --activate-pending-plan JSON --expected-plan-revision ID atomically
repartitions only untouched pending coverage in the same output directory.
An occupied schema-6 SR campaign needs the explicit sr_reco_manifest.py
upgrade-runtime-policy transition before pending units can change chunksize.
EOF
}

production_profile=""
profile_dry_run=false
profile_output_dir=""
profile_campaign_tag=""
profile_env_file=""
profile_resume=false
profile_reviewed_topeft_manager_head=""
profile_reviewed_topcoffea_manager_head=""
profile_chunksize=""
profile_wq_staging_base=""
profile_wq_quota_root=""
profile_wq_warning_bytes=""
profile_wq_min_free_bytes=""
profile_wq_poll_seconds=""
profile_wq_treereduction=""
profile_wq_processing_cores=""
profile_wq_accumulating_memory_mb=""
profile_wq_accumulating_cores=""
profile_selected_fragment=""
profile_sr_plan=""
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
    --chunksize|--wq-staging-base|--wq-quota-root|--wq-free-warning-bytes|--wq-min-free-bytes|--wq-poll-seconds|--wq-treereduction|--wq-processing-cores|--wq-accumulating-memory-mb|--wq-accumulating-cores)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 1
      fi
      case "$1" in
        --chunksize) profile_chunksize="$2" ;;
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
    --reviewed-topeft-manager-head)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: --reviewed-topeft-manager-head requires a value." >&2
        exit 1
      fi
      profile_reviewed_topeft_manager_head="$2"
      shift 2
      ;;
    --reviewed-topcoffea-manager-head)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: --reviewed-topcoffea-manager-head requires a value." >&2
        exit 1
      fi
      profile_reviewed_topcoffea_manager_head="$2"
      shift 2
      ;;
    --resume)
      profile_resume=true
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
    --sr-plan|--activate-pending-plan|--expected-plan-revision)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 1
      fi
      case "$1" in
        --sr-plan) profile_sr_plan="$2" ;;
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

case "${production_profile}" in
  run2_full|run3_full|run2_sr_ext|run3_sr_ext|run2_full_CR|run3_full_CR|t0_sr_statonly|t0_cr_statonly) ;;
  *)
    echo "ERROR: unsupported production profile '${production_profile}'." >&2
    exit 1
    ;;
esac

is_t0_statonly_profile() {
  [[ "${production_profile}" == "t0_sr_statonly" \
    || "${production_profile}" == "t0_cr_statonly" ]]
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../.." && pwd -P)
cd -- "${script_dir}"

validation_backend="${SRPLOT009_VALIDATION_BACKEND:-}"
validation_root="${SRPLOT009_VALIDATION_ROOT:-}"
validation_scenario="${SRPLOT009_VALIDATION_SCENARIO:-success}"
native_wq_log_dir=""
if [[ -n "${validation_backend}" ]]; then
  diagnostics_root=$(cd -- "${repository_root}/../reports/diagnostics" && pwd -P)
  case "${validation_root}" in /tmp/*|"${diagnostics_root}"/*) ;; *) echo "ERROR: SRPLOT009_VALIDATION_ROOT must be in an isolated validation namespace." >&2; exit 1 ;; esac
  case "${validation_backend}" in /tmp/*|"${diagnostics_root}"/*) ;; *) echo "ERROR: SRPLOT009_VALIDATION_BACKEND must be in an isolated validation namespace." >&2; exit 1 ;; esac
  [[ -x "${validation_backend}" ]] || { echo "ERROR: validation backend is not executable." >&2; exit 1; }
fi

###############################################################################
# Global configuration
###############################################################################

output_dir=""
chunk_size="100000"
if [[ "${production_profile}" == run?_sr_ext ]]; then
  case "${profile_chunksize}" in
    "") ;;
    100000|75000|50000|25000) chunk_size="${profile_chunksize}" ;;
    *) echo "ERROR: SR extension --chunksize must be 100000, 75000, 50000 or 25000." >&2; exit 1 ;;
  esac
elif [[ -n "${profile_chunksize}" ]]; then
  echo "ERROR: --chunksize is reserved for SR extension profiles." >&2
  exit 1
fi
if [[ -n "${profile_wq_staging_base}" ]]; then
  for guard_value in "${profile_wq_quota_root}" "${profile_wq_warning_bytes}" \
    "${profile_wq_min_free_bytes}" "${profile_wq_poll_seconds}"; do
    [[ -n "${guard_value}" ]] || { echo "ERROR: guarded staging requires quota, free-space warning/minimum and poll values." >&2; exit 1; }
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
if [[ -n "${profile_selected_fragment}" && "${production_profile}" != run?_sr_ext ]]; then
  echo "ERROR: --selected-fragment is supported only for SR extension profiles." >&2
  exit 1
fi
if [[ -n "${profile_sr_plan}${profile_activate_pending_plan}${profile_expected_plan_revision}" \
  && "${production_profile}" != run?_sr_ext ]]; then
  echo "ERROR: SR plan options require an SR extension profile." >&2
  exit 1
fi
if [[ -n "${profile_sr_plan}" && "${profile_resume}" == "true" ]]; then
  echo "ERROR: --sr-plan is for a fresh campaign; use explicit pending-plan activation." >&2
  exit 1
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
  && -z "${SRPLOT009_VALIDATION_BACKEND:-}" \
  && -z "${profile_wq_staging_base}" ]]; then
  echo "ERROR: Work Queue production requires guarded manager staging; implicit /tmp staging is forbidden." >&2
  exit 1
fi
if [[ "${profile_dry_run}" == false && -z "${SRPLOT009_VALIDATION_BACKEND:-}" \
  && -n "${profile_wq_staging_base}" ]]; then
  if [[ ! -d "${profile_wq_staging_base}" || "${profile_wq_staging_base}" != /* ]]; then
    echo "ERROR: staging base must be an existing absolute directory." >&2
    exit 1
  fi
  staging_real_path=$(realpath -e -- "${profile_wq_staging_base}") || exit 1
  if [[ "${staging_real_path}" == /tmp || "${staging_real_path}" == /tmp/* \
    || "${staging_real_path}" == /var/tmp || "${staging_real_path}" == /var/tmp/* \
    || "$(stat -c %d -- "${staging_real_path}")" == "$(stat -c %d -- /)" ]]; then
    echo "ERROR: /tmp or root-backed manager staging is forbidden for Work Queue production." >&2
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
production_environment_snapshot=false
production_topcoffea_git_commit=""
production_topcoffea_source_fingerprint=""
production_sumw2_options_path=""
production_sumw2_temporary_options=""
production_state_path=""
production_git_commit=""
production_postprocessor_topeft_git_commit=""
production_postprocessor_topcoffea_git_commit=""

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

if [[ "${production_profile}" == "run3_full" || "${production_profile}" == "run3_sr_ext" ]] \
  && { [[ "${output_dir}" == "/groups/klannon/apiccine/preappr_v9_260729" ]] \
    || [[ "${output_dir}" == "/groups/klannon/apiccine/rebin_fine_260818_v3" ]] \
    || [[ "${campaign_tag}" == "ANv9" ]] \
    || [[ "${campaign_tag}" == "rebin-fine-260818-v3" ]]; }; then
  echo "ERROR: ${production_profile} must use a fresh namespace, not a historical baseline or v3 campaign." >&2
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

if [[ -n "${profile_activate_pending_plan}" ]]; then
  if [[ ! -f "${output_dir}/.${production_profile}_campaign_state.json" ]]; then
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
  python "$(dirname -- "${BASH_SOURCE[0]}")/sr_reco_manifest.py" activate \
    "${output_dir}/.${production_profile}_campaign_state.json" \
    "${profile_activate_pending_plan}" "${profile_expected_plan_revision}"
  exit 0
fi

cr_pkl_base_tag="${campaign_tag}"
sr_pkl_base_tag="${campaign_tag}"

# Select the one region owned by the public component profile.
case "${production_profile}" in
  run2_full|run2_sr_ext)
    run_cr=false
    run_sr=true
    production_region="SR"
    production_np_mode="separate"
    production_component="run2"
    ;;
  run3_full|run3_sr_ext)
    run_cr=false
    run_sr=true
    production_region="SR"
    production_np_mode="separate"
    production_component="run3"
    ;;
  t0_sr_statonly)
    run_cr=false
    run_sr=true
    production_region="SR"
    production_np_mode="separate"
    production_component="t0_sr_statonly"
    ;;
  t0_cr_statonly)
    run_cr=true
    run_sr=false
    production_region="CR"
    production_np_mode="separate"
    production_component="t0_cr_statonly"
    ;;
  run2_full_CR)
    run_cr=true
    run_sr=false
    production_region="CR"
    production_np_mode="inline"
    production_component="run2"
    ;;
  run3_full_CR)
    run_cr=true
    run_sr=false
    production_region="CR"
    production_np_mode="inline"
    production_component="run3"
    ;;
esac

# Resolve and print commands without launching production.
dry_run="${profile_dry_run}"

# Shared CR/SR production switches.
do_systs=true
do_np=true
if is_t0_statonly_profile; then
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

# The complete Run-3 plan keeps the two off-Z charge families in separate
# source blocks so failures and output identities remain unambiguous.
run3_full_category_sets=(
  "2l 2lss_1tau 2los_1tau 4l"
  "3l_m_offZ"
  "3l_p_offZ"
  "3l_onZ_tau"
  "3l_fwd"
)

run3_full_category_var_set_names=(
  "sr_with_ptz_wtau_var_sets"
  "sr_offz_var_sets"
  "sr_offz_var_sets"
  "sr_onz_tau_var_sets"
  "sr_fwd_var_sets"
)

sr_year_sets=()
sr_year_category_set_names=()
sr_year_category_var_set_names=()
if [[ "${production_profile}" == "run2_full" ]]; then
  sr_year_sets=("run2")
  sr_year_category_set_names=("run3_full_category_sets")
  sr_year_category_var_set_names=("run3_full_category_var_set_names")
elif [[ "${production_profile}" == "run3_full" ]]; then
  sr_year_sets=(
    "2022 2022EE 2023 2023BPix"
  )
  sr_year_category_set_names=(
    "run3_full_category_sets"
  )
  sr_year_category_var_set_names=("run3_full_category_var_set_names")
elif [[ "${production_profile}" == "t0_sr_statonly" ]]; then
  sr_year_sets=(
    "2016APV 2016 2017 2018"
    "2022 2022EE 2023 2023BPix"
  )
  sr_year_category_set_names=(
    "run3_full_category_sets"
    "run3_full_category_sets"
  )
  sr_year_category_var_set_names=(
    "run3_full_category_var_set_names"
    "run3_full_category_var_set_names"
  )
fi

case "${production_profile}" in
  run2_full_CR) cr_year_sets=("2016APV 2016 2017 2018") ;;
  run3_full_CR) cr_year_sets=("2022 2022EE" "2023 2023BPix") ;;
  t0_cr_statonly) cr_year_sets=("2016APV 2016 2017 2018" "2022 2022EE" "2023 2023BPix") ;;
esac

production_state_filename=".${production_profile}_campaign_state.json"
production_block_ids=()
production_plan_year_exprs=()
production_plan_category_sets=()
production_plan_var_sets=()

case "${production_profile}" in
  run2_sr_ext|run3_sr_ext) ;; # Read the full JSON plan after state/environment preflight.
  run2_full|run3_full)
    profile_block_suffixes=(a b c d e)
    if [[ "${production_profile}" == "run2_full" ]]; then
      profile_year_expr="run2"
      profile_block_prefix="run2_full"
    else
      profile_year_expr="2022 2022EE 2023 2023BPix"
      profile_block_prefix="run3_full"
    fi
    for profile_index in "${!run3_full_category_sets[@]}"; do
      profile_var_set_name="${run3_full_category_var_set_names[profile_index]}"
      declare -n profile_var_set_ref="${profile_var_set_name}"
      production_block_ids+=("${profile_block_prefix}_${profile_block_suffixes[profile_index]}")
      production_plan_year_exprs+=("${profile_year_expr}")
      production_plan_category_sets+=("${run3_full_category_sets[profile_index]}")
      production_plan_var_sets+=("${profile_var_set_ref[0]}")
      unset -n profile_var_set_ref
    done
    ;;
  t0_sr_statonly)
    profile_block_suffixes=(a b c d e)
    profile_era_labels=(run2 run3)
    for profile_era_index in "${!sr_year_sets[@]}"; do
      profile_year_expr="${sr_year_sets[profile_era_index]}"
      profile_era_label="${profile_era_labels[profile_era_index]}"
      for profile_index in "${!run3_full_category_sets[@]}"; do
        profile_var_set_name="${run3_full_category_var_set_names[profile_index]}"
        declare -n profile_var_set_ref="${profile_var_set_name}"
        production_block_ids+=("t0_sr_statonly_${profile_era_label}_${profile_block_suffixes[profile_index]}")
        production_plan_year_exprs+=("${profile_year_expr}")
        production_plan_category_sets+=("${run3_full_category_sets[profile_index]}")
        production_plan_var_sets+=("${profile_var_set_ref[0]}")
        unset -n profile_var_set_ref
      done
    done
    ;;
  run2_full_CR|run3_full_CR|t0_cr_statonly)
    if [[ "${production_profile}" == "run2_full_CR" ]]; then
      profile_cr_year_sets=("2016APV 2016 2017 2018")
    elif [[ "${production_profile}" == "run3_full_CR" ]]; then
      profile_cr_year_sets=("2022 2022EE" "2023 2023BPix")
    else
      profile_cr_year_sets=(
        "2016APV 2016 2017 2018"
        "2022 2022EE"
        "2023 2023BPix"
      )
    fi
    profile_block_index=0
    for profile_year_expr in "${profile_cr_year_sets[@]}"; do
      for profile_category_index in "${!cr_category_sets[@]}"; do
        if (( profile_category_index == 0 )); then
          profile_var_set_name=cr_non_tau_var_sets
        else
          profile_var_set_name=cr_tau_var_sets
        fi
        declare -n profile_var_set_ref="${profile_var_set_name}"
        for profile_var_set in "${profile_var_set_ref[@]}"; do
          profile_block_index=$((profile_block_index + 1))
          production_block_ids+=("${production_profile}_block${profile_block_index}")
          production_plan_year_exprs+=("${profile_year_expr}")
          production_plan_category_sets+=("${cr_category_sets[profile_category_index]}")
          production_plan_var_sets+=("${profile_var_set}")
        done
        unset -n profile_var_set_ref
      done
    done
    ;;
esac

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
    if [[ "${production_profile}" == "run2_full" \
      || "${production_profile}" == *_CR \
      || "${production_profile}" == "t0_cr_statonly" ]]; then
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
  python - "$@" <<'PY'
import datetime
import json
import os
import re
import sys
from pathlib import Path
from sr_reco_manifest import artifact_receipt, capture_receipt, check_state, decorate_initial_state, source_runtime


VALID_STATUSES = {
    "planned",
    "source_running",
    "source_ready",
    "source_failed",
    "nonprompt_running",
    "nonprompt_failed",
    "success",
}
VALID_SOURCE_STATUSES = {"planned", "running", "ready", "failed"}
VALID_NONPROMPT_STATUSES = {"blocked", "planned", "running", "failed", "success"}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def atomic_write(path, payload):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read campaign state {path}: {exc}")


def read_plan(path, production_profile):
    blocks = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"cannot read generated {production_profile} plan {path}: {exc}")
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 8:
            fail(f"invalid generated {production_profile} plan row: {line!r}")
        block_id, years, categories, histograms, output_tag, output_name, nominal, nonprompt = fields
        block = {
            "id": block_id,
            "years": years.split(),
            "category_groups": categories.split(),
            "histograms": histograms.split(),
            "output_tag": output_tag,
            "output_name": output_name,
            "expected_outputs": [nominal, nonprompt],
        }
        if production_profile in {"run2_sr_ext", "run3_sr_ext"}:
            revision_match = re.search(r"_(r[0-9]+_[0-9a-f]{12})_", output_tag)
            if revision_match is None:
                fail(f"SR plan row lacks a revision identity: {block_id}")
            block["plan_revision"] = revision_match.group(1)
        block["expected_nominal_path"] = nominal
        block["expected_np_path"] = nonprompt
        blocks.append(block)
    if not blocks or len({block["id"] for block in blocks}) != len(blocks):
        fail(f"generated {production_profile} plan does not contain unique blocks")
    return blocks


def desired_state(arguments):
    plan_path = Path(arguments[0])
    production_profile, schema_version, tag, output_dir, commit, env_file, env_sha256, env_fingerprint, environment_mode, topcoffea_commit, topcoffea_source, ttgamma, do_systs, do_np, region, nonprompt_mode = arguments[1:]
    return {
        "schema_version": int(schema_version),
        "production_profile": production_profile,
        "campaign_tag": tag,
        "output_dir": output_dir,
        "topeft_git_commit": commit,
        "env_file": env_file,
        "env_file_sha256": env_sha256,
        "environment_fingerprint": env_fingerprint,
        "environment_mode": environment_mode,
        "topcoffea_git_commit": topcoffea_commit,
        "topcoffea_relevant_source_fingerprint": topcoffea_source,
        "ttgamma_sample_role_policy": ttgamma,
        "do_systs": do_systs == "true",
        "do_np": do_np == "true",
        "region": region,
        "nonprompt_mode": nonprompt_mode,
        "blocks": read_plan(plan_path, production_profile),
    }


def validate_state(state, desired, allow_manager_provenance_delta=False):
    for key in (
        "schema_version",
        "production_profile",
        "campaign_tag",
        "output_dir",
        "topeft_git_commit",
        "env_file",
        "env_file_sha256",
        "environment_fingerprint",
        "environment_mode",
        "topcoffea_git_commit",
        "topcoffea_relevant_source_fingerprint",
        "ttgamma_sample_role_policy",
        "do_systs",
        "do_np",
        "region",
        "nonprompt_mode",
    ):
        if key == "topeft_git_commit" and allow_manager_provenance_delta:
            recorded_source_commit = state.get(key)
            if not isinstance(recorded_source_commit, str) or re.fullmatch(
                r"[0-9a-f]{40}", recorded_source_commit
            ) is None:
                fail("campaign state does not contain a valid historical topeft source commit")
            continue
        if state.get(key) != desired[key]:
            fail(f"campaign state mismatch for {key}: recorded={state.get(key)!r} requested={desired[key]!r}")
    recorded_blocks = state.get("blocks")
    if not isinstance(recorded_blocks, list) or len(recorded_blocks) != len(desired["blocks"]):
        fail("campaign state block count does not match the live profile plan")
    for recorded, expected in zip(recorded_blocks, desired["blocks"]):
        for key in expected:
            if recorded.get(key) != expected[key]:
                fail(f"campaign state mismatch for block {expected['id']} field {key}")
        if recorded.get("status") not in VALID_STATUSES:
            fail(f"campaign state has invalid status for block {expected['id']}: {recorded.get('status')!r}")
        if recorded.get("source_status") not in VALID_SOURCE_STATUSES:
            fail(f"campaign state has invalid source status for block {expected['id']}")
        if recorded.get("nonprompt_status") not in VALID_NONPROMPT_STATUSES:
            fail(f"campaign state has invalid nonprompt status for block {expected['id']}")
        if not isinstance(recorded.get("transitions"), list):
            fail(f"campaign state lacks transition history for block {expected['id']}")
        expected_stage_statuses = {
            "planned": ("planned", "blocked"),
            "source_running": ("running", "blocked"),
            "source_ready": ("ready", "planned"),
            "source_failed": ("failed", "blocked"),
            "nonprompt_running": ("ready", "running"),
            "nonprompt_failed": ("ready", "failed"),
            "success": ("ready", "success"),
        }[recorded["status"]]
        if (recorded["source_status"], recorded["nonprompt_status"]) != expected_stage_statuses:
            fail(f"campaign state has contradictory stage statuses for block {expected['id']}")
    validate_full_receipts(state)


def validate_full_receipts(state):
    if state.get("production_profile") not in {"run2_full", "run3_full"} or state.get("schema_version") != 6:
        return
    if state.get("nonprompt_mode") != "separate":
        fail("new full-profile state requires separate nonprompt")
    for block in state["blocks"]:
        if block["status"] not in {"source_ready", "nonprompt_running", "nonprompt_failed", "success"}:
            continue
        if block.get("source_status") != "ready" or block.get("source_exit_code") != 0:
            fail("full-profile source receipt lacks a successful source child")
        artifact_receipt(block["expected_nominal_path"], stored_receipt=block.get("source_receipt"), verify_payload_sha256=False)
        if block["status"] == "success":
            if block.get("nonprompt_status") != "success" or block.get("nonprompt_exit_code") != 0:
                fail("full-profile nonprompt completion is inconsistent")
            artifact_receipt(block["expected_np_path"], stored_receipt=block.get("nonprompt_receipt"), verify_payload_sha256=False)
            sidecar = json.loads(Path(f"{block['expected_np_path']}.metadata.json").read_text(encoding="utf-8"))
            if not sidecar.get("synthetic") and not any(
                isinstance(item, dict) and item.get("pkl_sha256") == block["source_receipt"]["sha256"]
                for item in sidecar["lineage"]["inputs"]
            ):
                fail("full-profile nonprompt lineage does not bind source")
        elif block.get("nonprompt_receipt") is not None:
            fail("full-profile nonprompt receipt precedes success")


mode = sys.argv[1]
state_path = Path(sys.argv[2])

if mode in {"initialize", "validate"}:
    desired = desired_state(sys.argv[3:20])
    readonly = len(sys.argv) > 20 and sys.argv[20] == "true"
    allow_manager_provenance_delta = len(sys.argv) > 21 and sys.argv[21] == "true"
    if mode == "initialize":
        if state_path.exists():
            fail(f"refusing to overwrite existing {desired['production_profile']} campaign state: {state_path}")
        planned_blocks = desired["blocks"]
        state = {**desired, "blocks": []}
        timestamp = now_utc()
        state["created_at_utc"] = timestamp
        state["updated_at_utc"] = timestamp
        for block in planned_blocks:
            initialized = {
                **block,
                "status": "planned",
                "exit_code": None,
                "source_status": "planned",
                "source_exit_code": None,
                "source_signal": None,
                "source_command_argv": None,
                "source_started_at_utc": None,
                "source_ended_at_utc": None,
                "source_duration_seconds": None,
                "nonprompt_status": "blocked",
                "nonprompt_exit_code": None,
                "nonprompt_signal": None,
                "nonprompt_command_argv": None,
                "nonprompt_started_at_utc": None,
                "nonprompt_ended_at_utc": None,
                "nonprompt_duration_seconds": None,
                "native_work_queue_logs": {},
                "last_transition_utc": timestamp,
                "last_transition_detail": "campaign_initialized",
                "transitions": [
                    {
                        "timestamp_utc": timestamp,
                        "stage": "campaign",
                        "status": "planned",
                        "exit_code": None,
                        "detail": "campaign_initialized",
                    }
                ],
            }
            state["blocks"].append(initialized)
        if desired["production_profile"] in {"run2_sr_ext", "run3_sr_ext"}:
            state = decorate_initial_state(state, sys.argv[20])
        atomic_write(state_path, state)
    else:
        state = load(state_path)
        validate_state(
            state,
            desired,
            allow_manager_provenance_delta=allow_manager_provenance_delta,
        )
        for block in state["blocks"]:
            if block["source_status"] == "running":
                fail(f"block {block['id']} has an ambiguous interrupted source stage; state was not rewritten")
            elif block["nonprompt_status"] == "running":
                fail(f"block {block['id']} has an ambiguous interrupted nonprompt stage; state was not rewritten")
    raise SystemExit(0)

state = load(state_path)
if mode in {"env_file", "environment_mode"}:
    expected_profile, expected_tag, expected_output_dir = sys.argv[3:6]
    for key, expected in (
        ("production_profile", expected_profile),
        ("campaign_tag", expected_tag),
        ("output_dir", expected_output_dir),
    ):
        if state.get(key) != expected:
            fail(
                f"campaign state mismatch for {key}: "
                f"recorded={state.get(key)!r} requested={expected!r}"
            )
    if mode == "env_file":
        env_file = state.get("env_file")
        if not isinstance(env_file, str) or not env_file.startswith("/"):
            fail("campaign state does not contain an absolute frozen env_file")
        print(env_file)
    else:
        environment_mode = state.get("environment_mode")
        if environment_mode not in {
            "current_resolved",
            "explicit_snapshot",
        }:
            fail("campaign state does not contain a valid frozen environment_mode")
        print(environment_mode)
    raise SystemExit(0)

if mode == "historical_source_commit":
    expected_profile, expected_tag, expected_output_dir = sys.argv[3:6]
    for key, expected in (
        ("production_profile", expected_profile),
        ("campaign_tag", expected_tag),
        ("output_dir", expected_output_dir),
    ):
        if state.get(key) != expected:
            fail(
                f"campaign state mismatch for {key}: "
                f"recorded={state.get(key)!r} requested={expected!r}"
            )
    source_commit = state.get("topeft_git_commit")
    if (not isinstance(source_commit, str) or not source_commit or
            (state.get("production_profile") not in {"run2_sr_ext", "run3_sr_ext"} and
             re.fullmatch(r"[0-9a-f]{40}", source_commit) is None)):
        fail("campaign state does not contain a valid historical topeft source commit")
    print(source_commit)
    raise SystemExit(0)

if mode == "historical_full_execution_policy":
    expected_profile = sys.argv[3]
    if state.get("production_profile") != expected_profile or state.get("schema_version") not in {5, 6}:
        fail("unsupported historical full-profile state")
    if state["schema_version"] == 6 and state.get("nonprompt_mode") != "separate":
        fail("unsupported full-profile stage policy")
    if state.get("nonprompt_mode") not in {"inline", "separate"}:
        fail("invalid historical full-profile execution mode")
    print(f"{state['schema_version']}\t{state['nonprompt_mode']}")
    raise SystemExit(0)

if mode == "legacy_context_summary":
    block_id = sys.argv[3]
    matching = [block for block in state.get("blocks", []) if block.get("id") == block_id]
    if len(matching) != 1:
        fail(f"campaign state must contain exactly one block {block_id!r}")
    block = matching[0]
    if block.get("source_status") != "ready":
        fail(f"campaign block {block_id!r} does not contain a reusable ready source")
    source_command = block.get("source_command_argv")
    if not isinstance(source_command, list) or any(
        not isinstance(value, str) for value in source_command
    ):
        fail(f"campaign block {block_id!r} lacks source command provenance")

    def option_values(option):
        positions = [index for index, value in enumerate(source_command) if value == option]
        if len(positions) != 1:
            fail(f"source command must contain exactly one {option}")
        values = []
        for value in source_command[positions[0] + 1:]:
            if value.startswith("-"):
                break
            values.append(value)
        if not values:
            fail(f"source command has no values for {option}")
        return values

    selected_modes = [
        mode_name
        for flag, mode_name in (
            ("--all-analysis", "all"),
            ("--offZ-3l-split", "offz"),
            ("--tau-h-analysis", "tau"),
            ("--fwd-analysis", "fwd"),
        )
        if flag in source_command
    ]
    if len(selected_modes) > 1:
        fail("source command contains conflicting analysis modes")
    source_path = Path(block.get("expected_nominal_path", ""))
    sidecar_path = Path(str(source_path) + ".metadata.json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source_identity = sidecar["artifact"]["pkl_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        fail(f"cannot resolve source identity from {sidecar_path}: {exc}")
    summary = {
        "analysis_mode": selected_modes[0] if selected_modes else "default",
        "campaign_block": block_id,
        "category_groups": option_values("--category-groups"),
        "histogram_families": option_values("--hist-vars"),
        "producer_topeft_commit": state.get("topeft_git_commit"),
        "region": state.get("region"),
        "source_pkl_sha256": source_identity,
        "years": option_values("-y"),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0)

if mode == "status":
    block_id = sys.argv[3]
    for block in state.get("blocks", []):
        if block.get("id") == block_id:
            print(
                "\t".join(
                    (
                        str(block.get("status", "")),
                        str(block.get("source_status", "")),
                        str(block.get("nonprompt_status", "")),
                    )
                )
            )
            raise SystemExit(0)
    fail(f"campaign state does not contain block {block_id}")

if mode == "mark":
    block_id, stage, stage_status, exit_code, detail = sys.argv[3:8]
    extra = sys.argv[8:]
    if stage == "source" and stage_status == "running" and extra[:1] == ["--"]:
        extra = extra[1:]
    postprocessor_provenance = None
    if stage == "source" and stage_status not in VALID_SOURCE_STATUSES:
        fail(f"invalid requested source status {stage_status!r}")
    if stage == "nonprompt" and stage_status not in VALID_NONPROMPT_STATUSES:
        fail(f"invalid requested nonprompt status {stage_status!r}")
    if stage not in {"source", "nonprompt"}:
        fail(f"invalid requested stage {stage!r}")
    if (
        stage == "nonprompt"
        and stage_status == "running"
        and extra[:1] == ["--postprocessor-provenance"]
    ):
        if len(extra) < 5 or extra[3] != "--":
            fail("invalid postprocessor provenance arguments")
        postprocessor_topeft_commit, postprocessor_topcoffea_commit = extra[1:3]
        for label, value in (
            ("topeft", postprocessor_topeft_commit),
            ("topcoffea", postprocessor_topcoffea_commit),
        ):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                fail(f"invalid {label} postprocessor commit")
        postprocessor_provenance = {
            "topeft_git_commit": postprocessor_topeft_commit,
            "topcoffea_git_commit": postprocessor_topcoffea_commit,
        }
        extra = extra[4:]
    elif (
        stage == "nonprompt"
        and stage_status == "running"
        and state.get("production_profile") in {"t0_sr_statonly", "t0_cr_statonly"}
    ):
        fail("T0 stat-only nonprompt recovery requires postprocessor provenance")
    for block in state.get("blocks", []):
        if block.get("id") == block_id:
            if state.get("schema_version") in {7, 8} and stage == "source" and stage_status == "running":
                runtime = source_runtime(state, block, extra)
                block.setdefault("source_runtime_attempts", []).append(runtime)
                block["source_runtime"] = runtime
            parsed_exit_code = None if exit_code == "none" else int(exit_code)
            timestamp = now_utc()
            if stage == "source":
                block["source_status"] = stage_status
                block["source_exit_code"] = parsed_exit_code
                block["status"] = {
                    "planned": "planned",
                    "running": "source_running",
                    "ready": "source_ready",
                    "failed": "source_failed",
                }[stage_status]
                if stage_status == "ready":
                    block["nonprompt_status"] = "planned"
                    block["nonprompt_exit_code"] = None
                else:
                    block["nonprompt_status"] = "blocked"
                    block["nonprompt_exit_code"] = None
            else:
                block["nonprompt_status"] = stage_status
                block["nonprompt_exit_code"] = parsed_exit_code
                block["status"] = {
                    "blocked": "source_failed",
                    "planned": "source_ready",
                    "running": "nonprompt_running",
                    "failed": "nonprompt_failed",
                    "success": "success",
                }[stage_status]
            block["exit_code"] = parsed_exit_code
            block[f"{stage}_signal"] = (
                parsed_exit_code - 128
                if parsed_exit_code is not None and parsed_exit_code > 128
                else None
            )
            if stage_status == "running":
                block[f"{stage}_command_argv"] = extra
                block[f"{stage}_started_at_utc"] = timestamp
                block[f"{stage}_ended_at_utc"] = None
                block[f"{stage}_duration_seconds"] = None
            else:
                block[f"{stage}_ended_at_utc"] = timestamp
                if extra:
                    block[f"{stage}_duration_seconds"] = int(extra[0])
            output_readback = []
            for output in block.get("expected_outputs", []):
                output_path = Path(output)
                exists = output_path.exists()
                regular = output_path.is_file()
                size = output_path.stat().st_size if regular else None
                output_readback.append(
                    {"path": output, "exists": exists, "regular_file": regular, "size_bytes": size, "nonempty": bool(regular and size)}
                )
            block["expected_output_readback"] = output_readback
            if state.get("production_profile") in {"run2_sr_ext", "run3_sr_ext"}:
                if stage == "source" and stage_status == "ready":
                    capture_receipt(state, block, "source")
                if stage == "nonprompt" and stage_status == "success":
                    capture_receipt(state, block, "nonprompt")
            elif state.get("production_profile") in {"run2_full", "run3_full"} and state.get("schema_version") == 6:
                if stage == "source" and stage_status == "ready":
                    block["source_receipt"] = artifact_receipt(block["expected_nominal_path"])
                if stage == "nonprompt" and stage_status == "success":
                    block["nonprompt_receipt"] = artifact_receipt(block["expected_np_path"])
            block["last_transition_utc"] = timestamp
            block["last_transition_detail"] = detail
            if postprocessor_provenance is not None:
                block["nonprompt_postprocessor_provenance"] = postprocessor_provenance
            transition = {
                "timestamp_utc": block["last_transition_utc"],
                "stage": stage,
                "status": stage_status,
                "exit_code": parsed_exit_code,
                "signal": block[f"{stage}_signal"],
                "duration_seconds": block.get(f"{stage}_duration_seconds"),
                "detail": detail,
            }
            if stage == "nonprompt" and block.get("nonprompt_postprocessor_provenance"):
                transition["postprocessor_provenance"] = block[
                    "nonprompt_postprocessor_provenance"
                ]
            block.setdefault("transitions", []).append(transition)
            state["updated_at_utc"] = now_utc()
            if state.get("production_profile") in {"run2_sr_ext", "run3_sr_ext"}:
                check_state(state)
            else:
                validate_full_receipts(state)
            atomic_write(state_path, state)
            raise SystemExit(0)
    fail(f"campaign state does not contain block {block_id}")

if mode == "archive":
    block_id, stage, archive_dir = sys.argv[3:6]
    log_names = {
        "wq_debug_log": "debug.log",
        "wq_transactions_log": "tr.log",
        "wq_stats_log": "stats.log",
        "wq_tasks_accum_log": "tasks.log",
    }
    for block in state.get("blocks", []):
        if block.get("id") != block_id:
            continue
        archive = Path(archive_dir)
        metadata = {}
        for key, name in log_names.items():
            path = archive / name
            exists = path.exists()
            regular = path.is_file()
            stat = path.stat() if regular else None
            metadata[key] = {
                "path": str(path),
                "exists": exists,
                "regular_file": regular,
                "nonempty": bool(regular and stat.st_size),
                "size_bytes": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        block.setdefault("native_work_queue_logs", {})[stage] = metadata
        block["last_transition_utc"] = now_utc()
        block["last_transition_detail"] = f"{stage}_native_work_queue_logs_archived"
        state["updated_at_utc"] = now_utc()
        atomic_write(state_path, state)
        raise SystemExit(0)
    fail(f"campaign state does not contain block {block_id}")

def campaign_classification(payload):
    if payload.get("production_profile") in {"run2_sr_ext", "run3_sr_ext"}:
        check_state(payload)
    statuses = [block.get("status") for block in payload.get("blocks", [])]
    if any(status in {"source_running", "nonprompt_running"} for status in statuses):
        return "blocked_ambiguous"
    if any(status in {"planned", "source_ready"} for status in statuses):
        return "blocked_incomplete"
    if any(status in {"source_failed", "nonprompt_failed"} for status in statuses):
        return "complete_with_known_failures"
    if statuses and all(status == "success" for status in statuses):
        return "success"
    return "blocked_state_contradiction"

if mode == "classification":
    print(campaign_classification(state))
    raise SystemExit(0)

if mode == "finalize":
    classification = campaign_classification(state)
    selected_fragment = sys.argv[3] if len(sys.argv) > 3 else ""
    selected_complete = bool(selected_fragment) and any(
        block.get("id") == selected_fragment and block.get("status") == "success"
        for block in state.get("blocks", [])
    )
    final_exit_code = 0 if classification == "success" or selected_complete else 1
    statuses = [block.get("status") for block in state.get("blocks", [])]
    state["campaign_status"] = classification
    state["configured_block_count"] = len(statuses)
    state["attempted_block_count"] = sum(status not in {"planned", "source_ready"} for status in statuses)
    state["successful_block_count"] = statuses.count("success")
    state["known_failed_block_count"] = sum(status in {"source_failed", "nonprompt_failed"} for status in statuses)
    state["not_attempted_block_count"] = sum(status in {"planned", "source_ready"} for status in statuses)
    state["final_process_exit_code"] = final_exit_code
    state["completed_at_utc"] = now_utc()
    state["updated_at_utc"] = state["completed_at_utc"]
    atomic_write(state_path, state)
    print(f"{classification}\t{final_exit_code}")
    raise SystemExit(0)

if mode == "report":
    tsv_path, markdown_path = map(Path, sys.argv[3:5])
    classification = campaign_classification(state)
    if state.get("campaign_status") != classification:
        fail("campaign state must be finalized before deriving reports")
    headers = [
        "profile", "campaign_tag", "era", "block_id", "plan_revision", "categories", "variables",
        "source_status", "source_exit", "nonprompt_status", "nonprompt_exit",
        "final_block_status", "expected_nominal_path", "expected_np_path",
    ]
    rows = []
    for block in state["blocks"]:
        rows.append([
            state["production_profile"], state["campaign_tag"], " ".join(block["years"]), block["id"],
            block.get("plan_revision", ""),
            " ".join(block["category_groups"]), " ".join(block["histograms"]),
            block["source_status"], block.get("source_exit_code"), block["nonprompt_status"],
            block.get("nonprompt_exit_code"), block["status"], block["expected_nominal_path"], block["expected_np_path"],
        ])
    def atomic_text(path, text):
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    def clean(value):
        return "" if value is None else str(value).replace("\t", " ").replace("\n", " ")
    tsv = "\t".join(headers) + "\n" + "".join("\t".join(clean(item) for item in row) + "\n" for row in rows)
    atomic_text(tsv_path, tsv)
    summary = [
        f"# Campaign summary: {state['production_profile']}", "",
        f"- campaign_tag: `{state['campaign_tag']}`",
        f"- source_commit: `{state['topeft_git_commit']}`",
        f"- environment_sha256: `{state['env_file_sha256']}`",
        f"- configured: {state['configured_block_count']}",
        f"- attempted: {state['attempted_block_count']}",
        f"- successful: {state['successful_block_count']}",
        f"- known_failed: {state['known_failed_block_count']}",
        f"- not_attempted: {state['not_attempted_block_count']}",
        f"- final_classification: `{classification}`",
        f"- final_process_exit_code: {state['final_process_exit_code']}", "",
        f"- active_plan_revision: {state.get('active_sr_reco_revision', 'not_applicable')}",
        "Per-block details are serialized in `campaign_summary.tsv`.", "",
    ]
    atomic_text(markdown_path, "\n".join(summary))
    raise SystemExit(0)

if mode == "failure_snapshot":
    block_id, stage, snapshot_path = sys.argv[3:6]
    for block in state.get("blocks", []):
        if block.get("id") != block_id:
            continue
        lines = ["field\tvalue", f"block_id\t{block_id}", f"stage\t{stage}", f"exit_code\t{block.get(stage + '_exit_code')}", f"signal\t{block.get(stage + '_signal')}", f"start_time_utc\t{block.get(stage + '_started_at_utc')}", f"end_time_utc\t{block.get(stage + '_ended_at_utc')}", f"duration_seconds\t{block.get(stage + '_duration_seconds')}", f"campaign_updated_at_utc\t{state.get('updated_at_utc')}"]
        for index, item in enumerate(block.get("expected_output_readback", []), 1):
            for key in ("path", "exists", "regular_file", "size_bytes", "nonempty"):
                lines.append(f"expected_output_{index}_{key}\t{item.get(key)}")
        for key, item in block.get("native_work_queue_logs", {}).get("source", {}).items():
            lines.append(f"{key}\t{item.get('path')}")
        path = Path(snapshot_path)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        raise SystemExit(0)
    fail(f"campaign state does not contain block {block_id}")

fail(f"unsupported production state operation {mode!r}")
PY
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
    production_environment_mode=$(production_state_tool environment_mode \
      "${production_state_path}" "${production_profile}" "${campaign_tag}" "${output_dir}")
    if [[ -n "${requested_env_file}" ]]; then
      if [[ "${requested_env_file}" != /* ]]; then
        echo "ERROR: ${production_profile} --env-file must be an absolute path: ${requested_env_file}" >&2
        exit 1
      fi
      canonical_requested_env_file=$(readlink -f -- "${requested_env_file}" || true)
      if [[ -z "${canonical_requested_env_file}" || "${canonical_requested_env_file}" != "${frozen_env_file}" ]]; then
        echo "ERROR: resume --env-file does not match the exact archive frozen in campaign state." >&2
        exit 1
      fi
    fi
    requested_env_file="${frozen_env_file}"
    if [[ "${production_environment_mode}" == "explicit_snapshot" ]]; then
      validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
      production_environment_snapshot=true
    elif [[ "${production_environment_mode}" == "current_resolved" ]]; then
      validation_args=(--validate-env-file --env-file "${requested_env_file}")
    else
      echo "ERROR: ${production_profile} resume state has an unsupported environment mode." >&2
      exit 1
    fi
  elif [[ -n "${requested_env_file}" ]]; then
    if [[ "${requested_env_file}" != /* ]]; then
      echo "ERROR: ${production_profile} --env-file must be an absolute path: ${requested_env_file}" >&2
      exit 1
    fi
    canonical_requested_env_file=$(readlink -f -- "${requested_env_file}" || true)
    production_environment_mode=explicit_snapshot
    production_environment_snapshot=true
    validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
  else
    production_environment_mode=current_resolved
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
    || [[ -z "${production_topcoffea_git_commit}" ]] \
    || [[ -z "${production_topcoffea_source_fingerprint}" ]] \
    || [[ "${validation_status}" != "valid" ]]; then
    echo "ERROR: run_analysis.py did not return a complete valid environment identity." >&2
    exit 1
  fi

  if [[ "${profile_resume}" == "true" && "${production_env_file}" != "${frozen_env_file}" ]]; then
    echo "ERROR: validated resume environment path differs from campaign state." >&2
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

production_outputs_present() {
  local block_id="$1"
  local plan_path="$2"
  local nominal_path
  local nonprompt_path

  IFS=$'\t' read -r _ _ _ _ _ _ nominal_path nonprompt_path < <(
    awk -F '\t' -v block_id="${block_id}" '$1 == block_id {print; exit}' "${plan_path}"
  )
  [[ -n "${nominal_path}" && -n "${nonprompt_path}" && -s "${nominal_path}" && -s "${nonprompt_path}" ]]
}

production_any_output_exists() {
  local block_id="$1"
  local plan_path="$2"
  local nominal_path
  local nonprompt_path

  IFS=$'\t' read -r _ _ _ _ _ _ nominal_path nonprompt_path < <(
    awk -F '\t' -v block_id="${block_id}" '$1 == block_id {print; exit}' "${plan_path}"
  )
  [[ -e "${nominal_path}" || -e "${nonprompt_path}" ]]
}

production_assert_live_plan() {
  local expected_count
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    return 0
  fi
  case "${production_profile}" in
    run2_full|run3_full) expected_count=5 ;;
    t0_sr_statonly) expected_count=10 ;;
    t0_cr_statonly) expected_count=18 ;;
    run2_full_CR) expected_count=6 ;;
    run3_full_CR) expected_count=12 ;;
  esac
  if [[ -n "${profile_selected_fragment}" ]]; then
    expected_count=1
  fi
  if (( ${#production_block_ids[@]} != expected_count )) \
    || (( ${#production_plan_year_exprs[@]} != expected_count )) \
    || (( ${#production_plan_category_sets[@]} != expected_count )) \
    || (( ${#production_plan_var_sets[@]} != expected_count )); then
    echo "ERROR: live ${production_profile} packing no longer matches its configured block contract." >&2
    exit 1
  fi
}

resolve_t0_postprocessor_provenance() {
  local topcoffea_repository_root="${repository_root}/../topcoffea"
  local topeft_postprocessor_status
  local topcoffea_postprocessor_status

  is_t0_statonly_profile || return 0

  if [[ ! -d "${topcoffea_repository_root}/.git" ]]; then
    echo "ERROR: T0 stat-only postprocessor provenance requires the sibling topcoffea repository." >&2
    exit 1
  fi
  production_postprocessor_topeft_git_commit=$(git -C "${repository_root}" rev-parse HEAD)
  production_postprocessor_topcoffea_git_commit=$(git -C "${topcoffea_repository_root}" rev-parse HEAD)

  topeft_postprocessor_status=$(git -C "${repository_root}" status --porcelain --untracked-files=all -- \
    analysis/topeft_run2/run_data_driven.py \
    topeft/modules/dataDrivenEstimation.py \
    topeft/modules/data_driven_products.py \
    topeft/modules/histogram_artifact.py \
    topeft/modules/nominal_schema.py \
    topeft/modules/sumw2_policy.py)
  topcoffea_postprocessor_status=$(git -C "${topcoffea_repository_root}" status --porcelain --untracked-files=all -- \
    topcoffea/modules/sparseHist.py \
    topcoffea/modules/utils.py)
  if [[ -n "${topeft_postprocessor_status}" || -n "${topcoffea_postprocessor_status}" ]]; then
    if [[ "${dry_run}" == "true" ]]; then
      echo "Dry-run note: postprocessor source surfaces are dirty; no production child or provenance record will be created."
    elif [[ -n "${SRPLOT009_VALIDATION_BACKEND:-}" ]]; then
      echo "Validation-backend note: postprocessor source surfaces are dirty; the synthetic backend creates no production artifact."
    else
      echo "ERROR: T0 stat-only postprocessor source surfaces are dirty; commit provenance would be ambiguous." >&2
      exit 1
    fi
  fi
}

validate_manager_repository_delta() {
  local manager_label="$1"
  local manager_repository="$2"
  local recorded_commit="$3"
  local current_commit="$4"
  local reviewed_commit="$5"
  shift 5
  local changed_sensitive_paths=""

  if [[ "${recorded_commit}" == "${current_commit}" ]]; then
    echo "Manager compatibility ${manager_label}: PASS_EXACT_HEAD ${current_commit}"
    return 0
  fi
  if [[ ! -d "${manager_repository}/.git" ]] \
    || ! git -C "${manager_repository}" cat-file -e "${recorded_commit}^{commit}" 2>/dev/null \
    || ! git -C "${manager_repository}" cat-file -e "${current_commit}^{commit}" 2>/dev/null; then
    echo "HUMAN_REVIEW_REQUIRED: ${manager_label} manager provenance delta cannot be classified from local commits; campaign validity was not changed." >&2
    return 1
  fi
  if ! changed_sensitive_paths=$(git -C "${manager_repository}" diff \
    --no-ext-diff --no-textconv --name-only --no-renames \
    "${recorded_commit}" "${current_commit}" -- "$@" 2>/dev/null); then
    echo "HUMAN_REVIEW_REQUIRED: ${manager_label} manager provenance delta could not be compared; campaign validity was not changed." >&2
    return 1
  fi
  if [[ -n "${changed_sensitive_paths}" ]]; then
    echo "Manager-sensitive ${manager_label} paths changed between recorded and current HEAD:" >&2
    while IFS= read -r changed_path; do
      printf '  %s\n' "${changed_path}" >&2
    done <<< "${changed_sensitive_paths}"
    if [[ -z "${reviewed_commit}" ]]; then
      echo "HUMAN_REVIEW_REQUIRED: ${manager_label} manager delta has no exact-current-HEAD review; campaign validity was not changed." >&2
      return 1
    fi
    if [[ ! "${reviewed_commit}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "HUMAN_REVIEW_REQUIRED: reviewed ${manager_label} manager HEAD is malformed; campaign validity was not changed." >&2
      return 1
    fi
    if [[ "${reviewed_commit}" != "${current_commit}" ]]; then
      echo "HUMAN_REVIEW_REQUIRED: reviewed ${manager_label} manager HEAD is stale; reviewed=${reviewed_commit} current=${current_commit}; campaign validity was not changed." >&2
      return 1
    fi
    echo "Manager compatibility ${manager_label}: PASS_WITH_REVIEWED_MANAGER_DELTA recorded=${recorded_commit} current=${current_commit} reviewed=${reviewed_commit}"
    return 0
  fi
  echo "Manager compatibility ${manager_label}: PASS_WITH_MANAGER_PROVENANCE_DELTA recorded=${recorded_commit} current=${current_commit}"
}

validate_resume_manager_compatibility() {
  local recorded_topeft_commit="$1"
  local topcoffea_repository_root="${repository_root}/../topcoffea"
  local current_topcoffea_commit=""

  if [[ "${production_profile}" == run?_sr_ext ]]; then
    echo "SR manager provenance: historical commit is retained as metadata; native input and receipt checks govern resume."
    return 0
  fi

  validate_manager_repository_delta \
    TOPEFT "${repository_root}" "${recorded_topeft_commit}" "${production_git_commit}" \
    "${profile_reviewed_topeft_manager_head}" \
    analysis/topeft_run2/run_cr.sh \
    analysis/topeft_run2/sr_reco_manifest.py \
    analysis/topeft_run2/fullR2_run.sh \
    analysis/topeft_run2/fullR3_run.sh \
    analysis/topeft_run2/run_analysis.py \
    analysis/topeft_run2/analysis_processor.py \
    input_samples/cfgs/mc_signal_samples_NDSkim.cfg \
    input_samples/cfgs/mc_background_samples_NDSkim.cfg \
    input_samples/cfgs/mc_background_samples_cr_NDSkim.cfg \
    input_samples/cfgs/data_samples_NDSkim.cfg \
    ':(glob)input_samples/cfgs/NDSkim_*_background_samples.cfg' \
    ':(glob)input_samples/cfgs/NDSkim_*_background_samples_cr.cfg' \
    ':(glob)input_samples/cfgs/NDSkim_*_data_samples.cfg' \
    ':(glob)input_samples/cfgs/NDSkim_*_mc_signal_samples.cfg' \
    ':(glob)input_samples/cfgs/NDSkim_*_mc_signal_samples_sr.cfg'

  if [[ ! -d "${topcoffea_repository_root}/.git" ]] \
    || ! current_topcoffea_commit=$(git -C "${topcoffea_repository_root}" rev-parse HEAD 2>/dev/null); then
    echo "HUMAN_REVIEW_REQUIRED: live TOPCOFFEA manager helper provenance cannot be read; campaign validity was not changed." >&2
    return 1
  fi
  validate_manager_repository_delta \
    TOPCOFFEA "${topcoffea_repository_root}" \
    "${production_topcoffea_git_commit}" "${current_topcoffea_commit}" \
    "${profile_reviewed_topcoffea_manager_head}" \
    topcoffea/modules/remote_environment.py
}

prepare_production_campaign() {
  local plan_directory
  local schema_version=5
  local allow_manager_provenance_delta=false
  local recorded_manager_commit=""

  production_assert_live_plan
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    schema_version=8
  elif [[ "${production_profile}" == run?_full ]]; then
    schema_version=6
  fi
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    production_git_commit="not_observed_no_git"
  else
    production_git_commit=$(git -C "${repository_root}" rev-parse HEAD)
  fi
  resolve_t0_postprocessor_provenance
  production_state_path="${output_dir}/${production_state_filename}"
  if [[ "${profile_resume}" == "true" && "${production_profile}" == run?_full ]]; then
    IFS=$'\t' read -r schema_version production_np_mode < <(
      production_state_tool historical_full_execution_policy "${production_state_path}" "${production_profile}"
    )
  fi
  if [[ "${dry_run}" == "true" ]]; then
    production_plan_file=$(mktemp "${TMPDIR:-/tmp}/${production_profile}_plan.XXXXXX")
  else
    if [[ "${profile_resume}" == "false" ]]; then
      mkdir -- "${output_dir}"
    fi
    campaign_lock_path="${output_dir}/.${production_profile}_campaign.lock"
    if [[ -L "${campaign_lock_path}" ]]; then
      echo "ERROR: campaign writer lock path must not be a symbolic link: ${campaign_lock_path}" >&2
      exit 1
    fi
    exec {campaign_lock_fd}>> "${campaign_lock_path}"
    flock -n "${campaign_lock_fd}" || { echo "ERROR: campaign writer lock is held." >&2; exit 1; }
    plan_directory="${output_dir}"
    production_plan_file=$(mktemp "${plan_directory}/.${production_profile}_plan.XXXXXX")
  fi
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    if [[ "${profile_resume}" == "true" ]]; then
      python ./sr_reco_manifest.py render-active "${production_state_path}" "${production_plan_file}" "${chunk_size}"
    else
      python ./sr_reco_manifest.py render-new \
        "${profile_sr_plan:-./sr_reco_default_plan.json}" \
        "${production_profile}" "${campaign_tag}" "${output_dir}" "${production_plan_file}"
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
        echo "ERROR: unknown fragment '${profile_selected_fragment}' for ${production_profile}." >&2
        exit 1
      fi
    fi
  else
    write_production_plan "${production_plan_file}"
  fi

  if [[ "${profile_resume}" == "true" ]]; then
    if [[ ! -f "${production_state_path}" ]]; then
      echo "ERROR: ${production_profile} --resume requires campaign state: ${production_state_path}" >&2
      exit 1
    fi
    allow_manager_provenance_delta=true
    recorded_manager_commit=$(production_state_tool historical_source_commit \
      "${production_state_path}" "${production_profile}" "${campaign_tag}" "${output_dir}")
    validate_resume_manager_compatibility "${recorded_manager_commit}"
    if [[ "${production_profile}" != run?_sr_ext ]]; then
      production_state_tool validate \
      "${production_state_path}" \
      "${production_plan_file}" \
      "${production_profile}" \
      "${schema_version}" \
      "${campaign_tag}" \
      "${output_dir}" \
      "${production_git_commit}" \
      "${production_env_file}" \
      "${production_env_file_sha256}" \
      "${production_environment_fingerprint}" \
      "${production_environment_mode}" \
      "${production_topcoffea_git_commit}" \
      "${production_topcoffea_source_fingerprint}" \
      "${ttgamma_sample_role_policy}" \
      "${do_systs}" \
      "${do_np}" \
      "${production_region}" \
      "${production_np_mode}" \
      "${dry_run}" \
      "${allow_manager_provenance_delta}"
    fi
  elif [[ "${dry_run}" == "false" ]]; then
    production_state_tool initialize \
      "${production_state_path}" \
      "${production_plan_file}" \
      "${production_profile}" \
      "${schema_version}" \
      "${campaign_tag}" \
      "${output_dir}" \
      "${production_git_commit}" \
      "${production_env_file}" \
      "${production_env_file_sha256}" \
      "${production_environment_fingerprint}" \
      "${production_environment_mode}" \
      "${production_topcoffea_git_commit}" \
      "${production_topcoffea_source_fingerprint}" \
      "${ttgamma_sample_role_policy}" \
      "${do_systs}" \
      "${do_np}" \
      "${production_region}" \
      "${production_np_mode}" \
      "${chunk_size}"
  fi
}

prepare_production_sumw2_options() {
  if ! is_t0_statonly_profile; then
    if [[ "${profile_resume}" == "true" \
      && -e "${output_dir}/sumw2_full_diagnostics.yml" ]]; then
      echo "ERROR: ${production_profile} resume contains the legacy full_diagnostics sumw2 override; it is incompatible with the implicit production policy." >&2
      exit 1
    fi
    return
  fi
  if [[ "${dry_run}" == "true" ]]; then
    production_sumw2_temporary_options=$(mktemp /tmp/t0_statonly_sumw2.XXXXXX.yml)
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
    echo "ERROR: ${production_profile} sumw2 options do not match full_diagnostics." >&2
    exit 1
  fi
}

cleanup_production_plan() {
  if [[ -n "${production_plan_file:-}" && -f "${production_plan_file}" ]]; then
    rm -f -- "${production_plan_file}"
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

  # cmd_ref+=(--analysis-mode taufitter)

  if [[ "${do_systs}" == "true" ]]; then
    cmd_ref+=(--do-systs)
  fi

  if is_t0_statonly_profile; then
    cmd_ref+=(--record-raw-count)
  fi

  if [[ "${do_np}" == "true" ]]; then
    # --do-np configures and certifies the current nonprompt/sumw2 contract;
    # --defer-np prevents run_analysis.py from materializing the _np artifact
    # in the processor process. run_sr_block launches run_data_driven.py only
    # after that child exits and the source artifact is verified.
    if [[ "${production_np_mode}" == "separate" ]]; then
      cmd_ref+=(--do-np --defer-np)
    else
      cmd_ref+=(--do-np --np-postprocess=inline)
    fi
  fi

  cmd_ref+=(
    -p "${output_dir}"
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

  if [[ "${split_lep_flavor}" == "true" ]]; then
    cmd_ref+=(--split-lep-flavor)
  fi

  if [[ "${dry_run}" == "true" ]]; then
    cmd_ref+=(--dry-run)
  fi
}

append_guard_options() {
  local -n guarded_cmd_ref="$1"
  local block_id="$2"
  [[ -n "${profile_wq_staging_base}" ]] || return 0
  guarded_cmd_ref+=(
    --wq-guard
    --wq-filepath "${profile_wq_staging_base}"
    --wq-invocation-id "${campaign_tag}_${block_id}_source_attempt_1"
    --wq-quota-root "${profile_wq_quota_root}"
    --wq-free-warning-bytes "${profile_wq_warning_bytes}"
    --wq-min-free-bytes "${profile_wq_min_free_bytes}"
    --wq-poll-seconds "${profile_wq_poll_seconds}"
  )
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
    "${profile_wq_staging_base}/${campaign_tag}_${block_id}_source_attempt_1"
}

native_wq_log_names=(debug.log tr.log stats.log tasks.log)

set_native_wq_log_dir() {
  local block_id="$1"
  local stage="$2"
  native_wq_log_dir="${output_dir}/work_queue_logs/${production_component}/${block_id}/${stage}_native"
}

assert_native_wq_logs_clean() {
  local name
  mkdir -p -- "${native_wq_log_dir}"
  for name in "${native_wq_log_names[@]}"; do
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

run_cr_block() {
  local year_expr="$1"
  local var_set="$2"
  shift 2

  assert_supported_year_expr "${year_expr}"

  local years=()
  local vars=()
  local cats=("$@")
  local cat_tag
  local var_tag
  local pkl_tag
  local start_epoch
  local end_epoch
  local duration_seconds
  local source_exit_code=0
  local nonprompt_exit_code=0
  local production_block
  local source_path
  local nonprompt_path
  local source_sidecar_path
  local nonprompt_sidecar_path
  local plan_output_tag
  local production_status=""
  local source_status=""
  local nonprompt_status=""
  local source_reusable=false
  local nonprompt_mark_args=()

  read -r -a years <<< "${year_expr}"
  read -r -a vars <<< "${var_set}"

  cat_tag=$(join_by - "${cats[@]}")
  var_tag=$(join_by - "${vars[@]}")
  pkl_tag="${cr_pkl_base_tag}_${cat_tag}_${var_tag}"
  production_block=$(production_block_id "${year_expr}" "${cats[*]}" "${var_set}")
  IFS=$'\t' read -r _ _ _ _ plan_output_tag _ source_path nonprompt_path < <(
    awk -F '\t' -v block_id="${production_block}" '$1 == block_id {print; exit}' "${production_plan_file}"
  )
  pkl_tag="${plan_output_tag}"
  if [[ -z "${source_path}" || -z "${nonprompt_path}" ]]; then
    echo "ERROR: unable to resolve expected output paths for ${production_block}." >&2
    exit 1
  fi
  set_native_wq_log_dir "${production_block}" source
  source_sidecar_path="${source_path}.metadata.json"
  nonprompt_sidecar_path="${nonprompt_path}.metadata.json"

  if [[ "${dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    production_status="planned"
    source_status="planned"
    nonprompt_status="blocked"
  else
    IFS=$'\t' read -r production_status source_status nonprompt_status < <(
      production_state_tool status "${production_state_path}" "${production_block}"
    )
  fi

  if [[ "${production_status}" == "success" ]]; then
    if production_outputs_present "${production_block}" "${production_plan_file}" \
      && { [[ "${production_np_mode}" != "separate" ]] \
        || { [[ -s "${source_sidecar_path}" ]] && [[ -s "${nonprompt_sidecar_path}" ]]; }; }; then
      echo "----------------------------------------"
      echo "Skipping validated ${production_profile} block: ${production_block}"
      if [[ "${production_np_mode}" == "separate" ]]; then
        echo "Campaign state, source artifact, and separate data-driven artifact are complete."
      else
        echo "Campaign state and inline nonprompt artifacts are complete."
      fi
      echo "----------------------------------------"
      record_block_result \
        "SKIPPED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "0" "0"
      return 0
    fi
    if [[ ! -s "${source_path}" ]] \
      || { [[ "${production_np_mode}" == "separate" ]] && [[ ! -s "${source_sidecar_path}" ]]; }; then
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "failed" "none" "success_state_missing_expected_source"
    else
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "nonprompt" "failed" "none" "success_state_missing_expected_nonprompt"
    fi
    echo "ERROR: ${production_profile} state marks ${production_block} successful, but an expected artifact is missing or empty." >&2
    exit 1
  fi

  case "${source_status}" in
    ready)
      if [[ "${production_np_mode}" == "inline" ]]; then
        if [[ "${production_status}" == "source_ready" \
          && "${nonprompt_status}" == "planned" \
          && -s "${source_path}" \
          && -s "${nonprompt_path}" ]]; then
          production_state_tool mark \
            "${production_state_path}" "${production_block}" \
            "nonprompt" "success" "none" "resume_inline_outputs_complete"
          echo "----------------------------------------"
          echo "Skipping recovered ${production_profile} block: ${production_block}"
          echo "Campaign state source stage and inline nonprompt artifacts are complete."
          echo "----------------------------------------"
          record_block_result \
            "SKIPPED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
            "${pkl_tag}" "0" "0"
          return 0
        fi
        if [[ "${production_status}" != "source_ready" ]] \
          || [[ "${nonprompt_status}" != "planned" ]]; then
          echo "ERROR: ${production_block} has an unrecoverable inline nonprompt state; state was not resumed." >&2
          exit 1
        fi
        if [[ ! -s "${source_path}" ]]; then
          production_state_tool mark \
            "${production_state_path}" "${production_block}" \
            "source" "failed" "none" "source_ready_state_missing_expected_source"
        else
          production_state_tool mark \
            "${production_state_path}" "${production_block}" \
            "nonprompt" "failed" "none" "source_ready_state_missing_expected_inline_nonprompt"
        fi
        echo "ERROR: ${production_block} records an incomplete inline CR stage; state was not resumed." >&2
        exit 1
      else
        if [[ ! -s "${source_path}" || ! -s "${source_sidecar_path}" ]]; then
          production_state_tool mark \
            "${production_state_path}" "${production_block}" \
            "source" "failed" "none" "source_ready_state_missing_expected_source"
          echo "ERROR: ${production_block} records a reusable source, but it or its sidecar is missing or empty." >&2
          exit 1
        fi
        source_reusable=true
      fi
      ;;
    planned|failed)
      if production_any_output_exists "${production_block}" "${production_plan_file}"; then
        echo "ERROR: ${production_profile} block ${production_block} is ${production_status}, but an expected output path already exists. Refusing ambiguous overwrite." >&2
        exit 1
      fi
      ;;
    running)
      echo "ERROR: ${production_block} has an ambiguous interrupted source stage; state was not resumed." >&2
      exit 1
      ;;
    *)
      echo "ERROR: ${production_block} has invalid source status '${source_status}'." >&2
      exit 1
      ;;
  esac

  if [[ "${production_np_mode}" == "separate" ]] \
    && { [[ -e "${nonprompt_path}" ]] || [[ -e "${nonprompt_sidecar_path}" ]]; }; then
    echo "ERROR: ${production_block} is not successful, but its expected data-driven output path already exists. Refusing ambiguous overwrite." >&2
    exit 1
  fi

  echo "----------------------------------------"
  echo "Mode: CR"
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Variables: ${vars[*]}"
  echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
  echo "Campaign tag: ${campaign_tag}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "Source stage status: ${source_status}"
  echo "Data-driven stage status: ${nonprompt_status}"
  echo "Dry run: ${dry_run}"
  echo "----------------------------------------"

  local cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pkl_tag}"
    -s "${chunk_size}"
    --cr
    --hist-vars "${vars[@]}"
    --category-groups "${cats[@]}"
  )

  build_common_command_options cmd
  append_guard_options cmd "${production_block}"
  cmd+=(--wq-log-dir "${native_wq_log_dir}")

  local nonprompt_cmd=(
    python ./run_data_driven.py
    --input-pkl "${source_path}"
    --output-pkl "${nonprompt_path}"
  )

  start_epoch=$(date +%s)

  if [[ "${source_reusable}" == "false" ]]; then
    print_command "${cmd[@]}"
    if [[ "${dry_run}" == "true" ]]; then
      if [[ "${production_np_mode}" == "inline" ]]; then
        "${cmd[@]}"
        source_exit_code=0
      elif "${cmd[@]}"; then
        source_exit_code=0
      else
        source_exit_code=$?
      fi
    else
      if production_any_output_exists "${production_block}" "${production_plan_file}" \
        || [[ -e "${source_sidecar_path}" ]] \
        || [[ -e "${nonprompt_sidecar_path}" ]]; then
        echo "ERROR: ${production_block} expected output path already exists; refusing ambiguous overwrite." >&2
        exit 1
      fi
      assert_native_wq_logs_clean
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        source running none source_child_started "${cmd[@]}"

      # Keep child stdout/stderr attached directly to the caller. The
      # conditional invocation captures only the exit status.
      if run_production_source_child \
        "${production_block}" "${source_path}" "${nonprompt_path}" "${cmd[@]}"; then
        source_exit_code=0
      else
        source_exit_code=$?
      fi
    fi
  else
    echo "Reusing validated completed source for ${production_block}: ${source_path}"
  fi

  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if [[ "${dry_run}" == "true" ]]; then
    if (( source_exit_code != 0 )); then
      record_block_result \
        "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
      echo "ERROR: ${production_block} source dry-run resolution failed with exit code ${source_exit_code}." >&2
      return 0
    fi
    if [[ "${production_np_mode}" == "separate" ]]; then
      echo "Separate nonprompt/charge-flip command (not executed by dry-run):"
      printf ' %q' "${nonprompt_cmd[@]}"
      echo
    fi
    record_block_result \
      "DRY_RUN" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "0" "${duration_seconds}"
    echo
    return 0
  fi

  if [[ "${source_reusable}" == "false" ]] && (( source_exit_code != 0 )); then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source failed "${source_exit_code}" source_child_exit_nonzero "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    write_failure_snapshot "${production_block}" source
    record_block_result \
      "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
    echo \
      "ERROR: ${year_expr} CR failed for ${cat_tag} / ${var_tag} with exit code ${source_exit_code}; continuing with the next block." \
      >&2
    return 0
  fi

  if [[ "${source_reusable}" == "false" ]] \
    && { [[ ! -s "${source_path}" ]] \
      || { [[ "${production_np_mode}" == "separate" ]] \
        && { [[ ! -s "${source_sidecar_path}" ]] \
          || [[ -e "${nonprompt_path}" ]] \
          || [[ -e "${nonprompt_sidecar_path}" ]]; }; } \
      || { [[ "${production_np_mode}" == "inline" ]] && [[ ! -s "${nonprompt_path}" ]]; }; }; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source failed "${source_exit_code}" source_exit_zero_invalid_stage_outputs "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    write_failure_snapshot "${production_block}" source
    record_block_result \
      "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
    echo "ERROR: ${production_block} returned zero without the exact expected stage outputs; continuing with the next block." >&2
    return 0
  fi

  if [[ "${source_reusable}" == "false" ]]; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source ready "${source_exit_code}" source_child_exit_zero_expected_source_present "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    if ! complete_guarded_staging "${production_block}"; then
      echo "WARNING: guarded staging cleanup refused after native logs; source receipt remains valid and staging requires review." >&2
    fi
  fi

  if [[ "${production_np_mode}" == "inline" ]]; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      nonprompt success "${source_exit_code}" inline_nonprompt_exit_zero_expected_output_present "${duration_seconds}"
    record_block_result \
      "SUCCESS" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
    echo "${year_expr} CR done for ${cat_tag} / ${var_tag}"
    echo "----------------------------------------"
    echo
    return 0
  fi

  if is_t0_statonly_profile; then
    nonprompt_mark_args=(
      --postprocessor-provenance
      "${production_postprocessor_topeft_git_commit}"
      "${production_postprocessor_topcoffea_git_commit}"
      --
    )
  fi
  nonprompt_mark_args+=("${nonprompt_cmd[@]}")
  production_state_tool mark \
    "${production_state_path}" "${production_block}" \
    "nonprompt" "running" "none" "separate_data_driven_child_started_after_source_exit" "${nonprompt_mark_args[@]}"
  print_command "${nonprompt_cmd[@]}"
  if run_production_nonprompt_child \
    "${production_block}" "${source_path}" "${nonprompt_path}" "${nonprompt_cmd[@]}"; then
    nonprompt_exit_code=0
  else
    nonprompt_exit_code=$?
  fi

  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if (( nonprompt_exit_code == 0 )) \
    && [[ -s "${nonprompt_path}" ]] \
    && [[ -s "${nonprompt_sidecar_path}" ]]; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "nonprompt" "success" "${nonprompt_exit_code}" "separate_data_driven_exit_zero_expected_output_present" "${duration_seconds}"
    record_block_result \
      "SUCCESS" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${nonprompt_exit_code}" "${duration_seconds}"
    echo "${year_expr} CR source and separate nonprompt/charge-flip stages done for ${cat_tag} / ${var_tag}"
  else
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "nonprompt" "failed" "${nonprompt_exit_code}" "separate_data_driven_failed_or_missing_output" "${duration_seconds}"
    write_failure_snapshot "${production_block}" nonprompt
    record_block_result \
      "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${nonprompt_exit_code}" "${duration_seconds}"
    echo \
      "ERROR: ${production_block} separate nonprompt/charge-flip stage failed or its output is missing/empty; continuing with the next block." \
      >&2
  fi

  echo "----------------------------------------"
  echo
}

run_sr_block() {
  local year_expr="$1"
  local var_set="$2"
  shift 2

  assert_supported_year_expr "${year_expr}"

  local years=()
  local vars=()
  local cats=("$@")
  local cat_tag
  local var_tag
  local pkl_tag
  local source_path
  local nonprompt_path
  local source_sidecar_path
  local nonprompt_sidecar_path
  local plan_output_tag
  local start_epoch
  local end_epoch
  local duration_seconds
  local source_exit_code=0
  local nonprompt_exit_code=0
  local production_block=""
  local production_status=""
  local source_status=""
  local nonprompt_status=""
  local source_reusable=false
  local nonprompt_mark_args=()

  read -r -a years <<< "${year_expr}"
  read -r -a vars <<< "${var_set}"

  cat_tag=$(join_by - "${cats[@]}")
  var_tag=$(join_by - "${vars[@]}")
  pkl_tag="${sr_pkl_base_tag}_${cat_tag}_${var_tag}"

  production_block=$(production_block_id "${year_expr}" "${cats[*]}" "${var_set}")
  IFS=$'\t' read -r _ _ _ _ plan_output_tag _ source_path nonprompt_path < <(
    awk -F '\t' -v block_id="${production_block}" '$1 == block_id {print; exit}' "${production_plan_file}"
  )
  pkl_tag="${plan_output_tag}"
  if [[ -z "${source_path}" || -z "${nonprompt_path}" ]]; then
    echo "ERROR: unable to resolve expected output paths for ${production_block}." >&2
    exit 1
  fi
  set_native_wq_log_dir "${production_block}" source
  source_sidecar_path="${source_path}.metadata.json"
  nonprompt_sidecar_path="${nonprompt_path}.metadata.json"

  if [[ "${dry_run}" == "true" && "${profile_resume}" == "false" ]]; then
    production_status="planned"
    source_status="planned"
    nonprompt_status="blocked"
  else
    IFS=$'\t' read -r production_status source_status nonprompt_status < <(
      production_state_tool status "${production_state_path}" "${production_block}"
    )
  fi

  if [[ "${production_status}" == "success" ]]; then
    if [[ -s "${source_path}" && -s "${nonprompt_path}" ]] \
      && { [[ "${production_profile}" != "t0_sr_statonly" ]] \
        || { [[ -s "${source_sidecar_path}" ]] && [[ -s "${nonprompt_sidecar_path}" ]]; }; }; then
      echo "----------------------------------------"
      echo "Skipping validated ${production_profile} block: ${production_block}"
      echo "Campaign state, source artifact, and separate nonprompt artifact are complete."
      echo "----------------------------------------"
      record_block_result \
        "SKIPPED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "0" "0"
      return 0
    fi
    if [[ ! -s "${source_path}" ]] \
      || { [[ "${production_profile}" == "t0_sr_statonly" ]] && [[ ! -s "${source_sidecar_path}" ]]; }; then
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "failed" "none" "success_state_missing_expected_source"
    else
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "nonprompt" "failed" "none" "success_state_missing_expected_nonprompt"
    fi
    echo "ERROR: ${production_profile} state marks ${production_block} successful, but an expected artifact is missing or empty." >&2
    exit 1
  fi

  case "${source_status}" in
    ready)
      if [[ ! -s "${source_path}" ]] \
        || { [[ "${production_profile}" == "t0_sr_statonly" ]] && [[ ! -s "${source_sidecar_path}" ]]; }; then
        production_state_tool mark \
          "${production_state_path}" "${production_block}" \
          "source" "failed" "none" "source_ready_state_missing_expected_source"
        echo "ERROR: ${production_block} records a reusable source, but it is missing or empty." >&2
        exit 1
      fi
      source_reusable=true
      ;;
    planned|failed)
      if [[ -e "${source_path}" || -L "${source_path}" \
        || -e "${nonprompt_path}" || -L "${nonprompt_path}" ]] \
        || { [[ "${production_profile}" == "t0_sr_statonly" ]] \
          && { [[ -e "${source_sidecar_path}" ]] || [[ -L "${source_sidecar_path}" ]] \
            || [[ -e "${nonprompt_sidecar_path}" ]] || [[ -L "${nonprompt_sidecar_path}" ]]; }; }; then
        echo "ERROR: ${production_profile} block ${production_block} is ${production_status}, but an expected output path already exists. Refusing ambiguous overwrite." >&2
        exit 1
      fi
      ;;
    running)
      echo "ERROR: ${production_block} has an ambiguous interrupted source stage; run a non-dry --resume preflight before retrying." >&2
      exit 1
      ;;
    *)
      echo "ERROR: ${production_block} has invalid source status '${source_status}'." >&2
      exit 1
      ;;
  esac

  if [[ -e "${nonprompt_path}" || -L "${nonprompt_path}" ]] \
    || { [[ "${production_profile}" == "t0_sr_statonly" ]] \
      && { [[ -e "${nonprompt_sidecar_path}" ]] || [[ -L "${nonprompt_sidecar_path}" ]]; }; }; then
    echo "ERROR: ${production_block} is not successful, but its expected _np path already exists. Refusing ambiguous overwrite." >&2
    exit 1
  fi

  echo "----------------------------------------"
  echo "Mode: SR"
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    echo "production_block_id: ${production_block}"
    echo "expected_nominal_path: ${source_path}"
    echo "expected_np_path: ${nonprompt_path}"
  fi
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Variables: ${vars[*]}"
  echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
  echo "Campaign tag: ${campaign_tag}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "Source stage status: ${source_status}"
  echo "Nonprompt stage status: ${nonprompt_status}"
  echo "Dry run: ${dry_run}"
  echo "----------------------------------------"

  local source_cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pkl_tag}"
    -s "${chunk_size}"
    --sr
    --hist-vars "${vars[@]}"
    --category-groups "${cats[@]}"
  )

  build_common_command_options source_cmd
  append_guard_options source_cmd "${production_block}"
  source_cmd+=(--wq-log-dir "${native_wq_log_dir}")
  local nonprompt_cmd=(
    python ./run_data_driven.py
    --input-pkl "${source_path}"
    --output-pkl "${nonprompt_path}"
  )
  if [[ "${production_profile}" == "t0_sr_statonly" \
    && "${source_reusable}" == "true" \
    && -s "${source_sidecar_path}" ]] \
    && ! grep -Fq '"histogram_applicability"' "${source_sidecar_path}"; then
    nonprompt_cmd+=(
      --legacy-campaign-state "${production_state_path}"
      --legacy-campaign-block "${production_block}"
    )
    echo "Legacy context transported (no applicability verdict):"
    production_state_tool legacy_context_summary \
      "${production_state_path}" "${production_block}"
  fi

  start_epoch=$(date +%s)

  if [[ "${source_reusable}" == "false" ]]; then
    if [[ "${dry_run}" == "false" ]]; then
      assert_native_wq_logs_clean
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "running" "none" "source_child_started" \
        -- "${source_cmd[@]}"
    fi
    if [[ "${dry_run}" == "true" && "${production_profile}" == "run2_full" ]]; then
      printf 'SRPLOT009_BLOCK_COMMAND\t%s\t' "${production_block}"
      printf ' %q' "${source_cmd[@]}"
      printf '\n'
    fi
    print_command "${source_cmd[@]}"
    # This conditional waits for the heavy processor child to exit completely.
    # The separate data-driven process is not created until after this returns.
    if [[ "${dry_run}" == "true" ]]; then
      if "${source_cmd[@]}"; then
        source_exit_code=0
      else
        source_exit_code=$?
      fi
    elif run_production_source_child \
      "${production_block}" "${source_path}" "${nonprompt_path}" "${source_cmd[@]}"; then
      source_exit_code=0
    else
      source_exit_code=$?
    fi
  else
    echo "Reusing validated completed source for ${production_block}: ${source_path}"
  fi
  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if [[ "${dry_run}" == "true" ]]; then
    if (( source_exit_code != 0 )); then
      record_block_result \
        "FAILED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
      echo "ERROR: ${production_block} source dry-run resolution failed with exit code ${source_exit_code}." >&2
      return 0
    fi
    echo "Separate nonprompt command (not executed by dry-run):"
    printf ' %q' "${nonprompt_cmd[@]}"
    echo
    record_block_result \
      "DRY_RUN" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "0" "${duration_seconds}"
    echo "${year_expr} ${production_profile} two-stage dry-run resolved for ${cat_tag} / ${var_tag}"
    echo "----------------------------------------"
    echo
    return 0
  fi

  if [[ "${source_reusable}" == "false" ]]; then
    if (( source_exit_code != 0 )); then
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "failed" "${source_exit_code}" "source_child_exit_nonzero" "${duration_seconds}"
      archive_native_wq_logs "${production_block}" source
      write_failure_snapshot "${production_block}" source
      record_block_result \
        "FAILED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
      echo "ERROR: ${production_block} source production failed with exit code ${source_exit_code}; continuing." >&2
      return 0
    fi
    if [[ ! -s "${source_path}" ]] \
      || { [[ "${production_profile}" == "t0_sr_statonly" ]] && [[ ! -s "${source_sidecar_path}" ]]; } \
      || { [[ "${production_np_mode}" == "separate" ]] && [[ -e "${nonprompt_path}" ]]; } \
      || { [[ "${production_profile}" == "t0_sr_statonly" ]] \
        && [[ "${production_np_mode}" == "separate" ]] && [[ -e "${nonprompt_sidecar_path}" ]]; } \
      || { [[ "${production_np_mode}" == "inline" ]] && [[ ! -s "${nonprompt_path}" ]]; }; then
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "failed" "${source_exit_code}" "source_exit_zero_invalid_stage_outputs" "${duration_seconds}"
      archive_native_wq_logs "${production_block}" source
      write_failure_snapshot "${production_block}" source
      record_block_result \
        "FAILED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
      echo "ERROR: ${production_block} source stage did not produce exactly one nonempty source and no _np output." >&2
      return 0
    fi
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "source" "ready" "${source_exit_code}" "source_child_exit_zero_expected_source_present" "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    if ! complete_guarded_staging "${production_block}"; then
      echo "WARNING: guarded staging cleanup refused after native logs; source receipt remains valid and staging requires review." >&2
    fi
  fi

  if [[ "${production_np_mode}" == "inline" ]]; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "nonprompt" "success" "${source_exit_code}" "inline_nonprompt_exit_zero_expected_output_present" "${duration_seconds}"
    record_block_result \
      "SUCCESS" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${source_exit_code}" "${duration_seconds}"
    echo "${year_expr} SR source and inline nonprompt stages done for ${cat_tag} / ${var_tag}"
    echo "----------------------------------------"
    echo
    return 0
  fi

  if [[ "${production_profile}" == "t0_sr_statonly" ]]; then
    nonprompt_mark_args=(
      --postprocessor-provenance
      "${production_postprocessor_topeft_git_commit}"
      "${production_postprocessor_topcoffea_git_commit}"
      --
    )
  fi
  nonprompt_mark_args+=("${nonprompt_cmd[@]}")
  production_state_tool mark \
    "${production_state_path}" "${production_block}" \
    "nonprompt" "running" "none" "separate_nonprompt_child_started_after_source_exit" "${nonprompt_mark_args[@]}"
  print_command "${nonprompt_cmd[@]}"
  if run_production_nonprompt_child \
    "${production_block}" "${source_path}" "${nonprompt_path}" "${nonprompt_cmd[@]}"; then
    nonprompt_exit_code=0
  else
    nonprompt_exit_code=$?
  fi

  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if (( nonprompt_exit_code == 0 )) && [[ -s "${nonprompt_path}" ]] \
    && { [[ "${production_profile}" != "t0_sr_statonly" ]] || [[ -s "${nonprompt_sidecar_path}" ]]; }; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "nonprompt" "success" "${nonprompt_exit_code}" "separate_nonprompt_exit_zero_expected_output_present" "${duration_seconds}"
    record_block_result \
      "SUCCESS" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${nonprompt_exit_code}" "${duration_seconds}"
    echo "${year_expr} SR source and separate nonprompt stages done for ${cat_tag} / ${var_tag}"
  else
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      "nonprompt" "failed" "${nonprompt_exit_code}" "separate_nonprompt_failed_or_missing_output" "${duration_seconds}"
    write_failure_snapshot "${production_block}" nonprompt
    record_block_result \
      "FAILED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${nonprompt_exit_code}" "${duration_seconds}"
    echo \
      "ERROR: ${production_block} separate nonprompt stage failed or its _np output is missing/empty; continuing with the next block." \
      >&2
  fi

  echo "----------------------------------------"
  echo
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
  "run3_full SR category-to-variable mapping" \
  "${#run3_full_category_sets[@]}" \
  "${#run3_full_category_var_set_names[@]}"

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
  "${run3_full_category_var_set_names[@]}"; do
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
echo "env_file: ${production_env_file}"
echo "env_file_sha256: ${production_env_file_sha256}"
echo "environment_mode: ${production_environment_mode}"
echo "environment_policy: ${production_environment_mode}"
if is_t0_statonly_profile; then
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

echo "run3_full Run-3 SR category blocks:"
printf '  %s\n' "${run3_full_category_sets[@]}"

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

###############################################################################
# Main CR production
###############################################################################

if [[ "${run_cr}" == "true" ]]; then
  for year_expr in "${cr_year_sets[@]}"; do
    for category_index in "${!cr_category_sets[@]}"; do
      category_set="${cr_category_sets[category_index]}"
      category_var_set_name="${cr_category_var_set_names[category_index]}"

      declare -n category_var_sets="${category_var_set_name}"
      read -r -a cats <<< "${category_set}"

      for var_set in "${category_var_sets[@]}"; do
        run_cr_block "${year_expr}" "${var_set}" "${cats[@]}"
      done

      unset -n category_var_sets
    done
  done
else
  echo "Skipping CR production because run_cr=${run_cr}"
  echo
fi

###############################################################################
# Main SR production
###############################################################################

if [[ "${run_sr}" == "true" ]]; then
  if [[ "${production_profile}" == run?_sr_ext ]]; then
    for plan_index in "${!production_block_ids[@]}"; do
      current_fragment="${production_block_ids[plan_index]}"
      if [[ -n "${profile_selected_fragment}" \
        && "${current_fragment}" != "${profile_selected_fragment}" ]]; then
        continue
      fi
      read -r -a cats <<< "${production_plan_category_sets[plan_index]}"
      run_sr_block "${production_plan_year_exprs[plan_index]}" \
        "${production_plan_var_sets[plan_index]}" "${cats[@]}"
      if [[ -n "${profile_wq_staging_base}" ]] && (( run_failure_count > 0 )); then
        if [[ "${dry_run}" == "true" ]]; then
          echo "Guarded campaign stopped after the first failed dry-run fragment; remaining fragments were not started." >&2
          break
        else
          current_status_line=$(production_state_tool status "${production_state_path}" "${current_fragment}")
          IFS=$'\t' read -r _ current_source_status _ <<< "${current_status_line}"
          if [[ "${current_source_status}" == "failed" ]]; then
            echo "Guarded campaign stopped after a source-stage failure; remaining fragments were not started." >&2
            break
          elif [[ "${current_source_status}" != "ready" ]]; then
            echo "ERROR: ${current_fragment} has an unexpected post-block source status: ${current_source_status}." >&2
            exit 1
          fi
        fi
      fi
    done
  else
  for year_index in "${!sr_year_sets[@]}"; do
    year_expr="${sr_year_sets[year_index]}"
    category_set_array_name="${sr_year_category_set_names[year_index]}"
    category_var_map_array_name="${sr_year_category_var_set_names[year_index]}"

    declare -n active_category_sets="${category_set_array_name}"
    declare -n active_category_var_set_names="${category_var_map_array_name}"

    assert_parallel_array_lengths \
      "SR category-to-variable mapping for '${year_expr}'" \
      "${#active_category_sets[@]}" \
      "${#active_category_var_set_names[@]}"

    for category_index in "${!active_category_sets[@]}"; do
      category_set="${active_category_sets[category_index]}"
      category_var_set_name="${active_category_var_set_names[category_index]}"

      assert_array_defined "${category_var_set_name}"
      declare -n category_var_sets="${category_var_set_name}"
      read -r -a cats <<< "${category_set}"

      for chunk_index in "${!category_var_sets[@]}"; do
        var_set="${category_var_sets[chunk_index]}"
        if [[ -n "${profile_selected_fragment}" ]]; then
          profile_group_suffixes=(a b c d e)
          current_fragment="${production_profile}_${profile_group_suffixes[category_index]}_$((chunk_index + 1))"
          [[ "${current_fragment}" == "${profile_selected_fragment}" ]] || continue
        fi
        run_sr_block "${year_expr}" "${var_set}" "${cats[@]}"
        if [[ -n "${profile_wq_staging_base}" ]] && (( run_failure_count > 0 )); then
          echo "Guarded campaign stopped after the first failed fragment; remaining fragments were not started." >&2
          break 3
        fi
      done

      unset -n category_var_sets
    done

    unset -n active_category_sets
    unset -n active_category_var_set_names
  done
  fi
else
  echo "Skipping SR production because run_sr=${run_sr}"
  echo
fi

###############################################################################
# Final status
###############################################################################

print_run_summary

echo "campaign_tag: ${campaign_tag}"
echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
echo "output_dir: ${output_dir}"

if [[ "${dry_run}" == "true" && "${production_profile}" == "run2_full" ]]; then
  echo "dry_run_complete: five commands resolved; no environment, output root, scheduler, or production action was created"
elif [[ "${dry_run}" == "true" && "${production_profile}" == "t0_sr_statonly" ]]; then
  echo "dry_run_complete: ten commands resolved; no environment, output root, scheduler, or production action was created"
elif [[ "${dry_run}" == "true" && "${production_profile}" == "t0_cr_statonly" ]]; then
  echo "dry_run_complete: eighteen source commands and eighteen separate data-driven commands resolved; no environment, output root, scheduler, or production action was created"
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
