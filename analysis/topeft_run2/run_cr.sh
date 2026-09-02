#!/usr/bin/env bash
set -euo pipefail

matrix_profile=""
matrix_dry_run=false
matrix_output_dir=""
matrix_campaign_tag=""
matrix_env_file=""
matrix_resume=false
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
      --resume) matrix_resume=true; index=$((index + 1)) ;;
      -h|--help) index=$((index + 1)) ;;
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
    --env-file /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2/topeft-envs/env_spec_9d72aad444117c28.tar.gz \
    "${default_dry_run[@]}"
fi

matrix_parse_args "$@"
case "${matrix_profile}" in
  run2_full|run3_full|run2_full_CR|run3_full_CR|run2_run3_full|run2_run3_full_CR)
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
    if [[ -n "${matrix_env_file}" && "${matrix_env_file}" != "/users/apiccine/work/correction-lib/topeft/analysis/topeft_run2/topeft-envs/env_spec_9d72aad444117c28.tar.gz" ]]; then
      echo "ERROR: requested profiles are pinned to the required frozen snapshot archive." >&2
      exit 1
    fi
    if [[ "${matrix_resume}" == "true" && "${matrix_profile}" != "run3_full" ]]; then
      echo "ERROR: ${matrix_profile} has no automatic resume; inspect interrupted state first." >&2
      exit 1
    fi
    ;;
esac

case "${matrix_profile}" in
  run2_run3_full|run2_run3_full_CR)
    if [[ -e "${matrix_output_dir}" ]]; then
      echo "ERROR: combined output namespace already exists: ${matrix_output_dir}" >&2
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
    if [[ "${matrix_dry_run}" == "false" ]]; then
      mkdir -- "${matrix_output_dir}"
    fi
    component_common=(--env-file /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2/topeft-envs/env_spec_9d72aad444117c28.tar.gz)
    [[ "${matrix_dry_run}" == "true" ]] && component_common+=(--dry-run)
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
  [--output-dir PATH] [--campaign-tag TAG] [--env-file PATH] [--resume]

Public PROFILE values:
  run2_full, run3_full, run2_run3_full
  run2_full_CR, run3_full_CR, run2_run3_full_CR

With no arguments, run_cr.sh remains a backward-compatible alias for the fixed
five-block run2_full campaign. Combined profiles run Run 2 then Run 3 in
separate child namespaces. A safely classified Run-2-local failure or blocker
does not suppress the independent Run-3 component; a shared unsafe state does.

All six public profiles use the exact maintained frozen archive in snapshot
mode, Work Queue without a profile-level worker count, and explicit
full_diagnostics sumw2 storage. Explicit component and combined profiles
require a fresh absolute output directory and campaign tag.

run3_full is the canonical complete Run-3 SR source-production profile.
rebin_fine is the specialized six-block Run-2/Run-3 source-production profile
for fitting families whose bins changed and remains available as a legacy
profile. Resume remains available only where already maintained by the legacy
generic profile machinery.
EOF
}

production_profile=""
profile_dry_run=false
profile_output_dir=""
profile_campaign_tag=""
profile_env_file=""
profile_resume=false

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
    --resume)
      profile_resume=true
      shift
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
  run2_full|run3_full|run2_full_CR|run3_full_CR|rebin_fine) ;;
  *)
    echo "ERROR: unsupported production profile '${production_profile}'." >&2
    exit 1
    ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
cd -- "${script_dir}"

validation_backend="${SRPLOT009_VALIDATION_BACKEND:-}"
validation_root="${SRPLOT009_VALIDATION_ROOT:-}"
validation_scenario="${SRPLOT009_VALIDATION_SCENARIO:-success}"
native_wq_log_dir="${script_dir}"
if [[ -n "${validation_backend}" ]]; then
  case "${validation_root}" in /tmp/*) ;; *) echo "ERROR: SRPLOT009_VALIDATION_ROOT must be below /tmp." >&2; exit 1 ;; esac
  case "${validation_backend}" in /tmp/*) ;; *) echo "ERROR: SRPLOT009_VALIDATION_BACKEND must be below /tmp." >&2; exit 1 ;; esac
  [[ -x "${validation_backend}" ]] || { echo "ERROR: validation backend is not executable." >&2; exit 1; }
  native_wq_log_dir="${validation_root}/native_logs"
  mkdir -p -- "${native_wq_log_dir}"
fi

###############################################################################
# Global configuration
###############################################################################

output_dir=""
chunk_size="100000"

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

if [[ "${production_profile}" == "run3_full" ]] \
  && { [[ "${output_dir}" == "/groups/klannon/apiccine/preappr_v9_260729" ]] \
    || [[ "${output_dir}" == "/groups/klannon/apiccine/rebin_fine_260818_v3" ]] \
    || [[ "${campaign_tag}" == "ANv9" ]] \
    || [[ "${campaign_tag}" == "rebin-fine-260818-v3" ]]; }; then
  echo "ERROR: run3_full must use a fresh namespace, not a historical baseline or v3 campaign." >&2
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

cr_pkl_base_tag="${campaign_tag}"
sr_pkl_base_tag="${campaign_tag}"

# Select the one region owned by the public component profile. The legacy
# rebin_fine profile retains its maintained two-stage SR behavior.
case "${production_profile}" in
  run2_full)
    run_cr=false
    run_sr=true
    production_region="SR"
    production_np_mode="inline"
    production_component="run2"
    ;;
  run3_full|rebin_fine)
    run_cr=false
    run_sr=true
    production_region="SR"
    production_np_mode="separate"
    production_component="run3"
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

# Static memory-bounded source-production plan for the families whose fitting
# bins changed. The plan is intentionally limited to canonical live keys and
# excludes njets, whose missing-parton workflow is produced separately.
rebin_fine_category_sets=(
  "2lss_1tau 3l_m_offZ"
  "3l_p_offZ 3l_onZ_tau"
  "3l_fwd"
)

rebin_fine_2lss_1tau_3l_m_offz_var_sets=(
  "lj0pt ptll ptz_wtau"
)

rebin_fine_3l_p_offz_3l_onZ_tau_var_sets=(
  "lj0pt ptz ptll"
)

rebin_fine_3l_fwd_var_sets=(
  "lt"
)

rebin_fine_category_var_set_names=(
  "rebin_fine_2lss_1tau_3l_m_offz_var_sets"
  "rebin_fine_3l_p_offz_3l_onZ_tau_var_sets"
  "rebin_fine_3l_fwd_var_sets"
)

sr_year_sets=()
sr_year_category_set_names=()
sr_year_category_var_set_names=()
if [[ "${production_profile}" == "rebin_fine" ]]; then
  sr_year_sets=(
    "2016APV 2016 2017 2018"
    "2022 2022EE 2023 2023BPix"
  )
  sr_year_category_set_names=(
    "rebin_fine_category_sets"
    "rebin_fine_category_sets"
  )
  sr_year_category_var_set_names=(
    "rebin_fine_category_var_set_names"
    "rebin_fine_category_var_set_names"
  )
elif [[ "${production_profile}" == "run2_full" ]]; then
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
  sr_year_category_var_set_names=(
    "run3_full_category_var_set_names"
  )
fi

case "${production_profile}" in
  run2_full_CR) cr_year_sets=("2016APV 2016 2017 2018") ;;
  run3_full_CR) cr_year_sets=("2022 2022EE" "2023 2023BPix") ;;
esac

production_state_filename=".${production_profile}_campaign_state.json"
production_block_ids=()
production_plan_year_exprs=()
production_plan_category_sets=()
production_plan_var_sets=()

case "${production_profile}" in
  rebin_fine)
    production_block_ids=(run2_a run2_b run2_c run3_a run3_b run3_c)
    production_plan_year_exprs=(
      "2016APV 2016 2017 2018" "2016APV 2016 2017 2018" "2016APV 2016 2017 2018"
      "2022 2022EE 2023 2023BPix" "2022 2022EE 2023 2023BPix" "2022 2022EE 2023 2023BPix"
    )
    production_plan_category_sets=(
      "2lss_1tau 3l_m_offZ" "3l_p_offZ 3l_onZ_tau" "3l_fwd"
      "2lss_1tau 3l_m_offZ" "3l_p_offZ 3l_onZ_tau" "3l_fwd"
    )
    production_plan_var_sets=(
      "lj0pt ptll ptz_wtau" "lj0pt ptz ptll" "lt"
      "lj0pt ptll ptz_wtau" "lj0pt ptz ptll" "lt"
    )
    ;;
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
  run2_full_CR|run3_full_CR)
    if [[ "${production_profile}" == "run2_full_CR" ]]; then
      profile_cr_year_sets=("2016APV 2016 2017 2018")
    else
      profile_cr_year_sets=("2022 2022EE" "2023 2023BPix")
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
    if [[ "${production_profile}" == "run2_full" || "${production_profile}" == *_CR ]]; then
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
import sys
from pathlib import Path


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
        block["expected_nominal_path"] = nominal
        block["expected_np_path"] = nonprompt
        blocks.append(block)
    if not blocks or len({block["id"] for block in blocks}) != len(blocks):
        fail(f"generated {production_profile} plan does not contain unique blocks")
    return blocks


def desired_state(arguments):
    plan_path = Path(arguments[0])
    production_profile, schema_version, tag, output_dir, commit, env_file, env_sha256, env_fingerprint, topcoffea_commit, topcoffea_source, ttgamma, do_systs, do_np, region, nonprompt_mode = arguments[1:]
    return {
        "schema_version": int(schema_version),
        "production_profile": production_profile,
        "campaign_tag": tag,
        "output_dir": output_dir,
        "topeft_git_commit": commit,
        "env_file": env_file,
        "env_file_sha256": env_sha256,
        "environment_fingerprint": env_fingerprint,
        "topcoffea_git_commit": topcoffea_commit,
        "topcoffea_relevant_source_fingerprint": topcoffea_source,
        "ttgamma_sample_role_policy": ttgamma,
        "do_systs": do_systs == "true",
        "do_np": do_np == "true",
        "region": region,
        "nonprompt_mode": nonprompt_mode,
        "blocks": read_plan(plan_path, production_profile),
    }


def validate_state(state, desired):
    for key in (
        "schema_version",
        "production_profile",
        "campaign_tag",
        "output_dir",
        "topeft_git_commit",
        "env_file",
        "env_file_sha256",
        "environment_fingerprint",
        "topcoffea_git_commit",
        "topcoffea_relevant_source_fingerprint",
        "ttgamma_sample_role_policy",
        "do_systs",
        "do_np",
        "region",
        "nonprompt_mode",
    ):
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


mode = sys.argv[1]
state_path = Path(sys.argv[2])

if mode in {"initialize", "validate"}:
    desired = desired_state(sys.argv[3:19])
    readonly = len(sys.argv) > 19 and sys.argv[19] == "true"
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
        atomic_write(state_path, state)
    else:
        state = load(state_path)
        validate_state(state, desired)
        for block in state["blocks"]:
            if block["source_status"] == "running":
                fail(f"block {block['id']} has an ambiguous interrupted source stage; state was not rewritten")
            elif block["nonprompt_status"] == "running":
                fail(f"block {block['id']} has an ambiguous interrupted nonprompt stage; state was not rewritten")
    raise SystemExit(0)

state = load(state_path)
if mode == "env_file":
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
    env_file = state.get("env_file")
    if not isinstance(env_file, str) or not env_file.startswith("/"):
        fail("campaign state does not contain an absolute frozen env_file")
    print(env_file)
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
    if stage == "source" and stage_status not in VALID_SOURCE_STATUSES:
        fail(f"invalid requested source status {stage_status!r}")
    if stage == "nonprompt" and stage_status not in VALID_NONPROMPT_STATUSES:
        fail(f"invalid requested nonprompt status {stage_status!r}")
    if stage not in {"source", "nonprompt"}:
        fail(f"invalid requested stage {stage!r}")
    for block in state.get("blocks", []):
        if block.get("id") == block_id:
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
            block["last_transition_utc"] = timestamp
            block["last_transition_detail"] = detail
            block.setdefault("transitions", []).append(
                {
                    "timestamp_utc": block["last_transition_utc"],
                    "stage": stage,
                    "status": stage_status,
                    "exit_code": parsed_exit_code,
                    "signal": block[f"{stage}_signal"],
                    "duration_seconds": block.get(f"{stage}_duration_seconds"),
                    "detail": detail,
                }
            )
            state["updated_at_utc"] = now_utc()
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
    final_exit_code = 0 if classification == "success" else 1
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
        "profile", "campaign_tag", "era", "block_id", "categories", "variables",
        "source_status", "source_exit", "nonprompt_status", "nonprompt_exit",
        "final_block_status", "expected_nominal_path", "expected_np_path",
    ]
    rows = []
    for block in state["blocks"]:
        rows.append([
            state["production_profile"], state["campaign_tag"], " ".join(block["years"]), block["id"],
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
  local matrix_env_file=/users/apiccine/work/correction-lib/topeft/analysis/topeft_run2/topeft-envs/env_spec_9d72aad444117c28.tar.gz
  local matrix_env_sha256=8245afe4b3c28f4948039d383ad2176f1ee3ebb5e61bcdf1b49289452b025332
  local frozen_env_file=""
  local canonical_requested_env_file=""
  local direct_sha256=""
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
    if [[ "${production_profile}" != "rebin_fine" && "${requested_env_file}" != "${matrix_env_file}" ]]; then
      echo "ERROR: ${production_profile} resume state does not use the required frozen snapshot archive." >&2
      exit 1
    fi
    if [[ "${production_profile}" != "rebin_fine" ]]; then
      validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
    else
      validation_args=(--validate-env-file --env-file "${requested_env_file}")
    fi
  elif [[ -n "${requested_env_file}" ]]; then
    if [[ "${requested_env_file}" != /* ]]; then
      echo "ERROR: ${production_profile} --env-file must be an absolute path: ${requested_env_file}" >&2
      exit 1
    fi
    canonical_requested_env_file=$(readlink -f -- "${requested_env_file}" || true)
    if [[ "${production_profile}" != "rebin_fine" && "${canonical_requested_env_file}" != "${matrix_env_file}" ]]; then
      echo "ERROR: ${production_profile} is pinned to the required frozen snapshot archive." >&2
      exit 1
    fi
    if [[ "${production_profile}" != "rebin_fine" ]]; then
      validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
    else
      validation_args=(--validate-env-file --env-file "${requested_env_file}")
    fi
  elif [[ "${production_profile}" != "rebin_fine" ]]; then
    requested_env_file="${matrix_env_file}"
    canonical_requested_env_file="${matrix_env_file}"
    validation_args=(--validate-env-file --env-integrity-only --env-file "${requested_env_file}")
  else
    echo "ERROR: rebin_fine requires an explicit --env-file; no environment was built." >&2
    exit 1
  fi

  if [[ "${production_profile}" != "rebin_fine" ]]; then
    if [[ -n "${validation_backend}" ]]; then
      direct_sha256="${matrix_env_sha256}"
    else
      direct_sha256=$(sha256sum "${matrix_env_file}")
      direct_sha256="${direct_sha256%% *}"
    fi
    if [[ "${direct_sha256}" != "${matrix_env_sha256}" ]]; then
      echo "ERROR: ${production_profile} frozen snapshot archive SHA-256 mismatch." >&2
      exit 1
    fi
  fi

  if [[ -n "${validation_backend}" ]]; then
    if ! production_environment_validation=$("${validation_backend}" validate_environment "${validation_scenario}" "${requested_env_file}"); then
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
  if [[ "${production_profile}" != "rebin_fine" && "${production_env_file_sha256}" != "${matrix_env_sha256}" ]]; then
    echo "ERROR: maintained validator SHA-256 differs from the required frozen archive identity." >&2
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
  case "${production_profile}" in
    run2_full|run3_full) expected_count=5 ;;
    run2_full_CR) expected_count=6 ;;
    run3_full_CR) expected_count=12 ;;
    rebin_fine) expected_count=6 ;;
  esac
  if (( ${#production_block_ids[@]} != expected_count )) \
    || (( ${#production_plan_year_exprs[@]} != expected_count )) \
    || (( ${#production_plan_category_sets[@]} != expected_count )) \
    || (( ${#production_plan_var_sets[@]} != expected_count )); then
    echo "ERROR: live ${production_profile} packing no longer matches its configured block contract." >&2
    exit 1
  fi
}

prepare_production_campaign() {
  local plan_directory
  local schema_version=4

  production_assert_live_plan
  production_git_commit=$(git -C "${repository_root}" rev-parse HEAD)
  production_state_path="${output_dir}/${production_state_filename}"
  if [[ "${dry_run}" == "true" ]]; then
    production_plan_file=$(mktemp "/tmp/${production_profile}_plan.XXXXXX")
  else
    if [[ "${profile_resume}" == "false" ]]; then
      mkdir -- "${output_dir}"
    fi
    plan_directory="${output_dir}"
    production_plan_file=$(mktemp "${plan_directory}/.${production_profile}_plan.XXXXXX")
  fi
  write_production_plan "${production_plan_file}"

  if [[ "${profile_resume}" == "true" ]]; then
    if [[ ! -f "${production_state_path}" ]]; then
      echo "ERROR: ${production_profile} --resume requires campaign state: ${production_state_path}" >&2
      exit 1
    fi
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
      "${production_topcoffea_git_commit}" \
      "${production_topcoffea_source_fingerprint}" \
      "${ttgamma_sample_role_policy}" \
      "${do_systs}" \
      "${do_np}" \
      "${production_region}" \
      "${production_np_mode}" \
      "${dry_run}"
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
      "${production_topcoffea_git_commit}" \
      "${production_topcoffea_source_fingerprint}" \
      "${ttgamma_sample_role_policy}" \
      "${do_systs}" \
      "${do_np}" \
      "${production_region}" \
      "${production_np_mode}"
  fi
}

prepare_production_sumw2_options() {
  if [[ "${production_profile}" == "rebin_fine" ]]; then
    return
  fi
  if [[ "${dry_run}" == "true" ]]; then
    production_sumw2_temporary_options=$(mktemp /tmp/run3_full_sumw2.XXXXXX.yml)
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
    echo "ERROR: run3_full sumw2 options do not match full_diagnostics." >&2
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

  if [[ "${production_profile}" != "rebin_fine" ]]; then
    cmd_ref+=(
      --snapshot
      --options "${production_sumw2_options_path}"
      -x work_queue
    )
  fi

  if [[ "${split_lep_flavor}" == "true" ]]; then
    cmd_ref+=(--split-lep-flavor)
  fi

  if [[ "${dry_run}" == "true" ]]; then
    cmd_ref+=(--dry-run)
  fi
}

native_wq_log_names=(debug.log tr.log stats.log tasks.log)

assert_native_wq_logs_clean() {
  local name
  for name in "${native_wq_log_names[@]}"; do
    if [[ -e "${native_wq_log_dir}/${name}" ]]; then
      echo "ERROR: unexpected generic Work Queue log requires ownership reconciliation before a new invocation: ${native_wq_log_dir}/${name}" >&2
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
  local archive_dir="${output_dir}/work_queue_logs/${production_component}/${block_id}/${stage}"
  local name source_path destination_path source_size destination_size

  if [[ -e "${archive_dir}" ]]; then
    echo "ERROR: native Work Queue archive destination already exists: ${archive_dir}" >&2
    return 1
  fi
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
  local exit_code
  local production_block
  local source_path
  local nonprompt_path
  local plan_output_tag

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

  echo "----------------------------------------"
  echo "Mode: CR"
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Variables: ${vars[*]}"
  echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
  echo "Campaign tag: ${campaign_tag}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
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

  print_command "${cmd[@]}"
  start_epoch=$(date +%s)

  if [[ "${dry_run}" == "true" ]]; then
    "${cmd[@]}"
    end_epoch=$(date +%s)
    duration_seconds=$((end_epoch - start_epoch))
    record_block_result \
      "DRY_RUN" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "0" "${duration_seconds}"
    echo
    return 0
  fi

  if production_any_output_exists "${production_block}" "${production_plan_file}"; then
    echo "ERROR: ${production_block} expected output path already exists; refusing ambiguous overwrite." >&2
    exit 1
  fi
  assert_native_wq_logs_clean
  production_state_tool mark \
    "${production_state_path}" "${production_block}" \
    source running none source_child_started "${cmd[@]}"

  # Keep child stdout/stderr attached directly to the caller. The conditional
  # invocation only captures the exit status so set -e does not terminate the
  # campaign when one production block fails or its Python child is killed.
  if run_production_source_child "${production_block}" "${source_path}" "${nonprompt_path}" "${cmd[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if (( exit_code != 0 )); then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source failed "${exit_code}" source_child_exit_nonzero "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    write_failure_snapshot "${production_block}" source
    record_block_result \
      "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${exit_code}" "${duration_seconds}"
    echo \
      "ERROR: ${year_expr} CR failed for ${cat_tag} / ${var_tag} with exit code ${exit_code}; continuing with the next block." \
      >&2
  elif [[ ! -s "${source_path}" || ! -s "${nonprompt_path}" ]]; then
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source failed "${exit_code}" source_exit_zero_invalid_stage_outputs "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    write_failure_snapshot "${production_block}" source
    record_block_result \
      "FAILED" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${exit_code}" "${duration_seconds}"
    echo "ERROR: ${production_block} returned zero without both expected outputs; continuing with the next block." >&2
  else
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      source ready "${exit_code}" source_child_exit_zero_expected_source_present "${duration_seconds}"
    production_state_tool mark \
      "${production_state_path}" "${production_block}" \
      nonprompt success "${exit_code}" inline_nonprompt_exit_zero_expected_output_present "${duration_seconds}"
    archive_native_wq_logs "${production_block}" source
    record_block_result \
      "SUCCESS" "CR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
      "${pkl_tag}" "${exit_code}" "${duration_seconds}"
    echo "${year_expr} CR done for ${cat_tag} / ${var_tag}"
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
    if [[ -s "${source_path}" && -s "${nonprompt_path}" ]]; then
      echo "----------------------------------------"
      echo "Skipping validated ${production_profile} block: ${production_block}"
      echo "Campaign state, source artifact, and separate nonprompt artifact are complete."
      echo "----------------------------------------"
      record_block_result \
        "SKIPPED" "SR" "${year_expr}" "${cats[*]}" "${vars[*]}" \
        "${pkl_tag}" "0" "0"
      return 0
    fi
    if [[ ! -s "${source_path}" ]]; then
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
      if [[ ! -s "${source_path}" ]]; then
        production_state_tool mark \
          "${production_state_path}" "${production_block}" \
          "source" "failed" "none" "source_ready_state_missing_expected_source"
        echo "ERROR: ${production_block} records a reusable source, but it is missing or empty." >&2
        exit 1
      fi
      source_reusable=true
      ;;
    planned|failed)
      if [[ -e "${source_path}" || -e "${nonprompt_path}" ]]; then
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

  if [[ -e "${nonprompt_path}" ]]; then
    echo "ERROR: ${production_block} is not successful, but its expected _np path already exists. Refusing ambiguous overwrite." >&2
    exit 1
  fi

  echo "----------------------------------------"
  echo "Mode: SR"
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
  local nonprompt_cmd=(
    python ./run_data_driven.py
    --input-pkl "${source_path}"
    --output-pkl "${nonprompt_path}"
  )

  start_epoch=$(date +%s)

  if [[ "${source_reusable}" == "false" ]]; then
    if [[ "${dry_run}" == "false" ]]; then
      assert_native_wq_logs_clean
      production_state_tool mark \
        "${production_state_path}" "${production_block}" \
        "source" "running" "none" "source_child_started" "${source_cmd[@]}"
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
      || { [[ "${production_np_mode}" == "separate" ]] && [[ -e "${nonprompt_path}" ]]; } \
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

  production_state_tool mark \
    "${production_state_path}" "${production_block}" \
    "nonprompt" "running" "none" "separate_nonprompt_child_started_after_source_exit" "${nonprompt_cmd[@]}"
  print_command "${nonprompt_cmd[@]}"
  if run_production_nonprompt_child \
    "${production_block}" "${source_path}" "${nonprompt_path}" "${nonprompt_cmd[@]}"; then
    nonprompt_exit_code=0
  else
    nonprompt_exit_code=$?
  fi

  end_epoch=$(date +%s)
  duration_seconds=$((end_epoch - start_epoch))

  if (( nonprompt_exit_code == 0 )) && [[ -s "${nonprompt_path}" ]]; then
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
  "rebin_fine SR category-to-variable mapping" \
  "${#rebin_fine_category_sets[@]}" \
  "${#rebin_fine_category_var_set_names[@]}"

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
  "${run3_full_category_var_set_names[@]}" \
  "${rebin_fine_category_var_set_names[@]}"; do
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
if [[ "${profile_resume}" == "true" ]]; then
  if [[ "${production_profile}" != "rebin_fine" ]]; then
    echo "environment_policy: state_frozen_exact_archive_integrity_plus_snapshot"
  else
    echo "environment_policy: state_frozen_strict_single_archive"
  fi
elif [[ "${production_profile}" != "rebin_fine" ]]; then
  echo "environment_policy: exact_frozen_archive_integrity_plus_snapshot"
elif [[ -n "${profile_env_file}" ]]; then
  echo "environment_policy: explicit_single_archive"
else
  echo "environment_policy: legacy_profile_policy"
fi
if [[ "${production_profile}" != "rebin_fine" ]]; then
  echo "sumw2_storage_mode: full_diagnostics"
fi
echo "campaign_state: ${output_dir}/${production_state_filename}"
print_var_sets "CR non-tau" "${cr_non_tau_var_sets[@]}"
print_var_sets "CR tau" "${cr_tau_var_sets[@]}"
print_var_sets "SR with ptz_wtau" "${sr_with_ptz_wtau_var_sets[@]}"
print_var_sets "SR off-Z" "${sr_offz_var_sets[@]}"
print_var_sets "SR on-Z/tau" "${sr_onz_tau_var_sets[@]}"
print_var_sets "SR forward" "${sr_fwd_var_sets[@]}"

if [[ "${production_profile}" == "rebin_fine" ]]; then
  echo "rebin_fine SR category blocks (used for both Run 2 and Run 3):"
  printf '  %s\n' "${rebin_fine_category_sets[@]}"
else
  echo "run3_full Run-3 SR category blocks:"
  printf '  %s\n' "${run3_full_category_sets[@]}"
fi

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

      for var_set in "${category_var_sets[@]}"; do
        run_sr_block "${year_expr}" "${var_set}" "${cats[@]}"
      done

      unset -n category_var_sets
    done

    unset -n active_category_sets
    unset -n active_category_var_set_names
  done
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
fi

final_process_exit_code=0
if [[ "${dry_run}" == "false" ]]; then
  IFS=$'\t' read -r final_campaign_classification final_process_exit_code < <(
    production_state_tool finalize "${production_state_path}"
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
