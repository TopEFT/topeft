#!/usr/bin/env bash
# User-run Run 3 datacard production wrapper.
# Matrix and input provenance are derived from DATACARD023 serialized evidence.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "ERROR: this script must be executed, not sourced." >&2
    return 2
fi

set -o pipefail

repo_root="/users/apiccine/work/correction-lib/topeft"
expected_branch="te/datacard_split_family_provenance"
validated_source_baseline="be2106b1afe1b8a351fe8b18431369294d96b626"
expected_python="/users/apiccine/work/miniconda3/envs/clib-env/bin/python"
default_output_root="/groups/klannon/apiccine/preappr_v9_260729/ANv9_cards_run3_yawen_matrix"
missing_parton_runtime_path="data/missing_parton/missing_parton_run3.root"
missing_parton_physical_path="${repo_root}/topeft/data/missing_parton/missing_parton_run3.root"
missing_parton_sha256="446a157adb43fe2e2d20e059bb54fb1cbc76b4e2c1d134cd26673bf02bb3b43c"
make_cards_path="analysis/topeft_run2/make_cards.py"
make_cards_sha256="5e2624fffcb0575159b2402c43e5329cb04608958e27a5020c59d83ecda6d120"
datacard_tools_path="topeft/modules/datacard_tools.py"
datacard_tools_sha256="e7f8024c6eab1f1d2bba9c7e48ad99ac1e75f69951379f5be044b96f842f9bd3"
run3_years=(2022 2022EE 2023 2023BPix)

mixed_pkl="/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_2l-2lss_1tau-2los_1tau-4l_njets-lj0pt-ptz-ptz_wtau-lt_np.pkl.gz"
mixed_pkl_sha256="6cc3f6134c9c42b078c01940ccf073ac43be3fc20e3972b55d5fa3f8a9c563c4"
offz_pkl="/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_m_offZ-3l_p_offZ_njets-lj0pt-ptz-lt_np.pkl.gz"
offz_pkl_sha256="52d87e8eab6267e16f21718d39d5229c9cab2e051cc23f41ae8fb1e2b918af92"
onz_tau_pkl="/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_onZ_tau_njets-lj0pt-ptz-lt_np.pkl.gz"
onz_tau_pkl_sha256="fd1df4fe9c6393be6e351a099c1305a8e42f375aadf9dfc50cffff8d8abc4812"
fwd_pkl="/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_fwd_njets-lj0pt-ptz-lt_np.pkl.gz"
fwd_pkl_sha256="2254c1fc1fec1c078ec6172914696e929a5c8904eab5720cd7af2245e4e1c21f"

output_root="${default_output_root}"
only_job=""
continue_on_failure=0
dry_run=0
list_jobs=0
current_job=""
interrupted=0
campaign_failure=0
declare -A job_status
declare -A job_txt_count
declare -A job_root_count
declare -a all_job_ids=(mixed_01 mixed_02 mixed_03 mixed_04 3l_offz_01 3l_offz_02 3l_onz_tau_01 3l_onz_tau_02 3l_fwd_01)

usage() {
    cat <<'USAGE'
Usage: ./run_make_cards_run3_yawen_matrix.sh [options]

User-run production wrapper for the DATACARD023-qualified Run 3 matrix.

Options:
  --output-root PATH       Output root (default: qualified Run 3 campaign path)
  --only JOB_ID            Run exactly one listed job
  --continue-on-failure    Continue after a failed or blocked job
  --dry-run                Print the nine command vectors without creating output directories
  --list-jobs              Print job IDs, inputs, discriminants, and expected card counts
  -h, --help               Show this help

Dry-run and list-jobs do not create the output root or any production files.
Actual execution refuses source/input provenance drift and never deletes or
overwrites existing job outputs. Existing exact-valid outputs are recorded as
already_success; invalid or partial outputs are blocked for manual review.
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output-root)
            if [[ "$#" -lt 2 ]]; then echo "ERROR: --output-root needs a path." >&2; exit 2; fi
            output_root="$2"; shift 2 ;;
        --only)
            if [[ "$#" -lt 2 ]]; then echo "ERROR: --only needs a job ID." >&2; exit 2; fi
            only_job="$2"; shift 2 ;;
        --continue-on-failure) continue_on_failure=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --list-jobs) list_jobs=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_hash() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local observed=""
    if [[ ! -r "$path" ]]; then
        echo "ERROR: unreadable ${label}: ${path}" >&2
        return 1
    fi
    observed="$(sha256_file "$path")"
    if [[ "$observed" != "$expected" ]]; then
        echo "ERROR: SHA256 mismatch for ${label}: expected ${expected}, got ${observed}" >&2
        return 1
    fi
    return 0
}

require_clean_critical_paths() {
    local critical_paths=(
        "$make_cards_path"
        "$datacard_tools_path"
        "topeft/data/missing_parton/missing_parton_run3.root"
    )
    if ! git diff --quiet -- "${critical_paths[@]}"; then
        echo "ERROR: production-critical tracked files have unstaged modifications." >&2
        return 1
    fi
    if ! git diff --cached --quiet -- "${critical_paths[@]}"; then
        echo "ERROR: production-critical tracked files have staged modifications." >&2
        return 1
    fi
    return 0
}

preflight_provenance() {
    local active_python=""
    if [[ ! -d "$repo_root/.git" ]]; then
        echo "ERROR: repository root is unavailable: ${repo_root}" >&2
        return 1
    fi
    if [[ "$(pwd -P)" != "$repo_root" ]]; then
        echo "ERROR: execute from ${repo_root}; current directory is $(pwd -P)" >&2
        return 1
    fi
    current_branch="$(git rev-parse --abbrev-ref HEAD)" || return 1
    current_head="$(git rev-parse HEAD)" || return 1
    if [[ "$current_branch" != "$expected_branch" ]]; then
        echo "ERROR: expected branch ${expected_branch}, got ${current_branch}" >&2
        return 1
    fi
    if ! git merge-base --is-ancestor "$validated_source_baseline" HEAD; then
        echo "ERROR: validated source baseline is not an ancestor of HEAD." >&2
        return 1
    fi
    require_clean_critical_paths || return 1
    require_hash "make_cards.py" "$make_cards_path" "$make_cards_sha256" || return 1
    require_hash "datacard_tools.py" "$datacard_tools_path" "$datacard_tools_sha256" || return 1
    require_hash "Run 3 missing-parton payload" "$missing_parton_physical_path" "$missing_parton_sha256" || return 1
    require_hash "mixed input PKL" "$mixed_pkl" "$mixed_pkl_sha256" || return 1
    require_hash "3l offZ input PKL" "$offz_pkl" "$offz_pkl_sha256" || return 1
    require_hash "3l onZ tau input PKL" "$onz_tau_pkl" "$onz_tau_pkl_sha256" || return 1
    require_hash "3l forward input PKL" "$fwd_pkl" "$fwd_pkl_sha256" || return 1
    active_python="$(command -v python)"
    if [[ "$active_python" != "$expected_python" ]]; then
        echo "ERROR: activate clib-env before production; python resolves to ${active_python:-not found}" >&2
        return 1
    fi
    return 0
}

job_is_known() {
    local candidate="$1"
    local known=""
    for known in "${all_job_ids[@]}"; do
        [[ "$known" == "$candidate" ]] && return 0
    done
    return 1
}

print_command() {
    local item=""
    printf 'python'
    for item in "$@"; do
        printf ' %q' "$item"
    done
    printf '\n'
}

strip_selector_anchors() {
    local selector="$1"
    selector="${selector#^}"
    selector="${selector%\$}"
    printf '%s\n' "$selector"
}

build_expected_manifests() {
    local job_id="$1"
    local discriminant="$2"
    shift 2
    local selector=""
    local category=""
    local expected_dir="${expected_root}/${job_id}"
    mkdir -p "$expected_dir" || return 1
    : > "${expected_dir}/txt.txt"
    : > "${expected_dir}/root.txt"
    for selector in "$@"; do
        category="$(strip_selector_anchors "$selector")"
        printf '%s\n' "ttx_multileptons-${category}_${discriminant}.txt" >> "${expected_dir}/txt.txt"
        printf '%s\n' "ttx_multileptons-${category}_${discriminant}.root" >> "${expected_dir}/root.txt"
    done
}

validate_job_output() {
    local job_id="$1"
    local output_dir="$2"
    local expected_dir="${expected_root}/${job_id}"
    local observed_dir="${observed_root}/${job_id}"
    local file=""
    local validation_rc=0
    mkdir -p "$observed_dir" || return 1
    if [[ ! -d "$output_dir" ]]; then
        echo "missing output directory: ${output_dir}" > "${observed_dir}/validation.txt"
        return 1
    fi
    find "$output_dir" -maxdepth 1 -type f -name 'ttx_multileptons-*.txt' -printf '%f\n' | sort > "${observed_dir}/txt.txt"
    find "$output_dir" -maxdepth 1 -type f -name 'ttx_multileptons-*.root' -printf '%f\n' | sort > "${observed_dir}/root.txt"
    diff -u "${expected_dir}/txt.txt" "${observed_dir}/txt.txt" > "${observed_dir}/txt.diff"
    [[ "$?" -eq 0 ]] || validation_rc=1
    diff -u "${expected_dir}/root.txt" "${observed_dir}/root.txt" > "${observed_dir}/root.diff"
    [[ "$?" -eq 0 ]] || validation_rc=1
    while IFS= read -r file; do
        if ! grep -Eq '^\* autoMCStats 10$' "${output_dir}/${file}"; then
            echo "missing autoMCStats 10: ${file}" >> "${observed_dir}/validation.txt"
            validation_rc=1
        fi
    done < "${expected_dir}/txt.txt"
    if [[ "$validation_rc" -eq 0 ]]; then
        wc -l < "${observed_dir}/txt.txt" > "${observed_dir}/txt_count.txt"
        wc -l < "${observed_dir}/root.txt" > "${observed_dir}/root_count.txt"
        return 0
    fi
    return 1
}

record_job_status() {
    local job_id="$1"
    local status="$2"
    local txt_count="$3"
    local root_count="$4"
    job_status["$job_id"]="$status"
    job_txt_count["$job_id"]="$txt_count"
    job_root_count["$job_id"]="$root_count"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$status" "$txt_count" "$root_count" > "${status_root}/${job_id}.tsv"
    case "$status" in
        failed_*|blocked_*) campaign_failure=1 ;;
    esac
}

run_job() {
    local job_id="$1"
    local sibling_block="$2"
    local input_pkl="$3"
    local input_sha256="$4"
    local discriminant="$5"
    local expected_count="$6"
    shift 6
    local selectors=("$@")
    local output_dir="${output_root}/${job_id}"
    local log_file="${log_root}/${job_id}.log"
    local start_time=""
    local end_time=""
    local rc=0
    local cmd=(
        "$make_cards_path" "$input_pkl"
        --out-dir "$output_dir"
        --var-lst "$discriminant"
        --ch-lst "${selectors[@]}"
        --do-nuisance
        --do-mc-stat
        --skip-selected-wcs-check
        --year-coverage-policy warn
        --year "${run3_years[@]}"
        --miss-parton-file "$missing_parton_runtime_path"
        --sr-registry ALL_CH_LST_SR
    )

    if [[ "$list_jobs" -eq 1 ]]; then
        printf 'JOB id=%s sibling=%s discriminant=%s expected_txt=%s expected_root=%s input=%s\n' \
            "$job_id" "$sibling_block" "$discriminant" "$expected_count" "$expected_count" "$input_pkl"
        return 0
    fi

    if [[ -n "$only_job" && "$job_id" != "$only_job" ]]; then
        record_job_status "$job_id" "pending_not_selected" 0 0
        return 0
    fi

    if [[ "$campaign_failure" -ne 0 && "$continue_on_failure" -eq 0 && "$dry_run" -eq 0 ]]; then
        return 1
    fi

    if [[ "$dry_run" -eq 1 ]]; then
        printf 'DRY_RUN_JOB id=%s sibling=%s discriminant=%s expected_txt=%s expected_root=%s input_sha256=%s\n' \
            "$job_id" "$sibling_block" "$discriminant" "$expected_count" "$expected_count" "$input_sha256"
        printf 'DRY_RUN_COMMAND id=%s ' "$job_id"
        print_command "${cmd[@]}"
        return 0
    fi

    build_expected_manifests "$job_id" "$discriminant" "${selectors[@]}" || {
        record_job_status "$job_id" "failed_manifest_setup" 0 0
        return 1
    }

    if [[ -e "$output_dir" ]]; then
        if validate_job_output "$job_id" "$output_dir"; then
            record_job_status "$job_id" "already_success" "$expected_count" "$expected_count"
            echo "INFO: ${job_id} already has an exact valid output; preserving it."
            return 0
        fi
        record_job_status "$job_id" "blocked_existing_output" 0 0
        echo "ERROR: ${job_id} output exists but is invalid or partial: ${output_dir}" >&2
        return 1
    fi

    current_job="$job_id"
    start_time="$(date -Is)"
    printf 'START %s %s\n' "$start_time" "$job_id" | tee "$log_file"
    print_command "${cmd[@]}" | tee -a "$log_file"
    python "${cmd[@]}" 2>&1 | tee -a "$log_file"
    rc="${PIPESTATUS[0]}"
    end_time="$(date -Is)"
    if [[ "$rc" -ne 0 ]]; then
        printf 'END %s rc=%s\n' "$end_time" "$rc" | tee -a "$log_file"
        record_job_status "$job_id" "failed_command_rc_${rc}" 0 0
        current_job=""
        return 1
    fi
    if ! validate_job_output "$job_id" "$output_dir"; then
        record_job_status "$job_id" "failed_manifest_validation" 0 0
        current_job=""
        return 1
    fi
    record_job_status "$job_id" "success" "$expected_count" "$expected_count"
    current_job=""
    return 0
}

print_summary() {
    local summary_file="${output_root}/_summary.txt"
    local job_id=""
    local status=""
    local txt_count=0
    local root_count=0
    local successful_jobs=0
    local failed_jobs=0
    local blocked_jobs=0
    local interrupted_jobs=0
    local pending_jobs=0
    local validated_txt_total=0
    local validated_root_total=0
    local overall_status="FAILED_OR_INCOMPLETE"
    {
        echo "generated_at=$(date -Is)"
        echo "output_root=${output_root}"
        echo "current_branch=${current_branch}"
        echo "current_head=${current_head}"
        echo "validated_source_baseline=${validated_source_baseline}"
        echo "python_executable=$(command -v python)"
        echo "run3_years=${run3_years[*]}"
        echo "missing_parton_runtime_path=${missing_parton_runtime_path}"
        echo "missing_parton_sha256=${missing_parton_sha256}"
        echo "production_critical_file_sha256 ${make_cards_path} ${make_cards_sha256}"
        echo "production_critical_file_sha256 ${datacard_tools_path} ${datacard_tools_sha256}"
        echo "production_critical_file_sha256 topeft/data/missing_parton/missing_parton_run3.root ${missing_parton_sha256}"
        echo "input_pkl_path_and_sha256 mixed ${mixed_pkl} ${mixed_pkl_sha256}"
        echo "input_pkl_path_and_sha256 3l_offz ${offz_pkl} ${offz_pkl_sha256}"
        echo "input_pkl_path_and_sha256 3l_onz_tau ${onz_tau_pkl} ${onz_tau_pkl_sha256}"
        echo "input_pkl_path_and_sha256 3l_fwd ${fwd_pkl} ${fwd_pkl_sha256}"
        echo "job_id status txt root"
        for job_id in "${all_job_ids[@]}"; do
            status="${job_status[$job_id]:-pending}"
            txt_count="${job_txt_count[$job_id]:-0}"
            root_count="${job_root_count[$job_id]:-0}"
            printf '%s %s %s %s\n' "$job_id" "$status" "$txt_count" "$root_count"
            case "$status" in
                success|already_success)
                    successful_jobs=$((successful_jobs + 1))
                    validated_txt_total=$((validated_txt_total + txt_count))
                    validated_root_total=$((validated_root_total + root_count)) ;;
                failed_*) failed_jobs=$((failed_jobs + 1)) ;;
                blocked_*) blocked_jobs=$((blocked_jobs + 1)) ;;
                interrupted*) interrupted_jobs=$((interrupted_jobs + 1)) ;;
                *) pending_jobs=$((pending_jobs + 1)) ;;
            esac
        done
        [[ "$interrupted" -eq 1 ]] && interrupted_jobs=$((interrupted_jobs + 1))
        if [[ "$successful_jobs" -eq 9 && "$validated_txt_total" -eq 129 && "$validated_root_total" -eq 129 ]]; then
            overall_status="SUCCESS_129_OF_129"
        elif [[ "$interrupted" -eq 1 ]]; then
            overall_status="INTERRUPTED_PRESERVED"
        fi
        echo "successful_jobs=${successful_jobs}"
        echo "failed_jobs=${failed_jobs}"
        echo "blocked_jobs=${blocked_jobs}"
        echo "interrupted_jobs=${interrupted_jobs}"
        echo "pending_jobs=${pending_jobs}"
        echo "validated_success_txt_total=${validated_txt_total}"
        echo "validated_success_root_total=${validated_root_total}"
        echo "overall_status=${overall_status}"
    } | tee "$summary_file"
}

on_interrupt() {
    interrupted=1
    if [[ -n "$current_job" && -d "$status_root" ]]; then
        record_job_status "$current_job" "interrupted_preserved_partial_output" 0 0
    fi
    echo "INTERRUPTED: preserving all partial output; no automatic retry is attempted." >&2
    if [[ -d "$output_root" ]]; then
        print_summary
    fi
    exit 130
}

run_all_jobs() {
  run_job "mixed_01" "mixed" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_2l-2lss_1tau-2los_1tau-4l_njets-lj0pt-ptz-ptz_wtau-lt_np.pkl.gz" "6cc3f6134c9c42b078c01940ccf073ac43be3fc20e3972b55d5fa3f8a9c563c4" "lj0pt" 27 "^2lss_4t_m_4j\$" "^2lss_4t_m_5j\$" "^2lss_4t_m_6j\$" "^2lss_4t_m_7j\$" "^2lss_4t_p_4j\$" "^2lss_4t_p_5j\$" "^2lss_4t_p_6j\$" "^2lss_4t_p_7j\$" "^2lss_m_1tau_offZ_3j\$" "^2lss_m_1tau_offZ_4j\$" "^2lss_m_1tau_offZ_5j\$" "^2lss_m_1tau_offZ_6j\$" "^2lss_m_4j\$" "^2lss_m_5j\$" "^2lss_m_6j\$" "^2lss_m_7j\$" "^2lss_p_1tau_offZ_3j\$" "^2lss_p_1tau_offZ_4j\$" "^2lss_p_1tau_offZ_5j\$" "^2lss_p_1tau_offZ_6j\$" "^2lss_p_4j\$" "^2lss_p_5j\$" "^2lss_p_6j\$" "^2lss_p_7j\$" "^4l_2j\$" "^4l_3j\$" "^4l_4j\$"
  run_job "mixed_02" "mixed" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_2l-2lss_1tau-2los_1tau-4l_njets-lj0pt-ptz-ptz_wtau-lt_np.pkl.gz" "6cc3f6134c9c42b078c01940ccf073ac43be3fc20e3972b55d5fa3f8a9c563c4" "lt" 8 "^2lss_fwd_m_4j\$" "^2lss_fwd_m_5j\$" "^2lss_fwd_m_6j\$" "^2lss_fwd_m_7j\$" "^2lss_fwd_p_4j\$" "^2lss_fwd_p_5j\$" "^2lss_fwd_p_6j\$" "^2lss_fwd_p_7j\$"
  run_job "mixed_03" "mixed" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_2l-2lss_1tau-2los_1tau-4l_njets-lj0pt-ptz-ptz_wtau-lt_np.pkl.gz" "6cc3f6134c9c42b078c01940ccf073ac43be3fc20e3972b55d5fa3f8a9c563c4" "ptz" 1 "^2los_onZ_1tau_3j\$"
  run_job "mixed_04" "mixed" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_2l-2lss_1tau-2los_1tau-4l_njets-lj0pt-ptz-ptz_wtau-lt_np.pkl.gz" "6cc3f6134c9c42b078c01940ccf073ac43be3fc20e3972b55d5fa3f8a9c563c4" "ptz_wtau" 8 "^2lss_m_1tau_onZ_3j\$" "^2lss_m_1tau_onZ_4j\$" "^2lss_m_1tau_onZ_5j\$" "^2lss_m_1tau_onZ_6j\$" "^2lss_p_1tau_onZ_3j\$" "^2lss_p_1tau_onZ_4j\$" "^2lss_p_1tau_onZ_5j\$" "^2lss_p_1tau_onZ_6j\$"
  # The recorded off-Z input below predates the ptll schema. Replace it and
  # its checksum with the new canonical artifact before running this matrix;
  # the ptll lookup intentionally has no ptz fallback.
  run_job "3l_offz_01" "3l_offz" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_m_offZ-3l_p_offZ_njets-lj0pt-ptz-lt_np.pkl.gz" "52d87e8eab6267e16f21718d39d5229c9cab2e051cc23f41ae8fb1e2b918af92" "lj0pt" 16 "^3l_m_offZ_none_1b_2j\$" "^3l_m_offZ_none_1b_3j\$" "^3l_m_offZ_none_1b_4j\$" "^3l_m_offZ_none_1b_5j\$" "^3l_m_offZ_none_2b_2j\$" "^3l_m_offZ_none_2b_3j\$" "^3l_m_offZ_none_2b_4j\$" "^3l_m_offZ_none_2b_5j\$" "^3l_p_offZ_none_1b_2j\$" "^3l_p_offZ_none_1b_3j\$" "^3l_p_offZ_none_1b_4j\$" "^3l_p_offZ_none_1b_5j\$" "^3l_p_offZ_none_2b_2j\$" "^3l_p_offZ_none_2b_3j\$" "^3l_p_offZ_none_2b_4j\$" "^3l_p_offZ_none_2b_5j\$"
  run_job "3l_offz_02" "3l_offz" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_m_offZ-3l_p_offZ_njets-lj0pt-ptz-lt_np.pkl.gz" "52d87e8eab6267e16f21718d39d5229c9cab2e051cc23f41ae8fb1e2b918af92" "ptll" 32 "^3l_m_offZ_high_1b_2j\$" "^3l_m_offZ_high_1b_3j\$" "^3l_m_offZ_high_1b_4j\$" "^3l_m_offZ_high_1b_5j\$" "^3l_m_offZ_high_2b_2j\$" "^3l_m_offZ_high_2b_3j\$" "^3l_m_offZ_high_2b_4j\$" "^3l_m_offZ_high_2b_5j\$" "^3l_m_offZ_low_1b_2j\$" "^3l_m_offZ_low_1b_3j\$" "^3l_m_offZ_low_1b_4j\$" "^3l_m_offZ_low_1b_5j\$" "^3l_m_offZ_low_2b_2j\$" "^3l_m_offZ_low_2b_3j\$" "^3l_m_offZ_low_2b_4j\$" "^3l_m_offZ_low_2b_5j\$" "^3l_p_offZ_high_1b_2j\$" "^3l_p_offZ_high_1b_3j\$" "^3l_p_offZ_high_1b_4j\$" "^3l_p_offZ_high_1b_5j\$" "^3l_p_offZ_high_2b_2j\$" "^3l_p_offZ_high_2b_3j\$" "^3l_p_offZ_high_2b_4j\$" "^3l_p_offZ_high_2b_5j\$" "^3l_p_offZ_low_1b_2j\$" "^3l_p_offZ_low_1b_3j\$" "^3l_p_offZ_low_1b_4j\$" "^3l_p_offZ_low_1b_5j\$" "^3l_p_offZ_low_2b_2j\$" "^3l_p_offZ_low_2b_3j\$" "^3l_p_offZ_low_2b_4j\$" "^3l_p_offZ_low_2b_5j\$"
  run_job "3l_onz_tau_01" "3l_onz_tau" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_onZ_tau_njets-lj0pt-ptz-lt_np.pkl.gz" "fd1df4fe9c6393be6e351a099c1305a8e42f375aadf9dfc50cffff8d8abc4812" "lj0pt" 10 "^3l_1tau_1b_2j\$" "^3l_1tau_1b_3j\$" "^3l_1tau_1b_4j\$" "^3l_1tau_1b_5j\$" "^3l_1tau_2b_2j\$" "^3l_1tau_2b_3j\$" "^3l_1tau_2b_4j\$" "^3l_1tau_2b_5j\$" "^3l_onZ_2b_2j\$" "^3l_onZ_2b_3j\$"
  run_job "3l_onz_tau_02" "3l_onz_tau" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_onZ_tau_njets-lj0pt-ptz-lt_np.pkl.gz" "fd1df4fe9c6393be6e351a099c1305a8e42f375aadf9dfc50cffff8d8abc4812" "ptz" 6 "^3l_onZ_1b_2j\$" "^3l_onZ_1b_3j\$" "^3l_onZ_1b_4j\$" "^3l_onZ_1b_5j\$" "^3l_onZ_2b_4j\$" "^3l_onZ_2b_5j\$"
  run_job "3l_fwd_01" "3l_fwd" "/groups/klannon/apiccine/preappr_v9_260729/2022-2022EE-2023-2023BPixSRs_ANv9_3l_fwd_njets-lj0pt-ptz-lt_np.pkl.gz" "2254c1fc1fec1c078ec6172914696e929a5c8904eab5720cd7af2245e4e1c21f" "lt" 21 "^3l_m_offZ_1b_fwd_1j\$" "^3l_m_offZ_1b_fwd_2j\$" "^3l_m_offZ_1b_fwd_3j\$" "^3l_m_offZ_1b_fwd_4j\$" "^3l_m_offZ_2b_fwd_2j\$" "^3l_m_offZ_2b_fwd_3j\$" "^3l_m_offZ_2b_fwd_4j\$" "^3l_onZ_1b_fwd_1j\$" "^3l_onZ_1b_fwd_2j\$" "^3l_onZ_1b_fwd_3j\$" "^3l_onZ_1b_fwd_4j\$" "^3l_onZ_2b_fwd_2j\$" "^3l_onZ_2b_fwd_3j\$" "^3l_onZ_2b_fwd_4j\$" "^3l_p_offZ_1b_fwd_1j\$" "^3l_p_offZ_1b_fwd_2j\$" "^3l_p_offZ_1b_fwd_3j\$" "^3l_p_offZ_1b_fwd_4j\$" "^3l_p_offZ_2b_fwd_2j\$" "^3l_p_offZ_2b_fwd_3j\$" "^3l_p_offZ_2b_fwd_4j\$"
}

if [[ "$list_jobs" -eq 1 ]]; then
    echo "job_id sibling discriminant expected_txt expected_root input"
    run_all_jobs
    exit 0
fi

if [[ -n "$only_job" ]] && ! job_is_known "$only_job"; then
    echo "ERROR: unknown --only job: ${only_job}" >&2
    exit 2
fi

if [[ "$dry_run" -eq 1 ]]; then
    echo "DRY_RUN no output-root creation or production command execution."
    run_all_jobs
    exit 0
fi

if ! preflight_provenance; then
    echo "ERROR: provenance preflight failed; no output directory was created." >&2
    exit 1
fi

log_root="${output_root}/_logs"
status_root="${output_root}/_status"
expected_root="${output_root}/_expected"
observed_root="${output_root}/_observed"
mkdir -p "$output_root" "$log_root" "$status_root" "$expected_root" "$observed_root" || {
    echo "ERROR: cannot create production bookkeeping directories under ${output_root}" >&2
    exit 1
}
trap on_interrupt INT TERM

run_all_jobs
campaign_rc=$?
if [[ "$campaign_failure" -ne 0 ]]; then
    campaign_rc=1
fi
print_summary
if [[ "$campaign_rc" -ne 0 ]]; then
    exit "$campaign_rc"
fi
exit 0
