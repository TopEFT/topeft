#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
driver_path="${script_dir}/$(basename -- "${BASH_SOURCE[0]}")"
topeft_root="$(cd -- "${script_dir}/../.." && pwd -P)"
workspace_root="$(cd -- "${topeft_root}/.." && pwd -P)"
environment_wrapper="${workspace_root}/codex-run.sh"
python_env="${PYTHON_ENV:-/users/apiccine/work/miniconda3/envs/clib-env/bin/python}"

run2_periods=(UL16 UL16APV UL17 UL18)
executor="${MISSING_PARTON_EXECUTOR:-work_queue}"
environment_archive_dir_default="${script_dir}/topeft-envs"
environment_archive_pattern='env_*.tar.gz'

central_cfg="${topeft_root}/input_samples/cfgs/missing_parton_top22006_central_tzq_NDSkim.cfg"
private_cfg="${topeft_root}/input_samples/cfgs/missing_parton_top22006_private_tllq_NDSkim.cfg"
channel_list="${topeft_root}/topeft/channels/ch_lst.json"
analysis_processor="${script_dir}/analysis_processor.py"
run_analysis="${script_dir}/run_analysis.py"

output_root=""
resume_mode=false
print_plan=false
attempt=1
campaign_type="run2_top22006_missing_parton"
campaign_mutation_allowed=false

campaign_metadata=""
output_contract=""
status_file=""
state_history=""
retry_history=""
validation_log=""
execution_commands=""
cleanup_log=""
campaign_log=""

top22006_category_groups=()

usage() {
    cat <<'EOF'
Usage:
  run_run2_missing_parton_pkls_overnight.sh [--resume] [--print-plan] OUTPUT_ROOT

Options:
  --resume      Continue an interrupted campaign created by this driver.
  --print-plan  Print the validated two-role plan without creating OUTPUT_ROOT
                or launching analysis.
  -h, --help    Show this help text.

Environment:
  MISSING_PARTON_EXECUTOR  futures, work_queue, or taskvine (default: work_queue)
  PYTHON_ENV               Python executable used through codex-run.sh
  TOPEFT_ENVS_DIR          Must resolve exactly to SCRIPT_DIR/topeft-envs
EOF
}

timestamp_utc() {
    date -u +%Y%m%dT%H%M%SZ
}

timestamp_iso() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

quote_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_python() {
    "${environment_wrapper}" /bin/bash --noprofile --norc -c \
        'cd "$1"; shift; exec "$@"' \
        run2_missing_parton_driver "${topeft_root}" "${python_env}" "$@"
}

write_status() {
    local state="$1"
    local detail="${2:-}"
    [[ "${campaign_mutation_allowed}" == true ]] || return 0
    [[ -n "${status_file}" && -d "${output_root:-/nonexistent}" ]] || return 0
    local temporary_status="${status_file}.tmp.$$"
    {
        printf 'campaign_type=%s\n' "${campaign_type}"
        printf 'state=%s\n' "${state}"
        printf 'timestamp=%s\n' "$(timestamp_iso)"
        printf 'attempt=%s\n' "${attempt}"
        printf 'detail=%s\n' "${detail}"
    } > "${temporary_status}"
    mv -f -- "${temporary_status}" "${status_file}"
    printf '%s\t%s\t%s\t%s\n' "$(timestamp_iso)" "${attempt}" "${state}" "${detail}" >> "${state_history}"
}

record_retry() {
    printf '%s\t%s\t%s\n' "$(timestamp_iso)" "${attempt}" "resume_requested" >> "${retry_history}"
}

on_error() {
    local exit_code=$?
    local line_number="${1:-unknown}"
    write_status failed "line=${line_number};exit_code=${exit_code}"
    exit "${exit_code}"
}

on_signal() {
    local signal_name="$1"
    write_status interrupted "signal=${signal_name}"
    exit 130
}

trap 'on_error "$LINENO"' ERR
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

parse_args() {
    while (($#)); do
        case "$1" in
            --resume)
                resume_mode=true
                ;;
            --print-plan)
                print_plan=true
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                die "unsupported option: $1"
                ;;
            *)
                [[ -z "${output_root}" ]] || die "only one OUTPUT_ROOT may be specified"
                output_root="$1"
                ;;
        esac
        shift
    done

    if [[ "${resume_mode}" == true && "${print_plan}" == true ]]; then
        die "--resume and --print-plan are mutually exclusive"
    fi
    [[ -n "${output_root}" ]] || die "OUTPUT_ROOT is required"
    [[ "${output_root}" == /* ]] || die "OUTPUT_ROOT must be absolute"
    case "${executor}" in
        futures|work_queue|taskvine) ;;
        *) die "MISSING_PARTON_EXECUTOR must be futures, work_queue, or taskvine" ;;
    esac
}

validate_output_root() {
    local requested_root="${output_root}"
    local canonical_root

    [[ ! -L "${requested_root}" ]] || die "OUTPUT_ROOT must not be a symlink"
    canonical_root="$(realpath -m -- "${requested_root}")"
    [[ -n "${canonical_root}" ]] || die "failed to canonicalize OUTPUT_ROOT"

    case "${canonical_root}" in
        /|"${topeft_root}"|"${workspace_root}"|"${script_dir}"|"${HOME:-/definitely-not-home}")
            die "unsafe OUTPUT_ROOT: ${canonical_root}"
            ;;
        "${workspace_root}"/*|"${topeft_root}"/*|"${HOME:-/definitely-not-home}"/*)
            die "OUTPUT_ROOT must not be inside the workspace, repository, or home directory"
            ;;
    esac

    output_root="${canonical_root}"
}

assert_operational_prerequisites() {
    local required_command
    for required_command in git sha256sum gzip flock find realpath awk sed mktemp sort; do
        command -v "${required_command}" >/dev/null 2>&1 || die "required command not found: ${required_command}"
    done

    [[ -x "${environment_wrapper}" ]] || die "environment wrapper is not executable: ${environment_wrapper}"
    [[ -x "${python_env}" ]] || die "Python executable is not executable: ${python_env}"
    [[ -f "${run_analysis}" ]] || die "missing run_analysis.py: ${run_analysis}"
    [[ -f "${analysis_processor}" ]] || die "missing analysis_processor.py: ${analysis_processor}"
    [[ -f "${central_cfg}" ]] || die "missing central cfg: ${central_cfg}"
    [[ -f "${private_cfg}" ]] || die "missing private cfg: ${private_cfg}"
    [[ -f "${channel_list}" ]] || die "missing channel list: ${channel_list}"
}

campaign_paths() {
    campaign_metadata="${output_root}/campaign_metadata.txt"
    output_contract="${output_root}/output_contract.tsv"
    status_file="${output_root}/status.txt"
    state_history="${output_root}/state_history.tsv"
    retry_history="${output_root}/retry_history.tsv"
    validation_log="${output_root}/validation.tsv"
    execution_commands="${output_root}/execution_commands.tsv"
    cleanup_log="${output_root}/environment_archive_cleanup.tsv"
    campaign_log="${output_root}/logs/campaign.log"
}

acquire_campaign_lock() {
    mkdir -p -- "${output_root}/.campaign_lock"
    exec 9>"${output_root}/.campaign_lock/active.lock"
    flock -n 9 || die "another driver process holds the campaign lock"
    campaign_mutation_allowed=true
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

source_branch() {
    git -C "${topeft_root}" branch --show-current
}

source_commit() {
    git -C "${topeft_root}" rev-parse HEAD
}

write_campaign_metadata() {
    cat > "${campaign_metadata}" <<EOF
campaign_type=${campaign_type}
created_at=$(timestamp_iso)
driver_path=${driver_path}
driver_sha256=$(file_sha256 "${driver_path}")
source_branch=$(source_branch)
source_commit=$(source_commit)
analysis_processor_sha256=$(file_sha256 "${analysis_processor}")
run_analysis_sha256=$(file_sha256 "${run_analysis}")
channel_list_sha256=$(file_sha256 "${channel_list}")
central_cfg=${central_cfg}
central_cfg_sha256=$(file_sha256 "${central_cfg}")
private_cfg=${private_cfg}
private_cfg_sha256=$(file_sha256 "${private_cfg}")
periods=${run2_periods[*]}
executor=${executor}
category_source=${channel_list}:TOP22_006_CH_LST_SR
histogram_cli_contract=--hist-list njets
expected_histogram_contents=njets njets_sumw2
environment_archive_dir=${environment_archive_dir_default}
environment_archive_pattern=${environment_archive_pattern}
environment_archive_cleanup_policy=before_each_run_analysis_command
work_queue_runtime_log_directory=${script_dir}
work_queue_runtime_log_paths=${script_dir}/debug.log ${script_dir}/tr.log ${script_dir}/stats.log ${script_dir}/tasks.log
output_root=${output_root}
EOF
}

metadata_value() {
    local key="$1"
    awk -F= -v requested_key="${key}" '$1 == requested_key {sub(/^[^=]*=/, ""); print; exit}' "${campaign_metadata}"
}

require_resume_regular_file() {
    local path="$1"
    local description="$2"
    [[ ! -L "${path}" && -f "${path}" ]] || die "${description} must be a regular non-symlink file"
}

verify_resume_metadata() {
    require_resume_regular_file "${campaign_metadata}" "resume campaign_metadata.txt"
    [[ "$(metadata_value campaign_type)" == "${campaign_type}" ]] || die "resume campaign type mismatch"
    [[ "$(metadata_value driver_path)" == "${driver_path}" ]] || die "resume driver path mismatch"
    [[ "$(metadata_value driver_sha256)" == "$(file_sha256 "${driver_path}")" ]] || die "resume driver hash mismatch"
    [[ "$(metadata_value source_branch)" == "$(source_branch)" ]] || die "resume source branch mismatch"
    [[ "$(metadata_value source_commit)" == "$(source_commit)" ]] || die "resume source commit mismatch"
    [[ "$(metadata_value analysis_processor_sha256)" == "$(file_sha256 "${analysis_processor}")" ]] || die "resume processor hash mismatch"
    [[ "$(metadata_value run_analysis_sha256)" == "$(file_sha256 "${run_analysis}")" ]] || die "resume run_analysis.py hash mismatch"
    [[ "$(metadata_value channel_list_sha256)" == "$(file_sha256 "${channel_list}")" ]] || die "resume channel-list hash mismatch"
    [[ "$(metadata_value central_cfg_sha256)" == "$(file_sha256 "${central_cfg}")" ]] || die "resume central cfg hash mismatch"
    [[ "$(metadata_value private_cfg_sha256)" == "$(file_sha256 "${private_cfg}")" ]] || die "resume private cfg hash mismatch"
    [[ "$(metadata_value periods)" == "${run2_periods[*]}" ]] || die "resume period inventory mismatch"
    [[ "$(metadata_value executor)" == "${executor}" ]] || die "resume executor mismatch"
}

initialize_new_campaign() {
    if [[ -e "${output_root}" ]]; then
        [[ -d "${output_root}" ]] || die "OUTPUT_ROOT exists and is not a directory"
        [[ -z "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
            die "refusing to use a populated OUTPUT_ROOT without --resume"
    else
        mkdir -p -- "${output_root}"
    fi

    mkdir -p -- "${output_root}/top22006" "${output_root}/logs" "${output_root}/invalid_outputs"
    campaign_paths
    acquire_campaign_lock

    printf 'timestamp\tattempt\tstate\tdetail\n' > "${state_history}"
    printf 'timestamp\tattempt\taction\n' > "${retry_history}"
    printf 'timestamp\tphase\trole\tchunk\tpath\tresult\tsize\tsha256\tdetail\n' > "${validation_log}"
    printf 'timestamp\tattempt\tphase\trole\tchunk\tcommand\n' > "${execution_commands}"
    printf 'timestamp\tattempt\tcontext\taction\tpath\tdetail\n' > "${cleanup_log}"
    write_campaign_metadata
    write_status initialized "fresh_campaign"
}

initialize_existing_campaign() {
    [[ -d "${output_root}" ]] || die "--resume requires an existing OUTPUT_ROOT"
    campaign_paths
    require_resume_regular_file "${campaign_metadata}" "resume campaign_metadata.txt"
    require_resume_regular_file "${status_file}" "resume status.txt"
    require_resume_regular_file "${output_contract}" "resume output_contract.tsv"
    verify_resume_metadata
    verify_resume_contract
    ! grep -qx 'state=success' "${status_file}" || die "campaign already completed successfully"
    acquire_campaign_lock

    local previous_attempts=0
    if [[ -f "${retry_history}" ]]; then
        previous_attempts="$(awk 'NR > 1 {count++} END {print count + 0}' "${retry_history}")"
    fi
    attempt=$((previous_attempts + 2))
    record_retry
    write_status resumed "resume_requested"
}

read_json_keys() {
    local json_path="$1"
    local object_key="$2"
    run_python - "${json_path}" "${object_key}" <<'PY'
import json
import sys

json_path, object_key = sys.argv[1:]
with open(json_path, encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload[object_key]
if not isinstance(value, dict) or not value:
    raise SystemExit(f"{object_key} must be a nonempty object")
for key in value:
    print(key)
PY
}

load_category_groups() {
    mapfile -t top22006_category_groups < <(read_json_keys "${channel_list}" TOP22_006_CH_LST_SR)
    ((${#top22006_category_groups[@]} > 0)) || die "TOP22_006_CH_LST_SR is empty"
}

central_output_path() {
    printf '%s/top22006/missing_parton_top22006_run2_central_tzq_njets.pkl.gz\n' "${output_root}"
}

private_output_path() {
    printf '%s/top22006/missing_parton_top22006_run2_private_tllq_njets.pkl.gz\n' "${output_root}"
}

write_output_contract_to() {
    local contract_path="$1"
    {
        printf 'class\tphase\trole\tchunk\tpath\n'
        printf 'diagnostic\ttop22006\tcentral_tzq\tsingle\t%s\n' "$(central_output_path)"
        printf 'diagnostic\ttop22006\tprivate_tllq\tsingle\t%s\n' "$(private_output_path)"
    } > "${contract_path}"
}

write_output_contract() {
    write_output_contract_to "${output_contract}"
}

verify_resume_contract() {
    [[ -f "${output_contract}" ]] || die "resume output contract is missing"
    local expected_contract
    expected_contract="$(mktemp /tmp/run2_missing_parton_contract.XXXXXX)"
    write_output_contract_to "${expected_contract}"
    if ! cmp -s -- "${expected_contract}" "${output_contract}"; then
        rm -f -- "${expected_contract}"
        die "resume output contract does not match this driver"
    fi
    rm -f -- "${expected_contract}"
}

build_run_analysis_command() {
    local role="$1"
    local output_name="$2"
    local -n command_ref="$3"
    local cfg

    case "${role}" in
        central_tzq) cfg="${central_cfg}" ;;
        private_tllq) cfg="${private_cfg}" ;;
        *) die "unsupported role: ${role}" ;;
    esac

    command_ref=(
        "${environment_wrapper}" /bin/bash --noprofile --norc -c
        'cd "$1"; shift; exec "$@"'
        run2_missing_parton_driver
        "${script_dir}"
        "${python_env}"
        run_analysis.py
        "${cfg}"
        --years "${run2_periods[@]}"
        --skip-cr
        --hist-list njets
        --executor "${executor}"
        --outpath "${output_root}/top22006"
        --outname "${output_name}"
    )
    if [[ "${role}" == private_tllq ]]; then
        command_ref+=(--do-systs)
    fi
    command_ref+=(--category-groups "${top22006_category_groups[@]}")
}

print_execution_plan() {
    local temporary_contract
    local -a command=()
    temporary_contract="$(mktemp /tmp/run2_missing_parton_print_plan.XXXXXX)"
    write_output_contract_to "${temporary_contract}"

    printf 'plan_mode=non_submitting\n'
    printf 'source_branch=%s\n' "$(source_branch)"
    printf 'source_commit=%s\n' "$(source_commit)"
    printf 'driver_sha256=%s\n' "$(file_sha256 "${driver_path}")"
    printf 'analysis_processor_sha256=%s\n' "$(file_sha256 "${analysis_processor}")"
    printf 'periods=%s\n' "${run2_periods[*]}"
    printf 'executor=%s\n' "${executor}"
    printf 'resume_mode=%s\n' "${resume_mode}"
    printf 'central_cfg=%s\n' "${central_cfg}"
    printf 'private_cfg=%s\n' "${private_cfg}"
    printf 'category_source=%s:TOP22_006_CH_LST_SR\n' "${channel_list}"
    printf 'category_groups=%s\n' "${top22006_category_groups[*]}"
    printf 'histogram_cli_contract=--hist-list njets\n'
    printf 'expected_histogram_contents=njets njets_sumw2\n'
    printf 'output_root=%s\n' "${output_root}"
    printf 'output_contract_begin\n'
    cat "${temporary_contract}"
    printf 'output_contract_end\n'

    build_run_analysis_command central_tzq missing_parton_top22006_run2_central_tzq_njets command
    printf 'execution_command\tcentral_tzq\t'
    quote_command "${command[@]}"
    build_run_analysis_command private_tllq missing_parton_top22006_run2_private_tllq_njets command
    printf 'execution_command\tprivate_tllq\t'
    quote_command "${command[@]}"

    rm -f -- "${temporary_contract}"
}

record_operational_check() {
    local phase="$1"
    local role="$2"
    local chunk="$3"
    local path="$4"
    local result="$5"
    local size="$6"
    local checksum="$7"
    local detail="$8"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp_iso)" "${phase}" "${role}" "${chunk}" "${path}" \
        "${result}" "${size}" "${checksum}" "${detail}" >> "${validation_log}"
}

check_output_file() {
    local phase="$1"
    local role="$2"
    local chunk="$3"
    local path="$4"
    local size="0"
    local checksum=""

    if [[ ! -e "${path}" ]]; then
        record_operational_check "${phase}" "${role}" "${chunk}" "${path}" missing "${size}" "${checksum}" not_found
        return 1
    fi
    if [[ -L "${path}" || ! -f "${path}" ]]; then
        record_operational_check "${phase}" "${role}" "${chunk}" "${path}" invalid "${size}" "${checksum}" not_regular_file
        return 1
    fi
    size="$(stat -c %s -- "${path}")"
    if ((size <= 0)); then
        record_operational_check "${phase}" "${role}" "${chunk}" "${path}" invalid "${size}" "${checksum}" empty_file
        return 1
    fi
    if ! gzip -t -- "${path}"; then
        record_operational_check "${phase}" "${role}" "${chunk}" "${path}" invalid "${size}" "${checksum}" gzip_failed
        return 1
    fi
    checksum="$(file_sha256 "${path}")"
    record_operational_check "${phase}" "${role}" "${chunk}" "${path}" valid "${size}" "${checksum}" gzip_and_checksum_ok
}

quarantine_invalid_output() {
    local path="$1"
    [[ -e "${path}" || -L "${path}" ]] || return 0
    local quarantine_dir="${output_root}/invalid_outputs/attempt_$(printf '%03d' "${attempt}")"
    local destination="${quarantine_dir}/$(basename -- "${path}").$(timestamp_utc)"
    mkdir -p -- "${quarantine_dir}"
    mv -- "${path}" "${destination}"
}

run_step() {
    local phase="$1"
    local role="$2"
    local chunk="$3"
    shift 3
    local log_path="${output_root}/logs/$(printf '%03d' "${attempt}")_${phase}_${role}_${chunk}.log"
    local command_text
    command_text="$(quote_command "$@")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp_iso)" "${attempt}" "${phase}" "${role}" "${chunk}" "${command_text}" >> "${execution_commands}"
    "$@" > >(tee -a "${log_path}") 2>&1
}

record_environment_archive_cleanup() {
    local context="$1"
    local action="$2"
    local path="$3"
    local detail="$4"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp_iso)" "${attempt}" "${context}" "${action}" "${path}" "${detail}" >> "${cleanup_log}"
}

clear_stale_environment_tarballs() {
    local context="$1"
    local requested_dir="${TOPEFT_ENVS_DIR:-${environment_archive_dir_default}}"
    local expected_dir="${environment_archive_dir_default}"
    local resolved_requested=""
    local resolved_expected=""
    local enumeration_file=""
    local candidate
    local -a stale_archives=()

    if [[ -z "${requested_dir}" ]]; then
        record_environment_archive_cleanup "${context}" rejected "${requested_dir}" empty_directory
        return 1
    fi
    if [[ "${requested_dir}" != "${expected_dir}" ]]; then
        record_environment_archive_cleanup "${context}" rejected "${requested_dir}" override_must_match_default
        return 1
    fi
    if [[ -L "${requested_dir}" ]]; then
        record_environment_archive_cleanup "${context}" rejected "${requested_dir}" cache_directory_is_symlink
        return 1
    fi
    if [[ ! -e "${requested_dir}" ]]; then
        record_environment_archive_cleanup "${context}" no_match "${expected_dir}" cache_directory_absent
        return 0
    fi
    if [[ ! -d "${requested_dir}" || ! -r "${requested_dir}" || ! -x "${requested_dir}" ]]; then
        record_environment_archive_cleanup "${context}" rejected "${requested_dir}" cache_directory_not_accessible_directory
        return 1
    fi

    resolved_requested="$(cd -- "${requested_dir}" && pwd -P)" || {
        record_environment_archive_cleanup "${context}" failed "${requested_dir}" canonicalization_failed
        return 1
    }
    resolved_expected="$(cd -- "${expected_dir}" && pwd -P)" || {
        record_environment_archive_cleanup "${context}" failed "${expected_dir}" expected_canonicalization_failed
        return 1
    }

    case "${resolved_requested}" in
        ''|/|"${topeft_root}"|"${workspace_root}"|"${HOME:-/definitely-not-home}")
            record_environment_archive_cleanup "${context}" rejected "${resolved_requested}" unsafe_cache_directory
            return 1
            ;;
    esac
    if [[ "${resolved_requested}" != "${resolved_expected}" || "${resolved_expected}" != "${expected_dir}" ]]; then
        record_environment_archive_cleanup "${context}" rejected "${resolved_requested}" canonical_directory_mismatch
        return 1
    fi

    if ! enumeration_file="$(mktemp "${TMPDIR:-/tmp}/run2_missing_parton_environment_archives.XXXXXX")"; then
        record_environment_archive_cleanup "${context}" failed "${resolved_requested}" candidate_enumeration_tempfile_failed
        return 1
    fi
    if ! find "${resolved_requested}" -mindepth 1 -maxdepth 1 \
        \( -type f -o -type l \) -name "${environment_archive_pattern}" -print0 \
        | sort -z > "${enumeration_file}"; then
        record_environment_archive_cleanup "${context}" failed "${resolved_requested}" candidate_enumeration_failed
        rm -f -- "${enumeration_file}"
        return 1
    fi

    while IFS= read -r -d '' candidate; do
        if [[ -L "${candidate}" ]]; then
            record_environment_archive_cleanup "${context}" rejected "${candidate}" matching_symlink
            rm -f -- "${enumeration_file}"
            return 1
        elif [[ -f "${candidate}" ]]; then
            stale_archives+=("${candidate}")
        else
            record_environment_archive_cleanup "${context}" failed "${candidate}" candidate_changed_during_enumeration
            rm -f -- "${enumeration_file}"
            return 1
        fi
    done < "${enumeration_file}"

    if ! rm -f -- "${enumeration_file}"; then
        record_environment_archive_cleanup "${context}" failed "${enumeration_file}" candidate_enumeration_tempfile_cleanup_failed
        return 1
    fi

    if ((${#stale_archives[@]} == 0)); then
        record_environment_archive_cleanup "${context}" no_match "${resolved_requested}" "pattern=${environment_archive_pattern}"
        return 0
    fi

    for candidate in "${stale_archives[@]}"; do
        if rm -- "${candidate}"; then
            record_environment_archive_cleanup "${context}" removed "${candidate}" stale_environment_archive
        else
            record_environment_archive_cleanup "${context}" failed "${candidate}" removal_failed
            return 1
        fi
    done
}

execute_analysis_output() {
    local role="$1"
    local output_name="$2"
    local output_path="$3"
    local phase=top22006
    local chunk=single
    local -a command=()

    if [[ "${resume_mode}" == true ]] && check_output_file "${phase}" "${role}" "${chunk}" "${output_path}"; then
        printf 'resume_skip=%s\n' "${output_path}"
        return 0
    fi
    if [[ "${resume_mode}" == true ]]; then
        quarantine_invalid_output "${output_path}"
    fi
    [[ ! -e "${output_path}" && ! -L "${output_path}" ]] || die "refusing to overwrite output: ${output_path}"

    mkdir -p -- "$(dirname -- "${output_path}")"
    build_run_analysis_command "${role}" "${output_name}" command
    clear_stale_environment_tarballs "${phase}_${role}_${chunk}"
    run_step "${phase}" "${role}" "${chunk}" "${command[@]}"
    check_output_file "${phase}" "${role}" "${chunk}" "${output_path}" || \
        die "analysis did not create a valid output: ${output_path}"
}

execute_top22006() {
    write_status running "top22006_central_tzq"
    execute_analysis_output \
        central_tzq \
        missing_parton_top22006_run2_central_tzq_njets \
        "$(central_output_path)"

    write_status running "top22006_private_tllq"
    execute_analysis_output \
        private_tllq \
        missing_parton_top22006_run2_private_tllq_njets \
        "$(private_output_path)"
}

validate_final_outputs() {
    local expected_count=0
    local valid_count=0
    local class phase role chunk path
    while IFS=$'\t' read -r class phase role chunk path; do
        [[ "${class}" == class ]] && continue
        expected_count=$((expected_count + 1))
        if check_output_file final "${role}" "${chunk}" "${path}"; then
            valid_count=$((valid_count + 1))
        fi
    done < "${output_contract}"
    [[ "${expected_count}" -eq 2 ]] || die "output contract must contain exactly two outputs"
    [[ "${valid_count}" -eq "${expected_count}" ]] || die "final output validation failed"
}

write_checksum_inventory() {
    local manifest="${output_root}/pkl_manifest.sha256"
    local class phase role chunk path
    : > "${manifest}"
    while IFS=$'\t' read -r class phase role chunk path; do
        [[ "${class}" == class ]] && continue
        printf '%s  %s\n' "$(file_sha256 "${path}")" "${path}" >> "${manifest}"
    done < "${output_contract}"
}

main() {
    parse_args "$@"
    validate_output_root
    assert_operational_prerequisites
    load_category_groups

    if [[ "${print_plan}" == true ]]; then
        print_execution_plan
        exit 0
    fi

    if [[ "${resume_mode}" == true ]]; then
        initialize_existing_campaign
    else
        initialize_new_campaign
        write_output_contract
    fi

    exec > >(tee -a "${campaign_log}") 2>&1
    printf 'campaign_start=%s\n' "$(timestamp_iso)"
    printf 'output_root=%s\n' "${output_root}"
    printf 'executor=%s\n' "${executor}"
    printf 'periods=%s\n' "${run2_periods[*]}"

    execute_top22006
    write_status validating "final_output_contract"
    validate_final_outputs
    write_checksum_inventory
    write_status success "two_outputs_validated"
    printf 'campaign_complete=%s\n' "$(timestamp_iso)"
}

main "$@"
