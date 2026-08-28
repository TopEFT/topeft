#!/usr/bin/env bash
# Produce chunked Run 2 and Run 3 missing-parton source PKLs only.
# Merging, card production, and payload generation are intentionally separate stages.

set -Eeuo pipefail

script_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
driver_path="${script_dir}/$(basename -- "${BASH_SOURCE[0]}")"
topeft_root="$(cd -P -- "${script_dir}/../.." && pwd)"
workspace_root="$(cd -P -- "${topeft_root}/.." && pwd)"
environment_wrapper="${workspace_root}/codex-run.sh"
python_env="${PYTHON_ENV:-/users/apiccine/work/miniconda3/envs/clib-env/bin/python}"
executor="${MISSING_PARTON_EXECUTOR:-work_queue}"
all_analysis_chunks="${ALL_ANALYSIS_CHUNKS:-2}"
environment_archive_dir_default="${script_dir}/topeft-envs"
environment_archive_pattern='env_*.tar.gz'

run2_years=(UL16 UL16APV UL17 UL18)
run3_years=(2022 2022EE 2023 2023BPix)
channel_list="${topeft_root}/topeft/channels/ch_lst.json"
analysis_processor="${script_dir}/analysis_processor.py"
run_analysis="${script_dir}/run_analysis.py"

declare -A role_cfg=(
    [run2_central_tzq]="${topeft_root}/input_samples/cfgs/missing_parton_top22006_central_tzq_NDSkim.cfg"
    [run2_private_tllq]="${topeft_root}/input_samples/cfgs/missing_parton_top22006_private_tllq_NDSkim.cfg"
    [run3_central_tzq]="${topeft_root}/input_samples/cfgs/missing_parton_run3_central_tzq_NDSkim.cfg"
    [run3_private_tllq]="${topeft_root}/input_samples/cfgs/missing_parton_run3_private_tllq_NDSkim.cfg"
)
declare -A role_sumw2_storage_modes=(
    [run2_central_tzq]=production_central
    [run2_private_tllq]=production
    [run3_central_tzq]=production_central
    [run3_private_tllq]=production
)
declare -A role_sumw2_process_prefixes=(
    [run2_central_tzq]=tZq_central
    [run2_private_tllq]=tllq_private
    [run3_central_tzq]=tZq_central
    [run3_private_tllq]=tllq_private
)
roles=(run2_central_tzq run2_private_tllq run3_central_tzq run3_private_tllq)

output_root=""
resume_mode=false
print_plan_mode=false
campaign_mutation_allowed=false
status_file=""
metadata_file=""
output_contract_file=""
partition_file=""
validation_file=""
commands_file=""
manifest_file=""
state_history_file=""
environment_archive_cleanup_file=""
sumw2_options_dir=""
category_digest=""
declare -a all_analysis_groups=()
declare -a chunk_specs=()
declare -a resolved_command=()

usage() {
    cat <<'EOF'
Usage:
  run_missing_parton_pkls_all_eras.sh OUTPUT_ROOT
  run_missing_parton_pkls_all_eras.sh --resume OUTPUT_ROOT
  run_missing_parton_pkls_all_eras.sh --print-plan OUTPUT_ROOT

Produce source PKLs only: Run 2 and Run 3, central tZq and private tllq,
split deterministically across ALL_CH_LST_SR category chunks. This driver never
merges PKLs, runs make_cards.py, creates source cards, or generates payloads.

Options:
  --resume      Continue an incomplete campaign created by this driver.
  --print-plan  Resolve and print the non-submitting command/output plan.
  -h, --help    Show this help text.

Environment:
  ALL_ANALYSIS_CHUNKS      Integer >= 2 (default: 2).
  MISSING_PARTON_EXECUTOR  futures, work_queue, or taskvine (default: work_queue).
  PYTHON_ENV               Interpreter used only through correction-lib/codex-run.sh.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

timestamp_iso() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

quote_command() {
    local item
    for item in "$@"; do
        printf '%q ' "${item}"
    done
    printf '\n'
}

run_python() {
    "${environment_wrapper}" /bin/bash --noprofile --norc -c \
        'cd "$1"; shift; exec "$@"' \
        missing_parton_source_pkl_driver "${script_dir}" "${python_env}" "$@"
}

parse_args() {
    while (($#)); do
        case "$1" in
            --resume)
                [[ "${resume_mode}" == false && "${print_plan_mode}" == false ]] || die "select one campaign mode"
                resume_mode=true
                shift
                [[ $# -eq 1 ]] || die "--resume requires exactly one OUTPUT_ROOT"
                output_root="$1"
                ;;
            --print-plan)
                [[ "${resume_mode}" == false && "${print_plan_mode}" == false ]] || die "select one campaign mode"
                print_plan_mode=true
                shift
                [[ $# -eq 1 ]] || die "--print-plan requires exactly one OUTPUT_ROOT"
                output_root="$1"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            --*) die "unsupported option: $1" ;;
            *)
                [[ -z "${output_root}" ]] || die "only one OUTPUT_ROOT may be specified"
                output_root="$1"
                ;;
        esac
        shift
    done

    [[ -n "${output_root}" ]] || die "OUTPUT_ROOT is required"
    [[ "${output_root}" == /* ]] || die "OUTPUT_ROOT must be absolute"
    is_positive_integer "${all_analysis_chunks}" || die "ALL_ANALYSIS_CHUNKS must be an integer >= 2"
    (( all_analysis_chunks >= 2 )) || die "ALL_ANALYSIS_CHUNKS must be at least 2"
    case "${executor}" in futures|work_queue|taskvine) ;; *) die "unsupported MISSING_PARTON_EXECUTOR: ${executor}" ;; esac
}

validate_output_root() {
    [[ ! -L "${output_root}" ]] || die "OUTPUT_ROOT must not be a symlink"
    output_root="$(realpath -m -- "${output_root}")"
    case "${output_root}" in
        /|"${workspace_root}"|"${topeft_root}"|"${script_dir}"|"${HOME:-/definitely-not-home}" ) die "unsafe OUTPUT_ROOT: ${output_root}" ;;
        "${workspace_root}"/*|"${topeft_root}"/*|"${HOME:-/definitely-not-home}"/*) die "OUTPUT_ROOT must be outside the workspace, repository, and home" ;;
    esac
}

assert_prerequisites() {
    local path
    for path in "${environment_wrapper}" "${python_env}" "${run_analysis}" "${analysis_processor}" "${channel_list}"; do
        [[ -e "${path}" ]] || die "required path is missing: ${path}"
    done
    [[ -x "${environment_wrapper}" ]] || die "environment wrapper is not executable"
    [[ -x "${python_env}" ]] || die "Python interpreter is not executable"
    for path in "${role_cfg[@]}"; do [[ -f "${path}" ]] || die "required cfg is missing: ${path}"; done
    command -v git >/dev/null || die "git is required for provenance"
    command -v gzip >/dev/null || die "gzip is required for output validation"
    command -v sha256sum >/dev/null || die "sha256sum is required for provenance"
    command -v flock >/dev/null || die "flock is required for campaign ownership"
}

read_category_groups() {
    mapfile -t all_analysis_groups < <(
        run_python - "${channel_list}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
groups = payload.get("ALL_CH_LST_SR")
if not isinstance(groups, dict) or not groups:
    raise SystemExit("ALL_CH_LST_SR must be a nonempty object")
for group in groups:
    print(group)
PY
    )
    (( ${#all_analysis_groups[@]} >= all_analysis_chunks )) || die "ALL_ANALYSIS_CHUNKS exceeds ALL_CH_LST_SR category count"
    category_digest="$(printf '%s\n' "${all_analysis_groups[@]}" | sha256sum | awk '{print $1}')"
}

build_chunk_specs() {
    local total="${#all_analysis_groups[@]}"
    local start=0
    local index remaining chunks_left chunk_size
    chunk_specs=()
    for ((index = 0; index < all_analysis_chunks; index++)); do
        remaining=$((total - start))
        chunks_left=$((all_analysis_chunks - index))
        chunk_size=$(((remaining + chunks_left - 1) / chunks_left))
        chunk_specs+=("$(printf '%s ' "${all_analysis_groups[@]:start:chunk_size}")")
        start=$((start + chunk_size))
    done
}

campaign_paths() {
    metadata_file="${output_root}/campaign_metadata.txt"
    output_contract_file="${output_root}/output_contract.tsv"
    partition_file="${output_root}/category_partition.tsv"
    validation_file="${output_root}/validation.tsv"
    commands_file="${output_root}/execution_commands.tsv"
    manifest_file="${output_root}/pkl_manifest.sha256"
    status_file="${output_root}/status.txt"
    state_history_file="${output_root}/state_history.tsv"
    environment_archive_cleanup_file="${output_root}/environment_archive_cleanup.tsv"
    sumw2_options_dir="${output_root}/sumw2_options"
}

role_run() { printf '%s\n' "${1%%_*}"; }
role_name() { printf '%s\n' "${1#*_}"; }
role_systematics() { [[ "$1" == *_private_tllq ]] && printf 'yes\n' || printf 'no\n'; }
role_sumw2_storage_mode() {
    local role="$1"
    [[ -n "${role_sumw2_storage_modes[${role}]:-}" ]] || die "missing sumw2 storage mode for role: ${role}"
    printf '%s\n' "${role_sumw2_storage_modes[${role}]}"
}

role_sumw2_process_prefix() {
    local role="$1"
    [[ -n "${role_sumw2_process_prefixes[${role}]:-}" ]] || die "missing sumw2 process prefix for role: ${role}"
    printf '%s\n' "${role_sumw2_process_prefixes[${role}]}"
}

sumw2_options_path_for() {
    printf '%s/%s.yml\n' "${sumw2_options_dir}" "$1"
}

sumw2_options_sha256() {
    sumw2_options_content "$1" | sha256sum | awk '{print $1}'
}

sumw2_options_content() {
    local role="$1"
    printf 'sumw2_storage:\n  mode: %s\n  rules:\n    - process_prefixes:\n        - %s\n' \
        "$(role_sumw2_storage_mode "${role}")" "$(role_sumw2_process_prefix "${role}")"
    printf 'data_driven_products:\n  nonprompt:\n    enabled: false\n  flips:\n    enabled: false\n'
}

write_sumw2_options() {
    local role destination
    mkdir -p -- "${sumw2_options_dir}"
    for role in "${roles[@]}"; do
        destination="$(sumw2_options_path_for "${role}")"
        [[ ! -e "${destination}" && ! -L "${destination}" ]] || die "refusing to overwrite sumw2 options: ${destination}"
        sumw2_options_content "${role}" > "${destination}"
    done
}

raw_path_for() {
    local role="$1" chunk_label="$2" run role_name_value
    run="$(role_run "${role}")"
    role_name_value="$(role_name "${role}")"
    printf '%s/%s/raw/%s/missing_parton_%s_all_analysis_%s_chunk%s_njets.pkl.gz\n' \
        "${output_root}" "${run}" "${role_name_value}" "${run}" "${role_name_value}" "${chunk_label}"
}

write_partition() {
    local destination="$1" chunk_index group_index chunk_label
    local -a chunk_groups=()
    printf 'chunk\tposition\tcategory_group\n' > "${destination}"
    for ((chunk_index = 0; chunk_index < all_analysis_chunks; chunk_index++)); do
        printf -v chunk_label '%02d' "$((chunk_index + 1))"
        read -r -a chunk_groups <<< "${chunk_specs[chunk_index]}"
        for ((group_index = 0; group_index < ${#chunk_groups[@]}; group_index++)); do
            printf '%s\t%s\t%s\n' "${chunk_label}" "$((group_index + 1))" "${chunk_groups[group_index]}" >> "${destination}"
        done
    done
}

write_output_contract() {
    local destination="$1" role chunk_index chunk_label
    printf 'run\trole\tchunk\tpath\tcategory_digest\n' > "${destination}"
    for role in "${roles[@]}"; do
        for ((chunk_index = 0; chunk_index < all_analysis_chunks; chunk_index++)); do
            printf -v chunk_label '%02d' "$((chunk_index + 1))"
            printf '%s\t%s\t%s\t%s\t%s\n' "$(role_run "${role}")" "$(role_name "${role}")" "${chunk_label}" \
                "$(raw_path_for "${role}" "${chunk_label}")" "${category_digest}" >> "${destination}"
        done
    done
}

write_metadata() {
    local role
    {
        printf 'campaign_type=missing_parton_source_pkls_all_eras\n'
        printf 'created_at=%s\n' "$(timestamp_iso)"
        printf 'output_root=%s\n' "${output_root}"
        printf 'driver_path=%s\n' "${driver_path}"
        printf 'driver_sha256=%s\n' "$(file_sha256 "${driver_path}")"
        printf 'topeft_branch=%s\n' "$(git -C "${topeft_root}" branch --show-current)"
        printf 'topeft_head=%s\n' "$(git -C "${topeft_root}" rev-parse HEAD)"
        printf 'topcoffea_branch=%s\n' "$(git -C "${workspace_root}/topcoffea" branch --show-current)"
        printf 'topcoffea_head=%s\n' "$(git -C "${workspace_root}/topcoffea" rev-parse HEAD)"
        printf 'analysis_processor_sha256=%s\n' "$(file_sha256 "${analysis_processor}")"
        printf 'run_analysis_sha256=%s\n' "$(file_sha256 "${run_analysis}")"
        printf 'channel_list_sha256=%s\n' "$(file_sha256 "${channel_list}")"
        printf 'category_source=ALL_CH_LST_SR\n'
        printf 'category_count=%s\n' "${#all_analysis_groups[@]}"
        printf 'category_digest=%s\n' "${category_digest}"
        printf 'all_analysis_chunks=%s\n' "${all_analysis_chunks}"
        printf 'run2_years=%s\n' "${run2_years[*]}"
        printf 'run3_years=%s\n' "${run3_years[*]}"
        printf 'hist_list=njets\n'
        printf 'do_np=no\n'
        printf 'executor=%s\n' "${executor}"
        printf 'environment_archive_directory=%s\n' "${environment_archive_dir_default}"
        printf 'environment_archive_pattern=%s\n' "${environment_archive_pattern}"
        printf 'environment_archive_cleanup_policy=before_each_real_run_analysis_command;regular_files_only\n'
        printf 'private_sm_point_sumw2_semantics=sm_only_complete_event_contribution_squared\n'
        for role in "${roles[@]}"; do
            printf 'cfg_%s=%s\n' "${role}" "${role_cfg[${role}]}"
            printf 'cfg_%s_sha256=%s\n' "${role}" "$(file_sha256 "${role_cfg[${role}]}")"
            printf 'do_systs_%s=%s\n' "${role}" "$(role_systematics "${role}")"
            printf 'sumw2_storage_mode_%s=%s\n' "${role}" "$(role_sumw2_storage_mode "${role}")"
            printf 'sumw2_options_%s=%s\n' "${role}" "$(sumw2_options_path_for "${role}")"
            printf 'sumw2_options_%s_sha256=%s\n' "${role}" "$(sumw2_options_sha256 "${role}")"
        done
    } > "${metadata_file}"
}

write_status() {
    local state="$1" detail="${2:-}"
    [[ "${campaign_mutation_allowed}" == true ]] || return 0
    printf 'state=%s\ntimestamp=%s\ndetail=%s\n' "${state}" "$(timestamp_iso)" "${detail}" > "${status_file}"
    printf '%s\t%s\t%s\n' "$(timestamp_iso)" "${state}" "${detail}" >> "${state_history_file}"
}

on_error() {
    local code=$?
    write_status failed "line=${1};exit_code=${code}"
    exit "${code}"
}

trap 'on_error "$LINENO"' ERR
trap 'write_status interrupted "signal=INT"; exit 130' INT
trap 'write_status interrupted "signal=TERM"; exit 130' TERM

metadata_value() {
    awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "${metadata_file}"
}

verify_resume_metadata() {
    local role
    [[ -f "${metadata_file}" && ! -L "${metadata_file}" ]] || die "resume metadata is missing or unsafe"
    [[ "$(metadata_value campaign_type)" == missing_parton_source_pkls_all_eras ]] || die "resume campaign type mismatch"
    [[ "$(metadata_value output_root)" == "${output_root}" ]] || die "resume output root mismatch"
    [[ "$(metadata_value driver_path)" == "${driver_path}" ]] || die "resume driver path mismatch"
    [[ "$(metadata_value driver_sha256)" == "$(file_sha256 "${driver_path}")" ]] || die "resume driver hash mismatch"
    [[ "$(metadata_value topeft_head)" == "$(git -C "${topeft_root}" rev-parse HEAD)" ]] || die "resume topeft HEAD mismatch"
    [[ "$(metadata_value topeft_branch)" == "$(git -C "${topeft_root}" branch --show-current)" ]] || die "resume topeft branch mismatch"
    [[ "$(metadata_value topcoffea_head)" == "$(git -C "${workspace_root}/topcoffea" rev-parse HEAD)" ]] || die "resume topcoffea HEAD mismatch"
    [[ "$(metadata_value analysis_processor_sha256)" == "$(file_sha256 "${analysis_processor}")" ]] || die "resume processor hash mismatch"
    [[ "$(metadata_value run_analysis_sha256)" == "$(file_sha256 "${run_analysis}")" ]] || die "resume run_analysis hash mismatch"
    [[ "$(metadata_value channel_list_sha256)" == "$(file_sha256 "${channel_list}")" ]] || die "resume channel-list hash mismatch"
    [[ "$(metadata_value category_digest)" == "${category_digest}" ]] || die "resume category digest mismatch"
    [[ "$(metadata_value category_count)" == "${#all_analysis_groups[@]}" ]] || die "resume category count mismatch"
    [[ "$(metadata_value category_source)" == ALL_CH_LST_SR ]] || die "resume category source mismatch"
    [[ "$(metadata_value all_analysis_chunks)" == "${all_analysis_chunks}" ]] || die "resume chunk count mismatch"
    [[ "$(metadata_value executor)" == "${executor}" ]] || die "resume executor mismatch"
    [[ "$(metadata_value run2_years)" == "${run2_years[*]}" ]] || die "resume Run 2 years mismatch"
    [[ "$(metadata_value run3_years)" == "${run3_years[*]}" ]] || die "resume Run 3 years mismatch"
    [[ "$(metadata_value hist_list)" == njets ]] || die "resume histogram contract mismatch"
    [[ "$(metadata_value do_np)" == no ]] || die "resume nonprompt contract mismatch"
    for role in "${roles[@]}"; do
        [[ "$(metadata_value cfg_${role})" == "${role_cfg[${role}]}" ]] || die "resume cfg path mismatch: ${role}"
        [[ "$(metadata_value cfg_${role}_sha256)" == "$(file_sha256 "${role_cfg[${role}]}")" ]] || die "resume cfg hash mismatch: ${role}"
        [[ "$(metadata_value do_systs_${role})" == "$(role_systematics "${role}")" ]] || die "resume systematic contract mismatch: ${role}"
        [[ "$(metadata_value sumw2_storage_mode_${role})" == "$(role_sumw2_storage_mode "${role}")" ]] || die "resume sumw2 mode mismatch: ${role}"
        [[ "$(metadata_value sumw2_options_${role})" == "$(sumw2_options_path_for "${role}")" ]] || die "resume sumw2 options path mismatch: ${role}"
        [[ -f "$(sumw2_options_path_for "${role}")" && ! -L "$(sumw2_options_path_for "${role}")" ]] || die "resume sumw2 options are missing or unsafe: ${role}"
        [[ "$(metadata_value sumw2_options_${role}_sha256)" == "$(sumw2_options_sha256 "${role}")" ]] || die "resume sumw2 options contract mismatch: ${role}"
        [[ "$(file_sha256 "$(sumw2_options_path_for "${role}")")" == "$(sumw2_options_sha256 "${role}")" ]] || die "resume sumw2 options content mismatch: ${role}"
    done
    [[ -f "${output_contract_file}" && -f "${partition_file}" && -f "${status_file}" ]] || die "resume campaign contract is incomplete"
}

verify_resume_contract() {
    local temporary_contract temporary_partition
    temporary_contract="$(mktemp "${TMPDIR:-/tmp}/missing_parton_contract.XXXXXX")"
    temporary_partition="$(mktemp "${TMPDIR:-/tmp}/missing_parton_partition.XXXXXX")"
    write_output_contract "${temporary_contract}"
    write_partition "${temporary_partition}"
    cmp -s "${temporary_contract}" "${output_contract_file}" || die "resume output contract mismatch"
    cmp -s "${temporary_partition}" "${partition_file}" || die "resume category partition mismatch"
    rm -f -- "${temporary_contract}" "${temporary_partition}"
}

initialize_fresh_campaign() {
    if [[ -e "${output_root}" ]]; then
        [[ -d "${output_root}" && ! -L "${output_root}" ]] || die "OUTPUT_ROOT exists and is not a safe directory"
        [[ -z "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "refusing populated OUTPUT_ROOT without --resume"
    else
        mkdir -p -- "$(dirname -- "${output_root}")"
        mkdir -- "${output_root}"
    fi
    mkdir -p -- "${output_root}/run2/raw/central_tzq" "${output_root}/run2/raw/private_tllq" \
        "${output_root}/run3/raw/central_tzq" "${output_root}/run3/raw/private_tllq" "${output_root}/invalid_outputs"
    exec 9>"${output_root}/.campaign.lock"
    flock -n 9 || die "another campaign process holds this OUTPUT_ROOT"
    campaign_mutation_allowed=true
    printf 'timestamp\tstate\tdetail\n' > "${state_history_file}"
    printf 'timestamp\trun\trole\tchunk\tpath\tresult\tsize\tsha256\tdetail\n' > "${validation_file}"
    printf 'timestamp\trun\trole\tchunk\tcommand\n' > "${commands_file}"
    printf 'timestamp\tcontext\tresolved_directory\tpattern\tcandidate_path\taction\tresult\tdetail\n' \
        > "${environment_archive_cleanup_file}"
    write_sumw2_options
    write_metadata
    write_partition "${partition_file}"
    write_output_contract "${output_contract_file}"
    write_status initialized fresh_campaign
}

initialize_resume_campaign() {
    [[ -d "${output_root}" && ! -L "${output_root}" ]] || die "--resume requires a safe existing campaign directory"
    verify_resume_metadata
    [[ "$(awk -F= '$1 == "state" {print $2}' "${status_file}")" != success ]] || die "a successful campaign cannot be resumed"
    exec 9>"${output_root}/.campaign.lock"
    flock -n 9 || die "another campaign process holds this OUTPUT_ROOT"
    campaign_mutation_allowed=true
    verify_resume_contract
    write_status resumed resume_contract_verified
}

build_run_analysis_command() {
    local role="$1" chunk_index="$2" chunk_label run cfg
    local -a category_groups=()
    run="$(role_run "${role}")"
    cfg="${role_cfg[${role}]}"
    printf -v chunk_label '%02d' "$((chunk_index + 1))"
    resolved_command=(
        "${environment_wrapper}" /bin/bash --noprofile --norc -c
        'cd "$1"; shift; exec "$@"'
        missing_parton_source_pkl_driver "${script_dir}" "${python_env}" run_analysis.py
        "${cfg}" --years
    )
    if [[ "${run}" == run2 ]]; then resolved_command+=("${run2_years[@]}"); else resolved_command+=("${run3_years[@]}"); fi
    read -r -a category_groups <<< "${chunk_specs[chunk_index]}"
    resolved_command+=(--all-analysis --category-groups "${category_groups[@]}" --skip-cr --hist-list njets --executor "${executor}")
    resolved_command+=(--options "$(sumw2_options_path_for "${role}")")
    resolved_command+=(--outpath "$(dirname -- "$(raw_path_for "${role}" "${chunk_label}")")")
    resolved_command+=(--outname "$(basename -- "$(raw_path_for "${role}" "${chunk_label}")" .pkl.gz)")
    if [[ "${role}" == *_private_tllq ]]; then
        resolved_command+=(--do-systs)
    fi
}

record_environment_archive_cleanup() {
    local context="$1" resolved_directory="$2" candidate_path="$3"
    local action="$4" result="$5" detail="$6"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp_iso)" "${context}" "${resolved_directory}" \
        "${environment_archive_pattern}" "${candidate_path}" "${action}" \
        "${result}" "${detail}" >> "${environment_archive_cleanup_file}"
}

clear_stale_environment_archives() {
    local context="$1" requested_directory="${TOPEFT_ENVS_DIR:-${environment_archive_dir_default}}"
    local resolved_directory="" enumeration_file="" candidate_path
    local -a archive_candidates=() symlink_candidates=()

    if [[ "${requested_directory}" != "${environment_archive_dir_default}" ]]; then
        record_environment_archive_cleanup "${context}" "${requested_directory}" "-" rejected fail override_must_match_driver_cache
        return 1
    fi
    if [[ -L "${environment_archive_dir_default}" ]]; then
        record_environment_archive_cleanup "${context}" "${environment_archive_dir_default}" "-" rejected fail cache_directory_is_symlink
        return 1
    fi
    if [[ ! -e "${environment_archive_dir_default}" ]]; then
        record_environment_archive_cleanup "${context}" "${environment_archive_dir_default}" "-" no_match pass cache_directory_absent
        return 0
    fi
    if [[ ! -d "${environment_archive_dir_default}" || ! -r "${environment_archive_dir_default}" || ! -x "${environment_archive_dir_default}" ]]; then
        record_environment_archive_cleanup "${context}" "${environment_archive_dir_default}" "-" rejected fail cache_directory_not_accessible
        return 1
    fi
    resolved_directory="$(cd -P -- "${environment_archive_dir_default}" && pwd -P)" || {
        record_environment_archive_cleanup "${context}" "${environment_archive_dir_default}" "-" failed fail canonicalization_failed
        return 1
    }
    if [[ "${resolved_directory}" != "${environment_archive_dir_default}" ]]; then
        record_environment_archive_cleanup "${context}" "${resolved_directory}" "-" rejected fail canonical_path_mismatch
        return 1
    fi
    enumeration_file="$(mktemp "${TMPDIR:-/tmp}/missing_parton_environment_archives.XXXXXX")" || {
        record_environment_archive_cleanup "${context}" "${resolved_directory}" "-" failed fail enumeration_tempfile_failed
        return 1
    }
    if ! find "${resolved_directory}" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -name "${environment_archive_pattern}" -print0 | sort -z > "${enumeration_file}"; then
        rm -f -- "${enumeration_file}"
        record_environment_archive_cleanup "${context}" "${resolved_directory}" "-" failed fail candidate_enumeration_failed
        return 1
    fi
    while IFS= read -r -d '' candidate_path; do
        if [[ -L "${candidate_path}" ]]; then
            symlink_candidates+=("${candidate_path}")
        elif [[ -f "${candidate_path}" ]]; then
            archive_candidates+=("${candidate_path}")
        else
            rm -f -- "${enumeration_file}"
            record_environment_archive_cleanup "${context}" "${resolved_directory}" "${candidate_path}" failed fail candidate_changed_during_enumeration
            return 1
        fi
    done < "${enumeration_file}"
    rm -f -- "${enumeration_file}"
    if (( ${#symlink_candidates[@]} > 0 )); then
        for candidate_path in "${symlink_candidates[@]}"; do
            record_environment_archive_cleanup "${context}" "${resolved_directory}" "${candidate_path}" rejected fail matching_archive_symlink
        done
        return 1
    fi
    if (( ${#archive_candidates[@]} == 0 )); then
        record_environment_archive_cleanup "${context}" "${resolved_directory}" "-" no_match pass no_generated_archives
        return 0
    fi
    for candidate_path in "${archive_candidates[@]}"; do
        if rm -- "${candidate_path}"; then
            record_environment_archive_cleanup "${context}" "${resolved_directory}" "${candidate_path}" removed pass stale_environment_archive
        else
            record_environment_archive_cleanup "${context}" "${resolved_directory}" "${candidate_path}" failed fail removal_failed
            return 1
        fi
    done
}

print_plan() {
    local role chunk_index chunk_label temporary_contract temporary_partition
    temporary_contract="$(mktemp "${TMPDIR:-/tmp}/missing_parton_plan_contract.XXXXXX")"
    temporary_partition="$(mktemp "${TMPDIR:-/tmp}/missing_parton_plan_partition.XXXXXX")"
    write_output_contract "${temporary_contract}"
    write_partition "${temporary_partition}"
    printf 'plan_mode=non_submitting\n'
    printf 'category_source=ALL_CH_LST_SR\ncategory_count=%s\ncategory_digest=%s\n' "${#all_analysis_groups[@]}" "${category_digest}"
    printf 'all_analysis_chunks=%s\nexecutor=%s\nrun2_years=%s\nrun3_years=%s\nhist_list=njets\ndo_np=no\n' "${all_analysis_chunks}" "${executor}" "${run2_years[*]}" "${run3_years[*]}"
    for role in "${roles[@]}"; do
        printf 'sumw2_storage_mode_%s=%s\n' "${role}" "$(role_sumw2_storage_mode "${role}")"
        printf 'sumw2_rule_process_prefix_%s=%s\n' "${role}" "$(role_sumw2_process_prefix "${role}")"
        printf 'data_driven_products_%s=nonprompt=false,flips=false\n' "${role}"
    done
    printf '%s\n' '--- category_partition.tsv ---'; cat "${temporary_partition}"
    printf '%s\n' '--- output_contract.tsv ---'; cat "${temporary_contract}"
    printf '%s\n' '--- run_analysis_commands ---'
    for role in "${roles[@]}"; do
        for ((chunk_index = 0; chunk_index < all_analysis_chunks; chunk_index++)); do
            printf -v chunk_label '%02d' "$((chunk_index + 1))"
            build_run_analysis_command "${role}" "${chunk_index}"
            printf 'run_analysis\t%s\t%s\t' "${role}" "${chunk_label}"
            quote_command "${resolved_command[@]}"
        done
    done
    rm -f -- "${temporary_contract}" "${temporary_partition}"
}

check_output_file() {
    local role="$1" chunk_label="$2" path="$3" size checksum run role_name_value
    run="$(role_run "${role}")"; role_name_value="$(role_name "${role}")"
    if [[ -L "${path}" || ! -f "${path}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\tinvalid\t0\t-\tnot_regular_file\n' "$(timestamp_iso)" "${run}" "${role_name_value}" "${chunk_label}" "${path}" >> "${validation_file}"
        return 1
    fi
    size="$(stat -c %s -- "${path}")"
    if [[ "${size}" -le 0 ]] || ! gzip -t -- "${path}"; then
        printf '%s\t%s\t%s\t%s\t%s\tinvalid\t%s\t-\tgzip_or_size_failed\n' "$(timestamp_iso)" "${run}" "${role_name_value}" "${chunk_label}" "${path}" "${size}" >> "${validation_file}"
        return 1
    fi
    checksum="$(file_sha256 "${path}")"
    printf '%s\t%s\t%s\t%s\t%s\tvalid\t%s\t%s\tgzip_and_checksum_ok\n' "$(timestamp_iso)" "${run}" "${role_name_value}" "${chunk_label}" "${path}" "${size}" "${checksum}" >> "${validation_file}"
    return 0
}

quarantine_invalid_output() {
    local role="$1" chunk_label="$2" path="$3" destination
    destination="${output_root}/invalid_outputs/$(basename -- "${path}").$(timestamp_iso)"
    mv -- "${path}" "${destination}"
    printf '%s\t%s\t%s\t%s\t%s\tquarantined\t-\t-\t%s\n' "$(timestamp_iso)" "$(role_run "${role}")" "$(role_name "${role}")" "${chunk_label}" "${path}" "${destination}" >> "${validation_file}"
}

execute_chunk() {
    local role="$1" chunk_index="$2" chunk_label path command_text
    printf -v chunk_label '%02d' "$((chunk_index + 1))"
    path="$(raw_path_for "${role}" "${chunk_label}")"
    if [[ "${resume_mode}" == true && -e "${path}" ]]; then
        if check_output_file "${role}" "${chunk_label}" "${path}"; then return 0; fi
        quarantine_invalid_output "${role}" "${chunk_label}" "${path}"
    fi
    [[ ! -e "${path}" && ! -L "${path}" ]] || die "refusing to overwrite output: ${path}"
    build_run_analysis_command "${role}" "${chunk_index}"
    command_text="$(quote_command "${resolved_command[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp_iso)" "$(role_run "${role}")" "$(role_name "${role}")" "${chunk_label}" "${command_text}" >> "${commands_file}"
    clear_stale_environment_archives "${role}_${chunk_label}"
    "${resolved_command[@]}"
    check_output_file "${role}" "${chunk_label}" "${path}" || die "run_analysis.py did not create a valid output: ${path}"
}

validate_final_outputs() {
    local run role chunk path digest expected_count=0 valid_count=0
    while IFS=$'\t' read -r run role chunk path digest; do
        [[ "${run}" == run ]] && continue
        expected_count=$((expected_count + 1))
        if check_output_file "${run}_${role}" "${chunk}" "${path}"; then valid_count=$((valid_count + 1)); fi
    done < "${output_contract_file}"
    [[ "${expected_count}" -eq $((4 * all_analysis_chunks)) ]] || die "output contract count mismatch"
    [[ "${valid_count}" -eq "${expected_count}" ]] || die "final source-PKL validation failed"
}

write_manifest() {
    local run role chunk path digest
    : > "${manifest_file}"
    while IFS=$'\t' read -r run role chunk path digest; do
        [[ "${run}" == run ]] && continue
        printf '%s  %s\n' "$(file_sha256 "${path}")" "${path}" >> "${manifest_file}"
    done < "${output_contract_file}"
}

main() {
    parse_args "$@"
    validate_output_root
    assert_prerequisites
    read_category_groups
    build_chunk_specs
    campaign_paths
    if [[ "${print_plan_mode}" == true ]]; then print_plan; return 0; fi
    if [[ "${resume_mode}" == true ]]; then initialize_resume_campaign; else initialize_fresh_campaign; fi
    local role chunk_index
    write_status running source_pkl_chunks_only
    for role in "${roles[@]}"; do
        for ((chunk_index = 0; chunk_index < all_analysis_chunks; chunk_index++)); do execute_chunk "${role}" "${chunk_index}"; done
    done
    validate_final_outputs
    write_manifest
    write_status success "all_source_pkl_chunks_validated"
}

main "$@"
