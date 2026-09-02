#!/usr/bin/env bash

set -u
set -o pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root="/users/apiccine/work/correction-lib"
topeft_root="${workspace_root}/topeft"
topcoffea_root="${workspace_root}/topcoffea"
plotter_path="${script_dir}/run_plotter.sh"
plot_source_path="${script_dir}/make_cr_and_sr_plots.py"
codex_run_path="${workspace_root}/codex-run.sh"
python_path="/users/apiccine/work/miniconda3/envs/clib-env/bin/python"
sampler_path="${script_dir}/run3_processing_memory_sampler.py"
accepted_input_inventory="${workspace_root}/reports/diagnostics/SRPLOT_010_run3_processing_only_production_and_inventory/SRPLOT_010_input_pkl_inventory.tsv"
accepted_fitting_manifest="${workspace_root}/reports/diagnostics/SRPLOT_010_run3_processing_only_production_and_inventory/SRPLOT_010_preexisting_fitting_manifest.tsv"
expected_processing_oracle="${workspace_root}/reports/diagnostics/SRPLOT_010_run3_processing_only_production_and_inventory/SRPLOT_010_expected_processing_paths.tsv"
accepted_block1_manifest="${workspace_root}/reports/diagnostics/SRPLOT_010R2_reconcile_live_run3_processing_state_after_ambiguous_interruption/SRPLOT_010R2_processing_snapshot_2.tsv"
default_runtime_root="${workspace_root}/reports/diagnostics/SRPLOT_010_run3_processing_recovery_runtime"
workers=6

runtime_root="${default_runtime_root}"
fixture_root=""
stub_scenario=""
attempt_backend="production"
attempts_path=""
memory_samples_path=""
campaign_summary_path=""
task_plan_path=""
fitting_snapshot_path=""
output_inventory_path=""
logs_dir=""
time_dir=""

declare -a task_ids=()
declare -a task_blocks=()
declare -a task_pkls=()
declare -a task_output_dirs=()
declare -a task_year_aliases=()
declare -a logical_statuses=()
declare -a attempt_command=()
declare -a dependency_paths=()
declare -a dependency_sizes=()
declare -a dependency_sha256s=()

attempt_id=""
attempt_exit_code=""
attempt_pid=""
attempt_ppid=""
attempt_process_group_id=""
attempt_session_id=""
attempt_process_start_ticks=""
attempt_process_start_time=""
attempt_log_path=""
attempt_time_report_path=""
attempt_memory_summary_path=""
attempt_command_text=""
attempt_command_sha256=""
last_logical_status=""

usage() {
    cat <<'USAGE'
Usage:
  run3_processing_recovery_campaign.sh --print-plan
  run3_processing_recovery_campaign.sh --execute [--runtime-root ABSOLUTE_PATH]
  run3_processing_recovery_campaign.sh --status [--runtime-root ABSOLUTE_PATH]
  run3_processing_recovery_campaign.sh --finalize [--runtime-root ABSOLUTE_PATH]

Validation-only interface:
  run3_processing_recovery_campaign.sh --stub-scenario SCENARIO --runtime-root ABSOLUTE_PATH --fixture-root ABSOLUTE_PATH

Block 1 is an externally accepted immutable success and is verified but never
executed. The continuation contains exactly four serial Run-3 merged-njets
processing tasks for blocks 2-5. Existing block directories and fitting
artifacts are expected. Any existing intended block2-5 *_processing.* path
blocks execution. There is no resume or automatic-retry mode; an existing
runtime root is a recovery boundary.
USAGE
}

add_dependency() {
    dependency_paths+=("$1")
    dependency_sizes+=("$2")
    dependency_sha256s+=("$3")
}

initialize_dependency_contract() {
    add_dependency "${topeft_root}/analysis/topeft_run2/run_plotter.sh" 17557 e18360a7b41e1e19cdd083cbf9b996d9f7ddee6a9b7de52e27611f964686d52d
    add_dependency "${topeft_root}/analysis/topeft_run2/make_cr_and_sr_plots.py" 341262 824a3d86a1cc01c945e16e7c95e6c216beff5095c19ebf9aa69666c75d502d2e
    add_dependency "${topeft_root}/sitecustomize.py" 651 347fa4e392cebd474a962b2bc15b2b2c60341584a64fa175d89e3dcb2fffb9f5
    add_dependency "${topeft_root}/topeft/__init__.py" 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    add_dependency "${topeft_root}/topeft/modules/__init__.py" 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    add_dependency "${topeft_root}/topeft/modules/axes.py" 10473 824fca171e0b5a0f30de9ab9c41ce2f213feaffdebbafcdd3890e3eb01979495
    add_dependency "${topeft_root}/topeft/modules/axis_binning.py" 12016 f420c94949d60af355c6de0369808c0048ff49e43971e25b86a13108583c3947
    add_dependency "${topeft_root}/topeft/modules/compatibility.py" 816 73230293a33c13a4c7c8f7ee4bb598916f18c5ab57294be9788b7dea8d695f4b
    add_dependency "${topeft_root}/topeft/modules/data_driven_products.py" 73268 22b61dda7ff5d7a115e4031f5203e02c90234a1b82e47ba58dc4adec1028ebbf
    add_dependency "${topeft_root}/topeft/modules/datacard_tools.py" 94547 48bbb37f0d579d853abaefcde6f6a440da7b65c473b655d7e71da3d2bc01ad31
    add_dependency "${topeft_root}/topeft/modules/get_rate_systs.py" 3211 9a8e8372a1967d2d78380f39796745c634207aaa493ebf8c9b2bcb73f8235825
    add_dependency "${topeft_root}/topeft/modules/histogram_artifact.py" 118661 318f81f377bae96c8adb9e2fc6b30c053f1e43edfc98d2edbbfbfd4e07a301ad
    add_dependency "${topeft_root}/topeft/modules/missing_parton_contract.py" 24461 50920fc3744bdc898704bf89586c20c5b008a54e794e794133124b7ffc50dae0
    add_dependency "${topeft_root}/topeft/modules/nominal_schema.py" 33091 3e32d26212a34db4a03712d5450daf35871d532c3d36988184507771b3c6cf43
    add_dependency "${topeft_root}/topeft/modules/nonprompt_policy.py" 17876 692c4fda2b3584936f00896a7eb3160d69e5c7cc20dfb91c778290163023c8ba
    add_dependency "${topeft_root}/topeft/modules/paths.py" 226 b640638073b2678d2582a0e8622f9fded90a076d9a1022942dc3f6ea26cd94b9
    add_dependency "${topeft_root}/topeft/modules/production_sample_profile.py" 29681 b22a9ad24719949e9a93f063ecdc84ca9b96b79dcea2d19d7a4ef30878f78680
    add_dependency "${topeft_root}/topeft/modules/sumw2_policy.py" 26312 c8c433c5a75a6fab1ba1f0696ee98d64248aaacc4ba5233425d0f7374199ba10
    add_dependency "${topeft_root}/topeft/modules/yield_tools.py" 40306 328bc7e5d48fe5d08e45c8f204090fec6afc81e385e1b643aa4db9d1aad247e9
    add_dependency "${topeft_root}/topeft/params/cr_sr_plots_metadata.yml" 13302 65721a5e6cfb2a3ac6cbe3abd9313e0227357a46709ee1350bb5f958100c998b
    add_dependency "${topeft_root}/topeft/params/rate_systs.json" 1526 dfae896f93692a388033090ee5726201ba3a20bb9d204ab533efa3f803a018a5
    add_dependency "${topeft_root}/topeft/channels/ch_lst.json" 38344 5e1a49fbe018b821fe2e97bf7b6b78fb3335de13fb968baef3d08bd9c8061670
    add_dependency "${topcoffea_root}/topcoffea/__init__.py" 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    add_dependency "${topcoffea_root}/topcoffea/modules/__init__.py" 34 34144ce064d53f7836214976fbe493699569cf4e490edab6874029accfdf703d
    add_dependency "${topcoffea_root}/topcoffea/modules/HTMLGenerator.py" 21419 802403de5eef15eedef470a583c5d85a542d54c8152bfa7e22e7f5e3d73a0685
    add_dependency "${topcoffea_root}/topcoffea/modules/compat.py" 1558 cf53adda1d632662c3ecf4ae102771badf9c3d1f5f24b05d37443502a2dc24bf
    add_dependency "${topcoffea_root}/topcoffea/modules/eft_helper.py" 9208 abebee26207ba5c252529982fe50a76a18cf0f1acab36b2e1fd697d83ce10778
    add_dependency "${topcoffea_root}/topcoffea/modules/get_param_from_jsons.py" 299 8d5739b378f0f18052ce0df11f524ffe3eb60deb3a58c9ba02cd720c2f786a96
    add_dependency "${topcoffea_root}/topcoffea/modules/histEFT.py" 14518 2b8986c22a763186082504047a85ebc120569f2aa60a1dfd7f4b747cfa65f88d
    add_dependency "${topcoffea_root}/topcoffea/modules/hist_utils.py" 9270 ae04073c82be13e4dfff08791463d79256455d847a003897fc23ffae48112532
    add_dependency "${topcoffea_root}/topcoffea/modules/paths.py" 241 91eda5168f8751d9293faa2d4e666210b2cddec4b1544debe9bc5577943c4e09
    add_dependency "${topcoffea_root}/topcoffea/modules/sparseHist.py" 19398 a440fc15ecc7f301848995bf9b578c7aaba86259bc9007f557680be57133c87a
    add_dependency "${topcoffea_root}/topcoffea/modules/utils.py" 23554 b80113063ce3d2154de616515a2082165244ddcaa8dda654f3ea61748245309f
    add_dependency "${topcoffea_root}/topcoffea/scripts/make_html.py" 3598 d01b375ecf9c78dcc90e1e7de2cd7b4e9a4cea4e5de0e9491464fa24282912e5
    add_dependency "${topcoffea_root}/topcoffea/params/params.json" 1807 8020a216442a6c17940b3a3f23007c2ad7626b4dff43d4ff646eb37483be4c39
}

add_task() {
    task_ids+=("$1")
    task_blocks+=("$2")
    task_pkls+=("$3")
    task_output_dirs+=("$4")
    task_year_aliases+=("run3")
}

initialize_task_definitions() {
    local input_root="/groups/klannon/apiccine/run3_full_260819_v2_corrected_np_260822"
    local output_root="${topeft_root}/histos/SR_preappr_Aug31_resilient_run3/merged_njets"
    add_task "run3_block_2_processing" "block_2" \
        "${input_root}/2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_m_offZ_njets-lj0pt-ptll-lt_np.pkl.gz" \
        "${output_root}/block_2"
    add_task "run3_block_3_processing" "block_3" \
        "${input_root}/2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_p_offZ_njets-lj0pt-ptll-lt_np.pkl.gz" \
        "${output_root}/block_3"
    add_task "run3_block_4_processing" "block_4" \
        "${input_root}/2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_onZ_tau_njets-lj0pt-ptz-lt_np.pkl.gz" \
        "${output_root}/block_4"
    add_task "run3_block_5_processing" "block_5" \
        "${input_root}/2022-2022EE-2023-2023BPixSRs_run3-full-260819-v2_3l_fwd_njets-lj0pt-ptz-lt_np.pkl.gz" \
        "${output_root}/block_5"
}

validate_task_definitions() {
    local errors=0
    local index
    declare -A seen=()
    [[ ${#task_ids[@]} -eq 4 ]] || { echo "Preflight error: expected four continuation tasks." >&2; errors=1; }
    for index in "${!task_ids[@]}"; do
        [[ "${task_ids[$index]}" == "run3_block_$((index + 2))_processing" ]] || { echo "Preflight error: task identity/order mismatch at $((index + 1))." >&2; errors=1; }
        [[ "${task_blocks[$index]}" == "block_$((index + 2))" ]] || { echo "Preflight error: block mismatch at $((index + 1))." >&2; errors=1; }
        [[ -z "${seen[${task_ids[$index]}]+x}" ]] || { echo "Preflight error: duplicate task ${task_ids[$index]}." >&2; errors=1; }
        seen["${task_ids[$index]}"]=1
        if [[ "${attempt_backend}" == "production" ]]; then
            [[ "${task_output_dirs[$index]}" == "${topeft_root}/histos/SR_preappr_Aug31_resilient_run3/merged_njets/${task_blocks[$index]}" ]] || { echo "Preflight error: output directory mismatch for ${task_ids[$index]}." >&2; errors=1; }
        fi
    done
    [[ ${workers} -eq 6 ]] || { echo "Preflight error: worker count must be six." >&2; errors=1; }
    [[ ${errors} -eq 0 ]]
}

configure_runtime_paths() {
    attempts_path="${runtime_root}/attempt_events.tsv"
    memory_samples_path="${runtime_root}/memory_samples.tsv"
    campaign_summary_path="${runtime_root}/campaign_summary.tsv"
    task_plan_path="${runtime_root}/task_plan.tsv"
    fitting_snapshot_path="${runtime_root}/fitting_manifest_pre.tsv"
    output_inventory_path="${runtime_root}/processing_output_inventory.tsv"
    logs_dir="${runtime_root}/logs"
    time_dir="${runtime_root}/time"
}

shell_join() {
    local item
    local -a quoted=()
    for item in "$@"; do
        printf -v item '%q' "${item}"
        quoted+=("${item}")
    done
    local IFS=' '
    printf '%s' "${quoted[*]}"
}

stat_mtime_ns() {
    local timestamp
    timestamp=$(stat -c '%y' -- "$1") || return 1
    date --date="${timestamp}" +'%s%N'
}

build_attempt_command() {
    local index=$1
    local block_number=${task_blocks[$index]#block_}
    if [[ "${attempt_backend}" == "production" ]]; then
        attempt_command=(
            "${codex_run_path}" /bin/bash --noprofile --norc -c
            'python_bin=$1; shift; export PYTHON_BIN="$python_bin"; exec "$@"'
            "srplot008v7_run3_block${block_number}" "${python_path}" "${plotter_path}"
            -f "${task_pkls[$index]}"
            -o "${task_output_dirs[$index]}"
            -y "${task_year_aliases[$index]}"
            --sr
            --workers "${workers}"
            --channel-output merged-njets
            --binning processing
        )
        return 0
    fi

    local stub_output="${task_output_dirs[$index]}/${task_ids[$index]}_processing.png"
    if [[ "${stub_scenario}" == "process_identity_setup_failure" && ${index} -eq 0 ]]; then
        attempt_command=(/bin/bash --noprofile --norc -c 'echo identity_failure_stub_started; sleep 0.1')
    elif [[ "${stub_scenario}" == "known_failure" && ${index} -eq 1 ]]; then
        attempt_command=(/bin/bash --noprofile --norc -c 'echo deterministic_stub_failure; exit 7')
    elif [[ "${stub_scenario}" == "ambiguous" && ${index} -eq 1 ]]; then
        attempt_command=(/bin/bash --noprofile --norc -c 'echo ambiguous_stub_started; sleep 0.3')
    else
        attempt_command=(/bin/bash --noprofile --norc -c 'printf "stub_processing_output\n" >"$1"' stub "${stub_output}")
    fi
}

print_task_plan() {
    local index
    printf 'execution_order\tlogical_task\tblock\tpkl\tchannel_output\tbinning\tworkers\toutput_directory\texact_command\n'
    for index in "${!task_ids[@]}"; do
        build_attempt_command "${index}"
        printf '%d\t%s\t%s\t%s\tmerged-njets\tprocessing\t%d\t%s\t%s\n' \
            "$((index + 1))" "${task_ids[$index]}" "${task_blocks[$index]}" \
            "${task_pkls[$index]}" "${workers}" "${task_output_dirs[$index]}" \
            "$(shell_join "${attempt_command[@]}")"
    done
}

verify_static_source_contract() {
    local errors=0
    local index path observed_size observed_sha
    [[ ${#dependency_paths[@]} -gt 0 ]] || { echo "Preflight error: dependency contract is empty." >&2; errors=1; }
    for index in "${!dependency_paths[@]}"; do
        path=${dependency_paths[$index]}
        [[ -f "${path}" && ! -L "${path}" ]] || { echo "Dependency-integrity error: unavailable regular file ${path}." >&2; errors=1; continue; }
        observed_size=$(stat -c '%s' "${path}" 2>/dev/null)
        observed_sha=$(sha256sum "${path}" 2>/dev/null | awk '{print $1}')
        [[ "${observed_size}" == "${dependency_sizes[$index]}" && "${observed_sha}" == "${dependency_sha256s[$index]}" ]] || {
            echo "Dependency-integrity error: byte identity mismatch for ${path}." >&2
            errors=1
        }
    done
    [[ -x "${plotter_path}" && -x "${codex_run_path}" && -x "${python_path}" && -x "${sampler_path}" ]] || { echo "Preflight error: required executable is unavailable." >&2; errors=1; }
    [[ -x /usr/bin/setsid && -x /usr/bin/time ]] || { echo "Preflight error: setsid/time unavailable." >&2; errors=1; }
    command -v file >/dev/null 2>&1 || { echo "Preflight error: file command unavailable." >&2; errors=1; }
    validate_task_definitions || errors=1
    [[ ${errors} -eq 0 ]]
}

verify_block1_acceptance() {
    local errors=0 rows=0
    local snapshot_timestamp block path file_type size mtime_ns expected_sha snapshot_read_state current_sha
    declare -A accepted_paths=()
    while IFS=$'\t' read -r snapshot_timestamp block path file_type size mtime_ns expected_sha snapshot_read_state; do
        [[ "${snapshot_timestamp}" == "snapshot_timestamp" ]] && continue
        ((rows += 1))
        accepted_paths["${path}"]=1
        [[ "${block}" == "block_1" && "${file_type}" == "regular_file" && "${snapshot_read_state}" == "stable_during_read" ]] || { echo "Block1-acceptance error: malformed accepted row ${path}." >&2; errors=1; continue; }
        [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%s' "${path}" 2>/dev/null)" == "${size}" ]] || { echo "Block1-acceptance error: type/size mismatch ${path}." >&2; errors=1; continue; }
        current_sha=$(sha256sum "${path}" 2>/dev/null | awk '{print $1}')
        [[ "${current_sha}" == "${expected_sha}" ]] || { echo "Block1-acceptance error: hash mismatch ${path}." >&2; errors=1; }
    done <"${accepted_block1_manifest}"
    local expected_rows=111 block1_dir="${topeft_root}/histos/SR_preappr_Aug31_resilient_run3/merged_njets/block_1"
    [[ "${attempt_backend}" == "stub" ]] && expected_rows=1 && block1_dir="${fixture_root}/block_1"
    [[ ${rows} -eq ${expected_rows} ]] || { echo "Block1-acceptance error: expected ${expected_rows} accepted rows, found ${rows}." >&2; errors=1; }
    while IFS= read -r path; do
        [[ -n "${accepted_paths[${path}]+x}" ]] || { echo "Block1-acceptance error: unexpected processing path ${path}." >&2; errors=1; }
    done < <(find "${block1_dir}" -mindepth 1 \( -type f -o -type l \) -name '*_processing.*' -print)
    [[ ${errors} -eq 0 ]]
}

verify_input_identities() {
    local errors=0
    local rows=0
    local block artifact_class path expected_size observed_size expected_mtime observed_mtime expected_sha observed_sha identity_source sidecar_binding status current_mtime
    while IFS=$'\x1f' read -r block artifact_class path expected_size observed_size expected_mtime observed_mtime expected_sha observed_sha identity_source sidecar_binding status; do
        [[ "${block}" == "block" ]] && continue
        ((rows += 1))
        [[ -r "${path}" ]] || { echo "Preflight error: unreadable accepted input ${path}." >&2; errors=1; continue; }
        [[ "$(stat -c '%s' "${path}")" == "${expected_size}" ]] || { echo "Preflight error: size mismatch for ${path}." >&2; errors=1; }
        if [[ "${identity_source}" == "direct_sha256" ]]; then
            [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]] || { echo "Preflight error: direct hash mismatch for ${path}." >&2; errors=1; }
        elif [[ "${identity_source}" == "maintained_sidecar" ]]; then
            current_mtime=$(stat_mtime_ns "${path}" 2>/dev/null)
            [[ -n "${current_mtime}" && "${current_mtime}" == "${expected_mtime}" ]] || { echo "Preflight error: mtime_ns mismatch for ${path}." >&2; errors=1; }
            [[ "${sidecar_binding}" == "maintained_sidecar_bound" && "${status}" == "passed" ]] || { echo "Preflight error: maintained-sidecar binding is invalid for ${path}." >&2; errors=1; }
        else
            echo "Preflight error: unsupported identity authority ${identity_source} for ${path}." >&2
            errors=1
        fi
    done < <(tr '\t' '\037' <"${accepted_input_inventory}")
    [[ ${rows} -eq 10 ]] || { echo "Preflight error: expected ten input identity rows, found ${rows}." >&2; errors=1; }
    [[ ${errors} -eq 0 ]]
}

verify_fitting_manifest() {
    local destination=${1:-}
    local errors=0
    local rows=0
    local block path relative size mtime observed_sha accepted_sha status current_sha
    if [[ -n "${destination}" ]]; then
        printf 'block\tpath\tsize_bytes\tmtime_epoch_seconds\tsha256\n' >"${destination}" || return 1
    fi
    while IFS=$'\t' read -r block path relative size mtime observed_sha accepted_sha status; do
        [[ "${block}" == "block" ]] && continue
        ((rows += 1))
        current_sha=$(sha256sum "${path}" 2>/dev/null | awk '{print $1}')
        if [[ -z "${current_sha}" || "${current_sha}" != "${observed_sha}" || "${current_sha}" != "${accepted_sha}" ]]; then
            echo "Fitting-manifest error: ${path}." >&2
            errors=1
        fi
        if [[ -n "${destination}" && -n "${current_sha}" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\n' "${block}" "${path}" "$(stat -c '%s' "${path}")" "$(stat -c '%Y' "${path}")" "${current_sha}" >>"${destination}"
        fi
    done <"${accepted_fitting_manifest}"
    local expected_rows=358
    [[ "${attempt_backend}" == "stub" ]] && expected_rows=5
    [[ ${rows} -eq ${expected_rows} ]] || { echo "Fitting-manifest error: expected ${expected_rows} rows, found ${rows}." >&2; errors=1; }
    [[ ${errors} -eq 0 ]]
}

validate_processing_collisions() {
    local errors=0
    local rows=0
    local block source_fitting expected_path output_kind preexisting status
    while IFS=$'\t' read -r block source_fitting expected_path output_kind preexisting status; do
        [[ "${block}" == "block" ]] && continue
        [[ "${block}" =~ ^block_[2-5]$ ]] || continue
        ((rows += 1))
        if [[ -e "${expected_path}" ]]; then
            echo "Processing-collision error: intended output already exists: ${expected_path}." >&2
            errors=1
        fi
    done <"${expected_processing_oracle}"
    local expected_rows=247
    [[ "${attempt_backend}" == "stub" ]] && expected_rows=4
    [[ ${rows} -eq ${expected_rows} ]] || { echo "Processing-collision error: expected ${expected_rows} oracle rows, found ${rows}." >&2; errors=1; }
    [[ ${errors} -eq 0 ]]
}

validate_block_directory_membership() {
    local errors=0 output_dir current_path block1_dir
    block1_dir="${topeft_root}/histos/SR_preappr_Aug31_resilient_run3/merged_njets/block_1"
    [[ "${attempt_backend}" == "stub" ]] && block1_dir="${fixture_root}/block_1"
    local -a protected_dirs=("${block1_dir}" "${task_output_dirs[@]}")
    for output_dir in "${protected_dirs[@]}"; do
        while IFS= read -r current_path; do
            if awk -F '\t' -v wanted="${current_path}" 'NR > 1 && $2 == wanted {found=1} END {exit !found}' "${accepted_fitting_manifest}"; then
                continue
            fi
            if awk -F '\t' -v wanted="${current_path}" 'NR > 1 && $3 == wanted {found=1} END {exit !found}' "${expected_processing_oracle}"; then
                continue
            fi
            if awk -F '\t' -v wanted="${current_path}" 'NR > 1 && $3 == wanted {found=1} END {exit !found}' "${accepted_block1_manifest}"; then
                continue
            fi
            echo "Directory-membership error: unexpected path in protected block directory: ${current_path}." >&2
            errors=1
        done < <(find "${output_dir}" -mindepth 1 \( -type f -o -type l \) -print)
    done
    [[ ${errors} -eq 0 ]]
}

production_preflight() {
    local errors=0
    verify_static_source_contract || errors=1
    verify_input_identities || errors=1
    verify_block1_acceptance || errors=1
    verify_fitting_manifest || errors=1
    for output_dir in "${task_output_dirs[@]}"; do
        [[ -d "${output_dir}" && -w "${output_dir}" ]] || { echo "Preflight error: expected parent block directory unavailable: ${output_dir}." >&2; errors=1; }
    done
    validate_processing_collisions || errors=1
    validate_block_directory_membership || errors=1
    [[ ! -e "${runtime_root}" ]] || { echo "Preflight error: runtime root already exists and requires recovery classification: ${runtime_root}." >&2; errors=1; }
    [[ ${errors} -eq 0 ]]
}

initialize_runtime_files() {
    mkdir -p "${logs_dir}" "${time_dir}" || return 1
    printf 'logical_task\tattempt_id\tevent\ttimestamp\tpid\tppid\tprocess_group_id\tsession_id\tprocess_start_ticks\tprocess_start_time\tcommand_sha256\texit_code\telapsed_seconds\tterminal_state\texact_command\tlog_path\tlog_sha256\toutput_directory\ttime_report_path\tmemory_summary_path\toutput_validation_state\tfitting_manifest_integrity_state\tblock1_acceptance_state\tdependency_integrity_state\n' >"${attempts_path}" || return 1
    printf 'task_id\tattempt_kind\ttimestamp\tprocess_group_id\tprocess_count\taggregate_rss_kb\taggregate_pss_kb\tpss_status\tsystem_mem_available_kb\n' >"${memory_samples_path}" || return 1
    printf 'logical_task\tattempt_id\tpath\tsize_bytes\tsha256\tsemantic_state\n' >"${output_inventory_path}" || return 1
    print_task_plan >"${task_plan_path}" || return 1
    verify_fitting_manifest "${fitting_snapshot_path}" || return 1
    return 0
}

append_attempt_event() {
    local logical_task=$1 event=$2 timestamp=$3 exit_code=$4 elapsed=$5 terminal_state=$6 output_validation=$7 fitting_integrity=$8
    local log_sha=""
    [[ -f "${attempt_log_path}" ]] && log_sha=$(sha256sum "${attempt_log_path}" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${logical_task}" "${attempt_id}" "${event}" "${timestamp}" "${attempt_pid}" "${attempt_ppid}" \
        "${attempt_process_group_id}" "${attempt_session_id}" "${attempt_process_start_ticks}" "${attempt_process_start_time}" \
        "${attempt_command_sha256}" "${exit_code}" "${elapsed}" "${terminal_state}" "${attempt_command_text}" \
        "${attempt_log_path}" "${log_sha}" "${current_output_dir}" "${attempt_time_report_path}" \
        "${attempt_memory_summary_path}" "${output_validation}" "${fitting_integrity}" "passed" "passed" >>"${attempts_path}"
}

validate_task_outputs() {
    local task_id=$1 block=$2 output_dir=$3
    local errors=0 expected_count=0 actual_count=0
    local oracle_block source_fitting expected_path output_kind preexisting status semantic current_sha
    while IFS=$'\t' read -r oracle_block source_fitting expected_path output_kind preexisting status; do
        [[ "${oracle_block}" == "block" ]] && continue
        [[ "${oracle_block}" == "${block}" ]] || continue
        ((expected_count += 1))
        if [[ ! -s "${expected_path}" ]]; then
            errors=1
            continue
        fi
        if [[ "${attempt_backend}" == "production" && "${output_kind}" == "png" ]]; then
            file -b "${expected_path}" | grep -q '^PNG image data' || { errors=1; continue; }
        fi
        semantic="nonempty_stub"
        [[ "${attempt_backend}" == "production" ]] && semantic="readable_${output_kind}"
        current_sha=$(sha256sum "${expected_path}" | awk '{print $1}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${attempt_id}" "${expected_path}" "$(stat -c '%s' "${expected_path}")" "${current_sha}" "${semantic}" >>"${output_inventory_path}"
    done <"${expected_processing_oracle}"
    while IFS= read -r actual_path; do
        [[ -n "${actual_path}" ]] || continue
        if awk -F '\t' -v wanted="${actual_path}" 'NR > 1 && $2 == wanted {found=1} END {exit !found}' "${accepted_fitting_manifest}"; then
            continue
        fi
        if awk -F '\t' -v wanted="${actual_path}" 'NR > 1 && $3 == wanted {found=1} END {exit !found}' "${expected_processing_oracle}"; then
            ((actual_count += 1))
        else
            echo "Output-validation error: unexpected created path ${actual_path}." >&2
            errors=1
        fi
    done < <(find "${output_dir}" -mindepth 1 \( -type f -o -type l \) -print)
    [[ ${expected_count} -gt 0 && ${actual_count} -eq ${expected_count} ]] || errors=1
    [[ ${errors} -eq 0 ]]
}

run_attempt() {
    local index=$1
    local task_id=${task_ids[$index]}
    local block=${task_blocks[$index]}
    current_output_dir=${task_output_dirs[$index]}
    local start_time end_time start_seconds elapsed sampler_pid sampler_exit launch_state poll_index
    attempt_id="${task_id}_attempt_001"
    attempt_log_path="${logs_dir}/${attempt_id}.log"
    attempt_time_report_path="${time_dir}/${attempt_id}.txt"
    attempt_memory_summary_path="${time_dir}/${attempt_id}_memory_summary.tsv"
    last_logical_status=""
    attempt_exit_code=""
    attempt_command_text=""
    attempt_command_sha256=""
    attempt_pid=""
    attempt_ppid=""
    attempt_process_group_id=""
    attempt_session_id=""
    attempt_process_start_ticks=""
    attempt_process_start_time=""
    build_attempt_command "${index}" || return 1
    attempt_command_text=$(shell_join "${attempt_command[@]}")
    attempt_command_sha256=$(printf '%s' "${attempt_command_text}" | sha256sum | awk '{print $1}')
    : >"${attempt_log_path}"
    : >"${attempt_time_report_path}"
    : >"${attempt_memory_summary_path}"
    start_time=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    start_seconds=${SECONDS}
    append_attempt_event "${task_id}" "started" "${start_time}" "" "" "started" "not_evaluated" "precheck_passed"

    /usr/bin/setsid /bin/bash --noprofile --norc -c '
        kill -STOP "$$"
        log_path=$1
        shift
        exec > >(tee "${log_path}") 2>&1
        exec "$@"
    ' run3_processing_attempt "${attempt_log_path}" /usr/bin/time -v -o "${attempt_time_report_path}" "${attempt_command[@]}" &
    attempt_pid=$!
    launch_state=""
    for poll_index in {1..50}; do
        launch_state=$(ps -o stat= -p "${attempt_pid}" 2>/dev/null)
        [[ "${launch_state}" == *T* ]] && break
        sleep 0.1
    done
    attempt_ppid=$(ps -o ppid= -p "${attempt_pid}" 2>/dev/null | tr -d ' ')
    attempt_process_group_id=$(ps -o pgid= -p "${attempt_pid}" 2>/dev/null | tr -d ' ')
    attempt_session_id=$(ps -o sid= -p "${attempt_pid}" 2>/dev/null | tr -d ' ')
    attempt_process_start_ticks=$(sed 's/.*) //' "/proc/${attempt_pid}/stat" 2>/dev/null | awk '{print $20}')
    attempt_process_start_time=$(ps -o lstart= -p "${attempt_pid}" 2>/dev/null | sed 's/^ *//')
    if [[ "${attempt_backend}" == "stub" && "${stub_scenario}" == "process_identity_setup_failure" && ${index} -eq 0 ]]; then
        attempt_ppid=""
        attempt_process_group_id=""
        attempt_session_id=""
        attempt_process_start_ticks=""
        attempt_process_start_time=""
    fi
    [[ -n "${attempt_ppid}" && -n "${attempt_process_group_id}" && -n "${attempt_session_id}" && -n "${attempt_process_start_ticks}" && -n "${attempt_process_start_time}" ]] || {
        echo "Process-identity error: could not bind durable identity for ${task_id}." >&2
        last_logical_status="ambiguous_interruption"
        append_attempt_event "${task_id}" "identity_binding_failed" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "" "" "${last_logical_status}" "not_evaluated" "precheck_passed"
        if [[ "${attempt_backend}" == "stub" ]]; then
            kill -CONT "${attempt_pid}" 2>/dev/null || true
        fi
        return 75
    }
    append_attempt_event "${task_id}" "process_started" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "" "" "running" "not_evaluated" "precheck_passed"

    "${codex_run_path}" /bin/bash --noprofile --norc -c 'exec "$@"' run3_processing_sampler \
        "${python_path}" "${sampler_path}" \
        --process-group-id "${attempt_pid}" --task-id "${task_id}" --attempt-kind processing_primary \
        --samples-path "${memory_samples_path}" --summary-path "${attempt_memory_summary_path}" \
        --interval-seconds 0.1 --startup-grace-seconds 1.0 &
    sampler_pid=$!
    kill -CONT "${attempt_pid}" 2>/dev/null || true

    if [[ "${attempt_backend}" == "stub" && "${stub_scenario}" == "ambiguous" && ${index} -eq 1 ]]; then
        last_logical_status="ambiguous_interruption"
        return 75
    fi

    if wait "${attempt_pid}"; then
        attempt_exit_code=0
    else
        attempt_exit_code=$?
    fi
    if wait "${sampler_pid}"; then sampler_exit=0; else sampler_exit=$?; fi
    end_time=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    elapsed=$((SECONDS - start_seconds))
    local fitting_integrity="passed"
    local output_validation="not_evaluated"
    verify_fitting_manifest || fitting_integrity="failed"
    if [[ ${attempt_exit_code} -eq 0 && "${fitting_integrity}" == "passed" ]]; then
        if validate_task_outputs "${task_id}" "${block}" "${current_output_dir}"; then
            output_validation="passed"
            last_logical_status="success_processing"
        else
            output_validation="failed"
            last_logical_status="failed_output_validation"
        fi
    elif [[ "${fitting_integrity}" == "failed" ]]; then
        last_logical_status="failed_fitting_immutability"
    else
        last_logical_status="failed_processing"
    fi
    append_attempt_event "${task_id}" "terminal" "${end_time}" "${attempt_exit_code}" "${elapsed}" "${last_logical_status}" "${output_validation}" "${fitting_integrity}"
    return 0
}

reconstruct_statuses() {
    local index task terminal pid pgid sid ticks observed_pgid observed_sid observed_ticks
    reconstructed_statuses=()
    if [[ ! -r "${attempts_path}" ]]; then
        for task in "${task_ids[@]}"; do reconstructed_statuses+=("not_started"); done
        return 0
    fi
    for index in "${!task_ids[@]}"; do
        task=${task_ids[$index]}
        terminal=$(awk -F '\t' -v wanted="${task}" '$1 == wanted && $3 == "terminal" {state=$14} END {print state}' "${attempts_path}")
        if [[ -n "${terminal}" ]]; then
            reconstructed_statuses+=("${terminal}")
            continue
        fi
        pid=$(awk -F '\t' -v wanted="${task}" '$1 == wanted && $3 == "process_started" {value=$5} END {print value}' "${attempts_path}")
        pgid=$(awk -F '\t' -v wanted="${task}" '$1 == wanted && $3 == "process_started" {value=$7} END {print value}' "${attempts_path}")
        sid=$(awk -F '\t' -v wanted="${task}" '$1 == wanted && $3 == "process_started" {value=$8} END {print value}' "${attempts_path}")
        ticks=$(awk -F '\t' -v wanted="${task}" '$1 == wanted && $3 == "process_started" {value=$9} END {print value}' "${attempts_path}")
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            observed_pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')
            observed_sid=$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ')
            observed_ticks=$(sed 's/.*) //' "/proc/${pid}/stat" 2>/dev/null | awk '{print $20}')
            if [[ -n "${pgid}" && "${observed_pgid}" == "${pgid}" && "${observed_sid}" == "${sid}" && "${observed_ticks}" == "${ticks}" ]]; then
                reconstructed_statuses+=("active_attempt")
            else
                reconstructed_statuses+=("ambiguous_interruption")
            fi
        elif awk -F '\t' -v wanted="${task}" '$1 == wanted && ($3 == "started" || $3 == "process_started") {found=1} END {exit !found}' "${attempts_path}"; then
            reconstructed_statuses+=("ambiguous_interruption")
        else
            reconstructed_statuses+=("not_started")
        fi
    done
}

status_campaign() {
    local index
    reconstruct_statuses
    printf 'logical_task\tstatus\n'
    printf 'run3_block_1_processing\texternally_accepted_existing_success\n'
    for index in "${!task_ids[@]}"; do
        printf '%s\t%s\n' "${task_ids[$index]}" "${reconstructed_statuses[$index]}"
    done
}

finalize_campaign() {
    local index state success=0 failure=0 ambiguous=0 not_started=0 final_exit=0
    reconstruct_statuses
    if [[ ! -d "${runtime_root}" ]]; then
        printf 'finalization_state\tno_runtime_state\n'
        printf 'runtime_root\t%s\n' "${runtime_root}"
        return 3
    fi
    {
        printf 'metric\tvalue\n'
        printf 'logical_tasks\t5\n'
        printf 'externally_accepted_existing_successes\t1\n'
        printf 'continuation_tasks\t4\n'
        printf 'workers_per_task\t6\n'
        printf 'campaign_task_parallelism\t1\n'
        for index in "${!task_ids[@]}"; do
            state=${reconstructed_statuses[$index]}
            printf 'task_%d_%s\t%s\n' "$((index + 2))" "${task_ids[$index]}" "${state}"
            case "${state}" in
                success_processing) ((success += 1)) ;;
                failed_*) ((failure += 1)) ;;
                active_attempt|ambiguous_interruption) ((ambiguous += 1)) ;;
                not_started) ((not_started += 1)) ;;
            esac
        done
        if [[ ${ambiguous} -gt 0 ]]; then final_exit=75
        elif [[ ${failure} -gt 0 ]]; then final_exit=1
        elif [[ ${not_started} -gt 0 ]]; then final_exit=3
        else final_exit=0
        fi
        printf 'known_successes\t%d\n' "${success}"
        printf 'known_failures\t%d\n' "${failure}"
        printf 'ambiguous_or_active\t%d\n' "${ambiguous}"
        printf 'not_started\t%d\n' "${not_started}"
        printf 'final_exit_code\t%d\n' "${final_exit}"
    } >"${campaign_summary_path}"
    cat "${campaign_summary_path}"
    return "${final_exit}"
}

run_campaign() {
    local index final_exit=0 attempt_rc=0
    logical_statuses=()
    for index in "${!task_ids[@]}"; do
        echo "Continuation task $((index + 1))/4: ${task_ids[$index]}."
        last_logical_status=""
        attempt_rc=0
        if run_attempt "${index}"; then :; else attempt_rc=$?; fi
        if [[ ${attempt_rc} -ne 0 ]]; then
            if [[ "${last_logical_status}" == failed_* ]]; then
                logical_statuses+=("${last_logical_status}")
                echo "Campaign stopped on known failure; no later task will run." >&2
                finalize_campaign >/dev/null || true
                return 1
            fi
            if [[ "${last_logical_status}" != "ambiguous_interruption" ]]; then
                last_logical_status="ambiguous_interruption"
                append_attempt_event "${task_ids[$index]}" "unclassified_nonzero" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "" "" "${last_logical_status}" "not_evaluated" "not_evaluated"
            fi
            logical_statuses+=("${last_logical_status}")
            echo "Campaign stopped on unclassified or ambiguous nonzero attempt; no later task will run." >&2
            finalize_campaign >/dev/null || true
            return 75
        fi
        if [[ -z "${last_logical_status}" ]]; then
            last_logical_status="ambiguous_interruption"
            append_attempt_event "${task_ids[$index]}" "missing_current_classification" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "" "" "${last_logical_status}" "not_evaluated" "not_evaluated"
        fi
        logical_statuses+=("${last_logical_status}")
        if [[ "${last_logical_status}" == "ambiguous_interruption" ]]; then
            echo "Campaign stopped on ambiguous interruption; no later task will run." >&2
            finalize_campaign >/dev/null || true
            return 75
        fi
        if [[ "${last_logical_status}" == failed_* ]]; then
            echo "Campaign stopped on known failure; no later task will run." >&2
            finalize_campaign >/dev/null || true
            return 1
        fi
    done
    if finalize_campaign; then final_exit=0; else final_exit=$?; fi
    return "${final_exit}"
}

initialize_stub_fixture() {
    local index block task_id output fitting expected fitting_sha accepted_output accepted_sha dependency_file dependency_sha input_path input_sidecar input_size input_mtime sidecar_size sidecar_mtime sidecar_sha snapshot_read_state
    [[ -n "${fixture_root}" && "${fixture_root}" == /* ]] || { echo "Stub validation requires an absolute --fixture-root." >&2; return 3; }
    [[ ! -e "${fixture_root}" ]] || { echo "Stub fixture root already exists: ${fixture_root}." >&2; return 3; }
    mkdir -p "${fixture_root}"
    accepted_input_inventory="${fixture_root}/input_inventory.tsv"
    accepted_fitting_manifest="${fixture_root}/accepted_fitting_manifest.tsv"
    expected_processing_oracle="${fixture_root}/expected_processing_paths.tsv"
    accepted_block1_manifest="${fixture_root}/accepted_block1_manifest.tsv"
    printf 'block\tartifact_class\tpath\texpected_size_bytes\tobserved_size_bytes\texpected_mtime_ns\tobserved_mtime_ns\texpected_sha256\tobserved_sha256\tidentity_source\tsidecar_binding\tstatus\n' >"${accepted_input_inventory}"
    printf 'block\tpath\trelative_path\tsize_bytes\tmtime_ns\tsha256\taccepted_sha256\tstatus\n' >"${accepted_fitting_manifest}"
    printf 'block\tsource_fitting_path\texpected_processing_path\toutput_kind\tpreexisting\tstatus\n' >"${expected_processing_oracle}"
    printf 'snapshot_timestamp\tblock\tpath\tfile_type\tsize_bytes\tmtime_ns\tsha256\tsnapshot_read_state\n' >"${accepted_block1_manifest}"
    for block_number in 1 2 3 4 5; do
        block="block_${block_number}"
        task_id="run3_block_${block_number}_processing"
        output="${fixture_root}/${block}"
        mkdir -p "${output}"
        input_path="${fixture_root}/${task_id}.pkl.gz"
        input_sidecar="${input_path}.metadata.json"
        printf 'stub input %s\n' "${task_id}" >"${input_path}"
        printf '{"stub": "%s"}\n' "${task_id}" >"${input_sidecar}"
        input_size=$(stat -c '%s' "${input_path}")
        input_mtime=$(stat_mtime_ns "${input_path}")
        sidecar_size=$(stat -c '%s' "${input_sidecar}")
        sidecar_mtime=$(stat_mtime_ns "${input_sidecar}")
        sidecar_sha=$(sha256sum "${input_sidecar}" | awk '{print $1}')
        printf '%s\taccepted_run3_input_pkl\t%s\t%s\t%s\t%s\t%s\tstub_maintained_hash\t\tmaintained_sidecar\tmaintained_sidecar_bound\tpassed\n' \
            "${block}" "${input_path}" "${input_size}" "${input_size}" "${input_mtime}" "${input_mtime}" >>"${accepted_input_inventory}"
        printf '%s\taccepted_run3_input_sidecar\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tdirect_sha256\tsidecar_direct_hash_checked\tpassed\n' \
            "${block}" "${input_sidecar}" "${sidecar_size}" "${sidecar_size}" "${sidecar_mtime}" "${sidecar_mtime}" "${sidecar_sha}" "${sidecar_sha}" >>"${accepted_input_inventory}"
        fitting="${output}/${task_id}_fitting_control.txt"
        printf 'immutable fitting control %s\n' "${task_id}" >"${fitting}"
        fitting_sha=$(sha256sum "${fitting}" | awk '{print $1}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tpassed\n' "${block}" "${fitting}" "$(basename "${fitting}")" "$(stat -c '%s' "${fitting}")" "$(stat -c '%Y' "${fitting}")" "${fitting_sha}" "${fitting_sha}" >>"${accepted_fitting_manifest}"
        if [[ ${block_number} -eq 1 ]]; then
            accepted_output="${output}/${task_id}.png"
            printf 'accepted block1 processing output\n' >"${accepted_output}"
            accepted_sha=$(sha256sum "${accepted_output}" | awk '{print $1}')
            snapshot_read_state="stable_during_read"
            [[ "${stub_scenario}" == "wrong_block1_snapshot_state" ]] && snapshot_read_state="readable"
            printf 'stub\tblock_1\t%s\tregular_file\t%s\t%s\t%s\t%s\n' "${accepted_output}" "$(stat -c '%s' "${accepted_output}")" "$(( $(stat -c '%Y' "${accepted_output}") * 1000000000 ))" "${accepted_sha}" "${snapshot_read_state}" >>"${accepted_block1_manifest}"
        else
            index=$((block_number - 2))
            expected="${output}/${task_id}_processing.png"
            task_pkls[$index]="${input_path}"
            task_output_dirs[$index]="${output}"
            printf '%s\t%s\t%s\tpng\tno\tabsent\n' "${block}" "${fitting}" "${expected}" >>"${expected_processing_oracle}"
        fi
    done
    dependency_file="${fixture_root}/bound_plotting_dependency.txt"
    printf 'bound dependency\n' >"${dependency_file}"
    dependency_sha=$(sha256sum "${dependency_file}" | awk '{print $1}')
    dependency_paths=("${dependency_file}")
    dependency_sizes=("$(stat -c '%s' "${dependency_file}")")
    dependency_sha256s=("${dependency_sha}")
    if [[ "${stub_scenario}" == "collision" ]]; then
        printf 'preexisting collision\n' >"${fixture_root}/block_2/run3_block_2_processing_processing.png"
    elif [[ "${stub_scenario}" == "block1_mutation" ]]; then
        printf 'mutated after acceptance\n' >>"${fixture_root}/block_1/run3_block_1_processing.png"
    elif [[ "${stub_scenario}" == "fitting_mutation" ]]; then
        printf 'mutated after manifest\n' >>"${fixture_root}/block_1/run3_block_1_processing_fitting_control.txt"
    elif [[ "${stub_scenario}" == "dependency_mutation" ]]; then
        printf 'mutated dependency\n' >>"${dependency_file}"
    elif [[ "${stub_scenario}" == "same_size_changed_mtime" ]]; then
        touch -d '2030-01-01 00:00:00.123456789 UTC' "${fixture_root}/run3_block_2_processing.pkl.gz"
    elif [[ "${stub_scenario}" == "unrelated_tracked_mutation" ]]; then
        printf 'unrelated tracked-like change outside dependency set\n' >"${fixture_root}/unrelated_tracked_file.txt"
    fi
}

stub_preflight() {
    local errors=0 output_dir
    verify_static_source_contract || errors=1
    for output_dir in "${task_output_dirs[@]}"; do [[ -d "${output_dir}" ]] || errors=1; done
    verify_input_identities || errors=1
    verify_block1_acceptance || errors=1
    verify_fitting_manifest || errors=1
    validate_processing_collisions || errors=1
    validate_block_directory_membership || errors=1
    [[ ! -e "${runtime_root}" ]] || errors=1
    [[ ${errors} -eq 0 ]]
}

main() {
    local mode=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --print-plan|--execute|--status|--finalize)
                [[ -z "${mode}" ]] || { usage >&2; return 3; }
                mode=$1
                shift
                ;;
            --stub-scenario)
                [[ $# -ge 2 && -z "${mode}" ]] || { usage >&2; return 3; }
                mode="--stub-scenario"
                stub_scenario=$2
                attempt_backend="stub"
                shift 2
                ;;
            --runtime-root)
                [[ $# -ge 2 && "$2" == /* ]] || { usage >&2; return 3; }
                runtime_root=$2
                shift 2
                ;;
            --fixture-root)
                [[ $# -ge 2 && "$2" == /* ]] || { usage >&2; return 3; }
                fixture_root=$2
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                usage >&2
                return 3
                ;;
        esac
    done
    initialize_task_definitions
    initialize_dependency_contract
    configure_runtime_paths
    case "${mode}" in
        --print-plan)
            verify_static_source_contract || return 3
            print_task_plan
            ;;
        --execute)
            production_preflight || return 3
            initialize_runtime_files || return 3
            trap 'echo "Caller signal received; attempt state remains authoritative and no retry is automatic." >&2; exit 75' HUP INT TERM
            run_campaign
            ;;
        --status)
            status_campaign
            ;;
        --finalize)
            finalize_campaign
            ;;
        --stub-scenario)
            case "${stub_scenario}" in all_success|known_failure|ambiguous|process_identity_setup_failure|collision|block1_mutation|wrong_block1_snapshot_state|fitting_mutation|dependency_mutation|same_size_changed_mtime|unrelated_tracked_mutation|parent_exists) ;; *) echo "Unknown stub scenario: ${stub_scenario}." >&2; return 3 ;; esac
            initialize_stub_fixture || return $?
            if ! stub_preflight; then
                return 3
            fi
            initialize_runtime_files || return 3
            run_campaign
            ;;
        *)
            usage >&2
            return 3
            ;;
    esac
}

if main "$@"; then
    exit_code=0
else
    exit_code=$?
fi
exit "${exit_code}"
