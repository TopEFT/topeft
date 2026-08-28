#!/usr/bin/env bash

# Produce the nominal Run 3 JVM eta-phi CR diagnostics in one combined PKL.
# Keep dry_run enabled until a separately authorized production round.
set -euo pipefail

cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

output_dir="/groups/klannon/apiccine/preappr_v9_260729"
campaign_tag="ANv9_JVMCRttEtaPhi"
chunk_size="100000"
# A production launch requires explicit opt-in: ``dry_run=false ./run_jvm_cr.sh``.
dry_run="${dry_run:-true}"
ttgamma_sample_role_policy="split"

years=(2022 2022EE 2023 2023BPix)
hist_vars=(
    jet_eta_phi_before_veto
    jet_eta_phi_after_veto
)
category_groups=(
    2los_CRtt
)

clean_env_cache() {
    local cache_dir="topeft-envs"

    if [[ -d "$cache_dir" ]]; then
        find "$cache_dir" -maxdepth 1 \( -type f -o -type l \) -delete
    fi
}

validate_boolean() {
    local value="$1"
    local name="$2"

    if [[ "$value" != true && "$value" != false ]]; then
        echo "ERROR: ${name} must be true or false, got '${value}'." >&2
        exit 2
    fi
}

validate_years() {
    local year

    for year in "${years[@]}"; do
        case "$year" in
            2022|2022EE|2023|2023BPix) ;;
            *)
                echo "ERROR: unsupported Run 3 year '${year}'." >&2
                exit 2
                ;;
        esac
    done
}

validate_boolean "$dry_run" "dry_run"
validate_years

fullr3_command=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "$campaign_tag"
    -s "$chunk_size"
    --cr
    --hist-vars "${hist_vars[@]}"
    --category-groups "${category_groups[@]}"
    --ttgamma-sample-role-policy "$ttgamma_sample_role_policy"
    --sample-universe-wrapper "run_jvm_cr.sh -> fullR3_run.sh"
    -p "$output_dir"
    --all-analysis
)

if [[ "$dry_run" == true ]]; then
    fullr3_command+=(--dry-run)
else
    clean_env_cache
fi

printf 'Resolved command:\n'
printf ' %q' "${fullr3_command[@]}"
printf '\n'
printf 'Dry run: %s\n' "$dry_run"
printf 'Years: %s\n' "${years[*]}"
printf 'Category groups: %s\n' "${category_groups[*]}"
printf 'Histogram variables: %s\n' "${hist_vars[*]}"

start_time=$(date +%s)

if "${fullr3_command[@]}"; then
    exit_status=0
else
    exit_status=$?
fi

end_time=$(date +%s)

printf 'fullR3_run.sh exit status: %s\n' "$exit_status"
printf 'Elapsed seconds: %s\n' "$((end_time - start_time))"

exit "$exit_status"
