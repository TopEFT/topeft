#!/usr/bin/env bash
set -euo pipefail

cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

###############################################################################
# Global configuration
###############################################################################

output_dir="/groups/klannon/apiccine/photons"
chunk_size="100000"

# Use a branch-specific tag to avoid mixing baseline and feature outputs.
#
# Suggested usage:
#   on run3_test_mmerged_anpicci:
#     campaign_tag="photon_fbranch_baseline"
#
#   on te/ttgamma_conversion_photon_diagnostics:
#     campaign_tag="photon_fbranch_feature"
campaign_tag="photon_fbranch_baseline"

cr_pkl_base_tag="CR_${campaign_tag}"
sr_pkl_base_tag="SR_${campaign_tag}"

# Variables to be produced in both CR and SR pkls.
vars=(lj0pt ptz)
var_tag=$(IFS=-; echo "${vars[*]}")

###############################################################################
# CR configuration
###############################################################################

# Each entry is one independent year expression passed to fullR3_run.sh.
# Examples:
#   cr_year_sets=(2022EE 2018)        # two separate runs
#   cr_year_sets=("2022EE 2018")      # one combined run
cr_year_sets=(
  "2022EE 2018"
)

# Each entry is one independent subset of categories.
# The script will run all subsets for every year expression.
cr_category_sets=(
  "2los_CRtt"
  # "2l_CR 2los_CRtt 3l_CR"
  # "2los_CRZ 2l_CRflip"
  # "2los_1tau 1l_1tau_CRtt 1l_1tau_CRDY"
)

###############################################################################
# SR configuration
###############################################################################

# Each entry is one independent year expression passed to fullR3_run.sh.
# Examples:
#   sr_year_sets=(2022 2022EE 2023 2023BPix)              # separate runs
#   sr_year_sets=("2022 2022EE 2023 2023BPix")            # one combined run
sr_year_sets=(
  2022EE
  2018
  # "2022 2022EE 2023 2023BPix"
)

# Each entry is one independent subset of categories.
# Grouped entries produce one pkl per grouped set.
sr_category_sets=(
  "2l 2lss_1tau 2los_1tau 3l_m_offZ"
  "3l_p_offZ 3l_onZ_tau 3l_fwd 4l"
)

###############################################################################
# Helpers
###############################################################################

join_by() {
  local delimiter="$1"
  shift
  local IFS="$delimiter"
  echo "$*"
}

clean_env_cache() {
  if [[ -d topeft-envs ]]; then
    find topeft-envs -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -delete
  fi
}

assert_supported_year_expr() {
  local year_expr="$1"
  local year

  read -r -a years_in_expr <<< "${year_expr}"

  for year in "${years_in_expr[@]}"; do
    case "${year}" in
      2016APV|2016|2017|2018|2022|2022EE|2023|2023BPix) ;;
      *)
        cat >&2 <<EOF
ERROR: unsupported year token '${year}' in year expression '${year_expr}'.

Allowed year tokens:
  2016APV 2016 2017 2018 2022 2022EE 2023 2023BPix
EOF
        exit 1
        ;;
    esac
  done
}


run_cr_block() {
  local year_expr="$1"
  shift

  assert_supported_year_expr "${year_expr}"

  local years=()
  read -r -a years <<< "${year_expr}"

  local cats=("$@")
  local cat_tag
  local pkl_tag

  cat_tag=$(join_by - "${cats[@]}")
  pkl_tag="${cr_pkl_base_tag}_${cat_tag}_${var_tag}"

  echo "----------------------------------------"
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "----------------------------------------"

  clean_env_cache

  local cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pkl_tag}"
    -s "${chunk_size}"
    --cr
    --hist-vars "${vars[@]}"
    # --do-systs
    --do-np
    -p "${output_dir}"
    --category-groups "${cats[@]}"
    --suppress-forward-eta-stochastic-jer
    --all-analysis
    # --split-lep-flavor
  )

  echo "Executing:"
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}"

  echo "${year_expr} done"
  echo "----------------------------------------"
  echo
}

run_sr_block() {
  local year_expr="$1"
  shift

  assert_supported_year_expr "${year_expr}"

  local years=()
  read -r -a years <<< "${year_expr}"

  local cats=("$@")
  local cat_tag
  local pkl_tag

  cat_tag=$(join_by - "${cats[@]}")
  pkl_tag="${sr_pkl_base_tag}_${cat_tag}_${var_tag}"

  echo "----------------------------------------"
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "----------------------------------------"

  clean_env_cache

  local cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pkl_tag}"
    -s "${chunk_size}"
    --sr
    --hist-vars "${vars[@]}"
    # --do-systs
    --do-np
    -p "${output_dir}"
    --category-groups "${cats[@]}"
    --suppress-forward-eta-stochastic-jer
    --all-analysis
    # --split-lep-flavor
  )

  echo "Executing:"
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}"

  echo "${year_expr} done"
  echo "----------------------------------------"
  echo
}

# ###############################################################################
# # Main CR production
# ###############################################################################

# for year_expr in "${cr_year_sets[@]}"; do
#   for category_set in "${cr_category_sets[@]}"; do
#     read -r -a cats <<< "${category_set}"
#     run_cr_block "${year_expr}" "${cats[@]}"
#   done
# done

###############################################################################
# Main SR production
###############################################################################

for year_expr in "${sr_year_sets[@]}"; do
  for category_set in "${sr_category_sets[@]}"; do
    read -r -a cats <<< "${category_set}"
    run_sr_block "${year_expr}" "${cats[@]}"
  done
done