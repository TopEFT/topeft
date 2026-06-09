#!/usr/bin/env bash
set -euo pipefail

cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

###############################################################################
# Global configuration
###############################################################################

output_dir="/groups/klannon/apiccine/xANv4"
chunk_size="100000"

# Configurable first part of the pkl tag.
# pkl_base_tag="CR_t1met_fwdpt70_fulleta"
pkl_base_tag="CR_muonres"

# This tag should match what run_analysis.py will actually produce for --hist-list cr.
# With your current CR block:
#   hist_lst = ["ptz"]
#   + "ptz_wtau" when --tau-h-analysis or --all-analysis is enabled.
vars=(invmass l0ptcorr l0eta) # ptz ptz_wtau)
var_tag=$(IFS=-; echo "${vars[*]}")

years=(2023 2023BPix 2018)
# years=(2018) # 2016APV 2016 2017 2018)

# Each entry is one independent subset of categories.
# The script will run all subsets for every year.
category_sets=(
  # "2l_CR 2los_CRtt 3l_CR"
  "2los_CRZ 2l_CRflip"
  # "2los_1tau 1l_1tau_CRtt 1l_1tau_CRDY"
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

assert_run2_year() {
  case "$1" in
    2016APV|2016|2017|2018) ;;
    *)
      cat >&2 <<EOF
ERROR: this CR script is configured for Run 2 only, got year '$1'.

Before using this script for Run 3:
  - remove or replace this Run 2 year guard;
  - change pkl_base_tag from fwdpt40_noband_fulleta to the intended Run 3 tag,
    e.g. CR_t1met_fwdpt70_fulleta;
  - remove '--fwd-eta-band-pt-apply off' or replace it with the intended Run 3 policy
    (default auto already enables the eta-band pT tightening for Run 3);
  - re-check vars/category_sets for the intended Run 3 production.
EOF
      exit 1
      ;;
  esac
}

run_cr_block() {
  local year="$1"
  shift

  # assert_run2_year "${year}"

  local cats=("$@")
  local cat_tag
  local pkl_tag

  cat_tag=$(join_by - "${cats[@]}")
  pkl_tag="${pkl_base_tag}_${cat_tag}_${var_tag}"

  echo "----------------------------------------"
  echo "Year: ${year}"
  echo "Categories: ${cats[*]}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "----------------------------------------"

  clean_env_cache

local cmd=(
  ./fullR3_run.sh
  -y "${year}"
  -t "${pkl_tag}"
  -s "${chunk_size}"
  --cr
  --hist-vars "${vars[@]}"
  --do-systs
  # --do-np
  -p "${output_dir}"
  --category-groups "${cats[@]}"
  --suppress-forward-eta-stochastic-jer
  --tau-h-analysis
  --split-lep-flavor
)

  echo "Executing:"
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}"

  echo "${year} done"
  echo "----------------------------------------"
  echo
}

###############################################################################
# Main CR production
###############################################################################

for year in "${years[@]}"; do
  for category_set in "${category_sets[@]}"; do
    read -r -a cats <<< "${category_set}"
    run_cr_block "${year}" "${cats[@]}"
  done
done

###############################################################################
# Parking area for future CR runs
###############################################################################

# If you later change the run_analysis.py CR hist list, update vars accordingly.
#
# Example if you restore lj0pt/met/fwd0eta:
#
# vars=(lj0pt met fwd0eta ptz ptz_wtau)
# var_tag=$(IFS=-; echo "${vars[*]}")

# Non-tau CR groups, first block:
#
# vars=(fwd0eta)
# var_tag=$(IFS=-; echo "${vars[*]}")
# category_sets=(
#   "2l_CR 2l_CRflip 3l_CR"
# )
#
# for year in "${years[@]}"; do
#   for category_set in "${category_sets[@]}"; do
#     read -r -a cats <<< "${category_set}"
#     run_cr_block "${year}" "${cats[@]}"
#   done
# done

# Non-tau CR groups, second block:
#
# vars=(fwd0eta)
# var_tag=$(IFS=-; echo "${vars[*]}")
# category_sets=(
#   "2los_CRZ 2los_CRtt"
# )
#
# for year in "${years[@]}"; do
#   for category_set in "${category_sets[@]}"; do
#     read -r -a cats <<< "${category_set}"
#     run_cr_block "${year}" "${cats[@]}"
#   done
# done

###############################################################################
# Parking area for future SR runs
###############################################################################

# run_sr_block() {
#   local year_expr="$1"
#   shift
#
#   local cats=("$@")
#   local cat_tag
#   local pkl_tag
#
#   cat_tag=$(join_by - "${cats[@]}")
#   pkl_tag="SR_fwdfix_pt70ext_${cat_tag}"
#
#   echo "----------------------------------------"
#   echo "Years: ${year_expr}"
#   echo "SR categories: ${cats[*]}"
#   echo "Output tag: ${pkl_tag}"
#   echo "Output dir: ${output_dir}"
#   echo "----------------------------------------"
#
#   clean_env_cache
#
#   local cmd=(
#     ./fullR3_run.sh
#     -y "${year_expr}"
#     -t "${pkl_tag}"
#     -s "${chunk_size}"
#     --sr
#     --do-systs
#     --do-np
#     -p "${output_dir}"
#     --category-groups "${cats[@]}"
#     --suppress-forward-eta-stochastic-jer
#     --all-analysis
#   )
#
#   echo "Executing:"
#   printf ' %q' "${cmd[@]}"
#   echo
#
#   "${cmd[@]}"
#
#   echo "Done"
#   echo "----------------------------------------"
#   echo
# }
#
# run_sr_block "2022 2022EE 2023 2023BPix" 2l
# run_sr_block "2022 2022EE 2023 2023BPix" 2lss_1tau
# run_sr_block "2022 2022EE 2023 2023BPix" 2los_1tau
# run_sr_block "2022 2022EE 2023 2023BPix" 3l_m_offZ
# run_sr_block "2022 2022EE 2023 2023BPix" 3l_p_offZ
# run_sr_block "2022 2022EE 2023 2023BPix" 3l_onZ_tau
# run_sr_block "2022 2022EE 2023 2023BPix" 3l_fwd
# run_sr_block "2022 2022EE 2023 2023BPix" 4l