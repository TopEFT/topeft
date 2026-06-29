#!/usr/bin/env bash
set -euo pipefail

cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

###############################################################################
# Global configuration
###############################################################################

output_dir="/groups/klannon/apiccine/preappr_1l1tau_260613"
chunk_size="50000"

# Nominal TOP-23-002-like ttgamma sample-role strategy.
#
# Run 2:
#   TTGJets NLO                         -> production-like ttgamma
#   TTGamma_Dilept / TTGamma_SingleLept -> decay-like ttgamma
#   inclusive ttbar                     -> veto selected external-conversion-like leptons
#
# Run 3:
#   TTG-1Jets_PTG-*                     -> inclusive ttgamma treatment
#   inclusive ttbar                     -> veto selected external-conversion-like leptons
#
# The diagnostic Run 2 NLO-only policy is intentionally not used here.
ttgamma_sample_role_policy="split"

# Use a strategy-specific tag to avoid mixing baseline/feature/diagnostic outputs.
campaign_tag="preappr_sr"

cr_pkl_base_tag="${campaign_tag}"
sr_pkl_base_tag="${campaign_tag}"

# Execution switches.
#
# This script is currently configured for Yuyi's CR distribution request.
run_cr=false
run_sr=true

# Useful while checking resolved years/categories/histograms without launching production.
dry_run=false

# Shared CR/SR production switches.
#
# Yuyi's request is for distributions, and the previous colleague-facing setup used
# systematic variations for CR plotting. Keep nonprompt disabled unless explicitly needed.
do_systs=true
do_np=true

# Enable only if the colleague explicitly needs lepton-flavour split outputs.
split_lep_flavor=false

###############################################################################
# Histogram variable chunks
###############################################################################

# Each entry is one independent histogram chunk.
#
# Important:
#   These are intentionally strings. Each string is split into a real bash array
#   inside run_cr_block/run_sr_block before passing --hist-vars to fullR3_run.sh.
#
# Requested coverage:
#   - fwd0pt for all periods/regions;
#   - fwd0eta, lj0pt, lt, met, ptz for the relevant non-tau CRs;
#   - ptz_wtau and tau variables for tau CR coverage;
#   - invmass, j0eta, j0pt, l0/l1 variables, ljptsum, nbtagsl, njets for tau CRs.
cr_var_sets=(
  # "fwd0pt fwd0eta lj0pt lt met ptz nbtagsl l0conept l0eta"
  # "njets l1conept l1eta j0pt j0eta invmass ljptsum nbtagsm npvsGood"
  # "l0eta l0conept met lt njets ptz_wtau tau0Fpt tau0Tpt"
  # "lt"
  "lj0pt nbtagsl nbtagsm fwd0pt fwd0eta lt"
)

# SR variable chunks are configured separately from CR so SR campaigns can use
# dedicated histogram groups without changing the CR request.
sr_var_sets=(
  "njets lj0pt"
  "ptz ptz_wtau lt"
)

###############################################################################
# CR configuration
###############################################################################

# Keep year periods separate so the output pkls are period-specific.
#
# Yuyi requested Run 2 period-specific coverage and Run 3 tau-region coverage.
cr_year_sets=(
  # 2016APV
  # 2016
  # 2017
  # 2018
  # 2022
  # 2022EE
  2023
  2023BPix
)

# Current category names used by the analysis helpers.
#
# Mapping to Yuyi labels:
#   2los_Z      -> 2los_CRZ
#   2lss_flip   -> 2l_CRflip
#   2los_tt     -> 2los_CRtt
#   3l          -> 3l_CR
#   dy_tautau   -> 1l_1tau_CRDY
#   1l_1tau_tt  -> 1l_1tau_CRtt
#
# The aggregate 2los_1tau group is included for the 2los tau request.
# If the branch has explicit 2los_1tau_Ftau / 2los_1tau_Ttau category groups,
# they can be added as separate entries after confirming the exact names.
cr_category_sets=(
  # "2los_CRZ 2l_CR 2los_CRtt 2l_CRflip 3l_CR"
  "1l_1tau_CRtt 1l_1tau_CRDY 2los_1tau"
)

###############################################################################
# SR configuration
###############################################################################

# Kept available, but disabled by default for this request.
sr_year_sets=(
  # 2022
  # 2022EE
  # 2023
  # 2023BPix
  "2022 2022EE 2023 2023BPix"
  # 2016APV
  # 2016
  # 2017
  # 2018
)

sr_category_sets=(
  "2l"
  "2lss_1tau 2los_1tau"
  "3l_m_offZ"
  "3l_p_offZ"
  "3l_onZ_tau 4l"
  "3l_fwd"
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

bool_to_option_enabled() {
  case "$1" in
    true|false) ;;
    *)
      echo "ERROR: expected boolean true/false, got '$1'" >&2
      exit 1
      ;;
  esac
}

clean_env_cache() {
  if [[ -d topeft-envs ]]; then
    find topeft-envs -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -delete
  fi
}

assert_supported_year_expr() {
  local year_expr="$1"
  local year
  local years_in_expr=()

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

build_common_command_options() {
  local -n cmd_ref="$1"

  cmd_ref+=(--ttgamma-sample-role-policy "${ttgamma_sample_role_policy}")

  if [[ "${do_systs}" == "true" ]]; then
    cmd_ref+=(--do-systs)
  fi

  if [[ "${do_np}" == "true" ]]; then
    cmd_ref+=(--do-np)
  fi

  cmd_ref+=(
    -p "${output_dir}"
    --all-analysis
  )

  if [[ "${split_lep_flavor}" == "true" ]]; then
    cmd_ref+=(--split-lep-flavor)
  fi

  if [[ "${dry_run}" == "true" ]]; then
    cmd_ref+=(--dry-run)
  fi
}

run_cr_block() {
  local year_expr="$1"
  local var_set="$2"
  shift 2

  assert_supported_year_expr "${year_expr}"

  local years=()
  read -r -a years <<< "${year_expr}"

  local vars=()
  read -r -a vars <<< "${var_set}"

  local cats=("$@")
  local cat_tag
  local var_tag
  local pkl_tag

  cat_tag=$(join_by - "${cats[@]}")
  var_tag=$(join_by - "${vars[@]}")
  pkl_tag="${cr_pkl_base_tag}_${cat_tag}_${var_tag}"

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

  clean_env_cache

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
  "${cmd[@]}"

  echo "${year_expr} CR done for ${cat_tag} / ${var_tag}"
  echo "----------------------------------------"
  echo
}

run_sr_block() {
  local year_expr="$1"
  local var_set="$2"
  shift 2

  assert_supported_year_expr "${year_expr}"

  local years=()
  read -r -a years <<< "${year_expr}"

  local vars=()
  read -r -a vars <<< "${var_set}"

  local cats=("$@")
  local cat_tag
  local var_tag
  local pkl_tag

  cat_tag=$(join_by - "${cats[@]}")
  var_tag=$(join_by - "${vars[@]}")
  pkl_tag="${sr_pkl_base_tag}_${cat_tag}_${var_tag}"

  echo "----------------------------------------"
  echo "Mode: SR"
  echo "Years: ${year_expr}"
  echo "Categories: ${cats[*]}"
  echo "Variables: ${vars[*]}"
  echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
  echo "Campaign tag: ${campaign_tag}"
  echo "Output tag: ${pkl_tag}"
  echo "Output dir: ${output_dir}"
  echo "Dry run: ${dry_run}"
  echo "----------------------------------------"

  clean_env_cache

  local cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pkl_tag}"
    -s "${chunk_size}"
    --sr
    --hist-vars "${vars[@]}"
    --category-groups "${cats[@]}"
  )

  build_common_command_options cmd

  print_command "${cmd[@]}"
  "${cmd[@]}"

  echo "${year_expr} SR done for ${cat_tag} / ${var_tag}"
  echo "----------------------------------------"
  echo
}

###############################################################################
# Preflight summary
###############################################################################

bool_to_option_enabled "${run_cr}"
bool_to_option_enabled "${run_sr}"
bool_to_option_enabled "${dry_run}"
bool_to_option_enabled "${do_systs}"
bool_to_option_enabled "${do_np}"
bool_to_option_enabled "${split_lep_flavor}"

echo "========================================"
echo "run_cr.sh configuration"
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
print_var_sets "CR" "${cr_var_sets[@]}"
print_var_sets "SR" "${sr_var_sets[@]}"
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

###############################################################################
# Main CR production
###############################################################################

if [[ "${run_cr}" == "true" ]]; then
  for year_expr in "${cr_year_sets[@]}"; do
    for category_set in "${cr_category_sets[@]}"; do
      read -r -a cats <<< "${category_set}"

      for var_set in "${cr_var_sets[@]}"; do
        run_cr_block "${year_expr}" "${var_set}" "${cats[@]}"
      done
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
  for year_expr in "${sr_year_sets[@]}"; do
    for category_set in "${sr_category_sets[@]}"; do
      read -r -a cats <<< "${category_set}"

      for var_set in "${sr_var_sets[@]}"; do
        run_sr_block "${year_expr}" "${var_set}" "${cats[@]}"
      done
    done
  done
else
  echo "Skipping SR production because run_sr=${run_sr}"
  echo
fi

echo "run_cr.sh completed"
echo "campaign_tag: ${campaign_tag}"
echo "ttgamma sample-role policy: ${ttgamma_sample_role_policy}"
echo "output_dir: ${output_dir}"
