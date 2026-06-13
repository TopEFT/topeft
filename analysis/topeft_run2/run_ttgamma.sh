#!/usr/bin/env bash
set -euo pipefail

cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

###############################################################################
# CL007AT reduced-sample role-policy-v2 production
###############################################################################

dryRun=false
policyMode="split"

PrintUsage() {
  cat <<EOF
Usage: $0 [--dry-run] [--ttgamma-sample-role-policy POLICY]

Options:
  --dry-run
      Print resolved reduced-cfg contents and fullR3_run.sh commands only.
  --ttgamma-sample-role-policy POLICY
      Supported values: split, run2_nlo_inclusive.
      Default: split.

The run2_nlo_inclusive policy is diagnostic-only and must be requested
explicitly.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dryRun=true
      shift
      ;;
    --ttgamma-sample-role-policy)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "ERROR: --ttgamma-sample-role-policy requires a policy value" >&2
        exit 2
      fi
      policyMode="$2"
      shift 2
      ;;
    --ttgamma-sample-role-policy=*)
      policyMode="${1#--ttgamma-sample-role-policy=}"
      shift
      ;;
    -h|--help)
      PrintUsage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported argument: $1" >&2
      PrintUsage >&2
      exit 2
      ;;
  esac
done

case "${policyMode}" in
  split|run2_nlo_inclusive) ;;
  *)
    cat >&2 <<EOF
ERROR: policyMode must be one of:
  split
  run2_nlo_inclusive
EOF
    exit 2
    ;;
esac

###############################################################################
# Global configuration
###############################################################################

outputDir="/groups/klannon/apiccine/photons"
chunkSize="100000"

case "${policyMode}" in
  split)
    campaignTag="rolepolicy_v2"
    ttgammaSampleRolePolicy="split"
    ;;
  run2_nlo_inclusive)
    campaignTag="rolepolicy_v2_run2NloInclusive"
    ttgammaSampleRolePolicy="run2_nlo_inclusive"
    ;;
esac
srPklBaseTag="SR_${campaignTag}"

# Variables to be produced in the SR pkls.
histVars=(lj0pt ptz)
varTag=$(IFS=-; echo "${histVars[*]}")

# Run 3 comparison year. Use 2022EE to match the previous validation.
# Change to 2022 only if you intentionally want the pre-EE era.
run3Year="${CL007AT_RUN3_YEAR:-2022EE}"
if [[ "${policyMode}" == "split" ]]; then
  case "${run3Year}" in
    2022|2022EE) ;;
    *)
      echo "ERROR: CL007AT_RUN3_YEAR must be 2022 or 2022EE." >&2
      exit 2
      ;;
  esac
fi

###############################################################################
# SR configuration
###############################################################################

if [[ "${policyMode}" == "split" ]]; then
  # Each entry is one independent year expression passed to fullR3_run.sh.
  # Keep these separate because each year uses a different reduced cfg override.
  srYearSets=(
    "${run3Year}"
    2018
  )
else
  srYearSets=(
    2018
  )
fi

# Each entry is one independent subset of categories.
# Grouped entries produce one pkl per grouped set.
srCategorySets=(
  "2l 2lss_1tau 2los_1tau 3l_m_offZ"
  "3l_p_offZ 3l_onZ_tau 3l_fwd 4l"
)

###############################################################################
# Reduced cfg configuration
###############################################################################

repoRoot="/users/apiccine/work/correction-lib/topeft"
run2JsonRoot="${repoRoot}/input_samples/sample_jsons/background_samples/central_UL"
run3JsonRoot="${repoRoot}/input_samples/sample_jsons/background_samples/ND_SRskim${run3Year}"

cfgDir="/tmp/cl007at_${campaignTag}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${cfgDir}"

###############################################################################
# Helpers
###############################################################################

JoinBy() {
  local delimiter="$1"
  shift
  local IFS="${delimiter}"
  echo "$*"
}

CleanEnvCache() {
  if [[ -d topeft-envs ]]; then
    find topeft-envs -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -delete
  fi
}

AssertSupportedYearExpr() {
  local yearExpr="$1"
  local year

  read -r -a yearsInExpr <<< "${yearExpr}"

  for year in "${yearsInExpr[@]}"; do
    case "${year}" in
      2018|2022|2022EE) ;;
      *)
        cat >&2 <<EOF
ERROR: unsupported year token '${year}' in year expression '${yearExpr}'.

Allowed year tokens for this reduced ttgamma helper:
  2018 2022 2022EE
EOF
        exit 1
        ;;
    esac
  done
}

AssertJsonsExist() {
  local jsonPath
  for jsonPath in "$@"; do
    if [[ ! -f "${jsonPath}" ]]; then
      echo "ERROR: audited sample JSON is missing: ${jsonPath}" >&2
      exit 1
    fi
  done
}

WriteReducedCfg() {
  local year="$1"
  local cfgPath="$2"
  local -a jsons=()

  case "${year}" in
    2018)
      if [[ "${policyMode}" == "run2_nlo_inclusive" ]]; then
        jsons=(
          "${run2JsonRoot}/UL18_TTGJets_NDSkim.json"
          "${run2JsonRoot}/UL18_TTTo2L2Nu_NDSkim.json"
          "${run2JsonRoot}/UL18_TTToSemiLeptonic_NDSkim.json"
        )
      else
        jsons=(
          "${run2JsonRoot}/UL18_TTGJets_NDSkim.json"
          "${run2JsonRoot}/UL18_TTGamma_Dilept_NDSkim.json"
          "${run2JsonRoot}/UL18_TTGamma_SingleLept_NDSkim.json"
          "${run2JsonRoot}/UL18_TTTo2L2Nu_NDSkim.json"
          "${run2JsonRoot}/UL18_TTToSemiLeptonic_NDSkim.json"
        )
      fi
      ;;
    2022|2022EE)
      jsons=(
        "${run3JsonRoot}/TTG-1Jets_PTG-10to100_NDSkim_${year}.json"
        "${run3JsonRoot}/TTG-1Jets_PTG-100to200_NDSkim_${year}.json"
        "${run3JsonRoot}/TTG-1Jets_PTG-200_NDSkim_${year}.json"
        "${run3JsonRoot}/TTto2L2Nu_NDSkim_${year}.json"
        "${run3JsonRoot}/TTtoLNu2Q_NDSkim_${year}.json"
      )
      ;;
    *)
      echo "ERROR: unsupported year for reduced cfg: ${year}" >&2
      exit 2
      ;;
  esac

  AssertJsonsExist "${jsons[@]}"

  {
    echo "root://cmsxrootd.crc.nd.edu/"
    printf '%s\n' "${jsons[@]}"
  } > "${cfgPath}"
}

RunSrBlock() {
  local yearExpr="$1"
  local cfgPath="$2"
  shift 2

  AssertSupportedYearExpr "${yearExpr}"

  local years=()
  read -r -a years <<< "${yearExpr}"

  local cats=("$@")
  local catTag
  local pklTag
  local expectedPkl
  local expectedNpPkl
  local cmd=()

  catTag=$(JoinBy - "${cats[@]}")
  pklTag="${srPklBaseTag}_${catTag}_${varTag}"

  expectedPkl="${outputDir}/${yearExpr}SRs_${pklTag}.pkl.gz"
  expectedNpPkl="${outputDir}/${yearExpr}SRs_${pklTag}_np.pkl.gz"

  if [[ -e "${expectedPkl}" || -e "${expectedNpPkl}" ]]; then
    if [[ "${dryRun}" == "true" ]]; then
      echo "WARNING: existing pkl path would block a real run for tag ${pklTag}" >&2
      echo "Expected base pkl: ${expectedPkl}" >&2
      echo "Expected NP pkl:   ${expectedNpPkl}" >&2
    else
      echo "ERROR: refusing to overwrite an existing pkl for tag ${pklTag}" >&2
      echo "Expected base pkl: ${expectedPkl}" >&2
      echo "Expected NP pkl:   ${expectedNpPkl}" >&2
      exit 1
    fi
  fi

  echo "----------------------------------------"
  echo "Policy mode: ${policyMode}"
  echo "Campaign/tag: ${campaignTag}"
  echo "ttgamma sample-role policy: ${ttgammaSampleRolePolicy}"
  echo "Years: ${yearExpr}"
  echo "Categories: ${cats[*]}"
  echo "Cfg override: ${cfgPath}"
  if [[ "${dryRun}" == "true" ]]; then
    echo "Reduced cfg contents:"
    sed 's/^/  /' "${cfgPath}"
  fi
  echo "Output tag: ${pklTag}"
  echo "Output dir: ${outputDir}"
  echo "Expected base pkl: ${expectedPkl}"
  echo "Expected NP pkl:   ${expectedNpPkl}"
  echo "Dry run: ${dryRun}"
  echo "----------------------------------------"

  CleanEnvCache

  cmd=(
    ./fullR3_run.sh
    -y "${years[@]}"
    -t "${pklTag}"
    -s "${chunkSize}"
    --sr
    --cfg-override "${cfgPath}"
    --ttgamma-sample-role-policy "${ttgammaSampleRolePolicy}"
    --hist-vars "${histVars[@]}"
    --do-np
    -p "${outputDir}"
    --category-groups "${cats[@]}"
    --suppress-forward-eta-stochastic-jer
    --all-analysis
  )

  if [[ "${dryRun}" == "true" ]]; then
    cmd+=(--dry-run --pretend --nchunks 1)
  fi

  echo "Executing:"
  printf ' %q' "${cmd[@]}"
  echo

  "${cmd[@]}"

  echo "${yearExpr} done"
  echo "----------------------------------------"
  echo
}

###############################################################################
# Main SR production
###############################################################################

for yearExpr in "${srYearSets[@]}"; do
  cfgPath="${cfgDir}/ttgamma_ttbar_${yearExpr}.cfg"
  WriteReducedCfg "${yearExpr}" "${cfgPath}"

  for categorySet in "${srCategorySets[@]}"; do
    read -r -a cats <<< "${categorySet}"
    RunSrBlock "${yearExpr}" "${cfgPath}" "${cats[@]}"
  done
done

echo "run_ttgamma.sh completed"
echo "Policy mode: ${policyMode}"
echo "Campaign/tag: ${campaignTag}"
echo "ttgamma sample-role policy: ${ttgammaSampleRolePolicy}"
echo "Cfg directory: ${cfgDir}"
echo "Output directory: ${outputDir}"

if [[ "${dryRun}" == "true" ]]; then
  echo "No pkls were produced because --dry-run was requested."
fi
