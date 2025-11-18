#!/usr/bin/env bash
# Wrapper for launching run_analysis.py with TaskVine in the current branch.
#
# Expectations before running:
#   1. Activate the shared Conda environment shipped with this repository
#      (name: coffea20250703) so the topeft and topcoffea editable installs are
#      on PYTHONPATH.  The helper below attempts to activate it when possible.
#   2. Stage the packaged TaskVine environment tarball by running
#      `python -m topcoffea.modules.remote_environment` after activation.  The
#      script reuses the returned path as the --environment-file argument.
#   3. Launch a pool of TaskVine workers that point at the manager name used in
#      this script (defaults to "${USER}-taskvine-coffea") via vine_submit_workers
#      or vine_worker.  Workers should run in the same environment tarball
#      reported by the remote_environment helper.
#
# The run_analysis workflow emits histogram pickles keyed by
# (variable, channel, application, sample, systematic) 5-tuples; downstream
# tools assume this schema when combining outputs.

set -euo pipefail

PrintUsage() {
  cat <<'USAGE'
Usage: full_run.sh [-y YEAR [YEAR ...]] [-t TAG] [--cr | --sr] \
                   [--outdir PATH] [--manager NAME] [extra run_analysis args]

Examples:
  full_run.sh --cr -y run3 -t dev_validation
  full_run.sh --sr -y 2022 2022EE --outdir histos/run3_taskvine \
      --chunksize 80000 --prefix root://xrootd.site/

Notes:
  * YEARS accept explicit values (2022, 2022EE, UL17, etc.) or bundles
    (run2 -> UL16 UL16APV UL17 UL18, run3 -> 2022 2022EE).  Unknown entries
    are rejected so mis-typed eras fail fast.
  * Required input cfg/json files live under input_samples/cfgs/ and are
    selected automatically per year.  Add extra CLI arguments (for example
    --options configs/fullR2_run.yml:sr) after the recognized flags to pass
    through additional run_analysis toggles.
  * The default output name is <YEARS>_(CRs|SRs)_<TAG>, saved to the specified
    output directory with the 5-tuple histogram schema used throughout this
    branch.
USAGE
}

activate_env() {
  local target_env="coffea20250703"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$target_env" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$target_env" || true
  else
    echo "Warning: conda not available; ensure $target_env is already active." >&2
  fi
}

stage_environment() {
  python -m topcoffea.modules.remote_environment
}

main() {
  if [[ $# -eq 0 ]]; then
    PrintUsage
    return 0
  fi

  local default_year="2022"
  local default_tag
  default_tag=$(git rev-parse --short HEAD 2>/dev/null || date +%y%m%d)

  local flag_cr=false
  local flag_sr=false
  local -a extra_args=()
  local -a years=()
  local -a expanded_years=()
  local -a resolved_years=()
  local user_chunk_override=false
  local user_env_override=false
  local user_executor_override=false
  local outdir="histos"
  local manager_name="${USER:-coffea}-taskvine-coffea"
  local tag=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--year)
        shift
        if [[ $# -eq 0 || "$1" == -* ]]; then
          echo "Error: -y|--year requires at least one argument" >&2
          return 1
        fi
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -*)
              break
              ;;
            *)
              years+=("$1")
              shift
              ;;
          esac
        done
        ;;
      -t|--tag)
        tag="$2"
        shift 2
        ;;
      --outdir)
        outdir="$2"
        shift 2
        ;;
      --manager)
        manager_name="$2"
        shift 2
        ;;
      --cr)
        flag_cr=true
        shift
        ;;
      --sr)
        flag_sr=true
        shift
        ;;
      -s|--chunksize|--chunksize=*)
        user_chunk_override=true
        extra_args+=("$1")
        shift
        ;;
      -x|--executor|--executor=*)
        user_executor_override=true
        extra_args+=("$1")
        shift
        ;;
      --environment-file|--environment-file=*)
        user_env_override=true
        extra_args+=("$1")
        shift
        ;;
      -h|--help)
        PrintUsage
        return 0
        ;;
      --)
        shift
        extra_args+=("$@")
        break
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ "$flag_cr" == false && "$flag_sr" == false ]] || [[ "$flag_cr" == true && "$flag_sr" == true ]]; then
    echo "Error: specify exactly one of --cr or --sr" >&2
    echo
    PrintUsage
    return 1
  fi

  if [[ ${#years[@]} -eq 0 ]]; then
    echo "Warning: YEAR not provided, using default YEAR=$default_year"
    years=("$default_year")
  fi

  local year
  for year in "${years[@]}"; do
    case "${year,,}" in
      run2)
        expanded_years+=(UL16 UL16APV UL17 UL18)
        ;;
      run3)
        expanded_years+=(2022 2022EE)
        ;;
      *)
        expanded_years+=("$year")
        ;;
    esac
  done

  declare -A seen_year=()
  for year in "${expanded_years[@]}"; do
    if [[ -z "${seen_year[$year]:-}" ]]; then
      resolved_years+=("$year")
      seen_year[$year]=1
    fi
  done

  if [[ ${#resolved_years[@]} -eq 0 ]]; then
    echo "Error: no valid years resolved from the provided arguments." >&2
    return 1
  fi

  if [[ -z "$tag" ]]; then
    echo "Warning: TAG not provided, using default TAG=$default_tag"
    tag="$default_tag"
  fi

  local year_label
  year_label=$(IFS=-; echo "${resolved_years[*]}")

  local out_name
  if [[ "$flag_cr" == true ]]; then
    out_name="${year_label}_CRs_${tag}"
  else
    out_name="${year_label}_SRs_${tag}"
  fi

  local script_dir
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local repo_root
  repo_root=$(cd "$script_dir/../.." && pwd)
  cd "$script_dir"

  activate_env
  local env_tarball=""
  if [[ "$user_env_override" == false ]]; then
    env_tarball=$(stage_environment)
  fi

  local cfgs_path="$repo_root/input_samples/cfgs"
  local -a cfgs_list=()
  local -A run2_year_map=(
    [2016]=2016
    [2016apv]=2016APV
    [2017]=2017
    [2018]=2018
    [ul16]=UL16
    [ul16apv]=UL16APV
    [ul17]=UL17
    [ul18]=UL18
  )

  local -a run2_cfgs_sr=(
    "$cfgs_path/mc_signal_samples_NDSkim.cfg"
    "$cfgs_path/mc_background_samples_NDSkim.cfg"
    "$cfgs_path/data_samples_NDSkim.cfg"
  )
  local -a run2_cfgs_cr=(
    "$cfgs_path/mc_signal_samples_NDSkim.cfg"
    "$cfgs_path/mc_background_samples_NDSkim.cfg"
    "$cfgs_path/mc_background_samples_cr_NDSkim.cfg"
    "$cfgs_path/data_samples_NDSkim.cfg"
  )
  local -a run3_cfgs_2022_sr=(
    "$cfgs_path/2022_mc_signal_samples.cfg"
    "$cfgs_path/2022_mc_background_samples.cfg"
    "$cfgs_path/2022_data_samples.cfg"
  )
  local -a run3_cfgs_2022_cr=(
    "$cfgs_path/2022_mc_background_samples.cfg"
    "$cfgs_path/2022_data_samples.cfg"
  )
  local -a run3_cfgs_2022ee_sr=(
    "$cfgs_path/2022_mc_signal_samples.cfg"
    "$cfgs_path/2022_mc_background_samples.cfg"
    "$cfgs_path/2022EE_data_samples.cfg"
  )
  local -a run3_cfgs_2022ee_cr=(
    "$cfgs_path/2022_mc_background_samples.cfg"
    "$cfgs_path/2022EE_data_samples.cfg"
  )

  declare -A seen_cfgs=()
  add_cfg() {
    local cfg_file="$1"
    if [[ ! -f "$cfg_file" ]]; then
      echo "Error: required cfg/json file not found: $cfg_file" >&2
      return 1
    fi
    if [[ -n "${seen_cfgs[$cfg_file]:-}" ]]; then
      return 0
    fi
    cfgs_list+=("$cfg_file")
    seen_cfgs[$cfg_file]=1
    return 0
  }

  local year_key
  local run2_bundle_added=false
  for year in "${resolved_years[@]}"; do
    year_key=${year,,}
    if [[ -n "${run2_year_map[$year_key]:-}" ]]; then
      if [[ "$run2_bundle_added" == false ]]; then
        if [[ "$flag_cr" == true ]]; then
          for cfg in "${run2_cfgs_cr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        else
          for cfg in "${run2_cfgs_sr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        fi
        run2_bundle_added=true
      fi
      continue
    fi

    case "$year" in
      2022)
        if [[ "$flag_cr" == true ]]; then
          for cfg in "${run3_cfgs_2022_cr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        else
          for cfg in "${run3_cfgs_2022_sr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        fi
        ;;
      2022EE)
        if [[ "$flag_cr" == true ]]; then
          for cfg in "${run3_cfgs_2022ee_cr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        else
          for cfg in "${run3_cfgs_2022ee_sr[@]}"; do
            add_cfg "$cfg" || return 1
          done
        fi
        ;;
      *)
        echo "Error: unrecognized YEAR '$year'" >&2
        return 1
        ;;
    esac
  done

  local cfgs
  cfgs=$(IFS=,; echo "${cfgs_list[*]}")

  echo "Resolved years: ${resolved_years[*]}"
  echo "Resolved cfg inputs: $cfgs"
  echo "TaskVine manager: $manager_name"
  if [[ -n "$env_tarball" ]]; then
    echo "Environment archive: $env_tarball"
  fi

  local -a options=(
    --outname "$out_name"
    --outpath "$outdir"
    --nworkers 8
    --summary-verbosity brief
  )
  if [[ "$user_chunk_override" == false ]]; then
    options+=(--chunksize 50000)
  fi
  if [[ "$user_executor_override" == false ]]; then
    options+=(--executor taskvine)
  fi
  if [[ -n "$manager_name" ]]; then
    options+=(--manager-name "$manager_name")
  fi
  if [[ "$user_env_override" == false && -n "$env_tarball" ]]; then
    options+=(--environment-file "$env_tarball")
  fi
  if [[ "$flag_cr" == true ]]; then
    options+=(--skip-sr)
  else
    options+=(--skip-cr --do-systs)
  fi

  local -a run_cmd=(python run_analysis.py "$cfgs")
  run_cmd+=("${options[@]}")
  run_cmd+=("${extra_args[@]}")

  printf "\nRunning the following command:\n%s\n\n" "${run_cmd[*]}"
  time "${run_cmd[@]}"
}

main "$@"
exit_code=$?
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exit "$exit_code"
else
  return "$exit_code"
fi
