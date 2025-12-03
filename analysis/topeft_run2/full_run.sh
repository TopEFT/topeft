#!/usr/bin/env bash
# Unified wrapper for launching run_analysis.py with TaskVine, Coffea futures,
# or the local iterative executor now that Run-2 scenarios are driven by
# run2_scenarios.yaml.
#
# Expectations before running (same as before):
#   1. Activate the shared Conda environment shipped with this repository
#      (name: coffea2025) or another compatible setup so the topeft and
#      topcoffea editable installs are available. This script assumes the
#      environment is already active and does not attempt to activate it.
#   2. For TaskVine runs, stage the packaged environment tarball by running
#      `python -m topcoffea.modules.remote_environment` after activation.  The
#      script reuses the returned path as the --environment-file argument.
#   3. When using TaskVine, launch a pool of workers that point at the manager
#      name used in this script (defaults to "${USER}-taskvine-coffea") via
#      vine_submit_workers or vine_worker.  Workers should run in the same
#      environment tarball reported by the remote_environment helper.
#
# The run_analysis workflow emits histogram pickles keyed by
# (variable, channel, application, sample, systematic) 5-tuples; downstream
# tools assume this schema when combining outputs.

set -euo pipefail

PrintUsage() {
  cat <<'USAGE'
Usage: full_run.sh [-y YEAR [YEAR ...]] [-t TAG] [--cr | --sr] \
                   [--executor {taskvine,futures,iterative}] [--outdir PATH] [--manager NAME] \
                   [--samples PATH [PATH ...]] [--scenario NAME] [--log-level LEVEL] \
                   [--debug-logging] [--dry-run] [extra run_analysis args]

Examples:
  # Control-region TaskVine run over Run-3 bundles using defaults defined in the
  # Run-3 configs (user-supplied --options forwarded via extra args).
  full_run.sh --cr -y run3 -t dev_validation --executor taskvine \
      --outdir histos/run3_validation --dry-run

  # Signal-region futures launch for UL17 using explicit sample JSONs.
  full_run.sh --sr -y UL17 --executor futures --outdir histos/local_debug \
      --samples ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
      --scenario TOP_22_006 --chunksize 4000 --dry-run

  # Run-2 superset (all_analysis) using futures and the default Run-2 profile.
  full_run.sh --sr -y run2 --executor futures --outdir histos/run2_all \
      --scenario all_analysis --dry-run

Notes:
  * YEARS accept explicit values (2022, 2022EE, UL17, etc.) or bundles
    (run2 -> UL16 UL16APV UL17 UL18, run3 -> 2022 2022EE).  Unknown entries
    are rejected so mis-typed eras fail fast.
  * Run-2 invocations without an explicit --scenario will prefer the default
    run_analysis options file (analysis/topeft_run2/configs/fullR2_run.yml) and
    select the SR/CR profile automatically.  Provide --scenario or --options
    explicitly to override this behaviour.
  * Required input cfg/json files live under input_samples/cfgs/ and are
    selected automatically per year when --samples is not specified.  Supply
    --samples to point at bespoke JSONs.  Append additional run_analysis flags
    (for example --options configs/fullR2_run.yml:sr or --split-lep-flavor)
    after the recognized wrapper arguments.
  * The default output name is <YEARS>_(CRs|SRs)_<TAG>, saved to the specified
    output directory with the 5-tuple histogram schema used throughout this
    branch.  Use --dry-run to print the resolved command without launching
    Python.  Pass --debug-logging to forward the instrumentation flag (always
    DEBUG) or --log-level LEVEL to tweak the Python logging verbosity seen in
    run_analysis.py.
USAGE
}

check_active_env() {
  local target_env="coffea2025"
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "$target_env" ]]; then
    echo "Warning: CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV}' (expected '${target_env}' or a compatible environment)." >&2
  elif [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Note: no active conda environment detected; ensure '${target_env}' or another compatible environment is already activated." >&2
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
  local -a user_samples=()
  local -a scenario_args=()
  local scenario_specified=false
  local user_chunk_override=false
  local user_env_override=false
  local user_options_override=false
  local outdir="histos"
  local manager_name="${USER:-coffea}-taskvine-coffea"
  local tag=""
  local executor_choice=""
  local workers=""
  local futures_prefetch=1
  local futures_retries=0
  local futures_retry_wait=5.0
  local dry_run=false
  local debug_logging=false
  local log_level=""
  local auto_options_spec=""

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
      --workers)
        workers="$2"
        shift 2
        ;;
      --samples)
        shift
        if [[ $# -eq 0 || "$1" == -* ]]; then
          echo "Error: --samples requires at least one argument" >&2
          return 1
        fi
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -*)
              break
              ;;
            *)
              user_samples+=("$1")
              shift
              ;;
          esac
        done
        ;;
      --futures-prefetch)
        futures_prefetch="$2"
        shift 2
        ;;
      --futures-retries)
        futures_retries="$2"
        shift 2
        ;;
      --futures-retry-wait)
        futures_retry_wait="$2"
        shift 2
        ;;
      --executor)
        executor_choice="$2"
        shift 2
        ;;
      --executor=*)
        executor_choice="${1#*=}"
        shift
        ;;
      --taskvine)
        executor_choice="taskvine"
        shift
        ;;
      --futures)
        executor_choice="futures"
        shift
        ;;
      --iterative)
        executor_choice="iterative"
        shift
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
      -x)
        executor_choice="$2"
        shift 2
        ;;
      -h|--help)
        PrintUsage
        return 0
        ;;
      --dry-run)
        dry_run=true
        shift
        ;;
      --debug-logging)
        debug_logging=true
        shift
        ;;
      --log-level)
        log_level="$2"
        shift 2
        ;;
      --log-level=*)
        log_level="${1#*=}"
        shift
        ;;
      --debug-logging=*)
        local value="${1#*=}"
        if [[ "${value,,}" != "0" && "${value,,}" != "false" && -n "$value" ]]; then
          debug_logging=true
        fi
        shift
        ;;
      --options)
        user_options_override=true
        if [[ $# -lt 2 ]]; then
          echo "Error: --options expects a value" >&2
          return 1
        fi
        extra_args+=("$1" "$2")
        shift 2
        ;;
      --options=*)
        user_options_override=true
        extra_args+=("$1")
        shift
        ;;
      --scenario)
        scenario_specified=true
        shift
        if [[ $# -eq 0 || "$1" == -* ]]; then
          echo "Error: --scenario expects a name" >&2
          return 1
        fi
        scenario_args+=("$1")
        shift
        ;;
      --scenario=*)
        scenario_specified=true
        local scenario_value="${1#*=}"
        if [[ -z "$scenario_value" ]]; then
          echo "Error: --scenario expects a name" >&2
          return 1
        fi
        scenario_args+=("$scenario_value")
        shift
        ;;
      --environment-file|--environment-file=*)
        user_env_override=true
        extra_args+=("$1")
        shift
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

  if [[ "$user_options_override" == true && "$scenario_specified" == true ]]; then
    echo "Error: --scenario cannot be combined with --options; encode the scenario inside the options profile instead." >&2
    return 1
  fi

  if [[ "$user_options_override" == false && ${#extra_args[@]} -gt 0 ]]; then
    for passthrough_arg in "${extra_args[@]}"; do
      case "$passthrough_arg" in
        --options|--options=*)
          echo "Error: '--options' supplied after '--'. Pass --options before the passthrough separator so the wrapper can enforce the scenario/options guard (or drop --scenario)." >&2
          return 1
          ;;
      esac
    done
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
      --cr)
        flag_cr=true
        ;;
      --sr)
        flag_sr=true
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

  local executor
  if [[ -n "$executor_choice" ]]; then
    executor="$executor_choice"
  else
    executor="taskvine"
  fi

  if [[ "$executor" != "taskvine" && "$executor" != "futures" && "$executor" != "iterative" ]]; then
    echo "Error: executor must be one of taskvine, futures, or iterative (got '$executor')" >&2
    return 1
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

  check_active_env
  local env_tarball=""
  if [[ "$executor" == "taskvine" && "$user_env_override" == false && "$dry_run" == false ]]; then
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

  local has_run2=false
  local has_run3=false
  for year in "${resolved_years[@]}"; do
    local lower_year=${year,,}
    if [[ -n "${run2_year_map[$lower_year]:-}" ]]; then
      has_run2=true
    elif [[ "$year" == "2022" || "$year" == "2022EE" ]]; then
      has_run3=true
    fi
  done

  if [[ "$has_run2" == true && "$has_run3" == true ]]; then
    echo "Error: mixing Run-2 and Run-3 eras in a single invocation is not supported." >&2
    return 1
  fi

  if [[ "$scenario_specified" == false ]]; then
    # Default scenarios mirror the canonical Run-2/Run-3 bundles wired through
    # analysis/topeft_run2/scenario_registry.py.
    if [[ "$has_run2" == true ]]; then
      scenario_args=("TOP_22_006")
    elif [[ "$has_run3" == true ]]; then
      scenario_args=("fwd_analysis")
    fi
  fi

  if [[ ${#scenario_args[@]} -gt 1 ]]; then
    echo "Error: only one --scenario can be specified per run (requested: ${scenario_args[*]})." >&2
    return 1
  fi

  local scenario_name=""
  if [[ ${#scenario_args[@]} -eq 1 ]]; then
    scenario_name="${scenario_args[0]}"
  fi

  if [[ "$scenario_specified" == false && "$user_options_override" == false && \
        "$has_run2" == true && "$has_run3" == false ]]; then
    local run2_options_path="$script_dir/configs/fullR2_run.yml"
    if [[ -f "$run2_options_path" ]]; then
      local profile_name
      if [[ "$flag_cr" == true ]]; then
        profile_name="cr"
      else
        profile_name="sr"
      fi
      auto_options_spec="$run2_options_path:$profile_name"
    fi
  fi

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

  local -a input_specs=()
  if [[ ${#user_samples[@]} -gt 0 ]]; then
    for sample_path in "${user_samples[@]}"; do
      if [[ ! -e "$sample_path" ]]; then
        echo "Error: sample override not found: $sample_path" >&2
        return 1
      fi
      input_specs+=("$sample_path")
    done
  else
    input_specs=("${cfgs_list[@]}")
  fi

  if [[ ${#input_specs[@]} -eq 0 ]]; then
    echo "Error: no cfg/json inputs resolved" >&2
    return 1
  fi

  local cfgs
  cfgs=$(IFS=,; echo "${input_specs[*]}")

  local options_in_effect=false
  if [[ "$user_options_override" == true || -n "$auto_options_spec" ]]; then
    options_in_effect=true
  fi

  echo "Resolved years: ${resolved_years[*]}"
  echo "Resolved cfg inputs: $cfgs"
  echo "Executor: $executor"
  if [[ "$executor" == "taskvine" ]]; then
    echo "TaskVine manager: $manager_name"
    if [[ -n "$env_tarball" ]]; then
      echo "Environment archive: $env_tarball"
    elif [[ "$dry_run" == false && "$user_env_override" == false ]]; then
      echo "Warning: TaskVine environment tarball not set"
    fi
  else
    echo "Futures prefetch: $futures_prefetch | retries: $futures_retries"
  fi
  if [[ "$options_in_effect" == true ]]; then
    if [[ -n "$auto_options_spec" ]]; then
      echo "Options profile: $auto_options_spec (auto-selected)"
    else
      echo "Options profile: supplied via CLI"
    fi
    echo "Scenario: controlled by options profile"
  elif [[ -n "$scenario_name" ]]; then
    echo "Scenario: $scenario_name"
  fi

  if [[ -z "$workers" ]]; then
    case "$executor" in
      taskvine)
        workers=8
        ;;
      futures)
        workers=4
        ;;
      iterative)
        workers=1
        ;;
    esac
  fi

  local -a options=(
    --outname "$out_name"
    --outpath "$outdir"
    --nworkers "$workers"
    --summary-verbosity brief
    --executor "$executor"
  )

  if [[ "$user_chunk_override" == false ]]; then
    options+=(--chunksize 50000)
  fi
  if [[ "$executor" == "taskvine" ]]; then
    options+=(--manager-name "$manager_name")
    if [[ "$user_env_override" == false && -n "$env_tarball" ]]; then
      options+=(--environment-file "$env_tarball")
    fi
  elif [[ "$executor" == "futures" ]]; then
    options+=(--futures-prefetch "$futures_prefetch")
    options+=(--futures-retries "$futures_retries")
    options+=(--futures-retry-wait "$futures_retry_wait")
  fi

  if [[ "$flag_cr" == true ]]; then
    options+=(--skip-sr)
  else
    options+=(--skip-cr --do-systs)
  fi

  if [[ "$debug_logging" == true ]]; then
    options+=(--debug-logging)
  fi
  if [[ -n "$log_level" ]]; then
    options+=(--log-level "$log_level")
  fi

  local -a run_cmd=(python run_analysis.py "$cfgs")
  run_cmd+=("${options[@]}")
  if [[ -n "$auto_options_spec" ]]; then
    run_cmd+=(--options "$auto_options_spec")
  fi
  if [[ "$options_in_effect" == false && -n "$scenario_name" ]]; then
    run_cmd+=(--scenario "$scenario_name")
  fi
  run_cmd+=("${extra_args[@]}")

  printf "\nResolved command:\n%s\n\n" "${run_cmd[*]}"
  if [[ "$dry_run" == true ]]; then
    return 0
  fi

  time "${run_cmd[@]}"
}

main "$@"
exit_code=$?
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exit "$exit_code"
else
  return "$exit_code"
fi
