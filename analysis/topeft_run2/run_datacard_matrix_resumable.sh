#!/usr/bin/env bash
set -euo pipefail

main() {
  local wrapper_source
  local wrapper_directory
  local engine_python

  wrapper_source="${BASH_SOURCE[0]}"
  case "$wrapper_source" in
    */*) wrapper_directory="${wrapper_source%/*}" ;;
    *) wrapper_directory='.' ;;
  esac
  wrapper_directory="$(CDPATH= cd -- "$wrapper_directory" && pwd -P)" || return
  if command -v python >/dev/null 2>&1; then
    engine_python="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    engine_python="$(command -v python3)"
  else
    printf '%s\n' 'python or python3 is required for the runner engine' >&2
    return 127
  fi

  exec "$engine_python" "${wrapper_directory}/datacard_matrix_runner.py" "$@"
}

main "$@"
