#!/usr/bin/env bash
# Read-only view of the canonical status for one v3 datacard matrix manifest.
set -euo pipefail

usage() {
  printf 'Usage: %s [--once] [--refresh-seconds N] [--log-lines N] [--scheduler-status FILE] MANIFEST.json\n' "${0##*/}"
}

die() {
  printf 'Dashboard error: %s\n' "$*" >&2
  exit 2
}

require_positive_integer() {
  local label=$1 value=$2
  [[ $value =~ ^[0-9]+$ ]] && (( 10#$value > 0 )) || die "$label must be a positive integer"
}

once=false
refresh_seconds=5
log_lines=20
manifest=
scheduler_args=()
while (( $# )); do
  case $1 in
    --once) once=true; shift ;;
    --refresh-seconds)
      (( $# >= 2 )) || die '--refresh-seconds needs a value'
      refresh_seconds=$2; shift 2 ;;
    --log-lines)
      (( $# >= 2 )) || die '--log-lines needs a value'
      log_lines=$2; shift 2 ;;
    --scheduler-status)
      (( $# >= 2 )) || die '--scheduler-status needs a value'
      scheduler_args=(--scheduler-status "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [[ -z $manifest ]] || die 'exactly one manifest path is required'
       manifest=$1; shift ;;
  esac
done
[[ -n $manifest ]] || { usage >&2; exit 2; }
require_positive_integer refresh_seconds "$refresh_seconds"
require_positive_integer log_lines "$log_lines"
[[ -f $manifest && -r $manifest ]] || die "manifest is missing or unreadable: $manifest"
for required_command in jq sha256sum date tail sleep; do
  command -v "$required_command" >/dev/null 2>&1 || die "required command is missing: $required_command"
done

script_path=${BASH_SOURCE[0]}
[[ $script_path == */* ]] || script_path=$(command -v "$script_path")
script_dir=$(CDPATH= cd -- "${script_path%/*}" && pwd -P) || die 'cannot resolve script directory'
runner=$script_dir/../run_datacard_matrix_resumable.sh
[[ -x $runner ]] || die "canonical runner is missing or not executable: $runner"

jq -e '.schema == "topeft_datacard_matrix_v3" and (.rows | type == "array" and length > 0)' \
  "$manifest" >/dev/null 2>&1 || die "manifest is not readable v3 JSON: $manifest"
manifest_hash=$(sha256sum -- "$manifest") || die 'cannot hash manifest'
manifest_hash=${manifest_hash%% *}

progress_bar() {
  local done=$1 total=$2 width=28 index filled=0
  (( total == 0 )) || filled=$((done * width / total))
  printf '['
  for ((index = 0; index < width; index++)); do
    if (( index < filled )); then printf '#'; else printf '.'; fi
  done
  printf ']'
}

render_once() {
  local status_json summary complete_units active_units pending_units problem_units
  local total_units complete_targets active_targets total_targets lock_state
  local active_row active_attempt started_at log_path table row_state row_id attempt logical_row target_count channels
  if ! status_json=$("$runner" --status "$manifest" "${scheduler_args[@]}" 2>&1); then
    die "canonical --status failed: $status_json"
  fi
  if ! jq -e --arg manifest_hash "$manifest_hash" --slurpfile manifest "$manifest" '
    . as $status | $manifest[0] as $source |
    .schema == "topeft_datacard_matrix_read_only_result_v2" and
    .mode == "status" and .mutated == false and
    .manifest_sha256 == $manifest_hash and
    (.rows | type == "array") and
    ([.rows[] | [.row_id, .attempt_id]] | sort) ==
      ([$source.rows[] | [.row_id, .attempt_id]] | sort) and
    ([.rows[].row_id] | unique | length) == ($source.rows | length) and
    ([.rows[] | .status] | all(. == "complete" or . == "active" or
      . == "not_started" or . == "stale_owner_free_lock" or
      . == "partial_staging_attempt" or . == "complete_validated_staging_not_promoted" or
      . == "promoted_incomplete" or . == "promoted_complete_without_receipt" or
      . == "held_lock_active_or_unknown" or . == "invalid_receipt" or
      . == "canonical_collision_or_ambiguous_state")) and
    (.lock_held == null or (.lock_held | type == "boolean")) and
    (.owner_metadata == null or (.owner_metadata | type == "object"))
  ' <<<"$status_json" >/dev/null 2>&1; then
    die 'canonical --status returned invalid JSON, unknown state, or rows inconsistent with the manifest'
  fi

  summary=$(jq -r --slurpfile manifest "$manifest" '
    . as $status | $manifest[0] as $source |
    [
      ($status.rows | length),
      ([$status.rows[] | select(.status == "complete")] | length),
      ([$status.rows[] | select(.status == "active")] | length),
      ([$status.rows[] | select(.status == "not_started" or .status == "stale_owner_free_lock")] | length),
      ([$status.rows[] | select(.status != "complete" and .status != "active" and .status != "not_started" and .status != "stale_owner_free_lock")] | length),
      ([$source.rows[] | .physical_channels | length] | add // 0),
      ([$source.rows[] as $row | select(any($status.rows[]; .row_id == $row.row_id and .status == "complete")) | $row.physical_channels | length] | add // 0),
      ([$source.rows[] as $row | select(any($status.rows[]; .row_id == $row.row_id and .status == "active")) | $row.physical_channels | length] | add // 0),
      (if $status.lock_held == null then "unknown" else ($status.lock_held | tostring) end)
    ] | @tsv
  ' <<<"$status_json") || die 'cannot summarize canonical status'
  IFS=$'\t' read -r total_units complete_units active_units pending_units problem_units total_targets complete_targets active_targets lock_state <<<"$summary"

  active_row=$(jq -r '[.rows[] | select(.status == "active") | .row_id][0] // "-"' <<<"$status_json")
  active_attempt=$(jq -r '[.rows[] | select(.status == "active") | .attempt_id][0] // "-"' <<<"$status_json")
  started_at=$(jq -r --arg row "$active_row" --arg attempt "$active_attempt" '
    if .owner_metadata.current_row_id == $row and .owner_metadata.current_attempt_id == $attempt
    then .owner_metadata.current_row_started_at // "-" else "-" end
  ' <<<"$status_json")

  if [[ $once == false && -t 1 ]]; then printf '\033[H\033[2J'; fi
  printf 'DATACARD MATRIX DASHBOARD\n'
  printf 'Manifest       : %s\n' "${manifest##*/}"
  printf 'Logical rows   : %s\n' "$(jq -r '[.rows[].logical_row_id] | unique | join(", ")' "$manifest")"
  printf 'Control root   : %s\n' "$(jq -r '.control_root' "$manifest")"
  printf 'Output roots   : %s distinct (per-unit paths in manifest)\n' "$(jq '[.rows[].output_root] | unique | length' "$manifest")"
  printf 'Output context : %s\n' "$(jq -r '[.rows[].output_root | split("/")[:-1] | join("/")] | unique |
    if length == 1 then .[0] else "\(length) output parent directories" end' "$manifest")"
  printf 'Updated        : %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  printf 'Execution units: %s / %s ' "$complete_units" "$total_units"
  progress_bar "$complete_units" "$total_units"
  printf ' %s%%\n' "$((complete_units * 100 / total_units))"
  printf 'Physical targets: %s / %s ' "$complete_targets" "$total_targets"
  progress_bar "$complete_targets" "$total_targets"
  printf ' %s%%\n' "$((complete_targets * 100 / total_targets))"
  printf 'Active units   : %s (%s targets)  Pending: %s  Problems: %s  Lock: %s\n' \
    "$active_units" "$active_targets" "$pending_units" "$problem_units" "$lock_state"
  jq -r '.rows[] | select(.reservation != null) |
    "[\(.reservation.state)] Logical reservation: \(.row_id) Backend: \(.reservation.backend // "unknown") Scheduler: \(.reservation.scheduler_id // "not observed") (H2 receipt alone establishes completion)",
    (if .reservation.state == "HELD" then "  HELD terminal: removal pending; hold reason remains scheduler evidence; no automatic release."
     elif .reservation.state == "UNKNOWN" then "  BLOCKED: scheduler ownership is unresolved."
     elif .reservation.state == "GONE" then "  Next ownership attempt automatically reconciles H2 before reclaiming."
     else empty end),
    (if .reservation.reason then "  \(.reservation.error_code // "reservation_error"): \(.reservation.reason)" else empty end),
    (if .reservation.hold_evidence then "  Hold evidence: \(.reservation.hold_evidence | tojson)" else empty end)' <<<"$status_json"
  jq -r '.owner_metadata as $legacy | .rows[] | select(.status == "active") |
    (.owner_metadata // $legacy // {}) as $owner |
    "Current unit   : \(.row_id)  Attempt: \(.attempt_id)  Started: \($owner.current_row_started_at // "-")  Backend: \($owner.backend // "unknown")  Scheduler: \($owner.scheduler_id // "-")"' <<<"$status_json"
  printf '\n'
  printf 'Status source  : runner --status; live reservations are operational observations, validated H2 receipts establish completion. This view does not establish physics validation, memory/resource validation, or topology suitability.\n\n'

  printf '%-22s %-31s %-16s %-16s %4s  %s\n' STATE EXECUTION_UNIT LOGICAL_ROW ATTEMPT NCH PHYSICAL_CHANNELS
  printf '%-22s %-31s %-16s %-16s %4s  %s\n' ---------------------- ------------------------------- ---------------- ---------------- ---- -----------------
  table=$(jq -r --slurpfile manifest "$manifest" '
    . as $status | $manifest[0].rows[] as $row |
    ($status.rows[] | select(.row_id == $row.row_id and .attempt_id == $row.attempt_id)) as $state |
    [($state.status | if . == "complete" then "[OK]" elif . == "active" then "[ACTIVE]"
      elif . == "not_started" then "[PENDING]" elif . == "stale_owner_free_lock" then "[STALE OWNER]"
      elif . == "partial_staging_attempt" then "[PARTIAL]"
      elif . == "complete_validated_staging_not_promoted" then "[STAGED READY]"
      elif . == "promoted_incomplete" then "[PUBLISHING]"
      elif . == "promoted_complete_without_receipt" then "[RECEIPT PENDING]"
      elif . == "held_lock_active_or_unknown" then "[LOCK UNKNOWN]"
      elif . == "invalid_receipt" then "[INVALID RECEIPT]" else "[COLLISION]" end),
     $row.row_id, $row.attempt_id, $row.logical_row_id,
     ($row.physical_channels | length), ($row.physical_channels | join(","))] | @tsv
  ' <<<"$status_json") || die 'cannot join manifest rows with canonical status'
  while IFS=$'\t' read -r row_state row_id attempt logical_row target_count channels; do
    printf '%-22s %-31s %-16s %-16s %4s  %s\n' "$row_state" "$row_id" "$logical_row" "$attempt" "$target_count" "$channels"
  done <<<"$table"

  jq -r '.rows[] | select(.scheduler_metadata != null) |
    "Scheduler observation: \(.row_id) \(.scheduler_metadata.scheduler_id) \(.scheduler_metadata.state) at \(.scheduler_metadata.observed_at) (operational only)"' <<<"$status_json"

  printf '\nCURRENT LOG\n-----------\n'
  if [[ $active_row == - ]]; then
    printf 'No active execution unit.\n'
    return
  fi
  while IFS=$'\t' read -r active_row active_attempt; do
  log_path=$(jq -r --arg row "$active_row" --arg attempt "$active_attempt" \
    '[.rows[] | select(.row_id == $row and .attempt_id == $attempt) | .active_log_path][0] // empty' <<<"$status_json")
  if [[ -z $log_path ]]; then
    log_path=$(jq -r --arg row "$active_row" --arg attempt "$active_attempt" \
      '[.rows[] | select(.row_id == $row and .attempt_id == $attempt) | .log_path][0] // empty' "$manifest")
  fi
  [[ -n $log_path ]] || die "active unit has no log path in manifest: $active_row"
  if [[ ! -e $log_path ]]; then
    printf '[waiting for log] %s\n' "$log_path"
  elif [[ ! -f $log_path || ! -r $log_path ]]; then
    die "active log is not a readable file: $log_path"
  else
    printf '%s\n' "$log_path"
    tail -n "$log_lines" -- "$log_path" || die "cannot read active log: $log_path"
  fi
  done < <(jq -r '.rows[] | select(.status == "active") | [.row_id, .attempt_id] | @tsv' <<<"$status_json")

}

while :; do
  render_once
  [[ $once == false ]] || break
  sleep "$refresh_seconds"
done
