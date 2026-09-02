#!/usr/bin/env bash
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "$0")" && pwd)"
helper="$script_directory/project01_copy_one_file.sh"
usage(){ printf 'usage: %s --dry-run|--execute <manifest.tsv> <log_directory>\n' "$0" >&2; exit 64; }
[[ "$#" -eq 3 ]] || usage
mode="$1"; manifest_path="$2"; log_directory="$3"
[[ "$mode" == --dry-run || "$mode" == --execute ]] || usage
[[ -x "$helper" && -r "$manifest_path" ]] || { printf 'missing_helper_or_manifest\n' >&2; exit 66; }
entries=(); sizes=(); required_total=0; line_number=0
while IFS=$'\t' read -r store_entry expected_size extra || [[ -n "$store_entry$expected_size$extra" ]]; do
  line_number=$((line_number+1))
  [[ "$store_entry" == /store/* && "$expected_size" =~ ^[0-9]+$ && -z "$extra" ]] || { printf 'malformed_manifest_row line=%s\n' "$line_number" >&2; exit 68; }
  entries+=("$store_entry"); sizes+=("$expected_size"); required_total=$((required_total+expected_size))
done < "$manifest_path"
(( ${#entries[@]} > 0 )) || { printf 'empty_manifest\n' >&2; exit 68; }
for index in "${!entries[@]}"; do
  required=0; (( index == 0 )) && required="$required_total"; command_id="$(printf 'copy_pending_%05d' "$((index+1))")"
  [[ "$mode" == --dry-run ]] && printf '%s\tsource=/cms/cephfs/data%s\tdestination=/project01/ndcms/apiccine%s\texpected_size_bytes=%s\trequired_free_bytes=%s\n' "$command_id" "${entries[index]}" "${entries[index]}" "${sizes[index]}" "$required"
done
[[ "$mode" == --execute ]] || exit 0
[[ ! -e "$log_directory" && ! -L "$log_directory" ]] || { printf 'log_directory_already_exists\n' >&2; exit 69; }
mkdir -p -- "$log_directory"
for index in "${!entries[@]}"; do
  required=0; (( index == 0 )) && required="$required_total"; command_id="$(printf 'copy_pending_%05d' "$((index+1))")"; log_path="$log_directory/$command_id.log"
  { printf 'start_time: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; "$helper" --execute "${entries[index]}" "${sizes[index]}" "$required"; printf 'end_time: %s\nexit_code: 0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } 2>&1 | tee -a "$log_path" || { printf 'stopping_after_failure=%s\n' "$command_id" >&2; exit 1; }
done
