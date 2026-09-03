#!/usr/bin/env bash
set -euo pipefail
source_root=/cms/cephfs/data
destination_root=/project01/ndcms/apiccine
usage(){ printf 'usage: %s --dry-run|--execute <store_entry> <size> <required_free_bytes>\n' "$0" >&2; exit 64; }
[[ "$#" -eq 4 ]] || usage
mode="$1"; store_entry="$2"; expected_size="$3"; required_free_bytes="$4"
[[ "$mode" == --dry-run || "$mode" == --execute ]] || usage
[[ "$store_entry" == /store/* && "$expected_size" =~ ^[0-9]+$ && "$required_free_bytes" =~ ^[0-9]+$ ]] || { printf 'invalid_arguments\n' >&2; exit 65; }
source_path="$source_root$store_entry"
destination_path="$destination_root$store_entry"
if [[ "$mode" == --dry-run ]]; then printf 'source=%s\tdestination=%s\texpected_size_bytes=%s\trequired_free_bytes=%s\n' "$source_path" "$destination_path" "$expected_size" "$required_free_bytes"; exit 0; fi
[[ "$(hostname -f)" == glados.crc.nd.edu ]] || { printf 'host_gate_failed\n' >&2; exit 66; }
[[ -f "$source_path" && -r "$source_path" ]] || { printf 'source_gate_failed\n' >&2; exit 67; }
[[ "$(stat -c '%s' -- "$source_path")" == "$expected_size" ]] || { printf 'source_size_mismatch\n' >&2; exit 68; }
[[ ! -e "$destination_path" && ! -L "$destination_path" ]] || { printf 'destination_collision_before_directory_create\n' >&2; exit 73; }
destination_directory="$(dirname -- "$destination_path")"; destination_basename="$(basename -- "$destination_path")"
mkdir -p -- "$destination_directory"
[[ ! -e "$destination_path" && ! -L "$destination_path" ]] || { printf 'destination_collision_after_directory_create\n' >&2; exit 73; }
if (( required_free_bytes > 0 )); then available_bytes="$(df -B1 --output=avail "$destination_directory" | tail -n 1 | tr -d ' ')"; [[ "$available_bytes" =~ ^[0-9]+$ ]] && (( available_bytes >= required_free_bytes )) || { printf 'capacity_gate_failed\n' >&2; exit 72; }; fi
partial_path="$(mktemp -- "$destination_directory/.$destination_basename.project01_partial.XXXXXX")"
cp -- "$source_path" "$partial_path"
[[ "$(stat -c '%s' -- "$partial_path")" == "$expected_size" ]] || { printf 'partial_size_mismatch\n' >&2; exit 74; }
[[ ! -e "$destination_path" && ! -L "$destination_path" ]] || { printf 'destination_collision_before_publish\n' >&2; exit 73; }
ln -T -- "$partial_path" "$destination_path"
[[ "$(stat -c '%s' -- "$destination_path")" == "$expected_size" ]] || { printf 'final_size_mismatch\n' >&2; exit 75; }
rm -- "$partial_path"
printf 'state=success source=%q destination=%q size_bytes=%q\n' "$source_path" "$destination_path" "$expected_size"
