#!/usr/bin/env python

import argparse
import json
import time
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import topcoffea.modules.utils as utils
import topcoffea.modules.remote_environment as remote_environment
from topcoffea.modules.paths import topcoffea_path

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.data_driven_products import (
    DATA_DRIVEN_PRODUCT_NAMES,
    certify_data_driven_preflight,
    resolve_data_driven_products,
)
from topeft.modules.histogram_category_compatibility import (
    histogram_category_compatibility_error,
    validate_histogram_category_compatibility,
)
from topeft.modules.histogram_artifact import (
    lineage_input_from_sidecar,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import (
    NOMINAL_CONTAINER_LAYOUT,
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    canonicalize_nominal_keys,
    validate_nominal_mapping,
)
from topeft.modules.nonprompt_policy import (
    certify_active_nonprompt_policy,
    nonprompt_policy_error,
)
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.sumw2_policy import (
    resolve_sumw2_storage_mode,
    resolve_sumw2_storage_policy,
    sumw2_target,
)
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
    production_sample_profile_error,
    validate_active_sample_profile,
)
from topeft.modules.get_renormfact_envelope import raise_unsupported_renormfact_envelope
from topeft.modules.ttgamma_photon_history import (
    SPLIT_SAMPLE_ROLE_POLICY,
    SUPPORTED_SAMPLE_ROLE_POLICIES,
    get_ttgamma_sample_role_policy,
)
import analysis_processor
from analysis.topeft_run2.analysis_processor import (
    ANALYSIS_MODE_EXCLUSIVE_ERROR,
)

LST_OF_KNOWN_EXECUTORS = ["futures", "work_queue", "taskvine"]

WGT_VAR_LST = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    #"nSumOfWeights_renormfactUp",
    #"nSumOfWeights_renormfactDown",
]


def _certify_nonprompt_policy_before_executor(
    samples,
    *,
    nonprompt_enabled,
    configuration_source,
):
    """Hard-fail the active nonprompt universe before processor/executor setup."""

    if not nonprompt_enabled:
        return None
    try:
        return certify_active_nonprompt_policy(
            samples,
            configuration_source=configuration_source,
        )
    except nonprompt_policy_error as error:
        raise SystemExit(str(error)) from None

def _ensure_topcoffea_data_available(skip_check=False):
    if skip_check:
        return

    target_relpath = "data/pileup/pileup_2016GH.root"
    guidance = (
        "Topcoffea shared data files are missing. Re-run scripts/install_topcoffea.sh "
        "from the topeft checkout so the matching topcoffea branch (e.g. run3_test_mmerged) "
        "and its data bundles are installed, or pass --skip-topcoffea-data-check if your setup "
        "provides the resources elsewhere."
    )

    try:
        pileup_path = topcoffea_path(target_relpath)
    except FileNotFoundError as exc:
        raise SystemExit(f"{guidance} (lookup failed for {target_relpath}).") from exc

    if not os.path.exists(pileup_path):
        raise SystemExit(f"{guidance} (expected {pileup_path}).")


def _format_worker_exception(exception_obj):
    if exception_obj in (None, 0):
        return None

    try:
        return str(exception_obj)
    except Exception:
        return repr(exception_obj)


def _dedupe_preserve_order(values):
    if values is None:
        return None

    if isinstance(values, str):
        iterable = [values]
    else:
        iterable = values

    deduped_values = []
    seen_values = set()
    for raw_value in iterable:
        value = str(raw_value).strip()
        if not value or value in seen_values:
            continue
        deduped_values.append(value)
        seen_values.add(value)
    return deduped_values


def _format_name_list(names):
    return ", ".join(map(str, names)) if names else "<none>"


def _requested_products_for_histogram_preflight(
    data_driven_products_config,
    *,
    data_driven_products_present,
    legacy_do_np,
):
    """Resolve only the enabled product names needed by the early preflight."""

    if not data_driven_products_present:
        return DATA_DRIVEN_PRODUCT_NAMES if legacy_do_np else ()
    if not isinstance(data_driven_products_config, Mapping):
        return ()

    requested_products = []
    for product_name in DATA_DRIVEN_PRODUCT_NAMES:
        product_config = data_driven_products_config.get(product_name)
        if (
            isinstance(product_config, Mapping)
            and product_config.get("enabled") is True
        ):
            requested_products.append(product_name)
    return tuple(requested_products)


def _resolve_category_group_selection(
    *,
    category_groups,
    offz_3l_split,
    tau_h_analysis,
    fwd_analysis,
    all_analysis,
    skip_sr,
    skip_cr,
):
    requested_category_groups = _dedupe_preserve_order(category_groups)
    sr_block_name, cr_block_name = analysis_processor.resolve_category_dict_names(
        offz_3l_split,
        tau_h_analysis,
        fwd_analysis,
        all_analysis,
    )
    channel_config = analysis_processor.load_category_config()

    active_blocks = []
    for region_label, block_name, skip_region in (
        ("SR", sr_block_name, skip_sr),
        ("CR", cr_block_name, skip_cr),
    ):
        if skip_region:
            continue
        if block_name not in channel_config:
            raise RuntimeError(
                f"Resolved {region_label} ch_lst.json block '{block_name}' was not found in channels/ch_lst.json."
            )
        block = channel_config[block_name]
        active_blocks.append(
            {
                "region": region_label,
                "block_name": block_name,
                "block": block,
                "available_groups": list(block.keys()),
            }
        )

    if requested_category_groups is not None and not active_blocks:
        raise SystemExit(
            "Cannot use --category-groups when both --skip-sr and --skip-cr are set."
        )

    if requested_category_groups is not None:
        missing_groups = [
            group_name
            for group_name in requested_category_groups
            if not any(group_name in block_info["block"] for block_info in active_blocks)
        ]
        if missing_groups:
            active_block_summary = (
                "; ".join(
                    "{}={} [{}]".format(
                        block_info["region"],
                        block_info["block_name"],
                        _format_name_list(block_info["available_groups"]),
                    )
                    for block_info in active_blocks
                )
                if active_blocks
                else "<none>"
            )
            raise SystemExit(
                "Unknown or incompatible category group(s): {}. Active ch_lst.json block(s): {}.".format(
                    _format_name_list(missing_groups),
                    active_block_summary,
                )
            )

    selected_category_dicts = {"SR": {}, "CR": {}}
    for block_info in active_blocks:
        if requested_category_groups is None:
            selected_groups = block_info["available_groups"]
            selected_block = dict(block_info["block"])
        else:
            selected_groups = [
                group_name
                for group_name in requested_category_groups
                if group_name in block_info["block"]
            ]
            selected_block = {
                group_name: block_info["block"][group_name] for group_name in selected_groups
            }

        block_info["selected_groups"] = selected_groups
        block_info["selected_block"] = selected_block
        selected_category_dicts[block_info["region"]] = selected_block

    return {
        "requested_category_groups": requested_category_groups,
        "resolved_sr_block_name": sr_block_name,
        "resolved_cr_block_name": cr_block_name,
        "active_blocks": active_blocks,
        "sr_category_dict": selected_category_dicts["SR"],
        "cr_category_dict": selected_category_dicts["CR"],
    }


def _log_category_group_selection(category_group_selection):
    active_blocks = category_group_selection["active_blocks"]
    requested_category_groups = category_group_selection["requested_category_groups"]

    if not active_blocks:
        print("Category-group selection: no active ch_lst.json blocks (both SR and CR skipped).")
        return

    if requested_category_groups is None:
        print(
            "Category-group selection: no --category-groups filter requested; using all groups from each active block."
        )
    else:
        print(
            "Requested category groups (deduplicated user order): {}".format(
                _format_name_list(requested_category_groups)
            )
        )

    for block_info in active_blocks:
        print(
            "Resolved {} ch_lst.json block: {}".format(
                block_info["region"], block_info["block_name"]
            )
        )
        print(
            "Available {} category groups: {}".format(
                block_info["region"], _format_name_list(block_info["available_groups"])
            )
        )
        if requested_category_groups is None:
            print(
                "Selected {} category groups: all ({})".format(
                    block_info["region"],
                    _format_name_list(block_info["selected_groups"]),
                )
            )
        else:
            print(
                "Selected {} category groups: {}".format(
                    block_info["region"],
                    _format_name_list(block_info["selected_groups"]),
                )
            )


def _environment_rebuild_command(interpreter=None, script_path=None):
    return shlex.join(
        [
            interpreter or os.path.realpath(os.sys.executable),
            script_path or os.path.abspath(__file__),
            "--prepare-env-only",
            "--rebuild-env",
        ]
    )


def _print_environment_identity(validation):
    print(f"env_file: {validation['archive_path']}")
    print(f"env_file_sha256: {validation['archive_sha256']}")
    print(f"env_manifest: {validation['manifest_path']}")
    print(f"environment_fingerprint: {validation['environment_fingerprint']}")
    print(f"environment_validation_status: {validation['status']}")
    for package in validation.get("editable_packages", []):
        if package.get("package_name") == "topcoffea":
            print(f"topcoffea_git_commit: {package.get('git_commit')}")
            print(f"topcoffea_relevant_source_fingerprint: {package.get('watched_source_fingerprint')}")


def _strict_environment_error(validation):
    reasons = "\n".join(f"  - {reason}" for reason in validation["mismatches"])
    raise SystemExit(
        "environment_status: {}\nreason(s):\n{}\n\nCreate a compatible current environment with:\n\n"
        "  {}\n\nThen rerun with:\n  --env-file <new archive path>".format(
            validation["status"], reasons or "  - archive validation failed", _environment_rebuild_command()
        )
    )


def _resolve_environment_file(
    env_override,
    use_remote_env,
    extra_pip_local=None,
    rebuild_env=False,
    snapshot=False,
    integrity_only=False,
):
    extra_pip_local = extra_pip_local or {"topeft": ["topeft", "setup.py"]}
    if env_override:
        env_path = os.path.abspath(os.path.expanduser(env_override))
        integrity = remote_environment.validate_environment_archive(env_path, snapshot=snapshot)
        if integrity["status"] == "invalid_archive":
            _strict_environment_error(integrity)
        if integrity_only:
            if not integrity["usable"]:
                _strict_environment_error(integrity)
            _print_environment_identity(integrity)
            return env_path
        current_request = remote_environment.resolve_environment_request(
            extra_pip_local=extra_pip_local,
            unstaged="fail",
        )
        validation = remote_environment.validate_environment_archive(
            env_path, current_request, snapshot=snapshot
        )
        if not validation["usable"]:
            _strict_environment_error(validation)
        if snapshot:
            print("SNAPSHOT ENVIRONMENT MODE")
            print("compatibility enforcement: bypassed explicitly with --snapshot")
            for mismatch in validation["mismatches"]:
                print(f"snapshot_environment_mismatch: {mismatch}")
        _print_environment_identity(validation)
        return env_path

    if not use_remote_env:
        return None

    try:
        return remote_environment.get_environment(
            extra_pip_local=extra_pip_local,
            force=rebuild_env,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Failed to build a remote execution environment (poncho_package_create errored). "
            "Provide --env-file pointing to a known-good poncho tarball (e.g. generated from environment.yml) "
            "or rerun with --no-remote-env if workers already have the dependencies."
        ) from exc


def _prepare_work_queue_staging_directory(filepath_override=None):
    requested_path = filepath_override or f"/groups/klannon/{os.environ.get('USER', 'user')}/workers"
    path_preexisted = os.path.exists(requested_path)

    try:
        os.makedirs(requested_path, exist_ok=True)
        return requested_path, not path_preexisted
    except OSError as exc:
        print(
            "Warning: Failed to create Work Queue staging directory {} ({}). Falling back to a "
            "system temporary location.".format(requested_path, exc)
        )

    fallback_path = tempfile.gettempdir()
    print(f"Using fallback Work Queue staging directory: {fallback_path}")
    return fallback_path, False


def _cleanup_work_queue_staging_directory(path, eligible_for_cleanup):
    if not path or not eligible_for_cleanup:
        return

    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        print(
            "Warning: Failed to clean up Work Queue staging directory {} ({}). You may want to "
            "remove it manually.".format(path, exc)
        )


_REQUIRED_JSON_KEYS = (
    "files",
    "year",
    "xsec",
    "nEvents",
    "nGenEvents",
    "nSumOfWeights",
    "isData",
    "histAxisName",
    "treeName",
    "options",
)
_YEAR_CANONICAL_MAP = {
    "2016": "2016",
    "UL16": "2016",
    "UL2016": "2016",
    "2016APV": "2016APV",
    "UL16APV": "2016APV",
    "UL2016APV": "2016APV",
    "2017": "2017",
    "UL17": "2017",
    "UL2017": "2017",
    "2018": "2018",
    "UL18": "2018",
    "UL2018": "2018",
    "2022": "2022",
    "2022EE": "2022EE",
    "2023": "2023",
    "2023BPix": "2023BPix",
}
_TRUSTED_YEAR_HINT_KEYS = {"files", "path", "histAxisName"}
_DATE_TAG_PATTERN = re.compile(
    r"\b\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)20\d{2}\b",
    re.IGNORECASE,
)
_YEAR_STRONG_UL_PATTERN = re.compile(r"(UL(?:2016APV|16APV|2016|16|2017|17|2018|18))")
_YEAR_STRONG_RUN_ERA_PATTERN = re.compile(r"(Run(2016|2017|2018|2022|2023)[A-H])")
_YEAR_STRONG_DELIMITED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(2023BPix|2022EE|2016APV|UL2016APV|UL16APV|2016|2017|2018|2022|2023)(?![A-Za-z0-9])"
)
_YEAR_WEAK_PATTERN = re.compile(r"(?<![A-Za-z0-9])(20(?:16|17|18|22|23))(?![A-Za-z0-9])")


def _canonicalize_year_label(year_label):
    return _YEAR_CANONICAL_MAP.get(str(year_label), str(year_label))


def _strip_year_keys(payload):
    if isinstance(payload, dict):
        return {k: _strip_year_keys(v) for k, v in payload.items() if k != "year"}
    if isinstance(payload, list):
        return [_strip_year_keys(item) for item in payload]
    return payload


def _collect_trusted_year_scan_strings(payload):
    values = []

    def _walk(node):
        if isinstance(node, dict):
            files = node.get("files")
            if "files" in _TRUSTED_YEAR_HINT_KEYS and isinstance(files, list):
                for item in files:
                    if isinstance(item, str):
                        values.append(("files", item))
            for key in ("path", "histAxisName"):
                value = node.get(key)
                if key in _TRUSTED_YEAR_HINT_KEYS and isinstance(value, str):
                    values.append((key, value))
            for nested in node.values():
                _walk(nested)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return values


def _collapse_year_families(canonical_hits):
    collapsed = set(canonical_hits)
    if "2016APV" in collapsed:
        collapsed.discard("2016")
    if "2022EE" in collapsed:
        collapsed.discard("2022")
    if "2023BPix" in collapsed:
        collapsed.discard("2023")
    return collapsed


def _snippet_around(text, start, end, radius=24):
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right]
    return snippet.replace("\n", " ")


def _record_year_hit(hit_map, token, canonical, source_key, source_value, start, end):
    hit_map.setdefault(canonical, [])
    if len(hit_map[canonical]) < 3:
        hit_map[canonical].append(
            {
                "token": token,
                "key": source_key,
                "snippet": _snippet_around(source_value, start, end),
            }
        )


def _normalize_scan_value(source_key, source_value):
    if source_key not in {"files", "path"}:
        return source_value

    normalized = source_value.replace("\\", "/")
    return normalized.rsplit("/", 1)[0] if "/" in normalized else ""


def _extract_year_hits_from_trusted_content(payload):
    strong_hits = {}
    weak_hits = {}

    for source_key, source_value in _collect_trusted_year_scan_strings(payload):
        scan_source_value = _normalize_scan_value(source_key, source_value)
        scan_value = _DATE_TAG_PATTERN.sub(" ", scan_source_value)

        for match in _YEAR_STRONG_UL_PATTERN.finditer(scan_value):
            token = match.group(1)
            canonical = _canonicalize_year_label(token)
            _record_year_hit(
                strong_hits, token, canonical, source_key, scan_source_value, match.start(1), match.end(1)
            )

        for match in _YEAR_STRONG_RUN_ERA_PATTERN.finditer(scan_value):
            canonical = _canonicalize_year_label(match.group(2))
            _record_year_hit(
                strong_hits,
                match.group(1),
                canonical,
                source_key,
                scan_source_value,
                match.start(1),
                match.end(1),
            )

        for match in _YEAR_STRONG_DELIMITED_PATTERN.finditer(scan_value):
            token = match.group(1)
            canonical = _canonicalize_year_label(token)
            _record_year_hit(
                strong_hits, token, canonical, source_key, scan_source_value, match.start(1), match.end(1)
            )

        for match in _YEAR_WEAK_PATTERN.finditer(scan_value):
            token = match.group(1)
            canonical = _canonicalize_year_label(token)
            _record_year_hit(
                weak_hits, token, canonical, source_key, scan_source_value, match.start(1), match.end(1)
            )

    return strong_hits, weak_hits


def _format_year_hit_examples(hit_map, canonical_hits):
    examples = []
    for canonical in canonical_hits:
        for hit in hit_map.get(canonical, []):
            examples.append(
                "{token} [{key}] \"{snippet}\"".format(
                    token=hit["token"], key=hit["key"], snippet=hit["snippet"]
                )
            )
            if len(examples) >= 6:
                return examples
    return examples


def _debug_year_scan_selfcheck():
    cases = [
        ("basename_ignored", "/NAOD/sample/2022/subset/output_2023.root"),
        ("valid_2023_path", "/NAOD/sample/2023/subset/something.root"),
    ]

    for case_name, file_path in cases:
        payload = {"files": [file_path]}
        strong_hits, weak_hits = _extract_year_hits_from_trusted_content(payload)
        strong_keys = sorted(_collapse_year_families(set(strong_hits.keys())))
        weak_keys = sorted(_collapse_year_families(set(weak_hits.keys())))
        print(f"[DEBUG_YEAR_SCAN] {case_name}")
        print(f"  files[0]: {file_path}")
        print(f"  strong keys: {strong_keys}")
        print(f"  weak keys: {weak_keys}")

    case_a_strong, case_a_weak = _extract_year_hits_from_trusted_content(
        {"files": [cases[0][1]]}
    )
    case_a_collapsed_strong = _collapse_year_families(set(case_a_strong.keys()))
    if "2022" not in case_a_collapsed_strong:
        raise RuntimeError("debug year scan failed: 2022 directory did not produce a 2022 strong hit")
    if "2023" in case_a_strong or "2023" in case_a_weak:
        raise RuntimeError("debug year scan failed: basename contributed an unexpected 2023 hit")

    valid_strong, _ = _extract_year_hits_from_trusted_content({"files": [cases[1][1]]})
    if "2023" not in valid_strong:
        raise RuntimeError("debug year scan failed: 2023 path did not produce a 2023 strong hit")


def _validate_payload_schema(payload, json_path):
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"[ERROR] Invalid JSON payload in {json_path}: expected object, got {type(payload).__name__}."
        )

    for key in _REQUIRED_JSON_KEYS:
        if key not in payload:
            raise RuntimeError(
                f"[ERROR] Invalid JSON payload in {json_path}: missing required key '{key}'."
            )

    if not isinstance(payload["files"], list):
        raise RuntimeError(
            f"[ERROR] Invalid JSON payload in {json_path}: key 'files' must be list, got {type(payload['files']).__name__}."
        )

    if not isinstance(payload["year"], str):
        raise RuntimeError(
            f"[ERROR] Invalid JSON payload in {json_path}: key 'year' must be str, got {type(payload['year']).__name__}."
        )

    if not isinstance(payload["isData"], bool):
        raise RuntimeError(
            f"[ERROR] Invalid JSON payload in {json_path}: key 'isData' must be bool, got {type(payload['isData']).__name__}."
        )


def _validate_payload_year_tokens(payload, json_path):
    payload_year = str(payload["year"])
    canonical_payload_year = _canonicalize_year_label(payload_year)

    payload_without_year = _strip_year_keys(payload)
    strong_hits, weak_hits = _extract_year_hits_from_trusted_content(payload_without_year)
    collapsed_strong_hits = sorted(_collapse_year_families(set(strong_hits.keys())))

    if not collapsed_strong_hits:
        return None

    matching_tokens = sorted(
        {
            hit["token"]
            for canonical in collapsed_strong_hits
            for hit in strong_hits.get(canonical, [])
        }
    )
    examples = _format_year_hit_examples(strong_hits, collapsed_strong_hits)

    if len(collapsed_strong_hits) == 1 and collapsed_strong_hits[0] != canonical_payload_year:
        raise RuntimeError(
            (
                f"[ERROR] Year mismatch detected in {json_path}.\n"
                f"  payload year: {payload_year}\n"
                f"  canonical payload year: {canonical_payload_year}\n"
                f"  inferred canonical year set (strong): {collapsed_strong_hits[0]}\n"
                f"  detected year from internal JSON content: {collapsed_strong_hits[0]}\n"
                f"  matching tokens (strong): {', '.join(matching_tokens)}\n"
                f"  examples: {' | '.join(examples) if examples else 'n/a'}\n"
                "How to fix: ensure payload['year'] matches the year implied by internal metadata content."
            )
        )

    if len(collapsed_strong_hits) > 1:
        return {
            "path": os.path.abspath(json_path),
            "payload_year": payload_year,
            "canonical_payload_year": canonical_payload_year,
            "canonical_hits": collapsed_strong_hits,
            "tokens": matching_tokens,
            "examples": examples,
            "weak_canonical_hits": sorted(_collapse_year_families(set(weak_hits.keys()))),
        }

    return None


def _raise_if_missing_referenced_jsons(missing_referenced_jsons):
    if not missing_referenced_jsons:
        return

    seen = set()
    unique = []
    for cfg_file, missing in missing_referenced_jsons:
        key = (cfg_file, missing)
        if key in seen:
            continue
        seen.add(key)
        unique.append((cfg_file, missing))

    msg_lines = ["[ERROR] Missing referenced JSON file(s) while parsing cfg inputs:"]
    for cfg_file, missing in unique:
        msg_lines.append(f"  - {missing} (referenced from cfg: {cfg_file})")
    raise SystemExit("\n".join(msg_lines))


def _find_duplicate_input_files(samplesdict):
    file_to_samples = {}
    for sample_name, sample in samplesdict.items():
        redirector = sample.get("redirector", "")
        for file_path in sample.get("files", []):
            key = f"{redirector}{file_path}"
            file_to_samples.setdefault(key, set()).add(sample_name)

    return {
        file_path: sorted(sample_names)
        for file_path, sample_names in file_to_samples.items()
        if len(sample_names) > 1
    }


def _warn_duplicate_input_files(samplesdict, max_examples=10):
    duplicates = _find_duplicate_input_files(samplesdict)
    if not duplicates:
        return

    duplicate_items = sorted(duplicates.items())
    print(
        "[WARNING] Found {} input file path(s) reused across multiple samples "
        "(comparison uses redirector+file).".format(len(duplicate_items))
    )
    for file_path, sample_names in duplicate_items[:max_examples]:
        print(f"  - {file_path}")
        print(f"    samples: {', '.join(sample_names)}")

    if len(duplicate_items) > max_examples:
        print(f"  ... and {len(duplicate_items) - max_examples} more duplicated file path(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="You can customize your run")
    parser.add_argument(
        "jsonFiles",
        nargs="?",
        default="",
        help="Json file(s) containing files and metadata",
    )
    parser.add_argument(
        "--executor",
        "-x",
        default="work_queue",
        help="Which executor to use",
    )
    parser.add_argument(
        "--prefix",
        "-r",
        nargs="?",
        default="",
        help="Prefix or redirector to look for the files",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="To perform a test, run over a few events in a couple of chunks",
    )
    parser.add_argument(
        "--pretend",
        action="store_true",
        help="Read json files but do not execute the analysis",
    )
    parser.add_argument(
        "--nworkers",
        "-n",
        default=8,
        type=int,
        help="Number of workers",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Alias for --nworkers (kept for backwards compatibility)",
    )
    parser.add_argument(
        "--chunksize",
        "-s",
        default=100000,
        help="Number of events per chunk",
    )
    parser.add_argument(
        "--nchunks",
        "-c",
        default=None,
        help="You can choose to run only a number of chunks",
    )
    parser.add_argument(
        "--outname",
        "-o",
        default="plotsTopEFT",
        help="Name of the output file with histograms",
    )
    parser.add_argument(
        "--outpath",
        "-p",
        default="histos",
        help="Name of the output directory",
    )
    parser.add_argument(
        "--years",
        "-y",
        nargs="+",
        help="Limit processing to the specified data-taking years",
    )
    parser.add_argument(
        "--treename",
        default="Events",
        help="Name of the tree inside the files",
    )
    parser.add_argument(
        "--no-sumw2",
        action="store_true",
        help="Skip filling sum of weight-squared histograms",
    )
    parser.add_argument(
        "--do-systs",
        action="store_true",
        help="Compute systematic variations",
    )
    parser.add_argument(
        "--no-suppress-forward-eta-stochastic-jer",
        dest="suppress_forward_eta_stochastic_jer",
        action="store_false",
        default=True,
        help=(
            "Disable the default forward-eta stochastic JER mitigation in "
            "2.5 < abs(eta) < 3.0."
        ),
    )
    parser.add_argument(
        "--fwd-eta-band-pt-apply",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Control the forward-jet eta-band pT tightening. auto applies it for "
            "Run 3 only, on applies it for all years, and off disables it for all years."
        ),
    )
    parser.add_argument(
        "--split-lep-flavor",
        action="store_true",
        help="Split up categories by lepton flavor",
    )
    parser.add_argument(
        "--offZ-3l-split",
        dest="offZ_3l_split",
        action="store_true",
        help="Split up 3l offZ categories",
    )
    parser.add_argument(
        "--tau-h-analysis",
        dest="tau_h_analysis",
        action="store_true",
        help=(
            "Add hadronic tau channels, including the DY-like 1l+tau_h control region "
            "with opposite-sign pairs around the visible Z mass."
        ),
    )
    parser.add_argument(
        "--fwd-analysis",
        action="store_true",
        help="Add fwd channels",
    )
    parser.add_argument(
        "--all-analysis",
        action="store_true",
        help="Add all contributions",
    )
    parser.add_argument(
        "--skip-sr",
        action="store_true",
        help="Skip all signal region categories",
    )
    parser.add_argument(
        "--skip-cr",
        action="store_true",
        help="Skip all control region categories",
    )
    parser.add_argument(
        "--do-np",
        action="store_true",
        help=(
            "Perform nonprompt estimation on the output hist, and save a new hist "
            "with the np contribution included. Signal, background and data samples "
            "must all be processed together."
        ),
    )
    parser.add_argument(
        "--np-postprocess",
        choices=["inline", "defer", "skip"],
        default="inline",
        help=(
            "Control when the nonprompt post-processing step runs. "
            "Use 'inline' (default) to run immediately, 'defer' to print a direct "
            "follow-up command, or 'skip' to omit the step entirely."
        ),
    )
    parser.add_argument(
        "--do-renormfact-envelope",
        action="store_true",
        help=(
            "Deprecated unsupported option. It exits before any analysis work."
        ),
    )
    parser.add_argument(
        "--wc-list",
        action="extend",
        nargs="+",
        help="Specify a list of Wilson coefficients to use in filling histograms.",
    )
    parser.add_argument(
        "--hist-list",
        action="extend",
        nargs="+",
        help="Specify a list of histograms to fill."
    )
    parser.add_argument(
        "--category-groups",
        nargs="+",
        help=(
            "Filter the resolved ch_lst.json category block(s) to one or more named groups. "
            "Names are validated in run_analysis.py against the active SR/CR block selection before downstream processing starts. "
            "Accepts multiple group names and preserves user order after deduplication. "
            "When both SR and CR are active, a requested group may match only one region and leave the other region empty. "
            "When omitted, all groups in each resolved block are used."
        ),
    )
    parser.add_argument(
        "--ttgamma-sample-role-policy",
        choices=list(SUPPORTED_SAMPLE_ROLE_POLICIES),
        default=SPLIT_SAMPLE_ROLE_POLICY,
        help=(
            "Select the ttgamma conversion-overlap sample-role policy. "
            "Use 'split' for nominal Run 2 production/decay role splitting, "
            "or 'run2_nlo_inclusive' for the diagnostic Run 2 TTGJets-inclusive mode."
        ),
    )
    parser.add_argument(
        "--ecut",
        default=None,
        help="Energy cut threshold i.e. throw out events above this (GeV)",
    )
    parser.add_argument(
        "--port",
        default="9164-9170",
        help="Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--wq-filepath",
        #default="/tmp/${USER}-workers",
        default=None,
        help=(
            "Override the Work Queue staging directory (default: /tmp/${USER}-workers). The path will be "
            "created if missing; if creation fails a system temporary directory will be used instead."
        ),
    )
    parser.add_argument(
        "--noRun3MVA",
        action='store_false',
        default=True,
        help='Do not use the Run3 MVA for lepton selection. Default is to use it.',
    )
    parser.add_argument(
        "--options",
        default=None,
        help=(
            "YAML file that specifies supported options. Recognized YAML values "
            "replace overlapping command-line and parser-default values."
        ),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["standard", "taufitter"],
        default="standard",
        help=(
            "Select the analysis workflow configuration. Use 'standard' for the default behaviour "
            "or 'taufitter' to enable tau fitter specific handling."
        ),
    )
    parser.add_argument(
        "--skip-topcoffea-data-check",
        action="store_true",
        help=(
            "Bypass the startup sanity check that verifies the shared topcoffea data files are present. "
            "Use only for expert/custom setups."
        ),
    )
    parser.add_argument(
        "--sample-universe-wrapper",
        default="run_analysis.py",
        help=(
            "Portable identity of the maintained wrapper that selected the active "
            "sample cfgs (recorded in generated sidecar provenance)."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help=(
            "Path to a prebuilt poncho environment tarball to ship to workers instead of generating one. "
            "Start from the repository's environment.yml template when crafting a fallback to avoid repeated "
            "failures from unavailable upstream pins."
        ),
    )
    parser.add_argument(
        "--rebuild-env",
        action="store_true",
        help="Force recreation of the current automatic remote environment archive.",
    )
    parser.add_argument(
        "--prepare-env-only",
        action="store_true",
        help="Create and validate the current environment archive, then exit before analysis setup.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use an explicit historical archive after integrity validation, bypassing current compatibility only.",
    )
    parser.add_argument(
        "--validate-env-file",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--env-integrity-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-remote-env",
        dest="use_remote_env",
        action="store_false",
        help=(
            "Disable automatic poncho environment creation; rely on worker nodes to already provide the "
            "dependencies. Pair with --env-file to supply a known-good tarball (e.g. built from "
            "environment.yml) when remote packaging fails."
        ),
    )
    parser.add_argument(
        "--debug-year-scan",
        action="store_true",
        help="Run a lightweight self-check for year token extraction and exit.",
    )
    parser.set_defaults(use_remote_env=True)

    args = parser.parse_args()
    if args.rebuild_env and args.env_file:
        parser.error("--rebuild-env cannot be combined with --env-file")
    if args.snapshot and not args.env_file:
        parser.error("--snapshot requires --env-file")
    if args.snapshot and args.rebuild_env:
        parser.error("--snapshot cannot be combined with --rebuild-env")
    if args.snapshot and args.prepare_env_only:
        parser.error("--snapshot cannot be combined with --prepare-env-only")
    if args.validate_env_file and not args.env_file:
        parser.error("--validate-env-file requires --env-file")
    if args.validate_env_file and args.snapshot:
        parser.error("--validate-env-file cannot be combined with --snapshot")
    if args.env_integrity_only and not args.validate_env_file:
        parser.error("--env-integrity-only requires --validate-env-file")

    env_extra_pip_local = {"topeft": ["topeft", "setup.py"]}
    if args.prepare_env_only:
        environment_file = _resolve_environment_file(
            None,
            True,
            extra_pip_local=env_extra_pip_local,
            rebuild_env=args.rebuild_env,
        )
        request = remote_environment.resolve_environment_request(
            extra_pip_local=env_extra_pip_local,
            unstaged="fail",
        )
        validation = remote_environment.validate_environment_archive(environment_file, request)
        if not validation["usable"]:
            _strict_environment_error(validation)
        _print_environment_identity(validation)
        raise SystemExit(0)
    if args.validate_env_file:
        _resolve_environment_file(
            args.env_file,
            True,
            extra_pip_local=env_extra_pip_local,
            integrity_only=args.env_integrity_only,
        )
        raise SystemExit(0)
    resolved_explicit_env_file = None
    if args.env_file:
        resolved_explicit_env_file = _resolve_environment_file(
            args.env_file,
            True,
            extra_pip_local=env_extra_pip_local,
            snapshot=args.snapshot,
        )
    if args.do_renormfact_envelope:
        raise_unsupported_renormfact_envelope()
    if args.debug_year_scan:
        _debug_year_scan_selfcheck()
        raise SystemExit(0)
    if args.workers is not None:
        args.nworkers = args.workers
    jsonFiles = args.jsonFiles
    prefix = args.prefix
    executor_name = args.executor
    dotest = args.test
    nworkers = int(args.nworkers)
    chunksize = int(args.chunksize)
    nchunks = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname = args.outname
    outpath = args.outpath
    pretend = args.pretend
    treename = args.treename
    fill_sumw2 = not args.no_sumw2
    legacy_no_sumw2_present = bool(args.no_sumw2)
    legacy_no_sumw2_value = bool(args.no_sumw2)
    sumw2_storage_present = False
    sumw2_storage_config = None
    data_driven_products_present = False
    data_driven_products_config = None
    do_systs = args.do_systs
    suppress_forward_eta_stochastic_jer = args.suppress_forward_eta_stochastic_jer
    fwd_eta_band_pt_apply = args.fwd_eta_band_pt_apply
    split_lep_flavor = args.split_lep_flavor
    offZ_split = args.offZ_3l_split
    tau_h_analysis = args.tau_h_analysis
    fwd_analysis = args.fwd_analysis
    all_analysis = args.all_analysis
    skip_sr    = args.skip_sr
    skip_cr    = args.skip_cr
    do_np      = args.do_np
    np_postprocess_mode = args.np_postprocess
    useRun3MVA = args.noRun3MVA  # NB: default value is True, the arg starts with no because its usage prevents making selections with the run3 MVA
    do_renormfact_envelope = args.do_renormfact_envelope
    wc_lst = args.wc_list if args.wc_list is not None else []
    ecut = args.ecut
    port = args.port
    wq_filepath = args.wq_filepath
    hist_list = args.hist_list
    category_groups = args.category_groups
    ttgamma_sample_role_policy = args.ttgamma_sample_role_policy
    analysis_mode = args.analysis_mode
    env_file_override = args.env_file
    use_remote_env = args.use_remote_env
    skip_topcoffea_data_check = args.skip_topcoffea_data_check
    sample_universe_wrapper = args.sample_universe_wrapper

    if args.options:
        import yaml
        with open(args.options, 'r') as f:
            ops = yaml.load(f, Loader=yaml.Loader)
        jsonFiles = ops.pop("jsonFiles", jsonFiles)
        prefix = ops.pop("prefix", prefix)
        executor_name = ops.pop("executor", executor_name)
        dotest = ops.pop("test", dotest)
        nworkers = ops.pop("nworkers", nworkers)
        chunksize = ops.pop("chunksize", chunksize)
        nchunks = ops.pop("nchunks", nchunks)
        outname = ops.pop("outname", outname)
        outpath = ops.pop("outpath", outpath)
        pretend = ops.pop("pretend", pretend)
        treename = ops.pop("treename", treename)
        sumw2_storage_present = "sumw2_storage" in ops
        sumw2_storage_config = ops.pop("sumw2_storage", None)
        data_driven_products_present = "data_driven_products" in ops
        data_driven_products_config = ops.pop("data_driven_products", None)
        yaml_no_sumw2_present = "no_sumw2" in ops
        yaml_do_errors_present = "do_errors" in ops
        if yaml_no_sumw2_present and yaml_do_errors_present:
            raise ValueError(
                "Specify at most one legacy statistical flag: no_sumw2 or do_errors."
            )
        if args.no_sumw2 and (yaml_no_sumw2_present or yaml_do_errors_present):
            raise ValueError(
                "The command line and options file both explicitly set a legacy "
                "statistical flag."
            )
        if yaml_no_sumw2_present:
            yaml_no_sumw2_value = ops.pop("no_sumw2")
            if not isinstance(yaml_no_sumw2_value, bool):
                raise ValueError("Legacy no_sumw2 must be a boolean.")
            legacy_no_sumw2_present = True
            legacy_no_sumw2_value = yaml_no_sumw2_value
        elif yaml_do_errors_present:
            legacy_do_errors = ops.pop("do_errors")
            if not isinstance(legacy_do_errors, bool):
                raise ValueError("Legacy do_errors must be a boolean.")
            legacy_no_sumw2_present = True
            legacy_no_sumw2_value = not legacy_do_errors
        fill_sumw2 = not legacy_no_sumw2_value
        do_systs = ops.pop("do_systs", do_systs)
        suppress_forward_eta_stochastic_jer = ops.pop(
            "suppress_forward_eta_stochastic_jer",
            suppress_forward_eta_stochastic_jer,
        )
        fwd_eta_band_pt_apply = ops.pop("fwd_eta_band_pt_apply", fwd_eta_band_pt_apply)
        split_lep_flavor = ops.pop("split_lep_flavor", split_lep_flavor)
        offZ_split = ops.pop("offZ_split", offZ_split)
        tau_h_analysis = ops.pop("tau_h_analysis", tau_h_analysis)
        fwd_analysis = ops.pop("fwd_analysis", fwd_analysis)
        all_analysis = ops.pop("all_analysis", all_analysis)
        skip_sr = ops.pop("skip_sr", skip_sr)
        skip_cr = ops.pop("skip_cr", skip_cr)
        do_np = ops.pop("do_np", do_np)
        np_postprocess_mode = ops.pop("np_postprocess", np_postprocess_mode)
        do_renormfact_envelope = ops.pop("do_renormfact_envelope", do_renormfact_envelope)
        wc_lst = ops.pop("wc_list", wc_lst)
        hist_list = ops.pop("hist_list", hist_list)
        category_groups = ops.pop("category_groups", category_groups)
        ttgamma_sample_role_policy = ops.pop(
            "ttgamma_sample_role_policy",
            ttgamma_sample_role_policy,
        )
        port = ops.pop("port", port)
        wq_filepath = ops.pop("wq_filepath", wq_filepath)
        ecut = ops.pop("ecut", ecut)
        analysis_mode = ops.pop("analysis_mode", analysis_mode)
        env_file_override = ops.pop("env_file", env_file_override)
        use_remote_env = ops.pop("use_remote_env", use_remote_env)
        skip_topcoffea_data_check = ops.pop("skip_topcoffea_data_check", skip_topcoffea_data_check)
        if ops:
            unsupported_option_keys = ", ".join(sorted(str(key) for key in ops))
            raise ValueError(
                f"Unsupported YAML option key(s): {unsupported_option_keys}"
            )

    if do_renormfact_envelope:
        raise_unsupported_renormfact_envelope()

    try:
        validated_mode_flags = analysis_processor.validate_analysis_mode_flags(
            offZ_split,
            tau_h_analysis,
            fwd_analysis,
            all_analysis,
        )
    except ValueError as exc:
        raise SystemExit(ANALYSIS_MODE_EXCLUSIVE_ERROR) from exc

    offZ_split = validated_mode_flags["offz_3l_split"]
    tau_h_analysis = validated_mode_flags["tau_h_analysis"]
    fwd_analysis = validated_mode_flags["fwd_analysis"]
    all_analysis = validated_mode_flags["all_analysis"]
    try:
        ttgamma_sample_role_policy = get_ttgamma_sample_role_policy(
            ttgamma_sample_role_policy
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _ensure_topcoffea_data_available(skip_topcoffea_data_check)

    out_pkl_file = os.path.join(outpath, outname + ".pkl.gz")
    out_pkl_file_name_np = os.path.join(outpath, outname + "_np.pkl.gz")

    # Check if we have valid options
    if executor_name not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(
            f'The "{executor_name}" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.'
        )
    if dotest:
        if executor_name == "futures":
            nchunks = 2
            chunksize = 100
            nworkers = 1
            print(
                "Running a fast test with %i workers, %i chunks of %i events"
                % (nworkers, nchunks, chunksize)
            )
        else:
            raise Exception(
                f'The "test" option is not set up to work with the {executor_name} executor. Exiting.'
            )

    # Set the threshold for the ecut (if not applying a cut, should be None)
    ecut_threshold = ecut
    if ecut_threshold is not None:
        ecut_threshold = float(ecut)

    category_group_selection = _resolve_category_group_selection(
        category_groups=category_groups,
        offz_3l_split=offZ_split,
        tau_h_analysis=tau_h_analysis,
        fwd_analysis=fwd_analysis,
        all_analysis=all_analysis,
        skip_sr=skip_sr,
        skip_cr=skip_cr,
    )
    _log_category_group_selection(category_group_selection)

    if executor_name in ["work_queue", "taskvine"]:
        # construct wq port range
        port = list(map(int, port.split("-")))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert singale values into a range of one element
            port.append(port[0])

    port = port[0]
    # Figure out which hists to include
    if hist_list == ["ana"]:
        # Here we hardcode a list of hists used for the analysis
        hist_lst = ["njets", "lj0pt", "ptz"]
        if tau_h_analysis or all_analysis:
            hist_lst.append("ptz_wtau")
            # hist_lst.append("tau0Tpt")
            # hist_lst.append("tau0Fpt")
        if fwd_analysis or all_analysis:
            hist_lst.append("lt")
        # if "lepton_pt_vs_eta" not in hist_lst:
        #     hist_lst.append("lepton_pt_vs_eta")
        # if "l0_SeedEtaOrX_vs_SeedPhiOrY" not in hist_lst:
        #     hist_lst.append("l0_SeedEtaOrX_vs_SeedPhiOrY")
        # if "l0_eta_vs_phi" not in hist_lst:
        #     hist_lst.append("l0_eta_vs_phi")
        # if "l1_SeedEtaOrX_vs_SeedPhiOrY" not in hist_lst:
        #     hist_lst.append("l1_SeedEtaOrX_vs_SeedPhiOrY")
        # if "l1_eta_vs_phi" not in hist_lst:
        #     hist_lst.append("l1_eta_vs_phi")
        # if fill_sumw2 and "lepton_pt_vs_eta_sumw2" not in hist_lst:
        #     hist_lst.append("lepton_pt_vs_eta_sumw2")
        # if fill_sumw2 and "l0_SeedEtaOrX_vs_SeedPhiOrY_sumw2" not in hist_lst:
        #     hist_lst.append("l0_SeedEtaOrX_vs_SeedPhiOrY_sumw2")
        # if fill_sumw2 and "l0_eta_vs_phi_sumw2" not in hist_lst:
        #     hist_lst.append("l0_eta_vs_phi_sumw2")
        # if fill_sumw2 and "l1_SeedEtaOrX_vs_SeedPhiOrY_sumw2" not in hist_lst:
        #     hist_lst.append("l1_SeedEtaOrX_vs_SeedPhiOrY_sumw2")
        # if fill_sumw2 and "l1_eta_vs_phi_sumw2" not in hist_lst:
        #     hist_lst.append("l1_eta_vs_phi_sumw2")
    elif hist_list == ["cr"]:
        # Here we hardcode a list of hists used for the CRs
        hist_lst = [
            # "lj0pt",
            "ptz",
            "met",
            "lt",
            # # "ljptsum",
            # # "l0pt",
            # # "l0ptcorr",
            # "l0conept",
            # "l0eta",
            # # "l1pt",
            # # "l1ptcorr",
            # "l1conept",
            # "l1eta",
            # "j0pt",
            # "j0eta",
            # # "j1eta",
            # "fwd0eta",
            # "fwd0pt",
            # "njets",
            # "nbtagsl",
            # "nbtagsm",
            # "invmass",
            # # "npvs",
            # "npvsGood",
            # "l0_gen_pdgId",
            # "l1_gen_pdgId",
            # "l2_gen_pdgId",
            # "l0_genParent_pdgId",
            # "l1_genParent_pdgId",
            # "l2_genParent_pdgId",
            # "b0l_hFlav",
            # "b0m_hFlav",
            # "b0l_pFlav",
            # "b0m_pFlav",
            # "b1l_hFlav",
            # "b1m_hFlav",
            # "b1l_pFlav",
            # "b1m_pFlav",
            # "b0l_genhFlav",
            # "b0m_genhFlav",
            # "b0l_genpFlav",
            # "b0m_genpFlav",
            # "b1l_genhFlav",
            # "b1m_genhFlav",
            # "b1l_genpFlav",
            # "b1m_genpFlav",
            # "lepton_pt_vs_eta",
            # "l0_SeedEtaOrX_vs_SeedPhiOrY",
            # "l0_eta_vs_phi",
            # "l1_SeedEtaOrX_vs_SeedPhiOrY",
            # "l1_eta_vs_phi",
        ]
        if tau_h_analysis or all_analysis:
            # hist_lst.append("tau0Tpt")
            # hist_lst.append("tau0Fpt")
            hist_lst.append("ptz_wtau")
    else:
        # We want to specify a custom list
        # If we don't specify this argument, it will be None, and the processor will fill all hists
        hist_lst = hist_list

    print(
        "Resolved histogram list: {}".format(
            ", ".join(hist_lst) if hist_lst is not None else "all registered families"
        )
    )
    print(
        "Resolved ttgamma sample-role policy: {}".format(
            ttgamma_sample_role_policy
        )
    )

    runtime_histogram_families_for_preflight = (
        list(axes_info.keys()) + list(axes_info_2d.keys())
        if hist_lst is None
        else list(hist_lst)
    )
    runtime_histogram_families_for_preflight = [
        family[:-6] if family.endswith("_sumw2") else family
        for family in runtime_histogram_families_for_preflight
    ]
    runtime_histogram_families_for_preflight = list(
        dict.fromkeys(runtime_histogram_families_for_preflight)
    )
    requested_products_for_preflight = (
        _requested_products_for_histogram_preflight(
            data_driven_products_config,
            data_driven_products_present=data_driven_products_present,
            legacy_do_np=do_np,
        )
    )
    try:
        validate_histogram_category_compatibility(
            runtime_histogram_families_for_preflight,
            selected_category_dicts=(
                category_group_selection["sr_category_dict"],
                category_group_selection["cr_category_dict"],
            ),
            histogram_selection_explicit=hist_list is not None,
            requested_data_driven_products=requested_products_for_preflight,
        )
    except histogram_category_compatibility_error as error:
        raise SystemExit(str(error)) from None

    ### Load samples from json
    samplesdict = {}
    sample_sources = {}
    sample_payload_signatures = {}
    allInputFiles = []

    # NEW: keep track of missing JSONs referenced inside cfg files
    missing_referenced_jsons = []
    year_scan_warnings = []

    def _resolve_cfg_token_as_file(cfg_file, token):
        """
        Resolve token as a file path:
          1) as given (with ~ and env expansion)
          2) if relative, also try relative to cfg_file's directory
        Return the existing file path, or None if not found.
        """
        expanded = os.path.expandvars(os.path.expanduser(token))
        candidates = [expanded]
        if not os.path.isabs(expanded):
            candidates.append(os.path.join(os.path.dirname(cfg_file), expanded))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _record_missing_json(cfg_file, json_token):
        missing_referenced_jsons.append((cfg_file, json_token))

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = (
            jsonFile if not "/" in jsonFile else jsonFile[jsonFile.rfind("/") + 1 :]
        )
        if sampleName.endswith(".json"):
            sampleName = sampleName[:-5]

        source_json_path = os.path.abspath(jsonFile)
        with open(jsonFile, encoding="utf-8") as jf:
            payload = json.load(jf)
        _validate_payload_schema(payload, source_json_path)
        analysis_processor.resolve_eft_treatment(
            payload,
            sample_name=sampleName,
        )
        year_scan_warning = _validate_payload_year_tokens(payload, source_json_path)
        if year_scan_warning is not None:
            year_scan_warnings.append(year_scan_warning)
        payload_signature = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        hist_axis_name = payload.get("histAxisName")

        if sampleName in samplesdict:
            prev_payload = samplesdict[sampleName]
            prev_signature = sample_payload_signatures[sampleName]
            prev_source = sample_sources[sampleName]
            prev_hist_axis_name = prev_payload.get("histAxisName")
            prev_redirector = prev_payload.get("redirector", "")

            if prev_signature != payload_signature or prev_hist_axis_name != hist_axis_name:
                raise RuntimeError(
                    (
                        f'Colliding sample basename key "{sampleName}" while loading JSONs.\n'
                        f"  Existing json path: {prev_source}\n"
                        f"  New json path:      {source_json_path}\n"
                        f"  Existing histAxisName: {prev_hist_axis_name}\n"
                        f"  New histAxisName:      {hist_axis_name}\n"
                        "Refusing to continue because payloads are not identical."
                    )
                )

            if prev_redirector != prefix:
                raise RuntimeError(
                    (
                        f'Colliding sample basename key "{sampleName}" with conflicting redirector.\n'
                        f"  Existing json path: {prev_source}\n"
                        f"  New json path:      {source_json_path}\n"
                        f'  Existing redirector: "{prev_redirector}"\n'
                        f'  New redirector:      "{prefix}"\n'
                        "Refusing to continue because duplicate entries disagree."
                    )
                )

            # Duplicate identical entry: keep the first deterministic instance.
            return

        samplesdict[sampleName] = payload
        samplesdict[sampleName]["redirector"] = prefix
        sample_sources[sampleName] = source_json_path
        sample_payload_signatures[sampleName] = payload_signature

    if isinstance(jsonFiles, str) and "," in jsonFiles:
        jsonFiles = jsonFiles.replace(" ", "").split(",")
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith("/"):
                jsonFile += "/"
            # FIX: os.path.listdir -> os.listdir
            for f in os.listdir(jsonFile):
                if f.endswith(".json"):
                    allInputFiles.append(jsonFile + f)
        else:
            allInputFiles.append(jsonFile)

    resolved_input_jsons = list(allInputFiles)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f"[ERROR] Input file {f} not found!")
        # This input file is a json file, not a cfg
        if f.endswith(".json"):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(" >> Reading json from cfg file...", f)
                lines = fin.readlines()
                for l in lines:
                    if "#" in l:
                        l = l[: l.find("#")]
                    l = l.replace(" ", "").replace("\n", "")
                    if l == "":
                        continue

                    tokens = l.split(",") if "," in l else [l]
                    for token in tokens:
                        if token == "":
                            continue

                        resolved = _resolve_cfg_token_as_file(f, token)
                        if resolved is None:
                            # If it looks like a json, it must exist; do not silently treat it as a prefix.
                            if token.endswith(".json"):
                                _record_missing_json(f, token)
                            else:
                                prefix = token
                            continue

                        LoadJsonToSampleName(resolved, prefix)

    _raise_if_missing_referenced_jsons(missing_referenced_jsons)

    if year_scan_warnings:
        print(
            "[WARNING] Found ambiguous year-token matches in {} JSON payload(s); "
            "continuing because detected year is not unique.".format(
                len(year_scan_warnings)
            )
        )
        for warning in year_scan_warnings[:10]:
            print(f"  - path: {warning['path']}")
            print(
                "    payload year: {} (canonical: {})".format(
                    warning["payload_year"], warning["canonical_payload_year"]
                )
            )
            print(
                "    canonical hits: {} | matching tokens: {}".format(
                    ", ".join(warning["canonical_hits"]),
                    ", ".join(warning["tokens"]),
                )
            )
            if warning.get("examples"):
                print("    examples: {}".format(" | ".join(warning["examples"])))
            if warning.get("weak_canonical_hits"):
                print(
                    "    weak canonical hits (diagnostic only): {}".format(
                        ", ".join(warning["weak_canonical_hits"])
                    )
                )
        if len(year_scan_warnings) > 10:
            print(
                f"  ... and {len(year_scan_warnings) - 10} more ambiguous year-token match(es)."
            )

    requested_years = None
    if args.years:
        valid_year_choices = {
            "UL16",
            "UL16APV",
            "UL17",
            "UL18",
            "2016",
            "2016APV",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
            "run2",
            "run3",
        }
        year_synonyms = {
            "2016": {"2016", "UL16"},
            "UL16": {"2016", "UL16"},
            "2016APV": {"2016APV", "UL16APV"},
            "UL16APV": {"2016APV", "UL16APV"},
            "2017": {"2017", "UL17"},
            "UL17": {"2017", "UL17"},
            "2018": {"2018", "UL18"},
            "UL18": {"2018", "UL18"},
            "run2": {
                "2016",
                "2016APV",
                "2017",
                "2018",
                "UL16",
                "UL16APV",
                "UL17",
                "UL18",
            },
            "run3": {"2022", "2022EE", "2023", "2023BPix"},
        }

        requested_years = set()
        for year in args.years:
            year_str = str(year)
            if year_str not in valid_year_choices:
                raise ValueError(
                    "Invalid year selection \"{}\". Valid choices are: {}".format(
                        year_str, ", ".join(sorted(valid_year_choices))
                    )
                )

            requested_years.update(year_synonyms.get(year_str, {year_str}))

    print(">> Loaded a total of %i samples from json files." % len(samplesdict))

    if requested_years is not None:
        samplesdict = {
            name: sample
            for name, sample in samplesdict.items()
            if str(sample.get("year")) in requested_years
        }

        if not samplesdict:
            raise ValueError(
                "No samples remaining after applying the requested year filter."
            )

    _warn_duplicate_input_files(samplesdict)

    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():

        samplesdict[sname]["files"] = samplesdict[sname]["files"]  # [0:1]

        redirector = samplesdict[sname]["redirector"]
        flist[sname] = [(redirector + f) for f in samplesdict[sname]["files"]]
        samplesdict[sname]["year"] = samplesdict[sname]["year"]
        samplesdict[sname]["xsec"] = float(samplesdict[sname]["xsec"])
        samplesdict[sname]["nEvents"] = int(samplesdict[sname]["nEvents"])
        nevts_total += samplesdict[sname]["nEvents"]
        samplesdict[sname]["nGenEvents"] = int(samplesdict[sname]["nGenEvents"])
        samplesdict[sname]["nSumOfWeights"] = float(samplesdict[sname]["nSumOfWeights"])
        if not samplesdict[sname]["isData"]:
            for wgt_var in WGT_VAR_LST:
                # Check that MC samples have all needed weight sums (only needed if doing systs)
                if do_systs:
                    if wgt_var not in samplesdict[sname]:
                        raise Exception(f'Missing weight variation "{wgt_var}".')
                    else:
                        samplesdict[sname][wgt_var] = float(samplesdict[sname][wgt_var])
        # Print file info
        print(">> " + sname)
        print(
            "   - isData?      : %s" % ("YES" if samplesdict[sname]["isData"] else "NO")
        )
        print("   - year         : %s" % samplesdict[sname]["year"])
        # print("   - xsec         : %f" % samplesdict[sname]["xsec"])
        print("   - histAxisName : %s" % samplesdict[sname]["histAxisName"])
        # print("   - options      : %s" % samplesdict[sname]["options"])
        print("   - tree         : %s" % samplesdict[sname]["treeName"])
        print("   - nEvents      : %i" % samplesdict[sname]["nEvents"])
        print("   - nGenEvents   : %i" % samplesdict[sname]["nGenEvents"])
        # print("   - SumWeights   : %i" % samplesdict[sname]["nSumOfWeights"])
        # if not samplesdict[sname]["isData"]:
        #     for wgt_var in WGT_VAR_LST:
        #         if wgt_var in samplesdict[sname]:
        #             print(f"   - {wgt_var}: {samplesdict[sname][wgt_var]}")
        # print("   - Prefix       : %s" % samplesdict[sname]["redirector"])
        # print("   - nFiles       : %i" % len(samplesdict[sname]["files"]))
        for fname in samplesdict[sname]["files"]:
            print("     %s" % fname)

        if executor_name == "futures":
            break

    sample_years_from_inputs = sorted(
        {
            str(sample.get("year"))
            for sample in samplesdict.values()
            if sample.get("year") is not None
        }
    )

    runtime_histogram_families = runtime_histogram_families_for_preflight

    try:
        active_universe = build_active_sample_universe(
            samplesdict,
            input_paths=resolved_input_jsons,
            wrapper_identity=sample_universe_wrapper,
        )
        sumw2_mode = resolve_sumw2_storage_mode(
            sumw2_storage_config,
            sumw2_storage_present=sumw2_storage_present,
            legacy_no_sumw2_present=legacy_no_sumw2_present,
            legacy_no_sumw2_value=legacy_no_sumw2_value,
        )
        validate_active_sample_profile(
            active_universe,
            sumw2_mode,
            data_driven_products=data_driven_products_config,
            data_driven_products_present=data_driven_products_present,
            metadata_path=args.options,
        )
    except production_sample_profile_error as error:
        raise SystemExit(str(error)) from None

    configured_nonprompt_enabled = (
        bool(
            data_driven_products_config.get("nonprompt", {}).get("enabled")
        )
        if data_driven_products_present
        and isinstance(data_driven_products_config, Mapping)
        and isinstance(data_driven_products_config.get("nonprompt"), Mapping)
        else bool(do_np)
    )
    certified_nonprompt_policy = _certify_nonprompt_policy_before_executor(
        samplesdict,
        nonprompt_enabled=configured_nonprompt_enabled,
        configuration_source=args.options or "<command-line/default>",
    )

    resolved_data_driven_products = resolve_data_driven_products(
        data_driven_products_config,
        data_driven_products_present=data_driven_products_present,
        legacy_do_np=do_np,
        samples=samplesdict,
        runtime_families=runtime_histogram_families,
        metadata_path=args.options,
        nonprompt_policy=certified_nonprompt_policy,
    )
    if do_np and not resolved_data_driven_products.enabled_products():
        raise ValueError(
            "do_np requests data-driven postprocessing, but data_driven_products "
            "has no enabled product. Enable nonprompt or flips, or disable do_np."
        )

    required_sumw2_targets = []
    if analysis_mode == "taufitter":
        taufitter_families = ("tau0Fpt", "tau0Tpt")
        missing_taufitter_families = sorted(
            set(taufitter_families) - set(runtime_histogram_families)
        )
        if missing_taufitter_families:
            raise ValueError(
                "The taufitter workflow requires runtime histogram families: "
                + ", ".join(missing_taufitter_families)
            )
        for dataset_key, sample in samplesdict.items():
            for family in taufitter_families:
                required_sumw2_targets.append(
                    sumw2_target(dataset_key, sample["histAxisName"], family)
                )

    sumw2_policy = resolve_sumw2_storage_policy(
        sumw2_storage_config,
        samples=samplesdict,
        runtime_families=runtime_histogram_families,
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        analysis_mode=analysis_mode,
        sumw2_storage_present=sumw2_storage_present,
        legacy_no_sumw2_present=legacy_no_sumw2_present,
        legacy_no_sumw2_value=legacy_no_sumw2_value,
        consumer_requirements=required_sumw2_targets,
        implicit_production_requirements=(
            resolved_data_driven_products.required_targets()
        ),
        mode_resolution=sumw2_mode,
    )
    try:
        production_sample_contract = certify_production_sample_contract(
            active_universe,
            sumw2_policy,
            resolved_data_driven_products,
        )
    except production_sample_profile_error as error:
        raise SystemExit(str(error)) from None
    (
        requested_data_driven_products,
        resolved_data_driven_contract,
    ) = certify_data_driven_preflight(
        resolved_data_driven_products,
        sumw2_policy,
    )
    fill_sumw2 = bool(sumw2_policy.selected_families())
    print(
        "Resolved sumw2 storage: mode={} source={} targets={} families={}".format(
            sumw2_policy.requested_mode,
            sumw2_policy.source,
            len(sumw2_policy.resolved_targets),
            ",".join(sumw2_policy.selected_families()) or "<none>",
        )
    )
    print(
        "Certified production sample profile: profile={} wrapper={} cfgs={}".format(
            sumw2_policy.signal_sample_profile,
            active_universe.wrapper_identity,
            ",".join(
                identity["basename"]
                for identity in active_universe.serialized_cfg_identities()
            ),
        )
    )
    print(
        "Resolved nominal container: version={} layout={}".format(
            NOMINAL_CONTAINER_SCHEMA_VERSION,
            NOMINAL_CONTAINER_LAYOUT,
        )
    )
    print(
        "Resolved data-driven products: source={} enabled={}".format(
            resolved_data_driven_products.source,
            ",".join(resolved_data_driven_products.enabled_products()) or "<none>",
        )
    )

    inline_artifact_kind = (
        "flips_output"
        if resolved_data_driven_products.enabled_products() == ("flips",)
        else "nonprompt_output"
    )

    def _build_np_followup_command():
        command = [
            "python",
            "analysis/topeft_run2/run_data_driven.py",
            "--input-pkl",
            out_pkl_file,
            "--output-pkl",
            out_pkl_file_name_np,
        ]
        if inline_artifact_kind == "flips_output":
            command.append("--only-flips")
        return shlex.join(command)

    def _print_np_defer_instructions():
        print(
            "Nonprompt estimation deferred. The processor sidecar is discovered "
            "automatically from the input PKL.\n"
            "Run the following command to finalize the nonprompt histograms:\n  {}".format(
                _build_np_followup_command()
            )
        )

    if pretend:
        print("pretending...")
        if do_np and np_postprocess_mode == "defer":
            _print_np_defer_instructions()
        exit()

    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]["WCnames"]:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        # Yes, why not have the output be in correct English?
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + " and " + wc_lst[1]
        else:
            wc_print = ", ".join(wc_lst[:-1]) + ", and " + wc_lst[-1]
            print("Wilson Coefficients: {}.".format(wc_print))
    else:
        print("No Wilson coefficients specified")

    if hist_lst is None:
        print("Variables to be histogrammed: all (processor defaults)")
    else:
        print("Variables to be histogrammed: {}".format(", ".join(hist_lst)))

    wq_staging_dir = None
    wq_cleanup_after = False
    if executor_name in ["work_queue", "taskvine"]:
        environment_file = resolved_explicit_env_file or _resolve_environment_file(
            env_file_override,
            use_remote_env,
            extra_pip_local=env_extra_pip_local,
            rebuild_env=args.rebuild_env,
            snapshot=args.snapshot,
        )
    else:
        environment_file = None

    processor_instance = analysis_processor.AnalysisProcessor(
        samplesdict,
        wc_lst,
        hist_lst,
        ecut_threshold,
        fill_sumw2,
        do_systs,
        split_lep_flavor,
        skip_sr,
        skip_cr,
        offZ_split=offZ_split,
        tau_h_analysis=tau_h_analysis,
        fwd_analysis=fwd_analysis,
        all_analysis=all_analysis,
        useRun3MVA=useRun3MVA,
        tau_run_mode=analysis_mode,
        sr_category_dict=category_group_selection["sr_category_dict"],
        cr_category_dict=category_group_selection["cr_category_dict"],
        suppress_forward_eta_stochastic_jer=suppress_forward_eta_stochastic_jer,
        fwd_eta_band_pt_apply=fwd_eta_band_pt_apply,
        ttgamma_sample_role_policy=ttgamma_sample_role_policy,
        sumw2_policy=sumw2_policy,
    )

    if executor_name in ["work_queue", "taskvine"]:
        wq_staging_dir, wq_cleanup_after = _prepare_work_queue_staging_directory(wq_filepath)
        executor_args = {
            "manager_name": f"{os.environ['USER']}-workqueue-{outname}",
            # find a port to run work queue in this range:
            "port": port,
            "debug_log": "debug.log",
            "transactions_log": "tr.log",
            "stats_log": "stats.log",
            "tasks_accum_log": "tasks.log",
            "extra_input_files": ["analysis_processor.py"],
            "retries": 15,
            # use mid-range compression for chunks results.
            # Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            "compression": 1,
            # automatically find an adequate resource allocation for tasks.
            # tasks are first tried using the maximum resources seen of previously ran
            # tasks. on resource exhaustion, they are retried with the maximum resource
            # values, if specified below. if a maximum is not specified, the task waits
            # forever until a larger worker connects.
            "resources_mode": "auto",
            "resource_monitor": "measure",
            "split_on_exhaustion": True,
            #'filepath': wq_staging_dir,
            "filepath": '/tmp',
            # this resource values may be omitted when using
            # resources_mode: 'auto', but they do make the initial portion
            # of a workflow run a little bit faster.
            # Rather than using whole workers in the exploratory mode of
            # resources_mode: auto, tasks are forever limited to a maximum
            # of 8GB of mem and disk.
            #
            # NOTE: The very first tasks in the exploratory
            # mode will use the values specified here, so workers need to be at least
            # this large. If left unspecified, tasks will use whole workers in the
            # exploratory mode.
            # 'cores': 1,
            # 'disk': 10000,   #MB
            # 'memory': 16000, #MB
            # control the size of accumulation tasks.
            # "treereduction": 10,
            # terminate workers on which tasks have been running longer than average.
            # This is useful for temporary conditions on worker nodes where a task will
            # be finish faster is ran in another worker.
            # the time limit is computed by multipliying the average runtime of tasks
            # by the value of 'fast_terminate_workers'.  Since some tasks can be
            # legitimately slow, no task can trigger the termination of workers twice.
            #
            # warning: small values (e.g. close to 1) may cause the workflow to misbehave,
            # as most tasks will be terminated.
            #
            # Less than 1 disables it.
            "fast_terminate_workers": 0,
            # print messages when tasks are submitted, finished, etc.,
            # together with their resource allocation and usage. If a task
            # fails, its standard output is also printed, so we can turn
            # off print_stdout for all tasks.
            "verbose": True,
            "print_stdout": False,
        }

        if environment_file:
            executor_args["environment_file"] = environment_file

    # Run the processor and get the output
    tstart = time.time()

    def _ensure_nonempty_chunks():
        total_files = sum(len(files) for files in flist.values())
        if total_files == 0:
            raise SystemExit(
                "No input files were available to process; verify the sample JSON and prefix "
                "and retry with at least one file."
            )

        if nchunks == 0:
            raise SystemExit(
                "Requested zero chunks; increase --nchunks or drop the flag to process the full dataset."
            )

    if executor_name == "futures":
        futures_factory = getattr(processor, "futures_executor", None)
        if callable(futures_factory):
            exec_instance = futures_factory(workers=nworkers)
        else:
            exec_instance = processor.FuturesExecutor(workers=nworkers)
        _ensure_nonempty_chunks()
        runner = processor.Runner(
            exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks
        )
    elif executor_name == "work_queue":
        executor_instance = processor.WorkQueueExecutor(**executor_args)
        _ensure_nonempty_chunks()
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=nchunks,
            skipbadfiles=False,
            xrootdtimeout=180,
        )
    elif executor_name == "taskvine":
        try:
            executor_instance = processor.TaskVineExecutor(**executor_args)
        except AttributeError:
            raise RuntimeError("TaskVineExecutor not available.")
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=chunksize,
            maxchunks=nchunks,
            skipbadfiles=True,
            xrootdtimeout=300,
        )

    run_succeeded = False
    try:
        try:
            output = runner(flist, treename, processor_instance)
        except TypeError as exc:
            raise RuntimeError(
                "The executor returned no chunk results. Ensure that the input files produced at least "
                "one chunk and that the executor handled submissions correctly."
            ) from exc

        worker_exception = None
        if isinstance(output, dict):
            worker_exception = _format_worker_exception(output.get("exception"))

        if output is None:
            if worker_exception is not None:
                print(f"Executor reported a worker-side exception: {worker_exception}")
            else:
                print("Runner returned no output; no chunks appear to have been processed.")
            raise RuntimeError("Processing failed because no results were returned from the executor.")

        if worker_exception is not None:
            raise RuntimeError(
                f"Processing failed because a worker raised an exception: {worker_exception}"
            )

        print("Finished running the processor...")

        validate_nominal_mapping(
            output,
            runtime_families=runtime_histogram_families,
            schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
            policy=sumw2_policy,
        )
        output = canonicalize_nominal_keys(
            output,
            runtime_families=runtime_histogram_families,
            schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
        )

        dt = time.time() - tstart

        if executor_name in ["work_queue", "taskvine"]:
            print(
                "Processed {} events in {} seconds ({:.2f} evts/sec).".format(
                    nevts_total, dt, nevts_total / dt
                )
            )

        if executor_name == "futures":
            print(
                "Processing time: %1.2f s with %i workers (%.2f s cpu overall)"
                % (
                    dt,
                    nworkers,
                    dt * nworkers,
                )
            )

        # Save the output
        os.makedirs(outpath, exist_ok=True)
        print(f"\nSaving output in {out_pkl_file}...")
        processor_sidecar = write_histogram_artifact(
            out_pkl_file,
            histograms=output,
            artifact_kind="processor_output",
            sumw2_storage_provenance=sumw2_policy.to_provenance(),
            production_sample_contract=production_sample_contract,
            requested_data_driven_products=requested_data_driven_products,
            resolved_data_driven_contract=resolved_data_driven_contract,
        )
        print("Done!")

        # Run the data driven estimation, save the output
        if do_np:
            if np_postprocess_mode == "inline":
                print("\nDoing the nonprompt estimation...")
                ddp = DataDrivenProducer(
                    out_pkl_file,
                    "",
                    artifact_kind=inline_artifact_kind,
                )
                data_driven_histograms = ddp.getDataDrivenHistogram()
                print(f"Saving output in {out_pkl_file_name_np}...")
                write_histogram_artifact(
                    out_pkl_file_name_np,
                    histograms=data_driven_histograms,
                    artifact_kind=inline_artifact_kind,
                    sumw2_storage_provenance=sumw2_policy.to_provenance(),
                    lineage_inputs=[lineage_input_from_sidecar(processor_sidecar)],
                    input_sidecar=processor_sidecar,
                    transformation_context=ddp.get_transformation_context(
                        inline_artifact_kind
                    ),
                )
                print("Done!")
            elif np_postprocess_mode == "defer":
                print("\nDeferring the nonprompt estimation...")
                _print_np_defer_instructions()
            else:
                print("\nSkipping the nonprompt estimation as requested (--np-postprocess=skip).")
            run_succeeded = True
        else:
            run_succeeded = True
    finally:
        if run_succeeded and wq_cleanup_after:
            _cleanup_work_queue_staging_directory(wq_staging_dir, wq_cleanup_after)
