from __future__ import annotations

import json
import os
import shlex
from typing import Any, Dict, Mapping


DEFERRED_NP_METADATA_VERSION = 2
RUN_DATA_DRIVEN_ENTRYPOINT = "analysis/topeft_run2/run_data_driven.py"

_REQUIRED_KEYS = frozenset(
    {
        "metadata_version",
        "input_histogram",
        "output_histogram",
        "metadata_path",
        "np_postprocess",
        "pretend_mode",
        "do_np",
        "apply_renormfact_envelope",
        "resolved_years",
        "sample_years",
        "followup_command",
    }
)


def build_np_followup_command(metadata_path: str) -> str:
    return (
        f"python {shlex.quote(RUN_DATA_DRIVEN_ENTRYPOINT)} "
        f"--metadata-json {shlex.quote(metadata_path)}"
    )


def build_deferred_np_metadata(
    *,
    input_histogram: str,
    output_histogram: str,
    metadata_path: str,
    np_postprocess: str,
    pretend_mode: bool,
    do_np: bool,
    apply_renormfact_envelope: bool,
    resolved_years: list[str],
    sample_years: list[str],
    input_jsons: list[str],
    analysis_mode: str | None,
    hist_list: list[str],
    wc_list: list[str],
    executor: str,
    options_file: str | None,
    flags: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata_version": DEFERRED_NP_METADATA_VERSION,
        "input_histogram": input_histogram,
        "output_histogram": output_histogram,
        "metadata_path": metadata_path,
        "np_postprocess": np_postprocess,
        "pretend_mode": pretend_mode,
        "do_np": do_np,
        "apply_renormfact_envelope": apply_renormfact_envelope,
        "resolved_years": list(resolved_years or []),
        "sample_years": list(sample_years or []),
        "input_jsons": list(input_jsons or []),
        "analysis_mode": analysis_mode,
        "hist_list": list(hist_list or []),
        "wc_list": list(wc_list or []),
        "executor": executor,
        "options_file": options_file,
        "flags": dict(flags),
        "followup_command": build_np_followup_command(metadata_path),
    }


def load_deferred_np_metadata(metadata_path: str) -> Dict[str, Any]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path) as metadata_stream:
        payload = json.load(metadata_stream)
    return validate_deferred_np_metadata(payload)


def validate_deferred_np_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Deferred nonprompt metadata must be a JSON object.")

    version = payload.get("metadata_version")
    if version != DEFERRED_NP_METADATA_VERSION:
        raise ValueError(
            "Unsupported metadata schema version "
            f"{version!r}. Expected version {DEFERRED_NP_METADATA_VERSION} metadata."
        )

    missing = sorted(key for key in _REQUIRED_KEYS if key not in payload)
    if missing:
        raise ValueError(
            "Deferred nonprompt metadata is missing required keys: "
            + ", ".join(missing)
        )

    if payload["np_postprocess"] != "defer":
        raise ValueError(
            "Deferred nonprompt metadata must record np_postprocess='defer'."
        )
    if payload["do_np"] is not True:
        raise ValueError(
            "Metadata indicates nonprompt estimation was disabled (do_np=False). Nothing to do."
        )
    if not isinstance(payload["apply_renormfact_envelope"], bool):
        raise ValueError(
            "Deferred nonprompt metadata key 'apply_renormfact_envelope' must be a boolean."
        )

    input_histogram = _require_nonempty_string(payload, "input_histogram")
    output_histogram = _require_nonempty_string(payload, "output_histogram")
    _require_nonempty_string(payload, "metadata_path")
    _require_nonempty_string(payload, "followup_command")

    if os.path.normpath(input_histogram) == os.path.normpath(output_histogram):
        raise ValueError(
            "Deferred nonprompt metadata must not point input_histogram and "
            "output_histogram to the same path."
        )

    resolved_years = _require_string_list(payload, "resolved_years")
    sample_years = _require_string_list(payload, "sample_years")
    missing_years = sorted(set(resolved_years) - set(sample_years))
    if missing_years:
        raise ValueError(
            "Metadata contains requested years that are not present in the samples: "
            f"{missing_years}"
        )

    flags = payload.get("flags")
    if flags is not None and not isinstance(flags, Mapping):
        raise ValueError("Deferred nonprompt metadata key 'flags' must be a JSON object.")

    return dict(payload)


def _require_nonempty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Deferred nonprompt metadata key '{key}' must be a non-empty string."
        )
    return value


def _require_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"Deferred nonprompt metadata key '{key}' must be a list of strings."
        )
    return list(value)
