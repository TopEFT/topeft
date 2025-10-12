"""Helper utilities for :mod:`run_analysis`.

This module centralizes the normalization and configuration helpers that were
previously defined inline in :mod:`run_analysis`.  The goal is to make the main
script easier to read by isolating the type coercion, configuration merging and
sample loading logic.  The helpers are intentionally lightweight so they can be
shared across future scripts or tests without pulling in the full execution
stack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

try:  # pragma: no cover - typing import for static analyzers only
    import argparse
except ImportError:  # pragma: no cover - argparse is part of stdlib
    argparse = None  # type: ignore


# Default weight variations expected in sample metadata for systematic studies.
DEFAULT_WEIGHT_VARIATIONS = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    "nSumOfWeights_renormfactUp",
    "nSumOfWeights_renormfactDown",
]


def normalize_sequence(value: Any) -> List[str]:
    """Flatten ``value`` into a list of strings."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            result.extend(normalize_sequence(item))
        return result
    return [str(value)]


def unique_preserving_order(values: Iterable[str]) -> List[str]:
    """Return a list containing only the first occurrence of every value."""

    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def coerce_bool(value: Any) -> Optional[bool]:
    """Convert ``value`` into a boolean if possible."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def coerce_int(value: Any, allow_none: bool = False) -> Optional[int]:
    """Convert ``value`` to an integer, optionally accepting ``None``."""

    if value is None:
        if allow_none:
            return None
        raise ValueError("Integer value required")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            if allow_none:
                return None
            raise ValueError("Integer value required")
        return int(stripped)
    return int(value)


def coerce_json_files(value: Any) -> List[str]:
    """Normalize JSON file inputs to a list of paths."""

    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            tokens = [token for token in value.replace(" ", "").split(",") if token]
            return tokens
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        result: List[str] = []
        for item in value:
            result.extend(coerce_json_files(item))
        return result
    return [str(value)]


def coerce_optional_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float or ``None`` if unset."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return float(stripped)
    return float(value)


def coerce_port(value: Any) -> str:
    """Return a Work Queue port specification as ``min-max``."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable):
        ints = [str(coerce_int(v, allow_none=False)) for v in value if v is not None]
        if not ints:
            return ""
        if len(ints) == 1:
            return ints[0]
        return f"{ints[0]}-{ints[1]}"
    return str(value)


class SampleLoader:
    """Helper responsible for expanding input specifications and loading samples."""

    def __init__(self, default_prefix: str = "", weight_variables: Sequence[str] = ()):
        self.default_prefix = default_prefix
        self.weight_variables = list(weight_variables)

    def collect(self, json_files: Sequence[str]) -> List[Tuple[Path, str]]:
        """Expand JSON/CFG inputs into explicit sample specifications."""

        expanded: List[Tuple[Path, str]] = []
        for json_file in json_files:
            input_path = Path(json_file).expanduser()
            if not input_path.is_absolute():
                input_path = (Path.cwd() / input_path).resolve(strict=True)
            else:
                input_path = input_path.resolve(strict=True)
            expanded.extend(self._expand_path(input_path, self.default_prefix))
        return expanded

    def load(self, sample_specs: Sequence[Tuple[Path, str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
        """Load JSON metadata and normalize numeric fields for each sample."""

        samplesdict: Dict[str, Dict[str, Any]] = {}
        flist: Dict[str, List[str]] = {}
        for json_path, prefix in sample_specs:
            prefix = prefix or ""
            json_path = Path(json_path)
            if not json_path.is_file():
                raise FileNotFoundError(f"Input file {json_path} not found!")
            sample_name = json_path.stem
            with json_path.open() as jf:
                sample_info = json.load(jf)
            sample_info["redirector"] = prefix
            files = list(sample_info.get("files", []))
            sample_info["files"] = files
            flist[sample_name] = [prefix + f for f in files]
            sample_info["xsec"] = float(sample_info["xsec"])
            sample_info["nEvents"] = int(sample_info["nEvents"])
            sample_info["nGenEvents"] = int(sample_info["nGenEvents"])
            sample_info["nSumOfWeights"] = float(sample_info["nSumOfWeights"])
            if not sample_info.get("isData", False):
                for wgt_var in self.weight_variables:
                    if wgt_var in sample_info:
                        sample_info[wgt_var] = float(sample_info[wgt_var])
            samplesdict[sample_name] = sample_info
        return samplesdict, flist

    def _expand_path(self, path: Path, prefix: str) -> List[Tuple[Path, str]]:
        if path.is_dir():
            return [
                (json_path.resolve(strict=True), prefix)
                for json_path in sorted(path.glob("*.json"))
            ]
        if path.suffix.lower() == ".cfg":
            return self._parse_cfg_file(path.resolve(strict=True), prefix)
        if path.suffix.lower() == ".json":
            return [(path.resolve(strict=True), prefix)]
        raise ValueError(f"Unsupported input type: {path}")

    def _parse_cfg_file(self, cfg_path: Path, starting_prefix: str) -> List[Tuple[Path, str]]:
        entries: List[Tuple[Path, str]] = []
        prefix = starting_prefix
        with cfg_path.open() as stream:
            for raw_line in stream:
                stripped = raw_line.split("#", 1)[0].strip()
                if not stripped:
                    continue
                for token in (segment.strip() for segment in stripped.split(",")):
                    if not token:
                        continue
                    candidate = Path(token).expanduser()
                    if not candidate.is_absolute():
                        candidate = cfg_path.parent / candidate
                    suffix = candidate.suffix.lower()
                    if suffix == ".cfg":
                        entries.extend(
                            self._parse_cfg_file(candidate.resolve(strict=True), prefix)
                        )
                    elif suffix == ".json":
                        entries.append((candidate.resolve(strict=True), prefix))
                    elif candidate.exists():
                        resolved = candidate.resolve(strict=True)
                        if resolved.is_dir():
                            entries.extend(self._expand_path(resolved, prefix))
                        else:
                            entries.append((resolved, prefix))
                    else:
                        prefix = token
        return entries


def weight_variations_from_metadata(
    metadata: Mapping[str, Any],
    fallback: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return the list of weight-variation sum-of-weight keys defined in metadata."""

    variations: List[str] = []

    def _collect(entry: Mapping[str, Any]) -> None:
        if not isinstance(entry, Mapping):
            return
        for variation in entry.get("variations", []) or []:
            if isinstance(variation, Mapping):
                sumw = variation.get("sum_of_weights")
                if sumw:
                    variations.append(str(sumw))
        for group in entry.get("groups", []) or []:
            if isinstance(group, Mapping):
                _collect(group)

    if isinstance(metadata, Mapping):
        systematics = metadata.get("systematics")
        if isinstance(systematics, Mapping):
            for info in systematics.values():
                if isinstance(info, Mapping):
                    _collect(info)

    unique_variations = unique_preserving_order(variations)
    if unique_variations:
        return unique_variations

    if fallback is None:
        fallback = DEFAULT_WEIGHT_VARIATIONS
    return list(fallback)


@dataclass
class RunConfig:
    """Normalized configuration for ``run_analysis.py``."""

    json_files: List[str] = field(default_factory=list)
    prefix: str = ""
    executor: str = "work_queue"
    test: bool = False
    pretend: bool = False
    nworkers: int = 8
    chunksize: int = 100000
    nchunks: Optional[int] = None
    outname: str = "plotsTopEFT"
    outpath: str = "histos"
    treename: str = "Events"
    do_errors: bool = False
    do_systs: bool = False
    split_lep_flavor: bool = False
    scenario_names: List[str] = field(default_factory=list)
    channel_feature_tags: List[str] = field(default_factory=list)
    skip_sr: bool = False
    skip_cr: bool = False
    do_np: bool = False
    do_renormfact_envelope: bool = False
    wc_list: List[str] = field(default_factory=list)
    port: str = "9123-9130"
    ecut: Optional[float] = None


class RunConfigBuilder:
    """Merge defaults, CLI arguments and YAML overrides into a :class:`RunConfig`."""

    def __init__(self, defaults: Optional["argparse.Namespace"] = None):
        self.defaults = defaults

    def build(
        self,
        args: "argparse.Namespace",
        options_path: Optional[str],
    ) -> RunConfig:
        config = RunConfig()

        field_specs: Dict[str, Tuple[str, Any]] = {
            "jsonFiles": ("json_files", coerce_json_files),
            "json_files": ("json_files", coerce_json_files),
            "prefix": ("prefix", lambda v: "" if v is None else str(v)),
            "executor": ("executor", lambda v: "" if v is None else str(v)),
            "test": ("test", coerce_bool),
            "pretend": ("pretend", coerce_bool),
            "nworkers": ("nworkers", lambda v: coerce_int(v, allow_none=False)),
            "chunksize": ("chunksize", lambda v: coerce_int(v, allow_none=False)),
            "nchunks": ("nchunks", lambda v: coerce_int(v, allow_none=True)),
            "outname": ("outname", lambda v: "" if v is None else str(v)),
            "outpath": ("outpath", lambda v: "" if v is None else str(v)),
            "treename": ("treename", lambda v: "" if v is None else str(v)),
            "do_errors": ("do_errors", coerce_bool),
            "do_systs": ("do_systs", coerce_bool),
            "split_lep_flavor": ("split_lep_flavor", coerce_bool),
            "scenarios": ("scenario_names", normalize_sequence),
            "channel_features": ("channel_feature_tags", normalize_sequence),
            "skip_sr": ("skip_sr", coerce_bool),
            "skip_cr": ("skip_cr", coerce_bool),
            "do_np": ("do_np", coerce_bool),
            "do_renormfact_envelope": ("do_renormfact_envelope", coerce_bool),
            "wc_list": ("wc_list", normalize_sequence),
            "ecut": ("ecut", coerce_optional_float),
            "port": ("port", coerce_port),
        }

        def _apply_source(source: Mapping[str, Any]):
            for key, value in source.items():
                if key not in field_specs:
                    continue
                field_name, coercer = field_specs[key]
                coerced = coercer(value)
                setattr(config, field_name, coerced)

        if options_path:
            with open(options_path, "r") as handle:
                options = yaml.safe_load(handle) or {}
            if not isinstance(options, Mapping):
                raise TypeError("Options YAML must define a mapping of overrides")
            _apply_source(options)

        cli_attr_map = {
            "jsonFiles": "jsonFiles",
            "prefix": "prefix",
            "executor": "executor",
            "test": "test",
            "pretend": "pretend",
            "nworkers": "nworkers",
            "chunksize": "chunksize",
            "nchunks": "nchunks",
            "outname": "outname",
            "outpath": "outpath",
            "treename": "treename",
            "do_errors": "do_errors",
            "do_systs": "do_systs",
            "split_lep_flavor": "split_lep_flavor",
            "scenarios": "scenarios",
            "channel_features": "channel_features",
            "skip_sr": "skip_sr",
            "skip_cr": "skip_cr",
            "do_np": "do_np",
            "do_renormfact_envelope": "do_renormfact_envelope",
            "wc_list": "wc_list",
            "ecut": "ecut",
            "port": "port",
        }

        defaults = self.defaults
        cli_values: Dict[str, Any] = {}
        for key, attr_name in cli_attr_map.items():
            if not hasattr(args, attr_name):
                continue
            current_value = getattr(args, attr_name)
            default_value = getattr(defaults, attr_name, None) if defaults is not None else None
            if current_value != default_value:
                cli_values[key] = current_value

        _apply_source(cli_values)

        config.scenario_names = unique_preserving_order(config.scenario_names)
        if not config.scenario_names:
            config.scenario_names = ["TOP_22_006"]
        config.channel_feature_tags = unique_preserving_order(config.channel_feature_tags)
        config.wc_list = unique_preserving_order(config.wc_list)
        return config


__all__ = [
    "DEFAULT_WEIGHT_VARIATIONS",
    "RunConfig",
    "RunConfigBuilder",
    "SampleLoader",
    "coerce_bool",
    "coerce_int",
    "coerce_json_files",
    "coerce_optional_float",
    "coerce_port",
    "normalize_sequence",
    "unique_preserving_order",
    "weight_variations_from_metadata",
]
