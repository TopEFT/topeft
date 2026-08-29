"""Build the public 34-TTree missing-parton payload from njet datacards.

The producer compares private LO ``tllq`` cards with central NLO ``tZq``
cards. Stored values are fractional shifts; the DatacardMaker consumer adds
one and applies the resulting ``missing_parton`` nuisance to ``tllq`` and
``tHq``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    LEGACY_MISSING_PARTON_BRANCH,
    category_payload_layout,
    load_registry_payload_layout,
    validate_legacy_missing_parton_payload,
    validate_legacy_missing_parton_values,
    normalize_sr_registry,
    load_or_validate_selected_registry,
    SUPPORTED_SR_REGISTRIES,
)
from topeft.modules.paths import topeft_path


DEFAULT_YEARS = ("2016APV", "2016", "2017", "2018")
DEFAULT_CENTRAL_PROCESS = "tZq"
DEFAULT_PRIVATE_PROCESS = "tllq"
MISSING_PARTON_SYST_NAME = "missing_parton"
PHYSICAL_NJET_BIN_COUNT = 8
WEIGHTED_YIELD_ZERO_THRESHOLD = 1.0e-5
ROOT_TXT_RATE_REL_TOL = 1.0e-6
ROOT_TXT_RATE_ABS_TOL = 1.0e-6
LEGACY_CENTRAL_CARD_DIR = Path("parton_datacards/Run2/central_tZq")
LEGACY_PRIVATE_CARD_DIR = Path("parton_datacards/Run2/private_tllq")
LEGACY_OUTPUT_FILE = Path(
    topeft_path("data/missing_parton/missing_parton_test.root")
)

PROCESS_ALIASES = {
    DEFAULT_CENTRAL_PROCESS: (
        "tZq",
        "TZQB-Zto2L-4FS_MLL-30",
    ),
    DEFAULT_PRIVATE_PROCESS: ("tllq",),
}


class ConfigError(ValueError):
    """Raised when a requested producer configuration is not executable."""


@dataclass(frozen=True)
class CardFiles:
    root_path: Path
    txt_path: Path


@dataclass(frozen=True)
class selected_card_inventory(Mapping[str, CardFiles]):
    pairs: Mapping[str, CardFiles]
    unused_categories: tuple[str, ...]
    missing_root_categories: tuple[str, ...] = ()
    missing_txt_categories: tuple[str, ...] = ()

    def __getitem__(self, key: str) -> CardFiles:
        return self.pairs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class parsed_card:
    process_names: tuple[str, ...]
    rates: tuple[float, ...]
    rate_systematics: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class base_category_card_data:
    nominal_values: object
    shape_values: tuple[object, ...]
    bin_edges: object
    parsed_txt: parsed_card


@dataclass(frozen=True)
class category_payload_plan:
    base_channel: str
    central_process_name: str
    private_process_name: str
    central_integral: float
    private_integral: float
    neutralized_physical_njets: tuple[int, ...]
    stored_values: object
    jet_lst: tuple[str, ...] = ()
    terminal_mode: str = ""
    terminal_threshold: int = -1
    terminal_source_physical_njets: tuple[int, ...] = ()


@dataclass(frozen=True)
class payload_plan:
    categories: tuple[category_payload_plan, ...]
    registry: str = DEFAULT_SR_REGISTRY
    ch_lst_path: str = ""
    ch_lst_sha256: str = ""
    unused_central_categories: tuple[str, ...] = ()
    unused_private_categories: tuple[str, ...] = ()

    @property
    def values_by_category(self) -> Mapping[str, object]:
        return {
            category.base_channel: category.stored_values
            for category in self.categories
        }

    def to_printable_dict(self) -> dict[str, object]:
        return {
            "registry": self.registry,
            "ch_lst_path": self.ch_lst_path,
            "ch_lst_sha256": self.ch_lst_sha256,
            "category_count": len(self.categories),
            "base_category_order": [
                category.base_channel for category in self.categories
            ],
            "unused_input_categories": {
                "central": list(self.unused_central_categories),
                "private": list(self.unused_private_categories),
            },
            "missing_required_categories": [],
            "neutralized_bins": [
                {
                    "base_channel": category.base_channel,
                    "physical_njet": physical_njet,
                }
                for category in self.categories
                for physical_njet in category.neutralized_physical_njets
            ],
            "categories": [
                {
                    "base_channel": category.base_channel,
                    "central_process": category.central_process_name,
                    "private_process": category.private_process_name,
                    "central_integral": category.central_integral,
                    "private_integral": category.private_integral,
                    "neutralized_physical_njets": list(
                        category.neutralized_physical_njets
                    ),
                    "jet_lst": list(category.jet_lst),
                    "terminal_mode": category.terminal_mode,
                    "terminal_threshold": category.terminal_threshold,
                    "terminal_source_physical_njets": list(
                        category.terminal_source_physical_njets
                    ),
                    "stored_value_count": len(category.stored_values),
                    "stored_value_min": float(min(category.stored_values)),
                    "stored_value_max": float(max(category.stored_values)),
                }
                for category in self.categories
            ],
        }


@dataclass(frozen=True)
class ResolvedConfig:
    central_card_dir: Path
    private_card_dir: Path
    output_file: Path
    output_path: Path
    input_mode: str
    dry_run: bool
    overwrite: bool
    years: tuple[str, ...]
    time: bool
    var: str
    sr_registry: str

    def to_printable_dict(self) -> dict[str, object]:
        layout = load_registry_payload_layout(self.sr_registry)
        ch_lst_path = Path(topeft_path("channels/ch_lst.json"))
        return {
            "mode": "dry_run" if self.dry_run else "execute",
            "input_mode": self.input_mode,
            "central_card_dir": str(self.central_card_dir),
            "private_card_dir": str(self.private_card_dir),
            "central_process_aliases": list(
                PROCESS_ALIASES[DEFAULT_CENTRAL_PROCESS]
            ),
            "private_process_aliases": list(
                PROCESS_ALIASES[DEFAULT_PRIVATE_PROCESS]
            ),
            "base_category_count": len(layout.categories),
            "base_categories": list(layout.ordered_base_categories),
            "output_file": str(self.output_file),
            "output_path": str(self.output_path),
            "overwrite": self.overwrite,
            "years": list(self.years),
            "time": self.time,
            "var": self.var,
            "sr_registry": self.sr_registry,
            "ch_lst_json": str(ch_lst_path),
            "ch_lst_sha256": hashlib.sha256(ch_lst_path.read_bytes()).hexdigest(),
            "registry_layout": layout.to_printable_dict(),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a registry-selected missing-parton ROOT payload from "
            "central tZq and private tllq njet card directories."
        )
    )
    parser.add_argument(
        "--central-card-dir",
        "--central-dir",
        dest="central_card_dir",
        help=(
            "Directory containing central tZq ROOT/TXT pairs. "
            "When omitted with --private-card-dir, the historical "
            "parton_datacards/Run2 layout is used."
        ),
    )
    parser.add_argument(
        "--sr-registry",
        choices=SUPPORTED_SR_REGISTRIES,
        default=DEFAULT_SR_REGISTRY,
        help=f"SR registry for payload provenance (default: {DEFAULT_SR_REGISTRY}).",
    )
    parser.add_argument(
        "--private-card-dir",
        "--private-dir",
        dest="private_card_dir",
        help=(
            "Directory containing private tllq ROOT/TXT pairs. "
            "When omitted with --central-card-dir, the historical "
            "parton_datacards/Run2 layout is used."
        ),
    )
    parser.add_argument(
        "--output-file",
        "--output-payload",
        dest="output_file",
        help=(
            "Output ROOT payload. Defaults to the historical "
            "topeft/data/missing_parton/missing_parton_test.root path."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform complete inventory, card, numerical, and payload-plan "
            "validation without writing a ROOT file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        "--allow-overwrite",
        dest="overwrite",
        action="store_true",
        help="Atomically replace an existing output only after full validation.",
    )

    # Historical public flags retained at the argument-parsing boundary.
    parser.add_argument(
        "--years",
        default=[],
        action="extend",
        nargs="+",
        help="Historical year labels retained for CLI compatibility.",
    )
    parser.add_argument(
        "--time",
        "-t",
        action="store_true",
        help="Historical timestamp flag retained for CLI compatibility.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=".",
        help="Historical diagnostic output path retained for CLI compatibility.",
    )
    parser.add_argument(
        "--var",
        default="njets",
        help="Historical observable selector; payload production supports njets.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    explicit_central = args.central_card_dir is not None
    explicit_private = args.private_card_dir is not None
    if explicit_central != explicit_private:
        raise ConfigError(
            "--central-card-dir and --private-card-dir must be supplied together."
        )
    if args.var != "njets":
        raise ConfigError(
            "The compatibility-preserving missing-parton payload producer is "
            "defined only for --var njets."
        )

    input_mode = "explicit" if explicit_central else "legacy"
    central_card_dir = (
        Path(args.central_card_dir)
        if explicit_central
        else LEGACY_CENTRAL_CARD_DIR
    )
    private_card_dir = (
        Path(args.private_card_dir)
        if explicit_private
        else LEGACY_PRIVATE_CARD_DIR
    )
    sr_registry = normalize_sr_registry(args.sr_registry)
    load_or_validate_selected_registry(sr_registry)
    if sr_registry != DEFAULT_SR_REGISTRY and args.output_file is None:
        raise ConfigError("A nondefault --sr-registry requires an explicit --output-file.")
    output_file = (
        Path(args.output_file) if args.output_file else LEGACY_OUTPUT_FILE
    )
    years = tuple(args.years or DEFAULT_YEARS)
    config = ResolvedConfig(
        central_card_dir=central_card_dir,
        private_card_dir=private_card_dir,
        output_file=output_file,
        output_path=Path(args.output_path),
        input_mode=input_mode,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        years=years,
        time=bool(args.time),
        var=args.var,
        sr_registry=sr_registry,
    )
    if output_file.exists() and not config.overwrite:
        raise ConfigError(
            f"Refusing to overwrite existing output payload {output_file}. "
            "Pass --overwrite to authorize atomic replacement."
        )
    return config


def card_files_for_category(
    card_dir: str | Path,
    base_channel: str,
    var: str = "njets",
) -> CardFiles:
    stem = Path(card_dir) / f"ttx_multileptons-{base_channel}_{var}"
    return CardFiles(
        root_path=stem.with_suffix(".root"),
        txt_path=stem.with_suffix(".txt"),
    )


def _category_from_card_path(path: Path, *, extension: str, var: str) -> str:
    prefix = "ttx_multileptons-"
    suffix = f"_{var}.{extension}"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(
            f"Unexpected derivation-card filename {path.name!r}; expected "
            f"{prefix}<base_channel>{suffix}."
        )
    category = path.name[len(prefix) : -len(suffix)]
    if not category:
        raise ValueError(f"Empty base category in derivation-card path {path}.")
    return category


def _index_card_paths(
    paths: Iterable[Path],
    *,
    extension: str,
    var: str,
) -> Mapping[str, Path]:
    indexed_paths = []
    for path in paths:
        category = _category_from_card_path(
            Path(path),
            extension=extension,
            var=var,
        )
        indexed_paths.append((category, Path(path)))
    counts = Counter(category for category, _ in indexed_paths)
    duplicates = sorted(
        category for category, count in counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate {extension.upper()} derivation-card categories: "
            f"{duplicates!r}."
        )
    return {category: path for category, path in indexed_paths}


def discover_card_pairs(
    card_dir: str | Path,
    *,
    expected_categories: Sequence[str] | None = None,
    var: str = "njets",
    role: str,
    allow_missing: bool = False,
) -> selected_card_inventory:
    card_dir = Path(card_dir)
    if not card_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {role} derivation-card directory: {card_dir}."
        )

    root_paths = _index_card_paths(
        sorted(card_dir.glob(f"ttx_multileptons-*_{var}.root")),
        extension="root",
        var=var,
    )
    txt_paths = _index_card_paths(
        sorted(card_dir.glob(f"ttx_multileptons-*_{var}.txt")),
        extension="txt",
        var=var,
    )
    if expected_categories is None:
        expected_categories = load_registry_payload_layout().ordered_base_categories
    expected_categories = tuple(expected_categories)
    expected = set(expected_categories)
    observed_root = set(root_paths)
    observed_txt = set(txt_paths)
    missing_root = tuple(sorted(expected - observed_root))
    missing_txt = tuple(sorted(expected - observed_txt))
    unused_categories = tuple(sorted((observed_root | observed_txt) - expected))
    if (missing_root or missing_txt) and not allow_missing:
        raise ValueError(
            f"Invalid {role} derivation-card inventory in {card_dir}: "
            f"missing_root={list(missing_root)!r}, "
            f"missing_txt={list(missing_txt)!r}, "
            f"unused_categories={list(unused_categories)!r}. Selected payload "
            "builds require one ROOT/TXT pair for every registry category."
        )

    return selected_card_inventory(
        pairs=MappingProxyType(
            {
                category: CardFiles(
                    root_path=root_paths[category],
                    txt_path=txt_paths[category],
                )
                for category in expected_categories
                if category in root_paths and category in txt_paths
            }
        ),
        unused_categories=unused_categories,
        missing_root_categories=missing_root,
        missing_txt_categories=missing_txt,
    )


def root_key_without_cycle(root_key: str) -> str:
    return str(root_key).split(";", 1)[0].strip()


def process_from_root_key(root_key: str) -> str:
    object_name = root_key_without_cycle(root_key)
    if "_sm" not in object_name:
        return object_name.strip()
    return object_name.split("_sm", 1)[0].strip()


def systematic_from_root_key(root_key: str) -> str:
    object_name = root_key_without_cycle(root_key)
    if "_sm_" not in object_name:
        return ""
    return object_name.split("_sm_", 1)[1]


def process_aliases(process: str) -> tuple[str, ...]:
    normalized = str(process).strip()
    for aliases in PROCESS_ALIASES.values():
        if normalized in aliases:
            return aliases
    return (normalized,)


def matches_process_name(candidate: str, process: str) -> bool:
    return process_from_root_key(candidate) in process_aliases(process)


def is_nominal_key(root_key: str, process: str) -> bool:
    return (
        matches_process_name(root_key, process)
        and root_key_without_cycle(root_key).endswith("_sm")
    )


def is_shape_variation_key(root_key: str, process: str) -> bool:
    systematic = systematic_from_root_key(root_key)
    return (
        bool(systematic)
        and matches_process_name(root_key, process)
        and (systematic.endswith("Up") or systematic.endswith("Down"))
    )


def available_process_names(root_keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                process_from_root_key(root_key)
                for root_key in root_keys
                if "_sm" in root_key_without_cycle(root_key)
            }
        )
    )


def _find_existing_missing_parton_content(
    root_keys: Iterable[str],
    card_text: str,
) -> tuple[str, ...]:
    hits = [
        root_key_without_cycle(root_key)
        for root_key in root_keys
        if MISSING_PARTON_SYST_NAME in root_key_without_cycle(root_key).lower()
    ]
    for line in card_text.splitlines():
        fields = line.split()
        if fields and fields[0].lower().startswith(MISSING_PARTON_SYST_NAME):
            hits.append(line.strip())
    return tuple(hits)


def _parse_text_card(card_text: str, *, card_path: Path) -> parsed_card:
    process_names = None
    rates = None
    rate_systematics = []
    for raw_line in card_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "process":
            candidates = fields[1:]
            try:
                tuple(int(value) for value in candidates)
            except ValueError:
                process_names = tuple(candidates)
        elif fields[0] == "rate":
            try:
                rates = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric rate row in {card_path}: {line!r}."
                ) from exc
        elif len(fields) >= 2 and fields[1] == "lnN":
            rate_systematics.append((fields[0], tuple(fields[2:])))

    if process_names is None or rates is None:
        raise ValueError(
            f"Missing nominal process or rate row in text card {card_path}."
        )
    if len(process_names) != len(rates):
        raise ValueError(
            f"Process/rate column mismatch in {card_path}: "
            f"process_count={len(process_names)}, rate_count={len(rates)}."
        )
    for nuisance_name, values in rate_systematics:
        if len(values) != len(process_names):
            raise ValueError(
                f"lnN column mismatch for {nuisance_name!r} in {card_path}: "
                f"expected={len(process_names)}, found={len(values)}."
            )
    return parsed_card(
        process_names=process_names,
        rates=rates,
        rate_systematics=tuple(rate_systematics),
    )


def _validate_finite_template(
    values,
    *,
    card_path: Path,
    object_name: str,
    base_channel: str,
    role: str,
) -> None:
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        nonfinite = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValueError(
            f"Malformed {role} template: card={card_path}, "
            f"object={object_name!r}, base_channel={base_channel!r}, "
            f"shape={values.shape}, nonfinite_indices={nonfinite!r}."
        )


def _validate_txt_nominal_rate(
    parsed_txt: parsed_card,
    *,
    process: str,
    root_integral: float,
    card_path: Path,
    base_channel: str,
    role: str,
) -> str:
    matches = [
        (process_name, rate)
        for process_name, rate in zip(
            parsed_txt.process_names,
            parsed_txt.rates,
        )
        if matches_process_name(process_name, process)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one TXT semantic process for {process!r}: "
            f"card={card_path}, base_channel={base_channel!r}, role={role}, "
            f"matches={matches!r}, available={parsed_txt.process_names!r}."
        )
    process_name, txt_rate = matches[0]
    if not math.isfinite(txt_rate):
        raise ValueError(
            f"Non-finite TXT nominal rate in {card_path}: "
            f"process={process_name!r}, rate={txt_rate!r}."
        )
    if not math.isclose(
        root_integral,
        txt_rate,
        rel_tol=ROOT_TXT_RATE_REL_TOL,
        abs_tol=ROOT_TXT_RATE_ABS_TOL,
    ):
        absolute_difference = abs(root_integral - txt_rate)
        relative_difference = absolute_difference / max(
            abs(root_integral),
            abs(txt_rate),
            1.0,
        )
        raise ValueError(
            f"ROOT/TXT nominal disagreement: card={card_path}, "
            f"process={process_name!r}, base_channel={base_channel!r}, "
            f"role={role}, root_integral={root_integral}, txt_rate={txt_rate}, "
            f"absolute_difference={absolute_difference}, "
            f"relative_difference={relative_difference}, "
            f"rel_tol={ROOT_TXT_RATE_REL_TOL}, "
            f"abs_tol={ROOT_TXT_RATE_ABS_TOL}."
        )
    return process_name


def read_base_category_card(
    card_files: CardFiles,
    process: str,
    *,
    base_channel: str,
    role: str,
) -> tuple[base_category_card_data, str]:
    import numpy as np
    import uproot

    missing_files = [
        str(path)
        for path in (card_files.root_path, card_files.txt_path)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing card input file(s): " + ", ".join(missing_files)
        )

    card_text = card_files.txt_path.read_text(encoding="utf-8")
    parsed_txt = _parse_text_card(
        card_text,
        card_path=card_files.txt_path,
    )
    with uproot.open(card_files.root_path) as root_file:
        root_keys = list(root_file.keys())
        existing_missing_parton = _find_existing_missing_parton_content(
            root_keys,
            card_text,
        )
        if existing_missing_parton:
            raise ValueError(
                "Detected pre-existing missing_parton content in source card "
                f"{card_files.txt_path}: {existing_missing_parton[:5]!r}. "
                "Produce source cards with --skip-missing-parton-rate-syst."
            )

        nominal_keys = [
            root_key
            for root_key in root_keys
            if is_nominal_key(root_key, process)
        ]
        if len(nominal_keys) != 1:
            raise ValueError(
                f"Expected exactly one nominal ROOT semantic process for "
                f"{process!r}: card={card_files.root_path}, "
                f"base_channel={base_channel!r}, role={role}, "
                f"matches={[root_key_without_cycle(key) for key in nominal_keys]!r}, "
                f"available_processes={available_process_names(root_keys)!r}."
            )
        nominal_key = nominal_keys[0]
        nominal_values = np.asarray(
            root_file[nominal_key].values(flow=False),
            dtype=float,
        )
        bin_edges = np.asarray(
            root_file[nominal_key].axis().edges(flow=False),
            dtype=float,
        )
        _validate_finite_template(
            nominal_values,
            card_path=card_files.root_path,
            object_name=root_key_without_cycle(nominal_key),
            base_channel=base_channel,
            role=role,
        )

        shape_values = []
        for root_key in root_keys:
            if not is_shape_variation_key(root_key, process):
                continue
            object_name = root_key_without_cycle(root_key)
            if "fakes" in object_name:
                continue
            values = np.asarray(
                root_file[root_key].values(flow=False),
                dtype=float,
            )
            edges = np.asarray(
                root_file[root_key].axis().edges(flow=False),
                dtype=float,
            )
            _validate_finite_template(
                values,
                card_path=card_files.root_path,
                object_name=object_name,
                base_channel=base_channel,
                role=role,
            )
            if values.shape != nominal_values.shape or not np.array_equal(
                edges,
                bin_edges,
            ):
                raise ValueError(
                    f"Incompatible {role} shape template: "
                    f"card={card_files.root_path}, object={object_name!r}, "
                    f"base_channel={base_channel!r}, "
                    f"nominal_shape={nominal_values.shape}, "
                    f"shape={values.shape}, "
                    f"nominal_edges={bin_edges.tolist()}, "
                    f"shape_edges={edges.tolist()}."
                )
            shape_values.append(values)

    process_name = _validate_txt_nominal_rate(
        parsed_txt,
        process=process,
        root_integral=float(np.sum(nominal_values)),
        card_path=card_files.txt_path,
        base_channel=base_channel,
        role=role,
    )
    return (
        base_category_card_data(
            nominal_values=nominal_values,
            shape_values=tuple(shape_values),
            bin_edges=bin_edges,
            parsed_txt=parsed_txt,
        ),
        process_name,
    )


def validate_physical_njet_axis(
    values,
    bin_edges,
    *,
    base_channel: str,
    role: str,
) -> None:
    import numpy as np

    values = np.asarray(values)
    edges = np.asarray(bin_edges, dtype=float)
    expected_edges = np.arange(PHYSICAL_NJET_BIN_COUNT + 1, dtype=float)
    if (
        values.ndim != 1
        or len(values) != PHYSICAL_NJET_BIN_COUNT
        or not np.array_equal(edges, expected_edges)
    ):
        raise ValueError(
            f"Unexpected {role} physical-njet layout for {base_channel!r}: "
            f"expected eight 0..6 plus >=7 bins with edges "
            f"{expected_edges.tolist()}, found shape={values.shape}, "
            f"edges={edges.tolist()}."
        )


def _parse_rate_shift(value: str) -> tuple[float, float] | None:
    value = str(value).strip()
    if not value or value == "-":
        return None
    if "/" in value:
        low, high = value.split("/", 1)
        return 1.0 - float(low), float(high) - 1.0
    shift = float(value) - 1.0
    return shift, shift


def private_rate_errors(
    card_data: base_category_card_data,
    *,
    process: str = DEFAULT_PRIVATE_PROCESS,
) -> tuple[object, object]:
    import numpy as np

    nominal = np.asarray(card_data.nominal_values, dtype=float)
    down = np.zeros_like(nominal)
    up = np.zeros_like(nominal)
    for variation in card_data.shape_values:
        shift = np.asarray(variation, dtype=float) - nominal
        upward = shift > 0.0
        up[upward] = np.hypot(up[upward], shift[upward])
        down[~upward] = np.hypot(down[~upward], shift[~upward])

    for _, values in card_data.parsed_txt.rate_systematics:
        for process_name, raw_value in zip(
            card_data.parsed_txt.process_names,
            values,
        ):
            if not matches_process_name(process_name, process):
                continue
            parsed_shift = _parse_rate_shift(raw_value)
            if parsed_shift is None:
                continue
            down_shift, up_shift = parsed_shift
            down = np.hypot(down, nominal * down_shift)
            up = np.hypot(up, nominal * up_shift)

    if (
        not np.all(np.isfinite(down))
        or not np.all(np.isfinite(up))
        or np.any(down < 0.0)
        or np.any(up < 0.0)
    ):
        raise ValueError("Invalid private uncertainty values after card parsing.")
    return down, up


def calculate_missing_parton_per_bin(
    private_values,
    central_values,
    private_down_error,
    private_up_error,
    *,
    base_channel: str,
) -> tuple[object, object]:
    """Return per-bin missing-parton amounts and stored fractional shifts."""
    import numpy as np

    private_values = np.asarray(private_values, dtype=float)
    central_values = np.asarray(central_values, dtype=float)
    private_down_error = np.asarray(private_down_error, dtype=float)
    private_up_error = np.asarray(private_up_error, dtype=float)
    arrays = (
        private_values,
        central_values,
        private_down_error,
        private_up_error,
    )
    if any(array.shape != private_values.shape for array in arrays):
        raise ValueError(
            f"Missing-parton numerical array mismatch for {base_channel!r}: "
            f"shapes={[array.shape for array in arrays]!r}."
        )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            f"Non-finite missing-parton numerical input for {base_channel!r}."
        )
    if np.any(private_values <= -WEIGHTED_YIELD_ZERO_THRESHOLD):
        indices = np.flatnonzero(
            private_values <= -WEIGHTED_YIELD_ZERO_THRESHOLD
        ).tolist()
        raise ValueError(
            f"Materially negative private denominator for {base_channel!r} at physical "
            f"njet indices {indices!r}; no clipping or absolute-value fallback "
            "is allowed."
        )
    if np.any(private_down_error < 0.0) or np.any(private_up_error < 0.0):
        raise ValueError(
            f"Negative private uncertainty for {base_channel!r}."
        )

    missing_parton = np.zeros_like(private_values)
    stored_fraction = np.zeros_like(private_values)
    for index, (private, central, down_error, up_error) in enumerate(
        zip(
            private_values,
            central_values,
            private_down_error,
            private_up_error,
        )
    ):
        if abs(private) < WEIGHTED_YIELD_ZERO_THRESHOLD:
            continue
        effective_private = float(private)
        effective_central = (
            0.0
            if abs(central) < WEIGHTED_YIELD_ZERO_THRESHOLD
            else float(central)
        )

        delta = effective_private - effective_central
        selected_error = float(down_error if delta >= 0.0 else up_error)
        residual = max(delta * delta - selected_error * selected_error, 0.0)
        amount = math.sqrt(residual)
        fraction = amount / effective_private
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError(
                f"Invalid stored missing-parton fraction for "
                f"{base_channel!r}[{index}]: {fraction!r}."
            )
        if 1.0 + fraction <= 0.0:
            raise ValueError(
                f"Invalid consumer kappa_up for {base_channel!r}[{index}]: "
                f"stored_fraction={fraction!r}."
            )
        missing_parton[index] = amount
        stored_fraction[index] = fraction
    return missing_parton, stored_fraction


def _aggregate_card_population(
    card_data: base_category_card_data,
    *,
    physical_indices: Sequence[int],
) -> base_category_card_data:
    """Aggregate source-level card inputs before evaluating the formula."""
    import numpy as np

    physical_indices = tuple(int(index) for index in physical_indices)
    nominal_values = np.asarray(card_data.nominal_values, dtype=float)
    if (
        not physical_indices
        or min(physical_indices) < 0
        or max(physical_indices) >= len(nominal_values)
    ):
        raise ValueError(
            f"Invalid physical njet population {physical_indices!r} for source "
            f"shape {nominal_values.shape}."
        )
    selected_indices = list(physical_indices)
    return base_category_card_data(
        nominal_values=np.asarray(
            [float(np.sum(nominal_values[selected_indices]))]
        ),
        shape_values=tuple(
            np.asarray(
                [
                    float(
                        np.sum(
                            np.asarray(values, dtype=float)[selected_indices]
                        )
                    )
                ]
            )
            for values in card_data.shape_values
        ),
        bin_edges=np.asarray([0.0, 1.0]),
        parsed_txt=card_data.parsed_txt,
    )


def build_category_payload(
    *,
    base_channel: str,
    private_card: base_category_card_data,
    central_card: base_category_card_data,
    layout: category_payload_layout,
) -> object:
    import numpy as np

    validate_physical_njet_axis(
        private_card.nominal_values,
        private_card.bin_edges,
        base_channel=base_channel,
        role="private",
    )
    validate_physical_njet_axis(
        central_card.nominal_values,
        central_card.bin_edges,
        base_channel=base_channel,
        role="central",
    )
    if not np.array_equal(private_card.bin_edges, central_card.bin_edges):
        raise ValueError(
            f"Central/private njet bin-edge mismatch for {base_channel!r}: "
            f"private={private_card.bin_edges.tolist()}, "
            f"central={central_card.bin_edges.tolist()}."
        )

    if layout.base_sr_category != base_channel:
        raise ValueError(
            f"Payload layout/category mismatch: layout={layout.base_sr_category!r}, "
            f"input={base_channel!r}."
        )
    if not 0 <= layout.terminal_threshold < PHYSICAL_NJET_BIN_COUNT:
        raise ValueError(
            f"Registry terminal threshold {layout.terminal_threshold} for "
            f"{base_channel!r} is outside the maintained eight-bin source axis."
        )

    down_error, up_error = private_rate_errors(private_card)
    direct_stop = (
        layout.terminal_threshold
        if layout.terminal_is_inclusive
        else layout.public_array_length
    )
    _, direct_fractions = calculate_missing_parton_per_bin(
        np.asarray(private_card.nominal_values)[:direct_stop],
        np.asarray(central_card.nominal_values)[:direct_stop],
        np.asarray(down_error)[:direct_stop],
        np.asarray(up_error)[:direct_stop],
        base_channel=base_channel,
    )

    stored_values = direct_fractions
    if layout.terminal_is_inclusive:
        terminal_indices = tuple(
            range(layout.terminal_threshold, PHYSICAL_NJET_BIN_COUNT)
        )
        aggregate_private = _aggregate_card_population(
            private_card,
            physical_indices=terminal_indices,
        )
        aggregate_central = _aggregate_card_population(
            central_card,
            physical_indices=terminal_indices,
        )
        aggregate_down, aggregate_up = private_rate_errors(aggregate_private)
        _, aggregate_fraction = calculate_missing_parton_per_bin(
            aggregate_private.nominal_values,
            aggregate_central.nominal_values,
            aggregate_down,
            aggregate_up,
            base_channel=base_channel,
        )
        stored_values = np.concatenate((direct_fractions, aggregate_fraction))

    return validate_legacy_missing_parton_values(
        stored_values,
        base_channel=base_channel,
        expected_length=layout.public_array_length,
    )


def build_payload_plan(config: ResolvedConfig) -> payload_plan:
    import numpy as np

    layout = load_registry_payload_layout(config.sr_registry)
    expected_categories = layout.ordered_base_categories
    central_inventory = discover_card_pairs(
        config.central_card_dir,
        expected_categories=expected_categories,
        role="central",
        var=config.var,
        allow_missing=True,
    )
    private_inventory = discover_card_pairs(
        config.private_card_dir,
        expected_categories=expected_categories,
        role="private",
        var=config.var,
        allow_missing=True,
    )
    if any(
        (
            central_inventory.missing_root_categories,
            central_inventory.missing_txt_categories,
            private_inventory.missing_root_categories,
            private_inventory.missing_txt_categories,
        )
    ):
        raise ValueError(
            "Missing required registry-selected derivation-card categories: "
            f"central_missing_root={list(central_inventory.missing_root_categories)!r}, "
            f"central_missing_txt={list(central_inventory.missing_txt_categories)!r}, "
            f"private_missing_root={list(private_inventory.missing_root_categories)!r}, "
            f"private_missing_txt={list(private_inventory.missing_txt_categories)!r}."
        )
    validated_cards = {}
    for base_channel in expected_categories:
        private_card, private_process_name = read_base_category_card(
            private_inventory[base_channel],
            DEFAULT_PRIVATE_PROCESS,
            base_channel=base_channel,
            role="private",
        )
        central_card, central_process_name = read_base_category_card(
            central_inventory[base_channel],
            DEFAULT_CENTRAL_PROCESS,
            base_channel=base_channel,
            role="central",
        )
        validate_physical_njet_axis(
            private_card.nominal_values,
            private_card.bin_edges,
            base_channel=base_channel,
            role="private",
        )
        validate_physical_njet_axis(
            central_card.nominal_values,
            central_card.bin_edges,
            base_channel=base_channel,
            role="central",
        )
        if not np.array_equal(private_card.bin_edges, central_card.bin_edges):
            raise ValueError(
                f"Central/private njet bin-edge mismatch for {base_channel!r}: "
                f"private={private_card.bin_edges.tolist()}, "
                f"central={central_card.bin_edges.tolist()}."
            )
        validated_cards[base_channel] = (
            private_card,
            private_process_name,
            central_card,
            central_process_name,
        )

    category_plans = []
    for category_layout in layout.categories:
        base_channel = category_layout.base_sr_category
        (
            private_card,
            private_process_name,
            central_card,
            central_process_name,
        ) = validated_cards[base_channel]
        stored_values = build_category_payload(
            base_channel=base_channel,
            private_card=private_card,
            central_card=central_card,
            layout=category_layout,
        )
        neutralized_physical_njets = tuple(
            index
            for index, private_value in enumerate(private_card.nominal_values)
            if abs(float(private_value)) < WEIGHTED_YIELD_ZERO_THRESHOLD
        )
        category_plans.append(
            category_payload_plan(
                base_channel=base_channel,
                central_process_name=central_process_name,
                private_process_name=private_process_name,
                central_integral=float(
                    np.sum(central_card.nominal_values)
                ),
                private_integral=float(
                    np.sum(private_card.nominal_values)
                ),
                neutralized_physical_njets=neutralized_physical_njets,
                stored_values=stored_values,
                jet_lst=category_layout.jet_lst,
                terminal_mode=category_layout.terminal_mode,
                terminal_threshold=category_layout.terminal_threshold,
                terminal_source_physical_njets=(
                    tuple(
                        range(
                            category_layout.terminal_threshold,
                            PHYSICAL_NJET_BIN_COUNT,
                        )
                    )
                    if category_layout.terminal_is_inclusive
                    else (category_layout.terminal_threshold,)
                ),
            )
        )
    ch_lst_path = Path(topeft_path("channels/ch_lst.json"))
    return payload_plan(
        categories=tuple(category_plans),
        registry=layout.registry,
        ch_lst_path=str(ch_lst_path),
        ch_lst_sha256=hashlib.sha256(ch_lst_path.read_bytes()).hexdigest(),
        unused_central_categories=central_inventory.unused_categories,
        unused_private_categories=private_inventory.unused_categories,
    )


def _validate_in_memory_payload(
    payload_values: Mapping[str, object],
    *,
    sr_registry: str | None = None,
) -> None:
    layout = load_registry_payload_layout(sr_registry)
    expected_lengths = layout.public_lengths
    expected_keys = list(layout.ordered_base_categories)
    observed_keys = list(payload_values)
    if set(observed_keys) != set(expected_keys) or len(observed_keys) != len(
        expected_keys
    ):
        raise ValueError(
            f"Invalid in-memory legacy payload key set: "
            f"missing={sorted(set(expected_keys) - set(observed_keys))!r}, "
            f"unexpected={sorted(set(observed_keys) - set(expected_keys))!r}."
        )
    for base_channel in expected_keys:
        validate_legacy_missing_parton_values(
            payload_values[base_channel],
            base_channel=base_channel,
            expected_length=expected_lengths[base_channel],
        )


def write_legacy_payload_atomic(
    output_file: str | Path,
    payload_values: Mapping[str, object],
    *,
    overwrite: bool = False,
    sr_registry: str | None = None,
) -> str:
    """Write, validate, and atomically install a registry-selected payload."""
    import numpy as np
    import uproot

    output_file = Path(output_file)
    layout = load_registry_payload_layout(sr_registry)
    _validate_in_memory_payload(payload_values, sr_registry=layout.registry)
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output payload {output_file}. "
            "Pass --overwrite to authorize atomic replacement."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.",
        suffix=".tmp.root",
        dir=output_file.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with uproot.recreate(temporary_path) as payload_file:
            for base_channel in layout.ordered_base_categories:
                tree = payload_file.mktree(
                    base_channel,
                    {LEGACY_MISSING_PARTON_BRANCH: "float64"},
                )
                tree.extend(
                    {
                        LEGACY_MISSING_PARTON_BRANCH: np.asarray(
                            payload_values[base_channel],
                            dtype=np.float64,
                        )
                    }
                )
        validate_legacy_missing_parton_payload(
            temporary_path,
            sr_registry=layout.registry,
        )

        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Output payload appeared during validation: {output_file}. "
                "No replacement was performed."
            )
        os.replace(temporary_path, output_file)
        validate_legacy_missing_parton_payload(
            output_file,
            sr_registry=layout.registry,
        )
        return hashlib.sha256(output_file.read_bytes()).hexdigest()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_producer(config: ResolvedConfig) -> tuple[payload_plan, str | None]:
    plan = build_payload_plan(config)
    if config.dry_run:
        return plan, None
    output_sha256 = write_legacy_payload_atomic(
        config.output_file,
        plan.values_by_category,
        overwrite=config.overwrite,
        sr_registry=config.sr_registry,
    )
    return plan, output_sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        plan, output_sha256 = run_producer(config)
    except (ConfigError, FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))

    summary = {
        "resolved_config": config.to_printable_dict(),
        "payload_plan": plan.to_printable_dict(),
        "output_sha256": output_sha256,
        "write_performed": not config.dry_run,
    }
    if config.dry_run:
        print(
            "missing_parton.py dry-run validation succeeded; "
            "no ROOT payload was written."
        )
    else:
        print(
            f"Wrote validated registry-selected missing-parton payload: "
            f"{config.output_file} sha256={output_sha256}"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
