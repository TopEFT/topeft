#!/usr/bin/env python

import argparse
import json
import time
import cloudpickle
import gzip
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from topeft.modules.paths import topeft_path
from topcoffea.modules.paths import topcoffea_path

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import topcoffea.modules.utils as utils
import topcoffea.modules.remote_environment as remote_environment

from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.get_renormfact_envelope import get_renormfact_envelope
from topeft.modules.systematics import SystematicsHelper
from topeft.modules.channel_metadata import ChannelMetadataHelper
import analysis_processor

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
    "nSumOfWeights_renormfactUp",
    "nSumOfWeights_renormfactDown",
]


def _normalize_sequence(value: Any) -> List[str]:
    """Flatten ``value`` into a list of strings."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            result.extend(_normalize_sequence(item))
        return result
    return [str(value)]


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    """Return a list containing only the first occurrence of every value."""

    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _coerce_bool(value: Any) -> Optional[bool]:
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


def _coerce_int(value: Any, allow_none: bool = False) -> Optional[int]:
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


def _coerce_json_files(value: Any) -> List[str]:
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
            result.extend(_coerce_json_files(item))
        return result
    return [str(value)]


def _coerce_optional_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float or ``None`` if unset."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return float(stripped)
    return float(value)


def _coerce_port(value: Any) -> str:
    """Return a Work Queue port specification as ``min-max``."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable):
        ints = [str(_coerce_int(v, allow_none=False)) for v in value if v is not None]
        if not ints:
            return ""
        if len(ints) == 1:
            return ints[0]
        return f"{ints[0]}-{ints[1]}"
    return str(value)


def _coerce_optional_sequence(value: Any) -> Optional[List[str]]:
    """Normalize ``value`` into a list of strings or ``None``."""

    if value is None:
        return None
    return _normalize_sequence(value)


@dataclass
class RunConfig:
    """Normalized configuration for ``run_analysis.py``.

    The dataclass captures every command-line option and its YAML alias after
    type coercion. Values are seeded with the script defaults, then updated by
    command-line arguments, and finally overridden by entries in the optional
    YAML options file. The attributes mirror the existing local variables:

    ``json_files``
        Flattened list of input JSON/CFG paths from the positional argument or
        ``jsonFiles`` YAML entry.
    ``prefix``
        Redirector/prefix applied when expanding file paths. Config files may
        overwrite this during parsing.
    ``executor``
        Execution backend requested via ``--executor``. Validated against
        :data:`LST_OF_KNOWN_EXECUTORS` before dispatch.
    ``test``
        Fast-test flag (``--test``) that triggers the runtime overrides for
        ``nchunks``, ``chunksize`` and ``nworkers``.
    ``pretend``
        ``--pretend`` flag controlling whether the analysis exits after
        inspecting the inputs.
    ``nworkers``/``chunksize``/``nchunks``
        Worker and chunking controls coerced to integers, with ``nchunks`` being
        optional. ``--test`` mutates these fields downstream.
    ``outname``/``outpath``
        Output naming controls consumed when writing the pickled results.
    ``treename``
        Name of the tree to hand to coffea's runner.
    ``do_errors`` and ``do_systs``
        Booleans toggling covariance output and systematic processing.
    ``split_lep_flavor``
        Controls the flavor splitting logic in histogram key construction.
    ``scenario_names``/``channel_feature_tags``
        Scenario selection and feature tags resolved via metadata helpers. Empty
        scenarios default to ``["TOP_22_006"]`` for backward compatibility.
    ``skip_sr``/``skip_cr``
        Region filters inherited by all channel lookups.
    ``do_np``/``do_renormfact_envelope``
        Post-processing toggles. ``do_renormfact_envelope`` remains dependent on
        both ``do_systs`` and ``do_np`` as enforced in the main routine.
    ``wc_list``
        Normalized list of Wilson coefficients, extended from the sample
        metadata when not explicitly provided.
    ``hist_list``
        Optional list of histogram names used to specialize requests such as
        ``["ana"]`` or ``["cr"]``.
    ``port``
        Work Queue port range stored as ``min-max`` string; converted to an
        integer range when the backend requires it.
    ``ecut``
        Optional floating energy threshold applied during processor
        instantiation.
    """

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
    hist_list: Optional[List[str]] = None
    port: str = "9123-9130"
    ecut: Optional[float] = None


def build_run_config(
    args: argparse.Namespace,
    defaults: argparse.Namespace,
    options_path: Optional[str],
) -> RunConfig:
    """Return a fully merged :class:`RunConfig` instance.

    Parameters
    ----------
    args
        Parsed command-line arguments produced by ``argparse``.
    defaults
        Namespace containing the ``argparse`` default values. Used to detect
        which command-line options were explicitly provided by the user so
        they can take precedence over YAML overrides.
    options_path
        Path to a YAML file containing option overrides. When provided, values
        from the YAML file override the script defaults, while command-line
        arguments explicitly provided by the user take precedence over both the
        defaults and the YAML configuration.
    """

    config = RunConfig()

    field_specs: Dict[str, Any] = {
        "jsonFiles": ("json_files", _coerce_json_files),
        "json_files": ("json_files", _coerce_json_files),
        "prefix": ("prefix", lambda v: "" if v is None else str(v)),
        "executor": ("executor", lambda v: "" if v is None else str(v)),
        "test": ("test", _coerce_bool),
        "pretend": ("pretend", _coerce_bool),
        "nworkers": ("nworkers", lambda v: _coerce_int(v, allow_none=False)),
        "chunksize": ("chunksize", lambda v: _coerce_int(v, allow_none=False)),
        "nchunks": ("nchunks", lambda v: _coerce_int(v, allow_none=True)),
        "outname": ("outname", lambda v: "" if v is None else str(v)),
        "outpath": ("outpath", lambda v: "" if v is None else str(v)),
        "treename": ("treename", lambda v: "" if v is None else str(v)),
        "do_errors": ("do_errors", _coerce_bool),
        "do_systs": ("do_systs", _coerce_bool),
        "split_lep_flavor": ("split_lep_flavor", _coerce_bool),
        "scenarios": ("scenario_names", _normalize_sequence),
        "channel_features": ("channel_feature_tags", _normalize_sequence),
        "skip_sr": ("skip_sr", _coerce_bool),
        "skip_cr": ("skip_cr", _coerce_bool),
        "do_np": ("do_np", _coerce_bool),
        "do_renormfact_envelope": ("do_renormfact_envelope", _coerce_bool),
        "wc_list": ("wc_list", _normalize_sequence),
        "hist_list": ("hist_list", _coerce_optional_sequence),
        "ecut": ("ecut", _coerce_optional_float),
        "port": ("port", _coerce_port),
    }

    def _apply_source(source: Dict[str, Any]):
        for key, value in source.items():
            if key not in field_specs:
                continue
            field_name, coercer = field_specs[key]
            coerced = coercer(value)
            setattr(config, field_name, coerced)

    if options_path:
        with open(options_path, "r") as handle:
            options = yaml.safe_load(handle) or {}
        if not isinstance(options, dict):
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
        "hist_list": "hist_list",
        "ecut": "ecut",
        "port": "port",
    }

    cli_values: Dict[str, Any] = {}
    for key, attr_name in cli_attr_map.items():
        if not hasattr(args, attr_name):
            continue
        current_value = getattr(args, attr_name)
        default_value = getattr(defaults, attr_name, None)
        if current_value != default_value:
            cli_values[key] = current_value

    _apply_source(cli_values)

    config.scenario_names = _unique_preserving_order(config.scenario_names)
    if not config.scenario_names:
        config.scenario_names = ["TOP_22_006"]
    config.channel_feature_tags = _unique_preserving_order(config.channel_feature_tags)
    config.wc_list = _unique_preserving_order(config.wc_list)
    if config.hist_list is not None:
        config.hist_list = _normalize_sequence(config.hist_list)

    return config


def resolve_channel_groups(
    channel_helper,
    skip_sr,
    skip_cr,
    scenario_names=None,
    required_features=None,
):
    """Return the SR and CR channel groups along with their feature flags.

    Parameters
    ----------
    channel_helper : ChannelMetadataHelper
        Helper providing access to the grouped channel metadata.
    skip_sr : bool
        Whether signal-region categories should be skipped.
    skip_cr : bool
        Whether control-region categories should be skipped.
    scenario_names : Sequence[str], optional
        Scenario names defined in the metadata that specify which channel groups
        should participate in the run. Multiple entries are combined in order.
    required_features : Sequence[str], optional
        Feature tags used to locate additional channel groups. Any group whose
        declared features intersect with this set will be scheduled in addition
        to the selected scenarios.
    """

    sr_groups = []
    cr_groups = []
    active_features = set()
    seen_groups = set()
    required_feature_set = set(required_features or [])

    def _load_group(name):
        group = channel_helper.group(name)
        if name not in seen_groups:
            active_features.update(group.features)
            seen_groups.add(name)
        return group

    def _register_group(name):
        group = _load_group(name)
        if name.endswith("_CR"):
            if not skip_cr and group not in cr_groups:
                cr_groups.append(group)
        else:
            if not skip_sr and group not in sr_groups:
                sr_groups.append(group)

    scenario_names = list(scenario_names or [])
    for scenario_name in scenario_names:
        for group_name in channel_helper.scenario_groups(scenario_name):
            _register_group(group_name)

    if required_feature_set:
        feature_matched_groups = set()
        for group_name in channel_helper.group_names():
            group = channel_helper.group(group_name)
            if set(group.features) & required_feature_set:
                feature_matched_groups.add(group_name)
                _register_group(group_name)

        if feature_matched_groups:
            for scenario_name in channel_helper.scenario_names():
                scenario_groups = channel_helper.scenario_groups(scenario_name)
                if any(group_name in scenario_groups for group_name in feature_matched_groups):
                    for group_name in scenario_groups:
                        if group_name not in feature_matched_groups:
                            _register_group(group_name)

    if not sr_groups and not cr_groups and not seen_groups:
        raise ValueError(
            "No channel groups selected. Please specify a scenario or feature tag"
        )

    return sr_groups, cr_groups, frozenset(active_features)


def normalize_jet_category(jet_cat):
    """Return a standardized jet category suffix.

    Parameters
    ----------
    jet_cat : str
        Jet category string starting with one of ``=``, ``<``, or ``>``.

    Returns
    -------
    str
        Normalized jet category such as ``exactly_2j``.

    Raises
    ------
    ValueError
        If ``jet_cat`` does not start with a recognized comparison symbol.
    """

    jet_cat = str(jet_cat).strip()
    if jet_cat.startswith("="):
        tag = "exactly_"
    elif jet_cat.startswith("<"):
        tag = "atmost_"
    elif jet_cat.startswith(">"):
        tag = "atleast_"
    else:
        raise ValueError(f"jet_cat {jet_cat} misses =,<,> !")

    return f"{tag}{jet_cat[1:]}j"


def build_channel_dict(
    ch,
    appl,
    isData,
    skip_sr,
    skip_cr,
    channel_helper,
    scenario_names=None,
    required_features=None,
):

    import_sr_groups, import_cr_groups, active_features = resolve_channel_groups(
        channel_helper,
        skip_sr,
        skip_cr,
        scenario_names=scenario_names,
        required_features=required_features,
    )

    base_ch = ch

    # print("\nbase_ch:", base_ch)

    jet_suffix = None
    m = re.search(r"_(?:exactly_|atmost_|atleast_)?(\d+j)$", ch)
    if m:
        jet_suffix = m.group(1)
        base_ch = ch[: -(len(m.group(0)))]

    # print("jet_suffix:", jet_suffix)
    # print("base_ch:", base_ch)

    nlep_match = re.match(r"(\d+l)", base_ch)
    nlep_cat = nlep_match.group(1) if nlep_match else None

    def _find(groups):
        """Search the provided channel groups for matching metadata."""

        for group in groups:
            candidate_categories = []

            # Prefer categories whose region names directly match ``base_ch``.
            for category in group.categories():
                if any(region.name == base_ch for region in category.region_definitions):
                    candidate_categories.append(category)

            # Fall back to the lepton-multiplicity key if needed.
            if not candidate_categories and nlep_cat is not None:
                category = group.category(nlep_cat)
                if category is not None:
                    candidate_categories.append(category)

            for category in candidate_categories:
                appl_list = category.application_tags(isData)
                if appl not in appl_list:
                    continue
                for region in category.region_definitions:
                    if region.name != base_ch:
                        continue
                    jet_bins = category.jet_bins or [None]
                    for jet_cat in jet_bins:
                        jet_key = None
                        if jet_cat is not None:
                            jet_key = normalize_jet_category(jet_cat)
                            if jet_suffix and not jet_key.endswith(jet_suffix):
                                continue
                        elif jet_suffix:
                            continue
                        include_set = set(category.histogram_includes)
                        exclude_set = set(category.histogram_excludes)
                        include_set.update(region.include_histograms)
                        exclude_set.update(region.exclude_histograms)
                        if include_set:
                            exclude_set.difference_update(include_set)
                        features = set(active_features)
                        features.update(group.features)
                        return {
                            "jet_selection": jet_key,
                            "chan_def_lst": region.to_legacy_list(),
                            "lep_flav_lst": category.lepton_flavors,
                            "appl_region": appl,
                            "features": tuple(sorted(features)),
                            "channel_var_whitelist": tuple(sorted(include_set))
                            if include_set
                            else (),
                            "channel_var_blacklist": tuple(sorted(exclude_set)),
                        }
        return None

    ch_info = None
    if not skip_sr:
        ch_info = _find(import_sr_groups)
    if ch_info is None and not skip_cr:
        ch_info = _find(import_cr_groups)

    if ch_info is None:
        # Respect skip flags: if the requested application region was skipped,
        # return an empty dictionary so the caller can ignore this configuration.
        if (appl.startswith("isSR") and skip_sr) or (appl.startswith("isCR") and skip_cr):
            return {}
        raise ValueError(f"Channel {ch} with application {appl} not found")

    return ch_info


def build_channel_app_map(
    channel_helper,
    isData,
    skip_sr,
    skip_cr,
    scenario_names=None,
    required_features=None,
):
    """Extract channel names and their application regions from metadata."""
    import_sr_groups, import_cr_groups, _ = resolve_channel_groups(
        channel_helper,
        skip_sr,
        skip_cr,
        scenario_names=scenario_names,
        required_features=required_features,
    )

    def _collect(groups, result):
        for group in groups:
            for category in group.categories():
                appl_list = category.application_tags(isData)
                if not appl_list:
                    continue
                for region in category.region_definitions:
                    base_ch = region.name
                    jet_bins = category.jet_bins or [None]
                    for jet_cat in jet_bins:
                        if jet_cat is None:
                            continue
                        jet_suffix = normalize_jet_category(jet_cat)
                        clean_suffix = jet_suffix.split("_")[-1]
                        ch_name = f"{base_ch}_{clean_suffix}"
                        result.setdefault(ch_name, set()).update(appl_list)

    result = {}
    if not skip_sr:
        _collect(import_sr_groups, result)
    if not skip_cr:
        _collect(import_cr_groups, result)

    return {ch: sorted(list(apps)) for ch, apps in result.items()}


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
        help="Number of workers",
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
        "--treename",
        default="Events",
        help="Name of the tree inside the files",
    )
    parser.add_argument(
        "--do-errors",
        action="store_true",
        help="Save the w**2 coefficients",
    )
    parser.add_argument(
        "--do-systs",
        action="store_true",
        help="Compute systematic variations",
    )
    parser.add_argument(
        "--split-lep-flavor",
        action="store_true",
        help="Split up categories by lepton flavor",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help=(
            "Scenario name defined in metadata to select channel groups."
            " Defaults to 'TOP_22_006' when not provided. Can be supplied"
            " multiple times to combine scenarios."
        ),
    )
    parser.add_argument(
        "--channel-feature",
        dest="channel_features",
        action="append",
        help=(
            "Include channel groups advertising the specified feature tag"
            " (for example 'offz_split' or 'requires_tau')."
        ),
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
        "--do-renormfact-envelope",
        action="store_true",
        help=(
            "Perform renorm/fact envelope calculation on the output hist "
            "(saves the modified with the same name as the original)."
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
        help="Specify a list of histograms to fill.",
    )
    parser.add_argument(
        "--ecut",
        default=None,
        help="Energy cut threshold i.e. throw out events above this (GeV)",
    )
    parser.add_argument(
        "--port",
        default="9123-9130",
        help="Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.",
    )
    parser.add_argument(
        "--options",
        default=None,
        help="YAML file that specifies command-line options. Options explicitly set at command-line take precedence",
    )

    parser_defaults = parser.parse_args([])
    args = parser.parse_args()
    config = build_run_config(args, parser_defaults, args.options)

    # Check if we have valid options
    if config.executor not in LST_OF_KNOWN_EXECUTORS:
        raise Exception(
            f'The "{config.executor}" executor is not known. Please specify an executor from the known executors ({LST_OF_KNOWN_EXECUTORS}). Exiting.'
        )
    if config.do_renormfact_envelope:
        if not config.do_systs:
            raise Exception(
                "Error: Cannot specify do_renormfact_envelope if we are not including systematics."
            )
        if not config.do_np:
            raise Exception(
                "Error: Cannot specify do_renormfact_envelope if we have not already done the integration across the appl axis that occurs in the data driven estimator step."
            )
    if config.test:
        if config.executor == "futures":
            config.nchunks = 2
            config.chunksize = 100
            config.nworkers = 1
            print(
                "Running a fast test with %i workers, %i chunks of %i events"
                % (config.nworkers, config.nchunks, config.chunksize)
            )
        else:
            raise Exception(
                f'The "test" option is not set up to work with the {config.executor} executor. Exiting.'
            )

    # Set the threshold for the ecut (if not applying a cut, should be None)
    ecut_threshold = config.ecut
    if ecut_threshold is not None:
        ecut_threshold = float(ecut_threshold)

    port = config.port
    if config.executor in ["work_queue", "taskvine"]:
        # construct wq port range
        port = list(map(int, port.split("-")))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert single values into a range of one element
            port.append(port[0])

    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = (
            jsonFile if not "/" in jsonFile else jsonFile[jsonFile.rfind("/") + 1 :]
        )
        if sampleName.endswith(".json"):
            sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]["redirector"] = prefix

    json_files = config.json_files
    prefix = config.prefix

    for jsonFile in json_files:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith("/"):
                jsonFile += "/"
            for f in os.path.listdir(jsonFile):
                if f.endswith(".json"):
                    allInputFiles.append(jsonFile + f)
        else:
            allInputFiles.append(jsonFile)

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
                print(" >> Reading json from cfg file...")
                lines = fin.readlines()
                for l in lines:
                    if "#" in l:
                        l = l[: l.find("#")]
                    l = l.replace(" ", "").replace("\n", "")
                    if l == "":
                        continue
                    if "," in l:
                        l = l.split(",")
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)

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
                if config.do_systs:
                    if wgt_var not in samplesdict[sname]:
                        raise Exception(f'Missing weight variation "{wgt_var}".')
                    else:
                        samplesdict[sname][wgt_var] = float(samplesdict[sname][wgt_var])
        # # Print file info
        # print(">> " + sname)
        # print(
        #     "   - isData?      : %s" % ("YES" if samplesdict[sname]["isData"] else "NO")
        # )
        # print("   - year         : %s" % samplesdict[sname]["year"])
        # print("   - xsec         : %f" % samplesdict[sname]["xsec"])
        # print("   - histAxisName : %s" % samplesdict[sname]["histAxisName"])
        # print("   - options      : %s" % samplesdict[sname]["options"])
        # print("   - tree         : %s" % samplesdict[sname]["treeName"])
        # print("   - nEvents      : %i" % samplesdict[sname]["nEvents"])
        # print("   - nGenEvents   : %i" % samplesdict[sname]["nGenEvents"])
        # print("   - SumWeights   : %i" % samplesdict[sname]["nSumOfWeights"])
        # if not samplesdict[sname]["isData"]:
        #     for wgt_var in WGT_VAR_LST:
        #         if wgt_var in samplesdict[sname]:
        #             print(f"   - {wgt_var}: {samplesdict[sname][wgt_var]}")
        # print("   - Prefix       : %s" % samplesdict[sname]["redirector"])
        # print("   - nFiles       : %i" % len(samplesdict[sname]["files"]))
        # for fname in samplesdict[sname]["files"]:
        #     print("     %s" % fname)

    if config.pretend:
        print("pretending...")
        exit()

    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(config.wc_list) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]["WCnames"]:
                if wc not in config.wc_list:
                    config.wc_list.append(wc)

    if len(config.wc_list) > 0:
        # Yes, why not have the output be in correct English?
        if len(config.wc_list) == 1:
            wc_print = config.wc_list[0]
        elif len(config.wc_list) == 2:
            wc_print = config.wc_list[0] + " and " + config.wc_list[1]
        else:
            wc_print = ", ".join(config.wc_list[:-1]) + ", and " + config.wc_list[-1]
            print("Wilson Coefficients: {}.".format(wc_print))
    else:
        print("No Wilson coefficients specified")

    metadata_path = topeft_path("params/metadata.yml")
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    golden_jsons = metadata.get("golden_jsons", {}) if metadata else {}
    if not golden_jsons:
        raise ValueError("golden_jsons mapping missing from metadata.")

    var_defs = metadata["variables"]
    var_lst = list(var_defs)

    channels_metadata = metadata.get("channels") if metadata else None
    if not channels_metadata:
        raise ValueError("Channel metadata is missing from params/metadata.yml")
    channel_helper = ChannelMetadataHelper(channels_metadata)

    _, _, active_channel_features = resolve_channel_groups(
        channel_helper,
        skip_sr=config.skip_sr,
        skip_cr=config.skip_cr,
        scenario_names=config.scenario_names,
        required_features=config.channel_feature_tags,
    )

    samples_lst = list(samplesdict.keys())
    sample_years = {
        str(samplesdict[sample_name]["year"])
        for sample_name in samples_lst
        if "year" in samplesdict[sample_name]
    }

    syst_helper = SystematicsHelper(
        metadata,
        sample_years=sample_years,
        tau_analysis="requires_tau" in active_channel_features,
    )

    channel_app_map_mc = build_channel_app_map(
        channel_helper,
        isData=False,
        skip_sr=config.skip_sr,
        skip_cr=config.skip_cr,
        scenario_names=config.scenario_names,
        required_features=config.channel_feature_tags,
    )
    channel_app_map_data = build_channel_app_map(
        channel_helper,
        isData=True,
        skip_sr=config.skip_sr,
        skip_cr=config.skip_cr,
        scenario_names=config.scenario_names,
        required_features=config.channel_feature_tags,
    )

    hist_list_request = config.hist_list
    hist_lst = hist_list_request
    if hist_list_request == ["ana"]:
        hist_lst = ["njets", "lj0pt", "ptz"]
        if "requires_tau" in active_channel_features:
            hist_lst.append("ptz_wtau")
        if "requires_forward" in active_channel_features:
            hist_lst.append("lt")
    elif hist_list_request == ["cr"]:
        hist_lst = [
            "lj0pt",
            "ptz",
            "met",
            "ljptsum",
            "l0pt",
            "l0eta",
            "l1pt",
            "l1eta",
            "j0pt",
            "j0eta",
            "njets",
            "nbtagsl",
            "invmass",
        ]
        if "requires_tau" in active_channel_features:
            hist_lst.append("tau0pt")

    # raise RuntimeError("\n\nStopping here for debugging")

    available_systematics_by_sample_type = {
        "mc": syst_helper.names_by_type("mc", include_systematics=config.do_systs),
        "data": syst_helper.names_by_type("data", include_systematics=config.do_systs),
    }

    key_lst = []

    for sample in samples_lst:
        sample_info = samplesdict[sample]
        print("\nSample:", sample)
        # if not sample_info["isData"]:
        #     print("\nchannel_app_map_mc:", channel_app_map_mc)
        # else:
        #     print("\nchannel_app_map_data:", channel_app_map_data)

        ch_map = channel_app_map_data if sample_info["isData"] else channel_app_map_mc
        grouped_variations = syst_helper.grouped_variations_for_sample(
            sample_info, include_systematics=config.do_systs
        )
        sample_type_key = "data" if sample_info["isData"] else "mc"
        available_systematics = available_systematics_by_sample_type[sample_type_key]

        print("\n")
        print("grouped_variations:", {k.name: [v.name for v in vs] for k, vs in grouped_variations.items()})
        print("\n")

        for var in var_lst:
            var_info = var_defs[var].copy()
            for clean_ch, appl_list in ch_map.items():
                for appl in appl_list:
                    try:
                        channel_metadata = build_channel_dict(
                            clean_ch,
                            appl,
                            sample_info["isData"],
                            config.skip_sr,
                            config.skip_cr,
                            channel_helper,
                            scenario_names=config.scenario_names,
                            required_features=config.channel_feature_tags,
                        )
                    except ValueError:
                        channel_metadata = None

                    if not channel_metadata:
                        continue

                    whitelist = tuple(
                        channel_metadata.get("channel_var_whitelist") or ()
                    )
                    blacklist = set(
                        channel_metadata.get("channel_var_blacklist") or ()
                    )

                    if whitelist and var not in whitelist:
                        continue
                    if var in blacklist:
                        continue

                    for group_descriptor, variations in grouped_variations.items():
                        #print("\n", group_descriptor.name, [v.name for v in variations])
                        flavored_channel_names = ()
                        if config.split_lep_flavor:
                            flavored_candidates = []
                            if channel_metadata:
                                lep_flavors = channel_metadata.get("lep_flav_lst") or []
                                lep_chan_defs = channel_metadata.get("chan_def_lst") or []
                                jet_selection = channel_metadata.get("jet_selection")
                                lep_base = lep_chan_defs[0] if lep_chan_defs else None
                                if lep_base:
                                    for lep_flavor in lep_flavors:
                                        if not lep_flavor:
                                            continue
                                        flavored_name = analysis_processor.construct_cat_name(
                                            lep_base,
                                            njet_str=jet_selection,
                                            flav_str=lep_flavor,
                                        )
                                        flavored_candidates.append(flavored_name)
                            flavored_channel_names = tuple(flavored_candidates)

                        hist_keys = {}
                        for variation in variations:
                            syst_label = (
                                (group_descriptor.name, variation.name)
                                if len(variations) > 1
                                else variation.name
                            )
                            base_entry = (
                                var,
                                clean_ch,
                                appl,
                                sample,
                                syst_label,
                            )
                            key_entries = [base_entry]
                            if flavored_channel_names:
                                key_entries.extend(
                                    (
                                        var,
                                        flavored_name,
                                        appl,
                                        sample,
                                        syst_label,
                                    )
                                    for flavored_name in flavored_channel_names
                                )
                            hist_keys[variation.name] = tuple(key_entries)
                        key_lst.append(
                            (
                                sample,
                                var,
                                clean_ch,
                                appl,
                                group_descriptor,
                                tuple(variations),
                                hist_keys,
                                var_info,
                                available_systematics,
                            )
                        )
                        #break # For debugging, only run one group_descriptor
                    break # For debugging, only run one application region
                break  # For debugging, only run one channel
            break # For debugging, only run one variable
        break # For debugging, only run one sample

    if config.executor in ["work_queue", "taskvine"]:
        executor_args = {
            "manager_name": f"{os.environ['USER'].capitalize()}-{config.executor}-coffea",
            # find a port to run work queue in this range:
            "port": port,
            "debug_log": f"/tmp/{os.environ['USER']}/debug.log",
            "transactions_log": f"/tmp/{os.environ['USER']}/tr.log",
            "stats_log": f"/tmp/{os.environ['USER']}/stats.log",
            "tasks_accum_log": f"/tmp/{os.environ['USER']}/tasks.log",
            "environment_file": remote_environment.get_environment(
                extra_pip_local={"topeft": ["topeft", "setup.py"]},
                extra_conda=["pyyaml"],
            ),
            "extra_input_files": ["analysis_processor.py"],
            "retries": 15,
            # use mid-range compression for chunks results.
            # Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            "compression": 8,
            # automatically find an adequate resource allocation for tasks.
            # tasks are first tried using the maximum resources seen of previously ran
            # tasks. on resource exhaustion, they are retried with the maximum resource
            # values, if specified below. if a maximum is not specified, the task waits
            # forever until a larger worker connects.
            #'resource_monitor': True,
            "resource_monitor": "measure",
            "resources_mode": "auto",
            'filepath': f'/tmp/{os.environ["USER"]}', ##Placeholder to comment out if you don't want to save wq-factory dirs in $HOME
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
            # 'disk': 8000,   #MB
            # 'memory': 10000, #MB
            # control the size of accumulation tasks.
            #"treereduction": 10,
            'chunks_per_accum': 25,
            'chunks_accum_in_mem': 2,
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

    # Run the processor and get the output
    tstart = time.time()

    # print("chunksize: ", config.chunksize)
    # print("nchunks: ", config.nchunks)

    if config.executor == "futures":
        exec_instance = processor.futures_executor(workers=config.nworkers)
        runner = processor.Runner(
            exec_instance,
            schema=NanoAODSchema,
            chunksize=config.chunksize,
            maxchunks=config.nchunks,
        )
    elif config.executor == "work_queue":
        executor_instance = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=config.chunksize,
            maxchunks=config.nchunks,
            skipbadfiles=False,
            xrootdtimeout=180,
        )
    elif config.executor == "taskvine":
        try:
            executor_instance = processor.TaskVineExecutor(**executor_args)
        except AttributeError:
            raise RuntimeError("TaskVineExecutor not available.")
        runner = processor.Runner(
            executor_instance,
            schema=NanoAODSchema,
            chunksize=config.chunksize,
            maxchunks=config.nchunks,
            skipbadfiles=True,
            xrootdtimeout=300,
        )

    output = {}
    # print(f"Running over {len(key_lst)} configurations") #\n", key_lst)
    # for key in key_lst:
    #     print(" -", key[:-1])
    # print("\n\n\n\n\n\n")
    # raise RuntimeError("Stopping here for debugging")
    
    for key in key_lst:
        (
            sample,
            var,
            clean_ch,
            appl,
            group_descriptor,
            systematic_variations,
            hist_keys,
            var_info,
            available_systematics,
        ) = key
        sample_dict = samplesdict[sample]
        sample_flist = flist[sample][:1]

        channel_dict = build_channel_dict(
            clean_ch,
            appl,
            sample_dict["isData"],
            config.skip_sr,
            config.skip_cr,
            channel_helper,
            scenario_names=config.scenario_names,
            required_features=config.channel_feature_tags,
        )

        # If the channel dictionary is empty, this configuration corresponds
        # to a skipped signal/control region. Skip running the processor for it.
        if not channel_dict:
            continue

        golden_json_path = None
        if sample_dict["isData"]:
            year_key = str(sample_dict["year"])
            try:
                golden_json_relpath = golden_jsons[year_key]
            except KeyError as exc:
                raise ValueError(f"No golden JSON configured for data year '{year_key}'.") from exc
            golden_json_path = topcoffea_path(golden_json_relpath)
            if not os.path.exists(golden_json_path):
                raise FileNotFoundError(
                    f"Golden JSON file '{golden_json_path}' for year '{year_key}' was not found."
                )

        processor_instance = analysis_processor.AnalysisProcessor(
            sample_dict,
            config.wc_list,
            hist_keys=hist_keys,
            var_info=var_info,
            ecut_threshold=ecut_threshold,
            do_errors=config.do_errors,
            split_by_lepton_flavor=config.split_lep_flavor,
            channel_dict=channel_dict,
            golden_json_path=golden_json_path,
            systematic_variations=systematic_variations,
            available_systematics=available_systematics,
        )

        # # print("\nsample_dict:", sample_dict)
        # print("\nhist_keys:", hist_keys)
        # # print("\nchannel_helper groups:", channel_helper.group_names())
        # print("\nchannel_dict:", channel_dict)

        out = runner({sample: sample_flist}, config.treename, processor_instance)
        output.update(out)

        #break

    dt = time.time() - tstart

    if config.executor in ["work_queue", "taskvine"]:
        print(
            "Processed {} events in {} seconds ({:.2f} evts/sec).".format(
                nevts_total, dt, nevts_total / dt
            )
        )

    if config.executor == "futures":
        print(
            "Processing time: %1.2f s with %i workers (%.2f s cpu overall)"
            % (
                dt,
                config.nworkers,
                dt * config.nworkers,
            )
        )

    # Save the output
    if not os.path.isdir(config.outpath):
        os.system("mkdir -p %s" % config.outpath)
    out_pkl_file = os.path.join(config.outpath, config.outname + ".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(output, fout)
    print("Done!")

    # Run the data driven estimation, save the output
    if config.do_np:
        print("\nDoing the nonprompt estimation...")
        out_pkl_file_name_np = os.path.join(config.outpath, config.outname + "_np.pkl.gz")
        ddp = DataDrivenProducer(out_pkl_file, out_pkl_file_name_np)
        print(f"Saving output in {out_pkl_file_name_np}...")
        ddp.dumpToPickle()
        print("Done!")
        # Run the renorm fact envelope calculation
        if config.do_renormfact_envelope:
            print("\nDoing the renorm. fact. envelope calculation...")
            dict_of_histos = utils.get_hist_from_pkl(
                out_pkl_file_name_np, allow_empty=False
            )
            dict_of_histos_after_applying_envelope = get_renormfact_envelope(
                dict_of_histos
            )
            utils.dump_to_pkl(
                out_pkl_file_name_np, dict_of_histos_after_applying_envelope
            )
