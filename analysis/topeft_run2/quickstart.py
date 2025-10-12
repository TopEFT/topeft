"""Quickstart helpers for running the Run 2 analysis pipeline.

This module exposes a small convenience layer on top of the traditional
``run_analysis.py`` entry point.  The helpers intentionally choose conservative
defaults so that new users can validate their setup on a laptop or CI worker
without needing to understand every configuration knob exposed by the main
workflow.  The :func:`prepare_samples` function performs lightweight validation
of the sample JSONs while :func:`run_quickstart` executes the Coffea workflow
with a trimmed histogram selection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from topeft.modules.paths import topeft_path
from topeft.modules.channel_metadata import ChannelMetadataHelper

from .run_analysis_helpers import (
    RunConfig,
    SampleLoader,
    coerce_json_files,
    normalize_sequence,
    unique_preserving_order,
    weight_variations_from_metadata,
)
from .workflow import (
    DEFAULT_SCENARIO_NAME,
    ChannelPlanner,
    ExecutorFactory,
    HistogramPlanner,
    RunWorkflow,
)

_DEFAULT_VARIABLES: Tuple[str, ...] = ("lj0pt",)


@dataclass(frozen=True)
class PreparedSamples:
    """Container describing the input samples resolved for a quickstart run."""

    metadata: Mapping[str, MutableMapping[str, object]]
    samples: Mapping[str, Mapping[str, object]]
    file_lists: Mapping[str, Sequence[str]]
    json_files: Tuple[str, ...]
    prefix: str
    scenario_names: Tuple[str, ...]
    channel_features: Tuple[str, ...]
    treename: str
    weight_variations: Tuple[str, ...]
    variable_names: Tuple[str, ...]

    @property
    def total_events(self) -> int:
        """Total number of generated events advertised by the samples."""

        return int(
            sum(int(info.get("nEvents", 0)) for info in self.samples.values())
        )


def _load_metadata(metadata_path: Optional[str]) -> MutableMapping[str, object]:
    try:  # pragma: no cover - optional dependency resolution
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to use the quickstart helpers") from exc

    if metadata_path is None:
        metadata_path = topeft_path("params/metadata.yml")
    metadata_file = Path(metadata_path).expanduser().resolve()
    with metadata_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, MutableMapping):
        raise TypeError("Metadata YAML must define a mapping of configuration blocks")
    return copy.deepcopy(loaded)  # ensure callers can mutate safely


def _select_variables(
    metadata: MutableMapping[str, object],
    variables: Optional[Iterable[str]],
) -> Tuple[str, ...]:
    var_defs = metadata.get("variables")
    if not isinstance(var_defs, MutableMapping):
        raise TypeError("metadata['variables'] must be a mapping of histogram definitions")
    if variables is None:
        return tuple(str(name) for name in var_defs.keys())

    selected: MutableMapping[str, object] = type(var_defs)()  # preserve ordering type
    missing: list[str] = []
    for name in variables:
        if name in var_defs:
            selected[name] = copy.deepcopy(var_defs[name])
        else:
            missing.append(str(name))
    if missing:
        raise KeyError(
            "The following variables are not defined in metadata: " + ", ".join(missing)
        )
    metadata["variables"] = selected
    return tuple(selected.keys())


def _flatten_inputs(cfg_path: Iterable[str] | str) -> Sequence[str]:
    if isinstance(cfg_path, (str, bytes)):
        return [str(cfg_path)]
    if isinstance(cfg_path, Path):
        return [str(cfg_path)]
    try:
        iterator = iter(cfg_path)
    except TypeError:
        return [str(cfg_path)]
    flattened = []
    for item in iterator:
        flattened.extend(_flatten_inputs(item))
    return flattened


def prepare_samples(
    cfg_path: Iterable[str] | str,
    scenario: Iterable[str] | str = DEFAULT_SCENARIO_NAME,
    *,
    metadata_path: Optional[str] = None,
    prefix: str = "",
    channel_features: Optional[Iterable[str]] = None,
    variables: Optional[Iterable[str]] = None,
) -> PreparedSamples:
    """Resolve sample JSONs and metadata into a quickstart configuration bundle.

    Parameters
    ----------
    cfg_path:
        Path (or collection of paths) pointing to JSON files, directories or CFG
        lists describing the samples to process.
    scenario:
        Name (or iterable of names) of the metadata scenarios to enable.  By
        default the TOP-22-006 reinterpretation selections are used.
    metadata_path:
        Optional override for the metadata YAML file.  When omitted the project
        default under :mod:`topeft.modules.paths` is used.
    prefix:
        Redirector prefix prepended to each file path.  This can be used to point
        to XRootD endpoints such as ``root://cmsxrootd.fnal.gov/``.
    channel_features:
        Optional iterable with metadata feature tags (for example
        ``"requires_tau"``) that should be activated in addition to the scenario
        definition.
    variables:
        Optional iterable of histogram variable names to keep.  When omitted all
        variables advertised in the metadata are retained.
    """

    metadata = _load_metadata(metadata_path)
    variable_names = _select_variables(metadata, variables)

    weight_variations = tuple(
        weight_variations_from_metadata(metadata, None)
    )

    loader = SampleLoader(default_prefix=prefix, weight_variables=weight_variations)
    normalized_inputs = []
    for entry in _flatten_inputs(cfg_path):
        normalized_inputs.extend(coerce_json_files(entry))
    json_inputs = [str(path) for path in normalized_inputs if str(path)]
    if not json_inputs:
        raise ValueError("At least one sample JSON, directory or CFG file must be provided")
    sample_specs = loader.collect(json_inputs)
    samplesdict, file_lists = loader.load(sample_specs)

    tree_names = {
        str(info.get("treeName", ""))
        for info in samplesdict.values()
        if info.get("treeName")
    }
    treename = tree_names.pop() if len(tree_names) == 1 else "Events"
    if len(tree_names) > 1:
        raise ValueError(
            "The selected samples advertise different tree names; please choose a single dataset"
        )

    channels_metadata = metadata.get("channels")
    if not channels_metadata:
        raise ValueError("Channel metadata is missing from the metadata YAML")

    scenario_names = unique_preserving_order(normalize_sequence(scenario))
    if not scenario_names:
        scenario_names = [DEFAULT_SCENARIO_NAME]

    feature_tags = unique_preserving_order(normalize_sequence(channel_features))

    channel_helper = ChannelMetadataHelper(channels_metadata)
    channel_planner = ChannelPlanner(
        channel_helper,
        skip_sr=False,
        skip_cr=False,
        scenario_names=scenario_names,
        required_features=feature_tags,
    )
    channel_planner.resolve_groups()  # validate that scenarios and features exist

    return PreparedSamples(
        metadata=metadata,
        samples=samplesdict,
        file_lists=file_lists,
        json_files=tuple(str(path) for path in json_inputs),
        prefix=prefix,
        scenario_names=tuple(scenario_names),
        channel_features=tuple(feature_tags),
        treename=treename,
        weight_variations=weight_variations,
        variable_names=variable_names,
    )


def run_quickstart(
    output_dir: str,
    *,
    cfg_path: Iterable[str] | str,
    scenario: Iterable[str] | str = DEFAULT_SCENARIO_NAME,
    metadata_path: Optional[str] = None,
    prefix: str = "",
    channel_features: Optional[Iterable[str]] = None,
    variables: Optional[Iterable[str]] = _DEFAULT_VARIABLES,
    executor: str = "futures",
    nworkers: int = 1,
    chunksize: int = 50000,
    nchunks: Optional[int] = 2,
    outname: str = "quickstart",
    treename: Optional[str] = None,
    split_lep_flavor: bool = False,
    do_systs: bool = False,
    skip_sr: bool = False,
    skip_cr: bool = False,
    do_np: bool = False,
    wc_list: Optional[Iterable[str]] = None,
    pretend: bool = False,
    test: bool = False,
) -> RunConfig:
    """Execute the Run 2 processor with conservative defaults.

    The function returns the :class:`RunConfig` instance used for the execution so
    that callers can easily reproduce or tweak the configuration.
    """

    prepared = prepare_samples(
        cfg_path,
        scenario=scenario,
        metadata_path=metadata_path,
        prefix=prefix,
        channel_features=channel_features,
        variables=variables,
    )

    config = RunConfig(
        json_files=list(prepared.json_files),
        prefix=prepared.prefix,
        executor=executor,
        test=test,
        pretend=pretend,
        nworkers=nworkers,
        chunksize=chunksize,
        nchunks=nchunks,
        outname=outname,
        outpath=str(output_dir),
        treename=treename or prepared.treename,
        do_systs=do_systs,
        split_lep_flavor=split_lep_flavor,
        scenario_names=list(prepared.scenario_names),
        channel_feature_tags=list(prepared.channel_features),
        skip_sr=skip_sr,
        skip_cr=skip_cr,
        do_np=do_np,
        wc_list=unique_preserving_order(normalize_sequence(wc_list)),
    )

    sample_loader = SampleLoader(
        default_prefix=config.prefix,
        weight_variables=prepared.weight_variations,
    )

    channel_helper = ChannelMetadataHelper(prepared.metadata["channels"])
    channel_planner = ChannelPlanner(
        channel_helper,
        skip_sr=config.skip_sr,
        skip_cr=config.skip_cr,
        scenario_names=config.scenario_names,
        required_features=config.channel_feature_tags,
    )

    histogram_planner = HistogramPlanner(
        config=config,
        variable_definitions=prepared.metadata["variables"],
        channel_planner=channel_planner,
    )

    executor_factory = ExecutorFactory(config)

    workflow = RunWorkflow(
        config=config,
        metadata=prepared.metadata,
        sample_loader=sample_loader,
        channel_planner=channel_planner,
        histogram_planner=histogram_planner,
        executor_factory=executor_factory,
        weight_variations=prepared.weight_variations,
    )
    workflow.run()
    return config


__all__ = [
    "DEFAULT_SCENARIO_NAME",
    "PreparedSamples",
    "prepare_samples",
    "run_quickstart",
]
