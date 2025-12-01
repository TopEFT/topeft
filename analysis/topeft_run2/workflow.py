"""Workflow utilities for orchestrating Run 2 analyses.

This module provides a small collection of helper classes that encapsulate the
core steps performed by :mod:`analysis.topeft_run2.run_analysis`.  The helpers
are designed to be lightweight wrappers around the existing functionality while
making the orchestration of a run easier to understand and reuse from Python
code.  The main entry point is :class:`RunWorkflow` together with the
``run_workflow`` convenience function.  A detailed walkthrough of the execution
flow, systematic catalogue, and extension hooks lives in
``docs/analysis_processing.md``.

During planning the workflow records every histogram combination that will be
submitted to Coffea.  Each entry tracks the ``(sample, channel, variable,
application, systematic)`` tuple that uniquely identifies a histogram fill.
The combinations are exposed through :class:`HistogramPlan` and printed just
before task submission.  The ``summary_verbosity`` configuration controls
whether no summary (``"none"``), only a table (``"brief"``), or both a table
and structured YAML/JSON dump (``"full"``) are emitted.  When the
``log_tasks`` flag is enabled, the futures executor also emits a concise
single-line log echoing the identifying tuple for each submitted histogram
task.
"""

from __future__ import annotations

import importlib
import getpass
import gzip
import json
import logging
import os
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import topcoffea

from topeft.modules.executor import (
    build_futures_executor,
    build_taskvine_args,
    futures_runner_overrides,
    instantiate_taskvine_executor,
    parse_port_range,
    resolve_environment_file,
    taskvine_log_configurator,
)
from topeft.modules.runner_output import normalise_runner_output, tuple_dict_stats

logger = logging.getLogger(__name__)


def _import_topcoffea_submodule(submodule: str):
    module_name = f"{topcoffea.__name__}.modules.{submodule}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            (
                "Unable to import required topcoffea helper module '%s'. "
                "Ensure the sibling topcoffea checkout is available and on the "
                "'ch_update_calcoffea' branch."
            )
            % module_name
        ) from exc


_topcoffea_paths = _import_topcoffea_submodule("paths")
topcoffea_path = _topcoffea_paths.topcoffea_path
topcoffea_utils = _import_topcoffea_submodule("utils")

from .run_analysis_helpers import (
    DEFAULT_WEIGHT_VARIATIONS,
    RunConfig,
    SampleLoader,
    unique_preserving_order,
    weight_variations_from_metadata,
)
from .nanoevents_helpers import nanoevents_factory_from_root

DEFAULT_SCENARIO_NAME = "TOP_22_006"

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from topeft.modules.channel_metadata import ChannelMetadataHelper
    from topeft.modules.systematics import SystematicsHelper

LST_OF_KNOWN_EXECUTORS = ["futures", "iterative", "taskvine"]


class ChannelPlanner:
    """Resolve channel metadata into lookups used during processing."""

    def __init__(
        self,
        channel_helper: "ChannelMetadataHelper",
        *,
        skip_sr: bool = False,
        skip_cr: bool = False,
        scenario_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._channel_helper = channel_helper
        self._skip_sr = bool(skip_sr)
        self._skip_cr = bool(skip_cr)
        self._scenario_names = list(scenario_names or [])

        self._sr_groups = None
        self._cr_groups = None
        self._active_features: Optional[Tuple[str, ...]] = None
        self._channel_app_cache: Dict[bool, Dict[str, List[str]]] = {}

    @property
    def active_features(self) -> Tuple[str, ...]:
        """Return the set of metadata features activated for this run."""

        if self._active_features is None:
            self._resolve_groups()
        return self._active_features or ()

    def resolve_groups(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[str, ...]]:
        """Expose the resolved channel groups and active features."""

        return self._resolve_groups()

    def build_channel_dict(
        self,
        channel: str,
        application: str,
        *,
        is_data: bool,
    ) -> Mapping[str, Any]:
        """Return the metadata describing ``channel`` and ``application``."""

        sr_groups, cr_groups, active_features = self._resolve_groups()

        base_channel = channel
        jet_suffix = None

        import re

        match = re.search(r"_(?:exactly_|atmost_|atleast_)?(\d+j)$", channel)
        if match:
            jet_suffix = match.group(1)
            base_channel = channel[: -(len(match.group(0)))]

        nlep_match = re.match(r"(\d+l)", base_channel)
        nlep_cat = nlep_match.group(1) if nlep_match else None

        def _normalize_group(group_list: Iterable[Any]) -> Optional[Mapping[str, Any]]:
            for group in group_list:
                candidate_categories = []

                for category in group.categories():
                    if any(region.name == base_channel for region in category.region_definitions):
                        candidate_categories.append(category)

                if not candidate_categories and nlep_cat is not None:
                    category = group.category(nlep_cat)
                    if category is not None:
                        candidate_categories.append(category)

                for category in candidate_categories:
                    appl_list = category.application_tags(is_data)
                    if application not in appl_list:
                        continue
                    for region in category.region_definitions:
                        if region.name != base_channel:
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
                            chan_def_lst = self._normalize_channel_definition(
                                region.to_legacy_list(), active_features
                            )
                            return {
                                "jet_selection": jet_key,
                                "chan_def_lst": chan_def_lst,
                                "lep_flav_lst": category.lepton_flavors,
                                "appl_region": application,
                                "features": tuple(sorted(features)),
                                "channel_var_whitelist": tuple(sorted(include_set))
                                if include_set
                                else (),
                                "channel_var_blacklist": tuple(sorted(exclude_set)),
                            }
            return None

        channel_info: Optional[Mapping[str, Any]] = None
        if not self._skip_sr:
            channel_info = _normalize_group(sr_groups)
        if channel_info is None and not self._skip_cr:
            channel_info = _normalize_group(cr_groups)

        if channel_info is None:
            if (application.startswith("isSR") and self._skip_sr) or (
                application.startswith("isCR") and self._skip_cr
            ):
                return {}
            raise ValueError(f"Channel {channel} with application {application} not found")

        return channel_info

    def channel_app_map(self, *, is_data: bool) -> Mapping[str, List[str]]:
        """Return a mapping of channel names to application tags."""

        if is_data in self._channel_app_cache:
            return self._channel_app_cache[is_data]

        sr_groups, cr_groups, _ = self._resolve_groups()

        def _collect(groups: Iterable[Any], result: Dict[str, List[str]]) -> None:
            for group in groups:
                for category in group.categories():
                    appl_list = category.application_tags(is_data)
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
                            current = result.setdefault(ch_name, [])
                            for appl in appl_list:
                                if appl not in current:
                                    current.append(appl)

        result: Dict[str, List[str]] = {}
        if not self._skip_sr:
            _collect(sr_groups, result)
        if not self._skip_cr:
            _collect(cr_groups, result)

        self._channel_app_cache[is_data] = {k: sorted(v) for k, v in result.items()}
        return self._channel_app_cache[is_data]

    def _resolve_groups(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[str, ...]]:
        if self._sr_groups is not None and self._cr_groups is not None:
            return self._sr_groups, self._cr_groups, self._active_features or ()

        sr_groups: List[Any] = []
        cr_groups: List[Any] = []
        active_features = set()
        seen_groups = set()

        channel_helper = self._channel_helper

        def _load_group(name: str):
            group = channel_helper.group(name)
            if name not in seen_groups:
                active_features.update(group.features)
                seen_groups.add(name)
            return group

        def _register_group(name: str) -> None:
            group = _load_group(name)
            if name.endswith("_CR"):
                if not self._skip_cr and group not in cr_groups:
                    cr_groups.append(group)
            else:
                if not self._skip_sr and group not in sr_groups:
                    sr_groups.append(group)

        group_names = channel_helper.selected_group_names(self._scenario_names)
        for group_name in group_names:
            _register_group(group_name)

        if not sr_groups and not cr_groups and not seen_groups:
            raise ValueError("No channel groups selected. Please specify at least one scenario")

        self._sr_groups = tuple(sr_groups)
        self._cr_groups = tuple(cr_groups)
        self._active_features = tuple(sorted(active_features))
        return self._sr_groups, self._cr_groups, self._active_features

    @staticmethod
    def _normalize_channel_definition(
        chan_def_lst: Sequence[str], active_features: Iterable[str]
    ) -> List[str]:
        """Return ``chan_def_lst`` adjusted for active metadata features."""

        normalized = list(chan_def_lst)
        if "offz_split" in set(active_features) and "3l_offZ" in normalized:
            normalized = [
                "3l_offZ_split" if entry == "3l_offZ" else entry
                for entry in normalized
            ]
        return normalized


def normalize_jet_category(jet_cat: Any) -> str:
    """Return a standardized jet category suffix."""

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


@dataclass(frozen=True)
class HistogramTask:
    """Description of a single histogram filling task."""

    sample: str
    variable: str
    clean_channel: str
    application: str
    group_descriptor: Any
    variations: Tuple[Any, ...]
    hist_keys: Mapping[str, Tuple[Tuple[Any, ...], ...]]
    variable_info: Mapping[str, Any]
    available_systematics: Sequence[str]
    channel_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HistogramCombination:
    """Canonical tuple describing a histogram that will be filled."""

    sample: str
    channel: str
    variable: str
    application: str
    systematic: str


@dataclass(frozen=True)
class ChannelApplicationSelection:
    """Filtered description of a channel/application pair for a variable."""

    clean_channel: str
    application: str
    metadata: Mapping[str, Any]
    flavored_channels: Tuple[str, ...]


@dataclass(frozen=True)
class SystematicExpansion:
    """Expansion of grouped systematic variations for a histogram task."""

    group_descriptor: Any
    variations: Tuple[Any, ...]
    hist_keys: Mapping[str, Tuple[Tuple[Any, ...], ...]]
    summary_entries: Tuple[HistogramCombination, ...]


@dataclass
class SummaryAccumulator:
    """Collect and deduplicate histogram summary combinations."""

    seen: Set[HistogramCombination] = field(default_factory=set)
    entries: List[HistogramCombination] = field(default_factory=list)

    def add_entries(self, new_entries: Iterable[HistogramCombination]) -> None:
        for entry in new_entries:
            if entry in self.seen:
                continue
            self.seen.add(entry)
            self.entries.append(entry)

    def as_tuple(self) -> Tuple[HistogramCombination, ...]:
        return tuple(self.entries)


@dataclass(frozen=True)
class HistogramPlan:
    """Collection of histogram tasks computed for a workflow."""

    tasks: List[HistogramTask]
    histogram_names: Sequence[str]
    summary: Sequence[HistogramCombination]


class HistogramPlanner:
    """Compute the histogram tasks required for a run."""

    def __init__(
        self,
        *,
        config: RunConfig,
        variable_definitions: Mapping[str, MutableMapping[str, Any]],
        channel_planner: ChannelPlanner,
    ) -> None:
        self._config = config
        self._var_defs = variable_definitions
        self._channel_planner = channel_planner
        self._analysis_processor_module = None

    def plan(
        self,
        samplesdict: Mapping[str, Mapping[str, Any]],
        systematics_helper: "SystematicsHelper",
    ) -> HistogramPlan:
        hist_lst = unique_preserving_order(self._var_defs.keys())
        if not hist_lst:
            raise ValueError("Histogram selection resolved to an empty list")

        available_systematics_by_sample_type = {
            "mc": systematics_helper.names_by_type("mc", include_systematics=self._config.do_systs),
            "data": systematics_helper.names_by_type(
                "data", include_systematics=self._config.do_systs
            ),
        }

        tasks: List[HistogramTask] = []
        summary_accumulator = SummaryAccumulator()

        channel_map_mc = self._channel_planner.channel_app_map(is_data=False)
        channel_map_data = self._channel_planner.channel_app_map(is_data=True)

        for sample, sample_info in samplesdict.items():
            ch_map = channel_map_data if sample_info.get("isData") else channel_map_mc
            grouped_variations = systematics_helper.grouped_variations_for_sample(
                sample_info, include_systematics=self._config.do_systs
            )
            sample_type_key = "data" if sample_info.get("isData") else "mc"
            available_systematics = available_systematics_by_sample_type[sample_type_key]

            for var in hist_lst:
                var_info = dict(self._var_defs[var])
                selections = self._iter_channel_applications(
                    sample_info=sample_info,
                    variable=var,
                    channel_map=ch_map,
                )
                for selection in selections:
                    expansions = self._expand_systematics(
                        sample=sample,
                        variable=var,
                        clean_channel=selection.clean_channel,
                        application=selection.application,
                        grouped_variations=grouped_variations,
                        flavored_channel_names=selection.flavored_channels,
                    )

                    for expansion in expansions:
                        summary_accumulator.add_entries(expansion.summary_entries)
                        tasks.append(
                            HistogramTask(
                                sample=sample,
                                variable=var,
                                clean_channel=selection.clean_channel,
                                application=selection.application,
                                group_descriptor=expansion.group_descriptor,
                                variations=expansion.variations,
                                hist_keys=expansion.hist_keys,
                                variable_info=var_info,
                                available_systematics=available_systematics,
                                channel_metadata=selection.metadata,
                            )
                        )

        return HistogramPlan(
            tasks=tasks,
            histogram_names=hist_lst,
            summary=summary_accumulator.as_tuple(),
        )

    def _iter_channel_applications(
        self,
        *,
        sample_info: Mapping[str, Any],
        variable: str,
        channel_map: Mapping[str, Sequence[str]],
    ) -> Iterator[ChannelApplicationSelection]:
        is_data = sample_info.get("isData", False)
        for clean_ch, appl_list in channel_map.items():
            for appl in appl_list:
                try:
                    channel_metadata = self._channel_planner.build_channel_dict(
                        clean_ch,
                        appl,
                        is_data=is_data,
                    )
                except ValueError:
                    continue

                if not channel_metadata:
                    continue

                whitelist = tuple(channel_metadata.get("channel_var_whitelist") or ())
                blacklist = set(channel_metadata.get("channel_var_blacklist") or ())

                if whitelist and variable not in whitelist:
                    continue
                if variable in blacklist:
                    continue

                flavored_channels = self._resolve_flavored_channels(channel_metadata)

                yield ChannelApplicationSelection(
                    clean_channel=clean_ch,
                    application=appl,
                    metadata=channel_metadata,
                    flavored_channels=flavored_channels,
                )

    def _resolve_flavored_channels(
        self,
        channel_metadata: Mapping[str, Any],
    ) -> Tuple[str, ...]:
        if not self._config.split_lep_flavor:
            return ()

        analysis_processor_module = self._get_analysis_processor_module()
        flavored_candidates: List[str] = []
        lep_flavors = channel_metadata.get("lep_flav_lst") or []
        lep_chan_defs = channel_metadata.get("chan_def_lst") or []
        jet_selection = channel_metadata.get("jet_selection")
        lep_base = lep_chan_defs[0] if lep_chan_defs else None
        if lep_base:
            for lep_flavor in lep_flavors:
                if not lep_flavor:
                    continue
                flavored_name = analysis_processor_module.construct_cat_name(
                    lep_base,
                    njet_str=jet_selection,
                    flav_str=lep_flavor,
                )
                flavored_candidates.append(flavored_name)
        return tuple(flavored_candidates)

    def _get_analysis_processor_module(self):
        if self._analysis_processor_module is None:
            from . import analysis_processor as analysis_processor_module

            self._analysis_processor_module = analysis_processor_module
        return self._analysis_processor_module

    def _expand_systematics(
        self,
        *,
        sample: str,
        variable: str,
        clean_channel: str,
        application: str,
        grouped_variations: Mapping[Any, Sequence[Any]],
        flavored_channel_names: Sequence[str],
    ) -> Tuple[SystematicExpansion, ...]:
        expansions: List[SystematicExpansion] = []
        flavored_channel_names = tuple(flavored_channel_names)

        for group_descriptor, variations in grouped_variations.items():
            hist_keys: Dict[str, Tuple[Tuple[Any, ...], ...]] = {}
            summary_candidates: List[HistogramCombination] = []

            variations_tuple = tuple(variations)
            for variation in variations_tuple:
                syst_label = (
                    (group_descriptor.name, variation.name)
                    if len(variations_tuple) > 1
                    else variation.name
                )
                base_entry = (variable, clean_channel, application, sample, syst_label)
                key_entries: List[Tuple[Any, ...]] = [base_entry]
                if flavored_channel_names:
                    key_entries.extend(
                        (
                            variable,
                            flavored_name,
                            application,
                            sample,
                            syst_label,
                        )
                        for flavored_name in flavored_channel_names
                    )
                hist_keys[variation.name] = tuple(key_entries)

                for entry in key_entries:
                    systematic = entry[4]
                    if isinstance(systematic, tuple):
                        systematic_str = ":".join(str(component) for component in systematic)
                    else:
                        systematic_str = str(systematic)

                    summary_candidates.append(
                        HistogramCombination(
                            sample=str(entry[3]),
                            channel=str(entry[1]),
                            variable=str(entry[0]),
                            application=str(entry[2]),
                            systematic=systematic_str,
                        )
                    )

            expansions.append(
                SystematicExpansion(
                    group_descriptor=group_descriptor,
                    variations=variations_tuple,
                    hist_keys=hist_keys,
                    summary_entries=tuple(summary_candidates),
                )
            )

        return tuple(expansions)


class ExecutorFactory:
    """Create Coffea runners for the configured executor type."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    def create_runner(self) -> Any:
        import coffea.processor as processor
        from coffea.nanoevents import NanoAODSchema

        executor = (self._config.executor or "taskvine").lower()

        remote_environment = topcoffea.modules.remote_environment

        def _build_runner(exec_instance: Any, **runner_kwargs: Any) -> Any:
            return processor.Runner(
                executor=exec_instance,
                schema=NanoAODSchema,
                chunksize=self._config.chunksize,
                maxchunks=self._config.nchunks,
                **runner_kwargs,
            )

        runner_fields = set(getattr(processor.Runner, "__dataclass_fields__", {}))
        runner_kwargs: Dict[str, Any] = {}
        if "nanoevents_factory" in runner_fields:
            runner_kwargs["nanoevents_factory"] = partial(
                nanoevents_factory_from_root, mode="numpy"
            )

        if executor == "futures":
            workers = self._config.nworkers or 1
            exec_instance = build_futures_executor(
                processor,
                workers=workers,
                status=self._config.futures_status,
                tailtimeout=self._config.futures_tail_timeout,
            )

            runner_kwargs.update(
                futures_runner_overrides(
                    runner_fields,
                    memory=self._config.futures_memory,
                    prefetch=self._config.futures_prefetch,
                )
            )
            return _build_runner(exec_instance, **runner_kwargs)

        if executor == "iterative":
            try:
                exec_instance = processor.IterativeExecutor()
            except AttributeError:  # pragma: no cover - depends on coffea build
                exec_instance = processor.iterative_executor()
            return _build_runner(exec_instance, **runner_kwargs)

        if executor == "taskvine":
            port_min, port_max = parse_port_range(self._config.port)
            staging_dir = self._distributed_staging_dir(executor)
            logs_dir = self._executor_logs_dir(executor, staging_dir)
            manager_default = self._manager_name_base(executor)
            manager_name = self._config.manager_name or manager_default
            manager_template = self._config.manager_name_template
            if manager_template is None and manager_name:
                manager_template = f"{manager_name}-{{pid}}"
            environment_file = resolve_environment_file(
                self._config.environment_file,
                remote_environment,
                extra_pip_local={"topeft": ["topeft", "setup.py"]},
                extra_conda=["pyyaml"],
            )

            extra_input_files = self._processor_extra_input_files()
            taskvine_args = build_taskvine_args(
                staging_dir=staging_dir,
                logs_dir=logs_dir,
                manager_name=manager_name,
                manager_name_template=manager_template,
                extra_input_files=extra_input_files,
                resource_monitor=self._config.resource_monitor,
                resources_mode=self._config.resources_mode,
                environment_file=environment_file,
                print_stdout=self._config.taskvine_print_stdout,
                custom_init=taskvine_log_configurator(logs_dir),
            )
            exec_instance = instantiate_taskvine_executor(
                processor,
                taskvine_args,
                port_range=(port_min, port_max),
                negotiate_port=bool(self._config.negotiate_manager_port),
            )

            return _build_runner(
                exec_instance,
                skipbadfiles=True,
                xrootdtimeout=300,
                **runner_kwargs,
            )

        raise ValueError(f"Unknown executor '{executor}'")

    
    def _distributed_staging_dir(self, executor: str) -> Path:
        configured = getattr(self._config, "scratch_dir", None)
        if configured:
            staging = Path(configured).expanduser()
        else:
            base_dir = os.environ.get("TOPEFT_EXECUTOR_STAGING")
            if base_dir:
                staging = Path(base_dir).expanduser()
            else:
                staging = (
                    Path(tempfile.gettempdir())
                    / "topeft"
                    / self._manager_name_base(executor)
                )
        staging.mkdir(parents=True, exist_ok=True)
        return staging

    def _manager_name_base(self, executor: str) -> str:
        user = os.environ.get("USER")
        if not user:
            try:
                user = getpass.getuser()
            except Exception:  # pragma: no cover - best effort fallback
                user = "coffea"
        return f"{user}-{executor}-coffea"

    def _executor_logs_dir(self, executor: str, staging_dir: Path) -> Path:
        if executor == "taskvine":
            logs_dir = staging_dir / "logs" / "taskvine"
        else:
            logs_dir = staging_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _processor_extra_input_files(self) -> list[str]:
        try:
            package = importlib.import_module("analysis.topeft_run2")
        except ImportError:
            return ["analysis_processor.py"]

        package_file = getattr(package, "__file__", None)
        if not package_file:
            return ["analysis_processor.py"]

        package_dir = Path(package_file).resolve().parent
        candidates: set[str] = set()

        for module_path in sorted(package_dir.glob("analysis_processor*.py")):
            if module_path.name == "__init__.py":
                continue
            candidates.add(module_path.relative_to(package_dir).as_posix())

        helpers_dir = package_dir / "analysis_processor_helpers"
        if helpers_dir.is_dir():
            for helper_path in sorted(helpers_dir.rglob("*.py")):
                if helper_path.name == "__init__.py":
                    continue
                candidates.add(helper_path.relative_to(package_dir).as_posix())

        if not candidates:
            candidates.add("analysis_processor.py")

        return sorted(candidates)

class RunWorkflow:
    def __init__(
        self,
        *,
        config: RunConfig,
        metadata: Mapping[str, Any],
        sample_loader: SampleLoader,
        channel_planner: ChannelPlanner,
        histogram_planner: HistogramPlanner,
        executor_factory: ExecutorFactory,
        weight_variations: Sequence[str],
        metadata_path: str,
    ) -> None:
        self._config = config
        self._metadata = metadata
        self._sample_loader = sample_loader
        self._channel_planner = channel_planner
        self._histogram_planner = histogram_planner
        self._executor_factory = executor_factory
        self._weight_variations = list(weight_variations)
        self._metadata_path = metadata_path

    def _log_task_submission(self, task: HistogramTask) -> None:
        """Emit a concise log describing the histogram combinations for ``task``."""

        if self._config.executor != "futures" or not getattr(self._config, "log_tasks", False):
            return

        combination_labels: List[str] = []
        for entries in task.hist_keys.values():
            for var, channel, application, sample, systematic in entries:
                if isinstance(systematic, tuple):
                    systematic_label = ":".join(str(component) for component in systematic)
                else:
                    systematic_label = str(systematic)

                combination_labels.append(
                    "({0}, {1}, {2}, {3}, {4})".format(
                        str(sample),
                        str(channel),
                        str(var),
                        str(application),
                        systematic_label,
                    )
                )

        if not combination_labels:
            return

        unique_labels = unique_preserving_order(combination_labels)
        logger.info(
            "[futures] submitting histogram task for %s",
            ", ".join(unique_labels),
        )

    def _log_variation_recap(
        self,
        *,
        task_index: int,
        total_tasks: int,
        task: HistogramTask,
        summary_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        """Emit a single INFO log describing which variations actually ran."""

        def _unique_strings(values: Iterable[Any]) -> List[str]:
            normalized = []
            for value in values:
                if value in (None, ""):
                    continue
                normalized.append(str(value))
            return unique_preserving_order(normalized)

        def _flatten(entry_key: str) -> List[str]:
            return _unique_strings(
                value
                for entry in summary_entries
                for value in entry.get(entry_key, ())
            )

        def _format(values: Sequence[str]) -> str:
            return "[" + ", ".join(values) + "]" if values else "[]"

        if not summary_entries:
            logger.info(
                "Completed histogram task %d/%d: sample=%s channel=%s variable=%s application=%s (no variation summary returned)",
                task_index,
                total_tasks,
                task.sample,
                task.clean_channel,
                task.variable,
                task.application,
            )
            return

        requested_variations = _unique_strings(
            entry.get("requested_name") for entry in summary_entries
        )
        object_variations = _unique_strings(
            entry.get("object_variation") for entry in summary_entries
        )
        histogram_labels = _unique_strings(
            entry.get("histogram_label") for entry in summary_entries
        )
        executed_weight_variations = _flatten("executed_weight_variations")
        requested_weight_variations = _flatten("requested_weight_variations")
        skipped_weights = [
            weight
            for weight in requested_weight_variations
            if weight not in set(executed_weight_variations)
        ]

        logger.info(
            (
                "Completed histogram task %d/%d: sample=%s channel=%s variable=%s application=%s "
                "requested_variations=%s object_variations=%s executed_weight_variations=%s "
                "histogram_labels=%s skipped_weight_variations=%s"
            ),
            task_index,
            total_tasks,
            task.sample,
            task.clean_channel,
            task.variable,
            task.application,
            _format(requested_variations),
            _format(object_variations),
            _format(executed_weight_variations),
            _format(histogram_labels),
            _format(_unique_strings(skipped_weights)),
        )
    def run(self) -> None:
        from topeft.modules.systematics import SystematicsHelper
        from . import analysis_processor

        self._validate_config()

        sample_specs = self._sample_loader.collect(self._config.json_files)
        samplesdict, flist = self._sample_loader.load(sample_specs)

        if self._config.do_systs:
            self._ensure_weight_variations(samplesdict)

        nevts_total = sum(sample["nEvents"] for sample in samplesdict.values())

        if self._config.pretend:
            logger.info("Pretend mode active; skipping execution after configuration phase.")
            return

        self._ensure_wilson_coefficients(samplesdict)

        golden_jsons = self._metadata.get("golden_jsons", {}) if self._metadata else {}
        if not golden_jsons:
            raise ValueError(
                f"golden_jsons mapping missing from metadata ({self._metadata_path})."
            )

        var_defs = self._metadata.get("variables")
        if not isinstance(var_defs, Mapping):
            raise TypeError(
                "metadata['variables'] must be a mapping of histogram definitions "
                f"(source: {self._metadata_path})"
            )

        sample_years = {
            str(samplesdict[sample_name]["year"])
            for sample_name in samplesdict
            if "year" in samplesdict[sample_name]
        }

        active_features = set(self._channel_planner.active_features)
        systematics_helper = SystematicsHelper(
            self._metadata,
            sample_years=sample_years,
            tau_analysis="requires_tau" in active_features,
        )

        histogram_plan = self._histogram_planner.plan(samplesdict, systematics_helper)

        self._emit_histogram_summary(histogram_plan)

        runner = self._executor_factory.create_runner()

        output: Dict[str, Any] = {}
        ecut_threshold = self._config.ecut if self._config.ecut is None else float(self._config.ecut)

        tstart = time.time()

        total_tasks = len(histogram_plan.tasks)
        for idt, task in enumerate(histogram_plan.tasks):
            logger.info(
                "Starting histogram task %d/%d: sample=%s channel=%s variable=%s application=%s variations=%d",
                idt + 1,
                total_tasks,
                task.sample,
                task.clean_channel,
                task.variable,
                task.application,
                len(task.variations),
            )
            sample_dict = samplesdict[task.sample]
            sample_files = list(flist[task.sample])
            if self._config.executor == "futures":
                prefetch_files = self._config.futures_prefetch
                if prefetch_files is None or prefetch_files <= 0:
                    sample_flist = sample_files
                else:
                    sample_flist = sample_files[: int(prefetch_files)]
            else:
                sample_flist = sample_files

            channel_dict = task.channel_metadata
            if not channel_dict:
                continue

            if task.clean_channel != "3l_m_offZ_1b_2j":
                logging.info("Skipping task for channel %s", task.clean_channel)
                continue

            if self._config.debug_logging:
                logger.info("Channel %s metadata: %s", task.clean_channel, channel_dict)
                logger.info("Task detail: %s", task)

            golden_json_path = None
            if sample_dict.get("isData"):
                year_key = str(sample_dict["year"])
                try:
                    golden_json_relpath = golden_jsons[year_key]
                except KeyError as exc:
                    raise ValueError(
                        f"No golden JSON configured for data year '{year_key}' in "
                        f"{self._metadata_path}."
                    ) from exc
                golden_json_path = topcoffea_path(golden_json_relpath)
                if not os.path.exists(golden_json_path):
                    raise FileNotFoundError(
                        f"Golden JSON file '{golden_json_path}' for year '{year_key}' was not found."
                    )

            processor_instance = analysis_processor.AnalysisProcessor(
                sample_dict,
                self._config.wc_list,
                hist_keys=task.hist_keys,
                var_info=task.variable_info,
                ecut_threshold=ecut_threshold,
                do_errors=self._config.do_errors,
                split_by_lepton_flavor=self._config.split_lep_flavor,
                channel_dict=channel_dict,
                golden_json_path=golden_json_path,
                systematic_variations=task.variations,
                available_systematics=task.available_systematics,
                metadata_path=self._metadata_path,
                executor_mode=self._config.executor,
                debug_logging=bool(self._config.debug_logging),
            )

            import coffea.processor as processor

            if not isinstance(processor_instance, processor.ProcessorABC):
                raise TypeError(
                    "AnalysisProcessor is not an instance of coffea.processor.ProcessorABC. "
                    f"Active coffea.processor module: {getattr(processor, '__file__', 'unknown')}"
                )

            self._log_task_submission(task)

            attempt = 0
            max_retries = 0
            if (
                self._config.executor == "futures"
                and self._config.futures_retries
            ):
                max_retries = max(int(self._config.futures_retries), 0)
            retry_wait = 0.0
            if (
                self._config.executor == "futures"
                and self._config.futures_retry_wait is not None
            ):
                retry_wait = max(float(self._config.futures_retry_wait), 0.0)

            while True:
                try:
                    out = runner(
                        {task.sample: sample_flist},
                        processor_instance,
                        self._config.treename,
                        # coffea Runner.__call__ expects (fileset, processor_instance, treename)
                    )
                except Exception as exc:
                    if attempt >= max_retries:
                        raise
                    attempt += 1
                    logger.warning(
                        "[futures] task for %s failed (attempt %d/%d): %s",
                        task.sample,
                        attempt,
                        max_retries,
                        exc,
                    )
                    if retry_wait > 0:
                        time.sleep(retry_wait)
                    continue
                else:
                    break
            summary_payload = out.pop(
                analysis_processor.AnalysisProcessor.VARIATION_SUMMARY_KEY, ()
            )
            self._log_variation_recap(
                task_index=idt + 1,
                total_tasks=total_tasks,
                task=task,
                summary_entries=summary_payload or (),
            )
            output.update(out)

        dt = time.time() - tstart

        if self._config.executor == "taskvine" and nevts_total:
            logger.info(
                "Processed %d events in %.2f seconds (%.2f evts/sec)",
                nevts_total,
                dt,
                (nevts_total / dt) if dt else 0.0,
            )

        if self._config.executor == "futures":
            logger.info(
                "Processing time: %.2f s with %d workers (%.2f s cpu overall)",
                dt,
                self._config.nworkers,
                dt * self._config.nworkers,
            )

        self._store_output(output)

    def _emit_histogram_summary(self, plan: HistogramPlan) -> None:
        """Print the planned histogram combinations based on the configured verbosity."""

        verbosity = getattr(self._config, "summary_verbosity", "brief") or "brief"
        verbosity = str(verbosity).strip().lower()
        if verbosity not in {"none", "brief", "full"}:
            verbosity = "brief"

        if not plan.summary or verbosity == "none":
            return

        samples = unique_preserving_order(str(entry.sample) for entry in plan.summary)
        channel_pairs = unique_preserving_order(
            (str(entry.channel), str(entry.application)) for entry in plan.summary
        )
        variables = unique_preserving_order(str(entry.variable) for entry in plan.summary)
        systematics = unique_preserving_order(str(entry.systematic) for entry in plan.summary)

        def _format_values(values: Sequence[str]) -> str:
            return ", ".join(values) if values else "None"

        logger.info("Planned histogram summary:")
        logger.info("- Samples: %s", _format_values(samples))
        logger.info(
            "- Channels & applications: %s",
            _format_values([f"{channel} ({application})" for channel, application in channel_pairs]),
        )
        logger.info("- Variables: %s", _format_values(variables))
        logger.info("- Systematics: %s", _format_values(systematics))

        if verbosity == "brief":
            return

        headers = ("Sample", "Channel", "Variable", "Application", "Systematic")
        rows = [
            (
                str(entry.sample),
                str(entry.channel),
                str(entry.variable),
                str(entry.application),
                str(entry.systematic),
            )
            for entry in plan.summary
        ]

        column_widths = [
            max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
            for idx, header in enumerate(headers)
        ]
        row_format = "  ".join(f"{{:{width}}}" for width in column_widths)

        if getattr(self._config, "split_lep_flavor", False):
            logger.info(
                "Note: lepton-flavor channels reuse the processor task configured for their base channel when flavor splitting is enabled."
            )

        logger.info("Planned histogram combinations:")
        logger.info(row_format.format(*headers))
        logger.info("  ".join("-" * width for width in column_widths))
        for row in rows:
            logger.info(row_format.format(*row))

        summary_payload = [asdict(entry) for entry in plan.summary]

        logger.info("Structured summary:")
        try:
            import yaml  # type: ignore

            dumped = yaml.safe_dump(summary_payload, sort_keys=False).strip()
            logger.info("%s", dumped or "[]")
        except Exception:  # pragma: no cover - optional dependency
            logger.info("%s", json.dumps(summary_payload, indent=2))

    def _validate_config(self) -> None:
        if self._config.executor not in LST_OF_KNOWN_EXECUTORS:
            raise Exception(
                f'The "{self._config.executor}" executor is not known. Please specify an executor from the known executors '
                f"({LST_OF_KNOWN_EXECUTORS}). Exiting."
            )
        if self._config.do_renormfact_envelope:
            if not self._config.do_systs:
                raise Exception(
                    "Error: Cannot specify do_renormfact_envelope if we are not including systematics."
                )
            if not self._config.do_np:
                raise Exception(
                    "Error: Cannot specify do_renormfact_envelope if we have not already done the integration across the appl axis that occurs in the data driven estimator step."
                )
        if self._config.test:
            if self._config.executor == "futures":
                self._config.nchunks = 2
                self._config.chunksize = 100
                self._config.nworkers = 1
                logger.info(
                    "Running a fast futures test with %d workers, %d chunks of %d events",
                    self._config.nworkers,
                    self._config.nchunks,
                    self._config.chunksize,
                )
            elif self._config.executor == "iterative":
                self._config.nchunks = 2
                self._config.chunksize = 100
                logger.info(
                    "Running a fast iterative test with %d chunks of %d events",
                    self._config.nchunks,
                    self._config.chunksize,
                )
            else:
                raise Exception(
                    f'The "test" option is not set up to work with the {self._config.executor} executor. Exiting.'
                )

    def _ensure_weight_variations(self, samplesdict: Mapping[str, Mapping[str, Any]]) -> None:
        missing_default_variations = [
            variation
            for variation in DEFAULT_WEIGHT_VARIATIONS
            if variation not in self._weight_variations
        ]
        if missing_default_variations:
            warnings.warn(
                "Default sum-of-weights variations will not be processed: "
                + ", ".join(missing_default_variations),
                RuntimeWarning,
            )

        for sample_info in samplesdict.values():
            if sample_info.get("isData"):
                continue
            for wgt_var in self._weight_variations:
                if wgt_var not in sample_info:
                    raise Exception(f'Missing weight variation "{wgt_var}".')

    def _ensure_wilson_coefficients(self, samplesdict: Mapping[str, Mapping[str, Any]]) -> None:
        if self._config.wc_list:
            return
        for sample_info in samplesdict.values():
            for wc in sample_info.get("WCnames", []) or []:
                if wc not in self._config.wc_list:
                    self._config.wc_list.append(wc)
        if self._config.wc_list:
            if len(self._config.wc_list) == 1:
                wc_print = self._config.wc_list[0]
            elif len(self._config.wc_list) == 2:
                wc_print = " and ".join(self._config.wc_list)
            else:
                wc_print = ", ".join(self._config.wc_list[:-1]) + ", and " + self._config.wc_list[-1]
            logger.info("Wilson coefficients: %s", wc_print)
        else:
            logger.info("No Wilson coefficients specified")

    def _store_output(self, output: Mapping[str, Any]) -> None:
        if not os.path.isdir(self._config.outpath):
            os.system("mkdir -p %s" % self._config.outpath)
        out_pkl_file = os.path.join(self._config.outpath, self._config.outname + ".pkl.gz")

        serialised_output = normalise_runner_output(output)
        if isinstance(serialised_output, Mapping):
            total_bins, filled_bins = tuple_dict_stats(serialised_output)
            if total_bins:
                fill_fraction = (100 * filled_bins / total_bins)
                logger.info("Filled %.0f bins, nonzero bins: %1.1f %%", total_bins, fill_fraction)

        logger.info("Saving output in %s", out_pkl_file)
        with gzip.open(out_pkl_file, "wb") as fout:
            import cloudpickle

            cloudpickle.dump(serialised_output, fout)
        logger.info("Finished writing %s", out_pkl_file)

        if self._config.do_np:
            logger.info("Starting nonprompt estimation")
            out_pkl_file_name_np = os.path.join(
                self._config.outpath, self._config.outname + "_np.pkl.gz"
            )
            from topeft.modules.dataDrivenEstimation import DataDrivenProducer

            ddp = DataDrivenProducer(out_pkl_file, out_pkl_file_name_np)
            logger.info("Saving nonprompt output in %s", out_pkl_file_name_np)
            ddp.dumpToPickle()
            logger.info("Finished writing nonprompt output")
            if self._config.do_renormfact_envelope:
                logger.info("Applying renorm/fact envelope to nonprompt output")
                from topeft.modules.get_renormfact_envelope import (
                    get_renormfact_envelope,
                )

                dict_of_histos = topcoffea_utils.get_hist_from_pkl(
                    out_pkl_file_name_np, allow_empty=False
                )
                dict_of_histos_after_applying_envelope = get_renormfact_envelope(
                    dict_of_histos
                )
                topcoffea_utils.dump_to_pkl(
                    out_pkl_file_name_np, dict_of_histos_after_applying_envelope
                )


def run_workflow(config: RunConfig) -> None:
    """Convenience wrapper mirroring the behaviour of ``run_analysis.py``."""

    import yaml
    from topeft.modules.channel_metadata import ChannelMetadataHelper

    metadata_source = config.metadata_path
    if not metadata_source:
        raise ValueError(
            "RunConfig.metadata_path is not set. The scenario registry or options profile "
            "must select a metadata bundle before launching run_workflow."
        )
    candidate_path = Path(metadata_source).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    try:
        metadata_file = candidate_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Metadata file '{metadata_source}' could not be found."
        ) from exc

    config.metadata_path = str(metadata_file)

    with metadata_file.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    weight_variations = weight_variations_from_metadata(metadata, DEFAULT_WEIGHT_VARIATIONS)
    sample_loader = SampleLoader(
        default_prefix=config.prefix, weight_variables=weight_variations
    )

    channels_metadata = metadata.get("channels")
    if not channels_metadata:
        raise ValueError(
            f"Channel metadata is missing from the metadata YAML ({metadata_file})."
        )

    channel_helper = ChannelMetadataHelper(channels_metadata)
    scenario_names = unique_preserving_order(config.scenario_names)
    if not scenario_names:
        scenario_names = [DEFAULT_SCENARIO_NAME]
    config.scenario_names = list(scenario_names)

    channel_planner = ChannelPlanner(
        channel_helper,
        skip_sr=config.skip_sr,
        skip_cr=config.skip_cr,
        scenario_names=config.scenario_names,
    )

    var_defs = metadata.get("variables")
    if not isinstance(var_defs, Mapping):
        raise TypeError("metadata['variables'] must be a mapping of histogram definitions")

    histogram_planner = HistogramPlanner(
        config=config, variable_definitions=var_defs, channel_planner=channel_planner
    )

    executor_factory = ExecutorFactory(config)

    workflow = RunWorkflow(
        config=config,
        metadata=metadata,
        sample_loader=sample_loader,
        channel_planner=channel_planner,
        histogram_planner=histogram_planner,
        executor_factory=executor_factory,
        weight_variations=weight_variations,
        metadata_path=str(metadata_file),
    )
    workflow.run()


__all__ = [
    "ChannelPlanner",
    "HistogramPlanner",
    "HistogramPlan",
    "HistogramCombination",
    "HistogramTask",
    "ExecutorFactory",
    "RunWorkflow",
    "run_workflow",
    "normalize_jet_category",
]
