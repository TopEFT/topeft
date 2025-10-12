"""Workflow utilities for orchestrating Run 2 analyses.

This module provides a small collection of helper classes that encapsulate the
core steps performed by :mod:`analysis.topeft_run2.run_analysis`.  The helpers
are designed to be lightweight wrappers around the existing functionality while
making the orchestration of a run easier to understand and reuse from Python
code.  The main entry point is :class:`RunWorkflow` together with the
``run_workflow`` convenience function.
"""

from __future__ import annotations

import gzip
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TYPE_CHECKING

from .run_analysis_helpers import (
    DEFAULT_WEIGHT_VARIATIONS,
    RunConfig,
    SampleLoader,
    unique_preserving_order,
    weight_variations_from_metadata,
)

DEFAULT_SCENARIO_NAME = "TOP_22_006"

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from topeft.modules.channel_metadata import ChannelMetadataHelper
    from topeft.modules.systematics import SystematicsHelper

LST_OF_KNOWN_EXECUTORS = ["futures", "work_queue", "taskvine"]


class ChannelPlanner:
    """Resolve channel metadata into lookups used during processing."""

    def __init__(
        self,
        channel_helper: "ChannelMetadataHelper",
        *,
        skip_sr: bool = False,
        skip_cr: bool = False,
        scenario_names: Optional[Sequence[str]] = None,
        required_features: Optional[Sequence[str]] = None,
    ) -> None:
        self._channel_helper = channel_helper
        self._skip_sr = bool(skip_sr)
        self._skip_cr = bool(skip_cr)
        self._scenario_names = list(scenario_names or [])
        self._required_features = list(required_features or [])

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
                            return {
                                "jet_selection": jet_key,
                                "chan_def_lst": region.to_legacy_list(),
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
        required_feature_set = set(self._required_features)

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

        scenario_names = list(self._scenario_names)
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
            raise ValueError("No channel groups selected. Please specify a scenario or feature tag")

        self._sr_groups = tuple(sr_groups)
        self._cr_groups = tuple(cr_groups)
        self._active_features = tuple(sorted(active_features))
        return self._sr_groups, self._cr_groups, self._active_features


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
class HistogramPlan:
    """Collection of histogram tasks computed for a workflow."""

    tasks: List[HistogramTask]
    histogram_names: Sequence[str]


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

        channel_map_mc = self._channel_planner.channel_app_map(is_data=False)
        channel_map_data = self._channel_planner.channel_app_map(is_data=True)

        analysis_processor_module = None

        for sample, sample_info in samplesdict.items():
            ch_map = channel_map_data if sample_info.get("isData") else channel_map_mc
            grouped_variations = systematics_helper.grouped_variations_for_sample(
                sample_info, include_systematics=self._config.do_systs
            )
            sample_type_key = "data" if sample_info.get("isData") else "mc"
            available_systematics = available_systematics_by_sample_type[sample_type_key]

            for var in hist_lst:
                var_info = dict(self._var_defs[var])
                for clean_ch, appl_list in ch_map.items():
                    for appl in appl_list:
                        try:
                            channel_metadata = self._channel_planner.build_channel_dict(
                                clean_ch,
                                appl,
                                is_data=sample_info.get("isData", False),
                            )
                        except ValueError:
                            continue

                        if not channel_metadata:
                            continue

                        whitelist = tuple(channel_metadata.get("channel_var_whitelist") or ())
                        blacklist = set(channel_metadata.get("channel_var_blacklist") or ())

                        if whitelist and var not in whitelist:
                            continue
                        if var in blacklist:
                            continue

                        flavored_channel_names: Tuple[str, ...] = ()
                        if self._config.split_lep_flavor:
                            if analysis_processor_module is None:
                                from . import (
                                    analysis_processor as analysis_processor_module,
                                )
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
                            flavored_channel_names = tuple(flavored_candidates)

                        for group_descriptor, variations in grouped_variations.items():
                            hist_keys: Dict[str, Tuple[Tuple[Any, ...], ...]] = {}
                            for variation in variations:
                                syst_label = (
                                    (group_descriptor.name, variation.name)
                                    if len(variations) > 1
                                    else variation.name
                                )
                                base_entry = (var, clean_ch, appl, sample, syst_label)
                                key_entries: List[Tuple[Any, ...]] = [base_entry]
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

                            tasks.append(
                                HistogramTask(
                                    sample=sample,
                                    variable=var,
                                    clean_channel=clean_ch,
                                    application=appl,
                                    group_descriptor=group_descriptor,
                                    variations=tuple(variations),
                                    hist_keys=hist_keys,
                                    variable_info=var_info,
                                    available_systematics=available_systematics,
                                    channel_metadata=channel_metadata,
                                )
                            )

        return HistogramPlan(tasks=tasks, histogram_names=hist_lst)


class ExecutorFactory:
    """Create Coffea runners for the configured executor type."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    def create_runner(self) -> Any:
        from coffea import processor
        from coffea.nanoevents import NanoAODSchema
        from topcoffea.modules import remote_environment

        executor = self._config.executor

        if executor == "futures":
            exec_instance = processor.futures_executor(workers=self._config.nworkers)
            return processor.Runner(
                exec_instance,
                schema=NanoAODSchema,
                chunksize=self._config.chunksize,
                maxchunks=self._config.nchunks,
            )

        if executor in {"work_queue", "taskvine"}:
            port_min, port_max = self._parse_port_range(self._config.port)
            executor_args = {
                "manager_name": f"{os.environ['USER'].capitalize()}-{executor}-coffea",
                "port": [port_min, port_max],
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
                "compression": 8,
                "resource_monitor": "measure",
                "resources_mode": "auto",
                "filepath": f"/tmp/{os.environ['USER']}",
                "chunks_per_accum": 25,
                "chunks_accum_in_mem": 2,
                "fast_terminate_workers": 0,
                "verbose": True,
                "print_stdout": False,
            }

            if executor == "work_queue":
                exec_instance = processor.WorkQueueExecutor(**executor_args)
                return processor.Runner(
                    exec_instance,
                    schema=NanoAODSchema,
                    chunksize=self._config.chunksize,
                    maxchunks=self._config.nchunks,
                    skipbadfiles=False,
                    xrootdtimeout=180,
                )

            try:
                exec_instance = processor.TaskVineExecutor(**executor_args)
            except AttributeError as exc:  # pragma: no cover - depends on coffea build
                raise RuntimeError("TaskVineExecutor not available.") from exc
            return processor.Runner(
                exec_instance,
                schema=NanoAODSchema,
                chunksize=self._config.chunksize,
                maxchunks=self._config.nchunks,
                skipbadfiles=True,
                xrootdtimeout=300,
            )

        raise ValueError(f"Unknown executor '{executor}'")

    @staticmethod
    def _parse_port_range(port: str) -> Tuple[int, int]:
        try:
            tokens = [int(token) for token in port.split("-") if token]
        except ValueError as exc:
            raise ValueError("Port specification must be an integer or range") from exc
        if not tokens:
            raise ValueError("At least one port value should be specified.")
        if len(tokens) > 2:
            raise ValueError("More than one port range was specified.")
        if len(tokens) == 1:
            tokens.append(tokens[0])
        return tokens[0], tokens[1]


class RunWorkflow:
    """High-level orchestrator for running the Coffea processor."""

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
    ) -> None:
        self._config = config
        self._metadata = metadata
        self._sample_loader = sample_loader
        self._channel_planner = channel_planner
        self._histogram_planner = histogram_planner
        self._executor_factory = executor_factory
        self._weight_variations = list(weight_variations)

    def run(self) -> None:
        from topeft.modules.systematics import SystematicsHelper
        from topcoffea.modules.paths import topcoffea_path
        from . import analysis_processor

        self._validate_config()

        sample_specs = self._sample_loader.collect(self._config.json_files)
        samplesdict, flist = self._sample_loader.load(sample_specs)

        if self._config.do_systs:
            self._ensure_weight_variations(samplesdict)

        nevts_total = sum(sample["nEvents"] for sample in samplesdict.values())

        if self._config.pretend:
            print("pretending...")
            return

        self._ensure_wilson_coefficients(samplesdict)

        golden_jsons = self._metadata.get("golden_jsons", {}) if self._metadata else {}
        if not golden_jsons:
            raise ValueError("golden_jsons mapping missing from metadata.")

        var_defs = self._metadata.get("variables")
        if not isinstance(var_defs, Mapping):
            raise TypeError("metadata['variables'] must be a mapping of histogram definitions")

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

        runner = self._executor_factory.create_runner()

        output: Dict[str, Any] = {}
        ecut_threshold = self._config.ecut if self._config.ecut is None else float(self._config.ecut)

        tstart = time.time()

        for task in histogram_plan.tasks:
            sample_dict = samplesdict[task.sample]
            sample_flist = flist[task.sample][:1]

            channel_dict = task.channel_metadata
            if not channel_dict:
                continue

            golden_json_path = None
            if sample_dict.get("isData"):
                year_key = str(sample_dict["year"])
                try:
                    golden_json_relpath = golden_jsons[year_key]
                except KeyError as exc:
                    raise ValueError(
                        f"No golden JSON configured for data year '{year_key}'."
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
            )

            out = runner({task.sample: sample_flist}, self._config.treename, processor_instance)
            output.update(out)

        dt = time.time() - tstart

        if self._config.executor in ["work_queue", "taskvine"] and nevts_total:
            print(
                "Processed {} events in {} seconds ({:.2f} evts/sec).".format(
                    nevts_total, dt, nevts_total / dt if dt else 0
                )
            )

        if self._config.executor == "futures":
            print(
                "Processing time: %1.2f s with %i workers (%.2f s cpu overall)"
                % (dt, self._config.nworkers, dt * self._config.nworkers)
            )

        self._store_output(output)

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
                print(
                    "Running a fast test with %i workers, %i chunks of %i events"
                    % (self._config.nworkers, self._config.nchunks, self._config.chunksize)
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
            print(f"Wilson Coefficients: {wc_print}.")
        else:
            print("No Wilson coefficients specified")

    def _store_output(self, output: Mapping[str, Any]) -> None:
        if not os.path.isdir(self._config.outpath):
            os.system("mkdir -p %s" % self._config.outpath)
        out_pkl_file = os.path.join(self._config.outpath, self._config.outname + ".pkl.gz")
        print(f"\nSaving output in {out_pkl_file}...")
        with gzip.open(out_pkl_file, "wb") as fout:
            import cloudpickle

            cloudpickle.dump(output, fout)
        print("Done!")

        if self._config.do_np:
            print("\nDoing the nonprompt estimation...")
            out_pkl_file_name_np = os.path.join(
                self._config.outpath, self._config.outname + "_np.pkl.gz"
            )
            from topeft.modules.dataDrivenEstimation import DataDrivenProducer

            ddp = DataDrivenProducer(out_pkl_file, out_pkl_file_name_np)
            print(f"Saving output in {out_pkl_file_name_np}...")
            ddp.dumpToPickle()
            print("Done!")
            if self._config.do_renormfact_envelope:
                print("\nDoing the renorm. fact. envelope calculation...")
                from topeft.modules.get_renormfact_envelope import (
                    get_renormfact_envelope,
                )
                import topcoffea.modules.utils as utils

                dict_of_histos = utils.get_hist_from_pkl(
                    out_pkl_file_name_np, allow_empty=False
                )
                dict_of_histos_after_applying_envelope = get_renormfact_envelope(
                    dict_of_histos
                )
                utils.dump_to_pkl(
                    out_pkl_file_name_np, dict_of_histos_after_applying_envelope
                )


def run_workflow(config: RunConfig) -> None:
    """Convenience wrapper mirroring the behaviour of ``run_analysis.py``."""

    import yaml
    from topeft.modules.channel_metadata import ChannelMetadataHelper
    from topeft.modules.paths import topeft_path

    metadata_path = topeft_path("params/metadata.yml")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    weight_variations = weight_variations_from_metadata(metadata, DEFAULT_WEIGHT_VARIATIONS)
    sample_loader = SampleLoader(
        default_prefix=config.prefix, weight_variables=weight_variations
    )

    channels_metadata = metadata.get("channels")
    if not channels_metadata:
        raise ValueError("Channel metadata is missing from params/metadata.yml")

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
        required_features=config.channel_feature_tags,
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
    )
    workflow.run()


__all__ = [
    "ChannelPlanner",
    "HistogramPlanner",
    "HistogramPlan",
    "HistogramTask",
    "ExecutorFactory",
    "RunWorkflow",
    "run_workflow",
    "normalize_jet_category",
]

