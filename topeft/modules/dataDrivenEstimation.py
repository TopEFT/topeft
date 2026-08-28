import argparse
import copy
import gzip
import warnings
from collections import defaultdict

import cloudpickle
import numpy as np
from topcoffea.modules.hist_utils import iterate_hist_from_pkl

from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    data_driven_product_error,
    FLIPS_OUTPUT_ARTIFACT_KIND,
    generated_process_name,
    NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
    NONPROMPT_OUTPUT_ARTIFACT_KIND,
    parse_process_name,
    TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS,
    validate_requested_product_input,
)
from topeft.modules.nonprompt_policy import (
    certify_active_nonprompt_policy,
    nonprompt_policy_error,
)
from topeft.modules.histogram_artifact import (
    FLIPS_APPLICATION_REGION,
    NONPROMPT_APPLICATION_REGIONS,
    _PRODUCER_CONTEXT_TOKEN,
    derive_data_driven_applicability,
    lineage_input_from_sidecar,
    validate_histogram_artifact,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import (
    EFT_NOMINAL_SUFFIX,
    SCALAR_NOMINAL_SUFFIX,
    SUMW2_SUFFIX,
    evaluate_eft_histogram_at_wc,
)


def data_driven_product_for_application_region(application_region):
    """Return the maintained product consuming one application region."""

    region = str(application_region)
    if region == FLIPS_APPLICATION_REGION:
        return "flips"
    if region in NONPROMPT_APPLICATION_REGIONS:
        return "nonprompt"
    return None


class _producer_transformation_context(dict):
    """Marker for transformation evidence emitted only by DataDrivenProducer."""

    _producer_context_token = _PRODUCER_CONTEXT_TOKEN


class DataDrivenProducer:
    def __init__(
        self,
        inputHist,
        outputName,
        iterator_mode=False,
        dd_report=False,
        artifact_kind="nonprompt_output",
    ):
        self._input_source = inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.iterator_mode = iterator_mode
        self._dd_report_enabled = dd_report
        if artifact_kind not in TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS:
            raise RuntimeError(f"Unknown data-driven artifact kind {artifact_kind!r}.")
        self._artifact_kind = artifact_kind
        self._resolved_input_sidecar = None
        self._nonprompt_policy_migration = None
        self._dd_report_by_key = {} if dd_report else None
        self._input_artifact_validation = None
        if self._is_histogram_path(self._input_source):
            self._input_artifact_validation = validate_histogram_artifact(
                self._input_source
            )
            if self._input_artifact_validation["schema"] == "legacy_uniform":
                warnings.warn(
                    "Transforming a legacy uniform histogram PKL without a "
                    "schema-v2 sidecar; the output remains on the explicit legacy "
                    "path and no schema-v2 sidecar will be synthesized.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self._input_artifact_validation["metadata"]:
                resolution = validate_requested_product_input(
                    self._input_artifact_validation["metadata"],
                    artifact_kind=artifact_kind,
                )
                self._resolved_input_sidecar = resolution["effective_sidecar"]
                self._nonprompt_policy_migration = resolution["migration"]
        self._transformation_role_context = self._initialize_transformation_role_context()
        self._prompt_subtraction_execution_by_family = (
            self._initialize_prompt_subtraction_execution_context()
        )
        self._prompt_subtraction_coverage_validated = False
        (
            self._eft_prompt_processes_by_family,
            self._eft_prompt_projection_context,
        ) = self._initialize_eft_prompt_projection_context()
        self._eft_prompt_projections = self._build_eft_prompt_projections()
        if not self.iterator_mode:
            self.DDFakes()

    @staticmethod
    def _is_histogram_path(candidate):
        return isinstance(candidate, str) and candidate.endswith(('.pkl.gz', '.pkl'))

    def _iter_input_histograms(self):
        source = self._input_source
        if self._is_histogram_path(source):
            yield from iterate_hist_from_pkl(source, allow_empty=True)
            return

        if hasattr(source, 'items'):
            yield from source.items()
            return

        yield from source

    def _input_histogram_keys(self):
        source = self._input_source
        if self._is_histogram_path(source):
            return tuple(
                key
                for key, _histogram in iterate_hist_from_pkl(
                    source, allow_empty=True, materialize=False
                )
            )
        if hasattr(source, "keys"):
            return tuple(source.keys())
        raise TypeError(
            "Streaming nonprompt input must be a histogram path or keyed mapping."
        )

    def _initialize_transformation_role_context(self):
        if self._input_artifact_validation is None:
            return None
        input_sidecar = self._resolved_input_sidecar
        if not input_sidecar or "sumw2_content_manifest" not in input_sidecar:
            return None
        families = {}
        for family, manifest in input_sidecar["sumw2_content_manifest"][
            "families"
        ].items():
            families[family] = {
                "source_scalar_processes": list(
                    manifest["scalar_nominal_processes"]
                ),
                "source_eft_processes": list(manifest["eft_nominal_processes"]),
                "retained_scalar_processes": [],
                "retained_eft_processes": [],
                "generated_nonprompt_processes": [],
                "generated_flips_processes": [],
                "source_application_regions": None,
                "applicable_products": None,
            }
        return families

    @staticmethod
    def _build_prompt_subtraction_execution_plan(
        selected_processes,
        explicit_exclusions,
        family_inventories,
    ):
        selected = {str(process) for process in selected_processes}
        excluded = {str(process) for process in explicit_exclusions}
        plans = {}
        for family, inventory in family_inventories.items():
            scalar = {str(process) for process in inventory["scalar"]}
            eft = {str(process) for process in inventory["eft"]}
            sumw2 = {str(process) for process in inventory["sumw2"]}
            present = scalar | eft | sumw2
            selected_present = selected & present
            ambiguous = selected_present & scalar & eft
            scalar_route = selected_present & scalar
            eft_route = selected_present & eft
            unhandled = selected_present - scalar_route - eft_route
            if ambiguous:
                raise RuntimeError(
                    f"Family {family!r} has selected prompt process(es) with "
                    "ambiguous scalar and EFT nominal representations: "
                    + ", ".join(sorted(ambiguous))
                )
            if unhandled:
                raise RuntimeError(
                    f"Family {family!r} has selected prompt process(es) present "
                    "without a supported scalar or EFT nominal representation: "
                    + ", ".join(sorted(unhandled))
                )
            plans[family] = {
                "selected_processes": selected,
                "present_processes": present,
                "selected_present_processes": selected_present,
                "selected_absent_processes": selected - present,
                "scalar_processes": scalar_route,
                "eft_processes": eft_route,
                "excluded_processes": excluded & present,
                "ambiguous_processes": ambiguous,
                "unhandled_processes": unhandled,
                "executed_processes": set(),
                "nonprompt_applicable": False,
            }
        return plans

    def _initialize_prompt_subtraction_execution_context(self):
        if self._artifact_kind not in {
            NONPROMPT_OUTPUT_ARTIFACT_KIND,
            NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
        }:
            return {}
        if self._resolved_input_sidecar is not None:
            contract = self._resolved_input_sidecar["resolved_data_driven_contract"]
            policy = contract["nonprompt_policy"]
            family_inventories = {
                family: {
                    "scalar": manifest["scalar_nominal_processes"],
                    "eft": manifest["eft_nominal_processes"],
                    "sumw2": manifest["sumw2_processes"],
                }
                for family, manifest in self._resolved_input_sidecar[
                    "sumw2_content_manifest"
                ]["families"].items()
            }
            return self._build_prompt_subtraction_execution_plan(
                contract["resolved_prompt_process_set"],
                policy["explicit_exclusions"],
                family_inventories,
            )

        family_inventories = defaultdict(
            lambda: {"scalar": set(), "eft": set(), "sumw2": set()}
        )
        process_universe = set()
        for key, histogram in self._iter_input_histograms():
            family, component = self._family_from_nominal_key(key)
            if family is None:
                continue
            processes = {
                str(process) for process in self._axis_labels(histogram, "process")
            }
            family_inventories[family][component].update(processes)
            if component != "sumw2":
                process_universe.update(processes)
        if not process_universe:
            return {}
        try:
            certificate = certify_active_nonprompt_policy(
                sorted(process_universe),
                configuration_source="legacy_histogram_process_inventory",
            )
        except nonprompt_policy_error as error:
            raise RuntimeError(str(error)) from error
        return self._build_prompt_subtraction_execution_plan(
            certificate.resolved_prompt_process_set,
            certificate.explicit_exclusions,
            family_inventories,
        )

    def _initialize_eft_prompt_projection_context(self):
        empty_context = {
            "mode": "sm_point",
            "required_processes": [],
            "generated_nonprompt_eft_dependence": False,
        }
        if (
            self._artifact_kind
            not in {
                NONPROMPT_OUTPUT_ARTIFACT_KIND,
                NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
            }
        ):
            return {}, empty_context

        processes_by_family = {}
        projected_processes = set()
        for family, plan in self._prompt_subtraction_execution_by_family.items():
            family_processes = sorted(plan["eft_processes"])
            processes_by_family[family] = family_processes
            projected_processes.update(family_processes)
        return processes_by_family, {
            **empty_context,
            "required_processes": sorted(projected_processes),
        }

    @staticmethod
    def _filter_to_processes(histogram, process_names):
        allowed = {str(process) for process in process_names}
        observed = {
            str(process) for process in histogram.axes["process"]
        }
        return histogram.remove("process", sorted(observed - allowed))

    def _build_eft_prompt_projections(self):
        required_families = {
            family: processes
            for family, processes in self._eft_prompt_processes_by_family.items()
            if processes
        }
        if not required_families:
            return {}
        projections = {}
        for key, histogram in self._iter_input_histograms():
            if not key.endswith(EFT_NOMINAL_SUFFIX):
                continue
            family = key[: -len(EFT_NOMINAL_SUFFIX)]
            required_processes = required_families.get(family)
            if not required_processes:
                continue
            selected = self._filter_to_processes(histogram, required_processes)
            projections[family] = evaluate_eft_histogram_at_wc(selected, {})
        missing = sorted(set(required_families) - set(projections))
        if missing:
            raise RuntimeError(
                "Selected EFT prompt source sibling is missing for family/families: "
                + ", ".join(missing)
            )
        return projections

    @staticmethod
    def _family_from_nominal_key(key):
        if key.endswith(SCALAR_NOMINAL_SUFFIX):
            return key[: -len(SCALAR_NOMINAL_SUFFIX)], "scalar"
        if key.endswith(EFT_NOMINAL_SUFFIX):
            return key[: -len(EFT_NOMINAL_SUFFIX)], "eft"
        if key.endswith(SUMW2_SUFFIX):
            return key[: -len(SUMW2_SUFFIX)], "sumw2"
        if key in axes_info_2d:
            return key, "scalar"
        return None, None

    def _record_transformation_roles(
        self,
        key,
        output,
        *,
        generated_nonprompt_processes=(),
        generated_flips_processes=(),
    ):
        if self._transformation_role_context is None:
            return
        family, component = self._family_from_nominal_key(key)
        if family is None:
            return
        roles = self._transformation_role_context[family]
        output_processes = {
            str(process) for process in self._axis_labels(output, "process")
        }
        if component == "eft":
            roles["retained_eft_processes"] = sorted(
                output_processes & set(roles["source_eft_processes"])
            )
            return
        if component == "sumw2":
            return
        generated_nonprompt = {
            str(process) for process in generated_nonprompt_processes
        }
        generated_flips = {str(process) for process in generated_flips_processes}
        roles["retained_scalar_processes"] = sorted(
            output_processes & set(roles["source_scalar_processes"])
        )
        roles["generated_nonprompt_processes"] = sorted(generated_nonprompt)
        roles["generated_flips_processes"] = sorted(generated_flips)

    def _record_family_application_evidence(self, key, histogram):
        if self._transformation_role_context is None:
            return
        family, component = self._family_from_nominal_key(key)
        if family is None or component != "scalar":
            return
        source_application_regions = sorted(
            {str(region) for region in self._axis_labels(histogram, "appl")}
        )
        roles = self._transformation_role_context[family]
        if roles["source_application_regions"] is not None:
            if roles["source_application_regions"] != source_application_regions:
                raise RuntimeError(
                    f"Family {family!r} has inconsistent source application-region "
                    "evidence across scalar nominal inputs."
                )
            return
        roles["source_application_regions"] = source_application_regions
        roles["applicable_products"] = derive_data_driven_applicability(
            source_application_regions
        )

    def get_transformation_context(self, artifact_kind="nonprompt_output"):
        if self._transformation_role_context is None:
            raise RuntimeError(
                "Transformation roles are available only for validated schema-v2 inputs."
            )
        if artifact_kind not in TRANSFORMED_DATA_DRIVEN_ARTIFACT_KINDS:
            raise RuntimeError(
                f"Unknown data-driven artifact kind {artifact_kind!r}."
            )
        families = {}
        for family, raw_roles in self._transformation_role_context.items():
            if (
                raw_roles["source_application_regions"] is None
                or raw_roles["applicable_products"] is None
            ):
                raise RuntimeError(
                    f"Missing producer-generated application-region evidence for "
                    f"family {family!r}."
                )
            roles = {
                field_name: sorted(set(raw_roles[field_name]))
                for field_name in (
                    "source_scalar_processes",
                    "source_eft_processes",
                    "retained_scalar_processes",
                    "retained_eft_processes",
                    "generated_nonprompt_processes",
                    "generated_flips_processes",
                )
            }
            roles["source_application_regions"] = list(
                raw_roles["source_application_regions"]
            )
            roles["applicable_products"] = dict(
                raw_roles["applicable_products"]
            )
            if artifact_kind == FLIPS_OUTPUT_ARTIFACT_KIND:
                roles["retained_scalar_processes"] = []
                roles["generated_nonprompt_processes"] = []
            families[family] = roles
        return _producer_transformation_context(
            {
                "eft_prompt_projection": dict(self._eft_prompt_projection_context),
                "families": families,
            }
        )

    def get_prompt_subtraction_execution_evidence(self):
        if not self._prompt_subtraction_coverage_validated:
            raise RuntimeError(
                "Prompt-subtraction execution evidence is available only after "
                "the complete transformation has passed coverage validation."
            )
        families = {}
        for family, plan in self._prompt_subtraction_execution_by_family.items():
            evaluation_routes = {
                process: "scalar_nominal"
                for process in plan["scalar_processes"]
            }
            evaluation_routes.update(
                {
                    process: "eft_sm_point"
                    for process in plan["eft_processes"]
                }
            )
            families[family] = {
                "selected_processes": sorted(plan["selected_processes"]),
                "present_processes": sorted(plan["present_processes"]),
                "selected_present_processes": sorted(
                    plan["selected_present_processes"]
                ),
                "selected_absent_processes": sorted(
                    plan["selected_absent_processes"]
                ),
                "representation": dict(evaluation_routes),
                "nominal_evaluation_route": dict(evaluation_routes),
                "executed_processes": sorted(plan["executed_processes"]),
                "excluded_processes": sorted(plan["excluded_processes"]),
                "ambiguous_processes": sorted(plan["ambiguous_processes"]),
                "unhandled_processes": sorted(plan["unhandled_processes"]),
                "nonprompt_applicable": plan["nonprompt_applicable"],
            }
        return {"families": families}

    def get_effective_input_sidecar(self):
        return copy.deepcopy(self._resolved_input_sidecar)

    def _record_prompt_subtraction_execution(self, family, processes):
        plan = self._prompt_subtraction_execution_by_family.get(family)
        if plan is None:
            return
        duplicate = plan["executed_processes"] & set(processes)
        if duplicate:
            raise RuntimeError(
                f"Family {family!r} would subtract selected prompt process(es) "
                "through more than one nominal route: "
                + ", ".join(sorted(duplicate))
            )
        plan["executed_processes"].update(processes)

    @staticmethod
    def _validate_prompt_execution_groups(family, route, groups, expected):
        grouped = [
            process
            for processes in groups.values()
            for process in processes
        ]
        duplicates = sorted(
            process
            for process in set(grouped)
            if grouped.count(process) > 1
        )
        observed = set(grouped)
        if duplicates or observed != set(expected):
            raise RuntimeError(
                f"Family {family!r} prompt-subtraction {route} routing is not "
                "one-to-one with the selected execution set: "
                f"selected={sorted(expected)} routed={sorted(observed)} "
                f"duplicates={duplicates}."
            )

    def _group_selected_prompt_processes(
        self,
        family,
        route,
        selected_processes,
        allowed_outputs,
    ):
        groups = defaultdict(list)
        for process_name in sorted(selected_processes):
            _sample_name, year = self._parse_process(process_name)
            output_process = self._nonprompt_process_name(year)
            if allowed_outputs is not None and output_process not in allowed_outputs:
                raise RuntimeError(
                    f"Family {family!r} selected prompt process {process_name!r} "
                    f"has no certified nonprompt output route {output_process!r}."
                )
            groups[output_process].append(process_name)
        self._validate_prompt_execution_groups(
            family,
            route,
            groups,
            selected_processes,
        )
        return groups

    def _validate_prompt_subtraction_execution_coverage(self):
        for family, plan in self._prompt_subtraction_execution_by_family.items():
            expected = (
                plan["selected_present_processes"]
                if plan["nonprompt_applicable"]
                else set()
            )
            missing = expected - plan["executed_processes"]
            unexpected = plan["executed_processes"] - expected
            excluded = plan["executed_processes"] & plan["excluded_processes"]
            if missing or unexpected or excluded:
                raise RuntimeError(
                    f"Family {family!r} prompt-subtraction execution coverage failed: "
                    f"selected_present={sorted(expected)} "
                    f"executed={sorted(plan['executed_processes'])} "
                    f"missing={sorted(missing)} unexpected={sorted(unexpected)} "
                    f"excluded_executed={sorted(excluded)}."
                )
        self._prompt_subtraction_coverage_validated = True

    def _parse_process(self, process_name):
        try:
            return parse_process_name(str(process_name))
        except data_driven_product_error as error:
            raise RuntimeError(str(error)) from error

    def _build_process_metadata(self, histo):
        # Parse process names once per histogram and reuse the mapping across appl regions.
        process_metadata = {}
        for process_name in histo.axes["process"]:
            process_metadata[process_name] = self._parse_process(process_name)
        return process_metadata

    @staticmethod
    def _axis_labels(histo, axis_name):
        try:
            return list(histo.axes[axis_name])
        except Exception:
            return []

    @classmethod
    def dd_report_expected_regions_for_channel(cls, channel_name):
        if channel_name is None:
            return ()
        channel_label = str(channel_name or "").lower()
        if "2lss" in channel_label:
            return (
                ("sr", "isSR_2lSS"),
                ("nonprompt", "isAR_2lSS"),
                ("flips", "isAR_2lSS_OS"),
            )
        if "2los" in channel_label:
            return (
                ("sr", "isSR_2lOS"),
                ("nonprompt", "isAR_2lOS"),
            )
        if "1l" in channel_label:
            return (
                ("sr", "isSR_1l"),
                ("nonprompt", "isAR_1l"),
            )
        if "3l" in channel_label:
            return (
                ("sr", "isSR_3l"),
                ("nonprompt", "isAR_3l"),
            )
        if "4l" in channel_label:
            return (("sr", "isSR_4l"),)
        return ()

    @classmethod
    def _region_matches_channel(cls, region_name, channel_name):
        expected_regions = cls.dd_report_expected_regions_for_channel(channel_name)
        if not expected_regions:
            return True
        return any(region_name == expected_region for _, expected_region in expected_regions)

    def _select_histogram(self, histo, *, channel_name=None, process_name=None, systematic="nominal"):
        selected = histo
        if channel_name is not None:
            channel_axis = self._axis_labels(selected, "channel")
            if channel_axis and channel_name not in channel_axis:
                return None
            if channel_axis:
                selected = selected.integrate("channel", channel_name)
        if process_name is not None:
            process_axis = self._axis_labels(selected, "process")
            if process_axis and process_name not in process_axis:
                return None
            if process_axis:
                selected = selected.integrate("process", [process_name])
        if systematic is not None:
            systematic_axis = self._axis_labels(selected, "systematic")
            if systematic_axis and systematic not in systematic_axis:
                return None
            if systematic_axis:
                selected = selected.integrate("systematic", systematic)
        return selected

    @staticmethod
    def _selected_total(selected):
        if selected is None:
            return 0.0
        values = selected.values(flow=True)
        try:
            values = values[()]
        except Exception:
            pass
        return float(np.asarray(values).sum())

    def _total_for_selection(self, histo, *, channel_name=None, process_name=None, systematic="nominal"):
        selected = self._select_histogram(
            histo,
            channel_name=channel_name,
            process_name=process_name,
            systematic=systematic,
        )
        return self._selected_total(selected)

    def _has_selected_entries(self, histo, *, channel_name=None, process_name=None, systematic="nominal"):
        selected = self._select_histogram(
            histo,
            channel_name=channel_name,
            process_name=process_name,
            systematic=systematic,
        )
        if selected is None:
            return False
        if hasattr(selected, "view"):
            try:
                return bool(selected.view(as_dict=True, flow=True))
            except Exception:
                pass
        return self._selected_total(selected) != 0.0

    @staticmethod
    def _channel_labels_for_report(histo):
        channels = DataDrivenProducer._axis_labels(histo, "channel")
        return channels or [None]

    @staticmethod
    def _sorted_string_labels(labels):
        return tuple(sorted(str(label) for label in labels))

    @staticmethod
    def _is_effectively_zero(value):
        return abs(float(value)) < 1e-12

    @staticmethod
    def _init_dd_report(key, histo, *, empty=False):
        return {
            "key": key,
            "empty": empty,
            "channels": DataDrivenProducer._channel_labels_for_report(histo),
            "regions": DataDrivenProducer._sorted_string_labels(
                DataDrivenProducer._axis_labels(histo, "appl")
            ),
            "rows": [],
        }

    @staticmethod
    def _nonprompt_process_name(year):
        return generated_process_name("nonprompt", year)

    @staticmethod
    def _flips_process_name(year):
        return generated_process_name("flips", year)

    def _resolved_family_products(self, key):
        family, _component = self._family_from_nominal_key(key)
        if family is None and key.endswith("_sumw2"):
            family = key[: -len("_sumw2")]
        input_sidecar = (
            self._resolved_input_sidecar
        )
        if input_sidecar is None or "resolved_data_driven_contract" not in input_sidecar:
            return {
                "nonprompt": {
                    "enabled": self._artifact_kind
                    in {
                        NONPROMPT_OUTPUT_ARTIFACT_KIND,
                        NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
                    }
                    and not (
                        self._artifact_kind
                        == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                        and key.endswith("_sumw2")
                    ),
                    "generated_outputs": None,
                },
                "flips": {
                    "enabled": True,
                    "generated_outputs": None,
                },
            }
        if family not in input_sidecar["sumw2_content_manifest"]["families"]:
            raise RuntimeError(
                f"Missing resolved data-driven contract for family {family!r}."
            )
        products = input_sidecar["resolved_data_driven_contract"]["products"]
        return {
            "nonprompt": {
                **products["nonprompt"],
                "enabled": (
                    products["nonprompt"]["enabled"]
                    and self._artifact_kind
                    in {
                        NONPROMPT_OUTPUT_ARTIFACT_KIND,
                        NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND,
                    }
                    and not (
                        self._artifact_kind
                        == NONPROMPT_NOMINAL_REFERENCE_ARTIFACT_KIND
                        and key.endswith("_sumw2")
                    )
                ),
            },
            "flips": products["flips"],
        }

    @classmethod
    def _systematic_summary(cls, source_hist, used_hist):
        source_labels = cls._sorted_string_labels(cls._axis_labels(source_hist, "systematic"))
        used_labels = cls._sorted_string_labels(cls._axis_labels(used_hist, "systematic"))
        used_label_set = set(used_labels)
        dropped_labels = tuple(label for label in source_labels if label not in used_label_set)
        return {
            "kept": used_labels,
            "dropped": dropped_labels,
        }

    def _process_breakdown(self, histo, process_names, *, channel_name=None, systematic="nominal"):
        breakdown = []
        for process_name in sorted(process_names):
            total = self._total_for_selection(
                histo,
                channel_name=channel_name,
                process_name=process_name,
                systematic=systematic,
            )
            if self._is_effectively_zero(total):
                continue
            breakdown.append(
                {
                    "process": process_name,
                    "total": total,
                }
            )
        return breakdown

    def _record_sr_report(self, report, ident, hAR):
        for channel_name in report["channels"]:
            if not self._region_matches_channel(ident, channel_name):
                continue
            if not self._has_selected_entries(hAR, channel_name=channel_name, systematic="nominal"):
                continue
            report["rows"].append(
                {
                    "channel": channel_name,
                    "family": "sr",
                    "region": ident,
                    "retained_total": self._total_for_selection(
                        hAR,
                        channel_name=channel_name,
                        systematic="nominal",
                    ),
                }
            )

    def _record_flips_report(
        self,
        report,
        ident,
        hAR,
        hFlipsRaw,
        hFlipsUsed,
        output_process_names,
        source_processes_by_output,
    ):
        for channel_name in report["channels"]:
            if not self._region_matches_channel(ident, channel_name):
                continue
            for output_process in output_process_names:
                if not self._has_selected_entries(
                    hFlipsUsed,
                    channel_name=channel_name,
                    process_name=output_process,
                    systematic="nominal",
                ):
                    continue
                result_total = self._total_for_selection(
                    hFlipsUsed,
                    channel_name=channel_name,
                    process_name=output_process,
                    systematic="nominal",
                )
                report["rows"].append(
                    {
                        "channel": channel_name,
                        "family": "flips",
                        "region": ident,
                        "output_process": output_process,
                        "data_used": result_total,
                        "result": result_total,
                        "data_sources": self._process_breakdown(
                            hAR,
                            source_processes_by_output.get(output_process, []),
                            channel_name=channel_name,
                        ),
                        "systematics": self._systematic_summary(hFlipsRaw, hFlipsUsed),
                    }
                )

    def _record_nonprompt_report(
        self,
        report,
        ident,
        hAR,
        hDataUsed,
        hPromptSubRaw,
        hPromptSubScaled,
        hResult,
        output_process_names,
        data_source_processes_by_output,
        prompt_source_processes_by_output,
    ):
        for channel_name in report["channels"]:
            if not self._region_matches_channel(ident, channel_name):
                continue
            for output_process in output_process_names:
                has_data = self._has_selected_entries(
                    hDataUsed,
                    channel_name=channel_name,
                    process_name=output_process,
                    systematic="nominal",
                )
                has_prompt = self._has_selected_entries(
                    hPromptSubScaled,
                    channel_name=channel_name,
                    process_name=output_process,
                    systematic="nominal",
                )
                if not (has_data or has_prompt):
                    continue
                report["rows"].append(
                    {
                        "channel": channel_name,
                        "family": "nonprompt",
                        "region": ident,
                        "output_process": output_process,
                        "data_used": self._total_for_selection(
                            hDataUsed,
                            channel_name=channel_name,
                            process_name=output_process,
                            systematic="nominal",
                        ),
                        "prompt_sub_used": self._total_for_selection(
                            hPromptSubScaled,
                            channel_name=channel_name,
                            process_name=output_process,
                            systematic="nominal",
                        ) * -1.0,
                        "result": self._total_for_selection(
                            hResult,
                            channel_name=channel_name,
                            process_name=output_process,
                            systematic="nominal",
                        ),
                        "data_sources": self._process_breakdown(
                            hAR,
                            data_source_processes_by_output.get(output_process, []),
                            channel_name=channel_name,
                        ),
                        "prompt_sub_sources": self._process_breakdown(
                            hAR,
                            prompt_source_processes_by_output.get(output_process, []),
                            channel_name=channel_name,
                        ),
                        "prompt_sub_systematics": self._systematic_summary(
                            hPromptSubRaw,
                            hPromptSubScaled,
                        ),
                    }
                )

    def _build_data_driven_histogram(self, key, histo):
        self._record_family_application_evidence(key, histo)
        family, component = self._family_from_nominal_key(key)
        if key.endswith(EFT_NOMINAL_SUFFIX):
            output = None
            for appl in histo.axes["appl"]:
                selected = histo.integrate("appl", appl)
                if "isAR" in appl:
                    continue
                output = selected if output is None else output + selected
            if output is None:
                output = histo.integrate("appl")
                output.reset()
            self._record_transformation_roles(key, output)
            return output

        if histo.empty():  # histo is empty, so we just integrate over appl and keep an empty histo
            if self._dd_report_enabled and not key.endswith("_sumw2"):
                self._dd_report_by_key[key] = self._init_dd_report(key, histo, empty=True)
            print(f"[W]: Histogram {key} is empty, returning an empty histo")
            output = histo.integrate("appl")
            self._record_transformation_roles(key, output)
            return output

        process_metadata = self._build_process_metadata(histo)
        resolved_products = self._resolved_family_products(key)
        nonprompt_enabled = resolved_products["nonprompt"]["enabled"]
        flips_enabled = resolved_products["flips"]["enabled"]
        nonprompt_outputs = resolved_products["nonprompt"]["generated_outputs"]
        flips_outputs = resolved_products["flips"]["generated_outputs"]
        report = None
        if self._dd_report_enabled and not key.endswith("_sumw2"):
            report = self._init_dd_report(key, histo)

        # now for each year we actually perform the subtraction and integrate out the application regions
        newhist = None
        generated_nonprompt_processes = set()
        generated_flips_processes = set()
        executed_prompt_processes = set()
        for ident in histo.axes["appl"]:
            hAR = histo.integrate("appl", ident)
            product = data_driven_product_for_application_region(ident)

            if product is None:
                if str(ident).startswith("isAR"):
                    continue
                # if we are in the signal region, we just take the
                # whole histogram integrating out the application region axis
                if report is not None:
                    self._record_sr_report(report, ident, hAR)
                if newhist is None:
                    newhist = hAR
                else:
                    newhist += hAR
            elif product == "flips":
                if not flips_enabled:
                    continue
                # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                if flips_outputs is not None:
                    newNameDictData = {
                        output_process: list(
                            output_record["source_contributors"]["data"]
                        )
                        for output_process, output_record in flips_outputs.items()
                    }
                else:
                    newNameDictData = defaultdict(list)
                    for process_name in hAR.axes["process"]:
                        sampleName, year = process_metadata[process_name]
                        flips_name = self._flips_process_name(year)
                        if self.dataName == sampleName:
                            newNameDictData[flips_name].append(process_name)
                generated_flips_processes.update(newNameDictData)
                hFlips = hAR.group("process", newNameDictData)
                hFlipsRaw = hFlips

                # remove any up/down FF variations from the flip histo since we don't use that info
                syst_var_idet_rm_lst = []
                syst_var_idet_lst = list(hFlips.axes["systematic"])
                for syst_var_idet in syst_var_idet_lst:
                    if syst_var_idet != "nominal":
                        syst_var_idet_rm_lst.append(syst_var_idet)
                hFlips = hFlips.remove("systematic", syst_var_idet_rm_lst)

                if report is not None:
                    output_process_names = self._sorted_string_labels(
                        self._axis_labels(hFlips, "process")
                    )
                    self._record_flips_report(
                        report,
                        ident,
                        hAR,
                        hFlipsRaw,
                        hFlips,
                        output_process_names,
                        newNameDictData,
                    )

                # now adding them to the list of processes:
                if newhist is None:
                    newhist = hFlips
                else:
                    newhist += hFlips

            else:
                if not nonprompt_enabled:
                    continue
                if family in self._prompt_subtraction_execution_by_family:
                    self._prompt_subtraction_execution_by_family[family][
                        "nonprompt_applicable"
                    ] = True
                # if we are in the nonprompt application region, we also integrate the application region axis
                # and construct the new process 'nonprompt'
                # we look at data only, and rename it to fakes
                if nonprompt_outputs is not None:
                    newNameDictData = {
                        output_process: list(
                            output_record["source_contributors"]["data"]
                        )
                        for output_process, output_record in nonprompt_outputs.items()
                    }
                    scalar_processes = {
                        str(process) for process in hAR.axes["process"]
                    }
                    if component == "sumw2":
                        selected_scalar_processes = (
                            self._prompt_subtraction_execution_by_family[family][
                                "selected_processes"
                            ]
                            & scalar_processes
                        )
                    else:
                        selected_scalar_processes = (
                            self._prompt_subtraction_execution_by_family[family][
                                "scalar_processes"
                            ]
                        )
                    newNameDictNoData = self._group_selected_prompt_processes(
                        family,
                        "sumw2" if component == "sumw2" else "scalar",
                        selected_scalar_processes,
                        set(nonprompt_outputs),
                    )
                else:
                    newNameDictData = defaultdict(list)
                    newNameDictNoData = defaultdict(list)
                    if component == "sumw2":
                        resolved_prompt_processes = (
                            self._prompt_subtraction_execution_by_family[family][
                                "selected_processes"
                            ]
                            & {str(process) for process in hAR.axes["process"]}
                        )
                    else:
                        resolved_prompt_processes = (
                            self._prompt_subtraction_execution_by_family[family][
                                "scalar_processes"
                            ]
                        )
                    for process_name in hAR.axes["process"]:
                        sampleName, year = process_metadata[process_name]

                        nonprompt_name = self._nonprompt_process_name(year)
                        if self.dataName == sampleName:
                            newNameDictData[nonprompt_name].append(process_name)
                        elif str(process_name) in resolved_prompt_processes:
                            newNameDictNoData[nonprompt_name].append(process_name)
                        else:
                            pass
                            # print(f"We won't consider {sampleName} for the prompt subtraction in the appl. region")
                    self._validate_prompt_execution_groups(
                        family,
                        "sumw2" if component == "sumw2" else "scalar",
                        newNameDictNoData,
                        resolved_prompt_processes,
                    )
                generated_nonprompt_processes.update(newNameDictData)
                generated_nonprompt_processes.update(newNameDictNoData)
                hFakes = hAR.group("process", newNameDictData)
                # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                hPromptSub = hAR.group("process", newNameDictNoData)
                prompt_source_hist = hAR
                projection = self._eft_prompt_projections.get(family)
                if projection is not None and not key.endswith("_sumw2"):
                    projected_ar = projection.integrate("appl", ident)
                    projected_processes = {
                        str(process) for process in projected_ar.axes["process"]
                    }
                    selected_eft_processes = (
                        self._prompt_subtraction_execution_by_family[family][
                            "eft_processes"
                        ]
                    )
                    if projected_processes != selected_eft_processes:
                        raise RuntimeError(
                            f"Family {family!r} EFT nominal evaluation did not cover "
                            "the selected EFT execution route exactly: "
                            f"selected={sorted(selected_eft_processes)} "
                            f"evaluated={sorted(projected_processes)}."
                        )
                    projection_groups = self._group_selected_prompt_processes(
                        family,
                        "eft_sm_point",
                        selected_eft_processes,
                        (
                            set(nonprompt_outputs)
                            if nonprompt_outputs is not None
                            else None
                        ),
                    )
                    projected_prompt = projected_ar.group(
                        "process", projection_groups
                    )
                    try:
                        hPromptSub += projected_prompt
                        prompt_source_hist = hAR + projected_ar
                    except Exception as error:
                        raise RuntimeError(
                            f"Incompatible axes while evaluating selected EFT prompt "
                            f"sources at the SM point for family={family!r} "
                            f"application_region={ident!r}."
                        ) from error
                if component == "scalar":
                    executed_prompt_processes.update(
                        process
                        for processes in newNameDictNoData.values()
                        for process in processes
                    )
                    if projection is not None:
                        executed_prompt_processes.update(projected_processes)
                hPromptSubRaw = hPromptSub

                # remove the up/down variations (if any) from the prompt subtraction histo
                # but keep FFUp and FFDown, as these are the nonprompt up and down variations
                syst_var_idet_rm_lst = []
                syst_var_idet_lst = list(hPromptSub.axes["systematic"])
                for syst_var_idet in syst_var_idet_lst:
                    if (syst_var_idet != "nominal") and (not syst_var_idet.startswith("FF")):
                        syst_var_idet_rm_lst.append(syst_var_idet)
                hPromptSub = hPromptSub.remove("systematic", syst_var_idet_rm_lst)

                # now we actually make the subtraction
                # var(A - B) = var(A) + var(B)
                if not key.endswith("_sumw2"):
                    hPromptSub.scale(-1)
                hFakes += hPromptSub

                if report is not None:
                    output_process_names = self._sorted_string_labels(
                        self._axis_labels(hFakes, "process")
                    )
                    self._record_nonprompt_report(
                        report,
                        ident,
                        prompt_source_hist,
                        hAR.group("process", newNameDictData),
                        hPromptSubRaw,
                        hPromptSub,
                        hFakes,
                        output_process_names,
                        newNameDictData,
                        newNameDictNoData,
                    )
                # now adding them to the list of processes:
                if newhist is None:
                    newhist = hFakes
                else:
                    newhist += hFakes

        if report is not None:
            self._dd_report_by_key[key] = report
        if not key.endswith("_sumw2"):
            self._record_transformation_roles(
                key,
                newhist,
                generated_nonprompt_processes=generated_nonprompt_processes,
                generated_flips_processes=generated_flips_processes,
            )
        if component == "scalar" and executed_prompt_processes:
            self._record_prompt_subtraction_execution(
                family,
                executed_prompt_processes,
            )
        return newhist

    def iter_data_driven_histograms(self):
        if self.outHist is not None:
            yield from self.outHist.items()
            return

        seen_keys = set(self._input_histogram_keys())
        required_companions = set()
        for key in seen_keys:
            if key.endswith(SCALAR_NOMINAL_SUFFIX):
                family = key[: -len(SCALAR_NOMINAL_SUFFIX)]
                required_companions.add(f"{family}_sumw2")
            elif key in axes_info_2d:
                required_companions.add(f"{key}_sumw2")
        missing = sorted(required_companions - seen_keys)
        if missing:
            raise RuntimeError(
                "Nonprompt construction requires scalar statistical companions: "
                + ", ".join(missing)
            )
        for key, histo in self._iter_input_histograms():
            yield key, self._build_data_driven_histogram(key, histo)
        self._validate_prompt_subtraction_execution_coverage()

    def DDFakes(self):
        new_output = {}
        for key, histo in self.iter_data_driven_histograms():
            new_output[key] = histo
        self.outHist = new_output

    def dumpToPickle(self):
        if not self.outputName.endswith(".pkl.gz"):
            self.outputName = self.outputName + ".pkl.gz"
        if self.outHist is None:
            self.DDFakes()
        input_sidecar = (
            self._resolved_input_sidecar
        )
        if input_sidecar is not None:
            write_histogram_artifact(
                self.outputName,
                histograms=self.outHist,
                artifact_kind=self._artifact_kind,
                sumw2_storage_provenance=input_sidecar[
                    "sumw2_storage_provenance"
                ],
                lineage_inputs=[lineage_input_from_sidecar(input_sidecar)],
                input_sidecar=input_sidecar,
                transformation_context=self.get_transformation_context(
                    self._artifact_kind
                ),
            )
        else:
            with gzip.open(self.outputName, "wb") as fout:
                cloudpickle.dump(self.outHist, fout)


    def getDataDrivenHistogram(self):
        if self.outHist is None:
            self.DDFakes()
        return self.outHist

    def get_dd_report(self, key):
        if not self._dd_report_enabled:
            return None
        return self._dd_report_by_key.pop(key, None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    DataDrivenProducer(args.pkl_file_path, '')
