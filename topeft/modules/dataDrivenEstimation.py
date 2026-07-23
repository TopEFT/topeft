import argparse
import gzip
import warnings
from collections import defaultdict

import cloudpickle
import numpy as np
from topcoffea.modules.hist_utils import iterate_hist_from_pkl

from topcoffea.modules.get_param_from_jsons import GetParam
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.data_driven_products import (
    data_driven_product_error,
    generated_process_name,
    parse_process_name,
    validate_requested_product_input,
)
from topeft.modules.histogram_artifact import (
    lineage_input_from_sidecar,
    validate_histogram_artifact,
    write_histogram_artifact,
)
from topeft.modules.nominal_schema import (
    EFT_NOMINAL_SUFFIX,
    SCALAR_NOMINAL_SUFFIX,
    evaluate_eft_histogram_at_wc,
)
from topeft.modules.paths import topeft_path
get_te_param = GetParam(topeft_path("params/params.json"))

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
        if artifact_kind not in {"nonprompt_output", "flips_output"}:
            raise RuntimeError(f"Unknown data-driven artifact kind {artifact_kind!r}.")
        self._artifact_kind = artifact_kind
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
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
                validate_requested_product_input(
                    self._input_artifact_validation["metadata"],
                    artifact_kind=artifact_kind,
                )
        self._transformation_role_context = self._initialize_transformation_role_context()
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
        input_sidecar = self._input_artifact_validation["metadata"]
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
            }
        return families

    def _initialize_eft_prompt_projection_context(self):
        input_sidecar = (
            self._input_artifact_validation["metadata"]
            if self._input_artifact_validation is not None
            else None
        )
        empty_context = {
            "mode": "sm_point",
            "required_processes": [],
            "generated_nonprompt_eft_dependence": False,
        }
        if (
            not input_sidecar
            or self._artifact_kind != "nonprompt_output"
            or "resolved_data_driven_contract" not in input_sidecar
        ):
            return {}, empty_context

        contract = input_sidecar["resolved_data_driven_contract"]
        required_prompt_signals = set(
            contract["required_prompt_signal_processes"]
        )
        processes_by_family = {}
        projected_processes = set()
        for family, manifest in input_sidecar["sumw2_content_manifest"][
            "families"
        ].items():
            scalar_processes = set(manifest["scalar_nominal_processes"])
            eft_processes = set(manifest["eft_nominal_processes"])
            duplicates = sorted(
                required_prompt_signals & scalar_processes & eft_processes
            )
            if duplicates:
                raise RuntimeError(
                    f"Family {family!r} has required private EFT source(s) duplicated "
                    "in scalar and EFT nominal siblings: "
                    + ", ".join(duplicates)
                )
            missing = sorted(
                required_prompt_signals - scalar_processes - eft_processes
            )
            if missing:
                raise RuntimeError(
                    f"Family {family!r} is missing required private EFT source(s): "
                    + ", ".join(missing)
                )
            family_processes = sorted(required_prompt_signals & eft_processes)
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
                "Required private EFT source sibling is missing for family/families: "
                + ", ".join(missing)
            )
        return projections

    @staticmethod
    def _family_from_nominal_key(key):
        if key.endswith(SCALAR_NOMINAL_SUFFIX):
            return key[: -len(SCALAR_NOMINAL_SUFFIX)], "scalar"
        if key.endswith(EFT_NOMINAL_SUFFIX):
            return key[: -len(EFT_NOMINAL_SUFFIX)], "eft"
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
        generated_nonprompt = {
            str(process) for process in generated_nonprompt_processes
        }
        generated_flips = {str(process) for process in generated_flips_processes}
        roles["retained_scalar_processes"] = sorted(
            output_processes & set(roles["source_scalar_processes"])
        )
        roles["generated_nonprompt_processes"] = sorted(generated_nonprompt)
        roles["generated_flips_processes"] = sorted(generated_flips)

    def get_transformation_context(self, artifact_kind="nonprompt_output"):
        if self._transformation_role_context is None:
            raise RuntimeError(
                "Transformation roles are available only for validated schema-v2 inputs."
            )
        if artifact_kind not in {"nonprompt_output", "flips_output"}:
            raise RuntimeError(
                f"Unknown data-driven artifact kind {artifact_kind!r}."
            )
        families = {}
        for family, raw_roles in self._transformation_role_context.items():
            roles = {
                field_name: sorted(set(processes))
                for field_name, processes in raw_roles.items()
            }
            if artifact_kind == "flips_output":
                roles["retained_scalar_processes"] = []
                roles["generated_nonprompt_processes"] = []
            families[family] = roles
        return {
            "eft_prompt_projection": dict(self._eft_prompt_projection_context),
            "families": families,
        }

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
            self._input_artifact_validation["metadata"]
            if self._input_artifact_validation is not None
            else None
        )
        if input_sidecar is None or "resolved_data_driven_contract" not in input_sidecar:
            return {
                "nonprompt": {
                    "enabled": self._artifact_kind == "nonprompt_output",
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
                    and self._artifact_kind == "nonprompt_output"
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
        for ident in histo.axes["appl"]:
            hAR = histo.integrate("appl", ident)

            if "isAR" not in ident:
                # if we are in the signal region, we just take the
                # whole histogram integrating out the application region axis
                if report is not None:
                    self._record_sr_report(report, ident, hAR)
                if newhist is None:
                    newhist = hAR
                else:
                    newhist += hAR
            elif ident == "isAR_2lSS_OS":
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
                    newNameDictNoData = {
                        output_process: sorted(
                            set(prompt_processes) & scalar_processes
                        )
                        for output_process, output_record in nonprompt_outputs.items()
                        if (
                            prompt_processes := output_record["source_contributors"][
                                "prompt_mc"
                            ]
                        ) and set(prompt_processes) & scalar_processes
                    }
                else:
                    newNameDictData = defaultdict(list)
                    newNameDictNoData = defaultdict(list)
                    for process_name in hAR.axes["process"]:
                        sampleName, year = process_metadata[process_name]

                        nonprompt_name = self._nonprompt_process_name(year)
                        if self.dataName == sampleName:
                            newNameDictData[nonprompt_name].append(process_name)
                        elif sampleName in self.promptSubtractionSamples:
                            newNameDictNoData[nonprompt_name].append(process_name)
                        else:
                            pass
                            # print(f"We won't consider {sampleName} for the prompt subtraction in the appl. region")
                generated_nonprompt_processes.update(newNameDictData)
                generated_nonprompt_processes.update(newNameDictNoData)
                hFakes = hAR.group("process", newNameDictData)
                # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                hPromptSub = hAR.group("process", newNameDictNoData)
                prompt_source_hist = hAR
                family, _component = self._family_from_nominal_key(key)
                projection = self._eft_prompt_projections.get(family)
                if projection is not None and not key.endswith("_sumw2"):
                    projected_ar = projection.integrate("appl", ident)
                    projected_processes = {
                        str(process) for process in projected_ar.axes["process"]
                    }
                    projection_groups = {
                        output_process: sorted(
                            set(output_record["source_contributors"]["prompt_mc"])
                            & projected_processes
                        )
                        for output_process, output_record in nonprompt_outputs.items()
                        if set(output_record["source_contributors"]["prompt_mc"])
                        & projected_processes
                    }
                    projected_prompt = projected_ar.group(
                        "process", projection_groups
                    )
                    try:
                        hPromptSub += projected_prompt
                        prompt_source_hist = hAR + projected_ar
                    except Exception as error:
                        raise RuntimeError(
                            f"Incompatible axes while projecting private EFT prompt "
                            f"sources at the SM point for family={family!r} "
                            f"application_region={ident!r}."
                        ) from error
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
            self._input_artifact_validation["metadata"]
            if self._input_artifact_validation is not None
            else None
        )
        if input_sidecar is not None:
            write_histogram_artifact(
                self.outputName,
                histograms=self.outHist,
                artifact_kind="nonprompt_output",
                sumw2_storage_provenance=input_sidecar[
                    "sumw2_storage_provenance"
                ],
                lineage_inputs=[lineage_input_from_sidecar(input_sidecar)],
                input_sidecar=input_sidecar,
                transformation_context=self.get_transformation_context(
                    "nonprompt_output"
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
