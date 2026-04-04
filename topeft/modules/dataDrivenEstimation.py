import argparse
import gzip
import re
from collections import defaultdict

import cloudpickle
import numpy as np
from topcoffea.modules.hist_utils import iterate_hist_from_pkl

from topcoffea.modules.get_param_from_jsons import GetParam
from topcoffea.modules.utils import canonicalize_process_name
from topeft.modules.paths import topeft_path
get_te_param = GetParam(topeft_path("params/params.json"))

class DataDrivenProducer:
    _NAME_REGEX = r"^(?P<process>.*?)(?:UL)?(?P<year>(?:\d{2}(?:APV|EE|BPix)?|\d{4}(?:EE|BPix)?))$"
    _KNOWN_YEARS = {"16APV", "16", "17", "18", "2022", "2022EE", "2023", "2023BPix"}

    def __init__(self, inputHist, outputName, iterator_mode=False, dd_report=False):
        self._input_source = inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.iterator_mode = iterator_mode
        self._dd_report_enabled = dd_report
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
        self._name_pattern = re.compile(self._NAME_REGEX)
        self._dd_report_by_key = {} if dd_report else None
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

    def _parse_process(self, process_name):
        match = self._name_pattern.search(process_name)
        if not match:
            raise RuntimeError(f"Sample {process_name} does not match the naming convention.")

        sample_name = match.group("process")
        year = match.group("year").replace("central", "").replace("UL", "")
        if year not in self._KNOWN_YEARS:
            raise RuntimeError(
                f"Sample {process_name} does not match the naming convention, year \"{year}\" is unknown."
            )

        return sample_name, year

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
        if ("2022" in year) or ("2023" in year):
            raw_name = f"nonprompt{year}"
        else:
            raw_name = f"nonpromptUL{year}"
        return canonicalize_process_name(raw_name)

    @staticmethod
    def _flips_process_name(year):
        if year.startswith("202"):
            raw_name = f"flips{year}"
        else:
            raw_name = f"flipsUL{year}"
        return canonicalize_process_name(raw_name)

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
        if histo.empty():  # histo is empty, so we just integrate over appl and keep an empty histo
            if self._dd_report_enabled and not key.endswith("_sumw2"):
                self._dd_report_by_key[key] = self._init_dd_report(key, histo, empty=True)
            print(f"[W]: Histogram {key} is empty, returning an empty histo")
            return histo.integrate("appl")

        process_metadata = self._build_process_metadata(histo)
        report = None
        if self._dd_report_enabled and not key.endswith("_sumw2"):
            report = self._init_dd_report(key, histo)

        # now for each year we actually perform the subtraction and integrate out the application regions
        newhist = None
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
                # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                newNameDictData = defaultdict(list)
                for process_name in hAR.axes["process"]:
                    sampleName, year = process_metadata[process_name]
                    flips_name = self._flips_process_name(year)
                    if self.dataName == sampleName:
                        newNameDictData[flips_name].append(process_name)
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
                # if we are in the nonprompt application region, we also integrate the application region axis
                # and construct the new process 'nonprompt'
                # we look at data only, and rename it to fakes
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
                hFakes = hAR.group("process", newNameDictData)
                # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                hPromptSub = hAR.group("process", newNameDictNoData)
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
                        hAR,
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
        return newhist

    def iter_data_driven_histograms(self):
        if self.outHist is not None:
            yield from self.outHist.items()
            return

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
