import argparse
import gzip
import logging
import re
from collections import defaultdict

import cloudpickle
from topcoffea.modules.hist_utils import iterate_hist_from_pkl

from topeft.modules.paths import topeft_path
from topcoffea.modules.utils import canonicalize_process_name
from topcoffea.modules.get_param_from_jsons import GetParam
get_te_param = GetParam(topeft_path("params/params.json"))


logger = logging.getLogger(__name__)

class DataDrivenProducer:
    _NAME_REGEX = r"^(?P<process>.*?)(?:UL)?(?P<year>(?:\d{2}(?:APV|EE|BPix)?|\d{4}(?:EE|BPix)?))$"
    _KNOWN_YEARS = {"16APV", "16", "17", "18", "2022", "2022EE", "2023", "2023BPix"}

    def __init__(self, inputHist, outputName, iterator_mode=False):
        self._input_source = inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.iterator_mode = iterator_mode
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
        self._name_pattern = re.compile(self._NAME_REGEX)
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

    def _build_data_driven_histogram(self, key, histo):
        if histo.empty():  # histo is empty, so we just integrate over appl and keep an empty histo
            print(f"[W]: Histogram {key} is empty, returning an empty histo")
            return histo.integrate("appl")

        process_metadata = self._build_process_metadata(histo)

        # now for each year we actually perform the subtraction and integrate out the application regions
        newhist = None
        for ident in histo.axes["appl"]:
            hAR = histo.integrate("appl", ident)

            if "isAR" not in ident:
                # if we are in the signal region, we just take the
                # whole histogram integrating out the application region axis
                if newhist is None:
                    newhist = hAR
                else:
                    newhist += hAR
            elif ident == "isAR_2lSS_OS":
                # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                newNameDictData = defaultdict(list)
                for process_name in hAR.axes["process"]:
                    sampleName, year = process_metadata[process_name]
                    if year.startswith("202"):
                        raw_flips_name = f"flips{year}"
                    else:
                        raw_flips_name = f"flipsUL{year}"
                    flips_name = canonicalize_process_name(raw_flips_name)
                    if raw_flips_name == flips_name:
                        logger.debug("Process name '%s' already canonical", raw_flips_name)
                    if self.dataName == sampleName:
                        newNameDictData[flips_name].append(process_name)
                hFlips = hAR.group("process", newNameDictData)

                # remove any up/down FF variations from the flip histo since we don't use that info
                syst_var_idet_rm_lst = []
                syst_var_idet_lst = list(hFlips.axes["systematic"])
                for syst_var_idet in syst_var_idet_lst:
                    if syst_var_idet != "nominal":
                        syst_var_idet_rm_lst.append(syst_var_idet)
                hFlips = hFlips.remove("systematic", syst_var_idet_rm_lst)

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

                    if ("2022" in year) or ("2023" in year):
                        raw_nonprompt_name = f"nonprompt{year}"
                    else:
                        raw_nonprompt_name = f"nonpromptUL{year}"
                    nonprompt_name = canonicalize_process_name(raw_nonprompt_name)
                    if raw_nonprompt_name == nonprompt_name:
                        logger.debug("Process name '%s' already canonical", raw_nonprompt_name)
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
                # now adding them to the list of processes:
                if newhist is None:
                    newhist = hFakes
                else:
                    newhist += hFakes

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    DataDrivenProducer(args.pkl_file_path, '')
