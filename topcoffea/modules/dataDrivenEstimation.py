import argparse
import hist
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_param
import topcoffea.modules.utils as utils
import cloudpickle
from collections import defaultdict
import re
import gzip

class DataDrivenProducer:
    def __init__(self, inputHist, outputName):
        yt=YieldTools()
        if isinstance(inputHist, str) and inputHist.endswith('.pkl.gz'): # we are plugging a pickle file
            self.inhist=utils.get_hist_from_pkl(inputHist)
        else: # we already have the histogram
            self.inhist=inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.promptSubtractionSamples=get_param('prompt_subtraction_samples')
        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist

        self.outHist={}

        for key,histo in self.inhist.items():

            if histo.empty():  # histo is empty, so we just integrate over appl and keep an empty histo
                print(f'[W]: Histogram {key} is empty, returning an empty histo')
                self.outHist[key]=histo.integrate('appl')
                continue

            # First we are gonna scale all MC processes in  by the luminosity
            name_regex='(?P<process>.*)UL(?P<year>.*)'
            pattern=re.compile(name_regex)

            scale_dict={}
            for process in histo.axes["process"]:
                match = pattern.search(process)
                sampleName=match.group('process')
                year=match.group('year')
                if not match:
                    raise RuntimeError(f"Sample {process} does not match the naming convention.")
                if year not in ['16APV','16','17','18']:
                    raise RuntimeError(f"Sample {process} does not match the naming convention, year \"{year}\" is unknown.")

            # do these lines do anything? scale_dict is empty, and pre and postscale are never used.
            # prescale = histo.eval(values=None).copy()
            # histo.scale(scale_dict, axis=("process",))
            # postscale = histo.eval(values=None)

            # now for each year we actually perform the subtraction and integrate out the application regions
            newhist=None
            for ident in histo.axes["appl"]:
                hAR = histo.integrate("appl", ident)

                if 'isAR' not in ident:
                    # if we are in the signal region, we just take the
                    # whole histogram integrating out the application region axis
                    if newhist is None:
                        newhist = hAR
                    else:
                        newhist = newhist.union(hAR, "process")
                else:
                    if "isAR_2lSS_OS" == ident:
                        # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                        newNameDictData = defaultdict(list)
                        for process in hAR.axes["process"]:
                            match = pattern.search(process)
                            sampleName = match.group("process")
                            year = match.group("year")
                            nonPromptName = "flipsUL%s" % year
                            if self.dataName == sampleName:
                                newNameDictData[nonPromptName].append(process)
                        hFlips = hAR.group(
                            "process",
                            "process",
                            newNameDictData,
                        )

                        # remove any up/down FF variations from the flip histo since we don't use that info
                        syst_var_idet_rm_lst = []
                        for syst_var_idet in hFlips.axes["systematic"]:
                            if (syst_var_idet != "nominal"):
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
                        for process in hAR.axes["process"]:
                            match = pattern.search(process)
                            sampleName = match.group("process")
                            year = match.group("year")
                            nonPromptName = "nonpromptUL%s" % year
                            if self.dataName == sampleName:
                                newNameDictData[nonPromptName].append(process)
                            elif sampleName in self.promptSubtractionSamples:
                                newNameDictNoData[nonPromptName].append(process)
                            else:
                                print(
                                    f"We won't consider {sampleName} for the prompt subtraction in the appl. region"
                                )
                        hFakes = hAR.group(
                            "process",
                            "process",
                            newNameDictData,
                        )
                        # now we take all the stuff that is not data in the AR to make the prompt
                        # subtraction and assign them to nonprompt.
                        hPromptSub = hAR.group(
                            "process",
                            "process",
                            newNameDictNoData,
                        )

                        # remove the up/down variations (if any) from the prompt subtraction histo
                        # but keep FFUp and FFDown, as these are the nonprompt up and down variations
                        syst_var_idet_rm_lst = []
                        for syst_var_idet in hPromptSub.axes["systematic"]:
                            if (syst_var_idet != "nominal") and (not syst_var_idet.startswith("FF")):
                                syst_var_idet_rm_lst.append(syst_var_idet)
                        hPromptSub = hPromptSub.remove(
                            "systematic", syst_var_idet_rm_lst
                        )

                        # now we actually make the subtraction
                        hPromptSub.scale(-1)
                        hFakes += hPromptSub

                        # now adding them to the list of processes:
                        if newhist is None:
                            newhist = hFakes
                        else:
                            newhist = newhist.union(hFakes, "process")
            self.outHist[key] = newhist

    def dumpToPickle(self):
        if not self.outputName.endswith(".pkl.gz"):
            self.outputName = self.outputName + ".pkl.gz"
        with gzip.open(self.outputName, "wb") as fout:
            cloudpickle.dump(self.outHist, fout)

    def getDataDrivenHistogram(self):
        return self.outHist


if __name__ == "__main__":

    yt = YieldTools()

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    DataDrivenProducer(args.pkl_file_path, '')
