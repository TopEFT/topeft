import argparse
import topcoffea.modules.utils as utils
import cloudpickle
from collections import defaultdict
import re
import gzip

from topeft.modules.paths import topeft_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_te_param = GetParam(topeft_path("params/params.json"))

class DataDrivenProducer:
    def __init__(self, inputHist, outputName):
        if isinstance(inputHist, str) and inputHist.endswith('.pkl.gz'): # we are plugging a pickle file
            self.inhist=utils.get_hist_from_pkl(inputHist)
        else: # we already have the histogram
            self.inhist=inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist

        self.outHist={}

        for key,histo in self.inhist.items():

            if histo.empty(): # histo is empty, so we just integrate over appl and keep an empty histo
                print(f'[W]: Histogram {key} is empty, returning an empty histo')
                self.outHist[key]=histo.integrate('appl')
                continue

            # First we are gonna scale all MC processes in  by the luminosity
            name_regex = r'^(?P<process>.*?)(?:UL)?(?P<year>(?:\d{2}(?:APV)?|\d{4}))$'
            pattern=re.compile(name_regex)

            for process in histo.axes['process']:
                try:
                    match = pattern.search(process)
                    sampleName=match.group('process')
                    year=match.group('year')
                except AttributeError as ae:
                    print(f"The following exception occured due to missing match {ae} for process {process}")
                    print("Moving to the next process")
                if not match:
                    raise RuntimeError(f"Sample {process} does not match the naming convention.")
                year = year.replace("central", "").replace("UL", "")
                if year not in ['16APV','16','17','18','2022','2023']:
                    raise RuntimeError(f"Sample {process} does not match the naming convention, year \"{year}\" is unknown.")

            print("\n")
            print("process", process, "sampleName", sampleName, "year", year)
            print("\n")
            # now for each year we actually perform the subtraction and integrate out the application regions
            newhist=None
            for ident in histo.axes['appl']:
                hAR=histo.integrate('appl', ident)

                if 'isAR' not in ident:
                    # if we are in the signal region, we just take the
                    # whole histogram integrating out the application region axis
                    if newhist==None:
                        newhist=hAR
                    else:
                        newhist += hAR
                else:
                    if "isAR_2lSS_OS"==ident:
                        # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                        newNameDictData=defaultdict(list)
                        for process in hAR.axes['process']:
                            match = pattern.search(process)
                            sampleName=match.group('process')
                            year=match.group('year')
                            nonPromptName='flipsUL%s'%year
                            if self.dataName==sampleName:
                                newNameDictData[nonPromptName].append(process)
                        hFlips=hAR.group('process', newNameDictData)

                        # remove any up/down FF variations from the flip histo since we don't use that info
                        syst_var_idet_rm_lst = []
                        syst_var_idet_lst = list(hFlips.axes["systematic"])
                        for syst_var_idet in syst_var_idet_lst:
                            if (syst_var_idet != "nominal"):
                                syst_var_idet_rm_lst.append(syst_var_idet)
                        hFlips = hFlips.remove("systematic", syst_var_idet_rm_lst)

                        # now adding them to the list of processes:
                        if newhist==None:
                            newhist=hFlips
                        else:
                            newhist += hFlips


                    else:
                        # if we are in the nonprompt application region, we also integrate the application region axis
                        # and construct the new process 'nonprompt'
                        # we look at data only, and rename it to fakes
                        newNameDictData=defaultdict(list); newNameDictNoData=defaultdict(list)
                        for process in hAR.axes['process']:
                            match = pattern.search(process)
                            sampleName=match.group('process')
                            year=match.group('year')

                            if "2022" in year or "2023" in year:
                                nonPromptName='nonprompt%s'%year
                            else:
                                nonPromptName='nonpromptUL%s'%year
                            if self.dataName==sampleName:
                                newNameDictData[nonPromptName].append(process)
                            elif sampleName in self.promptSubtractionSamples:
                                newNameDictNoData[nonPromptName].append(process)
                            else:
                                print(f"We won't consider {sampleName} for the prompt subtraction in the appl. region")
                        hFakes=hAR.group('process', newNameDictData)
                        # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                        hPromptSub=hAR.group('process', newNameDictNoData)

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
                        if newhist==None:
                            newhist=hFakes
                        else:
                            newhist += hFakes

            self.outHist[key]=newhist

    def dumpToPickle(self):
        if not self.outputName.endswith(".pkl.gz"):
            self.outputName = self.outputName + ".pkl.gz"
        with gzip.open(self.outputName, "wb") as fout:
            cloudpickle.dump(self.outHist, fout)


    def getDataDrivenHistogram(self):
        return self.outHist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    args = parser.parse_args()

    DataDrivenProducer(args.pkl_file_path, '')
