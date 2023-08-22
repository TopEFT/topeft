import argparse
from coffea import hist
from topeft.modules.YieldTools import YieldTools
from topeft.modules.get_param_from_jsons import get_te_param
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
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist

        self.outHist={}

        for key,histo in self.inhist.items():

            if not len(histo.values()): # histo is empty, so we just integrate over appl and keep an empty histo
                print(f'[W]: Histogram {key} is empty, returning an empty histo')
                self.outHist[key]=histo.integrate('appl')
                continue

            # First we are gonna scale all MC processes in  by the luminosity
            name_regex='(?P<sample>.*)UL(?P<year>.*)'
            pattern=re.compile(name_regex)

            scale_dict={}
            for sample in histo.identifiers('sample'):
                match = pattern.search(sample.name)
                sampleName=match.group('sample')
                year=match.group('year')
                if not match:
                    raise RuntimeError(f"Sample {sample} does not match the naming convention.")
                if year not in ['16APV','16','17','18']:
                    raise RuntimeError(f"Sample {sample} does not match the naming convention, year \"{year}\" is unknown.")

            prescale=histo.values().copy()
            histo.scale( scale_dict, axis=('sample',))
            postscale=histo.values()

            # now for each year we actually perform the subtraction and integrate out the application regions
            newhist=None
            for ident in histo.identifiers('appl'):
                hAR=histo.integrate('appl', ident)

                if 'isAR' not in ident.name:
                    # if we are in the signal region, we just take the
                    # whole histogram integrating out the application region axis
                    if newhist==None:
                        newhist=hAR
                    else:
                        newhist.add(hAR)
                else:
                    if "isAR_2lSS_OS"==ident.name:
                        # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                        newNameDictData=defaultdict(list)
                        for sample in hAR.identifiers('sample'):
                            match = pattern.search(sample.name)
                            sampleName=match.group('sample')
                            year=match.group('year')
                            nonPromptName='flipsUL%s'%year
                            if self.dataName==sampleName:
                                newNameDictData[nonPromptName].append(sample.name)
                        hFlips=hAR.group('sample',  hist.Cat('sample','sample'), newNameDictData)

                        # remove any up/down FF variations from the flip histo since we don't use that info
                        syst_var_idet_rm_lst = []
                        syst_var_idet_lst = hFlips.identifiers("systematic")
                        for syst_var_idet in syst_var_idet_lst:
                            if (syst_var_idet.name != "nominal"):
                                syst_var_idet_rm_lst.append(syst_var_idet)
                        hFlips = hFlips.remove(syst_var_idet_rm_lst,"systematic")

                        # now adding them to the list of processes:
                        if newhist==None:
                            newhist=hFlips
                        else:
                            newhist.add( hFlips )


                    else:
                        # if we are in the nonprompt application region, we also integrate the application region axis
                        # and construct the new sample 'nonprompt'
                        # we look at data only, and rename it to fakes
                        newNameDictData=defaultdict(list); newNameDictNoData=defaultdict(list)
                        for sample in hAR.identifiers('sample'):
                            match = pattern.search(sample.name)
                            sampleName=match.group('sample')
                            year=match.group('year')
                            nonPromptName='nonpromptUL%s'%year
                            if self.dataName==sampleName:
                                newNameDictData[nonPromptName].append(sample.name)
                            elif sampleName in self.promptSubtractionSamples:
                                newNameDictNoData[nonPromptName].append(sample.name)
                            else:
                                print(f"We won't consider {sampleName} for the prompt subtraction in the appl. region")
                        hFakes=hAR.group('sample',  hist.Cat('sample','sample'), newNameDictData)
                        # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                        hPromptSub=hAR.group('sample', hist.Cat('sample','sample'), newNameDictNoData )

                        # remove the up/down variations (if any) from the prompt subtraction histo
                        # but keep FFUp and FFDown, as these are the nonprompt up and down variations
                        syst_var_idet_rm_lst = []
                        syst_var_idet_lst = hPromptSub.identifiers("systematic")
                        for syst_var_idet in syst_var_idet_lst:
                            if (syst_var_idet.name != "nominal") and (not syst_var_idet.name.startswith("FF")):
                                syst_var_idet_rm_lst.append(syst_var_idet)
                        hPromptSub = hPromptSub.remove(syst_var_idet_rm_lst,"systematic")
                        hPromptSub = hPromptSub.copy_sm()

                        # now we actually make the subtraction
                        hPromptSub.scale(-1)
                        hFakes.add(hPromptSub)
                        # now adding them to the list of processes:
                        if newhist==None:
                            newhist=hFakes
                        else:
                            newhist.add(hFakes)

            self.outHist[key]=newhist

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
