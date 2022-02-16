import argparse
from coffea import hist, processor
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_lumi, get_param
import cloudpickle
from collections import defaultdict 
import re, gzip

class DataDrivenProducer: 
    def __init__(self, inputHist, outputName):
        yt=YieldTools()
        if type(inputHist) == str and inputHist.endswith('.pkl.gz'): # we are plugging a pickle file
            self.inhist=yt.get_hist_from_pkl(inputHist)
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

                if self.dataName == sampleName:
                    continue # We do not scale data
                scale_dict[(sample, )] = 1000.0*get_lumi('20'+year)

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
                        newhist=newhist+hAR
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
                        
                        # now adding them to the list of processes: 
                        if newhist==None:
                            newhist=hFlips
                        else:
                            newhist=newhist+hFlips
                            

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
                        
                        # now we actually make the subtraction
                        hPromptSub.scale(-1)
                        hFakes=hFakes+hPromptSub
                        
                        # now adding them to the list of processes: 
                        if newhist==None:
                            newhist=hFakes
                        else:
                            newhist=newhist+hFakes

            # Scale back by 1/lumi all processes but data so they can be used transparently downstream
            # Mind that we scaled all mcs already above
            scaleDict={}
            for sample in newhist.identifiers('sample'):
                match = pattern.search(sample.name)
                sampleName=match.group('sample')
                if self.dataName == sampleName:
                    continue
                year=match.group('year')
                scaleDict[sample]=1.0/(1000.0*get_lumi('20'+year))
            newhist.scale( scaleDict, axis='sample')


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
