import argparse
from coffea import hist, processor
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_lumi
import cloudpickle
from collections import defaultdict 
import re 

class DataDrivenProducer: 
    def __init__(self, inputHist, outputName, doDDFakes=True, doDDFlips=False):
        self.yt=YieldTools()
        if type(inputHist) == str and inputHist.endswith('.pkl.gz'): # we are plugging a pickle file
            self.inhist=yt.get_hist_from_pkl(inputHist)
        else: # we already have the histogram
            self.inhist=inputHist
        self.outputName=outputName
        if doDDFlips: 
            raise RuntimeError("Data driven flips not yet implemented")
        self.doDDFlips=doDDFlips
        self.doDDFakes=doDDFakes
        self.dataName='data'
        self.chargeFlipName='chargeFip' # place holder, to implement in the future
        self.outHist=None

        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist
            
        self.outHist={}
        if 'SumOfEFTweights' in self.inhist:
            self.outHist['SumOfEFTweights']=self.inhist['SumOfEFTweights']
            SumOfEFTweights=self.inhist['SumOfEFTweights']
            SumOfEFTweights.set_sm()
            self.smsow = {proc: SumOfEFTweights.integrate('sample', proc).values()[()][0] for proc in SumOfEFTweights.identifiers('sample')}

        for key,histo in self.inhist.items():
            if key == 'SumOfEFTweights': 
                # we have already dealt with the sum of weights
                continue 

            if not len(histo.values()): # histo is empty, so we just integrate over appl and keep an empty histo
                print(f'[W]: Histogram {key} is empty, returning an empty histo')
                self.outHist[key]=histo.integrate('appl')
                continue

            if not self.doDDFakes:
                # if we are not gonna use the data-driven, then we don't care about the application region, so we get rid of it, and the associated dimension
                srs=[ histo.integrate('appl',ident) for ident in histo.identifiers('appl') if 'SR' in ident.name]
                if not len(srs):
                    raise RuntimeError(f"Histogram {key} does not have any signal region")
                newhist=srs[0]
                for h in srs[1:]:
                    newhist = newhist + h # sum doesnt work for some reason...
            else:

                # First we are gonna scale all MC processes in the AR by the luminosity
                # we wont do anything to stuff in the SR because we only want the scaling for the 
                # prompt subtraction
                name_regex='(?P<sample>.*)UL(?P<year>.*)'
                pattern=re.compile(name_regex)

                scale_dict={}
                for sample in histo.identifiers('sample'):
                    match = pattern.search(sample.name)
                    sampleName=match.group('sample')
                    year=match.group('year')
                    if not match: 
                        raise RuntimeError(f"Sample {sample} does not match the naming convention")
                    if year not in ['16','17','18']:
                        raise RuntimeError(f"Sample {sample} does not match the naming convention")

                    if self.dataName == sampleName or self.chargeFlipName == sampleName:
                        continue # We do not scale data or data-driven at all 

                    for appl in histo.integrate('sample',sample).identifiers('appl'):
                        if 'AR' not in appl.name: # we only care about AR 
                            continue
                        smweight = self.smsow[sample.name] if sample.name in self.smsow else 1 # dont reweight central samples
                        scale_dict[(appl, sample)] = 1000*get_lumi('20'+year)/smweight
                histo.scale( scale_dict, axis=('appl','sample'))


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
                        # if we are in the application region, we also integrate the application region axis
                        # and construct the new sample 'nonprompt'

                        # we look at data only, and rename it to fakes
                        newNameDictData=defaultdict(list); newNameDictNoData=defaultdict(list)
                        addedNonPrompts=[]
                        for sample in hAR.identifiers('sample'):
                            match = pattern.search(sample.name)
                            sampleName=match.group('sample')
                            year=match.group('year')
                            nonPromptName='nonpromptUL%s'%year
                            if self.dataName==sampleName:
                                newNameDictData[nonPromptName].append(sample.name)
                                addedNonPrompts.append( (nonPromptName, year)) # keep it to rescale later
                            else: 
                                # To do: hard code a list of samples we actually want to subtract? 
                                # if we run over, say, ttbar mc it doesn't make sense to subtract that
                                newNameDictNoData[nonPromptName].append(sample.name)
                        
                        hFakes=hAR.group('sample',  hist.Cat('sample','sample'), newNameDictData)
                    
                        # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                        hPromptSub=hAR.group('sample', hist.Cat('sample','sample'), newNameDictNoData )
                        
                        # now we actually make the subtraction
                        hPromptSub.scale(-1)
                        hFakes=hFakes+hPromptSub

                        #scale back by 1/lumi as if it were a MC, so it can be used transparently downstream
                        scaleDict={}
                        for name, year in addedNonPrompts:
                            scaleDict[name]=1/(1000*get_lumi('20'+year))
                        hFakes.scale( scaleDict, axis='sample')
                        

                        # now adding them to the list of processes: 
                        if newhist==None:
                            newhist=hFakes
                        else:
                            newhist=newhist+hFakes
            

            self.outHist[key]=newhist

        def dumpToPickle(self):
            with gzip.open(self.outputName + ".pkl.gz", "wb") as fout:
                cloudpickle.dump(self.outHist, fout)


        def getDataDrivenHistogram(self):
            return self.outHist


if __name__ == "__main__":

    yt = YieldTools()

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-y", "--year", default="2017", help = "The year of the sample")
    args = parser.parse_args()


    DataDrivenProducer(args.pkl_file_path, '',args.year)
