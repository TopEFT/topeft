import argparse
from coffea import hist, processor
from topcoffea.modules.YieldTools import YieldTools
from topcoffea.modules.GetValuesFromJsons import get_lumi
import cloudpickle

class DataDrivenProducer: 
    def __init__(self, inputHist, outputName, year, doDDFakes=True, doDDFlips=False):
        self.yt=YieldTools()
        if type(inputHist) == str: # we are plugging a pickle file
            self.inhist=yt.get_hist_from_pkl(inputHist)
        else: # we already have the histogram
            self.inhist=inputHist
        self.outputName=outputName
        if doDDFlips: 
            raise RuntimeError("Data driven flips not yet implemented")
        self.doDDFlips=doDDFlips
        self.doDDFakes=doDDFakes
        self.dataName='data'
        self.outHist=None
        self.year=year

        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist
            
        self.outHist={}
        for key,histo in self.inhist.items():
            if key == 'SumOfEFTweights': 
                # we will deal later with the sum of weights (to do) 
                continue 

            if not len(histo.values()): # histo is empty, so we just integrate over appl and keep an empty histo
                self.outHist[key]=histo.integrate('appl')
                continue

            if not self.doDDFakes:
                # if we are not gonna use the data-driven, then we don't care about the application region, so we get rid of it, and the associated dimension
                srs=[ histo.integrate('appl',ident) for ident in histo.identifiers('appl') if 'SR' in ident.name]
                if not len(srs):
                    continue # to do, this should be a runtime errror
                newhist=srs[0]
                for h in srs[1:]:
                    newhist = newhist + h # sum doesnt work for some reason...
            else:
                # this is where the fun part happens, we consider each application region separately
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
                        hFakes=hAR.group('sample',  hist.Cat('sample','sample'), {'nonprompt' : self.dataName})
                    
                        # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                        # To do: build a list of samples that we wanna subtract? 
                        hPromptSub=hAR.group('sample', hist.Cat('sample','sample'), {'nonprompt' : [x.name for x in hAR.identifiers('sample') if x.name != self.dataName]} )
                        
                        # now we actually make the subtraction, after normalizing by the luminosity!!
                        lumi = 1000.0*get_lumi(self.year)
                        hPromptSub.scale(-1*lumi)
                        hFakes=hFakes+hPromptSub

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
