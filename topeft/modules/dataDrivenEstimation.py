import argparse
import topcoffea.modules.utils as utils
import cloudpickle
from collections import defaultdict
import re
import gzip

from topeft.modules.paths import topeft_path
from topcoffea.modules.get_param_from_jsons import GetParam
from topeft.modules.nonprompt_unc_propagation_helper import modify_NP_photon_pt_eta_variance,modify_photon_pt_variance
get_te_param = GetParam(topeft_path("params/params.json"))

class DataDrivenProducer:
    def __init__(self, inputHist, outputName, do_np_ph=False):
        if isinstance(inputHist, str) and inputHist.endswith('.pkl.gz'): # we are plugging a pickle file
            self.inhist=utils.get_hist_from_pkl(inputHist)
        else: # we already have the histogram
            self.inhist=inputHist
        self.outputName=outputName
        self.verbose=False
        self.dataName='data'
        self.outHist=None
        self.closure=False #a boolean to indicate whether we are doing closure test for nonprompt photon estimation
        self.do_np_ph=do_np_ph #this controls whether we will do non-prompt photon estimation or not
        self.promptSubtractionSamples=get_te_param('prompt_subtraction_samples')
        self.promptPhSubtractionSamples=get_te_param('prompt_photon_subtraction_samples')
        self.DDFakes()

    def DDFakes(self):
        if self.outHist!=None:  # already some processing has been done, so using what is available
            self.inhist=self.outHist

        self.outHist={}

        #This dictionary collects photon_pt_eta and photon_pt_eta_sumw2 hists needed for FR/kMC stat uncertainty propagation
        required_hists_for_nonprompt_ph={}
        np_uncertainty_propagation_done = False

        for key,histo in self.inhist.items():
            if histo.empty(): # histo is empty, so we just integrate over appl and keep an empty histo
                print(f'[W]: Histogram {key} is empty, returning an empty histo')
                self.outHist[key]=histo.integrate('appl')
                continue

            # First we are gonna scale all MC processes in  by the luminosity
            name_regex='(?P<process>.*)UL(?P<year>.*)'
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
                if year not in ['16APV','16','17','18']:
                    raise RuntimeError(f"Sample {process} does not match the naming convention, year \"{year}\" is unknown.")

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

                    elif ident in ["isAR_2lSS","isAR_3l","isAR_2lOS_medph"]:
                        # if we are in the nonprompt application region, we also integrate the application region axis
                        # and construct the new process 'nonprompt'
                        # we look at data only, and rename it to fakes
                        newNameDictData=defaultdict(list); newNameDictNoData=defaultdict(list)
                        for process in hAR.axes['process']:
                            match = pattern.search(process)
                            sampleName=match.group('process')
                            year=match.group('year')
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

                        #Also make sure to remove nonpromptPh systematic uncertainty (if it exists) from hFakes cause it is not relevant
                        hFakes = hFakes.remove("systematic",["nonpromptPhUp","nonpromptPhDown"])

                        # now adding them to the list of processes:
                        if newhist==None:
                            newhist=hFakes
                        else:
                            newhist += hFakes

                    #isAR_2lOS_ph is the regular AR using which we estimate non-prompt photon in our signal region A
                    #isAR_R_LRCD is the "AR" corresponding to the "SR" L in the LRCD closure test
                    elif ident in ["isAR_2lOS_ph", "isAR_B_ABCD"] and self.do_np_ph:
                        print(f"\n\nWe are inside {ident} appl axis and we will do nonprompt photon estimation here")
                        newDataDict=defaultdict(list); newNonDataDict=defaultdict(list)
                        for process in hAR.axes['process']:
                            match = pattern.search(process)
                            sampleName=match.group('process')
                            year=match.group('year')
                            nonPromptPhName='nonpromptPhUL%s'%year
                            if self.dataName==sampleName:
                                newDataDict[nonPromptPhName].append(process)
                            elif sampleName in self.promptPhSubtractionSamples:
                                newNonDataDict[nonPromptPhName].append(process)
                            elif sampleName.startswith(("ZGToLLGISR", "ZGToLLGFSR")):
                                newNonDataDict[nonPromptPhName].append(process)
                            else:
                                print(f"We won't consider {sampleName} for the prompt photon subtraction in the appl. region")
                        hPhFakes=hAR.group('process', newDataDict)
                        # now we take all the stuff that is not data in the AR to make the prompt photon subtraction and assign them to nonprompt.
                        hPromptPhSub=hAR.group('process', newNonDataDict)

                        # remove the up/down variations (if any) from the prompt photon subtraction histo
                        # but keep nonPromptPhUp and nonPromptPhDown, as these are the nonprompt photon up and down variations
                        syst_var_idet_rm_lst_PromptPhSub = []
                        syst_var_idet_lst = list(hPromptPhSub.axes["systematic"])
                        for syst_var_idet in syst_var_idet_lst:
                            if (syst_var_idet != "nominal") and (not syst_var_idet.startswith("nonpromptPh")):
                                syst_var_idet_rm_lst_PromptPhSub.append(syst_var_idet)
                        hPromptPhSub = hPromptPhSub.remove("systematic", syst_var_idet_rm_lst_PromptPhSub)

                        #Also remove the up/down variations from the hPhFakes histo cause we will use this again later when we modify the photon pt MC stat uncertainty
                        syst_var_idet_rm_lst_PhFakes = []
                        syst_var_idet_lst = list(hPhFakes.axes["systematic"])
                        for syst_var_idet in syst_var_idet_lst:
                            if (syst_var_idet != "nominal") and (not syst_var_idet.startswith("nonpromptPh")):
                                syst_var_idet_rm_lst_PhFakes.append(syst_var_idet)
                        hPhFakes = hPhFakes.remove("systematic", syst_var_idet_rm_lst_PhFakes)

                        # now we actually make the subtraction
                        if not key.endswith("_sumw2"):
                            hPromptPhSub.scale(-1)
                        hPhFakes += hPromptPhSub
                        if key == "photon_pt_eta":
                            required_hists_for_nonprompt_ph["photon_pt_eta"] = hPhFakes
                        elif key == "photon_pt_eta_sumw2":
                            required_hists_for_nonprompt_ph["photon_pt_eta_sumw2"] = hPhFakes
                        if not np_uncertainty_propagation_done and "photon_pt_eta" in required_hists_for_nonprompt_ph and "photon_pt_eta_sumw2" in required_hists_for_nonprompt_ph:
                            modify_NP_photon_pt_eta_variance(required_hists_for_nonprompt_ph,closure=self.closure)
                            np_uncertainty_propagation_done = True
                            # Update newhist with the modified histogram. We only need the sumw2 histogram!
                            if newhist is None:
                                newhist = required_hists_for_nonprompt_ph["photon_pt_eta_sumw2"]
                            else:
                                newhist += required_hists_for_nonprompt_ph["photon_pt_eta_sumw2"]

                        # now adding them to the list of processes. We cannot add the sumw2 histogram yet:
                        if key not in ["photon_pt_eta_sumw2"]:
                            if newhist==None:
                                newhist=hPhFakes
                            else:
                                newhist += hPhFakes

            #For the sumw2 2D histogram, only dump it to the outHist dict if we have done the non-prompt uncertainty propagation
            if self.do_np_ph and key == "photon_pt_eta_sumw2":
                if np_uncertainty_propagation_done:
                    self.outHist[key]=newhist
            else:
                self.outHist[key]=newhist

        #This is where we modify the photon_pt2 histogram's variance for the nonpromptPhUL{year} contribution
        if self.do_np_ph:
            for year in ['16','16APV','17','18']:
                modify_photon_pt_variance(self.outHist,year)
            #At this point, we don't need to store photon_pt_eta and photon_pt_eta_sumw2 histograms anymore
            del self.outHist['photon_pt_eta'], self.outHist['photon_pt_eta_sumw2']

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
