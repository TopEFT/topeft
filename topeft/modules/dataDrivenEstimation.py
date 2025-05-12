import argparse
import topcoffea.modules.utils as utils
import cloudpickle
from collections import defaultdict
import re
import gzip

from topeft.modules.paths import topeft_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_te_param = GetParam(topeft_path("params/params.json"))

def hist_remove(histo, axis, lst):
    if axis == "channel":
        index = 1
    elif axis == "appl":
        index = 2
    elif axis == "process":
        index = 3
    elif axis == "systematic":
        index = 4
    else:
        print("axis not recognized")
        return

    keys = list(histo.keys())
    for key in keys:
        if key == 'meta':
            continue
        if key[index] in lst:
            del histo[key]
    return histo

def hist_integrate(old_histo, axis, lst=None):
    histo = old_histo
    del_keys = []
    keys = list(histo.keys())
    for key in keys:
        if key == 'meta':
            continue
        if axis == "channel":
            new_key = (key[0], "channel", key[2], key[3], key[4])
            if lst is not None and key[1] not in lst:
                del_keys.append(key)
                continue
            if new_key not in histo.keys():
                histo[new_key] = histo[key]
                del_keys.append(key)
            else:
                histo[new_key] += histo[key]
                del_keys.append(key)
        elif axis == "appl":
            new_key = (key[0], key[1], "appl", key[3], key[4])
            if lst is not None and key[2] not in lst:
                del_keys.append(key)
                continue
            if new_key not in histo.keys():
                histo[new_key] = histo[key]
                del_keys.append(key)
            else:
                histo[new_key] += histo[key]
                del_keys.append(key)
        elif axis == "process":
            new_key = (key[0], key[1], key[2], "process", key[4])
            if lst is not None and key[3] not in lst:
                del_keys.append(key)
                continue
            if new_key not in histo.keys():
                histo[new_key] = histo[key]
                del_keys.append(key)
            else:
                histo[new_key] += histo[key]
                del_keys.append(key)
        elif axis == "systematic":
            new_key = (key[0], key[1], key[2], key[3], "systematic")
            if lst is not None and key[4] not in lst:
                del_keys.append(key)
                continue
            if new_key not in histo.keys():
                histo[new_key] = histo[key]
                del_keys.append(key)
            else:
                histo[new_key] += histo[key]
                del_keys.append(key)
        else:
            print("axis not recognized")
            return
    for key in del_keys:
        del histo[key]
    return histo
            
def hist_group(old_histo, axis, lst):
    histo = old_histo
    del_keys = []
    keys = list(histo.keys())
    for key in keys:
        if key == 'meta':
            continue
        if axis == "channel":
            for item in lst.keys():
                new_key = (key[0], item, key[2], key[3], key[4])
                if key[1] not in lst[key]:
                    continue
                if new_key not in histo.keys():
                    histo[new_key] = histo[key]
                    del_keys.append(key)
                else:
                    histo[new_key] += histo[key]
                    del_keys.append(key)
        elif axis == "appl":
            for	item in	lst.keys():
                new_key = (key[0], key[1], item, key[3], key[4])
                if key[2] not in lst[key]:
                    continue
                if new_key not in histo.keys():
                    histo[new_key] = histo[key]
                    del_keys.append(key)
                else:
                    histo[new_key] += histo[key]
                    del_keys.append(key)
        elif axis == "process":
            for	item in	lst.keys():
                new_key = (key[0], key[1], key[2], item, key[4])
                if key[3] not in lst[key]:
                    continue
                if new_key not in histo.keys():
                    histo[new_key] = histo[key]
                    del_keys.append(key)
                else:
                    histo[new_key] += histo[key]
                    del_keys.append(key)
        elif axis == "systematic":
            for	item in	lst.keys():
                new_key = (key[0], key[1], key[2], key[3], item)
                if key[4] not in lst[key]:
                    continue
                if new_key not in histo.keys():
                    histo[new_key] = histo[key]
                    del_keys.append(key)
                else:
                    histo[new_key] += histo[key]
                    del_keys.append(key)
        else:
            print("axis not recognized")
            return
    for key in del_keys:
        del histo[key]
    return histo

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
        name_regex='(?P<process>.*)UL(?P<year>.*)'
        pattern=re.compile(name_regex)

        #for key,histo in self.inhist.items():
        #for key in self.inhist.keys():
            #if key == 'meta':
            #    continue
            #histo = self.inhist[key]
            
            #if histo.empty(): # histo is empty, so we just integrate over appl and keep an empty histo
            #    print(f'[W]: Histogram {key} is empty, returning an empty histo')
                #self.outHist[key]=histo.integrate('appl')
            #    self.outHist[key] = hist_integrate(histo, 'appl')
            #    continue
            
            # First we are gonna scale all MC processes in  by the luminosity
            #name_regex='(?P<process>.*)UL(?P<year>.*)'
            #pattern=re.compile(name_regex)
            
            #for process in histo.axes['process']:
        for process in self.inhist['meta']['process']:
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
        #for ident in histo.axes['appl']:
        for ident in self.inhist['meta']['appl']:
            hAR = hist_integrate(self.inhist, 'appl', ident)
            #hAR=histo.integrate('appl', ident)

            if 'isAR' not in ident:
                # if we are in the signal region, we just take the
                # whole histogram integrating out the application region axis
                if newhist==None:
                    newhist=hAR
                else:
                    for item in hAR.keys():
                        if item == 'meta':
                            continue
                        if item in newhist.keys():
                            print(item)
                            newhist[item] += hAR[item]
                        else:
                            newhist[item] = hAR[item]
            else:
                if "isAR_2lSS_OS"==ident:
                    # we are in the flips application region and theres no "prompt" subtraction, so we just have to rename data to flips, put it in the right axis and we are done
                    newNameDictData=defaultdict(list)
                    #for process in hAR.axes['process']:
                    for process in hAR['meta']['process']:
                        match = pattern.search(process)
                        sampleName=match.group('process')
                        year=match.group('year')
                        nonPromptName='flipsUL%s'%year
                        if self.dataName==sampleName:
                            newNameDictData[nonPromptName].append(process)
                    #hFlips=hAR.group('process', newNameDictData)
                    hFlips = hist_group(hAR, 'process', newNameDictData)
                    
                    # remove any up/down FF variations from the flip histo since we don't use that info
                    syst_var_idet_rm_lst = []
                    #syst_var_idet_lst = list(hFlips.axes["systematic"])
                    syst_var_idet_lst = hFlips['meta']["systematic"]
                    for syst_var_idet in syst_var_idet_lst:
                        if (syst_var_idet != "nominal"):
                            syst_var_idet_rm_lst.append(syst_var_idet)
                    #hFlips = hFlips.remove("systematic", syst_var_idet_rm_lst)
                    hFlips = hist_remove(hFlips, 'systematic', syst_var_idet_rm_lst)

                    # now adding them to the list of processes:
                    if newhist==None:
                        newhist=hFlips
                    else:
                        for item in hFlips.keys():
                            if item == 'meta':
                                continue
                            if item in newhist.keys():
                                newhist[item] += hFlips[item]
                            else:
                                newhist[item] = hFlips[item]


                else:
                    # if we are in the nonprompt application region, we also integrate the application region axis
                    # and construct the new process 'nonprompt'
                    # we look at data only, and rename it to fakes
                    newNameDictData=defaultdict(list); newNameDictNoData=defaultdict(list)
                    #for process in hAR.axes['process']:
                    for process in hAR['meta']['process']:
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
                    #hFakes=hAR.group('process', newNameDictData)
                    hFakes = hist_group(hAR, 'process', newNameDictData)
                    # now we take all the stuff that is not data in the AR to make the prompt subtraction and assign them to nonprompt.
                    #hPromptSub=hAR.group('process', newNameDictNoData)
                    hPromptSub = hist_group(hAR, 'process', newNameDictNoData)

                    # remove the up/down variations (if any) from the prompt subtraction histo
                    # but keep FFUp and FFDown, as these are the nonprompt up and down variations
                    syst_var_idet_rm_lst = []
                    #syst_var_idet_lst = list(hPromptSub.axes["systematic"])
                    syst_var_idet_lst = hFlips['meta']["systematic"]
                    for syst_var_idet in syst_var_idet_lst:
                        if (syst_var_idet != "nominal") and (not syst_var_idet.startswith("FF")):
                            syst_var_idet_rm_lst.append(syst_var_idet)
                    #hPromptSub = hPromptSub.remove("systematic", syst_var_idet_rm_lst)
                    hPromptSub = hist_remove(hPromptSub, 'systematic', syst_var_idet_rm_lst)
                    
                    # now we actually make the subtraction
                    # var(A - B) = var(A) + var(B)
                    for key in hPromptSub.keys():
                        if key == 'meta':
                            continue
                        if not key[0].endswith("_sumw2"):
                            hPromptSub[key].scale(-1)
                    #hFakes += hPromptSub
                    for item in hPromptSub.keys():
                        if item == 'meta':
                            continue
                        if item in hFakes.keys():
                            hFakes[item] += hPromptSub[item]
                        else:
                            hFakes[item] = hPromptSub[item]
                    # now adding them to the list of processes:
                    if newhist==None:
                        newhist=hFakes
                    else:
                        for item in hFakes.keys():
                            if item == 'meta':
                                continue
                            if item in newhist.keys():
                                newhist[item] += hFakes[item]
                            else:
                                newhist[item] = hFakes[item]

            self.outHist=newhist

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
