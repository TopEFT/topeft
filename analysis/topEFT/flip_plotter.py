import os
import copy
import topcoffea.modules.GetValuesFromJsons as getVal
import uproot3
from coffea import hist

from topcoffea.modules.YieldTools import YieldTools
import make_cr_and_sr_plots as mp


yt = YieldTools()

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
#parser.add_argument("filepath"          , default='histos/plotsTopEFT.pkl.gz', help = 'path of file with histograms')
parser.add_argument("--outpath" ,'-o'   , default='.', help = 'Path to the output directory')
args = parser.parse_args()

# Data
#hin_dict = yt.get_hist_from_pkl(args.filepath)
#hin_dict = yt.get_hist_from_pkl("histos/mar16_UL17UL18-data_njets-invmass.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar16_UL17UL18-data_njets-invmass_minPt15.pkl.gz")

# MC
#hin_dict_mc = yt.get_hist_from_pkl("histos/mar16_fullR2-DY_njets-invmass.pkl.gz") # MC
#hin_dict_mc = yt.get_hist_from_pkl("histos/mar16_UL17UL18-dy_njets-invmass.pkl.gz")
#hin_dict_mc = yt.get_hist_from_pkl("histos/mar16_UL17UL18-dy_njets-invmass_minPt15.pkl.gz")

###

#hin_dict_mc = yt.get_hist_from_pkl("histos/mar16_UL17UL18-dy_allKinCats.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar16_UL17UL18-data_allKinCats.pkl.gz")

#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi35.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi45.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi40.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi38.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi37.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/test_l1PtHi39.pkl.gz")


# Mar 18
#hin_dict = yt.get_hist_from_pkl("histos/mar18_UL17data_allKinCats_eventSelPtCutMaster.pkl.gz")

# Mar 22
#hin_dict = yt.get_hist_from_pkl("histos/mar22_UL17data_checkWithFlipWeightsTTH.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar22_UL17data_checkWithFlipWeightsMar21Fit7Cats.pkl.gz")

# Mar 23
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsFromTTH.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsMar21Fit7Cats.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsFromTTH_withSSZTruthChannel.pkl.gz.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsMar21Fit7Cats_withSSZTruthChannel.pkl.gz.pkl.gz")

#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsMar21Fit7Cats_withSSZTruthFlipPromptChannel.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsFromTTH_withSSZTruthFlipPromptChannel.pkl.gz")

# Mar 24
#hin_dict = yt.get_hist_from_pkl("histos/mar23_UL17DY_checkWithFlipWeightsMar23Fit13Cats_withSSZTruthFlipPromptChannel.pkl.gz")
#hin_dict = yt.get_hist_from_pkl("histos/mar24_UL17allBkgMC.pkl.gz")

# Week of Mar 28
#hin_dict = yt.get_hist_from_pkl("histos/mar28_UL17ttjets_channels-os-ss-ssTruth.pkl.gz")
hin_dict = yt.get_hist_from_pkl("histos/mar29_UL17DY_withSSZTruthMethod2FlipPromptChannel.pkl.gz")



integrate_map = {

    "inclusive" : None,

    "EH_EH" : ["EH_EH"],
    "BH_EH" : None, # Combining with the upper triangle element
    "EM_EH" : None, # Combining with the upper triangle element
    "BM_EH" : None, # Combining with the upper triangle element
    "EL_EH" : None, # Combining with the upper triangle element
    "BL_EH" : None, # Combining with the upper triangle element

    "EH_BH" : ["EH_BH","BH_EH"],
    "BH_BH" : ["BH_BH"],
    "EM_BH" : None, # Combining with the upper triangle element
    "BM_BH" : None, # Combining with the upper triangle element
    "EL_BH" : None, # Combining with the upper triangle element
    "BL_BH" : None, # Combining with the upper triangle element

    "EH_EM" : ["EH_EM","EM_EH"],
    "BH_EM" : ["BH_EM","EM_BH"],
    "EM_EM" : ["EM_EM"],
    "BM_EM" : None, # Combining with the upper triangle element
    "EL_EM" : None, # Combining with the upper triangle element
    "BL_EM" : None, # Combining with the upper triangle element

    "EH_BM" : ["EH_BM","BM_EH"],
    "BH_BM" : ["BH_BM","BM_BH"],
    "EM_BM" : ["EM_BM","BM_EM"],
    "BM_BM" : ["BM_BM"],
    "EL_BM" : None, # Combining with the upper triangle element
    "BL_BM" : None, # Combining with the upper triangle element

    "EH_EL" : ["EH_EL","EL_EH"],
    "BH_EL" : ["BH_EL","EL_BH"],
    "EM_EL" : ["EM_EL","EL_EM"],
    "BM_EL" : ["BM_EL","EL_BM"],
    "EL_EL" : None, # ["EL_EL"], # Skipping LL combos
    "BL_EL" : None, # Would combine with upper triangle element, but skipping LL combos

    "EH_BL" : ["EH_BL","BL_EH"],
    "BH_BL" : ["BH_BL","BL_BH"],
    "EM_BL" : ["EM_BL","BL_EM"],
    "BM_BL" : ["BM_BL","BL_BM"],
    "EL_BL" : None, # ["EL_BL","BL_EL"], # Skipping LL combos
    "BL_BL" : None, # ["BL_BL"],         # Skipping LL combos
}

outpath = args.outpath

# Get a MC name that corresponds to whatever data we're looking at
def get_mc_name(data_name):
    if "UL17" in data_name:
        return "DYJetsToLL_centralUL17"
    elif "UL18" in data_name:
        return "DY50_centralUL18"
    else: raise Exception

# Print summed values from a histo
def print_summed_hist_vals(in_hist):
    val_dict = {}
    for k,v in in_hist.values().items():
        val_dict[k[0]] = sum(v)
    for k,v in val_dict.items():
        print(f"\t{k}: {v}")
    print("\tFlip rate:", val_dict["sszTruthFlip"]/(val_dict["osz"] + val_dict["ssz"]))


# Main wrapper function
def make_plot():

    make_plots = True
    #make_plots = False
    save_root = False
    #save_root = True

    if save_root: fout = uproot3.create('flip_hists.root')

    sample_names_lst = yt.get_cat_lables(hin_dict,"sample")
    chan_names_lst = yt.get_cat_lables(hin_dict,"channel")
    cat_names_lst = yt.get_cat_lables(hin_dict,"kinematiccat")

    print("Samples:",sample_names_lst)
    print("Channels:",chan_names_lst)
    print("Cat names lst:",cat_names_lst)

    for histo_name,histo_orig in hin_dict.items():
        print(f"\nName: {histo_name}")
        #if histo_name != "njets": continue
        if histo_name != "invmass": continue

        # Loop over samples
        for sample_name in sample_names_lst:

            # Loop over lepton channels (i.e. ssz and osz)
            for lep_chan_name in chan_names_lst:

                # Copy (and rebin the ss)
                #histo = copy.deepcopy(histo_orig)
                #if lep_chan_name == "ssz" and histo_name == "invmass": histo = histo.rebin("invmass",10)
                #print("\nsum incl:", histo.integrate("kinematiccat","inclusive").values())
                #histo = histo.sum("kinematiccat")
                #print("\nsum all cats:", histo.values())

                # Integrate and make plot (overlay the categories)
                #savename = "_".join([sample_name,lep_chan_name,histo_name])
                #histo = histo.integrate("sample",sample_name)
                #histo = histo.integrate("channel",lep_chan_name)
                #fig = mp.make_single_fig(histo)
                #fig.savefig(os.path.join(outpath,savename))

                #'''
                # Print summed info
                histo_tmp = copy.deepcopy(histo_orig)
                #for k,v in histo_tmp.values().items(): print("k:",k)
                #histo_tmp = histo_tmp.sum("kinematiccat")
                histo_tmp = histo_tmp.integrate("kinematiccat","inclusive")
                histo_tmp = histo_tmp.rebin("invmass",10)
                histo_tmp.scale(mp.get_lumi_for_sample(sample_name)) # Only if MC
                for k,v in histo_tmp.values().items(): print("\n",k,v,"\n",sum(v),"\n")
                #'''

                # Loop over the kinematic categories
                print("\nLooping over kinematic categories:\n")
                for cat_name in cat_names_lst:

                    if "incl" in cat_name: continue # TMP
                    if integrate_map[cat_name] is None: continue
                    #if "UL17" not in sample_name and "UL18" not in sample_name: continue # TMP
                    if "UL17" not in sample_name: continue # TMP
                    #if lep_chan_name != "ssz": continue # TMP
                    #if lep_chan_name != "sszTruthFlip": continue # TMP
                    #if lep_chan_name == "sszTruthFlip": continue # TMP

                    # Copy and rebin
                    histo = copy.deepcopy(histo_orig)
                    #if lep_chan_name == "ssz" and histo_name == "invmass": histo = histo.rebin("invmass",10)
                    histo = histo.rebin("invmass",10)

                    # Integrate
                    histo = histo.integrate("sample",sample_name)
                    #histo = histo.integrate("channel",lep_chan_name)
                    #histo = histo.integrate("kinematiccat",cat_name)
                    histo = histo.integrate("kinematiccat",integrate_map[cat_name])
                    histo.scale(mp.get_lumi_for_sample(sample_name)) # For MC
                    print("\nSample, channel, cat:",sample_name,lep_chan_name,cat_name)
                    print_summed_hist_vals(histo)
                    continue

                    '''
                    # MC
                    histo_mc = copy.deepcopy(hin_dict_mc[histo_name])
                    histo_mc.scale(mp.get_lumi_for_sample(sample_name))
                    if lep_chan_name == "ssz" and histo_name == "invmass": histo_mc = histo_mc.rebin("invmass",10)
                    histo_mc = histo_mc.integrate("sample",get_mc_name(sample_name))
                    #histo_mc = histo_mc.integrate("channel",lep_chan_name)
                    histo_mc = histo_mc.integrate("kinematiccat",integrate_map[cat_name])
                    '''

                    # OS
                    #sample_name_os = histo_name.replace("ssz","osz")
                    #histo_os= copy.deepcopy(hin_dict[sample_name_os])
                    #histo_os.scale(mp.get_lumi_for_sample(sample_name))
                    #histo_os = histo_os.rebin("invmass",10)
                    #histo_os = histo_os.integrate("sample",get_mc_name(sample_name_os))
                    ##histo_mc = histo_mc.integrate("channel",lep_chan_name)
                    #histo_mc = histo_mc.integrate("kinematiccat",integrate_map[cat_name])


                    savename = "_".join([sample_name,lep_chan_name,histo_name,cat_name])

                    if lep_chan_name == "sszTruthFlip": continue # TMP
                    # Make plot
                    if make_plots:
                        #fig = mp.make_single_fig(histo[cat_name])
                        #fig = mp.make_single_fig(histo[lep_chan_name])

                        #fig = mp.make_single_fig(histo[lep_chan_name],histo[lep_chan_name.replace("ssz","osz")])
                        #fig = mp.make_single_fig(histo[lep_chan_name],histo[lep_chan_name.replace("sszTruthFlip","osz")],histo[lep_chan_name.replace("sszTruthFlip","ssz")])

                        if "osz" in lep_chan_name:
                            histo.integrate(lep_chan_name)
                            fig = mp.make_single_fig(histo)
                            #fig = mp.make_single_fig(histo[lep_chan_name])
                        elif "ssz" in lep_chan_name:
                            fig = mp.make_single_fig(histo[lep_chan_name],histo[lep_chan_name.replace("ssz","sszTruthFlip")])

                        #fig = mp.make_single_fig(histo[cat_name],histo_mc[cat_name])
                        #fig = mp.make_single_fig(histo[lep_chan_name],histo_mc[lep_chan_name])
                        fig.savefig(os.path.join(outpath,savename))

                    # Save output to root file
                    #if save_root: fout[savename] = hist.export1d(histo.integrate("kinematiccat",cat_name))
                    if save_root: fout[savename] = hist.export1d(histo.integrate("channel",lep_chan_name))

    if save_root: fout.close()

# Main function
def main():
    make_plot()

main()
