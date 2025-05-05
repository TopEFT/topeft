#Where is it used?: This script is used in dataDrivenEstimation script.
#What is the purpose?: The main purpose of this is to correctly handle the non-prompt photon MC stat uncertainty in the photon_pt histogram. This script makes sure that the FR and kMC uncertainties are included in the photon_pt histograms
#What does it take?: It takes the photon_pt_eta and photon_pt_eta_sumw2 histograms and uses it to modify the sumw2 values stored in the photon_pt_sumw2 histogram
#Future modifications: Potential places with pitfalls in the future are commented with "CAUTION" tag

import re
import numpy as np

from topeft.modules.paths import topeft_path

def load_numpy_files(file_path):
    f = np.load(file_path)

    val = f[f.files[0]]
    err = f[f.files[1]]

    return val, err

#This function takes a dictionary that has two histograms: photon_pt_eta and photon_pt_eta_sumw2. At this point, both of these histograms should only have a single process axis "nonpromptPhUL<year>" and the yield here will be with non-prompt photon estimation done. i.e. Data - Prompt MC in region B or R depending on whether one is doing (not doing) closure test
#CAUTION: The fr_file_path and kmc_file_path are hardcoded right now.
def modify_NP_photon_pt_eta_variance(dict_of_hists_for_NP_uncertainty, closure=False):
    print("Inside NP photon variance modification block")
    photon_pt_eta = dict_of_hists_for_NP_uncertainty["photon_pt_eta"]
    photon_pt_eta_sumw2 = dict_of_hists_for_NP_uncertainty["photon_pt_eta_sumw2"]

    for keys in list(photon_pt_eta.view(flow=True).keys()):
        for part in keys:
            if "nonpromptPhUL" in str(part):
                match = re.search(r'nonpromptPhUL(\d{2}APV|\d{2})', str(part))  # Check the exact structure
            if match: year = match.group(1)

        #We need to load the fake-rate and kMC files inside cause they depend on year!
        fr_file_path = topeft_path("data/photon_fakerates/")+f"fr_ph_UL{year}.npz"

        #Depending on whether we are doing closure test or not, the kMC file changes
        if closure:
            kmc_file_path  = topeft_path("data/photon_kmc_validation/")+f"kmc_ph_UL{year}.npz"

        else:
            kmc_file_path  = topeft_path("data/photon_kmc/")+f"kmc_ph_UL{year}.npz"

        #Load the FR and kMC files
        ph_fr_val, ph_fr_err = load_numpy_files(fr_file_path)
        ph_kmc_val, ph_kmc_err = load_numpy_files(kmc_file_path)

        #ph_ff is "photon fake factor"
        ph_ff_val = ph_fr_val*ph_kmc_val
        ph_ff_err = np.nan_to_num(ph_ff_val * np.sqrt(pow((ph_fr_err/ph_fr_val),2) + pow((ph_kmc_err/ph_kmc_val),2)),0)

        if any(str(part) == f"nonpromptPhUL{year}" for part in keys):
            sumw2_pt_eta_hist = photon_pt_eta_sumw2.view(flow=True)[keys]
            sumw_pt_eta_hist = photon_pt_eta.view(flow=True)[keys]

            #This is the master equation to calculate the final NP variance
            np_ph_var = ((sumw2_pt_eta_hist*pow(ph_ff_val,2)))+(pow(sumw_pt_eta_hist*ph_ff_err,2))
            np_ph_var = np.nan_to_num(np_ph_var,0)

            #this is how we modify the values of sumw2 and sumw. After this point the photon_pt_eta and and photon_pt_eta_sumw2 hists values will be modified by new value.
            sumw2_pt_eta_hist[:] = np_ph_var
            sumw_pt_eta_hist[:] = ph_ff_val*sumw_pt_eta_hist


#This histogram takes the output histogram dictionary called "outHist" in the dataDrivenEstimation script, where we will have a key called "photon_pt_sumw2". This histogram will also have the non-prompt estimation done, so it will have a key called "nonPromptPhUL<year>", and we need to modify the values associated with that key. The photon_pt_sumw2 histogram has same binning as FR/kMC
#Also note that at this point, the photon_pt_eta_sumw2 histogram will already have the values modified as described above
#CAUTION: Revisit this in this future cause we will just have one photon_pt histogram and the name will need to be changed accordingly
def modify_photon_pt_variance(outputHist,year):
    photon_pt_sumw2_hist = outputHist['photon_pt_sumw2']
    for k in list(outputHist['photon_pt_sumw2'].eval({}).keys()):
        if any(str(part) == f"nonpromptPhUL{year}" for part in k):
            #First, load the values of photon_pt_sumw2 hist
            photon_pt_sumw2 = photon_pt_sumw2_hist.eval({})[k]
            #Second, load the modified values from photon_pt_eta_sumw2 hist
            np_2D_ph_var = outputHist['photon_pt_eta_sumw2'].view(flow=True)[k]

            #WE first have to add over eta bins to get uncertainties associated with the pt bins
            new_sumw2_summed_over_eta = np.sum(np_2D_ph_var,1)  #At this point this array has a shape of 1 X (N_photon_pt_bins+1)

            #We are now ready to modify the values of the photon_pt_sumw2 histogram
            photon_pt_sumw2_hist[k] = new_sumw2_summed_over_eta[:,np.newaxis]
