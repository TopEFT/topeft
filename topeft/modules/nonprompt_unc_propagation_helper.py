#Where is it used?: This script is used in dataDrivenEstimation script.
#What is the purpose?: The main purpose of this is to correctly handle the non-prompt photon uncertainty in the photon_pt histogram. This script makes sure that the FR and kMC uncertainties are included in the photon_pt histograms
#What does it take?: It takes the photon_pt_eta and photon_pt_eta_sumw2 histograms and uses it to modify the values in the photon_pt2_sumw2 histogram
#Future modifications: Potential places with pitfalls in the future are commented with "CAUTION" tag

import re
import numpy as np

from topeft.modules.paths import topeft_path

#With overflow bins, the photon_pt_eta (and sumw2) 2D array has an original shape of (N_photon_pt_bins + 1) x (N_photon_eta_bins + 1)
#Similarly, with overflow bins, the photon_pt and (photon_pt_sumw2) has an original shape of (N_photon_pt_bins + 1) X 3, which is the shape of "target_arr"
#We take the 2D sumw2 histogram and sum over the eta bins, which gives us the "original_arr". This array has a shape of 1 x (N_photon_pt_bins + 1)
def get_desired_array_shape(original_arr, target_arr):
    #first make it a column matrix
    original_arr = original_arr.T

    #create an array of all zeros with shape of target_arr
    desired_arr = np.zeros_like(target_arr)

    desired_arr[:,1] = original_arr.flatten()

    return desired_arr

def load_numpy_files(file_path):
    f = np.load(file_path)

    val = f[f.files[0]]
    err = f[f.files[1]]  #be careful here cause this could be either error or variance

    return val, err

#This function takes a dictionary that has two histograms: photon_pt_eta and photon_pt_eta_sumw2. At this point, both of these histograms should only have a single process axis "nonpromptPhUL<year>" and the yield here will be with non-prompt photon estimation done. i.e. Data - Prompt MC in region B or R
#CAUTION: The fr_file_path and kmc_file_path are hardcoded right now.
def modify_NP_photon_pt_eta_variance(dict_of_hists_for_NP_uncertainty, closure=False):
    print("Inside NP variance calculation block")
    photon_pt_eta = dict_of_hists_for_NP_uncertainty["photon_pt_eta"]
    photon_pt_eta_sumw2 = dict_of_hists_for_NP_uncertainty["photon_pt_eta_sumw2"]

    for keys in list(photon_pt_eta.view(flow=True).keys()):
        for part in keys:
            if "nonpromptPhUL" in str(part):
                match = re.search(r'nonpromptPhUL(\d{2}APV|\d{2})', str(part))  # Check the exact structure
            if match: year = match.group(1)

        #We need to load the fake-rate and kMC files inside cause they depend on year!
        fr_file_path = topeft_path("data/photon_fakerates_gyR6uGhvfy/")+f"fr_ph_UL{year}.npz"

        #Depending on whether we are doing closure test or not, the kMC file changes
        if closure:
            kmc_file_path = topeft_path("data/photon_fakerates_jeJHI2cDh5/")+f"kmc_ph_UL{year}.npz"

        else:
            kmc_file_path = topeft_path("data/photon_fakerates_gB29WFMqFb/")+f"kmc_ph_UL{year}.npz"


        fr_val, fr_err = load_numpy_files(fr_file_path)
        kmc_val, kmc_err = load_numpy_files(kmc_file_path)

        ff_val = fr_val*kmc_val
        ff_err = ff_val * np.sqrt(pow((fr_err/fr_val),2) + pow((kmc_err/kmc_val),2))

        #Let's pad zeros as a first row and first column to make the fake factor arrays have same shape as the sumw and sumw2 arrays (which have overflows included)
        #CAUTION: In the future, these two lines can be entirely avoided if I use scikit-hep hists to make the FR and kMC files. The files loaded above were made with old coffea hist.
        ff_val_padded = np.pad(ff_val, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)
        ff_err_padded = np.pad(ff_err, pad_width=((1, 0), (1, 0)), mode='constant', constant_values=0)

        if any(str(part) == f"nonpromptPhUL{year}" for part in keys):
            sumw2_pt_eta_hist = photon_pt_eta_sumw2.view(flow=True)[keys]
            sumw_pt_eta_hist = photon_pt_eta.view(flow=True)[keys]

            #This is the master equation to calculate the final NP variance
            np_var = ((sumw2_pt_eta_hist*pow(ff_val_padded,2)))+(pow(sumw_pt_eta_hist*ff_err_padded,2))
            np_var = np.nan_to_num(np_var,0)

            #this is how we modify the values of sumw2 and sumw. After this point the photon_pt_eta and and photon_pt_eta_sumw2 hists values will be modified by new value.
            sumw2_pt_eta_hist[:] = np_var
            sumw_pt_eta_hist[:] = ff_val_padded*sumw_pt_eta_hist


#This histogram takes the output histogram dictionary called "outHist" in the dataDrivenEstimation script, where we will have a key called "photon_pt2_sumw2". This histogram will also have the non-prompt estimation done, so it will have a key called "nonPromptPhUL<year>", and we need to modify the values associated with that key. The photon_pt2_sumw2 histogram has same binning as FR/kMC
#Also note that at this point, the photon_pt_eta_sumw2 histogram will already have the values modified as described above
#CAUTION: Revisit this in this future cause we will just have one photon_pt histogram and the name will need to be changed accordingly
def modify_photon_pt_variance(outputHist,year):
    for k in list(outputHist['photon_pt2_sumw2'].view(flow=True).keys()):
        if any(str(part) == f"nonpromptPhUL{year}" for part in k):
            #First, load the values of photon_pt2_sumw2 hist
            photon_pt2_sumw2 = outputHist['photon_pt2_sumw2'].view(flow=True)[k]
            #Second, load the modified values from photon_pt_eta_sumw2 hist
            np_var = outputHist['photon_pt_eta_sumw2'].view(flow=True)[k]

            #WE first have to add over eta bins to get uncertainties associated with the pt bins
            new_sumw2_summed_over_eta = np.sum(np_var,1)  #At this point this array has a shape of 1 X (N_photon_pt_bins+1)

            #We need to get new_sumw2_summed_over_eta array to be the same shape as photon_pt2_sumw2 array
            desired_sumw2_summed_over_eta = get_desired_array_shape(new_sumw2_summed_over_eta, np.zeros((new_sumw2_summed_over_eta.shape[0], 3)))

            #We are now ready to modify the values of the photon_pt2_sumw2 histogram
            photon_pt2_sumw2[:] = desired_sumw2_summed_over_eta
