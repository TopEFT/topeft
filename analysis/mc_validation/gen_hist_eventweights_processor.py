'''
This script produces histograms of the log of the event weights
It assumes the MG weights are saved in the nanoGEN file.

Example:
python run_gen_hist_eventweights_processor.py ../../input_samples/sample_jsons/signal_samples/private_UL/2022_tllq_NewStPt4_nanoGEN.json -o 2022_tllq_NewStPt4 -x futures -r file:///cms/cephfs/data/ -p gen_hist_eventweights_processor.py
'''
#!/usr/bin/env python
import awkward as ak
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')
import hist
from hist import Hist
import topcoffea.modules.eft_helper as efth
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

from coffea import processor
from coffea.analysis_tools import PackedSelection

# Get the lumi for the given year
def get_lumi(year):
    lumi_dict = {
        "2016APV": 19.52,
        "2016": 16.81,
        "2017": 41.48,
        "2018": 59.83
    }
    if year not in lumi_dict.keys():
        raise Exception(f"(ERROR: Unknown year \"{year}\".")
    else:
        return(lumi_dict[year])

# Clean the objects
def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

# Create list of wc values in the correct order
def order_wc_values(wcs, ref_pts):
    '''Returns list of wc values in the same order as the list of wc names based on a dictionary
    '''
    wc_names = wcs
    ref_pts = ref_pts

    wc_values = []
    for i in wc_names:
        wc_values.append(ref_pts[i])

    return wc_values

# Calculate event weights from wc values and eft fit coefficients
def calc_event_weights(eft_coeffs, wc_vals):
    '''Returns an array that contains the event weight for each event.
    eft_coeffs: Array of eft fit coefficients for each event
    wc_vals: wilson coefficient values desired for the event weight calculation, listed in the same order as the wc_lst
             such that the multiplication with eft_coeffs is correct
             The correct ordering can be achieved with the order_wc_values function
    '''
    event_weight = np.empty_like(eft_coeffs)

    wcs = np.hstack((np.ones(1),wc_vals))
    wc_cross_terms = []
    index = 0
    for j in range(len(wcs)):
        for k in range (j+1):
            term = wcs[j]*wcs[k]
            wc_cross_terms.append(term)
    event_weight = np.sum(np.multiply(wc_cross_terms, eft_coeffs), axis=1)

    return event_weight

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        print("self._samples", self._samples)
        print("self._wc_names_lst", self._wc_names_lst)

        # Create the histograms with new scikit hist
        self._low = -12
        self._high = 3
        self._histo_dict = {
            "weights_SM_log"   : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_SM_log",  label='log(event weights at the SM)' ), storage="weight"),
            "weights_SMneg_log"   : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_SMneg_log",  label='log(event weights at the SMneg)' ), storage="weight"),
            "weights_SMabs_log"   : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_SMabs_log",  label='log(event weights at the SMabs)' ), storage="weight"),
            "weights_SMefth_log"   : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_SMefth_log",  label='log(event weights at the SMefth)' ), storage="weight"),
            "weights_SMcoeff_log"   : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_SMcoeff_log",  label='log(event weights at the SM)' ), storage="weight"),
            "weights_pt1_log"  : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_pt1_log", label='log(event weights at the pt1)'), storage="weight"),
            "weights_pt2_log"  : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_pt2_log", label='log(event weights at the pt2)'), storage="weight"),
            "weights_pt3_log"  : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_ptself._high_log", label='log(event weights at the ptself._high)'), storage="weight"),
            "weights_pt4_log"  : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_pt4_log", label='log(event weights at the pt4)'), storage="weight"),
            "weights_pt5_log"  : Hist(hist.axis.Regular(bins=120, start=self._low, stop=self._high, name="weights_pt5_log", label='log(event weights at the pt5)'), storage="weight"),
            #"events_100"       : {},
            #"events"       : {},
        }

        # Set the list of hists to to fill
        if hist_lst is None:
            self._hist_lst = list(self._histo_dict.keys())
        else:
            for h in hist_lst:
                if h not in self._histo_dict.keys():
                    raise Exception(f"Error: Cannot specify hist \"{h}\", it is not defined in self._histo_dict")
            self._hist_lst = hist_lst

        print("hist_lst: ", self._hist_lst)

    @property
    def columns(self):
        return self._columns

    def process(self, events):

        # Dataset parameters
        dataset = events.metadata['dataset']
        hist_axis_name = self._samples[dataset]["histAxisName"]

        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights']

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None

        # else:
        SM_pt = {wc: 0.0 for wc in self._wc_names_lst}
        #wc_lst_SM = order_wc_values(self._wc_names_lst, SM_pt)
        wc_lst_SM = np.zeros(len(self._wc_names_lst))
        event_weights_SM = calc_event_weights(eft_coeffs,wc_lst_SM)
        event_weights_SMcoeff = eft_coeffs[:,0]
        #wc_lst_SM = np.zeros(len(self._wc_names_lst))
        #event_weights_SM = efth.calc_eft_weights(eft_coeffs,wc_lst_SM)
        event_weights_SMefth = efth.calc_eft_weights(eft_coeffs,np.zeros(len(self._wc_names_lst)))

        eft_weight_names = [ x for x in events['LHEWeight'].fields if x.startswith('EFTrwgt') ][:10]
        if not eft_weight_names:
            wc_lst_pt1 = np.random.uniform(-10, 10, len(self._wc_names_lst))
            #wc_lst_pt1 = order_wc_values(self._wc_names_lst, rwgt1)
            event_weights_pt1 = calc_event_weights(eft_coeffs, wc_lst_pt1)
        else:
            wc_vals = eft_weight_names[0].split('_')[1:]
            wc_vals = np.array([float(wc) for wc in wc_vals[1::2]])
            event_weights_pt1 = efth.calc_eft_weights(eft_coeffs,wc_vals)

        #wc_lst_pt2 = order_wc_values(self._wc_names_lst, rwgt2)
        if not eft_weight_names:
            wc_lst_pt2 = np.random.uniform(-10, 10, len(self._wc_names_lst))
            event_weights_pt2 = calc_event_weights(eft_coeffs, wc_lst_pt2)
        else:
            wc_vals = eft_weight_names[1].split('_')[1:]
            wc_vals = np.array([float(wc) for wc in wc_vals[1::2]])
            event_weights_pt2 = efth.calc_eft_weights(eft_coeffs,wc_vals)

        #wc_lst_pt3 = order_wc_values(self._wc_names_lst, rwgt3)
        if not eft_weight_names:
            wc_lst_pt3 = np.random.uniform(-10, 10, len(self._wc_names_lst))
            event_weights_pt3 = calc_event_weights(eft_coeffs, wc_lst_pt3)
        else:
            wc_vals = eft_weight_names[2].split('_')[1:]
            wc_vals = np.array([float(wc) for wc in wc_vals[1::2]])
            event_weights_pt3 = efth.calc_eft_weights(eft_coeffs,wc_vals)

        #wc_lst_pt4 = order_wc_values(self._wc_names_lst, rwgt4)
        if not eft_weight_names:
            wc_lst_pt4 = np.random.uniform(-10, 10, len(self._wc_names_lst))
            event_weights_pt4 = calc_event_weights(eft_coeffs, wc_lst_pt4)
        else:
            wc_vals = eft_weight_names[3].split('_')[1:]
            wc_vals = np.array([float(wc) for wc in wc_vals[1::2]])
            event_weights_pt4 = efth.calc_eft_weights(eft_coeffs,wc_vals)

        #wc_lst_pt5 = order_wc_values(self._wc_names_lst, rwgt5)
        if not eft_weight_names:
            wc_lst_pt5 = np.random.uniform(-10, 10, len(self._wc_names_lst))
            event_weights_pt5 = calc_event_weights(eft_coeffs, wc_lst_pt5)
        else:
            wc_vals = eft_weight_names[4].split('_')[1:]
            wc_vals = np.array([float(wc) for wc in wc_vals[1::2]])
            event_weights_pt5 = efth.calc_eft_weights(eft_coeffs,wc_vals)

        counts = np.ones_like(event_weights_SM)

        ######## Fill histos ########
        hout = self._histo_dict
        variables_to_fill = {
            "weights_SM_log" : np.nan_to_num(np.log10(event_weights_SM ), nan=self._low-1),
            "weights_SMneg_log" : np.nan_to_num(np.log10(event_weights_SM[event_weights_SM < 0] ), nan=self._low-1),
            "weights_SMabs_log" : np.nan_to_num(np.log10(np.abs(event_weights_SM) ), nan=self._low-1),
            "weights_SMefth_log" : np.nan_to_num(np.log10(event_weights_SMefth ), nan=self._low-1),
            "weights_SMcoeff_log" : np.nan_to_num(np.log10(event_weights_SMcoeff ), nan=self._low-1),
            "weights_pt1_log": np.nan_to_num(np.log10(event_weights_pt1), nan=self._low-1),
            "weights_pt2_log": np.nan_to_num(np.log10(event_weights_pt2), nan=self._low-1),
            "weights_pt3_log": np.nan_to_num(np.log10(event_weights_pt3), nan=self._low-1),
            "weights_pt4_log": np.nan_to_num(np.log10(event_weights_pt4), nan=self._low-1),
            "weights_pt5_log": np.nan_to_num(np.log10(event_weights_pt5), nan=self._low-1),
        }

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            if 'neg' in var_name: hout[var_name].fill(var_values, weight=counts[event_weights_SM < 0])
            else: hout[var_name].fill(var_values, weight=counts)
        #if ak.any(event_weights_SM[event_weights_SM > 100]):
        #    hout['events_100'][events.metadata['filename']] = event_weights_SM[event_weights_SM > 100]
        #hout['events'][events.metadata['filename']] = event_weights_SM.tolist()

        return hout

    def postprocess(self, accumulator):
        return accumulator

