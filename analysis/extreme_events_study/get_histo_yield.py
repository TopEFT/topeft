import topcoffea.modules.utils as utils
from topcoffea.modules.YieldTools import YieldTools
from make_cr_and_sr_plots import get_lumi_for_sample


def get_bins_sum(histo):
    histo_summed = histo.sum("channel", overflow="all")
    yt = YieldTools()
    dict_of_histo = {"hist": histo_summed}
    all_samples = yt.get_cat_lables(dict_of_histo,"sample",h_name=yt.get_hist_list(dict_of_histo)[0])
    sample_lumi_dict = {}
    signal_sample = []
    for sample_name in all_samples:
        if "nonprompt" not in sample_name:
            signal_sample.append(sample_name)
        sample_lumi_dict[sample_name] = get_lumi_for_sample(sample_name)
    histo_summed.scale(sample_lumi_dict,axis="sample")
    histo_summed = histo_summed.integrate("sample",signal_sample,  overflow="all")
    histo_summed = histo_summed.sum("systematic", overflow="all")
    values_dict = histo_summed.values(overflow="all")
    for value in values_dict.values():
        values_ls = value.tolist()
        # The slice range is determined by the top values of the event quantity.
        # e.g. For lepton multiplicity, we want bins with nleps>4.
        bins_sum = sum(values_ls[11:])
        return bins_sum


# Run the function to print yields
def print_yields():
    # Get the histogram of the selected events
    hin_dict = utils.get_hist_from_pkl("path/to/topeft/output.pkl.gz",allow_empty=False)
    histo = hin_dict["njets"]

    # Specify the values of the WCs
    wc_ranges_differential = {
        'cQQ1' : 4.0,
        'cQei' : 4.0,
        'cQl3i': 5.5,
        'cQlMi': 4.0,
        'cQq11': 0.7,
        'cQq13': 0.35,
        'cQq81': 1.5,
        'cQq83': 0.6,
        'cQt1' : 4.0,
        'cQt8' : 8.0,
        'cbW'  : 3.0,
        'cpQ3' : 4.0,
        'cpQM' : 17.0,
        'cpt'  : 15.0,
        'cptb' : 9.0,
        'ctG'  : 0.8,
        'ctW'  : 1.5,
        'ctZ'  : 2.0,
        'ctei' : 4.0,
        'ctlSi': 5.0,
        'ctlTi': 0.9,
        'ctli' : 4.0,
        'ctp'  : 35.0,
        'ctq1' : 0.6,
        'ctq8' : 1.4,
        'ctt1' : 2.1,
    }

    # Initialize the output to print
    sum_dict = {}

    # Calculate the SM prediction
    sum_dict["SM"] = get_bins_sum(histo)

    # Calculate the yields modified by each WC
    for keyword, value in wc_ranges_differential.items():
        temp_dict = {}
        temp_dict[keyword] = value
        histo.set_wilson_coefficients(**temp_dict)
        bins_sum = get_bins_sum(histo)
        sum_dict[keyword] = bins_sum

    # Calculate the yields modified by all WCs
    histo.set_wilson_coefficients(**wc_ranges_differential)
    sum_dict["allWCs"] = get_bins_sum(histo)

    # Print the yields
    for key, value in sum_dict.items():
        print(key, value)

