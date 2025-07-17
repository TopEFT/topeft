'''
        This script is made specifically for run3 flip scale factor calculation based on l0eta distribution.
        Default eta bins are [-3, -1.479, 0, 1.479, 3], for +/- endcap/barrel regions. Hence there will be total of four sfs.

        Run the following command:
        python flip_sf_run3.py {/path/to/np.pkl.gz} -y {year}

'''

import argparse
import pickle
import gzip
import numpy as np
import hist
import boost_histogram as bh
import awkward as ak
import json

def load_pkl_file(pkl_file):
    with gzip.open(pkl_file, "rb") as f:
        return pickle.load(f)

def get_yields_in_eta_bins(hin_dict, proc_list, eta_bins, hist_name, channel_name):
    h = hin_dict[hist_name]
    yields = {}

    for proc in proc_list:
        yields[proc] = []

        try:
        # Slice to process and channel
            h_sel = h[{"process": proc, "channel": channel_name}]
        # If "l0eta" is not the only axis, integrate over the others
        #for ax in h_sel.axes:
        #    if ax.name != "l0eta":
        #        h_sel = h_sel.integrate(ax.name)
            eta_axis = h_sel.axes[hist_name]
            view = h_sel.view(flow=False)
            edges = eta_axis.edges
            view_array = list(view.values())[0]  # Extracts the array
            view_flatten = view_array.flatten().tolist()

        except Exception as e:
            print(f"  Error slicing/integrating for proc {proc}: {e}")
            yields[proc] = [(None, None)] * (len(eta_bins) - 1)
            continue

        for i in range(len(eta_bins) - 1):
            low, high = eta_bins[i], eta_bins[i + 1]
            val = 0.0
            err = 0.0

            bin_indices = [
                j for j, (lo, hi) in enumerate(zip(eta_axis.edges[:-1], eta_axis.edges[1:]))
                if hi > low and lo < high
            ]
            val = sum(view_flatten[j] for j in bin_indices)
            yields[proc].append((val, 0.0))
            
    return yields

def make_flipsf_json(eta_bins, scale_factors, year):
    if len(eta_bins) != len(scale_factors) + 1:
        raise ValueError("Number of scale factors must be one less than number of bin edges.")
    
    key_name = f"FlipSF_eta"
    sf_json = {
        key_name: {
            f"[{eta_bins[i]},{eta_bins[i+1]}]": scale_factors[i]
            for i in range(len(scale_factors))
        }
    }
    with open(f"flip_sf_{year}.json", "w") as f:
        json.dump(sf_json, f, indent=2)
    print(f"Scaling factors saved to flip_sf_{year}.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file", help="Path to the input pkl.gz file")
    parser.add_argument("--hist-name", default="l0eta", help="Histogram name")
    parser.add_argument("--channel", default="2lss_ee_CRflip_3j", help="Channel name")
    parser.add_argument("-y", "--year", default="2022", help = "The year of the sample")
    args = parser.parse_args()

    eta_bins = [-3, -1.479, 0, 1.479, 3]
    hin_dict = load_pkl_file(args.pkl_file)

    h = hin_dict[args.hist_name]
    proc_list = list(h.axes["process"])

    yields = get_yields_in_eta_bins(hin_dict, proc_list, eta_bins, hist_name=args.hist_name, channel_name=args.channel)

    flips = []
    data = []
    other = []

    for proc, vals in yields.items():
        if "flips" in proc:
            flips = [val for val, _ in vals]
        elif "data" in proc:
            data = [val for val, _ in vals]
        else:
            if not other:
                other = [val for val, _ in vals]
            else:
                other = [x + val for x, (val, _) in zip(other, vals)]

    # Compute (data - other) / flips
    scale_factors = []
    for d, o, f in zip(data, other, flips):
        if f != 0:
            sf = (d - o) / f
        else:
            sf = float("nan")  # or 0, or raise an error
        scale_factors.append(sf)
    make_flipsf_json(eta_bins, scale_factors, year=args.year)

    # Output
    print("flips =", flips)
    print("data  =", data)
    print("other =", other)
    print("SFs   =", scale_factors)



if __name__ == "__main__":
    main()

