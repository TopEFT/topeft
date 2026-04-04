import numpy as np
import argparse

import topcoffea.modules.utils as utils
from topcoffea.modules.utils import canonicalize_process_name
from topeft.modules.yield_tools import YieldTools
yt = YieldTools()


# The names of the 6 renorm and fact variations
RENORMFACT_VAR_LST = [
    "renormfactUp",
    "renormfactDown",
    "renormUp",
    "renormDown",
    "factUp",
    "factDown"
]

# Samples that do not include renorm and fact variations
NO_RENORMFACT_LST = [
    canonicalize_process_name(proc_name)
    for proc_name in [
        "dataUL16",
        "dataUL16APV",
        "dataUL17",
        "dataUL18",
        "flipsUL16",
        "flipsUL16APV",
        "flipsUL17",
        "flipsUL18",
        "nonpromptUL16",
        "nonpromptUL16APV",
        "nonpromptUL17",
        "nonpromptUL18",
    ]
]


# Get the most extreme renorm fact variations
def get_renormfact_envelope(dict_of_hists, *, verbose=True):
    out_hist_dict = {}
    if verbose:
        print("\nAll vars:",dict_of_hists.keys())
    for var_name, histo in dict_of_hists.items():
        out_hist_dict[var_name] = apply_renormfact_envelope_to_histogram(
            histo, verbose=verbose, hist_name=var_name
        )

    return out_hist_dict


def apply_renormfact_envelope_to_histogram(histo, *, verbose=True, hist_name=None):
    if verbose and hist_name is not None:
        print("\tVar name:", hist_name)

    process_lst = yt.get_cat_lables(histo, "process")
    cat_lst = yt.get_cat_lables(histo, "channel")
    if verbose:
        print("\nAll processes:", process_lst)
        print("\nAll cats:", cat_lst)

    hist_view = histo.view(as_dict=True, flow=True)

    for process_name in process_lst:
        if canonicalize_process_name(process_name) in NO_RENORMFACT_LST:
            continue
        for cat_name in cat_lst:
            if verbose:
                print("\t\t", process_name, cat_name)

            key_tup_nom = (process_name, cat_name, "nominal")
            if key_tup_nom not in hist_view:
                continue

            variation_keys = [
                (process_name, cat_name, rf_variation)
                for rf_variation in RENORMFACT_VAR_LST
            ]
            if any(var_key not in hist_view for var_key in variation_keys):
                continue

            dense_arr_nom = _get_selector_view(hist_view[key_tup_nom])
            variation_payload = [np.asarray(hist_view[var_key]) for var_key in variation_keys]
            variation_selector = np.stack(
                [_get_selector_view(payload) for payload in variation_payload], axis=0
            )
            diff_wrt_nom = variation_selector - dense_arr_nom[np.newaxis, ...]
            max_var_idx = np.argmax(diff_wrt_nom, axis=0)
            min_var_idx = np.argmin(diff_wrt_nom, axis=0)

            key_tup_rf_env_up = (process_name, cat_name, "renormfactUp")
            key_tup_rf_env_do = (process_name, cat_name, "renormfactDown")
            hist_view[key_tup_rf_env_up] = _select_extreme_payload(
                variation_payload, max_var_idx
            )
            hist_view[key_tup_rf_env_do] = _select_extreme_payload(
                variation_payload, min_var_idx
            )

            sumw2_payload = _load_sumw2_payload(histo, variation_keys)
            if sumw2_payload is None:
                histo._sumw2[key_tup_rf_env_up] = None
                histo._sumw2[key_tup_rf_env_do] = None
            else:
                histo._sumw2[key_tup_rf_env_up] = _select_extreme_payload(
                    sumw2_payload, max_var_idx
                )
                histo._sumw2[key_tup_rf_env_do] = _select_extreme_payload(
                    sumw2_payload, min_var_idx
                )

            del dense_arr_nom
            del variation_payload
            del variation_selector
            del diff_wrt_nom
            del max_var_idx
            del min_var_idx
            del sumw2_payload

    return histo.remove("systematic", ["factUp", "factDown", "renormUp", "renormDown"])


def _get_selector_view(arr):
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        return arr[..., 0]
    return arr


def _load_sumw2_payload(histo, variation_keys):
    payload = []
    for key_tup in variation_keys:
        arr = histo._sumw2[key_tup]
        if arr is None:
            return None
        payload.append(np.asarray(arr))
    return payload


def _select_extreme_payload(variation_payload, selected_indices):
    template = variation_payload[0]
    selected = np.empty_like(template)
    for variation_idx, payload in enumerate(variation_payload):
        mask = selected_indices == variation_idx
        if not np.any(mask):
            continue
        if payload.ndim == 1:
            selected[mask] = payload[mask]
        else:
            selected[mask, ...] = payload[mask, ...]
    return selected


# Example standalone usage of get_renormfact_envelope()
# Generally this function will be called from the run script
def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument("-n", "--output-name", default="histos_dict", help = "A name for the output file")
    args = parser.parse_args()

    # Get the envelope and write to an out pkl
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    hout_dict = get_renormfact_envelope(hin_dict)
    utils.dump_to_pkl(args.output_name,hout_dict)

if __name__ == "__main__":
    main()
