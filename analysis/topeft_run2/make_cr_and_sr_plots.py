import numpy as np
import os
import copy
import datetime
import argparse
import re
from collections import OrderedDict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mplhep as hep
import hist
from matplotlib.transforms import Bbox
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d

from topcoffea.scripts.make_html import make_html
import topcoffea.modules.utils as utils
from topeft.modules.yield_tools import YieldTools

from topcoffea.modules.paths import topcoffea_path
from topeft.modules.paths import topeft_path
import topeft.modules.get_rate_systs as grs
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
import yaml

with open(topeft_path("params/cr_sr_plots_metadata.yml")) as f:
    _META = yaml.safe_load(f)

DATA_ERR_OPS = _META["DATA_ERR_OPS"]
MC_ERROR_OPS = _META["MC_ERROR_OPS"]
if isinstance(MC_ERROR_OPS.get("edgecolor"), list):
    MC_ERROR_OPS["edgecolor"] = tuple(MC_ERROR_OPS["edgecolor"])
CR_CHAN_DICT = _META["CR_CHAN_DICT"]
SR_CHAN_DICT = _META["SR_CHAN_DICT"]
CR_GROUP_INFO = _META.get("CR_GRP_MAP", {})
SR_GROUP_INFO = _META.get("SR_GRP_MAP", {})
CR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in CR_GROUP_INFO.items()}
SR_GRP_PATTERNS = {k: v.get("patterns", []) for k, v in SR_GROUP_INFO.items()}
CR_GRP_MAP = {k: [] for k in CR_GRP_PATTERNS.keys()}
SR_GRP_MAP = {k: [] for k in SR_GRP_PATTERNS.keys()}
SR_SIGNAL_GROUP_KEYS = {"ttH", "ttlnu", "ttll", "tXq", "tttt"}
SIGNAL_WC_MATCHES = ("ttH", "tllq", "ttlnu", "ttll", "tHq", "tttt")
CR_KNOWN_CHANNELS = {chan for chans in CR_CHAN_DICT.values() for chan in chans}
SR_KNOWN_CHANNELS = {chan for chans in SR_CHAN_DICT.values() for chan in chans}
FILL_COLORS = {k: v.get("color") for k, v in {**CR_GROUP_INFO, **SR_GROUP_INFO}.items()}
WCPT_EXAMPLE = _META["WCPT_EXAMPLE"]
LUMI_COM_PAIRS = _META["LUMI_COM_PAIRS"]
PROC_WITHOUT_PDF_RATE_SYST = _META["PROC_WITHOUT_PDF_RATE_SYST"]
REGION_PLOTTING = _META.get("REGION_PLOTTING", {})

# This script takes an input pkl file that should have both data and background MC included.
# Use the -y option to specify a year, if no year is specified, all years will be included.
# There are various other options available from the command line.
# For example, to make unit normalized plots for 2018, with the timestamp appended to the directory name, you would run:
#     python make_cr_and_sr_plots.py -f histos/your.pkl.gz -o ~/www/somewhere/in/your/web/dir -n some_dir_name -y 2018 -t -u

yt = YieldTools()

######### Utility functions #########

# Takes a dictionary where the keys are catetory names and keys are lists of bin names in the category, and a string indicating what type of info (njets, or lepflav) to remove
# Returns a dictionary of the same structure, except with njet or lepflav info stripped off of the bin names
# E.g. if a value was ["cat_a_1j","cat_b_1j","cat_b_2j"] and we passed "njets", we should return ["cat_a","cat_b"]
def get_dict_with_stripped_bin_names(in_chan_dict,type_of_info_to_strip):
    out_chan_dict = {}
    for cat,bin_names in in_chan_dict.items():
        out_chan_dict[cat] = []
        for bin_name in bin_names:
            if type_of_info_to_strip == "njets":
                bin_name_no_njet = yt.get_str_without_njet(bin_name)
            elif type_of_info_to_strip == "lepflav":
                bin_name_no_njet = yt.get_str_without_lepflav(bin_name)
            else:
                raise Exception(f"Error: Unknown type of string to remove \"{type_of_info_to_strip}\".")
            if bin_name_no_njet not in out_chan_dict[cat]:
                out_chan_dict[cat].append(bin_name_no_njet)
    return (out_chan_dict)


def _apply_channel_transforms(channel_name, transformations):
    transformed = channel_name
    for transform in transformations:
        if transform == "njets":
            transformed = yt.get_str_without_njet(transformed)
        elif transform == "lepflav":
            transformed = yt.get_str_without_lepflav(transformed)
        else:
            raise ValueError(f"Unsupported channel transformation '{transform}'")
    return transformed


def _resolve_requested_variables(dict_of_hists, variables, context):
    """Return the ordered list of variables to process for a plotting function."""

    all_variables = list(dict_of_hists.keys())
    if not variables:
        return all_variables

    resolved = []
    missing = []
    for var_name in variables:
        if var_name in dict_of_hists:
            if var_name not in resolved:
                resolved.append(var_name)
        else:
            missing.append(var_name)

    for missing_name in missing:
        print(
            f"Warning: Requested variable '{missing_name}' not found in {context}; skipping."
        )

    return resolved


def validate_channel_group(histos, expected_labels, transformations, region, subgroup, variable):
    if not isinstance(histos, (list, tuple)):
        histos = [histos]

    available_channels = set()
    for histo in histos:
        if not isinstance(histo, (HistEFT, SparseHist)):
            continue
        if "channel" not in yt.get_axis_list(histo):
            continue
        available_channels.update(list(histo.axes["channel"]))

    if not available_channels:
        return

    expected_set = set(expected_labels)
    expected_transformed = {
        _apply_channel_transforms(label, transformations) for label in expected_set
    }

    region_known_channels = {
        "CR": CR_KNOWN_CHANNELS,
        "SR": SR_KNOWN_CHANNELS,
    }.get(region)
    transformed_known_channels = None
    if region_known_channels is not None:
        transformed_known_channels = {
            _apply_channel_transforms(label, transformations)
            for label in region_known_channels
        }

    stray_channels = set()

    for channel in available_channels:
        transformed = _apply_channel_transforms(channel, transformations)
        if transformed in expected_transformed:
            continue
        if region_known_channels is not None and channel in region_known_channels:
            continue
        if transformed_known_channels is not None and transformed in transformed_known_channels:
            continue
        stray_channels.add(channel)

    if stray_channels:
        var_str = f" for variable '{variable}'" if variable is not None else ""
        region_str = f"{region} " if region else ""
        raise ValueError(
            f"Found channel bins {sorted(stray_channels)} in {region_str}subgroup '{subgroup}'{var_str} that are not defined in the YAML configuration."
        )

def populate_group_map(samples, pattern_map):
    out = {k: [] for k in pattern_map}
    for proc_name in samples:
        matched = False
        for grp, patterns in pattern_map.items():
            for pat in patterns:
                if pat in proc_name:
                    out[grp].append(proc_name)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            print(f"Warning: Process name \"{proc_name}\" does not match any known group patterns. It will not be included in the grouping.")
            # If you want to raise an error instead, uncomment the next line
            # raise Exception(f"Error: Process name \"{proc_name}\" is not known.")
    return out


def _sample_in_signal_group(sample_name, sample_group_map, group_type):
    if group_type == "CR":
        return sample_name in sample_group_map.get("Signal", [])

    if group_type == "SR":
        for grp_key in SR_SIGNAL_GROUP_KEYS:
            if sample_name in sample_group_map.get(grp_key, []):
                return True

    return False


def _normalize_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _evaluate_channel_condition(condition, region_ctx):
    if condition == "not_split_by_lepflav":
        return not yt.is_split_by_lepflav(region_ctx.dict_of_hists)
    raise ValueError(
        f"Unsupported channel transformation condition '{condition}'"
    )


def _resolve_channel_transformations(region_ctx, var_name):
    rules = region_ctx.channel_rules
    transformations = []
    transformations.extend(rules.get("default", []))
    transformations.extend(rules.get("variables", {}).get(var_name, []))
    for cond_entry in rules.get("conditional", []):
        condition = cond_entry.get("when")
        if condition is None:
            continue
        if _evaluate_channel_condition(condition, region_ctx):
            transformations.extend(cond_entry.get("apply", []))
    ordered = []
    seen = set()
    for transform in transformations:
        if transform not in seen:
            ordered.append(transform)
            seen.add(transform)
    return ordered


def _apply_channel_dict_transformations(channel_dict, transformations):
    transformed_dict = copy.deepcopy(channel_dict)
    for transform in transformations:
        if transform == "njets":
            transformed_dict = get_dict_with_stripped_bin_names(
                transformed_dict, "njets"
            )
        elif transform == "lepflav":
            transformed_dict = get_dict_with_stripped_bin_names(
                transformed_dict, "lepflav"
            )
        else:
            raise ValueError(
                f"Unsupported channel transformation '{transform}'"
            )
    return transformed_dict


def _match_category(hist_cat, categories_cfg):
    if not categories_cfg:
        return True
    prefixes = _normalize_sequence(categories_cfg.get("prefixes"))
    if prefixes and any(hist_cat.startswith(pref) for pref in prefixes):
        return True
    equals = _normalize_sequence(categories_cfg.get("equals"))
    if equals and hist_cat in equals:
        return True
    contains = _normalize_sequence(categories_cfg.get("contains"))
    if contains and any(token in hist_cat for token in contains):
        return True
    return False


def _should_skip_category(rules, hist_cat, var_name):
    for rule in rules:
        if not _match_category(hist_cat, rule.get("categories")):
            continue
        includes = _normalize_sequence(rule.get("variable_includes"))
        if includes and not any(token in var_name for token in includes):
            continue
        excludes = _normalize_sequence(rule.get("variable_excludes"))
        if excludes and any(token in var_name for token in excludes):
            continue
        return True
    return False


def _collect_samples_to_remove(rules, hist_cat, region_ctx):
    samples = []
    for rule in rules:
        if not _match_category(hist_cat, rule.get("categories")):
            continue
        rule_groups = _normalize_sequence(rule.get("groups"))
        for group in rule_groups:
            samples.extend(region_ctx.group_map.get(group, []))
        samples.extend(_normalize_sequence(rule.get("samples")))
    ordered = []
    seen = set()
    for sample in samples:
        if sample not in seen:
            ordered.append(sample)
            seen.add(sample)
    return ordered


def _normalize_channel_rules(raw_rules):
    if raw_rules is None:
        return {"default": [], "variables": {}, "conditional": []}
    normalized = {
        "default": _normalize_sequence(raw_rules.get("default", [])),
        "variables": {
            key: _normalize_sequence(value)
            for key, value in raw_rules.get("variables", {}).items()
        },
        "conditional": [],
    }
    conditional_entries = []
    for entry in raw_rules.get("conditional", []):
        if not entry:
            continue
        when_key = entry.get("when")
        if when_key is None:
            continue
        conditional_entries.append(
            {"when": when_key, "apply": _normalize_sequence(entry.get("apply", []))}
        )
    normalized["conditional"] = conditional_entries
    return normalized


def _find_reference_hist_name(dict_of_hists):
    for hist_name in dict_of_hists:
        if not hist_name.endswith("_sumw2"):
            return hist_name
    raise ValueError("No histogram without '_sumw2' suffix was found.")


class RegionContext(object):
    def __init__(
        self,
        name,
        dict_of_hists,
        year,
        channel_map,
        group_patterns,
        group_map,
        all_samples,
        mc_samples,
        data_samples,
        samples_to_remove,
        sumw2_hists,
        signal_samples,
        unblind_default,
        lumi_pair,
        skip_variables=None,
        analysis_bins=None,
        channel_rules=None,
        sample_removal_rules=None,
        category_skip_rules=None,
        skip_sparse_2d=False,
        channel_mode="per-channel",
        variable_label="Variable",
        debug_channel_lists=False,
        sumw2_remove_signal=False,
        sumw2_remove_signal_when_blinded=False,
        use_mc_as_data_when_blinded=False,
    ):
        self.name = name
        self.dict_of_hists = dict_of_hists
        self.year = year
        self.channel_map = channel_map
        self.group_patterns = group_patterns
        self.group_map = group_map
        self.all_samples = all_samples
        self.mc_samples = mc_samples
        self.data_samples = data_samples
        self.samples_to_remove = samples_to_remove
        self.sumw2_hists = sumw2_hists
        self.signal_samples = signal_samples
        self.unblind_default = unblind_default
        self.lumi_pair = lumi_pair
        self.skip_variables = set() if skip_variables is None else set(skip_variables)
        self.analysis_bins = (
            {} if analysis_bins is None else copy.deepcopy(analysis_bins)
        )
        default_channel_rules = {"default": [], "variables": {}, "conditional": []}
        self.channel_rules = copy.deepcopy(
            channel_rules if channel_rules is not None else default_channel_rules
        )
        self.sample_removal_rules = (
            copy.deepcopy(sample_removal_rules)
            if sample_removal_rules is not None
            else []
        )
        self.category_skip_rules = (
            copy.deepcopy(category_skip_rules)
            if category_skip_rules is not None
            else []
        )
        self.skip_sparse_2d = bool(skip_sparse_2d)
        self.channel_mode = channel_mode
        self.variable_label = variable_label
        self.debug_channel_lists = bool(debug_channel_lists)
        self.sumw2_remove_signal = bool(sumw2_remove_signal)
        self.sumw2_remove_signal_when_blinded = bool(
            sumw2_remove_signal_when_blinded
        )
        self.use_mc_as_data_when_blinded = bool(use_mc_as_data_when_blinded)


def build_region_context(region,dict_of_hists,year,unblind=None):
    region_upper = region.upper()
    if region_upper not in ["CR","SR"]:
        raise ValueError(f"Unsupported region '{region}'.")

    mc_wl = []
    mc_bl = ["data"]
    data_wl = ["data"]
    data_bl = []
    if year is None:
        pass
    elif year == "2017":
        mc_wl.append("UL17")
        data_wl.append("UL17")
    elif year == "2018":
        mc_wl.append("UL18")
        data_wl.append("UL18")
    elif year == "2016":
        mc_wl.append("UL16")
        mc_bl.append("UL16APV")
        data_wl.append("UL16")
        data_bl.append("UL16APV")
    elif year == "2016APV":
        mc_wl.append("UL16APV")
        data_wl.append("UL16APV")
    elif year == "2022":
        mc_wl.append("2022")
        data_wl.append("2022")
    elif year == "2022EE":
        mc_wl.append("2022EE")
        data_wl.append("2022EE")
    elif year == "2023":
        mc_wl.append("2023")
        data_wl.append("2023")
    elif year == "2023BPix":
        mc_wl.append("2023BPix")
        data_wl.append("2023BPix")
    else:
        raise Exception(f"Error: Unknown year \"{year}\".")

    try:
        all_samples = yt.get_cat_lables(dict_of_hists, "process")
    except Exception:
        ref_hist = _find_reference_hist_name(dict_of_hists)
        all_samples = yt.get_cat_lables(dict_of_hists, "process", h_name=ref_hist)

    mc_samples = utils.filter_lst_of_strs(
        all_samples, substr_whitelist=mc_wl, substr_blacklist=mc_bl
    )
    data_samples = utils.filter_lst_of_strs(
        all_samples, substr_whitelist=data_wl, substr_blacklist=data_bl
    )
    samples_to_remove = {
        "mc": [sample for sample in all_samples if sample not in mc_samples],
        "data": [sample for sample in all_samples if sample not in data_samples],
    }

    sumw2_hists = {
        hist_name.replace("_sumw2", ""): hist_obj
        for hist_name, hist_obj in dict_of_hists.items()
        if hist_name.endswith("_sumw2") and hist_name.count("sumw2") == 1
    }

    lumi_pair = LUMI_COM_PAIRS.get(year)

    if unblind is None:
        resolved_unblind = region_upper == "CR"
    else:
        resolved_unblind = bool(unblind)

    region_plot_cfg = REGION_PLOTTING.get(region_upper, {})

    skip_variables = set(region_plot_cfg.get("skip_variables", []))
    analysis_bins = {}
    for var_name, spec in region_plot_cfg.get("analysis_bins", {}).items():
        if isinstance(spec, str):
            if spec not in axes_info:
                raise KeyError(
                    f"Analysis bin specification '{spec}' is not defined in axes_info."
                )
            analysis_bins[var_name] = axes_info[spec]["variable"]
        else:
            analysis_bins[var_name] = spec

    channel_rules = _normalize_channel_rules(
        region_plot_cfg.get("channel_transformations")
    )
    sample_removal_rules = region_plot_cfg.get("sample_removals", [])
    category_skip_rules = region_plot_cfg.get("category_skips", [])
    skip_sparse_2d = region_plot_cfg.get("skip_sparse_2d", False)
    channel_mode = region_plot_cfg.get("channel_mode", "per-channel")
    variable_label = region_plot_cfg.get("variable_label", "Variable")
    debug_channel_lists = region_plot_cfg.get("debug_channel_lists", False)
    sumw2_remove_signal = region_plot_cfg.get("sumw2_remove_signal", False)
    sumw2_remove_signal_when_blinded = region_plot_cfg.get(
        "sumw2_remove_signal_when_blinded", False
    )
    use_mc_as_data_when_blinded = region_plot_cfg.get(
        "use_mc_as_data_when_blinded", False
    )

    if region_upper == "CR":
        group_patterns = CR_GRP_PATTERNS
        channel_map = CR_CHAN_DICT
        group_map = populate_group_map(all_samples, group_patterns)
        signal_samples = sorted(set(group_map.get("Signal", [])))
        unblind_default = resolved_unblind
        global CR_GRP_MAP
        CR_GRP_MAP = group_map
    else:
        group_patterns = SR_GRP_PATTERNS
        channel_map = SR_CHAN_DICT
        group_map = populate_group_map(mc_samples + data_samples, group_patterns)
        signal_samples = sorted(
            {
                proc_name
                for group_name in SR_SIGNAL_GROUP_KEYS
                for proc_name in group_map.get(group_name, [])
            }
        )
        unblind_default = resolved_unblind
        global SR_GRP_MAP
        SR_GRP_MAP = group_map

    return RegionContext(
        region_upper,
        dict_of_hists,
        year,
        channel_map,
        group_patterns,
        group_map,
        all_samples,
        mc_samples,
        data_samples,
        samples_to_remove,
        sumw2_hists,
        signal_samples,
        unblind_default,
        lumi_pair,
        skip_variables,
        analysis_bins,
        channel_rules=channel_rules,
        sample_removal_rules=sample_removal_rules,
        category_skip_rules=category_skip_rules,
        skip_sparse_2d=skip_sparse_2d,
        channel_mode=channel_mode,
        variable_label=variable_label,
        debug_channel_lists=debug_channel_lists,
        sumw2_remove_signal=sumw2_remove_signal,
        sumw2_remove_signal_when_blinded=sumw2_remove_signal_when_blinded,
        use_mc_as_data_when_blinded=use_mc_as_data_when_blinded,
    )


def produce_region_plots(region_ctx,save_dir_path,variables,skip_syst_errs,unit_norm_bool,unblind=None):
    dict_of_hists = region_ctx.dict_of_hists
    context_label = f"{region_ctx.name} region"
    variables_to_plot = _resolve_requested_variables(
        dict_of_hists, variables, context_label
    )
    if variables is not None:
        print("Filtered variables:", variables_to_plot)

    print("\n\nAll samples:", region_ctx.all_samples)
    print("\nMC samples:", region_ctx.mc_samples)
    print("\nData samples:", region_ctx.data_samples)
    print("\nVariables:", list(dict_of_hists.keys()))

    unblind_flag = (
        region_ctx.unblind_default if unblind is None else bool(unblind)
    )
    stat_only_plots = 0
    stat_and_syst_plots = 0

    for var_name in variables_to_plot:
        if "sumw2" in var_name:
            continue
        if var_name in region_ctx.skip_variables:
            continue

        histo = dict_of_hists[var_name]
        is_sparse2d = _is_sparse_2d_hist(histo)
        if is_sparse2d and region_ctx.skip_sparse_2d:
            continue
        if is_sparse2d and (var_name not in axes_info_2d) and ("_vs_" not in var_name):
            print(
                f"Warning: Histogram '{var_name}' identified as sparse 2D but lacks metadata; falling back to 1D plotting."
            )
            is_sparse2d = False

        label = region_ctx.variable_label
        print(f"\n{label}: {var_name}")

        channel_transformations = _resolve_channel_transformations(
            region_ctx, var_name
        )
        channel_dict = _apply_channel_dict_transformations(
            region_ctx.channel_map, channel_transformations
        )

        hist_mc = histo.remove("process", region_ctx.samples_to_remove["mc"])
        hist_data = histo.remove("process", region_ctx.samples_to_remove["data"])
        hist_mc_sumw2_orig = region_ctx.sumw2_hists.get(var_name)
        if hist_mc_sumw2_orig is not None:
            hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
                "process", region_ctx.samples_to_remove["mc"]
            )
            if region_ctx.sumw2_remove_signal and region_ctx.signal_samples:
                existing_signal = [
                    sample
                    for sample in region_ctx.signal_samples
                    if sample in yt.get_cat_lables(hist_mc_sumw2_orig, "process")
                ]
                if existing_signal:
                    hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
                        "process", existing_signal
                    )
            if (
                region_ctx.sumw2_remove_signal_when_blinded
                and region_ctx.signal_samples
                and not unblind_flag
            ):
                hist_mc_sumw2_orig = hist_mc_sumw2_orig.remove(
                    "process", region_ctx.signal_samples
                )

        if region_ctx.debug_channel_lists:
            try:
                channels_lst = yt.get_cat_lables(dict_of_hists[var_name], "channel")
            except Exception:
                channels_lst = []
            print("channels:", channels_lst)

        for hist_cat, channel_bins in channel_dict.items():
            if _should_skip_category(
                region_ctx.category_skip_rules, hist_cat, var_name
            ):
                continue

            if region_ctx.channel_mode == "aggregate":
                print(f"\n\tCategory: {hist_cat}")

            validate_channel_group(
                [hist_mc, hist_data],
                channel_bins,
                channel_transformations,
                region=region_ctx.name,
                subgroup=hist_cat,
                variable=var_name,
            )

            save_dir_path_tmp = os.path.join(save_dir_path, hist_cat)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            if region_ctx.channel_mode == "aggregate":
                axes_to_integrate_dict = {"channel": channel_bins}
                try:
                    hist_mc_integrated = yt.integrate_out_cats(
                        yt.integrate_out_appl(hist_mc, hist_cat),
                        axes_to_integrate_dict,
                    )[{'channel': sum}]
                    hist_data_integrated = yt.integrate_out_cats(
                        yt.integrate_out_appl(hist_data, hist_cat),
                        axes_to_integrate_dict,
                    )[{'channel': sum}]
                except Exception:
                    continue
                hist_mc_sumw2_integrated = None
                if hist_mc_sumw2_orig is not None:
                    try:
                        hist_mc_sumw2_integrated = yt.integrate_out_cats(
                            yt.integrate_out_appl(hist_mc_sumw2_orig, hist_cat),
                            axes_to_integrate_dict,
                        )[{'channel': sum}]
                    except Exception:
                        hist_mc_sumw2_integrated = None

                samples_to_rm = _collect_samples_to_remove(
                    region_ctx.sample_removal_rules, hist_cat, region_ctx
                )
                hist_mc_integrated = hist_mc_integrated.remove(
                    "process", samples_to_rm
                )
                if hist_mc_sumw2_integrated is not None:
                    hist_mc_sumw2_integrated = hist_mc_sumw2_integrated.remove(
                        "process", samples_to_rm
                    )

                p_err_arr = None
                m_err_arr = None
                p_err_arr_ratio = None
                m_err_arr_ratio = None
                syst_err_mode = False
                if not (is_sparse2d or skip_syst_errs):
                    rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                        hist_mc_integrated,
                        region_ctx.group_map,
                        group_type=region_ctx.name,
                    )
                    shape_systs_summed_arr_m, shape_systs_summed_arr_p = get_shape_syst_arrs(
                        hist_mc_integrated,
                        group_type=region_ctx.name,
                    )
                    if var_name == "njets":
                        diboson_samples = region_ctx.group_map.get("Diboson", [])
                        if diboson_samples:
                            db_hist = (
                                hist_mc_integrated.integrate(
                                    "process", diboson_samples
                                )[{'process': sum}]
                                .integrate("systematic", "nominal")
                                .eval({})[()]
                            )
                            shape_systs_summed_arr_p = shape_systs_summed_arr_p + get_diboson_njets_syst_arr(
                                db_hist, bin0_njets=0
                            )
                            shape_systs_summed_arr_m = shape_systs_summed_arr_m + get_diboson_njets_syst_arr(
                                db_hist, bin0_njets=0
                            )
                    nom_arr_all = (
                        hist_mc_integrated[{"process": sum}]
                        .integrate("systematic", "nominal")
                        .eval({})[()][1:]
                    )
                    sqrt_sum_p = np.sqrt(
                        shape_systs_summed_arr_p + rate_systs_summed_arr_p
                    )[1:]
                    sqrt_sum_m = np.sqrt(
                        shape_systs_summed_arr_m + rate_systs_summed_arr_m
                    )[1:]
                    p_err_arr = nom_arr_all + sqrt_sum_p
                    m_err_arr = nom_arr_all - sqrt_sum_m
                    with np.errstate(divide="ignore", invalid="ignore"):
                        p_err_arr_ratio = np.where(
                            nom_arr_all > 0, p_err_arr / nom_arr_all, 1
                        )
                        m_err_arr_ratio = np.where(
                            nom_arr_all > 0, m_err_arr / nom_arr_all, 1
                        )
                    syst_err_mode = "total" if unblind_flag else True

                if is_sparse2d:
                    hist_mc_nominal = hist_mc_integrated[{"process": sum}].integrate(
                        "systematic", "nominal"
                    )
                    hist_data_nominal = hist_data_integrated[{"process": sum}].integrate(
                        "systematic", "nominal"
                    )
                    if hist_mc_nominal.empty():
                        print(
                            f"Empty histogram for hist_cat={hist_cat} var_name={var_name}, skipping 2D plot."
                        )
                        continue
                    if hist_data_nominal.empty():
                        print(
                            f"Empty data histogram for hist_cat={hist_cat} var_name={var_name}, skipping 2D plot."
                        )
                        continue
                    fig = make_sparse2d_fig(
                        hist_mc_nominal,
                        hist_data_nominal,
                        var_name,
                        channel_name=hist_cat,
                        lumitag=region_ctx.lumi_pair[0],
                        comtag=region_ctx.lumi_pair[1],
                        per_panel=True,
                    )
                else:
                    hist_mc_integrated = hist_mc_integrated.integrate(
                        "systematic", "nominal"
                    )
                    if hist_mc_sumw2_orig is not None and hist_mc_sumw2_integrated is not None:
                        hist_mc_sumw2_integrated = hist_mc_sumw2_integrated.integrate(
                            "systematic", "nominal"
                        )
                    hist_data_integrated = hist_data_integrated.integrate(
                        "systematic", "nominal"
                    )
                    if hist_mc_integrated.empty():
                        print(f"Empty {hist_mc_integrated=}")
                        continue
                    if hist_data_integrated.empty():
                        print(f"Empty {hist_data_integrated=}")
                        continue
                    x_range = (0, 250) if var_name == "ht" else None
                    group = {k: v for k, v in region_ctx.group_map.items() if v}
                    stacked_kwargs = {
                        "h_mc_sumw2": hist_mc_sumw2_integrated,
                        "syst_err": syst_err_mode,
                        "err_p_syst": p_err_arr,
                        "err_m_syst": m_err_arr,
                        "err_ratio_p_syst": p_err_arr_ratio,
                        "err_ratio_m_syst": m_err_arr_ratio,
                        "unblind": unblind_flag,
                        "set_x_lim": x_range,
                    }
                    bins_override = region_ctx.analysis_bins.get(var_name)
                    if bins_override is not None:
                        stacked_kwargs["bins"] = bins_override
                    fig = make_region_stacked_ratio_fig(
                        hist_mc_integrated,
                        hist_data_integrated,
                        unit_norm_bool,
                        var=var_name,
                        group=group,
                        lumitag=region_ctx.lumi_pair[0] if region_ctx.lumi_pair else None,
                        comtag=region_ctx.lumi_pair[1] if region_ctx.lumi_pair else None,
                        **stacked_kwargs,
                    )
                title = hist_cat + "_" + var_name
                if unit_norm_bool:
                    title = title + "_unitnorm"
                has_syst_inputs = any(
                    err is not None
                    for err in (
                        p_err_arr,
                        m_err_arr,
                        p_err_arr_ratio,
                        m_err_arr_ratio,
                    )
                )
                if isinstance(fig, dict):
                    combined_fig = fig["combined"]
                    combined_fig.savefig(
                        os.path.join(save_dir_path_tmp, title),
                        bbox_inches="tight",
                        pad_inches=0.05,
                    )
                    suffix_map = {"mc": "_MC", "data": "_data", "ratio": "_ratio"}
                    for key, panel_fig in fig.items():
                        if key == "combined":
                            continue
                        suffix = suffix_map.get(key, f"_{key}")
                        panel_fig.savefig(
                            os.path.join(save_dir_path_tmp, f"{title}{suffix}"),
                            bbox_inches="tight",
                            pad_inches=0.05,
                        )
                else:
                    fig.savefig(
                        os.path.join(save_dir_path_tmp, title),
                        bbox_inches="tight",
                        pad_inches=0.05,
                    )
                if has_syst_inputs:
                    stat_and_syst_plots += 1
                else:
                    stat_only_plots += 1
            elif region_ctx.channel_mode == "per-channel":
                channels = [
                    chan
                    for chan in channel_bins
                    if chan in hist_mc.axes["channel"]
                ]
                if not channels:
                    continue
                hist_mc_channel = hist_mc.integrate("channel", channels)[{'channel': sum}]
                hist_mc_integrated = hist_mc_channel.integrate(
                    "systematic", "nominal"
                )
                hist_mc_sumw2 = None
                if hist_mc_sumw2_orig is not None:
                    channels_sumw2 = [
                        chan
                        for chan in channel_bins
                        if chan in hist_mc_sumw2_orig.axes["channel"]
                    ]
                    if channels_sumw2:
                        hist_mc_sumw2 = hist_mc_sumw2_orig.integrate(
                            "channel", channels_sumw2
                        )[{'channel': sum}]
                        hist_mc_sumw2 = hist_mc_sumw2.integrate(
                            "systematic", "nominal"
                        )
                channels_data = [
                    chan
                    for chan in channel_bins
                    if chan in hist_data.axes["channel"]
                ]
                hist_data_channel = hist_data.integrate(
                    "channel", channels_data
                )[{'channel': sum}]
                hist_data_integrated = hist_data_channel.integrate(
                    "systematic", "nominal"
                )

                syst_err = False
                err_p_syst = None
                err_m_syst = None
                err_ratio_p_syst = None
                err_ratio_m_syst = None
                if not skip_syst_errs:
                    try:
                        rate_systs_summed_arr_m, rate_systs_summed_arr_p = get_rate_syst_arrs(
                            hist_mc_channel,
                            region_ctx.group_map,
                            group_type=region_ctx.name,
                        )
                        shape_systs_summed_arr_m, shape_systs_summed_arr_p = get_shape_syst_arrs(
                            hist_mc_channel,
                            group_type=region_ctx.name,
                        )
                    except Exception as exc:
                        print(
                            f"Warning: Failed to compute {region_ctx.name} systematics for {hist_cat} {var_name}: {exc}"
                        )
                    else:
                        nom_arr_all = (
                            hist_mc_channel[{"process": sum}]
                            .integrate("systematic", "nominal")
                            .eval({})[()][1:]
                        )
                        sqrt_sum_p = np.sqrt(
                            shape_systs_summed_arr_p + rate_systs_summed_arr_p
                        )[1:]
                        sqrt_sum_m = np.sqrt(
                            shape_systs_summed_arr_m + rate_systs_summed_arr_m
                        )[1:]
                        err_p_syst = nom_arr_all + sqrt_sum_p
                        err_m_syst = nom_arr_all - sqrt_sum_m
                        with np.errstate(divide="ignore", invalid="ignore"):
                            err_ratio_p_syst = np.where(
                                nom_arr_all > 0, err_p_syst / nom_arr_all, 1
                            )
                            err_ratio_m_syst = np.where(
                                nom_arr_all > 0, err_m_syst / nom_arr_all, 1
                            )
                        syst_err = True

                if not hist_mc_integrated.eval({}):
                    print("Warning: empty mc histo, continuing")
                    continue
                if unblind_flag and not hist_data_integrated.eval({}):
                    print("Warning: empty data histo, continuing")
                    continue

                hist_data_to_plot = (
                    hist_data_integrated
                    if (unblind_flag or not region_ctx.use_mc_as_data_when_blinded)
                    else copy.deepcopy(hist_mc_integrated)
                )
                year_str = region_ctx.year if region_ctx.year is not None else "ULall"
                title = f"{hist_cat}_{var_name}_{year_str}"
                bins_override = region_ctx.analysis_bins.get(var_name)
                default_bins = axes_info[var_name]["variable"] if var_name in axes_info else None
                stacked_kwargs = {
                    "group": region_ctx.group_map,
                    "lumitag": region_ctx.lumi_pair[0] if region_ctx.lumi_pair else None,
                    "comtag": region_ctx.lumi_pair[1] if region_ctx.lumi_pair else None,
                    "h_mc_sumw2": hist_mc_sumw2,
                    "syst_err": syst_err,
                    "err_p_syst": err_p_syst,
                    "err_m_syst": err_m_syst,
                    "err_ratio_p_syst": err_ratio_p_syst,
                    "err_ratio_m_syst": err_ratio_m_syst,
                    "unblind": unblind_flag,
                }
                bins_to_use = bins_override if bins_override is not None else default_bins
                if bins_to_use is not None:
                    stacked_kwargs["bins"] = bins_to_use
                fig = make_region_stacked_ratio_fig(
                    hist_mc_integrated,
                    hist_data_to_plot,
                    var=var_name,
                    unit_norm_bool=False,
                    **stacked_kwargs,
                )
                fig.savefig(
                    os.path.join(save_dir_path_tmp, title),
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                has_syst_inputs = any(
                    err is not None
                    for err in (
                        err_p_syst,
                        err_m_syst,
                        err_ratio_p_syst,
                        err_ratio_m_syst,
                    )
                )
                if has_syst_inputs:
                    stat_and_syst_plots += 1
                else:
                    stat_only_plots += 1
            else:
                raise ValueError(
                    f"Unsupported channel_mode '{region_ctx.channel_mode}'"
                )

            if "www" in save_dir_path_tmp:
                make_html(save_dir_path_tmp)

    print(
        f"[{region_ctx.name}] Produced {stat_and_syst_plots} plots with stat⊕syst uncertainties and {stat_only_plots} plots with stat-only bands",
        end="",
    )
    if save_dir_path:
        print(f" in {save_dir_path}")
    else:
        print()


def _ensure_list(values):
    if isinstance(values, str):
        return [values]
    return list(values)


# Group bins in a hist, returns a new hist
def group_bins(histo, bin_map, axis_name="process", drop_unspecified=False):

    bin_map_copy = copy.deepcopy(bin_map)  # Don't want to edit the original
    normalized_map = OrderedDict(
        (group, _ensure_list(bins)) for group, bins in bin_map_copy.items()
    )

    axis_categories = list(histo.axes[axis_name])
    axis_category_set = set(axis_categories)

    if not drop_unspecified:
        specified = {item for bins in normalized_map.values() for item in bins}
        for category in axis_categories:
            if category not in specified:
                normalized_map.setdefault(category, [category])

    requested = {item for bins in normalized_map.values() for item in bins}
    missing = sorted(requested - axis_category_set)
    if missing:
        raise ValueError(
            f"Requested {axis_name} bins not found in histogram: {', '.join(missing)}"
        )

    return histo.group(axis_name, normalized_map)


######### Functions for getting info from the systematics json #########

# Match a given sample name to whatever it is called in the json
# Will return None if a match is not found
def get_scale_name(sample_name,sample_group_map,group_type="CR"):
    scale_name_for_json = None
    if sample_name in sample_group_map.get("Conv", []):
        scale_name_for_json = "convs"
    elif sample_name in sample_group_map.get("Diboson", []):
        scale_name_for_json = "Diboson"
    elif sample_name in sample_group_map.get("Triboson", []):
        scale_name_for_json = "Triboson"
    elif _sample_in_signal_group(sample_name, sample_group_map, group_type):
        wc_matches = [proc_str for proc_str in SIGNAL_WC_MATCHES if proc_str in sample_name]
        if group_type == "CR":
            if len(wc_matches) == 1:
                scale_name_for_json = wc_matches[0]
        else:
            if wc_matches:
                # This should only match once, but maybe we should put a check to enforce this
                scale_name_for_json = wc_matches[0]
    return scale_name_for_json

# This function gets the tag that indicates how a particualr systematic is correlated
#   - For pdf_scale this corresponds to the initial state (e.g. gg)
#   - For qcd_scale this corresponds to the process type (e.g. VV)
# For any systemaitc or process that is not included in the correlations json we return None
def get_correlation_tag(uncertainty_name,proc_name,sample_group_map,group_type="CR"):
    proc_name_in_json = get_scale_name(proc_name,sample_group_map,group_type=group_type)
    corr_tag = None
    # Right now we only have two types of uncorrelated rate systematics
    if uncertainty_name in ["qcd_scale","pdf_scale"]:
        if proc_name_in_json is not None:
            if proc_name_in_json == "convs":
                # Special case for conversions since we estimate from LO sample, we do not have qcd uncty
                # Would be better to handle this in a more general way
                corr_tag = None
            else:
                corr_tag = grs.get_correlation_tag(uncertainty_name,proc_name_in_json)
    return corr_tag

# This function gets all of the the rate systematics from the json file
# Returns a dictionary with all of the uncertainties
# If the sample does not have an uncertainty in the json, an uncertainty of 0 is returned for that category
def get_rate_systs(sample_name,sample_group_map,group_type="CR"):

    # Figure out the name of the appropriate sample in the syst rate json (if the proc is in the json)
    scale_name_for_json = get_scale_name(sample_name,sample_group_map,group_type=group_type)

    # Get the lumi uncty for this sample (same for all samles)
    lumi_uncty = grs.get_syst("lumi")

    # Get the flip uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if sample_name in sample_group_map["Flips"]:
        flip_uncty = grs.get_syst("charge_flips","charge_flips_sm")
    else:
        flip_uncty = [1.0,1,0]

    # Get the scale uncty from the json (if there is not an uncertainty for this sample, return 1 since the uncertainties are multiplicative)
    if scale_name_for_json is not None:
        if scale_name_for_json in PROC_WITHOUT_PDF_RATE_SYST:
            # Special cases for when we do not have a pdf uncty (this is a really brittle workaround)
            # NOTE Someday should fix this, it's a really hardcoded and brittle and bad workaround
            pdf_uncty = [1.0,1,0]
        else:
            pdf_uncty = grs.get_syst("pdf_scale",scale_name_for_json)
        if scale_name_for_json == "convs":
            # Special case for conversions, since we estimate these from a LO sample, so we don't have an NLO uncty here
            # Would be better to handle this in a more general way
            qcd_uncty = [1.0,1,0]
        else:
            # In all other cases, use the qcd scale uncty that we have for the process
            qcd_uncty = grs.get_syst("qcd_scale",scale_name_for_json)
    else:
        pdf_uncty = [1.0,1,0]
        qcd_uncty = [1.0,1,0]

    out_dict = {"pdf_scale":pdf_uncty, "qcd_scale":qcd_uncty, "lumi":lumi_uncty, "charge_flips":flip_uncty}
    return out_dict


# Wrapper for getting plus and minus rate arrs
def get_rate_syst_arrs(base_histo,proc_group_map,group_type="CR"):

    # Fill dictionary with the rate uncertainty arrays (with correlated ones organized together)
    rate_syst_arr_dict = {}
    for rate_sys_type in grs.get_syst_lst():
        rate_syst_arr_dict[rate_sys_type] = {}
        for sample_name in yt.get_cat_lables(base_histo,"process"):

            # Build the plus and minus arrays from the rate uncertainty number and the nominal arr
            rate_syst_dict = get_rate_systs(sample_name,proc_group_map,group_type=group_type)
            thissample_nom_arr = base_histo.integrate("process",sample_name).integrate("systematic","nominal").eval({})[()]
            p_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][1]) - thissample_nom_arr # Difference between positive fluctuation and nominal
            m_arr = thissample_nom_arr*(rate_syst_dict[rate_sys_type][0]) - thissample_nom_arr # Difference between positive fluctuation and nominal

            # Put the arrays into the correlation dict (organizing correlated ones together)
            correlation_tag = get_correlation_tag(rate_sys_type,sample_name,proc_group_map,group_type=group_type)
            out_key_name = rate_sys_type
            if correlation_tag is not None: out_key_name += "_"+correlation_tag
            if out_key_name not in rate_syst_arr_dict[rate_sys_type]:
                rate_syst_arr_dict[rate_sys_type][out_key_name] = {"p":[],"m":[]}
            rate_syst_arr_dict[rate_sys_type][out_key_name]["p"].append(p_arr)
            rate_syst_arr_dict[rate_sys_type][out_key_name]["m"].append(m_arr)

    # Now sum the linearly correlated ones and then square everything
    all_rates_p_sumw2_lst = []
    all_rates_m_sumw2_lst = []
    for syst_name in rate_syst_arr_dict.keys():
        for correlated_syst_group in rate_syst_arr_dict[syst_name]:
            sum_p_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["p"])
            sum_m_arrs = sum(rate_syst_arr_dict[syst_name][correlated_syst_group]["m"])
            all_rates_p_sumw2_lst.append(sum_p_arrs*sum_p_arrs)
            all_rates_m_sumw2_lst.append(sum_m_arrs*sum_m_arrs)

    return [sum(all_rates_m_sumw2_lst),sum(all_rates_p_sumw2_lst)]

# Wrapper for getting plus and minus shape arrs
def get_shape_syst_arrs(base_histo,group_type="CR"):

    # Get the list of systematic base names (i.e. without the up and down tags)
    # Assumes each syst has a "systnameUp" and a "systnameDown" category on the systematic axis
    syst_var_lst = []
    all_syst_var_lst = yt.get_cat_lables(base_histo,"systematic")
    for syst_var_name in all_syst_var_lst:
        if syst_var_name.endswith("Up"):
            syst_name_base = "Up".join(syst_var_name.split("Up")[:-1])
            if syst_name_base not in syst_var_lst:
                syst_var_lst.append(syst_name_base)

    # Sum each systematic's contribtuions for all samples together (e.g. the ISR for all samples is summed linearly)
    p_arr_rel_lst = []
    m_arr_rel_lst = []
    for syst_name in syst_var_lst:
        # Skip the variation of renorm and fact together, since we're treating as independent
        if syst_name == "renormfact": continue

        relevant_samples_lst = yt.get_cat_lables(base_histo.integrate("systematic",syst_name+"Up"), "process") # The samples relevant to this syst
        n_arr = base_histo.integrate("process",relevant_samples_lst)[{'process': sum}].integrate("systematic","nominal").eval({})[()] # Sum of all samples for nominal variation

        # Special handling of renorm and fact
        # Uncorrelate these systs across the processes (though leave processes in groups like dibosons correlated to be consistent with SR)
        if (syst_name == "renorm") or (syst_name == "fact"):
            grp_map = CR_GRP_MAP if group_type == "CR" else SR_GRP_MAP
            p_arr_rel,m_arr_rel = get_decorrelated_uncty(syst_name,grp_map,relevant_samples_lst,base_histo,n_arr)

        # If the syst is not renorm or fact, just treat it normally (correlate across all processes)
        else:
            u_arr_sum = base_histo.integrate("process",relevant_samples_lst)[{"process": sum}].integrate("systematic",syst_name+"Up").eval({})[()]   # Sum of all samples for up variation
            d_arr_sum = base_histo.integrate("process",relevant_samples_lst)[{"process": sum}].integrate("systematic",syst_name+"Down").eval({})[()] # Sum of all samples for down variation

            u_arr_rel = u_arr_sum - n_arr # Diff with respect to nominal
            d_arr_rel = d_arr_sum - n_arr # Diff with respect to nominal
            p_arr_rel = np.where(u_arr_rel>0,u_arr_rel,d_arr_rel) # Just the ones that increase the yield
            m_arr_rel = np.where(u_arr_rel<0,u_arr_rel,d_arr_rel) # Just the ones that decrease the yield

        # Square and append this syst to the return lists
        p_arr_rel_lst.append(p_arr_rel*p_arr_rel) # Square each element in the arr and append the arr to the out list
        m_arr_rel_lst.append(m_arr_rel*m_arr_rel) # Square each element in the arr and append the arr to the out list

    return [sum(m_arr_rel_lst), sum(p_arr_rel_lst)]


# Special case for renorm and fact, as these are decorrelated across processes
# Sorry to anyone who tries to read this in the future, this function is very ad hoc and messy and hard to follow
# Just used in get_shape_syst_arrs()
# Here are a few notes:
#   - This is complicated, so I just symmetrized the errors
#   - The processes are generally correlated across groups (e.g. WZ and ZZ) since this is what's done in the datacard maker for the SR
#   - So the grouping generally follows what's in the CR group map, except in the case of signal
#       - Since in the SR all signal processes are uncorrelated for these systs, we also uncorrelate here
#       - Note there are caveats to this:
#           * In the SR, TTZToLL_M1to10 and TTToSemiLeptonic and TTTo2L2Nu are all grouped into ttll
#           * Here in the CR TTZToLL_M1to10 is part of signal group, but TTToSemiLeptonic and TTTo2L2Nu are in their own ttbar group
#           * So there are two differences with respect to how these processes are grouped in the SR:
#               1) Here TTToSemiLeptonic and TTTo2L2Nu are correlated with each other, but not with ttll
#               2) Here TTZToLL_M1to10 is grouped as part of signal (as in SR) but here _all_ signal processes are uncorrleated so here TTZToLL_M1to10 is uncorrelated with ttll while in SR they would be correlated
def get_decorrelated_uncty(syst_name,grp_map,relevant_samples_lst,base_histo,template_zeros_arr):

    # Initialize the array we will return (ok technically we return sqrt of this arr squared..)
    a_arr_sum = np.zeros_like(template_zeros_arr) # Just using this template_zeros_arr for its size

    # Loop over the groups of processes, generally the processes in the groups will be correlated and the different groups will be uncorrelated
    for proc_grp in grp_map.keys():
        proc_lst = grp_map[proc_grp]
        if proc_grp in ["Nonprompt","Flips","Data"]: continue # Renorm and fact not relevant here
        if proc_lst == []: continue # Nothing here

        # We'll keep all signal processes as uncorrelated, similar to what's done in SR
        if proc_grp == "Signal":
            for proc_name in proc_lst:
                if proc_name not in relevant_samples_lst: continue

                n_arr_proc = base_histo.integrate("process",proc_name)[{"process": sum}].integrate("systematic","nominal").eval({})[()]
                u_arr_proc = base_histo.integrate("process",proc_name)[{"process": sum}].integrate("systematic",syst_name+"Up").eval({})[()]
                d_arr_proc = base_histo.integrate("process",proc_name)[{"process": sum}].integrate("systematic",syst_name+"Down").eval({})[()]

                u_arr_proc_rel = u_arr_proc - n_arr_proc
                d_arr_proc_rel = d_arr_proc - n_arr_proc
                a_arr_proc_rel = (abs(u_arr_proc_rel) + abs(d_arr_proc_rel))/2.0

                a_arr_sum += a_arr_proc_rel*a_arr_proc_rel

        # Otherwise corrleated across groups (e.g. ZZ and WZ, as datacard maker does in SR)
        else:
            n_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic","nominal").eval({})[()]
            u_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic",syst_name+"Up").eval({})[()]
            d_arr_grp = base_histo.integrate("process",proc_lst)[{"process": sum}].integrate("systematic",syst_name+"Down").eval({})[()]
            u_arr_grp_rel = u_arr_grp - n_arr_grp
            d_arr_grp_rel = d_arr_grp - n_arr_grp
            a_arr_grp_rel = (abs(u_arr_grp_rel) + abs(d_arr_grp_rel))/2.0

            a_arr_sum += a_arr_grp_rel*a_arr_grp_rel

    # Before we move on, need to sqrt the outcome since later we'll square before adding in quadrature with other systs
    p_arr_rel =  np.sqrt(a_arr_sum)
    m_arr_rel = -np.sqrt(a_arr_sum)

    return [p_arr_rel,m_arr_rel]


# Get the squared arr for the jet dependent syst (e.g. for diboson jet dependent syst)
def get_diboson_njets_syst_arr(njets_histo_vals_arr,bin0_njets):

    # Get the list of njets vals for which we have SFs
    sf_int_lst = []
    diboson_njets_dict = grs.get_jet_dependent_syst_dict()
    sf_str_lst = list(diboson_njets_dict.keys())
    for s in sf_str_lst: sf_int_lst.append(int(s))
    min_njets = min(sf_int_lst) # The lowest njets bin we have a SF for
    max_njets = max(sf_int_lst) # The highest njets bin we have a SF for

    # Put the SFs into an array that matches the njets hist array
    sf_lst = []
    jet_idx = bin0_njets
    for idx in range(len(njets_histo_vals_arr)):
        if jet_idx < min_njets:
            # We do not apply the syst for these low jet bins
            sf_lst.append(1.0)
        elif jet_idx > max_njets:
            # For jet bins higher than the highest one in the dict, just use the val of the highest one
            sf_lst.append(diboson_njets_dict[str(max_njets)])
        else:
            # In this case, the exact jet bin should be included in the dict so use it directly
            sf_lst.append(diboson_njets_dict[str(jet_idx)])
        jet_idx = jet_idx + 1
    sf_arr = np.array(sf_lst)

    shift = abs(njets_histo_vals_arr - sf_arr*njets_histo_vals_arr)
    shift_sq = shift*shift # Return shift squared so we can combine with other syts in quadrature
    return shift*shift


def _is_sparse_2d_hist(histo):
    if not isinstance(histo, SparseHist):
        return False

    quadratic_axis = next(
        (ax for ax in histo.dense_axes if getattr(ax, "name", None) == "quadratic_term"),
        None,
    )
    if quadratic_axis is not None:
        try:
            # Skip the sparse 2D path only when the quadratic axis has a single bin.
            if histo.axes["quadratic_term"].size > 1:
                return True
        except (KeyError, AttributeError):
            # If the axis cannot be inspected reliably, keep the conservative 2D
            # classification to avoid mis-shaping 1D projections.
            return True

    dense_axes = [ax for ax in histo.dense_axes if ax is not quadratic_axis]
    return len(dense_axes) > 1


######### Plotting functions #########

def make_sparse2d_fig(
    h_mc,
    h_data,
    var,
    channel_name,
    lumitag="138",
    comtag="13",
    per_panel=False,
):
    axes_meta = axes_info_2d.get(var, {})
    axis_cfgs = axes_meta.get("axes", [])
    if len(axis_cfgs) < 2:
        raise ValueError(f"No 2D axis metadata configured for histogram '{var}'.")
    axis_labels = [cfg.get("label", cfg.get("name", "")) for cfg in axis_cfgs]
    cbar_label = axes_meta.get("cbar_label", "Events")
    ratio_meta = axes_meta.get("ratio", {})
    ratio_cbar_label = ratio_meta.get("cbar_label", "Data/MC")

    def _extract_weighted_values(histo):
        view = histo.view(flow=False, as_dict=True)
        if isinstance(view, dict):
            if len(view) == 1:
                view = next(iter(view.values()))
            else:
                # Fall back to the higher-level values helper when multiple
                # categorical entries remain. This preserves the dense layout
                # while still supporting weighted storages.
                view = histo.values(flow=False)

        if hasattr(view, "dtype") and view.dtype.fields:
            if "value" in view.dtype.fields:
                return np.asarray(view["value"], dtype=float)

        try:
            return np.asarray(view, dtype=float)
        except TypeError:
            return np.asarray(np.array(view), dtype=float)

    def _dense_edges(histo):
        return [np.asarray(ax.edges, dtype=float) for ax in histo.axes]

    mc_vals = _extract_weighted_values(h_mc)
    data_vals = _extract_weighted_values(h_data)
    ratio_vals = np.ones_like(data_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(data_vals, mc_vals, out=ratio_vals, where=mc_vals != 0)
    empty_mask = (mc_vals == 0) & (data_vals == 0)
    data_only_mask = (mc_vals == 0) & (data_vals != 0)
    ratio_vals[empty_mask | data_only_mask] = np.nan
    dense_edges = _dense_edges(h_mc)

    def _norm_from_meta(meta_cfg, values):
        if not meta_cfg:
            return None

        norm_cfg = meta_cfg.get("norm")
        if isinstance(norm_cfg, mpl.colors.Normalize):
            return copy.copy(norm_cfg)
        if callable(norm_cfg):
            generated = norm_cfg(values)
            if isinstance(generated, mpl.colors.Normalize):
                return generated

        zlim = meta_cfg.get("zlim")
        if zlim is not None:
            vmin, vmax = zlim
            finite_vals = values[np.isfinite(values)]
            if vmin is None:
                if finite_vals.size:
                    vmin = float(np.nanmin(finite_vals))
                else:
                    vmin = 0.0
            if vmax is None:
                if finite_vals.size:
                    vmax = float(np.nanmax(finite_vals))
                else:
                    vmax = 1.0
            return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        return None

    def _build_norm(values, dataset_key):
        dataset_meta = axes_meta.get(dataset_key, {})
        norm = _norm_from_meta(dataset_meta, values)
        if norm is None:
            norm = _norm_from_meta(axes_meta, values)
        if norm is None:
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size:
                vmax = float(np.nanmax(finite_vals))
            else:
                vmax = 0.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
        return norm

    mc_norm = _build_norm(mc_vals, "mc")
    data_norm = _build_norm(data_vals, "data")

    finite_ratio = ratio_vals[np.isfinite(ratio_vals)]
    if "zlim" in ratio_meta:
        ratio_low, ratio_high = ratio_meta["zlim"]
        span = max(abs(1.0 - ratio_low), abs(ratio_high - 1.0))
        if not np.isfinite(span) or span <= 0:
            span = 0.5
        ratio_vmin = 1.0 - span
        ratio_vmax = 1.0 + span
    else:
        if finite_ratio.size:
            max_dev = float(np.max(np.abs(finite_ratio - 1.0)))
        else:
            max_dev = 0.0
        half_range = max(max_dev, 0.5)
        ratio_vmin = 1.0 - half_range
        ratio_vmax = 1.0 + half_range
    ratio_norm = mpl.colors.TwoSlopeNorm(vmin=ratio_vmin, vcenter=1.0, vmax=ratio_vmax)

    def _apply_panel_margins(fig):
        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08)

    def _make_single_panel(values, norm, title, colorbar_label):
        fig = plt.figure(figsize=(10, 9))
        hep.style.use("CMS")
        ax = fig.add_subplot(111)
        hep.cms.label(ax=ax, lumi=lumitag, com=comtag, fontsize=20.0)
        artists = hep.hist2dplot(
            values,
            ax=ax,
            norm=norm,
            xbins=dense_edges[0],
            ybins=dense_edges[1],
        )
        mesh = getattr(artists, "mesh", None)
        if mesh is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cbar = fig.colorbar(mesh, cax=cax, norm=norm)
            cbar.set_label(colorbar_label, fontsize=18)
            cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel(axis_labels[0], fontsize=20)
        ax.set_ylabel(axis_labels[1], fontsize=20)
        ax.set_title(
            f"{channel_name} {title}" if channel_name else title,
            fontsize=22,
        )
        ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
        _apply_panel_margins(fig)
        return fig

    fig = plt.figure(figsize=(20, 12))
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0.15,
        left=0.06,
        right=0.98,
        top=0.96,
        bottom=0.08,
    )
    top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.12)

    hep.style.use("CMS")

    ax_mc = fig.add_subplot(top_gs[0])
    ax_data = fig.add_subplot(top_gs[1])
    ax_ratio = fig.add_subplot(outer_gs[1])

    axes_top = [ax_mc, ax_data]

    hep.cms.label(ax=ax_mc, lumi=lumitag, com=comtag, fontsize=20.0)
    for ax, plot_hist, title, norm in zip(
        axes_top,
        (mc_vals, data_vals),
        ("MC", "Data"),
        (mc_norm, data_norm),
    ):
        artists = hep.hist2dplot(
            plot_hist,
            ax=ax,
            norm=norm,
            xbins=dense_edges[0],
            ybins=dense_edges[1],
        )
        mesh = getattr(artists, "mesh", None)
        if mesh is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.04)
            cbar = fig.colorbar(mesh, cax=cax, norm=norm)
            cbar.set_label(cbar_label, fontsize=18)
            cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel(axis_labels[0], fontsize=20)
        ax.set_ylabel(axis_labels[1], fontsize=20)
        ax.set_title(
            f"{channel_name} {title}" if channel_name else title,
            fontsize=22,
        )
        ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    ratio_artists = hep.hist2dplot(
        ratio_vals,
        ax=ax_ratio,
        norm=ratio_norm,
        xbins=dense_edges[0],
        ybins=dense_edges[1],
    )
    ratio_mesh = getattr(ratio_artists, "mesh", None)
    if ratio_mesh is not None:
        divider = make_axes_locatable(ax_ratio)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        ratio_cbar = fig.colorbar(ratio_mesh, cax=cax, norm=ratio_norm)
        ratio_cbar.set_label(ratio_cbar_label, fontsize=18)
        ratio_cbar.ax.tick_params(labelsize=16)
    ax_ratio.set_xlabel(axis_labels[0], fontsize=20)
    ax_ratio.set_ylabel(axis_labels[1], fontsize=20)
    ax_ratio.set_title(
        f"{channel_name} Data/MC" if channel_name else "Data/MC",
        fontsize=22,
    )
    ax_ratio.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    for ax in axes_top:
        ax.set_ylabel(axis_labels[1], fontsize=20)
    _apply_panel_margins(fig)

    if not per_panel:
        return fig

    single_panel_figs = {
        "combined": fig,
        "mc": _make_single_panel(mc_vals, mc_norm, "MC", cbar_label),
        "data": _make_single_panel(data_vals, data_norm, "Data", cbar_label),
        "ratio": _make_single_panel(ratio_vals, ratio_norm, "Data/MC", ratio_cbar_label),
    }
    return single_panel_figs


# Takes two histograms and makes a region-level stacked ratio plot (with only one sparse axis, which should be "process").
# One histogram should encode the MC prediction while the other carries the data yields (or MC-substituted data when blinded).
def make_region_stacked_ratio_fig(
    h_mc,
    h_data,
    unit_norm_bool,
    axis='process',
    var='lj0pt',
    bins=None,
    group=None,
    set_x_lim=None,
    err_p=None,
    err_m=None,
    err_ratio_p=None,
    err_ratio_m=None,
    lumitag="138",
    comtag="13",
    h_mc_sumw2=None,
    syst_err=None,
    err_p_syst=None,
    err_m_syst=None,
    err_ratio_p_syst=None,
    err_ratio_m_syst=None,
    unblind=False,
):
    if bins is None:
        bins = []
    if group is None:
        group = {}
    default_colors = [
        "tab:blue", "darkgreen", "tab:orange", "tab:cyan", "tab:purple", "tab:pink",
        "tan", "mediumseagreen", "tab:red", "brown", "goldenrod", "yellow",
        "olive", "coral", "navy", "yellowgreen", "aquamarine", "black", "plum",
        "gray"
    ]

    # Determine which groups are actually present
    grouping = {proc: [p for p in group[proc] if p in h_mc.axes['process']]
                for proc in group if any(p in h_mc.axes['process'] for p in group[proc])}

    colors = []
    for i, proc in enumerate(grouping):
        c = FILL_COLORS.get(proc)
        if c is None:
            c = default_colors[i % len(default_colors)]
        colors.append(c)

    if err_p_syst is None and err_p is not None:
        err_p_syst = err_p
    if err_m_syst is None and err_m is not None:
        err_m_syst = err_m
    if err_ratio_p_syst is None and err_ratio_p is not None:
        err_ratio_p_syst = err_ratio_p
    if err_ratio_m_syst is None and err_ratio_m is not None:
        err_ratio_m_syst = err_ratio_m

    display_label = axes_info.get(var, {}).get("label", var)

    # Create the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 9),
        gridspec_kw={"height_ratios": (4, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    # Set up the colors for each stacked process

    # Normalize if we want to do that
    mc_norm_factor = 1.0
    if unit_norm_bool:
        sum_mc = 0
        sum_data = 0
        for sample in h_mc.eval({}):
            sum_mc = sum_mc + sum(h_mc.eval({})[sample])
        for sample in h_data.eval({}):
            sum_data = sum_data + sum(h_data.eval({})[sample])
        mc_norm_factor = 1.0/sum_mc
        h_mc.scale(mc_norm_factor)
        h_data.scale(1.0/sum_data)

    # Plot the MC
    years = {}
    for axis_name in h_mc.axes[axis]:
        name = axis_name.split('UL')[0].replace('_private', '').replace('_central', '')
        if name in years:
            years[name].append(axis_name)
        else:
            years[name] = [axis_name]
    hep.style.use("CMS")
    plt.sca(ax)
    cms_label = hep.cms.label(lumi=lumitag, com=comtag, fontsize=18.0)

    # Use the grouping information determined above
    def _get_grouped_vals(hist_obj, grouping_map):
        grouped_values = {}
        for proc_name, members in grouping_map.items():
            grouped_hist = hist_obj[{"process": members}][{"process": sum}]
            grouped_values[proc_name] = grouped_hist.as_hist({}).values(flow=True)[1:]
        return grouped_values

    mc_vals = _get_grouped_vals(h_mc, grouping)
    mc_sumw2_vals = {}
    if h_mc_sumw2 is not None:
        try:
            available_processes = set(h_mc_sumw2.axes[axis])
        except KeyError:
            available_processes = set()
        template = next(iter(mc_vals.values())) if mc_vals else h_mc[{"process": sum}].as_hist({}).values(flow=True)[1:]
        for proc_name, members in grouping.items():
            valid_members = [m for m in members if m in available_processes]
            missing_members = [m for m in members if m not in available_processes]

            grouped_vals = np.zeros_like(template)
            if valid_members:
                grouped_hist = h_mc_sumw2[{"process": valid_members}][{"process": sum}]
                grouped_vals = grouped_hist.as_hist({}).values(flow=True)[1:]
                if unit_norm_bool:
                    grouped_vals = grouped_vals * mc_norm_factor**2

            fallback_vals = np.zeros_like(template)
            if missing_members:
                fallback_hist = h_mc[{"process": missing_members}][{"process": sum}]
                fallback_vals = fallback_hist.as_hist({}).values(flow=True)[1:]
                if unit_norm_bool:
                    fallback_vals = fallback_vals * mc_norm_factor

            mc_sumw2_vals[proc_name] = grouped_vals + fallback_vals

    bins = h_data[{'process': sum}].as_hist({}).axes[var].edges
    bins = np.append(bins, [bins[-1] + (bins[-1] - bins[-2])*0.3])
    hep.histplot(
        list(mc_vals.values()),
        ax=ax,
        bins=bins,
        stack=True,
        density=unit_norm_bool,
        label=list(mc_vals.keys()),
        histtype='fill',
        color=colors,
    )

    #Plot the data
    hep.histplot(
        h_data[{'process':sum}].as_hist({}).values(flow=True)[1:],
        #error_opts = DATA_ERR_OPS,
        ax=ax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        label='Data',
        #flow='show',
        histtype='errorbar',
        **DATA_ERR_OPS,
    )

    # Make the ratio plot
    data_vals_flow = h_data[{'process': sum}].as_hist({}).values(flow=True)
    mc_vals_flow = h_mc[{"process": sum}].as_hist({}).values(flow=True)

    def _safe_divide(num, denom, default, zero_over_zero=None):
        num_arr = np.asarray(num, dtype=float)
        denom_arr = np.asarray(denom, dtype=float)
        out = np.full_like(num_arr, default, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            valid = denom_arr != 0
            np.divide(num_arr, denom_arr, out=out, where=valid)
        if zero_over_zero is not None:
            zero_zero_mask = (denom_arr == 0) & (num_arr == 0)
            out[zero_zero_mask] = zero_over_zero
        return out

    ratio_vals_flow = _safe_divide(
        data_vals_flow,
        mc_vals_flow,
        default=np.nan,
        zero_over_zero=1.0,
    )
    ratio_yerr_flow = _safe_divide(np.sqrt(data_vals_flow), data_vals_flow, default=0.0)
    ratio_yerr_flow[mc_vals_flow == 0] = np.nan

    hep.histplot(
        ratio_vals_flow[1:],
        yerr=ratio_yerr_flow[1:],
        #error_opts = DATA_ERR_OPS,
        ax=rax,
        bins=bins,
        stack=False,
        density=unit_norm_bool,
        #flow='show',
        histtype='errorbar',
        **DATA_ERR_OPS,
    )

    # Plot the syst error
    mc_totals = h_mc[{"process": sum}].as_hist({}).values(flow=True)[1:]
    if h_mc_sumw2 is not None:
        if mc_sumw2_vals:
            summed_mc_sumw2 = np.sum(list(mc_sumw2_vals.values()), axis=0)
        else:
            summed_mc_sumw2 = h_mc_sumw2[{"process": sum}].as_hist({}).values(flow=True)[1:]
            if unit_norm_bool:
                summed_mc_sumw2 = summed_mc_sumw2 * mc_norm_factor**2
    else:
        summed_mc_sumw2 = mc_totals * (mc_norm_factor if unit_norm_bool else 1)

    mc_stat_unc = np.sqrt(np.clip(summed_mc_sumw2, a_min=0, a_max=None))

    has_syst_arrays = all(
        arr is not None
        for arr in (err_p_syst, err_m_syst, err_ratio_p_syst, err_ratio_m_syst)
    )

    valid_modes = {"stat", "syst", "total"}
    if isinstance(syst_err, str) and syst_err.lower() in valid_modes:
        band_mode = syst_err.lower()
    elif isinstance(syst_err, bool):
        if syst_err and has_syst_arrays:
            band_mode = "total"
        else:
            band_mode = "stat"
    else:
        if has_syst_arrays:
            band_mode = "total"
        else:
            band_mode = "stat"

    def _append_last(arr):
        if arr is None or len(arr) == 0:
            return arr
        return np.append(arr, arr[-1])

    mc_stat_up = mc_totals + mc_stat_unc
    mc_stat_down = np.clip(mc_totals - mc_stat_unc, a_min=0, a_max=None)
    stat_fraction = _safe_divide(mc_stat_unc, mc_totals, default=0.0)
    ratio_stat_up = 1 + stat_fraction
    ratio_stat_down = 1 - stat_fraction

    mc_stat_band_up = _append_last(mc_stat_up)
    mc_stat_band_down = _append_last(mc_stat_down)
    ratio_stat_band_up = _append_last(ratio_stat_up)
    ratio_stat_band_down = _append_last(ratio_stat_down)

    syst_up = syst_down = ratio_syst_up = ratio_syst_down = None
    mc_total_band_up = mc_total_band_down = None
    ratio_total_band_up = ratio_total_band_down = None
    if has_syst_arrays:
        syst_up = np.asarray(err_p_syst)
        syst_down = np.asarray(err_m_syst)
        ratio_syst_up = np.asarray(err_ratio_p_syst)
        ratio_syst_down = np.asarray(err_ratio_m_syst)

        syst_up_diff = np.clip(syst_up - mc_totals, a_min=0, a_max=None)
        syst_down_diff = np.clip(mc_totals - syst_down, a_min=0, a_max=None)

        total_unc_up = np.sqrt(mc_stat_unc**2 + syst_up_diff**2)
        total_unc_down = np.sqrt(mc_stat_unc**2 + syst_down_diff**2)

        mc_total_band_up = _append_last(mc_totals + total_unc_up)
        mc_total_band_down = _append_last(np.clip(mc_totals - total_unc_down, a_min=0, a_max=None))

        total_up_fraction = _safe_divide(total_unc_up, mc_totals, default=0.0)
        total_down_fraction = _safe_divide(total_unc_down, mc_totals, default=0.0)
        ratio_total_up = 1 + total_up_fraction
        ratio_total_down = 1 - total_down_fraction
        ratio_total_band_up = _append_last(np.clip(ratio_total_up, a_min=0, a_max=None))
        ratio_total_band_down = _append_last(np.clip(ratio_total_down, a_min=0, a_max=None))

        ratio_syst_band_up = _append_last(ratio_syst_up)
        ratio_syst_band_down = _append_last(ratio_syst_down)
        mc_syst_band_up = _append_last(np.clip(syst_up, a_min=0, a_max=None))
        mc_syst_band_down = _append_last(np.clip(syst_down, a_min=0, a_max=None))
    else:
        ratio_syst_band_up = ratio_syst_band_down = None
        mc_syst_band_up = mc_syst_band_down = None

    stat_label = "Stat. unc."
    syst_label = "Syst. unc."
    total_label = "Stat. ⊕ syst. unc."

    ratio_band_handles = []

    if band_mode == "syst" and has_syst_arrays:
        if mc_syst_band_up is not None and mc_syst_band_down is not None:
            ax.fill_between(
                bins,
                mc_syst_band_down,
                mc_syst_band_up,
                step='post',
                facecolor='none',
                edgecolor='gray',
                label=syst_label,
                hatch='////',
            )
        if ratio_syst_band_up is not None and ratio_syst_band_down is not None:
            ratio_syst_handle = rax.fill_between(
                bins,
                ratio_syst_band_down,
                ratio_syst_band_up,
                step='post',
                facecolor='none',
                edgecolor='gray',
                label=syst_label,
                hatch='////',
            )
            ratio_band_handles.append(ratio_syst_handle)
    else:
        if mc_stat_band_up is not None and mc_stat_band_down is not None:
            ax.fill_between(
                bins,
                mc_stat_band_down,
                mc_stat_band_up,
                step='post',
                facecolor='gray',
                alpha=0.3,
                edgecolor='none',
                label=stat_label,
            )
        if ratio_stat_band_up is not None and ratio_stat_band_down is not None:
            ratio_stat_handle = rax.fill_between(
                bins,
                ratio_stat_band_down,
                ratio_stat_band_up,
                step='post',
                facecolor='gray',
                alpha=0.3,
                edgecolor='none',
                label=stat_label,
            )
            ratio_band_handles.append(ratio_stat_handle)

        show_total = band_mode == "total" and has_syst_arrays
        if show_total:
            if mc_total_band_up is not None and mc_total_band_down is not None:
                ax.fill_between(
                    bins,
                    mc_total_band_down,
                    mc_total_band_up,
                    step='post',
                    facecolor='none',
                    edgecolor='gray',
                    label=total_label,
                    hatch='////',
                )
            if ratio_total_band_up is not None and ratio_total_band_down is not None:
                ratio_total_handle = rax.fill_between(
                    bins,
                    ratio_total_band_down,
                    ratio_total_band_up,
                    step='post',
                    facecolor='none',
                    edgecolor='gray',
                    label=total_label,
                    hatch='////',
                )
                ratio_band_handles.append(ratio_total_handle)

    if ratio_band_handles:
        rax.legend(
            handles=ratio_band_handles,
            loc="upper left",
            fontsize=10,
            frameon=False,
        )

    # Scale the y axis and labels
    ax.autoscale(axis='y')
    ax.set_xlabel(None)
    ax.tick_params(axis='both', labelsize=18, width=1.5, length=6)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,6), useMathText=True)
    ax.yaxis.set_offset_position("left")
    ax.yaxis.offsetText.set_x(-0.07)
    ax.yaxis.offsetText.set_fontsize(18)

    rax.set_ylabel('Ratio', loc='center', fontsize=18)
    rax.set_ylim(0.5,1.5)
    rax.tick_params(axis='both', labelsize=18, width=1.5, length=6)

    fig.canvas.draw()
    xticks = rax.get_xticks()
    xtick_labels = [tick.get_text() for tick in rax.get_xticklabels()]
    if xtick_labels and len(xtick_labels) == len(xticks):
        xtick_labels[-1] = '>500'
        rax.xaxis.set_major_locator(FixedLocator(xticks))
        rax.xaxis.set_major_formatter(FixedFormatter(xtick_labels))

    ax_box = ax.get_position()
    rax_box = rax.get_position()
    ratio_label_fig = None
    ratio_label = rax.yaxis.label
    if ratio_label is not None:
        try:
            ratio_label_pos = np.asarray(ratio_label.get_position(), dtype=float)
            ratio_label_transform = ratio_label.get_transform()
            if ratio_label_transform is not None:
                ratio_label_display = ratio_label_transform.transform([ratio_label_pos])[0]
                ratio_label_fig = fig.transFigure.inverted().transform(ratio_label_display)
        except Exception:
            ratio_label_fig = None

    initial_events_anchor = (rax_box.x0 + rax_box.width, ax_box.y0 + ax_box.height)

    # Set the x axis lims
    if set_x_lim: plt.xlim(set_x_lim)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # Put a legend to the right of the current axis
    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.02), ncol=4, fontsize=16)

    def _finalize_layout(
        fig,
        ax,
        rax,
        legend,
        cms_label,
        display_label,
        label_artist=None,
        events_artist=None,
        ratio_anchor=None,
        events_anchor=None,
    ):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        legend_box = None
        if legend is not None:
            legend_bbox = legend.get_window_extent(renderer=renderer)
            legend_box = legend_bbox.transformed(fig.transFigure.inverted())

        cms_artists = ()
        if cms_label is not None:
            if isinstance(cms_label, (list, tuple)):
                cms_artists = tuple(cms_label)
            else:
                cms_artists = (cms_label,)

        cms_bboxes = []
        cms_positions = []
        for artist in cms_artists:
            if hasattr(artist, "get_window_extent"):
                cms_bbox = artist.get_window_extent(renderer=renderer)
                cms_bboxes.append(cms_bbox)
            if hasattr(artist, "get_position") and hasattr(artist, "get_transform"):
                try:
                    artist_pos = np.asarray(artist.get_position())
                except Exception:
                    continue
                artist_transform = artist.get_transform()
                if artist_transform is None:
                    continue
                try:
                    display_coords = artist_transform.transform(artist_pos)
                except Exception:
                    continue
                fig_coords = fig.transFigure.inverted().transform(display_coords)
                cms_positions.append((artist, fig_coords))

        if legend_box is not None and cms_bboxes:
            cms_box = Bbox.union(cms_bboxes).transformed(fig.transFigure.inverted())
            target_gap = 0.015
            gap = cms_box.y0 - legend_box.y1
            if gap < target_gap:
                delta = target_gap - gap
                ax_box = ax.get_position()
                rax_box = rax.get_position()
                max_shift = rax_box.y0
                shift = min(delta, max_shift)
                if shift > 0:
                    ax.set_position([ax_box.x0, ax_box.y0 - shift, ax_box.width, ax_box.height])
                    rax.set_position([rax_box.x0, rax_box.y0 - shift, rax_box.width, rax_box.height])
                    for artist, fig_coords in cms_positions:
                        artist.set_transform(fig.transFigure)
                        artist.set_position(fig_coords)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    legend_bbox = legend.get_window_extent(renderer=renderer)
                    legend_box = legend_bbox.transformed(fig.transFigure.inverted())
                    cms_bboxes = []
                    for artist in cms_artists:
                        if hasattr(artist, "get_window_extent"):
                            cms_bbox = artist.get_window_extent(renderer=renderer)
                            cms_bboxes.append(cms_bbox)
                    if cms_bboxes:
                        cms_box = Bbox.union(cms_bboxes).transformed(fig.transFigure.inverted())

        axis_bboxes = []
        for axis_obj in (ax, rax):
            try:
                bbox = axis_obj.get_tightbbox(renderer)
            except Exception:
                bbox = None
            if bbox is None:
                continue
            axis_bboxes.append(bbox.transformed(fig.transFigure.inverted()))

        if axis_bboxes:
            rightmost_extent = max(bbox.x1 for bbox in axis_bboxes)
        else:
            rightmost_extent = max(ax.get_position().x1, rax.get_position().x1)

        subplot_params = fig.subplotpars
        safety_margin = 0.003
        max_right = np.nextafter(1.0, 0.0)
        effective_right = min(max_right, rightmost_extent + safety_margin)

        if not np.isclose(effective_right, subplot_params.right):
            stored_positions = [ax.get_position().frozen(), rax.get_position().frozen()]

            plt.subplots_adjust(
                bottom=subplot_params.bottom,
                top=subplot_params.top,
                left=subplot_params.left,
                right=effective_right,
                hspace=subplot_params.hspace,
                wspace=subplot_params.wspace,
            )

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            adjusted = False
            for axis_obj, original_pos in zip((ax, rax), stored_positions):
                updated_pos = axis_obj.get_position()
                delta_y = original_pos.y0 - updated_pos.y0
                if not np.isclose(delta_y, 0.0):
                    axis_obj.set_position(
                        [
                            updated_pos.x0,
                            updated_pos.y0 + delta_y,
                            updated_pos.width,
                            updated_pos.height,
                        ]
                    )
                    adjusted = True

            if adjusted:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

            subplot_params = fig.subplotpars

        def _get_min_axis_y(renderer):
            bboxes = []
            for tick_label in rax.get_xticklabels():
                if not tick_label.get_visible():
                    continue
                text = tick_label.get_text()
                if not text:
                    continue
                bbox = tick_label.get_window_extent(renderer=renderer)
                bboxes.append(bbox.transformed(fig.transFigure.inverted()))

            axis_label = rax.xaxis.label
            if axis_label and axis_label.get_visible():
                axis_bbox = axis_label.get_window_extent(renderer=renderer)
                bboxes.append(axis_bbox.transformed(fig.transFigure.inverted()))

            if bboxes:
                return min(bbox.y0 for bbox in bboxes)

            return rax.get_position().y0

        label_fontsize = rax.yaxis.label.get_size() if rax.yaxis.label else 18
        renderer = fig.canvas.get_renderer()

        temp_text = fig.text(0, 0, display_label, fontsize=label_fontsize)
        temp_bbox = temp_text.get_window_extent(renderer=renderer)
        temp_bbox = temp_bbox.transformed(fig.transFigure.inverted())
        measured_height = temp_bbox.height
        temp_text.remove()

        margin = 0.002
        min_axis_y = _get_min_axis_y(renderer)
        label_y = min_axis_y - measured_height - margin

        subplot_params = fig.subplotpars
        new_bottom = max(0.0, label_y - margin)
        new_bottom = np.clip(new_bottom, 0.0, 1.0)

        if not np.isclose(new_bottom, subplot_params.bottom):
            plt.subplots_adjust(
                bottom=new_bottom,
                top=subplot_params.top,
                left=subplot_params.left,
                right=subplot_params.right,
                hspace=subplot_params.hspace,
                wspace=subplot_params.wspace,
            )

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            min_axis_y = _get_min_axis_y(renderer)
            label_y = min_axis_y - measured_height - margin

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_box = ax.get_position()
        rax_box = rax.get_position()

        ratio_label_fig_x = None
        ratio_label = rax.yaxis.label
        if ratio_label is not None:
            try:
                ratio_label_pos = np.asarray(ratio_label.get_position(), dtype=float)
                ratio_label_transform = ratio_label.get_transform()
                if ratio_label_transform is not None:
                    ratio_label_display = ratio_label_transform.transform([ratio_label_pos])[0]
                    ratio_label_fig = fig.transFigure.inverted().transform(ratio_label_display)
                    ratio_label_fig_x = ratio_label_fig[0]
            except Exception:
                ratio_label_fig_x = None

        if ratio_label_fig_x is None and ratio_anchor is not None:
            ratio_label_fig_x = ratio_anchor[0]

        events_x, events_y = events_anchor if events_anchor is not None else (None, None)

        if ratio_label_fig_x is not None:
            events_x = ratio_label_fig_x
        if events_x is None:
            events_x = rax_box.x0 + rax_box.width

        current_events_y = ax_box.y0 + ax_box.height
        if current_events_y is not None:
            events_y = current_events_y
        elif events_y is None:
            events_y = rax_box.y0 + rax_box.height

        if events_artist is None or not isinstance(events_artist, mpl.text.Text):
            events_artist = fig.text(
                events_x,
                events_y,
                "Events",
                ha="right",
                va="bottom",
                fontsize=label_fontsize,
                rotation=90,
            )
        else:
            events_artist.set_position((events_x, events_y))
            events_artist.set_text("Events")
            events_artist.set_fontsize(label_fontsize)
            events_artist.set_rotation(90)
            events_artist.set_ha("right")
            events_artist.set_va("bottom")

        if label_artist is None or not isinstance(label_artist, mpl.text.Text):
            label_artist = fig.text(
                rax_box.x0 + rax_box.width,
                label_y,
                display_label,
                ha="right",
                va="bottom",
                fontsize=label_fontsize,
            )
        else:
            label_artist.set_position((rax_box.x0 + rax_box.width, label_y))
            label_artist.set_text(display_label)
            label_artist.set_fontsize(label_fontsize)
            label_artist.set_ha("right")
            label_artist.set_va("bottom")

        return label_artist, events_artist

    label_artist = None
    events_artist = None
    for _ in range(2):
        label_artist, events_artist = _finalize_layout(
            fig,
            ax,
            rax,
            legend,
            cms_label,
            display_label,
            label_artist,
            events_artist,
            ratio_label_fig,
            initial_events_anchor,
        )

    return fig

# Takes a hist with one sparse axis and one dense axis, overlays everything on the sparse axis
def make_single_fig(histo,unit_norm_bool,axis=None,bins=[],group=[]):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    hep.style.use("CMS")
    plt.sca(ax)
    hep.cms.label(lumi='7.9804', com='13.6', fontsize=10.0)
    if axis is None:
        hep.histplot(
            histo.eval({})[()][1:-1],
            ax=ax,
            bins=bins,
            stack=False,
            density=unit_norm_bool,
            #clear=False,
            histtype='fill',
        )
    else:
        for axis_name in histo.axes[axis]:
            print(axis_name)
            hep.histplot(
                histo[{axis: axis_name}].eval({})[()][1:-1],
                bins=bins,
                stack=True,
                density=unit_norm_bool,
                label=axis_name,
            )
    plt.legend()
    ax.autoscale(axis='y')
    return fig

# Takes a hist with one sparse axis (axis_name) and one dense axis, overlays everything on the sparse axis
# Makes a ratio of each cateogory on the sparse axis with respect to ref_cat
def make_single_fig_with_ratio(histo,axis_name,cat_ref,var='lj0pt',err_p=None,err_m=None,err_ratio_p=None,err_ratio_m=None):
    # Create the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    # Make the main plot
    hep.histplot(
        histo,
        ax=ax,
        stack=False,
        clear=False,
    )

    # Make the ratio plot
    # TODO similar to L554
    #for cat_name in yt.get_cat_lables(histo,axis_name):
    #    hist.plotratio(
    #        num = histo.integrate(axis_name,cat_name),
    #        denom = histo.integrate(axis_name,cat_ref),
    #        ax = rax,
    #        unc = 'num',
    #        error_opts= {'linestyle': 'none','marker': '.', 'markersize': 10, 'elinewidth': 0},
    #        clear = False,
    #    )

    # Plot the syst error (if we have the necessary up/down variations)
    plot_syst_err = False
    if (err_p is not None) and (err_m is not None) and (err_ratio_p is not None) and (err_ratio_m is not None): plot_syst_err = True
    if plot_syst_err:
        bin_edges_arr = histo.axes[var].edges
        #err_p = np.append(err_p,0) # Work around off by one error
        #err_m = np.append(err_m,0) # Work around off by one error
        #err_ratio_p = np.append(err_ratio_p,0) # Work around off by one error
        #err_ratio_m = np.append(err_ratio_m,0) # Work around off by one error
        ax.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')
        ax.set_ylim(0.0,1.2*max(err_p))
        rax.fill_between(bin_edges_arr,err_ratio_m,err_ratio_p,step='post', facecolor='none', edgecolor='gray', label='Syst err', hatch='////')

    # Style
    ax.set_xlabel('')
    rax.axhline(1.0,linestyle="-",color="k",linewidth=1)
    rax.set_ylabel('Ratio')
    rax.autoscale(axis='y')

    return fig



###################### Wrapper function for signal plots with systematics ######################
# Wrapper function to loop over all categories and make plots for all variables
# Right now this function will only plot the signal samples
# By default, will make plots that show all systematics in the pkl file
def make_signal_systematic_plots(dict_of_hists,year,save_dir_path,variables=None):

    # If selecting a year, append that year to the wight list
    sig_wl = ["private"]
    if year is None: pass
    elif year == "2017":
        sig_wl.append("UL17")
    elif year == "2018":
        sig_wl.append("UL18")
    elif year == "2016":
        sig_wl.append("UL16") # NOTE: Right now this will plot both UL16 an UL16APV
    elif year == "2022":
        sig_wl.append("2022")
    elif year == "2022EE":
        sig_wl.append("2022EE")
    elif year == "2023":
        sig_wl.append("2023")
    elif year == "2023BPix":
        sig_wl.append("2023BPix")
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
    if len(sig_sample_lst) == 0: raise Exception("Error: No signal samples to plot.")
    samples_to_rm_from_sig_hist = []
    for sample_name in all_samples:
        if sample_name not in sig_sample_lst:
            samples_to_rm_from_sig_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nSig samples:",sig_sample_lst)
    print("\nAll systematics:",yt.get_cat_lables(dict_of_hists,"systematic",h_name=yt.get_hist_list(dict_of_hists)[0]))

    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    variables_to_plot = _resolve_requested_variables(
        dict_of_hists, variables, "make_signal_systematic_plots"
    )

    for idx, var_name in enumerate(variables_to_plot):
        if 'sumw2' in var_name: continue
        if _is_sparse_2d_hist(dict_of_hists[var_name]):
            continue
        if yt.is_split_by_lepflav(dict_of_hists): raise Exception("Not set up to plot lep flav for SR, though could probably do it without too much work")
        if (var_name in skip_lst): continue
        channel_transformations = []
        if (var_name == "njets"):
            # We do not keep track of jets in the sparse axis for the njets hists
            channel_transformations.append("njets")
            sr_cat_dict = get_dict_with_stripped_bin_names(SR_CHAN_DICT,"njets")
        else:
            sr_cat_dict = SR_CHAN_DICT
        print("\nVar name:", var_name)

        # Extract the signal hists
        hist_sig = dict_of_hists[var_name].remove("process", samples_to_rm_from_sig_hist)

        # If we only want to look at a subset of the systematics (Probably should be an option? For now, just uncomment if you want to use it)
        syst_subset_dict = {
            "nominal":["nominal"],
            "renormfactUp":["renormfactUp"],"renormfactDown":["renormfactDown"],
        }
        #hist_sig  = group_bins(hist_sig,syst_subset_dict,"systematic",drop_unspecified=True)

        # Make plots for each process
        for proc_name in sig_sample_lst:

            # Make a sub dir for this category
            save_dir_path_tmp = os.path.join(save_dir_path,proc_name)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            # Group categories
            hist_sig_grouped = group_bins(hist_sig,sr_cat_dict,"channel",drop_unspecified=True)

            # Make the plots
            for grouped_hist_cat in yt.get_cat_lables(hist_sig_grouped,axis="channel",h_name=var_name):

                if grouped_hist_cat in sr_cat_dict:
                    validate_channel_group(
                        hist_sig,
                        sr_cat_dict[grouped_hist_cat],
                        channel_transformations,
                        region="SR",
                        subgroup=grouped_hist_cat,
                        variable=var_name,
                    )

                # Integrate
                hist_sig_grouped_tmp = copy.deepcopy(hist_sig_grouped)
                hist_sig_grouped_tmp = yt.integrate_out_appl(hist_sig_grouped_tmp,grouped_hist_cat)
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("process",proc_name[{'process': sum}])
                hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("channel",grouped_hist_cat[{'channel': sum}])

                # Reweight (Probably should be an option? For now, just uncomment if you want to use it)
                #hist_sig_grouped_tmp.set_wilson_coefficients(**WCPT_EXAMPLE)

                # Make plots
                fig = make_single_fig_with_ratio(hist_sig_grouped_tmp,"systematic","nominal",var=var_name)
                title = proc_name+"_"+grouped_hist_cat+"_"+var_name
                fig.savefig(
                    os.path.join(save_dir_path_tmp, title),
                    bbox_inches="tight",
                    pad_inches=0.05,
                )

            # Make an index.html file if saving to web area
            if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


###################### Wrapper function for simple plots ######################
# Wrapper function to loop over categories and make plots for all variables
def make_simple_plots(dict_of_hists,year,save_dir_path,variables=None):

    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name="njets")

    variables_to_plot = _resolve_requested_variables(
        dict_of_hists, variables, "make_simple_plots"
    )

    for idx,var_name in enumerate(variables_to_plot):
        if 'sumw2' in var_name: continue
        if _is_sparse_2d_hist(dict_of_hists[var_name]):
            continue
        #if var_name == "njets": continue
        #if "parton" in var_name: save_tag = "partonFlavour"
        #if "hadron" in var_name: save_tag = "hadronFlavour"
        #if "hadron" not in var_name: continue
        #if var_name != "j0hadronFlavour": continue
        if var_name != "j0partonFlavour": continue

        histo_orig = dict_of_hists[var_name]

        # Loop over channels
        channels_lst = yt.get_cat_lables(dict_of_hists[var_name],"channel")
        for chan_name in channels_lst:

            histo = copy.deepcopy(histo_orig)

            histo = yt.integrate_out_appl(histo,chan_name)
            histo = histo.integrate("systematic","nominal")
            histo = histo.integrate("channel",chan_name)

            print("\n",chan_name)
            print(histo.eval({}))
            summed_histo = histo[{"process": sum}]
            print("sum:",sum(summed_histo.eval({})[()]))
            continue

            # Make a sub dir for this category
            save_tag = "placeholder" # Flake8 pointed out that save_tag is not defined, should figure out why at some point if this function is ever used again
            save_dir_path_tmp = os.path.join(save_dir_path,save_tag)
            if not os.path.exists(save_dir_path_tmp):
                os.mkdir(save_dir_path_tmp)

            fig = make_single_fig(histo, unit_norm_bool=False)
            title = chan_name + "_" + var_name
            fig.savefig(
                os.path.join(save_dir_path_tmp, title),
                bbox_inches="tight",
                pad_inches=0.05,
            )

            # Make an index.html file if saving to web area
            if "www" in save_dir_path: make_html(save_dir_path_tmp)


###################### Wrapper function for signal-focused plots ######################
# Wrapper function to loop over selected categories and make plots for all variables
# Right now this function will only plot the signal samples
# By default, will make two sets of plots: One with process overlay, one with channel overlay
def make_signal_plots(
    dict_of_hists,
    year,
    unit_norm_bool,
    save_dir_path,
    split_by_chan=True,
    split_by_proc=True,
    variables=None,
):

    # If selecting a year, append that year to the wight list
    sig_wl = ["private"]
    if year is None: pass
    elif year == "2017":
        sig_wl.append("UL17")
    elif year == "2018":
        sig_wl.append("UL18")
    elif year == "2016":
        sig_wl.append("UL16") # NOTE: Right now this will plot both UL16 an UL16APV
    elif year == "2022":
        sig_wl.append("2022")
    elif year == "2022EE":
        sig_wl.append("2022EE")
    elif year == "2023":
        sig_wl.append("2023")
    elif year == "2023BPix":
        sig_wl.append("2023BPix")
    else: raise Exception

    # Get the list of samples to actually plot (finding sample list from first hist in the dict)
    all_samples = yt.get_cat_lables(dict_of_hists,"process",h_name=yt.get_hist_list(dict_of_hists)[0])
    sig_sample_lst = utils.filter_lst_of_strs(all_samples,substr_whitelist=sig_wl)
    if len(sig_sample_lst) == 0: raise Exception("Error: No signal samples to plot.")
    samples_to_rm_from_sig_hist = []
    for sample_name in all_samples:
        if sample_name not in sig_sample_lst:
            samples_to_rm_from_sig_hist.append(sample_name)
    print("\nAll samples:",all_samples)
    print("\nSig samples:",sig_sample_lst)


    # Loop over hists and make plots
    skip_lst = [] # Skip this hist
    variables_to_plot = _resolve_requested_variables(
        dict_of_hists, variables, "make_signal_plots"
    )

    for idx,var_name in enumerate(variables_to_plot):
        #if yt.is_split_by_lepflav(dict_of_hists): raise Exception("Not set up to plot lep flav for SR, though could probably do it without too much work")
        if 'sumw2' in var_name: continue
        if _is_sparse_2d_hist(dict_of_hists[var_name]):
            continue
        if (var_name in skip_lst): continue
        if (var_name == "njets"):
            continue
            # We do not keep track of jets in the sparse axis for the njets hists
            sr_cat_dict = get_dict_with_stripped_bin_names(SR_CHAN_DICT,"njets")
        else:
            sr_cat_dict = SR_CHAN_DICT
        channel_transformations = []
        print("\nVar name:",var_name)
        print("sr_cat_dict:",sr_cat_dict)

        # Extract the signal hists, and integrate over systematic axis
        hist_sig = dict_of_hists[var_name].remove("process", samples_to_rm_from_sig_hist)
        hist_sig = hist_sig.integrate("systematic","nominal")

        # Make plots for each SR category
        if split_by_chan:
            for hist_cat in SR_CHAN_DICT.keys():
                if ((var_name == "ptz") and ("3l" not in hist_cat)): continue

                if hist_cat in sr_cat_dict:
                    validate_channel_group(
                        hist_sig,
                        sr_cat_dict[hist_cat],
                        channel_transformations,
                        region="SR",
                        subgroup=hist_cat,
                        variable=var_name,
                    )

                # Make a sub dir for this category
                save_dir_path_tmp = os.path.join(save_dir_path,hist_cat)
                if not os.path.exists(save_dir_path_tmp):
                    os.mkdir(save_dir_path_tmp)

                # Integrate to get the SR category we want to plot
                hist_sig_integrated_ch = yt.integrate_out_appl(hist_sig,hist_cat)
                # Skip missing channels (histEFT throws an exception)
                channels = [chan for chan in sr_cat_dict[hist_cat] if chan in hist_sig_integrated_ch.axes['channel']]
                if not channels: # Skip empty channels
                    continue
                hist_sig_integrated_ch = hist_sig_integrated_ch.integrate("channel",channels)[{'channel': sum}]
                hist_sig_integrated_ch = hist_sig_integrated_ch.integrate("process")

                # Make the plots
                if not hist_sig_integrated_ch.eval({}):
                    print("Warning: empty mc histo, continuing")
                    continue
                fig = make_single_fig(hist_sig_integrated_ch,unit_norm_bool,bins=axes_info[var_name]['variable'])
                title = hist_cat+"_"+var_name
                if unit_norm_bool: title = title + "_unitnorm"
                fig.savefig(
                    os.path.join(save_dir_path_tmp, title),
                    bbox_inches="tight",
                    pad_inches=0.05,
                )

                # Make an index.html file if saving to web area
                if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)


        # Make plots for each process
        if split_by_proc:
            for proc_name in sig_sample_lst:

                # Make a sub dir for this category
                save_dir_path_tmp = os.path.join(save_dir_path,proc_name)
                if not os.path.exists(save_dir_path_tmp):
                    os.mkdir(save_dir_path_tmp)

                # Group categories
                # Using new grouping approach in plot functions
                #hist_sig_grouped = group_bins(hist_sig,sr_cat_dict,"channel",drop_unspecified=True)
                hist_sig_grouped = hist_sig

                # Make the plots
                # Using new grouping approach in plot functions
                #for grouped_hist_cat in yt.get_cat_lables(hist_sig_grouped,axis="channel",h_name=var_name):
                for grouped_hist_cat in sr_cat_dict:
                    if grouped_hist_cat in sr_cat_dict:
                        validate_channel_group(
                            hist_sig_grouped,
                            sr_cat_dict[grouped_hist_cat],
                            channel_transformations,
                            region="SR",
                            subgroup=grouped_hist_cat,
                            variable=var_name,
                        )
                    if not any(cat in hist_sig_grouped.axes['channel'] for cat in sr_cat_dict[grouped_hist_cat]):
                        continue

                    # Integrate
                    hist_sig_grouped_tmp = copy.deepcopy(hist_sig_grouped)
                    hist_sig_grouped_tmp = yt.integrate_out_appl(hist_sig_grouped_tmp,grouped_hist_cat)
                    if proc_name not in list(hist_sig_grouped_tmp.axes["process"]):
                        print(f"Warning: mc histo missing {proc_name}, continuing")
                        continue
                    hist_sig_grouped_tmp = hist_sig_grouped_tmp.integrate("process",proc_name)
                    if not hist_sig_grouped_tmp.eval({}):
                        print("Warning: empty mc histo, continuing")
                        continue

                    # Make plots
                    fig = make_single_fig(hist_sig_grouped_tmp[{'channel': sr_cat_dict[grouped_hist_cat]}][{'channel': sum}],unit_norm_bool,bins=axes_info[var_name]['variable'])
                    title = proc_name+"_"+grouped_hist_cat+"_"+var_name
                    if unit_norm_bool: title = title + "_unitnorm"
                    fig.savefig(
                        os.path.join(save_dir_path_tmp, title),
                        bbox_inches="tight",
                        pad_inches=0.05,
                    )

                # Make an index.html file if saving to web area
                if "www" in save_dir_path_tmp: make_html(save_dir_path_tmp)



###################### Region plotting entry point ######################
# Execute the region-agnostic plotting pipeline for the requested region name.
# The caller provides the histogram dictionary that includes data and MC.
def run_plots_for_region(
    region_name,
    dict_of_hists,
    year,
    save_dir_path,
    *,
    skip_syst_errs=False,
    unit_norm_bool=False,
    variables=None,
    unblind=None,
):
    region_ctx = build_region_context(
        region_name,
        dict_of_hists,
        year,
        unblind=unblind,
    )
    produce_region_plots(
        region_ctx,
        save_dir_path,
        variables,
        skip_syst_errs,
        unit_norm_bool,
        unblind=unblind,
    )

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="plots", help = "A name for the output directory")
    parser.add_argument("-t", "--include-timestamp-tag", action="store_true", help = "Append the timestamp to the out dir name")
    parser.add_argument("-y", "--year", default=None, help = "The year of the sample")
    parser.add_argument("-u", "--unit-norm", action="store_true", help = "Unit normalize the plots")
    parser.add_argument("-s", "--skip-syst", default=False, action="store_true", help = "Skip systematic error bands in plots")
    parser.add_argument(
        "--unblind",
        dest="unblind",
        action="store_true",
        help="Force plots to include data yields even in normally blinded regions.",
    )
    parser.add_argument(
        "--blind",
        dest="unblind",
        action="store_false",
        help="Force plots to hide data yields even in normally unblinded regions.",
    )
    parser.set_defaults(unblind=None)
    region_group = parser.add_mutually_exclusive_group()
    region_group.add_argument(
        "--cr",
        dest="region_override",
        action="store_const",
        const="CR",
        help="Force control-region plotting, overriding filename-based detection.",
    )
    region_group.add_argument(
        "--sr",
        dest="region_override",
        action="store_const",
        const="SR",
        help="Force signal-region plotting, overriding filename-based detection.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional list of histogram variables to plot",
    )
    args = parser.parse_args()

    def _detect_region_from_path(path):
        if not path:
            return None, False
        filename = os.path.basename(path)
        uppercase = filename.upper()
        matched_regions = []
        for region in ("CR", "SR"):
            # Accept filenames where the region token is directly followed by
            # qualifiers such as a year (e.g. "SR2018") or run tag (e.g. "CRRun2").
            # We only guard against being embedded within a longer alphanumeric
            # token by ensuring the preceding character is not an uppercase
            # letter or digit.
            pattern = re.compile(rf"(?<![A-Z0-9]){region}")
            if pattern.search(uppercase):
                matched_regions.append(region)
        if len(matched_regions) == 1:
            return matched_regions[0], False
        if len(matched_regions) > 1:
            return None, True
        return None, False

    detected_region, ambiguous_region = _detect_region_from_path(args.pkl_file_path)
    resolved_region = args.region_override or detected_region or "CR"
    if ambiguous_region and not args.region_override:
        print(
            "Warning: Detected both 'CR' and 'SR' tokens in the input filename. "
            "Defaulting to 'CR'; please pass --cr or --sr to specify explicitly."
        )

    if args.unblind is None:
        resolved_unblind = resolved_region == "CR"
        blinding_source = f"default for {resolved_region} region"
    elif args.unblind:
        resolved_unblind = True
        blinding_source = "command-line --unblind override"
    else:
        resolved_unblind = False
        blinding_source = "command-line --blind override"

    print(f"Resolved plotting region: {resolved_region}")
    print(
        "Resolved blinding mode: {} ({})".format(
            "unblinded" if resolved_unblind else "blinded", blinding_source
        )
    )

    normalized_variables = []
    if args.variables is not None:
        seen_variables = set()
        for value in args.variables:
            if value is None:
                continue
            cleaned = value.strip()
            if not cleaned or cleaned in seen_variables:
                continue
            seen_variables.add(cleaned)
            normalized_variables.append(cleaned)
    selected_variables = normalized_variables if normalized_variables else None

    # Whether or not to unit norm the plots
    unit_norm_bool = args.unit_norm

    # Make a tmp output directory in curren dir a different dir is not specified
    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    save_dir_path = args.output_path
    outdir_name = args.output_name
    if args.include_timestamp_tag:
        outdir_name = outdir_name + "_" + timestamp_tag
    save_dir_path = os.path.join(save_dir_path,outdir_name)
    os.mkdir(save_dir_path)

    # Get the histograms
    hin_dict = utils.get_hist_from_pkl(args.pkl_file_path,allow_empty=False)
    # Print info about histos
    #yt.print_hist_info(args.pkl_file_path,"nbtagsl")
    #exit()

    print("\nMaking plots for year:",args.year)
    print("Output dir:",save_dir_path)
    print("Variables to plot:", selected_variables if selected_variables else "All")
    print("\n\n")

    # Make the plots
    run_plots_for_region(
        resolved_region,
        hin_dict,
        args.year,
        save_dir_path,
        skip_syst_errs=args.skip_syst,
        unit_norm_bool=unit_norm_bool,
        variables=selected_variables,
        unblind=resolved_unblind,
    )
    #make_signal_plots(hin_dict,args.year,unit_norm_bool,save_dir_path)
    #make_signal_systematic_plots(hin_dict,args.year,save_dir_path)
    #make_simple_plots(hin_dict,args.year,save_dir_path)

if __name__ == "__main__":
    main()
