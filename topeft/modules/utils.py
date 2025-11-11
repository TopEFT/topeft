"""General utility helpers for :mod:`topeft`.

This module also exposes cached wrappers for the rate systematic helpers so
callers do not re-read ``params/rate_systs.json`` on every invocation.
"""

import os
import re
import json
import gzip
import pickle
import time
from functools import lru_cache
from types import MappingProxyType

import cloudpickle
import uproot


pjoin = os.path.join


############## Floats manipulations and tools ##############

# Get percent difference
def get_pdiff(a,b,in_percent=False):
    #p = (float(a)-float(b))/((float(a)+float(b))/2)
    if ((a is None) or (b is None)):
        p = None
    elif b == 0:
        p = None
    else:
        p = (float(a)-float(b))/float(b)
        if in_percent:
            p = p*100.0
    return p

############## Strings manipulations and tools ##############


def canonicalize_process_name(process_name):
    """Return *process_name* with only the leading alphabetic token lowercased.

    Examples:
        ``NonPromptUL16`` becomes ``nonpromptUL16`` while ``Flips2023BPix``
        becomes ``flips2023BPix``.

    Args:
        process_name (str): The process identifier to canonicalize.

    Returns:
        str: The canonicalized process name. Strings without a leading
        alphabetic token are returned unchanged.
    """

    match = re.match(r"([A-Za-z]+)(.*)", process_name)
    if not match:
        return process_name
    prefix, remainder = match.groups()
    return prefix.lower() + remainder


# Match strings using one or more regular expressions
def regex_match(lst,regex_lst):
    # NOTE: For the regex_lst patterns, we use the raw string to generate the regular expression.
    #       This means that any regex special characters in the regex_lst should be properly
    #       escaped prior to calling this function.
    # NOTE: The input list is assumed to be a list of str objects and nothing else!
    if len(regex_lst) == 0: return lst[:]
    matches = []
    to_match = [re.compile(r"{}".format(x)) for x in regex_lst]
    for s in lst:
        for pat in to_match:
            m = pat.search(s)
            if m is not None:
                matches.append(s)
                break
    return matches

# Get a subset of the elements from a list of strings given a whitelist and/or blacklist of substrings
def filter_lst_of_strs(in_lst,substr_whitelist=[],substr_blacklist=[]):

    # Check all elements are strings
    if not (all(isinstance(x,str) for x in in_lst) and all(isinstance(x,str) for x in substr_whitelist) and all(isinstance(x,str) for x in substr_blacklist)):
        raise Exception("Error: This function only filters lists of strings, one of the elements in one of the input lists is not a str.")
    for elem in substr_whitelist:
        if elem in substr_blacklist:
            raise Exception(f"Error: Cannot whitelist and blacklist the same element (\"{elem}\").")

    # Append to the return list
    out_lst = []
    for element in in_lst:
        blacklisted = False
        whitelisted = True
        for substr in substr_blacklist:
            if substr in element:
                # If any of the substrings are in the element, blacklist it
                blacklisted = True
        for substr in substr_whitelist:
            if substr not in element:
                # If any of the substrings are NOT in the element, do not whitelist it
                whitelisted = False
        if whitelisted and not blacklisted:
            out_lst.append(element)

    return out_lst


############## Dirs and root files manipulations and tools ##############

def get_files(top_dir,**kwargs):
    '''
        Description:
            Walks through an entire directory structure searching for files. Returns a list of
            matching files with absolute path included.

            Can optionally be given list of regular
            expressions to skip certain directories/files or only match certain types of files
    '''
    ignore_dirs  = kwargs.pop('ignore_dirs',[])
    match_files  = kwargs.pop('match_files',[])
    ignore_files = kwargs.pop('ignore_files',[])
    recursive    = kwargs.pop('recursive',False)
    verbose      = kwargs.pop('verbose',False)
    found = []
    if verbose:
        print(f"Searching in {top_dir}")
        print(f"\tRecurse: {recursive}")
        print(f"\tignore_dirs: {ignore_dirs}")
        print(f"\tmatch_files: {match_files}")
        print(f"\tignore_files: {ignore_files}")
    for root, dirs, files in os.walk(top_dir):
        if recursive:
            if ignore_dirs:
                dir_matches = regex_match(dirs,regex_lst=ignore_dirs)
                for m in dir_matches:
                    if verbose:
                        print(f"\tSkipping directory: {m}")
                    dirs.remove(m)
        else:
            dirs.clear()
        files = regex_match(files,match_files)
        if ignore_files:
            file_matches = regex_match(files,regex_lst=ignore_files)
            for m in file_matches:
                if verbose:
                    print(f"\tSkipping file: {m}")
                files.remove(m)     # Removes 'm' from the file list, not the actual file on disk
        for f in files:
            fpath = os.path.join(root,f)
            found.append(fpath)
    return found


# Cached rate systematic helpers ################################################

from topeft.modules.get_rate_systs import (
    get_correlation_tag as _get_correlation_tag,
    get_jet_dependent_syst_dict as _get_jet_dependent_syst_dict,
    get_syst as _get_syst,
    get_syst_lst as _get_syst_lst,
)


@lru_cache(maxsize=None)
def cached_get_syst(syst_name, proc_name=None, literal=False):
    """Cached variant of :func:`topeft.modules.get_rate_systs.get_syst`.

    Returns:
        str | tuple[float, float]: Either the literal systematic string or an
        immutable pair of down/up scaling factors, depending on *literal*.
    """

    result = _get_syst(syst_name, proc_name=proc_name, literal=literal)
    if literal:
        return result
    return tuple(result)


@lru_cache(maxsize=None)
def cached_get_syst_lst():
    """Cached variant of :func:`topeft.modules.get_rate_systs.get_syst_lst`."""

    return tuple(_get_syst_lst())


@lru_cache(maxsize=None)
def cached_get_correlation_tag(syst_type, proc_name):
    """Cached variant of :func:`topeft.modules.get_rate_systs.get_correlation_tag`."""

    return _get_correlation_tag(syst_type, proc_name)


@lru_cache(maxsize=None)
def cached_get_jet_dependent_syst_dict(process="Diboson"):
    """Cached variant of :func:`topeft.modules.get_rate_systs.get_jet_dependent_syst_dict`."""

    source = _get_jet_dependent_syst_dict(process)
    return MappingProxyType(dict(source))


_existing_all = globals().get("__all__")
if isinstance(_existing_all, (list, tuple)):
    _base_exports = list(_existing_all)
else:
    _base_exports = [name for name in globals() if not name.startswith("_")]

__all__ = [
    *_base_exports,
    "cached_get_syst",
    "cached_get_syst_lst",
    "cached_get_correlation_tag",
    "cached_get_jet_dependent_syst_dict",
]

# Extracts event information from a root file
'''
def get_info(fname, tree_name="Events"):
    # The info we want to get
    raw_events = 0  # The raw number of entries as reported by TTree.num_entries
    gen_events = 0  # Number of gen events according to 'genEventCount' or set to raw_events if not found
    sow_events = 0  # Sum of weights
    sow_lhe_wgts = None # Sum of LHE weights
    is_data = False

    print(f"Opening with uproot: {fname}")
    try:
        # This both opens and ensures f.close() on exit
        with uproot.open(fname) as f:
            tree = f[tree_name]  # KeyInFileError if missing
            is_data = "genWeight" not in tree
            raw_events = int(tree.num_entries)

            if is_data:
                # Data doesn't have gen or weighted events!
                gen_events = raw_events
                sow_events = raw_events
            else:
                gen_events = raw_events
                sow_events = sum(tree["genWeight"])

                if "Runs" in f:
                    # Instead get event from the "Runs" tree
                    runs = f["Runs"]
                    gen_key = "genEventCount" if "genEventCount" in runs else "genEventCount_"
                    sow_key = "genEventSumw"   if "genEventSumw"   in runs else "genEventSumw_"
                    try:
                        gen_events = sum(runs[gen_key].array())
                        sow_events = sum(runs[sow_key].array())
                        lhe          = runs["LHEScaleSumw"].array()
                        sow_lhe_wgts = sum(runs[sow_key].array() * lhe)
                    except KeyError as e:
                        print(f"\tMissing branch in Runs tree: {e}, using default sums")
    
    except (OSError, uproot.KeyInFileError) as err:
        # File‐not‐found, unreadable, or missing tree
        print(f"\tCouldn’t process {fname!r}: {err}")
    
    finally:
        # Ensure that we return the default values if the file was not readable
        if raw_events == 0:
            gen_events = 0
            sow_events = 0
            sow_lhe_wgts = None
            is_data = False
        
        # Return the info we found
        print(f"\tFound {raw_events} raw events, {gen_events} gen events, {sow_events} sum of weights, {sow_lhe_wgts} sum of LHE weights, is_data={is_data}")
        return [raw_events, gen_events, sow_events, sow_lhe_wgts, is_data]
'''

def get_info(fname, tree_name="Events", max_retries=10, retry_delay=30):
    raw_events = 0
    gen_events = 0
    sow_events = 0
    sow_lhe_wgts = None
    is_data = False

    print(f"Opening with uproot: {fname}")
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            # This both opens and ensures f.close() on exit
            with uproot.open(fname) as f:
                tree = f[tree_name]
                is_data = "genWeight" not in tree
                raw_events = int(tree.num_entries)

                if is_data:
                    gen_events = raw_events
                    sow_events = raw_events
                else:
                    gen_events = raw_events
                    sow_events = sum(tree["genWeight"])
                    if "Runs" in f:
                        runs = f["Runs"]
                        gen_key = "genEventCount" if "genEventCount" in runs else "genEventCount_"
                        sow_key = "genEventSumw"   if "genEventSumw"   in runs else "genEventSumw_"
                        try:
                            gen_events = sum(runs[gen_key].array())
                            sow_events = sum(runs[sow_key].array())
                            lhe = runs["LHEScaleSumw"].array()
                            sow_lhe_wgts = sum(runs[sow_key].array() * lhe)
                        except KeyError as e:
                            print(f"\tMissing branch in Runs tree: {e}, using default sums")
            # success!
            print(f"\tRemote reading of {fname!r} succeeded on attempt {attempt}/{max_retries}.")
            break

        except Exception as err:
            msg = str(err).lower()
            if "operation expired" in msg or "no servers are available to read the file" in msg or "unable to read" in msg or "couldn't process":
                print(f"\tNetwork issue on attempt {attempt}/{max_retries} ({err}), retrying in {retry_delay}s …")
                time.sleep(retry_delay)
                continue
            else:
                print(f"\tCouldn’t process {fname!r}: {err}")
                break
    else:
        # This block runs if we never 'break'—i.e. all retries exhausted
        print(f"\tGiving up on file {fname!r} after {max_retries} retries due to repeated network errors.")

    # Ensure defaults if nothing was read
    if raw_events == 0:
        gen_events = 0
        sow_events = 0
        sow_lhe_wgts = None
        is_data = False

    print(f"\tFound {raw_events} raw events, {gen_events} gen events, "
          f"{sow_events} sum of weights, {sow_lhe_wgts} sum of LHE weights, is_data={is_data}")
    return [raw_events, gen_events, sow_events, sow_lhe_wgts, is_data]


# Get the list of WC names from an EFT sample naod root file
def get_list_of_wc_names(fname):
    ''' Retruns a list of the WC names from WCnames, (retruns [] if not an EFT sample) '''
    wc_names_lst = []
    tree = uproot.open(f'{fname}:Events')
    if 'WCnames' not in tree.keys():
        wc_names_lst = []
    else:
        wc_info = tree['WCnames'].array(entry_stop=1)[0]
        for idx,i in enumerate(wc_info):
            h = hex(i)[2:]                                 # Get rid of the first two characters
            wc_fragment = bytes.fromhex(h).decode('utf-8') # From: https://stackoverflow.com/questions/3283984/decode-hex-string-in-python-3
            # The WC names that are longer than 4 letters are too long to be encoded in a 64-bit integer:
            #   - They're instead stored in two subsequent entries in the list
            #   - This means that the decoded names in wc_info go like this [... 'ctlT' , '-i' ...]
            #   - The leading '-' indicates the given fragment is the trailing end of the previous WC name
            #   - The following logic is supposed to put those fragments back together into the WC name
            if not wc_fragment.startswith("-"):
                wc_names_lst.append(wc_fragment)
            else:
                leftover = wc_fragment[1:]                    # This leftover part of the WC goes with the previous one (but get rid of leading '-')
                wc_names_lst[-1] = wc_names_lst[-1]+leftover  # So append this trailing fragment to the leading framgenet to reconstruct the WC name
    return wc_names_lst


# Moves a list of files to the specified target directory
def move_files(files,target):
    width = len(max(files,key=len))
    for src in files:
        dst = os.path.join(target,src)
        os.rename(src,dst)


# Removes files from tdir which match any of the regex in targets list
def clean_dir(tdir,targets,dry_run=False):
    fnames = regex_match(get_files(tdir),targets)
    if len(fnames) == 0: return
    print(f"Removing files from: {tdir}")
    print(f"\tTargets: {targets}")
    for fn in fnames:
        fpath = os.path.join(tdir,fn)
        if not dry_run:
            print(f"\tRemoving {fn}")
            os.remove(fpath)
        else:
            print(f"\tRemoving {fpath}")


############## Jason and config manipulations and tools ##############

# Read from a sample json file
def load_sample_json_file(fpath):
    if not os.path.exists(fpath):
        raise RuntimeError(f"fpath '{fpath}' does not exist!")
    with open(fpath) as f:
        jsn = json.load(f)
    jsn['redirector'] = None
    # Cleanup any spurious double slashes
    for i,fn in enumerate(jsn['files']):
        fn = fn.replace("//","/")
        jsn['files'][i] = fn
    # Make sure that the json was unpacked correctly
    jsn['xsec']          = float(jsn['xsec'])
    jsn['nEvents']       = int(jsn['nEvents'])
    jsn['nGenEvents']    = int(jsn['nGenEvents'])
    jsn['nSumOfWeights'] = float(jsn['nSumOfWeights'])
    return jsn


# Generate/Update a dictionary for storing info from a cfg file
def update_cfg(jsn,name,**kwargs):
    cfg = kwargs.pop('cfg',{})
    max_files = kwargs.pop('max_files',0)
    cfg[name] = {}
    cfg[name].update(jsn)
    if max_files:
        # Only keep the first "max_files"
        del cfg[name]['files'][max_files:]
    # Inject/Modify info related to the json sample
    for k,v in kwargs.items():
        cfg[name][k] = v
    return cfg


# Read from a cfg file
def read_cfg_file(fpath,cfg={},max_files=0):
    cfg_dir,fname = os.path.split(fpath)
    if not cfg_dir:
        raise RuntimeError(f"No cfg directory in {fpath}")
    if not os.path.exists(cfg_dir):
        raise RuntimeError(f"{cfg_dir} does not exist!")
    xrd_src = None
    with open(fpath) as f:
        print(' >> Reading json from cfg file...')
        for l in f:
            l = l.strip().split("#")[0]     # Chop off anything after a comment
            if not len(l): continue         # Ignore fully commented lines
            if l.startswith("root:") or l.startswith("http:") or l.startswith("https:"):
                # Note: This implicitly assumes that a redirector line will appear before any json
                #   paths in the cfg file
                xrd_src = l
            elif l.startswith("file://"):
                xrd_src = l.replace("file://","")
            else:
                sample = os.path.basename(l)
                sample = sample.replace(".json","")
                full_path = pjoin(cfg_dir,l)
                jsn = load_sample_json_file(full_path)
                cfg = update_cfg(jsn,sample,cfg=cfg,max_files=max_files,redirector=xrd_src)
    return cfg


############## Pickle manipulations and tools ##############

# Save to a pkl file
def dump_to_pkl(out_name,out_file):
    if not out_name.endswith(".pkl.gz"):
        out_name = out_name + ".pkl.gz"
    print(f"\nSaving output to {out_name}...")
    with gzip.open(out_name, "wb") as fout:
        cloudpickle.dump(out_file, fout)
    print("Done.\n")


def get_hist_dict_non_empty(h):
    print(h.keys())
    return {k: v for k, v in h.items()}# if not v.empty()}


# Get the dictionary of hists from the pkl file (e.g. that a processor outputs)
def get_hist_from_pkl(path_to_pkl, allow_empty=True):
    h = pickle.load(gzip.open(path_to_pkl))
    if not allow_empty:
        h = get_hist_dict_non_empty(h)
    return h


############## Dictionary manipulations and tools ##############

# Takes two dictionaries, returns the list of lists [common keys, keys unique to d1, keys unique to d2]
def get_common_keys(dict1,dict2):

    common_lst = []
    unique_1_lst = []
    unique_2_lst = []

    # Find common keys, and keys unique to d1
    for k1 in dict1.keys():
        if k1 in dict2.keys():
            common_lst.append(k1)
        else:
            unique_1_lst.append(k1)

    # Find keys unique to d2
    for k2 in dict2.keys():
        if k2 not in common_lst:
            unique_2_lst.append(k2)

    return [common_lst,unique_1_lst,unique_2_lst]


# Check if the contents of two dictionaries of lists agree
# Assumes structure d = {k1: [i1,i2,...], ...}
def dict_comp(in_dict1,in_dict2,strict=False):

    def all_d1_in_d2(d1,d2):
        agree = True
        for k1,v1 in d1.items():
            if k1 not in d2:
                agree = False
                break
            for i1 in v1:
                if i1 not in d2[k1]:
                    agree = False
                    break
        return agree

    dicts_match = all_d1_in_d2(in_dict1,in_dict2) and all_d1_in_d2(in_dict2,in_dict1)
    print_str = f"The two dictionaries do not agree.\n\tDict 1:{in_dict1}\n\tDict 2:{in_dict2}"

    if not dicts_match:
        if strict: raise Exception("Error: "+print_str)
        else: print("Warning: "+print_str)

    return dicts_match


# Takes as input a dictionary {"k": {"subk":[val,err]}} and returns {"k":{"subk":val}}
def strip_errs(in_dict):
    out_dict = {}
    for k in in_dict.keys():
        out_dict[k] = {}
        for subk in in_dict[k]:
            out_dict[k][subk] = in_dict[k][subk][0]
    return out_dict


# Takes as input a dictionary {"k":{"subk":val}} and returns {"k": {"subk":[val,None]}}
def put_none_errs(in_dict):
    out_dict = {}
    for k in in_dict.keys():
        out_dict[k] = {}
        for subk in in_dict[k]:
            out_dict[k][subk] = [in_dict[k][subk],None]
    return out_dict


# Takes yield dicts and prints it
# Note:
#   - This function also now optionally takes a tolerance value
#   - Checks if the differences are larger than that value
#   - Returns False if any of the values are too large
#   - Should a different function handle this stuff?
def print_yld_dicts(ylds_dict,tag,show_errs=False,tolerance=None):
    ret = True
    print(f"\n--- {tag} ---\n")
    for proc in ylds_dict.keys():
        print(proc)
        for cat in ylds_dict[proc].keys():
            print(f"    {cat}")
            val , err = ylds_dict[proc][cat]

            # We don't want to check if the val is small
            if tolerance is None:
                if show_errs:
                    #print(f"\t{val} +- {err}")
                    print(f"\t{val} +- {err} -> {err/val}")
                else:
                    print(f"\t{val}")

            # We want to check if the val is small
            else:
                if (val is None) or (abs(val) < abs(tolerance)):
                    print(f"\t{val}")
                else:
                    print(f"\t{val} -> NOTE: This is larger than tolerance ({tolerance})!")
                    ret = False
    return ret


# Wrapper around get_diff_between_dicts for nested dicts
# Returns a dictionary in the same format (currently does not propagate errors for percent diff, just returns None)
#   dict = {
#       k : {
#           subk : [val,var]
#       }
#   }
# Note: This function makes use of utils.get_common_keys and utils.get_pdiff
def get_diff_between_nested_dicts(dict1,dict2,difftype,inpercent=False):

    # Get list of keys common to both dictionaries
    common_keys, d1_keys, d2_keys = get_common_keys(dict1,dict2)
    if len(d1_keys+d2_keys) > 0:
        print(f"\nWARNING, keys {d1_keys+d2_keys} are not in both dictionaries.")

    ret_dict = {}
    for k in common_keys:
        ret_dict[k] = get_diff_between_dicts(dict1[k],dict2[k],difftype,inpercent)

    return ret_dict


# Get the difference between values in a dictionary, currently can get either percent diff, or absolute diff, or sum
# Returns a dictionary in the same format (currently does not propagate errors for percent diff, just returns None)
#   dict = {
#       k : [val,var]
#   }
# Note: This function makes use of utils.get_common_keys and utils.get_pdiff
def get_diff_between_dicts(dict1,dict2,difftype,inpercent=False):

    # Get list of sub keys common to both sub dictionaries
    common_keys, d1_keys, d2_keys = get_common_keys(dict1,dict2)
    if len(d1_keys+d2_keys) > 0:
        print(f"\tWARNING, sub keys {d1_keys+d2_keys} are not in both dictionaries.")

    ret_dict = {}
    for k in common_keys:
        v1,e1 = dict1[k]
        v2,e2 = dict2[k]
        if difftype == "percent_diff":
            ret_diff = get_pdiff(v1,v2,in_percent=inpercent)
            ret_err = None
        elif difftype == "absolute_diff":
            ret_diff = v1 - v2
            if (e1 is not None) and (e2 is not None):
                ret_err = e1 - e2 # Assumes these are variances not errors (i.e. already squared)
            else:
                ret_err = None
        elif difftype == "sum":
            ret_diff = v1 + v2
            if (e1 is not None) and (e2 is not None):
                ret_err = e1 + e2 # Assumes these are variances not errors (i.e. already squared)
            else:
                ret_err = None
        else:
            raise Exception(f"Unknown diff type: {difftype}. Exiting...")

        ret_dict[k] = [ret_diff,ret_err]

    return ret_dict
