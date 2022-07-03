import os
import re
import json
import gzip
import pickle
import cloudpickle

pjoin = os.path.join

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

# Save to a pkl file
def dump_to_pkl(out_name,out_file):
    if not out_name.endswith(".pkl.gz"):
        out_name = out_name + ".pkl.gz"
    print(f"\nSaving output to {out_name}...")
    with gzip.open(out_name, "wb") as fout:
        cloudpickle.dump(out_file, fout)
    print("Done.\n")

# Get the dictionary of hists from the pkl file (e.g. that a processor outputs)
def get_hist_from_pkl(path_to_pkl,allow_empty=True):
    h = pickle.load( gzip.open(path_to_pkl) )
    if not allow_empty:
        h = {k:v for k,v in h.items() if v.values() != {}}
    return h

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
