import os
import re
import json

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
            if l.startswith("root:"):
                # Note: This implicitly assumes that a redirector line will appear before any json
                #   paths in the cfg file
                xrd_src = l
            else:
                sample = os.path.basename(l)
                sample = sample.replace(".json","")
                full_path = pjoin(cfg_dir,l)
                jsn = load_sample_json_file(full_path)
                cfg = update_cfg(jsn,sample,cfg=cfg,max_files=max_files,redirector=xrd_src)
    return cfg
