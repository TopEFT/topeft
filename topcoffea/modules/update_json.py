import os
import json

from topcoffea.modules.utils import load_sample_json_file

def update_json(fname,dry_run=False,outname=None,verbose=False,**kwargs):
    '''
        Description:
            Attempts to open a json file, modify one or more of the outermost keys, and then save
            the new json. If dry_run is set to true, then skip writing to an output file. If outname
            is None then the file name will be set to the original and overwrite it.
        Arguments:
            fname -- The full file path to the json file that is to be updated/modified
            dry_run -- If true will do everything except actually write the resulting json to file
            outname -- If not None will write to a json file with this name
            verbose -- If true will include additional print outs about what is being updated
            **kwargs -- The key/value pairs to update in original json. The keys must already exist
                        in the original file or a KeyError will be raised
        Note:
            fname will in general will be the full file path to the desired file, so don't expect it
            to be saved in the same directory as the original w/o making sure the file path is correct
    '''
    jsn = load_sample_json_file(fname)
    jsn.pop('redirector',None)   # Don't currently store this info in the json
    if verbose:
        h,t = os.path.split(fname)
        print(f"Updating {t}")
    for k,new in kwargs.items():
        if not k in jsn:
            raise KeyError(f"Unknown json key specified: {k}")
        old = jsn[k]
        if not isinstance(old,type(new)):
            raise TypeError(f"New should at least be a base class of old: {type(old)} vs {type(new)}")
        if verbose:
            if isinstance(old,list):
                s = f"\tOld {k}: [\n"
                s += ",\n".join([f"\t\t{x}" for x in old])
                s += "\n\t]"
                print(s)
                s = f"\tNew {k}: [\n"
                s += ",\n".join([f"\t\t{x}" for x in new])
                s += "\n\t]"
                print(s)
            else:
                print(f"\t{k}: {old} --> {new}")
        jsn[k] = new
    if dry_run:
        return
    new_file = fname if outname is None else outname
    with open(new_file,'w') as f:
        print(f'>> Writing updated file to {new_file}')
        json.dump(jsn,f,indent=2)