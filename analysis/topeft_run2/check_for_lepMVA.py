import os
import json

from coffea.nanoevents import NanoEventsFactory

from topcoffea.modules.utils import regex_match

CFG_PATH = "../../input_samples/cfgs/"

pjoin = os.path.join

def read_cfg(fpath,rgx=[]):
    r = {
        "src_xrd": "",
        "jsons": [],
    }
    with open(fpath) as f:
        for l in f:
            l = l.strip().split("#")[0]
            if not len(l):
                continue
            if rgx and len(regex_match([l],rgx)) == 0:
                continue
            if l.startswith("root:"):
                r['src_xrd'] = l
            else:
                r['jsons'].append(l)
    return r


def main():
    cfgs_to_check = [
        "data_samples_NDSkim.cfg",
        "mc_background_samples_NDSkim.cfg",
        "mc_signal_samples_NDSkim.cfg",
    ]

    for cfg_name in cfgs_to_check:
        cfg_fpath = pjoin(CFG_PATH,cfg_name)
        print(f"cfg path: {cfg_fpath}")
        cfg = read_cfg(cfg_fpath)
        njsns = len(cfg["jsons"])
        for idx,jsn_fpath in enumerate(cfg["jsons"]):
            with open(jsn_fpath) as jf:
                jsn = json.load(jf)
                root_files = ["/hadoop" + x.replace("//","/") for x in jsn["files"]]
                root_file = root_files[0]
                print(f"\t[{idx+1:>2}/{njsns}] Checking: {root_file}")
                nano_events = NanoEventsFactory.from_root(root_file,entry_stop=5).events()
                e = nano_events.Electron
                print("\t\te.mvaTTHUL",e.mvaTTHUL)
main()
