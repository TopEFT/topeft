import json
from topcoffea.modules.paths import topcoffea_path

# Return the lumi from the json/lumi.json file for a given year
def get_lumi(year):
    lumi_json = topcoffea_path("json/lumi.json")
    with open(lumi_json) as f_lumi:
       lumi = json.load(f_lumi)
       lumi = lumi[year]
    return lumi

# Retrun the param value from params.json for a given param name
def get_cut(param_name):
    param_json = topcoffea_path("json/params.json")
    with open(param_json) as f_params:
       params = json.load(f_params)
       param_val = params[param_name]
    return param_val
