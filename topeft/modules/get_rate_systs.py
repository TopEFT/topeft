import json
from topeft.modules.paths import topeft_path

# Retrun the param value from params.json for a given param name
def get_te_param(param_name):
    param_json = topeft_path("params/params.json")
    with open(param_json) as f_params:
        params = json.load(f_params)
        param_val = params[param_name]
    return param_val


# Get the systematic value from the rate_systs json
#   - If literal is True, return the literal string, e.g. "0.88/1.13"
#   - If literal is False, return a pair of floats e.g. [0.88,1.13] for down and up
def get_syst(syst_name,proc_name=None,literal=False):
    syst_json = topeft_path("json/rate_systs.json")
    with open(syst_json) as f_systs:
        rate_systs_dict = json.load(f_systs)["rate_uncertainties"]

        # Try to get the param from the dict
        if syst_name in rate_systs_dict.keys():
            syst_obj = rate_systs_dict[syst_name]
            if syst_name == "lumi":
                # Note we'll return this regardless of what proc name (if any) was passed
                ret_obj = syst_obj
            elif proc_name in rate_systs_dict[syst_name]:
                ret_obj = rate_systs_dict[syst_name][proc_name]
            else:
                raise Exception(f"Error: Unknown proc name \"{proc_name}\", known processes for this syst \"{syst_name}\" are: {list(rate_systs_dict[syst_name].keys())}")
        else:
            raise Exception(f"Error: Unknown syst name \"{syst_name}\", known systs are: {list(rate_systs_dict.keys())}")

        # Now get the output as a str and a pair of floats
        ret_obj_str = str(ret_obj)
        ret_obj_str_split = ret_obj_str.split("/")
        if len(ret_obj_str_split) == 2:
            ret_obj_pair = [float(ret_obj_str_split[0]),float(ret_obj_str_split[1])]
        elif len(ret_obj_str_split) == 1:
            ret_obj_pair = [1.0/float(ret_obj_str_split[0]),float(ret_obj_str_split[0])]
        else:
            raise Exception(f"Error: Syst string \"{ret_obj_str}\" is of an unknown format.")

        if literal:
            return ret_obj_str
        else:
            return ret_obj_pair


# Just jet the list of rate syst keys included in the rate rate syst json
def get_syst_lst():
    syst_json = topeft_path("json/rate_systs.json")
    with open(syst_json) as f_systs:
        rate_systs_dict = json.load(f_systs)["rate_uncertainties"]
        rate_syst_lst = list(rate_systs_dict.keys())
        return rate_syst_lst


# Get the correlation group a process belongs to for a given systematic type (pdf or qcd)
def get_correlation_tag(syst_type,proc_name):
    syst_json = topeft_path("json/rate_systs.json")
    with open(syst_json) as f_systs:
        corr_dict = json.load(f_systs)["correlations"]
        if proc_name in corr_dict.keys():
            if syst_type in corr_dict[proc_name].keys():
                return corr_dict[proc_name][syst_type]
            else: raise Exception(f"Error: Unknown syst type \"{syst_type}\", known systematics with correlations are: {list(corr_dict[proc_name].keys())}")
        else: raise Exception(f"Error: Unknown proc name \"{proc_name}\", known processes are: {list(corr_dict.keys())}")

# Get the dict of jet-dependent scaling factors
def get_jet_dependent_syst_dict(process="Diboson"):
    syst_json = topeft_path("json/rate_systs.json")
    with open(syst_json) as f_systs:
        diboson_njets_dict = json.load(f_systs)["diboson_njets"]
        return (diboson_njets_dict[process])
