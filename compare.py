import yaml
import json
from pathlib import Path

# Paths – tweak as needed

region = "TOP22_006_CH_LST_SR"
new_metadata = "metadata_TOP_22_006.yaml"
base = Path(".")
params_yml = base / "topeft" / "params" / "metadata.yml"
top_scenario_yml = base / "analysis" / "metadata" / new_metadata

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def compare_ch_lst_cr():
    meta = load_yaml(params_yml)
    top_meta = load_yaml(top_scenario_yml)

    orig_cr = meta["channels"]["groups"][region]
    new_cr = top_meta["channels"]["groups"][region]

    # Very dumb comparison first
    if orig_cr == new_cr:
        print(f"{region}: metadata.yml and {new_metadata} MATCH exactly ✅")
    else:
        print(f"{region}: mismatch between metadata.yml and {new_metadata} ❌")
        print("  - original lepton_categories:",
              [r["lepton_category"] for r in orig_cr["regions"]])
        print("  - new lepton_categories:",
              [r["lepton_category"] for r in new_cr["regions"]])

if __name__ == "__main__":
    compare_ch_lst_cr()
