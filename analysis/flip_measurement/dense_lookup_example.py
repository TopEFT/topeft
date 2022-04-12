import gzip
import pickle
from coffea import lookup_tools

pkl_lst = [
    "../../topcoffea/data/fliprates/flip_probs_topcoffea_UL16APV.pkl.gz",
    "../../topcoffea/data/fliprates/flip_probs_topcoffea_UL16.pkl.gz",
    "../../topcoffea/data/fliprates/flip_probs_topcoffea_UL17.pkl.gz",
    "../../topcoffea/data/fliprates/flip_probs_topcoffea_UL18.pkl.gz",
]

# Before scaling
pkl_lst_old = [
    "../../topcoffea/data/fromTTH/fliprates/flip_probs_apr06BinningAN19127_UL16APV.pkl.gz",
    "../../topcoffea/data/fromTTH/fliprates/flip_probs_apr06BinningAN19127_UL16.pkl.gz",
    "../../topcoffea/data/fromTTH/fliprates/flip_probs_apr06BinningAN19127_UL17.pkl.gz",
    "../../topcoffea/data/fromTTH/fliprates/flip_probs_apr06BinningAN19127_UL18.pkl.gz",
]

# Dump the values currently in the committed fliprates pkl files
for pkl_file in pkl_lst:
    print("Looking at:",pkl_file)
    with gzip.open(pkl_file) as fin:
        hin = pickle.load(fin)
        print(hin.values())

'''
# Example of looking up values using the dense lookup tools
with gzip.open("test_ratio.pkl.gz") as fin:
    hin = pickle.load(fin)
    print(hin.values())
    print(hin.axis("pt").edges())
    print(hin.axis("eta").edges())
    x = lookup_tools.dense_lookup.dense_lookup(hin.values()[()],[hin.axis("pt").edges(),hin.axis("eta").edges()])
    print("x",x)
    print("x",type(x))
    print(x(-35,1.9))
    print(x(-35,-7))
    print(x(500,1.9))
'''
