import setuptools

setuptools.setup(
    name='topcoffea',
    version='0.0.0',
    description='Top quark analyses using the Coffea framework',
    packages=setuptools.find_packages(),
    # Include data files (Note: "include_package_data=True" does not seem to work)
    package_data={
        "topcoffea" : [
            "cfg/*.cfg",
            "json/*",
            "data/scaleFactors/*.root",
            "data/fromTTH/fakerate/*.root",
            "data/fromTTH/fliprates/*.root",
            "data/fromTTH/lepSF/*/*/*.root",
            "data/fromTTH/lepSF/*/*/*/*.root",
            "data/leptonSF/elec/*.root",
            "data/leptonSF/muon/*.root",
            "data/leptonSF/muon/*.json",
            "data/triggerSF/*.pkl.gz"
            "data/JEC/*.txt",
            "data/JER/*.txt",
            "data/btagSF/UL/*.pkl.gz",
            "data/btagSF/UL/*.csv",
            "data/btagSF/*.csv",
            "data/pileup/*.root",
            "data/MuonScale/*txt",
            "data/goldenJsons/*.txt",
            "data/triggerSF/*.pkl.gz"
        ],
    }
)

