import setuptools

setuptools.setup(
    name='topeft',
    version='0.0.0',
    description='Top quark analyses using the Coffea framework',
    packages=setuptools.find_packages(),
    # Include data files (Note: "include_package_data=True" does not seem to work)
    package_data={
        "topeft" : [
            "cfg/*.cfg",
            "json/*",
            "data/scaleFactors/*.root",
            "data/fliprates/*.pkl.gz",
            "data/fromTTH/fakerate/*.root",
            "data/leptonSF/*/*.root",
            "data/leptonSF/*/*.json",
            "data/triggerSF/*.pkl.gz",
            "data/JEC/*.txt",
            "data/JER/*.txt",
            "data/btagSF/UL/*.pkl.gz",
            "data/btagSF/UL/*.csv",
            "data/pileup/*.root",
            "data/MuonScale/*txt",
            "data/goldenJsons/*.txt",
            "data/TauSF/*.json",
            "data/topmva/lepid_weights/*.bin",
        ],
    }
)

