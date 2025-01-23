import setuptools

setuptools.setup(
    name='topeft',
    version='0.0.0',
    description='Analysis code for top quark EFT analyses',
    packages=setuptools.find_packages(),
    # Include data files (Note: "include_package_data=True" does not seem to work)
    package_data={
        "topeft" : [
            "channels/*",
            "params/*",
            "data/fliprates/*.pkl.gz",
            "data/triggerSF/*.pkl.gz",
            "data/btagSF/UL/*.pkl.gz",
            "data/btagSF/UL/*.csv",
            "modules/*.json"
        ],
    }
)

