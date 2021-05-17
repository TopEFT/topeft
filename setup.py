import setuptools

setuptools.setup(
    name='topcoffea',
    version='0.0.0',
    description='Top quark analyses using the Coffea framework',
    packages=setuptools.find_packages(),
    package_data={
        "topcoffea" : ["cfg/*.cfg","data/scaleFactors/*.root"],
    }
)

