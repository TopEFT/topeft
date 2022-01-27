mport subprocess
import filecmp
import topcoffea.modules.dataDrivenEstimation as dataDrivenEstimation
from work_queue import Factory
from os.path import exists
from os import getcwd

def test_topcoffea():
    args = [
        "time",
        "python",
        "analysis/topEFT/run.py",
        "topcoffea/json/test_samples/UL17_private_ttH_for_CI.json",
        "-o",
        "output_check_yields",
        "-p",
        "analysis/topEFT/histos/"
    ]

    # Run TopCoffea
    subprocess.run(args)

    assert(exists('analysis/topEFT/histos/output_check_yields.pkl.gz'))


