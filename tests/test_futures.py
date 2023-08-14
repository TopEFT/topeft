import subprocess
import topcoffea.modules.dataDrivenEstimation as dataDrivenEstimation
from os.path import exists
from topcoffea.modules.comp_datacard import comp_datacard

def test_topcoffea():
    args = [
        "time",
        "python",
        "analysis/topEFT/run_topeft.py",
        "-x",
        "futures",
        "topcoffea/json/test_samples/UL17_private_ttH_for_CI.json",
        "-o",
        "output_check_yields",
        "-p",
        "analysis/topEFT/histos/",
        "--prefix",
        "http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/",
    ]

    # Run TopCoffea
    subprocess.run(args)

    assert (exists('analysis/topEFT/histos/output_check_yields.pkl.gz'))


def test_nonprompt():
    a=dataDrivenEstimation.DataDrivenProducer('analysis/topEFT/histos/output_check_yields.pkl.gz', 'analysis/topEFT/histos/output_check_yields_nonprompt')
    a.dumpToPickle() # Do we want to write this file when testing in CI? Maybe if we ever save the CI artifacts

    assert (exists('analysis/topEFT/histos/output_check_yields_nonprompt.pkl.gz'))

def test_datacardmaker():
    args = [
        "time",
        "python",
        "analysis/topEFT/make_cards.py",
        "analysis/topEFT/histos/output_check_yields_nonprompt.pkl.gz",
        "-d",
        "histos",
        "--var-lst",
        "lj0pt",
        "--do-nuisance",
        "--ch-lst",
        "2lss_p_4j",
        "--skip-selected-wcs-check"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert (comp_datacard('histos/ttx_multileptons-2lss_p_4j_lj0pt.txt','analysis/topEFT/test/ttx_multileptons-2lss_p_4j_lj0pt.txt'))
