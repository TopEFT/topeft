import subprocess
import topcoffea.modules.dataDrivenEstimation as dataDrivenEstimation
from os.path import exists
from os import getcwd
from topcoffea.modules.comp_datacard import comp_datacard

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


def test_nonprompt():
    a=dataDrivenEstimation.DataDrivenProducer('analysis/topEFT/histos/output_check_yields.pkl.gz', 'analysis/topEFT/histos/output_check_yields_nonprompt')
    a.dumpToPickle() # Do we want to write this file when testing in CI? Maybe if we ever save the CI artifacts

    assert(exists('analysis/topEFT/histos/output_check_yields_nonprompt.pkl.gz'))

def test_datacard_2l():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields_nonprompt.pkl.gz",
        "--var-lst",
        "njets",
        "-j",
        "0",
    ]

    # Run datacard maker
    subprocess.run(args)

    assert(comp_datacard('histos/ttx_multileptons-2lss_2b_p.txt','analysis/topEFT/test/ttx_multileptons-2lss_2b_p_ref.txt'))

def test_datacard_2l_ht():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "--var-lst",
        "ht",
        "-j",
        "0"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert(comp_datacard('histos/ttx_multileptons-2lss_p_2b_4j_ht.txt','analysis/topEFT/test/ttx_multileptons-2lss_p_2b_4j_ht_ref.txt'))

def test_datacard_3l():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "--var-lst",
        "njets",
        "-j",
        "10"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert(comp_datacard('histos/ttx_multileptons-3l_sfz_1b.txt','analysis/topEFT/test/ttx_multileptons-3l_sfz_1b_ref.txt'))

def test_datacard_3l_ptbl():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "--var-lst",
        "ptbl",
        "-j",
        "40"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert(comp_datacard('histos/ttx_multileptons-3l_onZ_1b_2j_ptbl.txt','analysis/topEFT/test/ttx_multileptons-3l_onZ_1b_2j_ptbl_ref.txt'))
