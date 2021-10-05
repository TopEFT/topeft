import subprocess
import filecmp

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

def test_datacard():
    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "-j",
        "0"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert filecmp.cmp('histos/ttx_multileptons-2lss_p_2b.txt', 'histos/ttx_multileptons-2lss_p_2b_ref.txt')

    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "-j",
        "9"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert filecmp.cmp('histos/ttx_multileptons-2lss_p_4j_2b_ht.txt', 'histos/ttx_multileptons-2lss_p_4j_2b_ht_ref.txt')

    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "-j",
        "6"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert filecmp.cmp('histos/ttx_multileptons-3l_sfz_1b.txt', 'histos/ttx_multileptons-3l_sfz_1b_ref.txt')

    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "-j",
        "24"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert filecmp.cmp('histos/ttx_multileptons-3l_onZ_1b_2j_ptbl.txt', 'histos/ttx_multileptons-3l_onZ_1b_2j_ptbl_ref.txt')
