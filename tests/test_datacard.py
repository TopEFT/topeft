import subprocess
import filecmp

def test_datacard():
    args = [
        "time",
        "python",
        "analysis/topEFT/run.py",
        ".topcoffea/json/test_samples/UL17_private_ttH_for_CI.json",
        " -o output_check_yields"
    ]

    # Run TopCoffea
    subprocess.run(args)

    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/plotsTopEFT.pkl.gz",
        "--job 0"
    ]

    assert filecmp.cmp('histos/ttx_multileptons-2lss_p_2b.txt', 'histos/ttx_multileptons-2lss_p_2b_ref.txt')

    # Run datacard maker
    subprocess.run(args)

    args = [
        "python",
        "analysis/topEFT/datacard_maker.py",
        "analysis/topEFT/histos/plotsTopEFT.pkl.gz",
        "--job 9"
    ]

    # Run datacard maker
    subprocess.run(args)

    assert filecmp.cmp('histos/ttx_multileptons-2lss_p_4j_2b_ht.txt', 'histos/ttx_multileptons-2lss_p_4j_2b_ht_ref.txt')
