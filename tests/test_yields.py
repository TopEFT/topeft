import subprocess
from os.path import exists
from os import getcwd

def test_make_yields_after_processor():
    assert(exists('analysis/topEFT/histos/output_check_yields.pkl.gz')) # Make sure the input pkl file exists

    args = [
        "python",
        "analysis/topEFT/get_yield_json.py",
        "-f",
        "analysis/topEFT/histos/output_check_yields.pkl.gz",
        "-n",
        "analysis/topEFT/output_check_yields"
    ]

    # Produce json
    subprocess.run(args)
    assert(exists('analysis/topEFT/output_check_yields.json'))

def test_compare_yields_after_processor():
    args = [
        "python",
        "analysis/topEFT/comp_yields.py",
        "analysis/topEFT/output_check_yields.json",
        "analysis/topEFT/test/UL17_private_ttH_for_CI_yields.json",
        "-t1",
        "New yields",
        "-t2",
        "Ref yields"
    ]

    # Run comparison
    out = subprocess.run(args, stdout=True)
    assert(out.returncode == 0) # Returns 0 if all pass
