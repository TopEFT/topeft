import subprocess
from os.path import exists

def test_make_yields_after_processor():
    assert (exists('analysis/topeft_run2/histos/output_check_yields.pkl.gz')) # Make sure the input pkl file exists

    args = [
        "python",
        "analysis/topeft_run2/get_yield_json.py",
        "-f",
        "analysis/topeft_run2/histos/output_check_yields.pkl.gz",
        "-n",
        "analysis/topeft_run2/output_check_yields"
    ]

    # Produce json
    subprocess.run(args)
    assert (exists('analysis/topeft_run2/output_check_yields.json'))

def test_compare_yields_after_processor():
    args = [
        "python",
        "analysis/topeft_run2/comp_yields.py",
        "analysis/topeft_run2/output_check_yields.json",
        "analysis/topeft_run2/test/UL17_private_ttH_for_CI_yields.json",
        "-t1",
        "New yields",
        "-t2",
        "Ref yields"
    ]

    # Run comparison
    out = subprocess.run(args, stdout=True)
    assert (out.returncode == 0) # Returns 0 if all pass
