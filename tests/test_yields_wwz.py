import subprocess
from os.path import exists

def test_make_yields_after_processor():
    assert (exists('analysis/wwz/histos/output_check_yields.pkl.gz')) # Make sure the input pkl file exists

    args = [
        "python",
        "analysis/wwz/get_wwz_counts.py",
        "-f",
        "analysis/wwz/histos/output_check_yields.pkl.gz",
        "-n",
        "analysis/wwz/output_check_yields"
    ]

    # Produce json
    subprocess.run(args)
    assert (exists('analysis/wwz/output_check_yields.json'))

def test_compare_yields_after_processor():
    args = [
        "python",
        "analysis/wwz/comp_json_yields.py",
        "analysis/wwz/output_check_yields.json",
        "analysis/wwz/ref_for_ci/counts_wwz_ref.json",
        "-t1",
        "New yields",
        "-t2",
        "Ref yields"
    ]

    # Run comparison
    out = subprocess.run(args, stdout=True)
    assert (out.returncode == 0) # Returns 0 if all pass
