import subprocess
from os.path import exists

def test_topcoffea():
    args = [
        "time",
        "python",
        "analysis/wwz/run_wwz4l.py",
        "-x",
        "futures",
        "topcoffea/json/test_samples/UL17_WWZJetsTo4L2Nu_forCI.json",
        "-o",
        "output_check_yields",
        "-p",
        "analysis/wwz/histos/"
    ]

    # Run TopCoffea
    subprocess.run(args)

    assert (exists('analysis/wwz/histos/output_check_yields.pkl.gz'))
