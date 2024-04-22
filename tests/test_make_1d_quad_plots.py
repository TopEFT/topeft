import subprocess
import glob

def test_make_1d_quad_plots():
    args = [
        "python",
        "analysis/topeft_run2/make_1d_quad_plots.py",
        "-i",
        "ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root",
        "-r",
        "./"
    ]

    # Run make_1d_quad
    assert subprocess.run(args)

    glob.glob('tmp_quad_plos*')
