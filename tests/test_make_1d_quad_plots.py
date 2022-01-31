import subprocess
import glob

def test_make_1d_quad_plots():
    args = [
        "python",
        "analysis/topEFT/make_1d_quad_plots.py",
        "-i",
        "ttHJet_UL17_R1B14_NAOD-00000_10194.root",
        "-r",
        "./"
    ]

    # Run make_1d_quad
    subprocess.run(args)

    glob.glob('tmp_quad_plos*')
