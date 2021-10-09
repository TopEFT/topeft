import subprocess
import glob

def test_make_1d_quad_plots():
    args = [
        "python",
        "analysis/topEFT/make_1d_quad_plots.py",
        "-i",
        "NAOD-00000_18449.root",
        "-r",
        "/afs/crc.nd.edu/user/b/byates2/topcoffea/"
    ]

    # Run make_1d_quad
    subprocess.run(args)

    glob.glob('tmp_quad_plos*')
