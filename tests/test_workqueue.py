import subprocess
import filecmp
from work_queue import Factory
from os.path import exists
from os import getcwd

def test_topcoffea_wq():
    port=9123
    factory = Factory("local", manager_host_port="localhost:{}".format(port))
    factory.max_workers=1
    factory.min_workers=1
    factory.cores=2

    args = [
        "time",
        "python",
        "work_queue_run.py",
        "../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json",
        "-o",
        "output_check_yields_wq",
        "-p",
        "../../analysis/topEFT/histos/",
        "--prefix",
        "http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/",
        "--port",
        "9123-9124",
        "--chunksize",
        "500",
        "--nchunks",
        "1"
    ]

    # Run TopCoffea
    with factory:
        subprocess.run(args, cwd="analysis/topEFT", timeout=300)

    assert(exists('analysis/topEFT/histos/output_check_yields_wq.pkl.gz'))
