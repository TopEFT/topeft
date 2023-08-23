import subprocess
from work_queue import Factory
from os.path import exists


def test_topcoffea_wq():
    port=9123
    factory = Factory("local", manager_host_port="localhost:{}".format(port))
    factory.max_workers=1
    factory.min_workers=1
    factory.cores=2

    factory.extra_options="--max-backoff 15"

    args = [
        "time",
        "python",
        "run_topeft.py",
        "input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json",
        "-o",
        "output_check_yields_wq",
        "-p",
        "../../analysis/topeft_run2/histos/",
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
        subprocess.run(args, cwd="analysis/topeft_run2", timeout=400)

    assert (exists('analysis/topeft_run2/histos/output_check_yields_wq.pkl.gz'))
