import os
#import subprocess

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.utils import get_files

pjoin = os.path.join

def test_get_files():
    # Should only find files in the top most directory
    path = topcoffea_path("json")
    file_list = get_files(
        top_dir=path,
        recursive=False,
        verbose=True
    )

    assert (len(file_list) == 4)
    assert (pjoin(path,"lumi.json") in file_list)
    assert (pjoin(path,"params.json") in file_list)
    assert (pjoin(path,"rate_systs.json") in file_list)
    assert (pjoin(path,"README.md") in file_list)

    # Check to see if regex matching for file names is working
    path = topcoffea_path("json")
    file_list = get_files(
        top_dir=path,
        match_files=[".*\\.json"],
        recursive=False,
        verbose=True
    )

    assert (len(file_list) == 3)
    assert (pjoin(path,"lumi.json") in file_list)
    assert (pjoin(path,"params.json") in file_list)
    assert (pjoin(path,"rate_systs.json") in file_list)

    # Check if the 'recursive' and 'ignore_dirs' options are working
    path = topcoffea_path("json")
    file_list = get_files(
        top_dir=path,
        match_files=[".*\\.json"],
        recursive=True,
        # Should only recurse into "sync_samples" and "test_samples"
        ignore_dirs=["background_samples","data_samples","signal_samples","wwz_analysis_samples"],
        verbose=True
    )

    assert (len(file_list) == 8)
    assert (pjoin(path,"lumi.json") in file_list)
    assert (pjoin(path,"params.json") in file_list)
    assert (pjoin(path,"rate_systs.json") in file_list)
    assert (pjoin(path,"sync_samples/ttHJetToNonbb_sync.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_for_CI.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_hadoop_for_CI.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_hadoop_for_CI_NDSkim.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_WWZJetsTo4L2Nu_forCI.json") in file_list)

    # Check if the 'ignore_files' option is working as intended
    path = topcoffea_path("json")
    file_list = get_files(
        top_dir=path,
        match_files=[".*\\.json"],
        ignore_files=["lumi\\.json","params\\.json"],
        recursive=True,
        # Should only recurse into "sync_samples" and "test_samples"
        ignore_dirs=["background_samples","data_samples","signal_samples","wwz_analysis_samples"],
        verbose=True
    )

    assert (len(file_list) == 6)
    assert (pjoin(path,"lumi.json") not in file_list)
    assert (pjoin(path,"params.json") not in file_list)
    assert (pjoin(path,"rate_systs.json") in file_list)
    assert (pjoin(path,"sync_samples/ttHJetToNonbb_sync.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_for_CI.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_hadoop_for_CI.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_private_ttH_hadoop_for_CI_NDSkim.json") in file_list)
    assert (pjoin(path,"test_samples/UL17_WWZJetsTo4L2Nu_forCI.json") in file_list)

    # Recurse through the entire json directory ignoring all json files. Should only match the
    #   "README.md" file
    path = topcoffea_path("json")
    file_list = get_files(
        top_dir=path,
        ignore_files=[".*\\.json"],
        recursive=True
    )

    assert (len(file_list) == 1)
    assert (pjoin(path,"README.md") in file_list)

# def test_make_skim_jsons():
#     # This test can only be run from a machine that has the ND hadoop cluster mounted on it
#     if os.environ['HOSTNAME'] != 'earth.crc.nd.edu':
#         return

#     args = [
#         "python",
#         "analysis/topEFT/make_skim_jsons.py",
#         "--json-dir",
#         topcoffea_path("json/data_samples"),
#         "--file",
#         "analysis/topEFT/ND_data_skim_locations.txt",
#         "--output-dir",
#         "."
#     ]
#     cmd_str = " ".join(args)
#     print(f"Test command: {cmd_str}")
#     subprocess.run(args)

if __name__ == "__main__":
    test_get_files()
