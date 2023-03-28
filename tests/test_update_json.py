import os

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.update_json import update_json
from topcoffea.modules.utils import load_sample_json_file

def test_update_json():
    src_fname = topcoffea_path("json/test_samples/UL17_private_ttH_for_CI.json")
    dst_fname = "tmp_test_file.json"

    assert (os.path.exists(src_fname))

    if os.path.exists(dst_fname):
        os.remove(dst_fname)

    # Check to make sure that dry_run doesn't actually generate a file
    update_json(src_fname,dry_run=True,outname=dst_fname,verbose=True)

    assert (not os.path.exists(dst_fname))

    # No updates, should just create a copy
    update_json(src_fname,outname=dst_fname,verbose=True)

    assert (os.path.exists(dst_fname))
    a = open(src_fname).read().strip()
    b = open(dst_fname).read().strip()
    assert (a==b)

    updates = {
        "xsec": 909.090,
        "WCnames": ["ctG","ctZ","cpt"],
        "year": "1999",
        "files": [],
        "isData": False
    }

    # Apply the above updates and overwrite the previous test file
    update_json(src_fname,outname=dst_fname,verbose=True,**updates)

    # Check that the updates were all applied correctly
    test_jsn = load_sample_json_file(dst_fname)
    for k,v in updates.items():
        assert (test_jsn[k] == v)
