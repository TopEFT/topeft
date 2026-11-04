"""Range/receipt and native-observer composition, without a scientific campaign."""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

from analysis.topeft_run2 import event_scan as scan


@pytest.fixture
def unit_fixture(tmp_path, monkeypatch):
    root = tmp_path / "sample.root"
    root.write_bytes(b"inert fixture")
    sample, cfg, sr, target = [tmp_path / name for name in ("sample.json", "sample.cfg", "sr.json", "targets.json")]
    sample.write_text(json.dumps(dict(histAxisName="process", year="2022", WCnames=["c"], files=["/sample.root"])))
    cfg.write_text("fixture")
    sr.write_text('{"ALL_CH_LST_SR": {}}')
    group = dict(group_id="group", dataset="sample", process="process", metadata_year="2022",
                 year_or_era="2022", tree_name="Events")
    owner = dict(group_id="group", source_channel="channel", fit_variable="x",
                 source_edges="[0, 100]", source_bin="1", native_term_index="2",
                 negative_row_id="target", term="c*c")
    paths = [Path(scan.__file__).resolve(), Path(scan.native.__file__).resolve(),
             scan.native.PROCESSOR, sample, cfg, sr]
    target.write_text(json.dumps(dict(schema_version=1, group=group, ownership=[owner],
        sr_config_path=str(sr), calibrations=[], input_sha256={str(p): scan.native.sha256(p) for p in paths})))
    source = dict(path=str(root), tree_name="Events", tree_entries=100001, root_uuid="fixture",
                  device=1, inode=2, size=root.stat().st_size, mtime_ns=root.stat().st_mtime_ns)
    monkeypatch.setattr(scan, "source_identity", lambda *args: dict(source))
    unit = scan.make_unit(root, sample, cfg, target, 5, 7)
    return unit, source


def test_range_defaults_and_generic_large_start():
    assert scan.entry_range() == (0, None)
    assert scan.entry_range(tree_entries=12) == (0, 12)
    assert scan.entry_range(50000, None, 70000) == (50000, 70000)
    assert scan.entry_range(3, 7, 12) == (3, 7)


def test_source_identity_portable_across_mounts_and_local_replacement_detected(tmp_path,monkeypatch):
    import sys
    root=tmp_path/'portable.root';root.write_bytes(b'header fixture')
    root=root.resolve();original_stat=Path.stat
    state=dict(device=1,inode=2,replace=False)
    class header:
        file=SimpleNamespace(uuid='same_root_uuid')
        def __enter__(self):return self
        def __getitem__(self,key):return SimpleNamespace(num_entries=10)
        def __exit__(self,*args):
            if state['replace']:state['inode']+=1
    def observed_stat(path,*args,**kwargs):
        real=original_stat(path,*args,**kwargs)
        if path==root:
            return SimpleNamespace(st_dev=state['device'],st_ino=state['inode'],
                                   st_size=real.st_size,st_mtime_ns=real.st_mtime_ns)
        return real
    monkeypatch.setattr(Path,'stat',observed_stat)
    monkeypatch.setitem(sys.modules,'uproot',SimpleNamespace(open=lambda *a:header()))
    first=scan.source_identity(root)
    state.update(device=41,inode=99)
    assert scan.source_identity(root)==first
    assert 'device' not in first and 'inode' not in first
    state['replace']=True
    with pytest.raises(ValueError,match='source changed'):
        scan.source_identity(root)


@pytest.mark.parametrize("start,stop,total", [
    (-1, None, 10), (True, None, 10), (1.2, None, 10),
    (0, 0, 10), (2, 1, 10), (1, True, 10), (1, 2.5, 10),
    (10, None, 10), (0, 11, 10), (0, None, 0)])
def test_invalid_range_before_output(start, stop, total):
    with pytest.raises(ValueError):
        scan.entry_range(start, stop, total)


def test_direct_and_manifest_unit_are_identical(unit_fixture, tmp_path):
    unit, _ = unit_fixture
    manifest = tmp_path / "units.json"
    manifest.write_text(json.dumps(dict(schema_version=1, units=[unit])))
    assert scan.select_unit(manifest, unit["unit_id"]) == unit
    again = scan.make_unit(unit["source_identity"]["path"], unit["sample_json"],
                          unit["analysis_config"], unit["target_manifest"], 5, 7)
    assert again == unit
    manifest.write_text(json.dumps(dict(schema_version=1, units=[unit, unit])))
    with pytest.raises(ValueError, match="duplicate"):
        scan.select_unit(manifest, unit["unit_id"])


def test_source_and_config_changes_invalidate(unit_fixture, monkeypatch):
    unit, source = unit_fixture
    monkeypatch.setattr(scan, "source_identity", lambda *a: dict(source, root_uuid="changed"))
    with pytest.raises(ValueError, match="source identity"):
        scan.validate_unit(unit, revalidate_source=True)
    Path(unit["analysis_config"]).write_text("changed")
    with pytest.raises(ValueError, match="input identity"):
        scan.validate_unit(unit)


def receipt(source, a, b, **changes):
    return dict(dict(source_identity=source, science_identity="one_contract", status="COMPLETE",
                     range_complete=True, requested_range=[a,b], processed_range=[a,b]), **changes)


def test_coverage_requires_identity_nonoverlap_and_no_gaps():
    source = dict(tree_entries=10, root_uuid="one")
    assert scan.coverage([receipt(source,5,10), receipt(source,0,5)], source)["whole_file_complete"]
    assert not scan.coverage([receipt(source,3,5)], source,3,5)["whole_file_complete"]
    cases = [
        [receipt(source,0,5), receipt(source,0,5)],
        [receipt(source,0,6), receipt(source,5,10)],
        [receipt(source,0,4), receipt(source,5,10)],
        [receipt(source,0,9)],
        [receipt(dict(source,root_uuid="wrong"),0,10)],
        [receipt(source,0,10,status="STOPPED_NO_RETRY",range_complete=False)],
        [receipt(source,0,10,science_identity="other"),receipt(source,0,10)],
        [receipt(source,0,10,requested_range=[0,9])],
    ]
    for rows in cases:
        with pytest.raises(ValueError):
            scan.coverage(rows, source)


def install_fake_runtime(monkeypatch, fail_second=False):
    module = SimpleNamespace()
    calls = []
    class events:
        def __init__(self, a, b):
            self.run=[1]*(b-a); self.luminosityBlock=[2]*(b-a)
            self.event=list(range(a,b))
        def __len__(self):return len(self.event)
    def process(ev):
        n=len(ev)
        module._diagnostic_capture(dict(ch_name="channel",dense_axis_name="x",
            syst_var="nominal",wgt_fluct="nominal",appl="isSR_3l",
            all_cuts_mask=ak.Array([True]*n),combined_axis_mask=None,
            values_cut_map={"x":ak.Array([10]*n)},
            eft_coeffs_cut=np.tile(np.array([[-.25,0,-.5]],dtype=np.float32),(n,1)),
            weights_flat=ak.Array([2.]*n),events=ev))
    def factory(contract,sample):
        selectors=scan.native.selectors_for(contract["ownership"])["group"]
        return module,SimpleNamespace(process=process),selectors,{"channel":"isSR_3l"},"instrumented"
    def loader(unit,a,b,dataset):
        calls.append((a,b))
        if fail_second and len(calls)==2:raise RuntimeError("interrupted fixture")
        return events(a,b)
    monkeypatch.setattr(scan,"_processor",factory)
    monkeypatch.setattr(scan,"_events",loader)
    monkeypatch.setattr(scan.native,"install_io_guard",lambda calibrations:Counter())
    return calls


def test_worker_streams_native_payload_and_marks_only_requested_range(unit_fixture,tmp_path,monkeypatch):
    unit,_=unit_fixture
    calls=install_fake_runtime(monkeypatch)
    output=tmp_path/"result"
    result=scan.run_unit(unit,output,chunk_entries=1,reserve_bytes=0)
    assert calls==[(5,6),(6,7)]
    assert result["status"]=="COMPLETE" and result["range_complete"]
    assert not result["whole_file_complete"]
    assert result["requested_range"]==result["processed_range"]==[5,7]
    with gzip.open(output/"events.tsv.gz","rt") as f:rows=list(csv.DictReader(f,delimiter="\t"))
    assert [int(row["entry"]) for row in rows]==[5,6]
    assert all(float(row["q0"])==-.5 and json.loads(row["qii"])==[-1.] for row in rows)
    assert json.loads((output/"sums.json").read_text())[0]["partial_qii"]==-2
    with pytest.raises(ValueError,match="collision"):
        scan.run_unit(unit,output,reserve_bytes=0)


def test_interrupted_chunk_cannot_be_reused_as_complete(unit_fixture,tmp_path,monkeypatch):
    unit,_=unit_fixture
    install_fake_runtime(monkeypatch,fail_second=True)
    result=scan.run_unit(unit,tmp_path/"partial",chunk_entries=1,reserve_bytes=0)
    assert result["status"]=="STOPPED_NO_RETRY"
    assert result["completed_chunks"]==[[5,6]] and result["processed_range"]==[5,6]
    assert not result["range_complete"] and not result["whole_file_complete"]
    with pytest.raises(ValueError):
        scan.coverage([result],unit["source_identity"],5,7)


def test_no_round_specific_resume_branch():
    import ast
    import re
    from analysis.topeft_run2 import event_scan_condor
    for module in (scan,event_scan_condor):
        source=Path(module.__file__).read_text()
        assert not re.search(r"output_[0-9]+[.]root",source)
        assert not any(isinstance(node,ast.Constant) and node.value==50000
                       for node in ast.walk(ast.parse(source)))
