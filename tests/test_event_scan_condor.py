"""Adapter tests compose the maintained protocol with synthetic scheduler replies."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from analysis.topeft_run2 import event_scan_condor as condor
from analysis.topeft_run2 import event_scan as scan
from test_event_scan import unit_fixture


@pytest.fixture
def execution(unit_fixture,tmp_path,monkeypatch):
    unit,_=unit_fixture
    manifest=tmp_path/"units.json"
    manifest.write_text(json.dumps(dict(schema_version=1,units=[unit])))
    monkeypatch.setattr(condor.scheduler,"resolve_target",
                        lambda:dict(pool="collector.fixture.invalid",schedd="submit.fixture.invalid"))
    path=condor.prepare(manifest,unit["unit_id"],tmp_path/"attempt","/bin/true",sys.executable)
    return path,condor.read_descriptor(path)


def test_prepare_has_no_scheduler_effect_and_preserves_scientific_unit(execution,unit_fixture):
    path,descriptor=execution
    assert descriptor["unit"]==unit_fixture[0]
    assert not (path.parent/"submission_intent.json").exists()
    text=(path.parent/"execution.sub").read_text()
    assert text.count("queue 1\n")==1 and "max_retries = 0" in text
    assert "request_cpus = 1" in text and "allowed_execute_duration = 3600" in text
    assert "event_scan_condor --worker" in text
    assert "make_cr_and_sr_plots" not in text
    assert "executable = /usr/bin/true" in text or "executable = /bin/true" in text
    assert descriptor["token"] in text


def test_one_submit_reuses_identity_and_never_resubmits(execution,monkeypatch):
    path,descriptor=execution;calls=[]
    def submit(command,**kw):
        calls.append(command)
        intent=json.loads((path.parent/"submission_intent.json").read_text())
        assert intent["token"]==descriptor["token"]
        assert command[:5]==["condor_submit","-pool","collector.fixture.invalid","-name","submit.fixture.invalid"]
        return SimpleNamespace(returncode=0,stdout="123.0\n",stderr="")
    monkeypatch.setattr(condor.scheduler.subprocess,"run",submit)
    assert condor.submit(path)["job"]=="123.0"
    assert json.loads((path.parent/"scheduler_identity.json").read_text())["job"]=="123.0"
    with pytest.raises(ValueError,match="no resubmit"):condor.submit(path)
    assert len(calls)==1


def test_ambiguous_submit_keeps_intent_and_refuses_second_attempt(execution,monkeypatch):
    path,_=execution;calls=[]
    def timeout(command,**kw):
        calls.append(command)
        raise subprocess.TimeoutExpired(command,60)
    monkeypatch.setattr(condor.scheduler.subprocess,"run",timeout)
    with pytest.raises(condor.durable.ActiveChildUnresolved):condor.submit(path)
    assert (path.parent/"submission_intent.json").exists()
    assert not (path.parent/"scheduler_identity.json").exists()
    with pytest.raises(ValueError,match="no resubmit"):condor.submit(path)
    assert len(calls)==1


def test_worker_exclusive_marker_and_same_scanner(execution,monkeypatch):
    path,descriptor=execution;calls=[]
    monkeypatch.setattr(condor,"worker_identity",lambda token:"123.0")
    def run(unit,output,**kwargs):
        calls.append(unit);output.mkdir()
        (output/"receipt.json").write_text('{"status":"COMPLETE"}')
        return dict(status="COMPLETE")
    monkeypatch.setattr(scan,"run_unit",run)
    assert condor.worker(path)==0
    with pytest.raises(FileExistsError):condor.worker(path)
    assert calls==[descriptor["unit"]]
    assert json.loads((path.parent/"worker_receipt.json").read_text())["job"]=="123.0"


def test_worker_binds_exact_job_ad_token(execution,tmp_path,monkeypatch):
    _,descriptor=execution
    path=tmp_path/"job.ad"
    path.write_text('ClusterId = 42\nProcId = 0\nPlotExecutionToken = "'+descriptor["token"]+'"\n')
    monkeypatch.setenv("_CONDOR_JOB_AD",str(path))
    assert condor.worker_identity(descriptor["token"])=="42.0"
    with pytest.raises(ValueError,match="contradiction"):condor.worker_identity("0"*64)


def test_changed_manifest_or_descriptor_prevents_worker(execution,monkeypatch):
    path,descriptor=execution
    changed=dict(descriptor);changed["unit"]=dict(changed["unit"],entry_start=1)
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(scan,"run_unit",lambda *a,**k:pytest.fail("science launched"))
    with pytest.raises(ValueError,match="descriptor changed"):condor.worker(path)
