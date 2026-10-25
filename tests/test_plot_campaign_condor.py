import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import hist
import awkward as ak
import pytest
from PIL import Image

from analysis.topeft_run2 import make_cr_and_sr_plots as plotter
from analysis.topeft_run2 import plot_campaign_condor as campaign


def _monitor_fixture(tmp_path, *, duplicate=False, wrong_root=False):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(parents=True)
    entries = []
    for index, workers in enumerate((1, 4), 1):
        unit_id = f"unit_{index}"
        descriptor_path = diagnostics / "dispatch" / unit_id / "units" / f"{unit_id}.json"
        descriptor_path.parent.mkdir(parents=True)
        output_root = tmp_path / ("other" if wrong_root and index == 2 else "campaign")
        descriptor = {"schema": "plot_campaign_unit_v1", "unit_id": unit_id,
                      "principal_pkl": f"/input/{unit_id}.pkl.gz", "campaign_id": output_root.name,
                      "campaign_manifest": "/input/campaign.tsv", "output_root": str(output_root),
                      "workers": workers,
                      "expected_figures": 3, "row_ids": [unit_id]}
        descriptor_path.write_bytes(campaign._json_bytes(descriptor))
        entries.append({"unit_id": unit_id, "scheduler_id": "1.0" if duplicate else f"{index}.0",
                        "profile": f"W{workers}", "request_cpus": workers,
                        "request_memory_mb": 64000, "request_disk_mb": 10000,
                        "descriptor_path": str(descriptor_path),
                        "descriptor_sha256": campaign._sha256(descriptor_path),
                        "principal_pkl_path": descriptor["principal_pkl"],
                        "output_root": str(output_root), "rows": 1, "slots": 3})
    (diagnostics / "campaign_submission_map.json").write_bytes(campaign._json_bytes({
        "units": entries, "planned_units": 2, "planned_rows": 2, "planned_slots": 6}))
    return diagnostics


def test_monitor_queue_history_unknown_and_progress(tmp_path, monkeypatch, capsys):
    diagnostics = _monitor_fixture(tmp_path)
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    (output_root / "partial.png").write_bytes(b"not opened")
    (output_root / "receipt_v2.json").write_text("{}")
    ads = {("condor_q", "1.0"): {"ClusterId": 1, "ProcId": 0, "JobStatus": 1,
                                 "NumJobStarts": 2, "RequestCpus": 1},
           ("condor_q", "2.0"): {"ClusterId": 2, "ProcId": 0, "JobStatus": 2,
                                 "RequestCpus": 4}}
    calls = []
    def query(job, command):
        calls.append((command, job))
        return ads.get((command, job))
    monkeypatch.setattr(campaign, "_query_monitor_job", query)
    snapshot = campaign.observe_monitor(diagnostics)
    assert snapshot["campaign_status"] == "ACTIVE_NO_IMMEDIATE_ATTENTION"
    assert snapshot["counts"]["idle"] == snapshot["counts"]["running"] == 1
    assert [(unit["profile"], unit["workers"]) for unit in snapshot["units"]] == [
        ("W1", 1), ("W4", 4)]
    assert snapshot["units"][0]["ads"]["NumJobStarts"] == 2
    assert snapshot["progress"]["png_count"] == 1
    assert snapshot["progress"]["label"].startswith("informational_only")
    campaign._print_monitor(snapshot, json_mode=False)
    human = capsys.readouterr().out
    assert "ACTIVE_NO_IMMEDIATE_ATTENTION" in human
    assert "idle=1 running=1" in human
    campaign._print_monitor(snapshot, json_mode=True)
    machine = json.loads(capsys.readouterr().out)
    assert machine["counts"] == snapshot["counts"]
    assert machine["campaign_status"] == snapshot["campaign_status"]
    assert calls == [("condor_q", "1.0"), ("condor_q", "2.0")]

    ads[("condor_q", "1.0")]["JobStatus"] = 5
    ads[("condor_q", "1.0")]["HoldReason"] = "quota"
    del ads[("condor_q", "2.0")]
    ads[("condor_history", "2.0")] = {"ClusterId": 2, "ProcId": 0,
                                       "JobStatus": 4, "ExitCode": 0}
    snapshot = campaign.observe_monitor(diagnostics)
    assert snapshot["campaign_status"] == "ACTIVE_WITH_ATTENTION_REQUIRED"
    assert snapshot["attention_units"] == ["unit_1"]
    assert snapshot["units"][1]["source"] == "history"
    assert snapshot["units"][1]["exit_result"] == "exit 0"
    ads[("condor_q", "1.0")]["JobStatus"] = 4
    ads[("condor_q", "1.0")]["ExitCode"] = 7
    snapshot = campaign.observe_monitor(diagnostics)
    assert snapshot["campaign_status"] == "ALL_TERMINAL_READY_FOR_COLLECTION"
    assert snapshot["units"][0]["exit_result"] == "exit 7"
    del ads[("condor_history", "2.0")]
    snapshot = campaign.observe_monitor(diagnostics)
    assert snapshot["campaign_status"] == "MONITOR_INCOMPLETE_OR_UNRESOLVED"
    assert snapshot["units"][1]["state"] == "UNKNOWN"


def test_monitor_rejects_bad_mapping_and_scheduler_identity(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="duplicate"):
        campaign._monitor_units(_monitor_fixture(tmp_path / "duplicate", duplicate=True))
    with pytest.raises(ValueError, match="Inconsistent"):
        campaign._monitor_units(_monitor_fixture(tmp_path / "roots", wrong_root=True))
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, '[{"ClusterId": 9, "ProcId": 0}]', ""))
    with pytest.raises(ValueError, match="another scheduler identity"):
        campaign._query_monitor_job("1.0", "condor_q")


def test_monitor_watch_reuses_observation_and_interrupts(tmp_path, monkeypatch, capsys):
    diagnostics = _monitor_fixture(tmp_path)
    seen = []
    monkeypatch.setattr(campaign, "observe_monitor", lambda path: seen.append(path) or
                        {"campaign_status": "ACTIVE_NO_IMMEDIATE_ATTENTION"})
    monkeypatch.setattr(campaign, "_print_monitor", lambda snapshot, **kwargs: None)
    def interrupt(interval):
        raise KeyboardInterrupt
    monkeypatch.setattr(campaign.time, "sleep", interrupt)
    monkeypatch.setattr(campaign.sys, "argv", ["plot_campaign_condor.py", "monitor",
                        "--diagnostics-dir", str(diagnostics), "--watch", "60"])
    assert campaign.main() == 0
    assert seen == [diagnostics]
    source = inspect.getsource(campaign.observe_monitor) + inspect.getsource(campaign.main)
    assert all(command not in source for command in ("condor_submit", "condor_rm",
                "condor_release", "condor_hold", "condor_qedit"))


_accepted_manifest = (
    Path(__file__).resolve().parents[2]
    / "reports/diagnostics/SR_BASELINE_ALL_SR_MANUAL_HANDOFF_028/coverage_input_manifest.tsv"
)


def _fixture_manifest(tmp_path, *, run="run3"):
    pkl = tmp_path / "principal_SR.pkl.gz"
    pkl.write_bytes(b"inert input: never deserialize")
    pkl_digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
    sidecar = Path(str(pkl) + ".metadata.json")
    sidecar.write_text(json.dumps({"artifact": {"pkl_sha256": pkl_digest}}))
    sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest = tmp_path / "campaign.tsv"
    header = "\t".join(campaign._header)
    year = "2022" if run == "run3" else "2018"
    rows = [
        ("one_njets", "njets", run, "combined", "owner", year, "njets", "1"),
        ("two_other", "other", run, year, "owner", year, "ptz", "1"),
    ]
    manifest.write_text(header + "\n" + "\n".join("\t".join((
        *row, str(pkl), pkl_digest, str(sidecar), sidecar_digest,
    )) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_run2_public_row_path_omits_block_but_preserves_unit_identity(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    fields = manifest.read_text().splitlines()
    first = fields[1].split("\t")
    second = fields[2].split("\t")
    for row, owner in ((first, "run2_block1"), (second, "run2_block2")):
        row[0] = owner + "_combined_njets"
        row[1:7] = ["njets", "run2", "combined", owner, "2018", "njets"]
    manifest.write_text(fields[0] + "\n" + "\t".join(first) + "\n"
                        + "\t".join(second) + "\n")
    rows, digest = campaign.read_campaign(manifest)
    assert [campaign._output_relative(row) for row in rows] == [
        "run2/combined/njets", "run2/combined/njets"]
    unit = campaign.derive_units(rows, digest)[0]
    assert unit["row_ids"] == sorted(row["unit"] for row in rows)
    assert unit["outputs"] == ["run2/combined/njets"] * 2
    assert all(row["owner"].startswith("run2_block") for row in unit["rows"])
    assert campaign._output_relative({**rows[0], "run": "run3"}) == (
        "run3/combined/run2_block1/njets")
    planned = ["run2/combined/njets/category/plot.png"]
    skip = {"outcome": "skipped", "reason": "empty-data-content",
            "region": "SR", "mode": "aggregate", "row_id": rows[0]["unit"],
            "variable": "njets", "paths": planned}
    accounting = campaign._reconcile_outputs(
        {"expected_figures": 1, "rows": [rows[0]]}, planned, [skip], [], 0)
    assert (accounting["planned_slot_count"], accounting["allowed_skip_count"],
            accounting["unaccounted_count"]) == (1, 1, 0)
    with pytest.raises(RuntimeError, match="not bound to its logical row"):
        campaign._reconcile_outputs(
            {"expected_figures": 1, "rows": [rows[0]]}, planned,
            [{**skip, "paths": ["run2/combined/run2_block1/njets/category/plot.png"]}],
            [], 0)
    duplicate_owner = second.copy()
    duplicate_owner[4] = "run2_block1"
    manifest.write_text(fields[0] + "\n" + "\t".join(first) + "\n"
                        + "\t".join(duplicate_owner) + "\n")
    with pytest.raises(ValueError, match="Duplicate logical row or output directory"):
        campaign.read_campaign(manifest)


def test_accepted_manifest_partitions_exactly_and_prepare_never_opens_pkls(
        tmp_path, monkeypatch):
    rows, manifest_hash = campaign.read_campaign(_accepted_manifest)
    units = campaign.derive_units(rows, manifest_hash)
    assert len(rows) == 100
    assert sum(row["kind"] == "njets" for row in rows) == 50
    assert sum(row["kind"] == "other" for row in rows) == 50
    assert len(units) == 10
    manifest_paths = {row["pkl"] for row in rows}
    assert len(manifest_paths) == 10
    assert sum(unit["expected_figures"] for unit in units) == 910
    assert {row["unit"] for unit in units for row in unit["rows"]} == {
        row["unit"] for row in rows}
    assert sum(len(unit["rows"]) for unit in units) == 100
    assert all(len({row["pkl"] for row in unit["rows"]}) == 1 for unit in units)

    original_sha = campaign._sha256

    def no_production_pkl_read(path):
        assert not str(path).endswith(".pkl.gz")
        return original_sha(path)

    monkeypatch.setattr(campaign, "_sha256", no_production_pkl_read)
    output_root = campaign._repo_root / "histos" / "test_prepare_only_uncreated"
    assert not output_root.exists()
    destination = tmp_path / "dispatch"
    result = campaign.prepare(
        _accepted_manifest, destination, output_root, request_cpus=3,
        request_memory="8GB", request_disk="4GB", max_runtime=3600, workers=2,
    )
    assert result["submitted"] is False
    assert (result["rows"], result["units"], result["expected_figures"]) == (100, 10, 910)
    assert not output_root.exists()
    items = (destination / "items.txt").read_text().splitlines()
    assert len(items) == 10
    descriptors = [json.loads(path.read_text()) for path in sorted((destination / "units").glob("*.json"))]
    assert len(descriptors) == 10
    input_record = json.loads((destination / "campaign_inputs.json").read_text())
    assert set(input_record["campaign_principal_pkl_paths"]) == manifest_paths
    assert set(input_record["rows_by_principal_pkl"]) == manifest_paths
    assert {descriptor["principal_pkl"] for descriptor in descriptors} == manifest_paths
    assert all(set(descriptor["campaign_principal_pkl_paths"]) == manifest_paths
               for descriptor in descriptors)
    for item, descriptor in zip(sorted(items), sorted(descriptors, key=lambda value: value["unit_id"])):
        unit_id, digest = item.split()
        path = destination / "units" / (unit_id + ".json")
        assert original_sha(path) == digest
        assert descriptor["unit_id"] == unit_id
        assert descriptor["principal_pkl"].startswith("/groups/klannon/apiccine/")
        assert descriptor["output_root"] == str(output_root)
        assert len(descriptor["row_ids"]) == len(descriptor["rows"])
    submit = (destination / "plots.sub").read_text()
    assert submit.count("universe = vanilla") == 1
    assert "should_transfer_files = NO" in submit
    assert "request_cpus = 3" in submit
    assert "request_memory = 8GB" in submit
    assert "request_disk = 4GB" in submit
    assert "allowed_execute_duration = 3600" in submit
    assert "queue unit_id, unit_sha256 from " in submit
    assert "condor_submit" not in submit


def test_one_unit_prepare_keeps_whole_campaign_input_set(tmp_path):
    rows, manifest_hash = campaign.read_campaign(_accepted_manifest)
    units = campaign.derive_units(rows, manifest_hash)
    chosen = next(unit for unit in units if unit["unit_id"] == "pkl_60965b09ae555381")
    destination = tmp_path / "pilot"
    result = campaign.prepare(
        _accepted_manifest, destination,
        campaign._repo_root / "histos" / "test_pilot_prepare_only_uncreated",
        request_cpus=1, request_memory="1GB", request_disk="1GB",
        max_runtime=60, workers=1, unit_id=chosen["unit_id"],
    )
    assert result["units"] == 1
    assert result["queued_rows"] == len(chosen["rows"])
    assert result["expected_figures"] == chosen["expected_figures"]
    assert len((destination / "items.txt").read_text().splitlines()) == 1
    descriptor = json.loads((destination / "units" / (chosen["unit_id"] + ".json")).read_text())
    assert descriptor["principal_pkl"] == chosen["principal_pkl"]
    assert len(descriptor["campaign_principal_pkl_paths"]) == 10
    assert len(json.loads((destination / "campaign_inputs.json").read_text())[
        "campaign_principal_pkl_paths"]) == 10
    assert "should_transfer_files = NO" in (destination / "plots.sub").read_text()


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("run", ["run2", "run3"])
def test_run_unit_selects_once_and_reuses_all_views(tmp_path, monkeypatch, workers, run):
    manifest = _fixture_manifest(tmp_path, run=run)
    rows, manifest_hash = campaign.read_campaign(manifest)
    unit = campaign.derive_units(rows, manifest_hash)[0]
    output_root = tmp_path / f"histos_{workers}" / "campaign"
    monkeypatch.setattr(campaign, "_histos_root", lambda path: str(path))
    monkeypatch.setattr(campaign, "_source_identity", lambda: {"head": "fixture"})
    descriptor = {
        "schema": "plot_campaign_unit_v1", "campaign_id": "campaign",
        "campaign_manifest": str(manifest), "output_root": str(output_root),
        "source_identity": {"head": "prepared, different from runtime",
                            "frontend_sha256": "0" * 64,
                            "plotter_sha256": "f" * 64},
        "campaign_principal_pkl_paths": [unit["principal_pkl"]],
        "workers": workers, **unit,
    }
    descriptor_path = tmp_path / f"unit_{workers}.json"
    descriptor_path.write_bytes(campaign._json_bytes(descriptor))
    Path(unit["sidecar"]).write_text("prepare-time sidecar provenance changed")
    loader_calls = []
    rendered = []

    def selected_loader(paths, **kwargs):
        loader_calls.append((paths, kwargs))
        assert len(loader_calls) == 1
        assert kwargs["selected_families"] == ["njets", "ptz"]
        def make_histogram(variable, weight):
            histogram = plotter.SparseHist(
                hist.axis.StrCategory([], name="process", growth=True),
                hist.axis.StrCategory([], name="channel", growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(2, 0, 2, name=variable),
            )
            histogram.fill(process="ttH_central2022", channel="2lss_p",
                           systematic="nominal", **{variable: 0.5}, weight=weight)
            return histogram
        return {"njets__scalar_nominal": make_histogram("njets", 11),
                "njets_sumw2": make_histogram("njets", 1),
                "ptz__scalar_nominal": make_histogram("ptz", 22),
                "ptz_sumw2": make_histogram("ptz", 2)}, {
            "schema": "split_sibling_v1", "runtime_histogram_families": ["njets", "ptz"]}

    def render(args, parser, *, loaded_view, outcome_collector,
               negative_report_suffix):
        assert len(loader_calls) == 1
        family = args.variables[0]
        assert negative_report_suffix == (next(
            row["unit"] for row in rows if row["variables"] == family)
            if run == "run2" else None)
        mapping, report = loaded_view
        assert set(mapping) == {family + "__scalar_nominal", family + "_sumw2"}
        assert report["runtime_histogram_families"] == [family]
        rendered.append((family, float(ak.sum(mapping[family + "__scalar_nominal"].values())),
                         args.workers))
        path = Path(args.output_path) / args.output_name
        path.mkdir(parents=True)
        Image.new("RGB", (1, 1)).save(path / (family + ".png"))
        outcome_collector["planned"].append(family + ".png")
        return 0

    monkeypatch.setattr(plotter, "load_and_merge_histogram_pkls", selected_loader)
    monkeypatch.setattr(plotter, "run_with_args", render)
    receipt = campaign.run_unit(descriptor_path, campaign._sha256(descriptor_path))
    assert len(loader_calls) == 1
    assert rendered == [("njets", 11.0, workers), ("ptz", 22.0, workers)]
    assert receipt["logical_view_count"] == 2
    assert receipt["expected_output_count"] == receipt["observed_output_count"] == 2
    assert receipt["planned_slot_count"] == receipt["produced_png_count"] == 2
    assert receipt["allowed_skip_count"] == receipt["unaccounted_count"] == 0
    assert receipt["zero_byte_or_unreadable_output_count"] == 0
    assert receipt["return_code"] == 0
    assert receipt["topeft_source_identity"] != receipt["runtime_topeft_source_identity"]
    for row in rows:
        relative = campaign._output_relative(row)
        assert (output_root / relative / (row["variables"] + ".png")).is_file()
        if run == "run2":
            assert "/owner/" not in relative
    with pytest.raises(FileExistsError, match="Occupied campaign output"):
        campaign.run_unit(descriptor_path, campaign._sha256(descriptor_path))
    assert len(loader_calls) == 1


def test_missing_declared_principal_pkl_fails_without_substitution(tmp_path, monkeypatch):
    manifest = _fixture_manifest(tmp_path)
    rows, manifest_hash = campaign.read_campaign(manifest)
    unit = campaign.derive_units(rows, manifest_hash)[0]
    descriptor = {
        "schema": "plot_campaign_unit_v1", "campaign_id": "campaign",
        "campaign_manifest": str(manifest),
        "output_root": str(tmp_path / "histos" / "campaign"),
        "source_identity": {"head": "old"},
        "campaign_principal_pkl_paths": [unit["principal_pkl"]],
        "workers": 1, **unit,
    }
    descriptor_path = tmp_path / "unit.json"
    descriptor_path.write_bytes(campaign._json_bytes(descriptor))
    Path(unit["principal_pkl"]).unlink()
    (tmp_path / "replacement.pkl.gz").write_bytes(b"not a campaign member")
    monkeypatch.setattr(campaign, "_histos_root", lambda path: str(path))
    monkeypatch.setattr(plotter, "load_and_merge_histogram_pkls",
                        lambda *args, **kwargs: pytest.fail("loader must not be called"))
    with pytest.raises(FileNotFoundError, match="Declared principal PKL"):
        campaign.run_unit(descriptor_path, campaign._sha256(descriptor_path))
    assert not Path(descriptor["output_root"]).exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_structured_maintained_skip_reconciles_with_drawable_png(tmp_path, workers):
    def histogram(variable, data_weight):
        output = plotter.SparseHist(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.Regular(2, 0, 2, name=variable),
        )
        for process, weight in (("ttH_central2022", 5.0),
                                ("data2022", data_weight)):
            output.fill(process=process, channel="2lss_p", systematic="nominal",
                        **{variable: 0.5}, weight=weight)
        return output

    outcomes = {"planned": [], "skipped": []}
    plotter.run_plots_for_region(
        "SR", {"njets": histogram("njets", 0.0),
               "ptz": histogram("ptz", 7.0)},
        years=["2022"], save_dir_path=str(tmp_path), variables=["njets", "ptz"],
        channel_output="merged", skip_syst_errs=True, unblind=True,
        negative_weight_report=False, outcome_collector=outcomes, workers=workers,
    )
    assert outcomes["skipped"]
    assert all(record["reason"] == "empty-data-content"
               for record in outcomes["skipped"])
    assert all(record["variable"] == "njets" for record in outcomes["skipped"])
    assert any(path.endswith("_ptz_processing.png") for path in outcomes["planned"])
    inventory, invalid = campaign._inventory(tmp_path, outcomes["planned"])
    skipped = [{**record, "row_id": "synthetic_row"}
               for record in outcomes["skipped"]]
    accounting = campaign._reconcile_outputs(
        {"expected_figures": len(outcomes["planned"])},
        outcomes["planned"], skipped, inventory, invalid,
    )
    assert accounting["produced_png_count"] > 0
    assert accounting["allowed_skip_count"] > 0
    assert accounting["unaccounted_count"] == 0


def test_output_reconciliation_fails_closed(tmp_path):
    planned = ["row/one.png", "row/two.png"]
    row = tmp_path / "row"
    row.mkdir()
    Image.new("RGB", (1, 1)).save(row / "one.png")
    descriptor = {"expected_figures": 2}
    skip = {"outcome": "skipped", "reason": "empty-data-content",
            "region": "SR", "category": "category", "variable": "njets",
            "mode": "aggregate", "row_id": "row", "has_mc": True,
            "has_data_like": False, "mc_empty": False, "data_empty": True,
            "paths": [planned[1]]}

    def reconcile(skips):
        inventory, invalid = campaign._inventory(tmp_path, planned)
        return campaign._reconcile_outputs(
            descriptor, planned, skips, inventory, invalid,
        )

    with pytest.raises(RuntimeError, match="unaccounted"):
        reconcile([])
    with pytest.raises(RuntimeError, match="unexpected_skips"):
        reconcile([{**skip, "paths": ["row/foreign.png"]}])
    with pytest.raises(RuntimeError, match="duplicate_skips=\\['row/two.png'\\]"):
        reconcile([skip, skip])
    with pytest.raises(RuntimeError, match="conflicting_outcomes=\\['row/one.png'\\]"):
        reconcile([{**skip, "paths": [planned[0]]}])
    with pytest.raises(RuntimeError, match="Invalid maintained"):
        reconcile([{**skip, "reason": "arbitrary"}])
    (row / "two.png").write_bytes(b"")
    with pytest.raises(RuntimeError, match="invalid_png=1"):
        reconcile([])
    (row / "two.png").write_bytes(b"not a PNG")
    with pytest.raises(RuntimeError, match="invalid_png=1"):
        reconcile([])


def test_mixed_unit_njets_view_matches_selective_channel_semantics():
    from topcoffea.modules.utils import regex_match

    selected_channel = "2lss_p"
    extra_channel = "unrelated_extra_channel"
    patterns = plotter._sr_njets_channel_patterns()
    assert regex_match([selected_channel], patterns) == [selected_channel]
    assert not regex_match([extra_channel], patterns)

    def make_histogram(variable, offset):
        output = plotter.SparseHist(
            hist.axis.StrCategory([], name="process", growth=True),
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.Regular(2, 0, 2, name=variable),
        )
        for channel, factor in ((selected_channel, 1), (extra_channel, 100)):
            for process, process_factor in (("ttH_central2022", 2),
                                            ("data2022", 3),
                                            ("nonprompt2022", 5),
                                            ("flips2022", 7)):
                for systematic, syst_factor in (("nominal", 1),
                                                ("JESUp", 1.2),
                                                ("JESDown", 0.8)):
                    output.fill(process=process, channel=channel,
                                systematic=systematic, **{variable: 0.5},
                                weight=(offset + process_factor) * factor * syst_factor)
        return output

    retained = {
        "njets__scalar_nominal": make_histogram("njets", 1),
        "njets_sumw2": make_histogram("njets", 9),
        "ptz__scalar_nominal": make_histogram("ptz", 3),
        "ptz_sumw2": make_histogram("ptz", 11),
    }
    report = {"runtime_histogram_families": ["njets", "ptz"]}
    njets_row = {"kind": "njets", "variables": "njets"}
    other_row = {"kind": "other", "variables": "ptz"}
    original = {key: ak.to_list(histogram.values(flow=True))
                for key, histogram in retained.items()}
    reference = {
        key: histogram.prune("channel", regex_match(list(histogram.axes["channel"]), patterns))
        for key, histogram in retained.items() if key.startswith("njets")
    }

    for order in ((njets_row, other_row), (other_row, njets_row)):
        for row in order:
            view, view_report = campaign._selected_view(retained, report, row)
            assert view_report["runtime_histogram_families"] == [row["variables"]]
            assert set(view) == set(reference if row["kind"] == "njets" else
                                    {key: value for key, value in retained.items()
                                     if key.startswith("ptz")})
            for key, histogram in view.items():
                expected = reference[key] if row["kind"] == "njets" else retained[key]
                assert list(histogram.axes["channel"]) == list(expected.axes["channel"])
                assert list(histogram.axes["process"]) == list(expected.axes["process"])
                assert list(histogram.axes["systematic"]) == list(expected.axes["systematic"])
                assert ak.to_list(histogram.values(flow=True)) == ak.to_list(expected.values(flow=True))
                if row["kind"] == "njets":
                    assert list(histogram.axes["channel"]) == [selected_channel]
            if row["kind"] == "njets":
                reference_context = plotter.build_region_context(
                    "SR", reference, ["2022"], unblind=True,
                    channel_mode_override="aggregate")
                candidate_context = plotter.build_region_context(
                    "SR", view, ["2022"], unblind=True,
                    channel_mode_override="aggregate")
                assert candidate_context.all_samples == reference_context.all_samples
                assert candidate_context.mc_samples == reference_context.mc_samples
                assert candidate_context.data_samples == reference_context.data_samples
                assert candidate_context.channel_map == reference_context.channel_map
                assert set(candidate_context.sumw2_hists) == set(reference_context.sumw2_hists)
        for key, histogram in retained.items():
            assert ak.to_list(histogram.values(flow=True)) == original[key]
            assert extra_channel in histogram.axes["channel"]


def test_render_worker_path_has_no_pkl_loader_call():
    for function in (plotter._initialize_render_worker,
                     plotter._render_variable_from_worker, plotter.produce_region_plots):
        source = inspect.getsource(function)
        assert "load_and_merge_histogram_pkls(" not in source
        assert "iterate_hist_from_pkl(" not in source
