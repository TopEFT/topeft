import hashlib
import inspect
import json
from pathlib import Path

import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots as plotter
from analysis.topeft_run2 import plot_campaign_condor as campaign


_accepted_manifest = (
    Path(__file__).resolve().parents[2]
    / "reports/diagnostics/SR_BASELINE_ALL_SR_MANUAL_HANDOFF_028/coverage_input_manifest.tsv"
)


def _fixture_manifest(tmp_path):
    pkl = tmp_path / "principal_SR.pkl.gz"
    pkl.write_bytes(b"inert input: never deserialize")
    pkl_digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
    sidecar = Path(str(pkl) + ".metadata.json")
    sidecar.write_text(json.dumps({"artifact": {"pkl_sha256": pkl_digest}}))
    sidecar_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest = tmp_path / "campaign.tsv"
    header = "\t".join(campaign._header)
    rows = [
        ("one_njets", "njets", "run3", "combined", "owner", "2022", "njets", "1"),
        ("two_other", "other", "run3", "2022", "owner", "2022", "ptz", "1"),
    ]
    manifest.write_text(header + "\n" + "\n".join("\t".join((
        *row, str(pkl), pkl_digest, str(sidecar), sidecar_digest,
    )) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_accepted_manifest_partitions_exactly_and_prepare_never_opens_pkls(
        tmp_path, monkeypatch):
    rows, manifest_hash = campaign.read_campaign(_accepted_manifest)
    units = campaign.derive_units(rows, manifest_hash)
    assert len(rows) == 100
    assert sum(row["kind"] == "njets" for row in rows) == 50
    assert sum(row["kind"] == "other" for row in rows) == 50
    assert len(units) == 10
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


@pytest.mark.parametrize("workers", [1, 3])
def test_run_unit_selects_once_and_reuses_all_views(tmp_path, monkeypatch, workers):
    manifest = _fixture_manifest(tmp_path)
    rows, manifest_hash = campaign.read_campaign(manifest)
    unit = campaign.derive_units(rows, manifest_hash)[0]
    output_root = tmp_path / f"histos_{workers}" / "campaign"
    monkeypatch.setattr(campaign, "_histos_root", lambda path: str(path))
    monkeypatch.setattr(campaign, "_source_identity", lambda: {"head": "fixture"})
    descriptor = {
        "schema": "plot_campaign_unit_v1", "campaign_id": "campaign",
        "campaign_manifest": str(manifest), "output_root": str(output_root),
        "source_identity": {"head": "fixture"}, "workers": workers, **unit,
    }
    descriptor_path = tmp_path / f"unit_{workers}.json"
    descriptor_path.write_bytes(campaign._json_bytes(descriptor))
    loader_calls = []
    rendered = []

    def selected_loader(paths, **kwargs):
        loader_calls.append((paths, kwargs))
        assert len(loader_calls) == 1
        assert kwargs["selected_families"] == ["njets", "ptz"]
        return {"njets__scalar_nominal": 11, "njets_sumw2": 1,
                "ptz__scalar_nominal": 22, "ptz_sumw2": 2}, {
            "schema": "split_sibling_v1", "runtime_histogram_families": ["njets", "ptz"]}

    def render(args, parser, *, loaded_view):
        assert len(loader_calls) == 1
        family = args.variables[0]
        mapping, report = loaded_view
        assert set(mapping) == {family + "__scalar_nominal", family + "_sumw2"}
        assert report["runtime_histogram_families"] == [family]
        rendered.append((family, mapping[family + "__scalar_nominal"], args.workers))
        path = Path(args.output_path) / args.output_name
        path.mkdir(parents=True)
        (path / (family + ".png")).write_bytes(b"PNG fixture")
        return 0

    monkeypatch.setattr(plotter, "load_and_merge_histogram_pkls", selected_loader)
    monkeypatch.setattr(plotter, "run_with_args", render)
    receipt = campaign.run_unit(descriptor_path, campaign._sha256(descriptor_path))
    assert len(loader_calls) == 1
    assert rendered == [("njets", 11, workers), ("ptz", 22, workers)]
    assert receipt["logical_view_count"] == 2
    assert receipt["expected_output_count"] == receipt["observed_output_count"] == 2
    assert receipt["zero_byte_or_unreadable_output_count"] == 0
    assert receipt["return_code"] == 0
    with pytest.raises(FileExistsError, match="Occupied campaign output"):
        campaign.run_unit(descriptor_path, campaign._sha256(descriptor_path))
    assert len(loader_calls) == 1


def test_render_worker_path_has_no_pkl_loader_call():
    for function in (plotter._initialize_render_worker,
                     plotter._render_variable_from_worker, plotter.produce_region_plots):
        source = inspect.getsource(function)
        assert "load_and_merge_histogram_pkls(" not in source
        assert "iterate_hist_from_pkl(" not in source
