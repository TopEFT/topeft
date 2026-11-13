"""Single-source nominal EFT scans using the maintained fill observer.

Direct CLI and execution manifests share the same immutable half-open range
contract. This module owns acquisition/coverage, never selections or weights.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import time
import traceback
from contextlib import redirect_stdout

from analysis.topeft_run2 import eft_event_diagnostics as native
from analysis.topeft_run2 import plot_campaign_backend as durable

require = native.require


def entry_range(start=0, stop=None, tree_entries=None):
    require(type(start) is int and start >= 0, "entry_start must be an integer >= 0")
    require(stop is None or type(stop) is int and stop > start,
            "entry_stop must be an integer > entry_start")
    if tree_entries is not None:
        require(type(tree_entries) is int and 0 <= start < tree_entries,
                "empty or beyond-tree entry range")
        stop = tree_entries if stop is None else stop
        require(stop <= tree_entries, "entry_stop beyond tree")
    return start, stop


def source_identity(path, tree_name="Events"):
    """Metadata-only read; no event arrays and no remote source support."""
    import uproot
    path = Path(path).resolve(strict=True)
    before = path.stat()
    with uproot.open(str(path)) as root:
        entries, uuid = int(root[tree_name].num_entries), str(root.file.uuid)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "source changed during identity read")
    # Mount device/inode observations are node-local. Use them above to detect
    # replacement during a read, never as the cross-node scientific identity.
    return dict(path=str(path), tree_name=tree_name, tree_entries=entries, root_uuid=uuid,
                size=after.st_size, mtime_ns=after.st_mtime_ns)


def validate_inputs(contract):
    require(contract.get("schema_version") == 1, "unsupported target contract")
    bindings = contract["input_sha256"]
    required = [str(Path(__file__).resolve()), str(Path(native.__file__).resolve()),
                str(native.PROCESSOR), contract["sr_config_path"]]
    require(all(p in bindings for p in required), "missing code/config binding")
    for path, digest in bindings.items():
        require(Path(path).is_absolute() and native.sha256(path) == digest,
                "input identity changed: " + path)
    for calibration in contract["calibrations"]:
        require(bindings.get(calibration["path"]) == calibration["sha256"],
                "calibration not bound")
    owners = contract["ownership"]
    group = contract["group"]
    require(bool(owners) and all(r["group_id"] == group["group_id"] for r in owners),
            "missing or mixed target ownership")
    keys = [(r["negative_row_id"], r["source_channel"], r["fit_variable"],
             int(r["source_bin"])) for r in owners]
    require(len(keys) == len(set(keys)), "duplicate target membership")


def make_unit(source_root, sample_json, analysis_config, target_manifest,
              entry_start=0, entry_stop=None):
    entry_range(entry_start, entry_stop)
    target_manifest = Path(target_manifest).resolve(strict=True)
    contract = json.loads(target_manifest.read_text())
    validate_inputs(contract)
    sample_json, analysis_config = (str(Path(p).resolve(strict=True))
                                    for p in (sample_json, analysis_config))
    require(all(p in contract["input_sha256"] for p in (sample_json, analysis_config)),
            "unbound sample/config")
    source = source_identity(source_root, contract["group"]["tree_name"])
    entry_range(entry_start, entry_stop, source["tree_entries"])
    body = dict(schema_version=1, source_identity=source, sample_json=sample_json,
                analysis_config=analysis_config, target_manifest=str(target_manifest),
                target_manifest_sha256=native.sha256(target_manifest),
                entry_start=entry_start, entry_stop=entry_stop)
    return dict(body, unit_id="unit_" + durable.identity(body)[:24])


def validate_unit(unit, *, revalidate_source=False):
    require(unit.get("schema_version") == 1, "unsupported execution unit")
    body = {k: v for k, v in unit.items() if k != "unit_id"}
    require(unit.get("unit_id") == "unit_" + durable.identity(body)[:24],
            "execution unit identity changed")
    source = unit["source_identity"]
    require(Path(source["path"]).is_absolute(), "source must be local absolute path")
    entry_range(unit["entry_start"], unit["entry_stop"], source["tree_entries"])
    require(native.sha256(unit["target_manifest"]) == unit["target_manifest_sha256"],
            "target manifest changed")
    contract = json.loads(Path(unit["target_manifest"]).read_text())
    validate_inputs(contract)
    require(all(p in contract["input_sha256"] for p in
                (unit["sample_json"], unit["analysis_config"])), "unbound sample/config")
    if revalidate_source:
        require(source_identity(source["path"], source["tree_name"]) == source,
                "source identity changed")
    return contract


def select_unit(manifest, unit_id):
    data = json.loads(Path(manifest).read_text())
    require(data.get("schema_version") == 1 and isinstance(data.get("units"), list)
            and bool(data["units"]), "unsupported or empty execution manifest")
    ids = [unit["unit_id"] for unit in data["units"]]
    require(len(ids) == len(set(ids)), "duplicate execution unit")
    selected = [u for u in data["units"] if u["unit_id"] == unit_id]
    require(len(selected) == 1, "unknown execution unit")
    validate_unit(selected[0])
    return selected[0]


def coverage(receipts, current_source, start=0, stop=None):
    """Require exact complete coverage of the requested interval, not a count."""
    start, stop = entry_range(start, stop, current_source["tree_entries"])
    require(bool(receipts), "coverage gap: no receipts")
    sciences = {r["science_identity"] for r in receipts}
    require(len(sciences) == 1, "incompatible scientific contracts")
    intervals = []
    for receipt in receipts:
        require(receipt["source_identity"] == current_source, "source identity changed")
        require(receipt["status"] == "COMPLETE" and receipt["range_complete"],
                "incomplete range receipt")
        a, b = receipt["processed_range"]
        entry_range(a, b, current_source["tree_entries"])
        require(receipt["requested_range"] == [a, b], "requested/actual range contradiction")
        intervals.append((a, b))
    cursor = start
    for a, b in sorted(intervals):
        require(a >= cursor, "duplicate or overlapping coverage")
        require(a == cursor, "coverage gap")
        require(b <= stop, "coverage outside requested interval")
        cursor = b
    require(cursor == stop, "coverage gap at end")
    return dict(processed_range=[start, stop], range_complete=True,
                whole_file_complete=(start == 0 and stop == current_source["tree_entries"]))


def _processor(contract, sample):
    from topeft.modules.missing_parton_contract import load_missing_parton_channel_contract
    group, owners = contract["group"], contract["ownership"]
    require(sample["histAxisName"] == group["process"] and
            str(sample["year"]) == str(group["metadata_year"]), "sample identity mismatch")
    for row in owners:
        factors = row["term"].split("*")
        require(len(factors) == 2 and factors[0] == factors[1] and
                native.diagonal_index(sample["WCnames"], factors[0]) ==
                int(row["native_term_index"]), "native EFT term/order mismatch")
    channel_contract = load_missing_parton_channel_contract(config_path=contract["sr_config_path"])
    config = json.loads(Path(contract["sr_config_path"]).read_text())["ALL_CH_LST_SR"]
    selectors = native.selectors_for(owners)[group["group_id"]]
    bases = {channel_contract.base_channel(key[0]) for key in selectors}
    selected = {name: value for name, value in config.items()
                if any(row[0] in bases for row in value["lep_chan_lst"])}
    require(bases.issubset({row[0] for v in selected.values() for row in v["lep_chan_lst"]}),
            "unbound source category")
    module, instrumented_hash = native.observed_processor(
        lambda _: None, contract["input_sha256"][str(native.PROCESSOR)])
    processor = module.AnalysisProcessor({group["dataset"]: sample}, list(sample["WCnames"]),
        hist_lst=sorted({key[1] for key in selectors}), fill_sumw2_hist=False,
        do_systematics=False, skip_signal_regions=False, skip_control_regions=True,
        all_analysis=True, useRun3MVA=True, sr_category_dict=selected)
    applications = {key[0]: channel_contract.expected_sr_appl(key[0]) for key in selectors}
    return module, processor, selectors, applications, instrumented_hash


def _events(unit, begin, end, dataset):
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
    source = unit["source_identity"]
    return NanoEventsFactory.from_root(source["path"], treepath="/" + source["tree_name"],
        entry_start=begin, entry_stop=end, schemaclass=NanoAODSchema,
        metadata={"dataset": dataset}, uproot_options={"timeout": 120, "num_workers": 1}).events()


def run_unit(unit, output_dir, *, chunk_entries=10000, time_limit=3600,
             max_output_bytes=268435456, reserve_bytes=1073741824):
    """One worker for both backends; no retries, overwrite or inferred completion."""
    require(type(chunk_entries) is int and chunk_entries > 0 and
            0 < time_limit <= 108000 and max_output_bytes > 0 and reserve_bytes >= 0,
            "invalid execution bound")
    contract = validate_unit(unit, revalidate_source=True)
    source = unit["source_identity"]
    begin, stop = entry_range(unit["entry_start"], unit["entry_stop"], source["tree_entries"])
    sample = json.loads(Path(unit["sample_json"]).read_text())
    require(any(source["path"].endswith(p) for p in sample["files"]),
            "source not owned by sample")
    output = Path(output_dir).resolve()
    require(not output.exists() and not output.is_symlink(), "output collision; no retry/resume")
    require(os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize >= reserve_bytes,
            "insufficient output capacity")
    opened = native.install_io_guard([r["path"] for r in contract["calibrations"]])
    output.mkdir(mode=0o700)
    started, cursor, chunks, error = time.monotonic(), begin, [], None
    capture, instrumented_hash = None, None
    with open(output / "runtime.log", "x") as log, redirect_stdout(log):
        try:
            module, processor, selectors, applications, instrumented_hash = _processor(contract, sample)
            group = contract["group"]
            context = dict(context_id=unit["unit_id"], group_id=group["group_id"],
                local_root_path=source["path"], process=group["process"], year_or_era=group["year_or_era"])
            with gzip.open(output / "events.tsv.gz", "xt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=native.EVENT_FIELDS, delimiter="\t")
                writer.writeheader()
                capture = native.fill_capture(context, selectors, applications, writer)
                module._diagnostic_capture = capture
                for a in range(begin, stop, chunk_entries):
                    require(time.monotonic() - started < time_limit, "time boundary; no retry")
                    require(os.statvfs(output).f_bavail * os.statvfs(output).f_frsize >= reserve_bytes
                            and (output / "events.tsv.gz").stat().st_size < max_output_bytes,
                            "storage boundary; preserve partial output")
                    b = min(stop, a + chunk_entries)
                    capture.entry_start = a
                    events = _events(unit, a, b, group["dataset"])
                    require(len(events) == b - a, "incomplete event read")
                    processor.process(events)
                    del events
                    stream.flush()
                    cursor = b
                    chunks.append([a, b])
            require(source_identity(source["path"], source["tree_name"]) == source,
                    "source identity changed during scan")
            validate_inputs(contract)
            require(time.monotonic() - started <= time_limit, "time boundary; no retry")
            require((output / "events.tsv.gz").stat().st_size <= max_output_bytes,
                    "output size boundary; preserve partial output")
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    rows = [] if capture is None else [
        dict(negative_row_id=rid, partial_qii=capture.sums[rid].value(),
             sign_counts=dict(capture.signs[rid]))
        for rid in sorted({r["negative_row_id"] for r in contract["ownership"]})]
    durable.exclusive_json(output / "sums.json", rows)
    for name in ("events.tsv.gz", "sums.json"):
        if (output / name).is_file():
            with (output / name).open("rb") as stream:
                os.fsync(stream.fileno())
    complete = error is None and cursor == stop
    receipt = dict(schema_version=1, unit_id=unit["unit_id"],
        status="COMPLETE" if complete else "STOPPED_NO_RETRY",
        source_identity=source, requested_range=[begin, stop],
        requested_entry_stop=unit["entry_stop"], processed_range=[begin, cursor],
        completed_chunks=chunks, range_complete=complete,
        whole_file_complete=complete and begin == 0 and stop == source["tree_entries"],
        sample=contract["group"], target_manifest=unit["target_manifest"],
        target_manifest_sha256=unit["target_manifest_sha256"], targets=contract["ownership"],
        input_sha256=contract["input_sha256"], instrumented_source_sha256=instrumented_hash,
        science_identity=durable.identity(dict(target=unit["target_manifest_sha256"],
                                             inputs=contract["input_sha256"])),
        selected_fill_rows=0 if capture is None else capture.rows,
        calibration_open_counts=dict(opened), error=error, retries=0,
        elapsed_seconds=time.monotonic()-started,
        artifact_sha256={name: native.sha256(output/name) for name in
                         ("events.tsv.gz", "sums.json") if (output/name).is_file()})
    durable.exclusive_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    direct = actions.add_parser("scan")
    for name in ("source-root", "sample-json", "analysis-config", "target-manifest"):
        direct.add_argument("--"+name, type=Path, required=True)
    direct.add_argument("--entry-start", type=int, default=0)
    direct.add_argument("--entry-stop", type=int)
    run = actions.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--unit-id", required=True)
    for command in (direct, run):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--chunk-entries", type=int, default=10000)
        command.add_argument("--time-limit", type=int, default=3600)
    merge = actions.add_parser("coverage")
    merge.add_argument("--receipt", type=Path, action="append", required=True)
    merge.add_argument("--entry-start", type=int, default=0)
    merge.add_argument("--entry-stop", type=int)
    args = parser.parse_args(argv)
    if args.action == "coverage":
        receipts = [json.loads(p.read_text()) for p in args.receipt]
        for path, receipt in zip(args.receipt, receipts):
            require(all(native.sha256(path.parent/name) == digest
                        for name, digest in receipt["artifact_sha256"].items()), "receipt artifact changed")
        source = receipts[0]["source_identity"]
        result = coverage(receipts, source_identity(source["path"], source["tree_name"]),
                          args.entry_start, args.entry_stop)
        print(json.dumps(result, sort_keys=True))
        return 0
    unit = (make_unit(args.source_root, args.sample_json, args.analysis_config,
                      args.target_manifest, args.entry_start, args.entry_stop)
            if args.action == "scan" else select_unit(args.manifest, args.unit_id))
    receipt = run_unit(unit, args.output_dir, chunk_entries=args.chunk_entries, time_limit=args.time_limit)
    print(json.dumps(dict(status=receipt["status"], unit_id=unit["unit_id"],
                         output_dir=str(args.output_dir.resolve()))))
    return 0 if receipt["range_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
