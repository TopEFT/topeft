"""Local, nominal EFT event diagnostics using the maintained processor.

Observe its fill boundary rather than reproducing selections or weights. See
docs/reference/eft_event_diagnostics.md for the input, provenance and failure
contracts. This module never loads analysis pickles or writes histograms.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
PROCESSOR = Path(__file__).with_name("analysis_processor.py")
EVENT_FIELDS = (
    "fill_id context_id group_id source_file entry run lumi event process era "
    "channel variable source_bin value w_analysis weight_dtype coefficient_dtype "
    "p0 q0 indices pii qii sign_w sign_pii sign_qii ordinary_factors"
).split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def dump_yaml(path, value):
    with open(path, "x") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def table_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def write_table(path, rows, fields):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sign(value):
    require(math.isfinite(value), "Nonfinite event factor")
    return int(value > 0) - int(value < 0)


def diagonal_index(wc_names, operator):
    """Resolve the actual source WC order through the native packed helper."""
    import numpy as np
    from topcoffea.modules.eft_helper import quadratic_factors_to_term

    require(len(wc_names) == len(set(wc_names)), "Duplicate WC names")
    factor = wc_names.index(operator) + 1
    return int(quadratic_factors_to_term(np.asarray([factor, factor])))


def combined_producer_views(components, family, years, channels):
    """Transport origin tags through the native combined process universe.

    Zero-valued tags retain every combined category AND sparse key. Removing
    the other origin's categories changes native group ownership (notably
    Run3 ttll); isolated component makers cannot define physical S/E0.
    Returned histograms are in-memory evidence, never written as payloads.
    """
    import copy
    import numpy as np
    from topeft.modules.datacard_tools import DatacardMaker
    from topeft.modules.nominal_schema import materialize_nominal_family

    require(bool(channels) and len(channels) == len(set(channels)), "Explicit unique channels required")
    full = materialize_nominal_family(components, family)
    require(not full.empty(), "Empty combined family")
    full_keys = set(full.view(flow=True))
    tagged = {"combined": full}
    origin_keys = {}
    for origin in ("scalar", "eft"):
        key = f"{family}__{origin}_nominal"
        tag = copy.deepcopy(full)
        tag.reset()
        if key in components:
            component = materialize_nominal_family(
                {key: components[key]}, family, wc_names=full.wc_names)
            views = component.view(flow=True)
            require(set(views) <= full_keys, "Origin support outside combined family")
            origin_keys[origin] = [dict(k._asdict()) for k in views]
            destination = tag.view(flow=True)
            for coordinate, values in views.items():
                require(values.shape == destination[coordinate].shape, "Origin shape mismatch")
                destination[coordinate][...] = values
        else:
            origin_keys[origin] = []
        require(set(tag.view(flow=True)) == full_keys, "Origin tag lost combined sparse support")
        require(tuple(tag.axes['process']) == tuple(full.axes['process']), "Origin tag lost process order")
        tagged[origin] = tag
    for key, values in full.view(flow=True).items():
        require(np.allclose(values, tagged['scalar'].view(flow=True)[key] +
                            tagged['eft'].view(flow=True)[key], rtol=1e-10, atol=1e-10),
                "Source origin tags do not reconstruct combined coefficients")

    class period_observer(DatacardMaker):
        def read(self, fpath=None, hists=None):
            self.diagnostic_periods = {}
            for name in self._prepare_source_periods(fpath=fpath, hists=hists):
                self.diagnostic_periods[name] = self.hists[name]
                self.hists[name] = self.correlate_years(self.hists[name], add_luminosity_shapes=False)

    result = {"wc_names": list(full.wc_names), "origin_source_keys": origin_keys,
              "family": family, "views": {}}
    for origin, source in tagged.items():
        # Native empty-family skipping is retained; an all-zero tag contributes zero.
        if source.empty():
            result['views'][origin] = None
            continue
        maker = period_observer(hists={family: source}, year_lst=list(years),
                                do_nuisance=False, var_lst=[family],
                                channel_patterns=['^' + c + '$' for c in channels],
                                verbose=False, binning_mode='fitting')
        result['views'][origin] = maker
    return result


def combined_bin_coefficients(bundle, channel, process, period=None):
    """Physical bins include the native overflow and exclude source underflow.

    Missing origin/period support is zero only relative to a present combined
    final process; missing combined support raises instead of inventing a bin.
    """
    import numpy as np

    family = bundle['family']
    arrays = {}
    for origin in ('combined', 'scalar', 'eft'):
        maker = bundle['views'][origin]
        arrays[origin] = None
        if maker is None:
            continue
        histogram = (maker.hists if period is None else maker.diagnostic_periods)[family]
        label = process if period is None else f'{process}_{period}'
        selected = histogram.integrate('channel', [channel])
        selected = maker.binning_view(selected, family, channel)
        labels = {maker._sparse_key_mapping(k).get('process') for k in selected.categorical_keys}
        if label not in labels:
            continue
        projection = maker._scaling_histogram_for_json(selected, channel, label)
        require(list(projection.wc_names) == bundle['wc_names'], 'WC order changed')
        arrays[origin] = np.asarray(projection.values(flow=True), dtype=float)[1:, 1:-1]
    require(arrays['combined'] is not None, 'Combined final process/period absent')
    full = arrays['combined']
    for origin in ('scalar', 'eft'):
        if arrays[origin] is None:
            arrays[origin] = np.zeros_like(full)
        require(arrays[origin].shape == full.shape, 'Origin final bin shape differs')
    require(np.isfinite(full).all(), 'Nonfinite combined coefficients')
    require(np.allclose(arrays['scalar'] + arrays['eft'], full, atol=1e-10, rtol=1e-10),
            'Combined origin transport does not close')
    require((arrays['scalar'][:, 1:] == 0).all(), 'Scalar nonconstant coefficient')
    return arrays


def package_bin_representation(signed_a0, nominal_value):
    """Describe representation only; sanitation does not redefine signed A0.

    None denotes a directly established absent template, never a failed read.
    A sanitation-compatible state needs package lineage before acceptance.
    """
    require(math.isfinite(signed_a0), 'Nonfinite signed A0')
    if nominal_value is None:
        return 'ABSENT_TEMPLATE_REQUIRES_LINEAGE'
    require(math.isfinite(nominal_value), 'Nonfinite ROOT nominal')
    if math.isclose(signed_a0, nominal_value, rel_tol=1e-10, abs_tol=1e-10):
        return 'SIGNED_A0_REPRESENTED'
    if signed_a0 < 0 and nominal_value == 0:
        return 'SANITIZED_ZERO_REQUIRES_LINEAGE'
    return 'REPRESENTATION_MISMATCH'


def event_products(coefficients, weights):
    """Preserve the flatten/repeat/multiply dtype semantics of HistEFT.fill."""
    import numpy as np

    coefficients, weights = np.asarray(coefficients), np.asarray(weights)
    require(coefficients.ndim == 2 and weights.ndim == 1, "Invalid factor shape")
    require(len(coefficients) == len(weights), "Unpaired event factors")
    require(np.isfinite(coefficients).all() and np.isfinite(weights).all(),
            "Nonfinite event factors")
    return (coefficients.ravel() * np.repeat(weights, coefficients.shape[1])).reshape(
        coefficients.shape
    )


def weight_product(factors):
    require(bool(factors), "Empty ordinary-weight factor record")
    require(all(math.isfinite(float(x)) for x in factors.values()), "Nonfinite factor")
    return math.prod(float(x) for x in factors.values())


def btag_decomposition(rows):
    """Reconstruct serialized Method-1a factors, including the native zero guard."""
    groups = defaultdict(list)
    for row in rows:
        em, el, sm, sl = (float(row[k]) for k in ("eM", "eL", "SFM", "SFL"))
        require(all(math.isfinite(x) for x in (em, el, sm, sl)), "Nonfinite BTV input")
        category = row["category"]
        require(category in {"MEDIUM", "LOOSE_NOT_MEDIUM", "NOT_LOOSE"},
                "Unknown Method-1a category")
        mc = em if category == "MEDIUM" else el - em if category == "LOOSE_NOT_MEDIUM" else 1 - el
        data = em * sm if category == "MEDIUM" else el * sl - em * sm if category == "LOOSE_NOT_MEDIUM" else 1 - el * sl
        groups[row["subgroup"]].append({"mc_factor": mc, "data_factor": data,
                                       "category": category})
    require(bool(groups), "Empty BTV decomposition")
    result = {}
    for name, jets in groups.items():
        raw_mc = math.prod(j["mc_factor"] for j in jets)
        mc = 1.0 if raw_mc == 0 else raw_mc
        data = math.prod(j["data_factor"] for j in jets)
        result[name] = {"raw_mc": raw_mc, "mc_after_guard": mc, "data": data,
                        "weight": data / mc, "jets": jets}
    return {"subgroups": result, "btagSF": math.prod(v["weight"] for v in result.values())}


def mechanism(counts, closed):
    """Classify the sources of negative qii, retaining all sign pairs separately."""
    if not closed:
        return "UNRESOLVED"
    ordinary = counts.get("-1,1,-1", 0) > 0
    coefficient = counts.get("1,-1,-1", 0) > 0
    if ordinary and coefficient:
        return "MIXED"
    if ordinary:
        return "ORDINARY_WEIGHT_ORIGIN"
    if coefficient:
        return "EFT_COEFFICIENT_ORIGIN"
    return "UNRESOLVED"


class compensated_sum:
    """Bounded-memory Neumaier accumulation for signed event contributions."""

    def __init__(self):
        self.total = self.correction = 0.0

    def add(self, value):
        summed = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - summed) + value
        else:
            self.correction += (value - summed) + self.total
        self.total = summed

    def value(self):
        return self.total + self.correction


class wide_packed_selection:
    """Compose native PackedSelection shards as in the accepted event observer."""

    def __init__(self, dtype="uint64"):
        from coffea.analysis_tools import PackedSelection

        self.factory, self.dtype = PackedSelection, dtype
        self.shards = [PackedSelection(dtype=dtype)]
        self.owner = {}

    @property
    def names(self):
        return list(self.owner)

    def add(self, name, selection, fill_value=False):
        require(name not in self.owner, "Duplicate selection name")
        shard = self.shards[-1]
        if len(shard.names) >= shard.maxitems:
            shard = self.factory(dtype=self.dtype)
            self.shards.append(shard)
        shard.add(name, selection, fill_value=fill_value)
        self.owner[name] = len(self.shards) - 1

    def combine(self, names, operation):
        import numpy as np

        require(bool(names), "An explicit nonempty selection is required")
        groups = defaultdict(list)
        for name in names:
            groups[self.owner[name]].append(name)
        arrays = [getattr(self.shards[i], operation)(*groups[i]) for i in sorted(groups)]
        reducer = np.logical_and if operation == "all" else np.logical_or
        return reducer.reduce(arrays)

    def all(self, *names):
        return self.combine(names, "all")

    def any(self, *names):
        return self.combine(names, "any")


def observed_processor(callback, expected_sha256):
    """Instrument only the observation point and terminal histogram accumulation.

    The on-disk source is never modified. No selection, correction or weight
    expression is copied into this diagnostic implementation.
    """
    require(sha256(PROCESSOR) == expected_sha256, "Processor identity changed")
    source = PROCESSOR.read_text()
    fill = "                                        # Fill the histos\n"
    terminal = "                                            nominal_histogram.fill(**axes_fill_info_dict)"
    selection = "PackedSelection(dtype='uint64')"
    require(source.count(fill) == source.count(terminal) == 1, "Fill boundary drift")
    require(source.count(selection) == 2, "Selection construction drift")
    source = source.replace(selection, "_diagnostic_selection(dtype='uint64')")
    source = source.replace(fill, "                                        _diagnostic_capture(locals())\n" + fill)
    source = source.replace(terminal, "                                            pass  # diagnostic owns the selected rows")
    name = "analysis.topeft_run2._eft_diagnostic_processor"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module._diagnostic_selection = wide_packed_selection
    module._diagnostic_capture = callback
    sys.modules[name] = module
    exec(compile(source, str(PROCESSOR), "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def install_io_guard(calibrations):
    """Block network and non-allowlisted pickle opens before loading runtime code."""
    allowed = {str(Path(p).resolve()) for p in calibrations}
    opened = Counter()

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise PermissionError("Network is forbidden in event diagnostics")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if ".pkl" in path.lower() or ".pickle" in path.lower():
                resolved = str(Path(path).resolve())
                mode = args[1] or ""
                flags = args[2] if len(args) > 2 else 0
                writing = any(x in mode for x in "wax+") or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                if resolved not in allowed or writing:
                    raise PermissionError("Forbidden pickle access: " + path)
                opened[resolved] += 1

    sys.addaudithook(audit)
    return opened


def source_context(manifest):
    """Reopen immutable ownership evidence and filter only by the bound worklist."""
    for path, digest in manifest["input_sha256"].items():
        require(sha256(path) == digest, "Input identity changed: " + path)
    targets = list(table_rows(manifest["worklist"]))
    require(sha256(manifest["worklist"]) == manifest["worklist_sha256"], "Worklist changed")
    ids = {r["negative_row_id"] for r in targets}
    require(len(ids) == len(targets), "Duplicate target identity")
    ownership = [r for r in table_rows(manifest["ownership"]) if r["negative_row_id"] in ids]
    require({r["negative_row_id"] for r in ownership} == ids, "Missing source ownership")
    group_ids = {r["group_id"] for r in ownership}
    groups = {g["group_id"]: g for g in load_yaml(manifest["groups"]) if g["group_id"] in group_ids}
    contexts = [c for c in table_rows(manifest["contexts"]) if c["group_id"] in group_ids]
    require({c["group_id"] for c in contexts} == group_ids, "Missing source context")
    require(len(contexts) == len({c["local_root_path"] for c in contexts}), "Duplicate physical source")
    return targets, ownership, groups, contexts


def discover(args):
    prior, output = args.prior_dir.resolve(), args.output_dir.resolve()
    binding = load_yaml(args.binding)
    require(binding["s0_gate"] == "PASS_WITH_OWNER_OVERRIDES", "Source binding not passed")
    old = prior / "class_b_22552_negative_term_era_worklist.tsv.gz"
    ownership = prior / "class_b_source_ownership.tsv.gz"
    contexts = prior / "class_b_source_context_worklist.tsv.gz"
    groups_path = prior / "class_b_source_groups.yaml"
    exclusion_path = prior / "selection_semantic_delta.yaml"
    accepted = load_yaml(prior / "class_b_context_deduplication_readback.yaml")
    for name, digest in accepted["output_sha256"].items():
        require(sha256(prior / name) == digest, "Prior worklist identity changed")
    year = list(table_rows(args.year_era))
    expected = {(r["target_id"], r["year_or_era"]): r for r in year if float(r["Eii_year_or_era"]) < 0}
    old_rows = list(table_rows(old))
    require(len(expected) == len(old_rows), "Negative population count differs")
    require(set(expected) == {(r["target_id"], r["year_or_era"]) for r in old_rows}, "Negative population differs")
    for row in old_rows:
        require(row["Eii_year_or_era"] == expected[row["target_id"], row["year_or_era"]]["Eii_year_or_era"],
                "Accepted target literal changed")
    excluded = [r for r in old_rows if r["source_channel"] == args.exclude_channel and r["fit_variable"] == args.exclude_variable]
    excluded_ids = {r["negative_row_id"] for r in excluded}
    require(excluded_ids == set(load_yaml(exclusion_path)["affected_ids"]), "Exclusion identity mismatch")
    require(excluded_ids == set(binding["accepted_evidence"]["excluded_row_ids"]), "Exclusion scope changed")
    retained = [r for r in old_rows if r["negative_row_id"] not in excluded_ids]
    require(retained, "Empty worklist")
    worklist = output / "class_b_current_fit_worklist.tsv.gz"
    excluded_path = output / "excluded_current_fit_rows.tsv.gz"
    write_table(worklist, retained, list(old_rows[0]))
    write_table(excluded_path, excluded, list(old_rows[0]))
    inputs = [args.year_era.resolve(), old, ownership, contexts, groups_path, exclusion_path,
              args.binding.resolve(), args.calibrations.resolve()]
    manifest = {"schema_version": 1, "worklist": str(worklist), "worklist_sha256": sha256(worklist),
                "excluded_manifest": str(excluded_path), "excluded_sha256": sha256(excluded_path),
                "ownership": str(ownership), "contexts": str(contexts), "groups": str(groups_path),
                "year_era": str(args.year_era.resolve()), "binding": str(args.binding.resolve()),
                "calibrations": str(args.calibrations.resolve()),
                "input_sha256": {str(p): sha256(p) for p in inputs},
                "absolute_tolerance": args.atol, "relative_tolerance": args.rtol,
                "tolerance_provenance": "predeclared before event reads; inherited 011 1e-10 default; negative sign additionally required"}
    targets, own, groups, ctx = source_context(manifest)
    ownership_keys = set()
    for row in own:
        group = groups[row["group_id"]]
        term = row["term"].split("*")
        require(len(term) == 2 and term[0] == term[1], "Non-diagonal term")
        require(diagonal_index(group["wc_names"], term[0]) == int(row["native_term_index"]), "Native index mismatch")
        key = tuple(row[k] for k in ("negative_row_id", "group_id", "source_channel", "fit_variable", "source_bin"))
        require(key not in ownership_keys, "Duplicate target/source-bin membership")
        ownership_keys.add(key)
    for group in groups.values():
        for field in ("sample_json", "cfg"):
            require(sha256(group[field]) == group[field + "_sha256"], "Configuration drift")
    for context in ctx:
        path = Path(context["local_root_path"])
        require(path.is_absolute() and path.is_file(), "Not an exact local ROOT file")
        stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (int(context["expected_file_size_bytes"]), int(context["expected_file_mtime_ns"])), "ROOT input stat changed")
    manifest["counts"] = {"negative_contributions": len(targets), "unique_terms": len({r["target_id"] for r in targets}),
                          "excluded_contributions": len(excluded), "source_contexts": len(ctx),
                          "source_groups": len(groups), "ownership_rows": len(own),
                          "expected_entries": sum(int(c["expected_entries"]) for c in ctx)}
    dump_yaml(output / "class_b_worklist_manifest.yaml", manifest)
    print(json.dumps(manifest["counts"]), flush=True)


def selectors_for(ownership):
    import numpy as np

    selectors = defaultdict(dict)
    for row in ownership:
        key = (row["source_channel"], row["fit_variable"])
        edges = np.asarray(json.loads(row["source_edges"]))
        require(len(edges) > 1 and np.isfinite(edges).all() and (np.diff(edges) > 0).all(), "Invalid source bin edges")
        groups = selectors[row["group_id"]]
        if key not in groups:
            groups[key] = (edges, defaultdict(list))
        require(np.array_equal(groups[key][0], edges), "Incompatible source axes")
        groups[key][1][int(row["source_bin"])].append(row)
    return selectors


class fill_capture:
    def __init__(self, context, selectors, applications, writer):
        self.context, self.selectors, self.applications, self.writer = context, selectors, applications, writer
        self.entry_start = self.rows = 0
        self.sums = defaultdict(compensated_sum)
        self.signs = defaultdict(Counter)

    def __call__(self, values):
        import awkward as ak
        import numpy as np

        key = (values["ch_name"], values["dense_axis_name"])
        if key not in self.selectors or values["syst_var"] != "nominal" or values["wgt_fluct"] != "nominal":
            return
        if values["appl"] != self.applications[key[0]]:
            return
        edges, bins = self.selectors[key]
        indices = np.flatnonzero(ak.to_numpy(values["all_cuts_mask"]))
        if values["combined_axis_mask"] is not None:
            indices = indices[np.asarray(ak.to_numpy(values["combined_axis_mask"]), dtype=bool)]
        coordinates = np.asarray(ak.to_numpy(values["values_cut_map"][key[1]]))
        source_bins = np.searchsorted(edges, coordinates, side="right")
        selected = np.isin(source_bins, list(bins)) & np.isfinite(coordinates)
        coefficients = np.asarray(values["eft_coeffs_cut"])
        weights = np.asarray(ak.to_numpy(values["weights_flat"]))
        require(len(indices) == len(coordinates) == len(weights) == len(coefficients), "Unpaired fill arrays")
        # Only retained memberships are multiplied/serialized; no global BTV scan.
        for j in np.flatnonzero(selected):
            idx, source_bin = int(indices[j]), int(source_bins[j])
            memberships = bins[source_bin]
            terms = sorted({int(r["native_term_index"]) for r in memberships})
            product = event_products(coefficients[j:j + 1], weights[j:j + 1])[0]
            w, p0, q0 = float(weights[j]), float(coefficients[j, 0]), float(product[0])
            pii = [float(coefficients[j, k]) for k in terms]
            qii = [float(product[k]) for k in terms]
            context, events = self.context, values["events"]
            entry = self.entry_start + idx
            fill_id = f"{context['context_id']}:{entry}:{key[0]}:{key[1]}:{source_bin}"
            factors = {}
            if w < 0:
                obj = values["weights_object"]
                factors = {name: float(ak.to_numpy(obj.partial_weight(include=[name]))[idx]) for name in obj._weights}
                require(math.isclose(weight_product(factors), w, rel_tol=1e-12, abs_tol=1e-14), "Ordinary factors do not reconstruct weight")
            self.writer.writerow({
                "fill_id": fill_id, "context_id": context["context_id"], "group_id": context["group_id"],
                "source_file": context["local_root_path"], "entry": entry, "run": int(events.run[idx]),
                "lumi": int(events.luminosityBlock[idx]), "event": int(events.event[idx]),
                "process": context["process"], "era": context["year_or_era"], "channel": key[0],
                "variable": key[1], "source_bin": source_bin, "value": float(coordinates[j]),
                "w_analysis": w, "weight_dtype": weights.dtype.str, "coefficient_dtype": coefficients.dtype.str,
                "p0": p0, "q0": q0, "indices": json.dumps(terms), "pii": json.dumps(pii), "qii": json.dumps(qii),
                "sign_w": sign(w), "sign_pii": json.dumps([sign(x) for x in pii]),
                "sign_qii": json.dumps([sign(x) for x in qii]), "ordinary_factors": json.dumps(factors),
            })
            self.rows += 1
            for row in memberships:
                k, rid = int(row["native_term_index"]), row["negative_row_id"]
                value, coefficient = float(product[k]), float(coefficients[j, k])
                self.sums[rid].add(value)
                self.signs[rid][f"{sign(w)},{sign(coefficient)},{sign(value)}"] += 1


def closure_rows(targets, sums, signs, completed_groups, manifest):
    for target in targets:
        rid = target["negative_row_id"]
        complete = set(target["source_group_ids"].split("|")).issubset(completed_groups)
        observed, expected = sums[rid].value(), float(target["Eii_year_or_era"])
        closed = complete and observed < 0 and math.isclose(observed, expected, rel_tol=manifest["relative_tolerance"], abs_tol=manifest["absolute_tolerance"])
        yield {"negative_row_id": rid, "target_id": target["target_id"], "era": target["year_or_era"],
               "accepted_Eii_y": target["Eii_year_or_era"], "observed_completed_contexts": observed,
               "residual": observed - expected, "complete_ownership": complete, "closed": closed,
               "absolute_tolerance": manifest["absolute_tolerance"], "relative_tolerance": manifest["relative_tolerance"],
               "mechanism": mechanism(signs[rid], closed), "sign_counts": json.dumps(dict(signs[rid]), sort_keys=True)}


def event_closure(args, opened_calibrations):
    import awkward as ak
    import uproot
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
    from topeft.modules.missing_parton_contract import load_missing_parton_channel_contract

    manifest = load_yaml(args.manifest)
    targets, ownership, groups, contexts = source_context(manifest)
    binding = load_yaml(manifest["binding"])
    for path, digest in binding["source_hashes"].items():
        require(sha256(REPO.parent / path) == digest, "Source drift: " + path)
    for item in load_yaml(manifest["calibrations"])["files"]:
        require(sha256(item["path"]) == item["sha256"], "Calibration identity changed")
    output = args.output_dir.resolve()
    names = ["class_b_event_contributions.tsv.gz", "class_b_target_closure.tsv", "class_b_mechanism_classification.tsv",
             "class_b_unresolved_targets.tsv", "class_b_execution_receipt.yaml", "class_b_context_progress.jsonl"]
    require(all(not (output / n).exists() for n in names), "Output collision; no overwrite or resume")
    selectors = selectors_for(ownership)
    contract = load_missing_parton_channel_contract()
    applications = {r["source_channel"]: contract.expected_sr_appl(r["source_channel"]) for r in ownership}
    module, instrumented_hash = observed_processor(lambda _: None, binding["source_hashes"]["topeft/analysis/topeft_run2/analysis_processor.py"])
    config = module.load_category_config()["ALL_CH_LST_SR"]
    by_group = defaultdict(list)
    for context in contexts:
        by_group[context["group_id"]].append(context)
    order = sorted(by_group, key=lambda gid: (sum(int(c["expected_entries"]) for c in by_group[gid]), gid))
    totals, signs = defaultdict(compensated_sum), defaultdict(Counter)
    done, started, completed_groups, chunks = [], [], [], []
    count = emitted = 0
    start = time.monotonic()
    error = None
    with gzip.open(output / names[0], "xt", newline="", compresslevel=6) as stream, \
            open(output / names[-1], "x") as progress, open(os.devnull, "w") as quiet:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS, delimiter="\t")
        writer.writeheader()
        try:
            for gid in order:
                group = groups[gid]
                require(sha256(group["sample_json"]) == group["sample_json_sha256"], "Sample metadata drift")
                require(sha256(group["cfg"]) == group["cfg_sha256"], "Sample config drift")
                sample = json.loads(Path(group["sample_json"]).read_text())
                bases = {contract.base_channel(key[0]) for key in selectors[gid]}
                selected_config = {name: value for name, value in config.items()
                                   if any(row[0] in bases for row in value["lep_chan_lst"])}
                require(bases.issubset({row[0] for v in selected_config.values() for row in v["lep_chan_lst"]}), "Unbound source category")
                with redirect_stdout(quiet):
                    processor = module.AnalysisProcessor({group["dataset"]: sample}, list(sample["WCnames"]),
                        hist_lst=sorted({key[1] for key in selectors[gid]}), fill_sumw2_hist=False,
                        do_systematics=False, skip_signal_regions=False, skip_control_regions=True,
                        all_analysis=True, useRun3MVA=True, sr_category_dict=selected_config)
                for context in sorted(by_group[gid], key=lambda c: c["local_root_path"]):
                    path = Path(context["local_root_path"])
                    stat = path.stat()
                    require((stat.st_size, stat.st_mtime_ns) == (int(context["expected_file_size_bytes"]), int(context["expected_file_mtime_ns"])), "ROOT identity changed")
                    capture = fill_capture(context, selectors[gid], applications, writer)
                    module._diagnostic_capture = capture
                    started.append(context["context_id"])
                    progress.write(json.dumps({"context_id": context["context_id"], "state": "STARTED", "path": str(path)}) + "\n")
                    progress.flush()
                    entries = int(context["expected_entries"])
                    with uproot.open(str(path)) as root_file:
                        require(root_file[context["tree_name"]].num_entries == entries, "ROOT entry count changed")
                    for begin in range(0, entries, args.chunk_entries):
                        require(time.monotonic() - start < args.time_limit, "TIME_BOUNDARY; no retry")
                        free = os.statvfs(output).f_bavail * os.statvfs(output).f_frsize
                        require(free >= args.reserve_bytes and (output / names[0]).stat().st_size < args.max_output_bytes,
                                "STORAGE_BOUNDARY; preserve partial output")
                        end = min(entries, begin + args.chunk_entries)
                        capture.entry_start = begin
                        with redirect_stdout(quiet):
                            events = NanoEventsFactory.from_root(str(path), treepath="/" + context["tree_name"],
                                entry_start=begin, entry_stop=end, schemaclass=NanoAODSchema,
                                metadata={"dataset": group["dataset"]}, uproot_options={"timeout": 120, "num_workers": 1}).events()
                            require(len(events) == end - begin, "Incomplete entry range")
                            processor.process(events)
                        count += len(events)
                        del events
                        stream.flush()
                        chunk = {"context_id": context["context_id"], "entry_start": begin, "entry_stop": end}
                        chunks.append(chunk)
                        progress.write(json.dumps(dict(chunk, state="CHUNK_COMPLETE")) + "\n")
                        progress.flush()
                    for rid, accumulator in capture.sums.items():
                        totals[rid].add(accumulator.value())
                        signs[rid].update(capture.signs[rid])
                    emitted += capture.rows
                    done.append(context["context_id"])
                    progress.write(json.dumps({"context_id": context["context_id"], "state": "COMPLETE", "entries": entries, "rows": capture.rows}) + "\n")
                    progress.flush()
                    os.fsync(progress.fileno())
                    print(json.dumps({"completed_contexts": len(done), "total_contexts": len(contexts), "entries": count,
                                      "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
                completed_groups.append(gid)
                closed_rows = list(closure_rows(targets, totals, signs, set(completed_groups), manifest))
                failures = [row for row in closed_rows if row["complete_ownership"] and not row["closed"]]
                require(not failures, "CLOSURE_MISMATCH: " + json.dumps(failures[:3]))
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    rows = list(closure_rows(targets, totals, signs, set(completed_groups), manifest))
    write_table(output / names[1], rows, list(rows[0]))
    write_table(output / names[2], [{k: row[k] for k in ("negative_row_id", "mechanism", "sign_counts")} for row in rows],
                ["negative_row_id", "mechanism", "sign_counts"])
    write_table(output / names[3], [row for row in rows if not row["closed"]], list(rows[0]))
    receipt = {"status": "COMPLETE_PENDING_INDEPENDENT_READBACK" if error is None else "STOPPED_NO_RETRY",
               "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
               "collector_sha256": sha256(__file__), "instrumented_source_sha256": instrumented_hash,
               "contexts_started": started, "contexts_completed": done, "groups_completed": completed_groups,
               "completed_entry_ranges": chunks, "entries_processed": count, "rows_in_completed_contexts": emitted,
               "elapsed_seconds": time.monotonic() - start, "error": error, "attempt_count": 1, "retries": 0,
               "calibration_open_counts": dict(opened_calibrations), "analysis_reference_pkl_reads": 0,
               "network_operations": 0, "scheduler_operations": 0, "histogram_outputs": 0,
               "artifact_sha256": {name: sha256(output / name) for name in names if name != names[4]}}
    dump_yaml(output / names[4], receipt)
    print(json.dumps({"status": receipt["status"], "error": error, "closed_targets": sum(row["closed"] for row in rows)}), flush=True)
    return 0 if error is None else 1


def frozen_readback(args):
    """Regression of frozen artifacts only; never opens a scientific payload."""
    base = args.frozen_dir
    r1 = base / "R6C1_CH235_EVENT_LEVEL_MC_STAT_CONTINUATION"
    r2 = base / "R6C2_CH235_ANALYSIS_WEIGHT_DECOMPOSITION"
    r4 = base / "R6C4_CH235_BTAG_ONLY_DECOMPOSITION"
    events = list(table_rows(r1 / "ch235_event_target_rows_R6C1.tsv.gz"))
    aggregates = list(table_rows(r1 / "ch235_event_aggregate_by_era_R6C1.tsv"))
    for row in events:
        for p, q in (("p0", "q0"), ("pii", "qii")):
            require(math.isclose(float(row["w_scalar"]) * float(row[p]), float(row[q]), rel_tol=1e-12, abs_tol=1e-18), "Frozen R6C1 product mismatch")
    for agg in aggregates:
        chosen = events if agg["era"] == "aggregate" else [r for r in events if r["era"] == agg["era"]]
        for key in ("q0", "qii"):
            require(math.isclose(math.fsum(float(r[key]) for r in chosen), float(agg["sum_" + key]), rel_tol=1e-12, abs_tol=1e-18), "Frozen aggregate mismatch")
            require(math.isclose(math.fsum(float(r[key]) ** 2 for r in chosen), float(agg["sum_" + key + "_sq"]), rel_tol=1e-12, abs_tol=1e-24), "Frozen sumw2 mismatch")
    components = list(table_rows(r2 / "analysis_weight_components_R6C2.tsv"))
    factors = {r["component_name_as_defined_by_source"]: float(r["component_value_for_event"]) for r in components}
    prior = load_yaml(r2 / "analysis_weight_readback_R6C2.yaml")
    product = weight_product(factors)
    require(math.isclose(product, prior["accepted_R6C1_w_analysis"], rel_tol=1e-12), "Frozen R6C2 product mismatch")
    terms = list(table_rows(r4 / "btag_method1a_event_terms_R6C4.tsv"))
    btag = btag_decomposition(terms)
    accepted = load_yaml(r4 / "btag_method1a_event_summary_R6C4.yaml")["event"]
    require(math.isclose(btag["btagSF"], accepted["accepted_btagSF_R6C2"], rel_tol=1e-12), "Frozen BTV mismatch")
    negatives = [j for group in btag["subgroups"].values() for j in group["jets"] if j["category"] == "LOOSE_NOT_MEDIUM" and j["data_factor"] < 0]
    require(bool(negatives), "Missing frozen BTV corner case")
    receipt = {"status": "PASS", "R6C1_event_count": len(events), "R6C1_products_and_sums": "PASS",
               "MP009A3_SM_sumw2_algebra": "sum((w*p0)^2), checked against frozen R6C1",
               "R6C2_serialized_factor_count": len(factors), "R6C2_product": product,
               "R6C2_coverage_limit": prior["full_component_coverage"],
               "R6C2_note": "Only the preserved 19-factor artifact is tested; missing nominal unit-factor rows are not fabricated.",
               "R6C4": btag, "payload_reads": 0,
               "input_sha256": {str(p): sha256(p) for p in [r1 / "ch235_event_target_rows_R6C1.tsv.gz", r1 / "ch235_event_aggregate_by_era_R6C1.tsv", r2 / "analysis_weight_components_R6C2.tsv", r2 / "analysis_weight_readback_R6C2.yaml", r4 / "btag_method1a_event_terms_R6C4.tsv", r4 / "btag_method1a_event_summary_R6C4.yaml"]}}
    dump_yaml(args.output, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "R6C4" and k != "input_sha256"}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="operation", required=True)
    discover_parser = actions.add_parser("discover")
    for name in ("prior-dir", "year-era", "binding", "calibrations", "output-dir"):
        discover_parser.add_argument("--" + name, type=Path, required=True)
    discover_parser.add_argument("--exclude-channel", required=True)
    discover_parser.add_argument("--exclude-variable", required=True)
    discover_parser.add_argument("--atol", type=float, default=1e-10)
    discover_parser.add_argument("--rtol", type=float, default=1e-10)
    run_parser = actions.add_parser("event-closure")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--chunk-entries", type=int, default=10000)
    run_parser.add_argument("--time-limit", type=int, default=108000)
    run_parser.add_argument("--max-output-bytes", type=int, default=2 * 1024**3)
    run_parser.add_argument("--reserve-bytes", type=int, default=750 * 1024**2)
    for operation in ("weight-decompose", "btag-decompose"):
        sub = actions.add_parser(operation)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    read_parser = actions.add_parser("readback")
    read_parser.add_argument("--frozen-dir", type=Path, required=True)
    read_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    calibrations = []
    if args.operation == "event-closure":
        require(args.chunk_entries > 0 and 0 < args.time_limit <= 108000, "Invalid execution bound")
        calibration_record = load_yaml(load_yaml(args.manifest)["calibrations"])
        require(calibration_record["status"] == "OWNER_AUTHORIZED_READ_ONLY", "Calibration read scope not granted")
        calibrations = [row["path"] for row in calibration_record["files"]]
    opened = install_io_guard(calibrations)
    if args.operation == "discover":
        require(math.isfinite(args.atol) and math.isfinite(args.rtol) and args.atol >= 0 and args.rtol >= 0, "Invalid tolerance")
        discover(args)
    elif args.operation == "event-closure":
        return event_closure(args, opened)
    elif args.operation == "readback":
        frozen_readback(args)
    elif args.operation == "btag-decompose":
        dump_yaml(args.output, btag_decomposition(list(table_rows(args.input))))
    else:
        rows = list(table_rows(args.input))
        factors = {r["component_name_as_defined_by_source"]: float(r["component_value_for_event"]) for r in rows}
        require(len(factors) == len(rows), "Duplicate weight factor")
        dump_yaml(args.output, {"factors": factors, "product": weight_product(factors), "input_sha256": sha256(args.input)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
