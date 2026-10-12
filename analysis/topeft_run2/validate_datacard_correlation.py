"""Compare source-period nuisance algebra with one serialized datacard pair.

The comparison uses DatacardMaker's preparation and card-facing transforms,
but constructs the final year-group sums here, without correlate_years.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import hist
import numpy as np
import uproot

from topcoffea.modules.utils import get_hist_from_pkl
from topeft.modules.datacard_tools import (
    DatacardMaker,
    _card_numerical_split_view,
    _sanitize_negative_template_bins,
    resolve_shape_nuisance_identity,
)


# uproot writes the writer's weighted hist.Hist as a float64 TH1D. This bound
# covers ordinary float64 summation and fitting rebin order in the focused path.
COMPARISON_RTOL = 1e-12
COMPARISON_ATOL = 1e-10


def compare_arrays(expected, observed):
    """Compare finite float64 ROOT contents with a small arithmetic envelope."""
    expected = np.asarray(expected)
    observed = np.asarray(observed)
    return (
        expected.shape == observed.shape
        and expected.dtype == np.dtype("float64")
        and observed.dtype == np.dtype("float64")
        and np.all(np.isfinite(expected))
        and np.all(np.isfinite(observed))
        and bool(np.allclose(expected, observed, rtol=COMPARISON_RTOL,
                             atol=COMPARISON_ATOL))
    )


def _group_map(groups, periods):
    period_set = set(periods)
    if not periods or len(period_set) != len(periods):
        raise ValueError("Source periods must be nonempty and unique")
    if not groups:
        raise ValueError("At least one final correlation group is required")
    seen = set()
    for label, members in groups.items():
        if not label or not members or label not in members:
            raise ValueError(f"Invalid final correlation group {label!r}: {members!r}")
        if len(members) != len(set(members)) or not set(members) <= period_set:
            raise ValueError(f"Invalid source members for group {label!r}: {members!r}")
        if seen & set(members):
            raise ValueError(f"Overlapping source periods in group {label!r}")
        seen.update(members)
    if seen != period_set:
        raise ValueError(f"Unassigned source periods: {sorted(period_set - seen)!r}")


def _card_shapes(card_path, process_name, prefix):
    lines = Path(card_path).read_text(encoding="utf-8").splitlines()
    process_rows = [line.split()[1:] for line in lines
                    if line.split()[:1] == ["process"]]
    name_rows = [row for row in process_rows
                 if row and not all(part.lstrip("-").isdigit() for part in row)]
    if len(name_rows) != 1 or name_rows[0].count(process_name) != 1:
        raise ValueError(f"Card process {process_name!r} is missing or ambiguous")
    column = name_rows[0].index(process_name)
    names = []
    for line in lines:
        fields = line.split()
        if len(fields) < 2 or fields[1] != "shape" or not fields[0].startswith(prefix):
            continue
        if len(fields[2:]) != len(name_rows[0]):
            raise ValueError(f"Malformed shape row {fields[0]!r}")
        if fields[column + 2] not in ("1", "-"):
            raise ValueError(f"Ambiguous shape assignment {fields[0]!r}")
        if fields[column + 2] == "1":
            names.append(fields[0])
    return names


def _source_rows(histogram, channel, process, periods):
    """Inventory exact sparse coordinates before independent group summation."""
    rows = {}
    process_names = defaultdict(list)
    for name in histogram.axes["process"]:
        if DatacardMaker.get_process(name) == process:
            process_names[DatacardMaker.get_year(name)].append(name)
    for period in periods:
        if len(process_names[period]) != 1:
            raise ValueError(
                f"Expected one grouped process for {process!r}/{period!r}; "
                f"found {process_names[period]!r}"
            )
    for key, payload in histogram.view(flow=True).items():
        values = key._asdict()
        if values.get("channel") != channel:
            continue
        period = DatacardMaker.get_year(values["process"])
        if values["process"] != process_names.get(period, [None])[0]:
            continue
        other = tuple(sorted((name, value) for name, value in values.items()
                             if name not in ("process", "systematic")))
        coordinate = (period, values["systematic"], other)
        if coordinate in rows:
            raise ValueError(f"Duplicate source coordinate {coordinate!r}")
        rows[coordinate] = (key, np.array(payload, copy=True))
    return rows, process_names


def _expected_histogram(maker, variable, channel, process, periods, groups,
                        prefix, required_bases):
    source = _card_numerical_split_view({variable: maker.hists[variable]})[variable]
    rows, process_names = _source_rows(source, channel, process, periods)
    bases = set(required_bases)
    for _, systematic, _ in rows:
        if systematic.startswith(prefix) and systematic.endswith(("Up", "Down")):
            for period in periods:
                for direction in ("Up", "Down"):
                    suffix = f"_{period}{direction}"
                    if systematic.endswith(suffix):
                        bases.add(systematic[:-len(suffix)])
    if not bases:
        raise ValueError(f"No source nuisance bases under {prefix!r}")
    if any(not base.startswith(prefix) for base in bases):
        raise ValueError("Required base lies outside the selected prefix")

    nominal_coordinates = {
        period: {other for row_period, systematic, other in rows
                 if row_period == period and systematic == "nominal"}
        for period in periods
    }
    first = nominal_coordinates[periods[0]]
    if not first or any(nominal_coordinates[p] != first for p in periods):
        raise ValueError(f"Missing or incompatible nominal source support: "
                         f"{nominal_coordinates!r}")

    final_systematics = ["nominal"] + [
        f"{base}_{label}{direction}"
        for base in sorted(bases) for label in groups
        for direction in ("Up", "Down")
    ]
    categorical_axes = [
        hist.axis.StrCategory(
            [process] if axis.name == "process" else
            final_systematics if axis.name == "systematic" else list(axis),
            name=axis.name, growth=True,
        )
        for axis in source.categorical_axes
    ]
    expected = source.empty_from_axes(categorical_axes=categorical_axes)
    source_support = {}
    expected_names = set()
    for other in sorted(first):
        exemplar = rows[(periods[0], "nominal", other)][0]
        for base in sorted(bases):
            for label, members in groups.items():
                for direction in ("Up", "Down"):
                    name = f"{base}_{label}{direction}"
                    contributions = []
                    for period in periods:
                        systematic = f"{base}_{period}{direction}" if period in members else "nominal"
                        coordinate = (period, systematic, other)
                        if coordinate not in rows:
                            raise ValueError(f"Missing source support {coordinate!r}")
                        contributions.append(rows[coordinate][1])
                    payload = np.sum(np.stack(contributions), axis=0)
                    target = exemplar._replace(process=process, systematic=name)
                    expected[target] = payload
                    expected_names.add(f"{base}_{label}")
                    source_support[str((base, label, direction, other))] = [
                        (period, f"{base}_{period}{direction}" if period in members else "nominal")
                        for period in periods
                    ]
        nominal = np.sum(np.stack([rows[(period, "nominal", other)][1]
                                   for period in periods]), axis=0)
        expected[exemplar._replace(process=process, systematic="nominal")] = nominal
    return expected, source_support, expected_names, sorted(bases), process_names


def validate_correlation(source_pkl, card_txt, template_root, variable, channel,
                         periods, groups, prefix, process, *, binning_mode="fitting",
                         negative_bins="crop", required_bases=(),
                         enable_legacy_run3_vsmu_recovery=False):
    """Return a structured, fail-closed comparison for one semantic process."""
    report = {
        "inputs": {"source_pkl": str(source_pkl), "card_txt": str(card_txt),
                   "template_root": str(template_root), "variable": variable,
                   "channel": channel, "periods": list(periods), "groups": groups,
                   "prefix": prefix, "process": process,
                   "binning_mode": binning_mode, "negative_bins": negative_bins},
        "comparison": {"dtype": "float64", "rtol": COMPARISON_RTOL,
                       "atol": COMPARISON_ATOL},
        "source_support": {}, "expected_identities": [], "root_identities": [],
        "card_identities": [], "comparisons": {}, "missing": [], "extra": [],
        "duplicate": [], "forbidden": [], "pruned": [], "errors": [], "passed": False,
    }
    try:
        report["artifact_identity"] = {
            label: {"resolved_path": str(Path(path).resolve()),
                    "size_bytes": Path(path).stat().st_size,
                    "mtime_ns": Path(path).stat().st_mtime_ns}
            for label, path in (("source_pkl", source_pkl), ("card_txt", card_txt),
                                ("template_root", template_root))
        }
        _group_map(groups, periods)
        if negative_bins not in ("crop", "keep"):
            raise ValueError(f"Unknown negative-bin mode {negative_bins!r}")
        maker = DatacardMaker(
            hists={}, year_lst=list(periods), var_lst=[variable], do_nuisance=True,
            skip_missing_parton_rate_syst=True, verbose=False,
            binning_mode=binning_mode,
            enable_legacy_run3_vsmu_recovery=enable_legacy_run3_vsmu_recovery,
        )
        # Load only the selected nominal family. The correlation decision does
        # not inspect or gate on a stored statistical companion.
        loaded = get_hist_from_pkl(source_pkl, allow_empty=True)
        selected_keys = {variable, f"{variable}__scalar_nominal",
                         f"{variable}__eft_nominal"}
        selected_hists = {key: value for key, value in loaded.items()
                          if key in selected_keys}
        maker.read_source_periods(hists=selected_hists)
        if variable not in maker.hists:
            raise ValueError(f"Source variable {variable!r} is absent")
        expected_hist, support, names, bases, process_names = _expected_histogram(
            maker, variable, channel, process, periods, groups, prefix,
            required_bases,
        )
        report["source_support"] = support
        report["source_processes"] = {p: process_names[p] for p in periods}
        report["source_bases"] = bases

        # These operations follow analyze(): channel, fitting binning, process,
        # metadata SR selection, SM decomposition, and optional sanitation.
        selected = maker.binning_view(
            expected_hist.integrate("channel", [channel]), variable, channel
        ).integrate("process", [process])
        selected = maker.select_final_sr_appl(selected, channel, process=process)
        templates = maker.decompose(selected, None, [])["sm"]
        maker.validate_sparse_axes_for_card(templates, channel, f"{process}_sm")
        if negative_bins == "crop":
            templates = _sanitize_negative_template_bins(templates)
        expected_arrays = {}
        for key, payload in templates.items():
            systematic = key.systematic
            if systematic != "nominal" and not systematic.startswith(prefix):
                raise ValueError(f"Unexpected expected systematic {systematic!r}")
            _, final_name = resolve_shape_nuisance_identity(
                systematic, maker.suffix, maker.run_decorrelate
            )
            identity = (f"{process}_sm" if systematic == "nominal"
                        else f"{process}_sm_{final_name}")
            if identity in expected_arrays:
                raise ValueError(f"Colliding expected identity {identity!r}")
            if np.sum(payload[0]) == 0:
                report["pruned"].append(identity)
                continue
            expected_arrays[identity] = np.asarray(payload[0][1:], dtype=np.float64)
        expected_card = sorted(names)
        report["expected_identities"] = sorted(expected_arrays)
        report["expected_card_identities"] = expected_card
        if report["pruned"]:
            report["errors"].append("Selected identity was pruned by the writer's zero-sum predicate")

        selected_prefix = f"{process}_sm_{prefix}"
        with uproot.open(template_root) as root:
            keys = [key.split(";", 1)[0] for key in root.keys(cycle=True)]
            counts = Counter(key for key in keys if key.startswith(selected_prefix))
            report["duplicate"] = sorted(key for key, count in counts.items() if count > 1)
            selected_keys = sorted(counts)
            report["root_identities"] = selected_keys
            actual_names = set(selected_keys) | ({f"{process}_sm"} if f"{process}_sm" in keys else set())
            report["missing"] = sorted(set(expected_arrays) - actual_names)
            report["extra"] = sorted(set(selected_keys) - set(expected_arrays))
            forbidden = {f"{process}_sm_{base}_{period}{direction}"
                         for base in bases for period in periods
                         for direction in ("Up", "Down")}
            forbidden -= set(expected_arrays)
            report["forbidden"] = sorted(forbidden & set(selected_keys))
            for identity in sorted(set(expected_arrays) & actual_names):
                if root.classname_of(identity) != "TH1D":
                    raise ValueError(f"Unsupported ROOT object {identity!r}: "
                                     f"{root.classname_of(identity)!r}")
                observed = np.asarray(root[identity].values(flow=False))
                expected_value = expected_arrays[identity]
                passed = compare_arrays(expected_value, observed)
                report["comparisons"][identity] = {
                    "expected": expected_value.tolist(), "observed": observed.tolist(),
                    "observed_dtype": str(observed.dtype), "passed": passed,
                }
        card_names = _card_shapes(card_txt, f"{process}_sm", prefix)
        report["card_identities"] = sorted(card_names)
        card_counts = Counter(card_names)
        report["duplicate"] += sorted(name for name, count in card_counts.items() if count > 1)
        report["missing"] += sorted(set(expected_card) - set(card_names))
        report["extra"] += sorted(set(card_names) - set(expected_card))
        report["forbidden"] += sorted(forbidden_name for forbidden_name in card_names
                                      if forbidden_name not in expected_card)
        report["passed"] = not any((report["errors"], report["missing"],
                                    report["extra"], report["duplicate"],
                                    report["forbidden"])) and all(
                                        item["passed"] for item in report["comparisons"].values()
                                    ) and len(report["comparisons"]) == len(expected_arrays)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError,
            AttributeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["classification"] = "passed" if report["passed"] else "validation_failed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-pkl", "card", "root", "variable", "channel",
                   "prefix", "process", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--period", action="append", required=True)
    parser.add_argument("--group", action="append", required=True,
                        help="Final label and source periods: LABEL=PERIOD,PERIOD")
    parser.add_argument("--required-base", action="append", default=[])
    parser.add_argument("--binning-mode", choices=("processing", "fitting"),
                        default="fitting")
    parser.add_argument("--negative-bins", choices=("crop", "keep"), default="crop")
    parser.add_argument("--enable-legacy-run3-vsmu-recovery", action="store_true")
    args = parser.parse_args(argv)
    groups = {}
    for item in args.group:
        if "=" not in item:
            parser.error(f"Invalid --group {item!r}")
        label, members = item.split("=", 1)
        if label in groups:
            parser.error(f"Duplicate group label {label!r}")
        groups[label] = members.split(",")
    result = validate_correlation(
        args.source_pkl, args.card, args.root, args.variable, args.channel,
        args.period, groups, args.prefix, args.process,
        binning_mode=args.binning_mode, negative_bins=args.negative_bins,
        required_bases=args.required_base,
        enable_legacy_run3_vsmu_recovery=args.enable_legacy_run3_vsmu_recovery,
    )
    output = Path(args.output)
    if output.resolve() in {Path(path).resolve()
                            for path in (args.source_pkl, args.card, args.root)}:
        parser.error("Report path must differ from each input artifact")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Correlation validation: {result['classification']}; "
          f"{len(result['comparisons'])} arrays; {len(result['errors'])} errors")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
