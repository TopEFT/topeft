"""Post-production datacard identity, metadata, and ordering primitives.

Inputs are already resolved and ordered by the caller. This module neither
discovers package files nor runs the legacy packaging commands.
"""

import copy
import hashlib
import math
import re
from pathlib import Path


_channel_pattern = re.compile(r"ch([1-9][0-9]*)\Z")
_physical_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_mapping_keys = {"physical_name", "per_era_chN"}
_combined_keys = {
    "era", "physical_name", "per_era_chN", "combined_chN",
    "combined_order_index", "destination_txt_name", "destination_root_name",
}


def sha256_file(path):
    """Return the lowercase SHA256 digest of a file, reading in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_name(value):
    if not isinstance(value, str) or _physical_pattern.fullmatch(value) is None:
        raise ValueError("invalid physical target identity")
    return value


def _channel_index(value):
    match = _channel_pattern.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("invalid chN label")
    return int(match.group(1))


def _per_era_rows(rows):
    """Parse supplied mapping shape and namespace without building a mapping."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("per-era mapping must be a nonempty list")
    names = set()
    indexes = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _mapping_keys:
            raise ValueError("invalid per-era mapping row")
        name = _physical_name(row["physical_name"])
        index = _channel_index(row["per_era_chN"])
        if name in names or index in indexes:
            raise ValueError("duplicate per-era identity or chN")
        names.add(name)
        indexes.add(index)
    if indexes != set(range(1, len(rows) + 1)):
        raise ValueError("noncontiguous per-era chN namespace")
    return rows


def build_per_era_mapping(physical_names):
    """Assign ch1..chN to an already canonical ordered physical target list."""
    if not isinstance(physical_names, (list, tuple)) or not physical_names:
        raise ValueError("physical identities must be a nonempty sequence")
    seen = set()
    result = []
    for index, name in enumerate(physical_names, 1):
        name = _physical_name(name)
        if name in seen:
            raise ValueError("duplicate physical target identity")
        seen.add(name)
        result.append({"physical_name": name, "per_era_chN": f"ch{index}"})
    return result


def verify_per_era_mapping(observed, physical_names):
    """Check row positions and identities directly against the supplied order."""
    _per_era_rows(observed)
    if not isinstance(physical_names, (list, tuple)) or len(observed) != len(physical_names):
        raise ValueError("per-era target count differs")
    if len(set(physical_names)) != len(physical_names):
        raise ValueError("duplicate expected physical identity")
    for position, (row, name) in enumerate(zip(observed, physical_names), 1):
        if row["physical_name"] != _physical_name(name) or _channel_index(row["per_era_chN"]) != position:
            raise ValueError("per-era mapping identity or order differs")
    return True


def _selected_unit(unit):
    if not isinstance(unit, dict):
        raise ValueError("selected-WC unit must be an object")
    for process, wcs in unit.items():
        if not isinstance(process, str) or not isinstance(wcs, list) or any(not isinstance(wc, str) for wc in wcs):
            raise ValueError("invalid selected-WC entry")
        if len(set(wcs)) != len(wcs):
            raise ValueError("duplicate WC within source unit")


def consolidate_selected_wcs(source_units):
    """Union successful unit WCs in unit, process, and WC encounter order."""
    result = {}
    for unit in source_units:
        _selected_unit(unit)
        for process, wcs in unit.items():
            target = result.setdefault(process, [])
            for wc in wcs:
                if wc not in target:
                    target.append(wc)
    return result


def verify_per_era_selected_wcs(observed, source_units):
    """Check observed process/WC order from source occurrences independently."""
    if not isinstance(observed, dict):
        raise ValueError("selected-WC output must be an object")
    occurrences = {}
    for unit in source_units:
        _selected_unit(unit)
        for process, wcs in unit.items():
            if process not in occurrences:
                occurrences[process] = []
            occurrences[process].extend(wcs)
    if set(observed) != set(occurrences):
        raise ValueError("selected-WC process identities differ")
    for process in occurrences:
        value = observed[process]
        if not isinstance(value, list) or any(not isinstance(wc, str) for wc in value):
            raise ValueError("invalid observed selected-WC list")
        if len(value) != len(set(value)):
            raise ValueError("duplicate observed selected WC")
        first_positions = {wc: occurrences[process].index(wc) for wc in set(occurrences[process])}
        if set(value) != set(first_positions) or [first_positions[wc] for wc in value] != sorted(first_positions.values()):
            raise ValueError("selected-WC union or order differs")
    return True


def _scaling_record(record):
    if not isinstance(record, dict) or not {"channel", "process", "parameters", "scaling"} <= set(record):
        raise ValueError("incomplete scaling record")
    if not isinstance(record["channel"], str) or not record["channel"].strip():
        raise ValueError("invalid scaling channel")
    if not isinstance(record["process"], str) or not record["process"].strip():
        raise ValueError("invalid scaling process")
    if not isinstance(record["parameters"], list) or any(not isinstance(item, str) for item in record["parameters"]):
        raise ValueError("invalid scaling parameters")
    if not isinstance(record["scaling"], list):
        raise ValueError("invalid scaling payload")
    for row in record["scaling"]:
        if not isinstance(row, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in row
        ):
            raise ValueError("invalid scaling coefficient")


def consolidate_scaling_records(source_units, mapping):
    """Relabel only physical channels, preserving source record and payload order."""
    _per_era_rows(mapping)
    labels = {row["physical_name"]: row["per_era_chN"] for row in mapping}
    result = []
    seen = set()
    for unit in source_units:
        if not isinstance(unit, list):
            raise ValueError("scaling unit must be a list")
        for record in unit:
            _scaling_record(record)
            identity = (record["channel"], record["process"])
            if identity in seen or record["channel"] not in labels:
                raise ValueError("duplicate or unmapped scaling identity")
            seen.add(identity)
            transformed = copy.deepcopy(record)
            transformed["channel"] = labels[record["channel"]]
            result.append(transformed)
    return result


def verify_per_era_scalings(observed, source_units, mapping):
    """Compare each output record to its physical source via inverse mapping."""
    _per_era_rows(mapping)
    if not isinstance(observed, list):
        raise ValueError("scaling output must be a list")
    physical_by_channel = {row["per_era_chN"]: row["physical_name"] for row in mapping}
    source_records = []
    source_keys = set()
    for unit in source_units:
        if not isinstance(unit, list):
            raise ValueError("scaling unit must be a list")
        for record in unit:
            _scaling_record(record)
            key = (record["channel"], record["process"])
            if key in source_keys or record["channel"] not in set(physical_by_channel.values()):
                raise ValueError("duplicate or unmapped source scaling")
            source_keys.add(key)
            source_records.append(record)
    if len(observed) != len(source_records):
        raise ValueError("scaling record count differs")
    observed_keys = set()
    for actual, source in zip(observed, source_records):
        _scaling_record(actual)
        key = (actual["channel"], actual["process"])
        if key in observed_keys or physical_by_channel.get(actual["channel"]) != source["channel"]:
            raise ValueError("scaling identity or order differs")
        observed_keys.add(key)
        if {key: value for key, value in actual.items() if key != "channel"} != {
            key: value for key, value in source.items() if key != "channel"
        }:
            raise ValueError("scaling payload differs")
    return True


def build_combined_mapping(run2_mapping, run3_mapping):
    """Assign combined channels and canonical destination names."""
    _per_era_rows(run2_mapping)
    _per_era_rows(run3_mapping)
    for rows in (run2_mapping, run3_mapping):
        if any(_channel_index(row["per_era_chN"]) != position for position, row in enumerate(rows, 1)):
            raise ValueError("per-era mapping rows are out of canonical order")
    result = []
    offset = len(run2_mapping)
    for era, rows, era_offset in (("run2", run2_mapping, 0), ("run3", run3_mapping, offset)):
        for row in rows:
            number = _channel_index(row["per_era_chN"]) + era_offset
            name = row["physical_name"]
            basename = f"{'Run2' if era == 'run2' else 'Run3'}_ttx_multileptons-{name}"
            result.append({
                "era": era, "physical_name": name, "per_era_chN": row["per_era_chN"],
                "combined_chN": f"ch{number}", "combined_order_index": number,
                "destination_txt_name": basename + ".txt",
                "destination_root_name": basename + ".root",
            })
    return result


def verify_combined_mapping(observed, run2_mapping, run3_mapping):
    """Check domains, offsets, names, and row order without the builder."""
    _per_era_rows(run2_mapping)
    _per_era_rows(run3_mapping)
    for rows in (run2_mapping, run3_mapping):
        if any(_channel_index(row["per_era_chN"]) != position for position, row in enumerate(rows, 1)):
            raise ValueError("per-era source mapping rows are out of canonical order")
    if not isinstance(observed, list) or len(observed) != len(run2_mapping) + len(run3_mapping):
        raise ValueError("combined mapping count differs")
    by_era = {"run2": run2_mapping, "run3": run3_mapping}
    seen_physical = {"run2": set(), "run3": set()}
    seen_local = {"run2": set(), "run3": set()}
    seen_combined = set()
    seen_order = set()
    seen_destinations = set()
    for position, row in enumerate(observed, 1):
        if not isinstance(row, dict) or set(row) != _combined_keys:
            raise ValueError("invalid combined mapping row")
        era = row["era"]
        if era not in by_era:
            raise ValueError("invalid combined era")
        local = _channel_index(row["per_era_chN"])
        expected_number = local + (len(run2_mapping) if era == "run3" else 0)
        combined = _channel_index(row["combined_chN"])
        name = _physical_name(row["physical_name"])
        if local > len(by_era[era]) or by_era[era][local - 1]["physical_name"] != name:
            raise ValueError("combined physical identity differs from source")
        if type(row["combined_order_index"]) is not int or row["combined_order_index"] != position or combined != position or expected_number != position:
            raise ValueError("combined channel order or offset differs")
        if (name in seen_physical[era] or local in seen_local[era] or
                combined in seen_combined or position in seen_order):
            raise ValueError("duplicate combined identity")
        seen_physical[era].add(name)
        seen_local[era].add(local)
        seen_combined.add(combined)
        seen_order.add(position)
        stem = ("Run2" if era == "run2" else "Run3") + "_ttx_multileptons-" + name
        if row["destination_txt_name"] != stem + ".txt" or row["destination_root_name"] != stem + ".root":
            raise ValueError("combined destination name differs")
        if row["destination_txt_name"] in seen_destinations or row["destination_root_name"] in seen_destinations:
            raise ValueError("duplicate combined destination")
        seen_destinations.update((row["destination_txt_name"], row["destination_root_name"]))
    for era, rows in by_era.items():
        if seen_physical[era] != {row["physical_name"] for row in rows} or seen_local[era] != set(range(1, len(rows) + 1)):
            raise ValueError("incomplete combined era domain")
    return True


def build_ordered_card_inputs(combined_mapping):
    """Return canonical cards/ paths in declared combined order."""
    if not isinstance(combined_mapping, list):
        raise ValueError("combined mapping must be a list")
    indexes = [row["combined_order_index"] for row in combined_mapping]
    if sorted(indexes) != list(range(1, len(indexes) + 1)):
        raise ValueError("combined order is incomplete")
    return ["cards/" + row["destination_txt_name"] for row in sorted(combined_mapping, key=lambda row: row["combined_order_index"])]


def verify_ordered_card_inputs(observed, combined_mapping):
    """Check serialized lines against mapping rows without the list writer."""
    if isinstance(observed, str):
        lines = observed.splitlines()
    elif isinstance(observed, list):
        lines = observed
    else:
        raise ValueError("ordered inputs must be text or a line list")
    if not isinstance(combined_mapping, list) or len(lines) != len(combined_mapping):
        raise ValueError("ordered input count differs")
    if len(set(lines)) != len(lines):
        raise ValueError("duplicate ordered input")
    rows_by_index = {}
    for row in combined_mapping:
        index = row.get("combined_order_index") if isinstance(row, dict) else None
        if type(index) is not int or index in rows_by_index:
            raise ValueError("invalid combined order index")
        rows_by_index[index] = row
    if set(rows_by_index) != set(range(1, len(lines) + 1)):
        raise ValueError("noncontiguous combined order index")
    for position, line in enumerate(lines, 1):
        name = rows_by_index[position].get("destination_txt_name")
        if not isinstance(name, str) or not name.endswith(".txt") or "/" in name or line != "cards/" + name:
            raise ValueError("ordered input path or order differs")
    return True
