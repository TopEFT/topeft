"""Producer-only coverage for the versioned embedded 1D statistical layout.

Records are sets so Coffea's ordinary accumulator unions coverage across chunks.
Each record is (family, component, dataset, process, channel, appl, systematic,
state). States describe nominal selected-zero, nominal selected-nonzero, or
unavailable variance. A systematic fill is always unavailable. Combining chunks
retains nonzero if any chunk was nonzero; it never infers selection from yield.
No numerical second moments or policy identity are encoded in these records.
"""

import numpy as np

EMBEDDED_SCHEMA_VERSION = 3
EMBEDDED_LAYOUT = "split_multicell_embedded_v1"
COVERAGE_KEY = "__embedded_sumw2_coverage__"


def fill_embedded_nominal(histogram, coverage, *, family, component, dataset,
                          selected, **fill_values):
    """Fill content unconditionally and record availability independently of yield."""
    is_nominal_path = fill_values["systematic"] == "nominal"
    fill_sumw2 = is_nominal_path and selected
    histogram.fill(fill_sumw2=fill_sumw2, **fill_values)
    state = "unavailable"
    if fill_sumw2:
        weight = np.asarray(fill_values["weight"], dtype=float)
        coefficients = fill_values.get("eft_coeff")
        if coefficients is not None:
            weight = weight * np.asarray(coefficients)[:, 0]
        state = "selected_nonzero" if np.any(np.square(weight) != 0) else "selected_zero"
    coverage.add((family, component, dataset, fill_values["process"],
                  fill_values["channel"], fill_values["appl"],
                  fill_values["systematic"], state))


def coverage_manifest(coverage):
    """Canonical JSON records; nonzero wins when zero and nonzero chunks combine."""
    merged = {}
    for record in coverage:
        key, state = record[:-1], record[-1]
        previous = merged.get(key)
        if previous is not None and (previous == "unavailable") != (state == "unavailable"):
            raise ValueError("Conflicting embedded sumw2 coverage for one target.")
        if previous != "selected_nonzero":
            merged[key] = state
    return [list(key) + [state] for key, state in sorted(merged.items())]


def validate_coverage(histograms, *, runtime_families, policy=None):
    """Check concrete policy selections and numerical cells against fill coverage."""
    from topeft.modules.nominal_schema import scalar_nominal_key, eft_nominal_key
    from topeft.modules.axes import info_2d

    coverage = histograms.get(COVERAGE_KEY)
    if not isinstance(coverage, set):
        raise ValueError("Embedded layout requires a coverage set.")
    cells = {}
    for record in coverage:
        if (not isinstance(record, tuple) or len(record) != 8
                or any(not isinstance(value, str) for value in record)):
            raise ValueError("Malformed embedded sumw2 coverage record.")
        family, component, dataset, process, channel, appl, systematic, state = record
        if family not in runtime_families or family in info_2d:
            raise ValueError("Embedded coverage has an unknown or 2D family.")
        if component not in {"scalar_nominal", "eft_nominal"}:
            raise ValueError("Invalid embedded nominal component.")
        if state not in {"selected_zero", "selected_nonzero", "unavailable"}:
            raise ValueError("Invalid embedded variance availability.")
        selected = state != "unavailable"
        if systematic != "nominal" and selected:
            raise ValueError("Systematic paths cannot contain embedded sumw2.")
        if policy is not None:
            if dataset not in policy.resolved_datasets or process not in policy.resolved_processes:
                raise ValueError("Embedded coverage target is outside the resolved policy universe.")
            if selected != (systematic == "nominal" and policy.selects(dataset, process, family)):
                raise ValueError("Embedded coverage disagrees with concrete sumw2 policy selection.")
        key = (family, component, process, channel, systematic, appl)
        cells.setdefault(key, set()).add(state)
    coverage_manifest(coverage)
    observed = set()
    for family in runtime_families:
        if family in info_2d:
            continue
        for component, name in (("scalar_nominal", scalar_nominal_key(family)),
                                ("eft_nominal", eft_nominal_key(family))):
            histogram = histograms.get(name)
            if histogram is None:
                continue
            for category, variance in histogram.nominal_sumw2(flow=True).items():
                coordinates = dict(zip(histogram.categorical_axes.name, category))
                key = (family, component, coordinates["process"], coordinates["channel"],
                       coordinates["systematic"], coordinates["appl"])
                observed.add(key)
                if key not in cells:
                    raise ValueError("Embedded numerical cell has no coverage record.")
                values = np.asarray(variance)
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError("Embedded sumw2 must be finite and nonnegative.")
                if bool(np.any(values != 0)) != ("selected_nonzero" in cells[key]):
                    raise ValueError("Embedded numerical variance disagrees with coverage.")
    if set(cells) != observed:
        raise ValueError("Embedded coverage has no matching numerical cell.")
