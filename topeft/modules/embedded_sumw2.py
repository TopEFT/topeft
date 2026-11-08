"""Producer-only coverage for the versioned embedded statistical layout.

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


def fill_embedded_2d(histogram, coverage, *, family, dataset, selected,
                     value_weight, second_moment, **fill_values):
    """Store 2D values and independent nominal moments with explicit coverage."""
    fill_sumw2 = fill_values["systematic"] == "nominal" and selected
    selected_second_moment = second_moment if fill_sumw2 else 0.0
    histogram.fill_with_moments(
        value_weight=value_weight, second_moment=selected_second_moment, **fill_values,
    )
    state = "unavailable"
    if fill_sumw2:
        state = "selected_nonzero" if np.any(np.asarray(second_moment) != 0) else "selected_zero"
    coverage.add((family, "scalar_nominal", dataset, fill_values["process"],
                  fill_values["channel"], fill_values["appl"],
                  fill_values["systematic"], state))


def scale_embedded_family(histograms, family, factor):
    """Scale numerical cells and retain explicit availability, including at zero.

    Direct nonzero histogram scaling leaves coverage valid. Use this mapping
    boundary when scaling to zero so selected-nonzero records become selected-zero;
    unavailable records must never become selected merely because their bins vanish.
    """
    from topeft.modules.axes import info_2d
    from topeft.modules.nominal_schema import get_nominal_components

    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("Embedded scaling requires a finite factor.")
    components = get_nominal_components(histograms, family, schema_version=EMBEDDED_SCHEMA_VERSION)
    scaled = {}
    for component, histogram in components.items():
        key = family if family in info_2d else f"{family}__{component}"
        scaled[key] = histogram.copy().scale(factor)
    histograms.update(scaled)
    if factor == 0:
        histograms[COVERAGE_KEY] = {
            record[:-1] + ("selected_zero",)
            if record[0] == family and record[-1] == "selected_nonzero" else record
            for record in histograms[COVERAGE_KEY]
        }
    return histograms


def embedded_sumw2_view(histograms, family, *, provenance, flow=True, wc_values=None):
    """Return copied numerical components, dataset coverage and resolved provenance.

    A zero variance does not establish availability. Read the coverage records,
    including unavailable datasets sharing a process, before using the moments.
    EFT variances remain SM moments even when yields are evaluated at other WCs.
    """
    import copy
    from topeft.modules.axes import info_2d
    from topeft.modules.nominal_schema import get_nominal_components, validate_nominal_mapping
    from topeft.modules.sumw2_policy import resolved_policy_from_provenance

    policy = resolved_policy_from_provenance(provenance)
    if family not in policy.runtime_histogram_families:
        raise ValueError("Family is outside the resolved policy universe.")
    validate_nominal_mapping(histograms, runtime_families=policy.runtime_histogram_families,
                             schema_version=EMBEDDED_SCHEMA_VERSION, policy=policy)
    result = {}
    for component, histogram in get_nominal_components(
            histograms, family, schema_version=EMBEDDED_SCHEMA_VERSION).items():
        if family in info_2d:
            views = histogram.view(flow=flow)
            values = {key: cell.value.copy() for key, cell in views.items()}
            variances = {key: cell.variance.copy() for key, cell in views.items()}
        else:
            values = {key: value.copy() for key, value in histogram.eval(
                {} if wc_values is None else wc_values).items()}
            variances = histogram.nominal_sumw2(flow=flow)
            # HistEFT.eval includes flows; match the requested numerical view.
            if not flow:
                axis = histogram.dense_axis
                start = int(axis.traits.underflow)
                values = {key: value[start:start + len(axis)] for key, value in values.items()}
        result[component] = {"values": values, "variances": copy.deepcopy(variances)}
    return {"components": result,
            "coverage": coverage_manifest({r for r in histograms[COVERAGE_KEY] if r[0] == family}),
            "provenance": copy.deepcopy(dict(provenance))}


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
        if family not in runtime_families:
            raise ValueError("Embedded coverage has an unknown family.")
        if (component not in {"scalar_nominal", "eft_nominal"}
                or (family in info_2d and component != "scalar_nominal")):
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
        components = (
            (("scalar_nominal", family),) if family in info_2d else
            (("scalar_nominal", scalar_nominal_key(family)),
             ("eft_nominal", eft_nominal_key(family)))
        )
        for component, name in components:
            histogram = histograms.get(name)
            if histogram is None:
                continue
            variances = (
                {key: cell.variance for key, cell in histogram.view(flow=True).items()}
                if family in info_2d else histogram.nominal_sumw2(flow=True)
            )
            for category, variance in variances.items():
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
