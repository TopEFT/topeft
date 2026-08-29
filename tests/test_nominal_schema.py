from __future__ import annotations

import gc
import pickle
import weakref
from collections import OrderedDict

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.nominal_schema import (
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    canonicalize_nominal_keys,
    eft_nominal_key,
    evaluate_nominal_at_wc,
    get_eft_nominal,
    get_nominal_components,
    get_scalar_nominal,
    map_nominal_components,
    materialize_legacy_histogram_dict,
    materialize_scalar_histogram_dict,
    merge_nominal_mappings,
    scalar_nominal_key,
    validate_histogram_compatibility,
    validate_nominal_family,
    validate_nominal_mapping,
)


def _categorical_axes():
    return (
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
    )


def _sparse(family="njets", process="background", weight=2.0, *, companion=False):
    dense_name = f"{family}_sumw2" if companion else family
    output = SparseHist(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name=dense_name),
        storage="Double",
    )
    output.fill(
        process=process,
        channel="3l_onZ_1b",
        systematic="nominal",
        appl="isSR",
        **{dense_name: np.asarray([0.5])},
        weight=np.asarray([weight]),
    )
    return output


def _eft(family="njets", process="signal", coefficients=None, weight=2.0):
    output = HistEFT(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name=family),
        wc_names=["ctG"],
        label="Events",
    )
    if coefficients is None:
        coefficients = np.asarray([[1.5, 2.0, 3.0]])
    output.fill(
        process=process,
        channel="3l_onZ_1b",
        systematic="nominal",
        appl="isSR",
        **{family: np.asarray([0.5])},
        weight=np.asarray([weight]),
        eft_coeff=coefficients,
    )
    return output


def _total(histogram):
    if isinstance(histogram, HistEFT):
        values = histogram.eval({})
    else:
        values = histogram.view(flow=True, as_dict=True)
    return sum(float(np.asarray(value).sum()) for value in values.values())


def test_scalar_eft_and_mixed_family_access_and_wc_evaluation():
    scalar = _sparse(weight=4.0)
    eft = _eft(weight=2.0)
    mapping = OrderedDict(
        ((scalar_nominal_key("njets"), scalar), (eft_nominal_key("njets"), eft))
    )

    assert tuple(get_nominal_components(mapping, "njets")) == (
        "scalar_nominal",
        "eft_nominal",
    )
    assert get_scalar_nominal(mapping, "njets") is scalar
    assert get_eft_nominal(mapping, "njets") is eft
    assert _total(evaluate_nominal_at_wc(mapping, "njets", {})) == pytest.approx(7.0)
    assert _total(evaluate_nominal_at_wc(mapping, "njets", {"ctG": 1.0})) == pytest.approx(17.0)


@pytest.mark.parametrize("present", ["scalar", "eft"])
def test_missing_optional_component_is_valid(present):
    key = scalar_nominal_key("njets") if present == "scalar" else eft_nominal_key("njets")
    value = _sparse() if present == "scalar" else _eft()
    mapping = {key: value}
    validate_nominal_family(mapping, "njets")
    assert tuple(get_nominal_components(mapping, "njets")) == (f"{present}_nominal",)


def test_neither_component_original_collision_and_duplicate_process_are_rejected():
    with pytest.raises(ValueError, match="no nominal component"):
        validate_nominal_family({}, "njets")
    with pytest.raises(ValueError, match="duplicate authoritative"):
        validate_nominal_family(
            {"njets": _eft(), scalar_nominal_key("njets"): _sparse()}, "njets"
        )
    with pytest.raises(ValueError, match="duplicate process"):
        validate_nominal_family(
            {
                scalar_nominal_key("njets"): _sparse(process="same"),
                eft_nominal_key("njets"): _eft(process="same"),
            },
            "njets",
        )


def test_companion_selection_partial_and_orphan_rules_are_strict():
    mapping = {
        scalar_nominal_key("njets"): _sparse(process="background"),
        "njets_sumw2": _sparse(process="background", companion=True, weight=4.0),
    }
    validate_nominal_family(
        mapping,
        "njets",
        companion_selected=True,
        selected_processes=("background",),
    )
    with pytest.raises(ValueError, match="policy-unselected"):
        validate_nominal_family(mapping, "njets", companion_selected=False)
    with pytest.raises(ValueError, match="orphan"):
        validate_nominal_family({"njets_sumw2": mapping["njets_sumw2"]}, "njets")

    weighted = SparseHist(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets_sumw2"),
        storage="Weight",
    )
    with pytest.raises(TypeError, match="storage='Double'"):
        validate_nominal_family(
            {
                scalar_nominal_key("njets"): _sparse(),
                "njets_sumw2": weighted,
            },
            "njets",
        )

    with pytest.raises(ValueError, match="partial required coverage"):
        validate_nominal_family(
            {
                scalar_nominal_key("njets"): _sparse(process="background"),
                eft_nominal_key("njets"): _eft(process="signal"),
                "njets_sumw2": _sparse(
                    process="background", companion=True, weight=4.0
                ),
            },
            "njets",
            companion_selected=True,
            selected_processes=("background", "signal"),
        )


def test_two_dimensional_family_remains_one_sparse_nominal():
    family = "lepton_pt_vs_eta"
    nominal = SparseHist(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="lepton_pt_vs_eta_pt"),
        hist.axis.Regular(2, 0.0, 2.0, name="lepton_pt_vs_eta_abseta"),
        storage="Double",
    )
    mapping = {family: nominal}
    validate_nominal_family(mapping, family)
    assert get_scalar_nominal(mapping, family) is nominal
    with pytest.raises(ValueError, match="2D family"):
        validate_nominal_family({scalar_nominal_key(family): nominal}, family)


def test_merge_optional_components_pickle_roundtrip_and_key_order_are_deterministic():
    scalar_chunk = OrderedDict(((scalar_nominal_key("njets"), _sparse(weight=2.0)),))
    eft_chunk = OrderedDict(((eft_nominal_key("njets"), _eft(weight=3.0)),))
    reopened = pickle.loads(pickle.dumps(eft_chunk))
    merged = merge_nominal_mappings(
        (scalar_chunk, reopened),
        runtime_families=("njets",),
    )
    reverse = merge_nominal_mappings(
        (reopened, scalar_chunk),
        runtime_families=("njets",),
    )
    expected = (scalar_nominal_key("njets"), eft_nominal_key("njets"))
    assert tuple(merged) == expected
    assert tuple(reverse) == expected
    assert _total(evaluate_nominal_at_wc(merged, "njets", {})) == pytest.approx(6.5)


def test_same_type_addition_and_axis_or_wc_order_mismatch():
    first = {scalar_nominal_key("njets"): _sparse(weight=2.0)}
    second = {scalar_nominal_key("njets"): _sparse(weight=3.0)}
    merged = merge_nominal_mappings((first, second), runtime_families=("njets",))
    assert _total(merged[scalar_nominal_key("njets")]) == pytest.approx(5.0)

    incompatible_axis = _sparse(family="met")
    with pytest.raises(ValueError, match="Dense axes"):
        validate_histogram_compatibility(
            first[scalar_nominal_key("njets")], incompatible_axis, key="njets"
        )
    reordered = HistEFT(
        *_categorical_axes(),
        hist.axis.Regular(2, 0.0, 2.0, name="njets"),
        wc_names=["cpt", "ctG"],
        label="Events",
    )
    with pytest.raises(ValueError, match="WC ordering"):
        validate_histogram_compatibility(
            HistEFT(
                *_categorical_axes(),
                hist.axis.Regular(2, 0.0, 2.0, name="njets"),
                wc_names=["ctG", "cpt"],
                label="Events",
            ),
            reordered,
            key="njets__eft_nominal",
        )


def test_mapping_transform_canonicalization_and_schema_mismatch():
    mapping = OrderedDict(
        (
            ("other", 1),
            (eft_nominal_key("njets"), _eft()),
            (scalar_nominal_key("njets"), _sparse()),
        )
    )
    transformed = map_nominal_components(mapping, "njets", lambda value: value.copy())
    canonical = canonicalize_nominal_keys(
        transformed, runtime_families=("njets",)
    )
    assert tuple(canonical) == (
        scalar_nominal_key("njets"),
        eft_nominal_key("njets"),
        "other",
    )
    with pytest.raises(ValueError, match="Split sibling keys require"):
        get_nominal_components(mapping, "njets", schema_version=None)


def test_plotting_scalar_view_combines_scalar_and_eft_at_requested_wc():
    source = {
        scalar_nominal_key("njets"): _sparse(weight=4.0),
        eft_nominal_key("njets"): _eft(weight=2.0),
        "njets_sumw2": _sparse(companion=True, weight=25.0),
    }
    view = materialize_scalar_histogram_dict(
        source,
        runtime_families=("njets",),
        wc_values={"ctG": 1.0},
        require_companions=("njets",),
    )
    assert tuple(view) == ("njets", "njets_sumw2")
    assert isinstance(view["njets"], SparseHist)
    assert _total(view["njets"]) == pytest.approx(17.0)
    assert _total(view["njets_sumw2"]) == pytest.approx(25.0)


def test_transient_compatibility_view_is_lossless_unserialized_and_releasable():
    scalar = _sparse(weight=4.0)
    eft = _eft(weight=2.0)
    scalar_reference = weakref.ref(scalar)
    eft_reference = weakref.ref(eft)
    source = {
        scalar_nominal_key("njets"): scalar,
        eft_nominal_key("njets"): eft,
        "njets_sumw2": _sparse(companion=True, weight=25.0),
    }
    view = materialize_legacy_histogram_dict(
        source,
        runtime_families=("njets",),
        require_companions=("njets",),
    )
    assert tuple(view) == ("njets", "njets_sumw2")
    assert _total(view["njets"]) == pytest.approx(7.0)
    serialized = pickle.dumps(source)
    assert scalar_nominal_key("njets").encode() in serialized
    assert "njets" not in pickle.loads(serialized)

    del scalar, eft, source
    gc.collect()
    assert scalar_reference() is None
    assert eft_reference() is None
    view_reference = weakref.ref(view["njets"])
    del view
    gc.collect()
    assert view_reference() is None


def test_malformed_unknown_component_is_rejected():
    with pytest.raises(ValueError, match="orphan or unresolved"):
        validate_nominal_mapping(
            {
                scalar_nominal_key("njets"): _sparse(),
                scalar_nominal_key("met"): _sparse(family="met"),
            },
            runtime_families=("njets",),
            schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
        )
