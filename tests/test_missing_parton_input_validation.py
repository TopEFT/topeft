from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import uproot

from topeft.modules.missing_parton_contract import (
    LEGACY_MISSING_PARTON_BASE_CHANNELS,
    load_registry_payload_layout,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_input_validation_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_card(
    directory,
    *,
    category="2lss_p",
    process="tllq",
    nominal=None,
    edges=None,
    shapes=(),
    txt_processes=None,
    txt_rates=None,
    extra_text="",
    extra_root_objects=(),
):
    nominal = np.asarray(
        np.arange(1.0, 9.0) if nominal is None else nominal,
        dtype=float,
    )
    edges = np.asarray(
        np.arange(len(nominal) + 1, dtype=float) if edges is None else edges,
        dtype=float,
    )
    stem = Path(directory) / f"ttx_multileptons-{category}_njets"
    with uproot.recreate(stem.with_suffix(".root")) as root_file:
        root_file[f"{process}_sm"] = (nominal, edges)
        for name, values, shape_edges in shapes:
            root_file[f"{process}_sm_{name}"] = (
                np.asarray(values, dtype=float),
                np.asarray(shape_edges, dtype=float),
            )
        for name, values, object_edges in extra_root_objects:
            root_file[name] = (
                np.asarray(values, dtype=float),
                np.asarray(object_edges, dtype=float),
            )

    txt_processes = tuple(txt_processes or (process,))
    txt_rates = tuple(
        txt_rates
        or (
            float(np.sum(nominal)),
        )
    )
    stem.with_suffix(".txt").write_text(
        "process "
        + " ".join(f"{name}_sm" for name in txt_processes)
        + "\n"
        + "process "
        + " ".join(str(index) for index in range(len(txt_processes)))
        + "\n"
        + "rate "
        + " ".join(str(rate) for rate in txt_rates)
        + "\n"
        + extra_text,
        encoding="utf-8",
    )
    return stem


def read(module, stem, *, process="tllq", role="private"):
    return module.read_base_category_card(
        module.CardFiles(
            stem.with_suffix(".root"),
            stem.with_suffix(".txt"),
        ),
        process,
        base_channel="2lss_p",
        role=role,
    )


def test_default_inventory_uses_all_registry_json_order(tmp_path):
    module = load_module()
    for category in LEGACY_MISSING_PARTON_BASE_CHANNELS:
        stem = tmp_path / f"ttx_multileptons-{category}_njets"
        stem.with_suffix(".root").touch()
        stem.with_suffix(".txt").touch()

    inventory = module.discover_card_pairs(tmp_path, role="private")

    expected = load_registry_payload_layout().ordered_base_categories
    assert list(inventory) == list(expected)
    assert len(inventory) == 34
    assert inventory.unused_categories == ()


def test_missing_root_or_txt_partner_fails(tmp_path):
    module = load_module()
    (tmp_path / "ttx_multileptons-a_njets.root").touch()

    with pytest.raises(ValueError, match="missing_txt"):
        module.discover_card_pairs(
            tmp_path,
            expected_categories=("a",),
            role="private",
        )


def test_inventory_failure_reports_every_missing_selected_category(tmp_path):
    for extension in ("root", "txt"):
        (tmp_path / f"ttx_multileptons-a_njets.{extension}").touch()

    module = load_module()
    with pytest.raises(ValueError) as exc_info:
        module.discover_card_pairs(
            tmp_path,
            expected_categories=("a", "b", "c"),
            role="private",
        )

    message = str(exc_info.value)
    assert "missing_root=['b', 'c']" in message
    assert "missing_txt=['b', 'c']" in message


def test_duplicate_semantic_category_fails():
    module = load_module()
    paths = (
        Path("/one/ttx_multileptons-a_njets.root"),
        Path("/two/ttx_multileptons-a_njets.root"),
    )

    with pytest.raises(ValueError, match="Duplicate ROOT"):
        module._index_card_paths(
            paths,
            extension="root",
            var="njets",
        )


def test_unselected_category_is_ignored_and_reported(tmp_path):
    module = load_module()
    for category in ("a", "unexpected"):
        stem = tmp_path / f"ttx_multileptons-{category}_njets"
        stem.with_suffix(".root").touch()
        stem.with_suffix(".txt").touch()

    inventory = module.discover_card_pairs(
        tmp_path,
        expected_categories=("a",),
        role="central",
    )

    assert list(inventory) == ["a"]
    assert inventory.unused_categories == ("unexpected",)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_nonfinite_nominal_shape_content_fails(tmp_path, bad_value):
    module = load_module()
    nominal = np.arange(1.0, 9.0)
    nominal[3] = bad_value
    stem = write_card(tmp_path, nominal=nominal, txt_rates=(36.0,))

    with pytest.raises(ValueError, match="Malformed private template"):
        read(module, stem)


def test_nonfinite_shape_content_fails(tmp_path):
    module = load_module()
    shape = np.arange(1.0, 9.0)
    shape[4] = np.nan
    stem = write_card(
        tmp_path,
        shapes=(("shapeUp", shape, np.arange(9.0)),),
    )

    with pytest.raises(ValueError, match="Malformed private template"):
        read(module, stem)


def test_shape_count_or_edges_must_match_nominal(tmp_path):
    module = load_module()
    stem = write_card(
        tmp_path,
        shapes=(("shapeUp", np.arange(7.0), np.arange(8.0)),),
    )

    with pytest.raises(ValueError, match="Incompatible private shape template"):
        read(module, stem)

    other = tmp_path / "other"
    other.mkdir()
    stem = write_card(
        other,
        shapes=(("shapeUp", np.arange(1.0, 9.0), np.arange(8.0, 17.0)),),
    )
    with pytest.raises(ValueError, match="Incompatible private shape template"):
        read(module, stem)


def test_root_txt_rate_mismatch_fails_with_tolerance_context(tmp_path):
    module = load_module()
    stem = write_card(tmp_path, txt_rates=(36.1,))

    with pytest.raises(ValueError, match="ROOT/TXT nominal disagreement") as exc_info:
        read(module, stem)

    message = str(exc_info.value)
    assert "rel_tol=1e-06" in message
    assert "abs_tol=1e-06" in message


def test_root_txt_rounding_within_documented_tolerance_is_accepted(tmp_path):
    module = load_module()
    nominal = np.zeros(8)
    stem = write_card(tmp_path, nominal=nominal, txt_rates=(0.9e-6,))

    card_data, process_name = read(module, stem)

    assert process_name == "tllq_sm"
    assert np.array_equal(card_data.nominal_values, nominal)


def test_preexisting_missing_parton_txt_row_fails(tmp_path):
    module = load_module()
    stem = write_card(
        tmp_path,
        extra_text="missing_parton lnN 1.10\n",
    )

    with pytest.raises(ValueError, match="pre-existing missing_parton"):
        read(module, stem)


def test_preexisting_missing_parton_root_object_fails(tmp_path):
    module = load_module()
    stem = write_card(
        tmp_path,
        extra_root_objects=(
            (
                "tllq_sm_missing_partonUp",
                np.arange(1.0, 9.0),
                np.arange(9.0),
            ),
        ),
    )

    with pytest.raises(ValueError, match="pre-existing missing_parton"):
        read(module, stem)


def test_physical_njet_axis_is_exactly_eight_bins(tmp_path):
    module = load_module()
    stem = write_card(
        tmp_path,
        nominal=np.arange(1.0, 8.0),
        edges=np.arange(8.0),
        txt_rates=(28.0,),
    )
    card_data, _ = read(module, stem)

    with pytest.raises(ValueError, match="expected eight"):
        module.validate_physical_njet_axis(
            card_data.nominal_values,
            card_data.bin_edges,
            base_channel="2lss_p",
            role="private",
        )


def test_central_private_bin_edges_must_match():
    module = load_module()
    parsed = module.parsed_card(
        process_names=("tllq_sm",),
        rates=(36.0,),
        rate_systematics=(),
    )
    private = module.base_category_card_data(
        nominal_values=np.arange(1.0, 9.0),
        shape_values=(),
        bin_edges=np.arange(9.0),
        parsed_txt=parsed,
    )
    central = module.base_category_card_data(
        nominal_values=np.arange(1.0, 9.0),
        shape_values=(),
        bin_edges=np.arange(1.0, 10.0),
        parsed_txt=parsed,
    )

    with pytest.raises(ValueError, match="physical-njet layout|bin-edge mismatch"):
        module.build_category_payload(
            base_channel="2lss_p",
            private_card=private,
            central_card=central,
            layout=load_registry_payload_layout().categories_by_name["2lss_p"],
        )
