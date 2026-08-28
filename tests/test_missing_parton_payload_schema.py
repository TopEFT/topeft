from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import uproot

from topeft.modules.missing_parton_contract import (
    LEGACY_MISSING_PARTON_BRANCH,
    load_registry_payload_layout,
)


module_path = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)
payload_directory = (
    Path(__file__).resolve().parents[1]
    / "topeft"
    / "data"
    / "missing_parton"
)
current_payload_paths = {
    "run2": payload_directory / "missing_parton_run2.root",
    "run3": payload_directory / "missing_parton_run3.root",
}
current_payload_semantic_digests = {
    "run2": "936a7316894257a5dcac31c345c60ea273d27cb672c71fbce6382fe5df534a24",
    "run3": "8ddf59420ed47828551803ef7b168ae1dec02e1402418801ab5ec2efc90de332",
}
run3_test_mmerged_payload_commit = (
    "2469053a8d7ab0b42c86c68000f51b6e7f6dafff"
)
run3_test_mmerged_payload_semantic_digests = {
    "run2": "08895d0ba12fab53609b8732e992bb5da2736e215c70f6b7c9db97af6b3bc5e8",
    "run3": "884d10b315f56444a0d205dfcce0cd5b19180d9967acaa3d0c06f17633804ff5",
}
run3_test_mmerged_tree_order = (
    "2los_onZ_1tau",
    "2lss_4t_m",
    "2lss_4t_p",
    "2lss_fwd_m",
    "2lss_fwd_p",
    "2lss_m_1tau_offZ",
    "2lss_m_1tau_onZ",
    "2lss_m",
    "2lss_p_1tau_offZ",
    "2lss_p_1tau_onZ",
    "2lss_p",
    "3l_1tau_1b",
    "3l_1tau_2b",
    "3l_m_offZ_1b_fwd",
    "3l_m_offZ_2b_fwd",
    "3l_m_offZ_high_1b",
    "3l_m_offZ_high_2b",
    "3l_m_offZ_low_1b",
    "3l_m_offZ_low_2b",
    "3l_m_offZ_none_1b",
    "3l_m_offZ_none_2b",
    "3l_onZ_1b_fwd",
    "3l_onZ_1b",
    "3l_onZ_2b_fwd",
    "3l_onZ_2b",
    "3l_p_offZ_1b_fwd",
    "3l_p_offZ_2b_fwd",
    "3l_p_offZ_high_1b",
    "3l_p_offZ_high_2b",
    "3l_p_offZ_low_1b",
    "3l_p_offZ_low_2b",
    "3l_p_offZ_none_1b",
    "3l_p_offZ_none_2b",
    "4l",
)
run3_test_mmerged_array_lengths = {
    "2los_onZ_1tau": 4,
    "2lss_4t_m": 8,
    "2lss_4t_p": 8,
    "2lss_fwd_m": 8,
    "2lss_fwd_p": 8,
    "2lss_m_1tau_offZ": 7,
    "2lss_m_1tau_onZ": 7,
    "2lss_m": 8,
    "2lss_p_1tau_offZ": 7,
    "2lss_p_1tau_onZ": 7,
    "2lss_p": 8,
    "3l_1tau_1b": 6,
    "3l_1tau_2b": 6,
    "3l_m_offZ_1b_fwd": 6,
    "3l_m_offZ_2b_fwd": 5,
    "3l_m_offZ_high_1b": 6,
    "3l_m_offZ_high_2b": 6,
    "3l_m_offZ_low_1b": 6,
    "3l_m_offZ_low_2b": 6,
    "3l_m_offZ_none_1b": 6,
    "3l_m_offZ_none_2b": 6,
    "3l_onZ_1b_fwd": 5,
    "3l_onZ_1b": 6,
    "3l_onZ_2b_fwd": 5,
    "3l_onZ_2b": 6,
    "3l_p_offZ_1b_fwd": 6,
    "3l_p_offZ_2b_fwd": 5,
    "3l_p_offZ_high_1b": 6,
    "3l_p_offZ_high_2b": 6,
    "3l_p_offZ_low_1b": 6,
    "3l_p_offZ_low_2b": 6,
    "3l_p_offZ_none_1b": 6,
    "3l_p_offZ_none_2b": 6,
    "4l": 5,
}
run3_test_mmerged_terminal_population_by_index = {
    "2los_onZ_1tau": {3: ">=3"},
    "2lss_4t_m": {7: ">=7"},
    "2lss_4t_p": {7: ">=7"},
    "2lss_fwd_m": {7: ">=7"},
    "2lss_fwd_p": {7: ">=7"},
    "2lss_m_1tau_offZ": {6: ">=6"},
    "2lss_m_1tau_onZ": {6: ">=6"},
    "2lss_m": {7: ">=7"},
    "2lss_p_1tau_offZ": {6: ">=6"},
    "2lss_p_1tau_onZ": {6: ">=6"},
    "2lss_p": {7: ">=7"},
    "3l_1tau_1b": {5: ">=5"},
    "3l_1tau_2b": {5: ">=5"},
    "3l_m_offZ_1b_fwd": {4: "=4", 5: ">=5"},
    "3l_m_offZ_2b_fwd": {4: ">=4"},
    "3l_m_offZ_high_1b": {5: ">=5"},
    "3l_m_offZ_high_2b": {5: ">=5"},
    "3l_m_offZ_low_1b": {5: ">=5"},
    "3l_m_offZ_low_2b": {5: ">=5"},
    "3l_m_offZ_none_1b": {5: ">=5"},
    "3l_m_offZ_none_2b": {5: ">=5"},
    "3l_onZ_1b_fwd": {4: ">=4"},
    "3l_onZ_1b": {5: ">=5"},
    "3l_onZ_2b_fwd": {4: ">=4"},
    "3l_onZ_2b": {5: ">=5"},
    "3l_p_offZ_1b_fwd": {4: "=4", 5: ">=5"},
    "3l_p_offZ_2b_fwd": {4: ">=4"},
    "3l_p_offZ_high_1b": {5: ">=5"},
    "3l_p_offZ_high_2b": {5: ">=5"},
    "3l_p_offZ_low_1b": {5: ">=5"},
    "3l_p_offZ_low_2b": {5: ">=5"},
    "3l_p_offZ_none_1b": {5: ">=5"},
    "3l_p_offZ_none_2b": {5: ">=5"},
    "4l": {4: ">=4"},
}


@dataclass(frozen=True)
class payload_schema_contract:
    name: str
    tree_order: tuple[str, ...]
    array_lengths: Mapping[str, int]
    branch_name: str
    branch_typename: str
    branch_dtype: str
    terminal_population_by_index: Mapping[str, Mapping[int, str]]
    provenance_semantic_digests: Mapping[str, str]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_payload_schema_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_payload_schema():
    layout = load_registry_payload_layout("ALL_CH_LST_SR")
    terminal_population_by_index = {}
    for category in layout.categories:
        terminal_label = (
            f">={category.terminal_threshold}"
            if category.terminal_is_inclusive
            else f"={category.terminal_threshold}"
        )
        terminal_population_by_index[category.base_sr_category] = {
            category.terminal_threshold: terminal_label
        }
    return payload_schema_contract(
        name="current",
        tree_order=layout.ordered_base_categories,
        array_lengths=dict(layout.public_lengths),
        branch_name=LEGACY_MISSING_PARTON_BRANCH,
        branch_typename="double",
        branch_dtype="float64",
        terminal_population_by_index=terminal_population_by_index,
        provenance_semantic_digests=current_payload_semantic_digests,
    )


def run3_test_mmerged_payload_schema():
    return payload_schema_contract(
        name="run3_test_mmerged",
        tree_order=run3_test_mmerged_tree_order,
        array_lengths=run3_test_mmerged_array_lengths,
        branch_name=LEGACY_MISSING_PARTON_BRANCH,
        branch_typename="double",
        branch_dtype="float64",
        terminal_population_by_index=(
            run3_test_mmerged_terminal_population_by_index
        ),
        provenance_semantic_digests=(
            run3_test_mmerged_payload_semantic_digests
        ),
    )


def payload_semantic_digest(values):
    serialized = json.dumps(
        [[key, np.asarray(values[key]).tolist()] for key in sorted(values)],
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _validate_payload_against_schema(payload_path, schema):
    payload_path = Path(payload_path)
    values = {}
    with uproot.open(payload_path) as payload_file:
        observed_order = tuple(
            payload_file.keys(recursive=False, cycle=False)
        )
        if observed_order != schema.tree_order:
            raise ValueError(
                f"Invalid {schema.name} missing-parton tree order: "
                f"expected={schema.tree_order!r}, observed={observed_order!r}."
            )
        if set(observed_order) != set(schema.tree_order):
            raise ValueError(
                f"Invalid {schema.name} missing-parton tree set."
            )
        classnames = payload_file.classnames(
            recursive=False,
            cycle=False,
        )
        for category in schema.tree_order:
            if classnames[category] != "TTree":
                raise ValueError(
                    f"Invalid {schema.name} object type for {category!r}: "
                    f"{classnames[category]!r}."
                )
            tree = payload_file[category]
            if tree.keys() != [schema.branch_name]:
                raise ValueError(
                    f"Invalid {schema.name} branch set for {category!r}: "
                    f"{tree.keys()!r}."
                )
            branch = tree[schema.branch_name]
            if branch.typename != schema.branch_typename:
                raise ValueError(
                    f"Invalid {schema.name} branch type for {category!r}: "
                    f"{branch.typename!r}."
                )
            array = branch.array(library="np")
            if str(array.dtype) != schema.branch_dtype:
                raise ValueError(
                    f"Invalid {schema.name} branch dtype for {category!r}: "
                    f"{array.dtype!s}."
                )
            if len(array) != schema.array_lengths[category]:
                raise ValueError(
                    f"Invalid {schema.name} array length for {category!r}: "
                    f"expected={schema.array_lengths[category]}, "
                    f"observed={len(array)}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Non-finite {schema.name} value for {category!r}."
                )
            values[category] = array
    return values


def validate_current_missing_parton_schema(payload_path):
    return _validate_payload_against_schema(
        payload_path,
        current_payload_schema(),
    )


def validate_current_missing_parton_payload(payload_path, *, era):
    values = validate_current_missing_parton_schema(payload_path)
    observed_digest = payload_semantic_digest(values)
    expected_digest = current_payload_semantic_digests[era]
    if observed_digest != expected_digest:
        raise ValueError(
            f"Invalid current {era} semantic digest: "
            f"expected={expected_digest}, observed={observed_digest}."
        )
    return values


def validate_run3_test_mmerged_missing_parton_payload(payload_path):
    return _validate_payload_against_schema(
        payload_path,
        run3_test_mmerged_payload_schema(),
    )


def synthetic_payload(*, offset=0.0):
    schema = current_payload_schema()
    return {
        category: np.linspace(
            offset,
            offset + 0.01 * (schema.array_lengths[category] - 1),
            schema.array_lengths[category],
            dtype=np.float64,
        )
        for category in schema.tree_order
    }


def write_synthetic_payload(
    output,
    schema,
    *,
    tree_order=None,
    length_overrides=None,
    omitted_categories=(),
    extra_categories=(),
    branch_type="float64",
    nonfinite_category=None,
):
    tree_order = schema.tree_order if tree_order is None else tree_order
    length_overrides = {} if length_overrides is None else length_overrides
    omitted_categories = set(omitted_categories)
    with uproot.recreate(output) as payload_file:
        for category in tree_order:
            if category in omitted_categories:
                continue
            length = length_overrides.get(
                category,
                schema.array_lengths[category],
            )
            array = np.arange(length, dtype=np.dtype(branch_type))
            if category == nonfinite_category:
                array[-1] = np.nan
            tree = payload_file.mktree(
                category,
                {schema.branch_name: branch_type},
            )
            tree.extend({schema.branch_name: array})
        for category in extra_categories:
            tree = payload_file.mktree(
                category,
                {schema.branch_name: branch_type},
            )
            tree.extend(
                {schema.branch_name: np.asarray([0.0], dtype=branch_type)}
            )


def test_synthetic_writer_produces_exact_current_34_tree_schema(tmp_path):
    module = load_module()
    output = tmp_path / "missing_parton.root"

    output_sha256 = module.write_legacy_payload_atomic(
        output,
        synthetic_payload(),
    )
    validated = validate_current_missing_parton_schema(output)
    schema = current_payload_schema()

    assert len(output_sha256) == 64
    assert tuple(validated) == schema.tree_order
    with uproot.open(output) as payload_file:
        assert tuple(
            payload_file.keys(recursive=False, cycle=False)
        ) == schema.tree_order
        assert set(
            payload_file.classnames(recursive=False, cycle=False).values()
        ) == {"TTree"}


def test_writer_has_no_extra_directories_or_scalar_histograms(tmp_path):
    module = load_module()
    output = tmp_path / "missing_parton.root"
    module.write_legacy_payload_atomic(output, synthetic_payload())

    with uproot.open(output) as payload_file:
        classnames = payload_file.classnames(
            recursive=True,
            cycle=False,
        )

    assert all(
        classname in {"TTree", "TBranch"}
        for classname in classnames.values()
    )
    assert not any(
        classname.startswith("TH1") for classname in classnames.values()
    )
    assert sum(
        classname == "TTree" for classname in classnames.values()
    ) == 34


def test_132_key_scalar_schema_is_rejected(tmp_path):
    output = tmp_path / "scalar_payload.root"
    with uproot.recreate(output) as payload_file:
        for index in range(132):
            payload_file[f"final_channel_{index}"] = (
                np.asarray([0.1]),
                np.asarray([0.0, 1.0]),
            )

    with pytest.raises(ValueError, match="current missing-parton tree order"):
        validate_current_missing_parton_schema(output)


def test_existing_output_rejected_without_overwrite_and_replaced_with_opt_in(
    tmp_path,
):
    module = load_module()
    output = tmp_path / "missing_parton.root"
    first_sha256 = module.write_legacy_payload_atomic(
        output,
        synthetic_payload(offset=0.0),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.write_legacy_payload_atomic(
            output,
            synthetic_payload(offset=0.2),
        )

    second_sha256 = module.write_legacy_payload_atomic(
        output,
        synthetic_payload(offset=0.2),
        overwrite=True,
    )

    assert first_sha256 != second_sha256
    validated = validate_current_missing_parton_schema(output)
    assert validated["2los_onZ_1tau"][0] == pytest.approx(0.2)


def test_invalid_payload_leaves_existing_output_unchanged(tmp_path):
    module = load_module()
    output = tmp_path / "missing_parton.root"
    module.write_legacy_payload_atomic(output, synthetic_payload())
    original = output.read_bytes()
    invalid = synthetic_payload()
    invalid["2los_onZ_1tau"] = np.asarray([0.0, -0.1, 0.0, 0.0])

    with pytest.raises(ValueError, match="Negative stored"):
        module.write_legacy_payload_atomic(
            output,
            invalid,
            overwrite=True,
        )

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.root"))


@pytest.mark.parametrize("era", ("run2", "run3"))
def test_checked_in_current_payloads_match_strict_contract(era):
    validated = validate_current_missing_parton_payload(
        current_payload_paths[era],
        era=era,
    )

    assert len(validated) == 34
    assert tuple(validated) == current_payload_schema().tree_order


def test_current_forward_terminal_contract_is_complete_greater_equal_four():
    schema = current_payload_schema()
    for category in (
        "3l_m_offZ_1b_fwd",
        "3l_p_offZ_1b_fwd",
    ):
        assert schema.array_lengths[category] == 5
        assert schema.terminal_population_by_index[category] == {4: ">=4"}
        assert 5 not in schema.terminal_population_by_index[category]


def test_frozen_run3_test_mmerged_manifest_identifies_immutable_commit():
    schema = run3_test_mmerged_payload_schema()

    assert run3_test_mmerged_payload_commit == (
        "2469053a8d7ab0b42c86c68000f51b6e7f6dafff"
    )
    assert len(schema.tree_order) == 34
    assert set(schema.tree_order) == set(current_payload_schema().tree_order)
    assert schema.provenance_semantic_digests == {
        "run2": "08895d0ba12fab53609b8732e992bb5da2736e215c70f6b7c9db97af6b3bc5e8",
        "run3": "884d10b315f56444a0d205dfcce0cd5b19180d9967acaa3d0c06f17633804ff5",
    }


def test_run3_test_mmerged_synthetic_fixture_matches_strict_contract(tmp_path):
    schema = run3_test_mmerged_payload_schema()
    output = tmp_path / "run3_test_mmerged.root"
    write_synthetic_payload(output, schema)

    validated = validate_run3_test_mmerged_missing_parton_payload(output)

    assert tuple(validated) == schema.tree_order
    for category in (
        "3l_m_offZ_1b_fwd",
        "3l_p_offZ_1b_fwd",
    ):
        assert schema.array_lengths[category] == 6
        assert schema.terminal_population_by_index[category] == {
            4: "=4",
            5: ">=5",
        }


def test_current_contract_rejects_run3_test_mmerged_schema(tmp_path):
    output = tmp_path / "run3_test_mmerged.root"
    write_synthetic_payload(
        output,
        run3_test_mmerged_payload_schema(),
    )

    with pytest.raises(ValueError, match="current missing-parton tree order"):
        validate_current_missing_parton_schema(output)


def test_run3_test_mmerged_contract_rejects_current_schema(tmp_path):
    output = tmp_path / "current.root"
    write_synthetic_payload(output, current_payload_schema())

    with pytest.raises(
        ValueError,
        match="run3_test_mmerged missing-parton tree order",
    ):
        validate_run3_test_mmerged_missing_parton_payload(output)


def test_current_contract_rejects_alphabetically_reordered_payload(tmp_path):
    schema = current_payload_schema()
    output = tmp_path / "alphabetical.root"
    write_synthetic_payload(
        output,
        schema,
        tree_order=tuple(sorted(schema.tree_order)),
    )

    with pytest.raises(ValueError, match="current missing-parton tree order"):
        validate_current_missing_parton_schema(output)


def test_current_contract_rejects_extra_forward_index(tmp_path):
    schema = current_payload_schema()
    output = tmp_path / "length_six_current.root"
    write_synthetic_payload(
        output,
        schema,
        length_overrides={"3l_m_offZ_1b_fwd": 6},
    )

    with pytest.raises(ValueError, match="current array length"):
        validate_current_missing_parton_schema(output)


def test_run3_test_mmerged_contract_rejects_length_five_forward(tmp_path):
    schema = run3_test_mmerged_payload_schema()
    output = tmp_path / "length_five_legacy.root"
    write_synthetic_payload(
        output,
        schema,
        length_overrides={"3l_m_offZ_1b_fwd": 5},
    )

    with pytest.raises(
        ValueError,
        match="run3_test_mmerged array length",
    ):
        validate_run3_test_mmerged_missing_parton_payload(output)


@pytest.mark.parametrize(
    ("omitted_categories", "extra_categories"),
    [
        (("4l",), ()),
        ((), ("unsupported_extra_tree",)),
    ],
    ids=("missing_tree", "extra_tree"),
)
def test_current_contract_rejects_missing_or_extra_tree(
    tmp_path,
    omitted_categories,
    extra_categories,
):
    schema = current_payload_schema()
    output = tmp_path / "wrong_tree_set.root"
    write_synthetic_payload(
        output,
        schema,
        omitted_categories=omitted_categories,
        extra_categories=extra_categories,
    )

    with pytest.raises(ValueError, match="current missing-parton tree order"):
        validate_current_missing_parton_schema(output)


def test_current_contract_rejects_wrong_branch_type(tmp_path):
    output = tmp_path / "float32.root"
    write_synthetic_payload(
        output,
        current_payload_schema(),
        branch_type="float32",
    )

    with pytest.raises(ValueError, match="current branch type"):
        validate_current_missing_parton_schema(output)


def test_current_contract_rejects_nonfinite_value(tmp_path):
    output = tmp_path / "nonfinite.root"
    write_synthetic_payload(
        output,
        current_payload_schema(),
        nonfinite_category="4l",
    )

    with pytest.raises(ValueError, match="Non-finite current value"):
        validate_current_missing_parton_schema(output)
