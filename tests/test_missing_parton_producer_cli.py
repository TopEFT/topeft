from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_producer_cli_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_config(module, *arguments):
    parser = module.build_arg_parser()
    return module.resolve_config(parser.parse_args(list(arguments)))


def test_historical_flags_remain_accepted_and_resolve_legacy_paths():
    module = load_module()

    config = parse_config(
        module,
        "--years",
        "2022",
        "2023",
        "--time",
        "--output-path",
        "/tmp/diagnostics",
        "--var",
        "njets",
        "--dry-run",
    )

    assert config.input_mode == "legacy"
    assert config.central_card_dir == module.LEGACY_CENTRAL_CARD_DIR
    assert config.private_card_dir == module.LEGACY_PRIVATE_CARD_DIR
    assert config.years == ("2022", "2023")
    assert config.time is True
    assert config.output_path == Path("/tmp/diagnostics")
    assert config.var == "njets"


def test_explicit_card_directories_override_legacy_resolution(tmp_path):
    module = load_module()
    central = tmp_path / "central"
    private = tmp_path / "private"

    config = parse_config(
        module,
        "--central-card-dir",
        str(central),
        "--private-card-dir",
        str(private),
        "--dry-run",
    )

    assert config.input_mode == "explicit"
    assert config.central_card_dir == central
    assert config.private_card_dir == private


def test_candidate_cli_aliases_are_additive(tmp_path):
    module = load_module()
    output = tmp_path / "payload.root"

    config = parse_config(
        module,
        "--central-dir",
        str(tmp_path / "central"),
        "--private-dir",
        str(tmp_path / "private"),
        "--output-payload",
        str(output),
        "--allow-overwrite",
        "--dry-run",
    )

    assert config.output_file == output
    assert config.overwrite is True


def test_explicit_card_directories_must_be_supplied_together(tmp_path):
    module = load_module()

    with pytest.raises(module.ConfigError, match="must be supplied together"):
        parse_config(
            module,
            "--central-card-dir",
            str(tmp_path / "central"),
        )


def test_non_njets_observable_is_rejected_after_argument_parsing():
    module = load_module()

    with pytest.raises(module.ConfigError, match="only for --var njets"):
        parse_config(module, "--var", "ptz")


def test_existing_output_is_rejected_without_overwrite(tmp_path):
    module = load_module()
    output = tmp_path / "payload.root"
    output.write_bytes(b"existing")

    with pytest.raises(module.ConfigError, match="Refusing to overwrite"):
        parse_config(module, "--output-file", str(output), "--dry-run")


def test_existing_output_is_accepted_only_with_overwrite(tmp_path):
    module = load_module()
    output = tmp_path / "payload.root"
    output.write_bytes(b"existing")

    config = parse_config(
        module,
        "--output-file",
        str(output),
        "--overwrite",
        "--dry-run",
    )

    assert config.output_file == output
    assert config.overwrite is True


def test_dry_run_builds_complete_plan_and_never_calls_writer(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    calls = []
    expected_plan = module.payload_plan(categories=())
    config = module.ResolvedConfig(
        central_card_dir=tmp_path / "central",
        private_card_dir=tmp_path / "private",
        output_file=tmp_path / "payload.root",
        output_path=tmp_path,
        input_mode="explicit",
        dry_run=True,
        overwrite=False,
        years=("2022",),
        time=False,
        var="njets",
        sr_registry=module.DEFAULT_SR_REGISTRY,
    )

    def build_plan(observed_config):
        calls.append(("build", observed_config))
        return expected_plan

    def fail_writer(*args, **kwargs):
        raise AssertionError("dry-run attempted to write a payload")

    monkeypatch.setattr(module, "build_payload_plan", build_plan)
    monkeypatch.setattr(module, "write_legacy_payload_atomic", fail_writer)

    plan, output_sha256 = module.run_producer(config)

    assert calls == [("build", config)]
    assert plan is expected_plan
    assert output_sha256 is None
    assert not config.output_file.exists()


def test_dry_run_plan_prints_neutralized_physical_bins():
    module = load_module()
    category = module.category_payload_plan(
        base_channel="2lss_m_1tau_onZ",
        central_process_name="tZq_sm",
        private_process_name="tllq_sm",
        central_integral=1.0,
        private_integral=0.0,
        neutralized_physical_njets=(2, 7),
        stored_values=np.zeros(7),
    )

    plan = module.payload_plan(categories=(category,))
    printable = plan.to_printable_dict()

    assert printable["neutralized_bins"] == [
        {"base_channel": "2lss_m_1tau_onZ", "physical_njet": 2},
        {"base_channel": "2lss_m_1tau_onZ", "physical_njet": 7},
    ]
    assert printable["categories"][0]["neutralized_physical_njets"] == [2, 7]


def test_invalid_input_leaves_existing_output_byte_for_byte_unchanged(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    output = tmp_path / "payload.root"
    original = b"pre-existing-payload"
    output.write_bytes(original)
    config = module.ResolvedConfig(
        central_card_dir=tmp_path / "central",
        private_card_dir=tmp_path / "private",
        output_file=output,
        output_path=tmp_path,
        input_mode="explicit",
        dry_run=False,
        overwrite=True,
        years=("2022",),
        time=False,
        var="njets",
        sr_registry=module.DEFAULT_SR_REGISTRY,
    )
    monkeypatch.setattr(
        module,
        "build_payload_plan",
        lambda _: (_ for _ in ()).throw(ValueError("invalid input")),
    )

    with pytest.raises(ValueError, match="invalid input"):
        module.run_producer(config)

    assert output.read_bytes() == original


def test_plan_reports_missing_categories_from_both_source_roles_before_write(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    config = parse_config(
        module,
        "--output-file",
        str(tmp_path / "payload.root"),
    )

    def discover(_, *, role, **kwargs):
        missing = "missing_central" if role == "central" else "missing_private"
        return module.selected_card_inventory(
            pairs={},
            unused_categories=(),
            missing_root_categories=(missing,),
            missing_txt_categories=(missing,),
        )

    monkeypatch.setattr(module, "discover_card_pairs", discover)
    monkeypatch.setattr(
        module,
        "read_base_category_card",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("card read preceded complete inventory validation")
        ),
    )
    monkeypatch.setattr(
        module,
        "write_legacy_payload_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("writer was called")
        ),
    )

    with pytest.raises(ValueError, match="Missing required") as exc_info:
        module.run_producer(config)

    message = str(exc_info.value)
    assert "missing_central" in message
    assert "missing_private" in message
    assert not config.output_file.exists()


@pytest.mark.parametrize(
    "sr_registry",
    (
        "TOP22_006_CH_LST_SR",
        "TAU_CH_LST_SR",
        "OFFZ_SPLIT_CH_LST_SR",
        "FWD_CH_LST_SR",
        "ALL_CH_LST_SR",
    ),
)
def test_sr_registry_choices_resolve_and_are_recorded(
    tmp_path,
    sr_registry,
):
    module = load_module()
    output = tmp_path / f"{sr_registry}.root"

    config = parse_config(
        module,
        "--sr-registry",
        sr_registry,
        "--output-file",
        str(output),
        "--dry-run",
    )

    printable = config.to_printable_dict()
    assert config.sr_registry == sr_registry
    assert printable["sr_registry"] == sr_registry
    assert Path(printable["ch_lst_json"]).name == "ch_lst.json"
    assert printable["ch_lst_sha256"] == hashlib.sha256(
        Path(printable["ch_lst_json"]).read_bytes()
    ).hexdigest()


def test_nondefault_registry_requires_explicit_output(tmp_path):
    module = load_module()

    with pytest.raises(module.ConfigError, match="requires an explicit --output-file"):
        parse_config(module, "--sr-registry", "FWD_CH_LST_SR", "--dry-run")

    config = parse_config(
        module,
        "--sr-registry",
        "FWD_CH_LST_SR",
        "--output-file",
        str(tmp_path / "fwd.root"),
        "--dry-run",
    )
    assert config.sr_registry == "FWD_CH_LST_SR"


def test_producer_parser_rejects_unknown_registry():
    module = load_module()

    with pytest.raises(SystemExit):
        module.build_arg_parser().parse_args(
            ["--sr-registry", "UNKNOWN_SR_REGISTRY"]
        )


def test_nondefault_dry_run_is_allowed_and_never_writes(monkeypatch, tmp_path):
    module = load_module()
    config = parse_config(
        module,
        "--sr-registry",
        "FWD_CH_LST_SR",
        "--output-file",
        str(tmp_path / "fwd.root"),
        "--dry-run",
    )
    expected_plan = module.payload_plan(categories=())
    monkeypatch.setattr(module, "build_payload_plan", lambda observed: expected_plan)
    monkeypatch.setattr(
        module,
        "write_legacy_payload_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run attempted to write")
        ),
    )

    plan, output_sha256 = module.run_producer(config)

    assert plan is expected_plan
    assert output_sha256 is None
    assert not config.output_file.exists()


def test_nondefault_generation_reaches_plan_and_registry_bound_writer(
    monkeypatch,
    tmp_path,
):
    module = load_module()
    config = parse_config(
        module,
        "--sr-registry",
        "FWD_CH_LST_SR",
        "--output-file",
        str(tmp_path / "fwd.root"),
    )
    expected_plan = module.payload_plan(
        categories=(),
        registry="FWD_CH_LST_SR",
    )
    calls = []
    monkeypatch.setattr(module, "build_payload_plan", lambda _: expected_plan)

    def writer(output_file, payload_values, **kwargs):
        calls.append((output_file, payload_values, kwargs))
        return "synthetic-sha256"

    monkeypatch.setattr(module, "write_legacy_payload_atomic", writer)

    plan, output_sha256 = module.run_producer(config)

    assert plan is expected_plan
    assert output_sha256 == "synthetic-sha256"
    assert calls == [
        (
            config.output_file,
            {},
            {"overwrite": False, "sr_registry": "FWD_CH_LST_SR"},
        )
    ]
    assert not config.output_file.exists()


@pytest.mark.parametrize(
    "sr_registry",
    (
        "TOP22_006_CH_LST_SR",
        "TAU_CH_LST_SR",
        "OFFZ_SPLIT_CH_LST_SR",
        "FWD_CH_LST_SR",
        "ALL_CH_LST_SR",
    ),
)
def test_selected_registry_builds_only_selected_plan_and_reports_unused_inputs(
    monkeypatch,
    tmp_path,
    sr_registry,
):
    module = load_module()
    config = parse_config(
        module,
        "--sr-registry",
        sr_registry,
        "--output-file",
        str(tmp_path / f"{sr_registry}.root"),
    )
    expected_layout = module.load_registry_payload_layout(sr_registry)
    observed_inventories = {}

    def discover(_, *, expected_categories, var, role, allow_missing):
        assert allow_missing is True
        observed_inventories[role] = tuple(expected_categories)
        return module.selected_card_inventory(
            pairs={
                category: module.CardFiles(
                    tmp_path / f"{role}-{category}.root",
                    tmp_path / f"{role}-{category}.txt",
                )
                for category in expected_categories
            },
            unused_categories=(f"unused_{role}",),
        )

    def read_card(_, process, *, base_channel, role):
        if role == "private":
            nominal = np.arange(10.0, 18.0)
            process_name = "tllq_sm"
        else:
            nominal = np.arange(1.0, 9.0)
            process_name = "tZq_sm"
        card = module.base_category_card_data(
            nominal_values=nominal,
            shape_values=(),
            bin_edges=np.arange(9.0),
            parsed_txt=module.parsed_card(
                process_names=(process_name,),
                rates=(float(np.sum(nominal)),),
                rate_systematics=(),
            ),
        )
        return card, process_name

    writer_calls = []
    monkeypatch.setattr(module, "discover_card_pairs", discover)
    monkeypatch.setattr(module, "read_base_category_card", read_card)
    monkeypatch.setattr(
        module,
        "write_legacy_payload_atomic",
        lambda output_file, payload_values, **kwargs: writer_calls.append(
            (output_file, payload_values, kwargs)
        )
        or "synthetic-sha256",
    )

    plan, output_sha256 = module.run_producer(config)

    expected_categories = expected_layout.ordered_base_categories
    assert observed_inventories == {
        "central": expected_categories,
        "private": expected_categories,
    }
    assert tuple(plan.values_by_category) == expected_categories
    assert plan.registry == sr_registry
    assert plan.unused_central_categories == ("unused_central",)
    assert plan.unused_private_categories == ("unused_private",)
    assert [
        len(plan.values_by_category[category]) for category in expected_categories
    ] == [
        expected_layout.categories_by_name[category].public_array_length
        for category in expected_categories
    ]
    assert writer_calls[0][0] == config.output_file
    assert tuple(writer_calls[0][1]) == expected_categories
    assert writer_calls[0][2]["sr_registry"] == sr_registry
    assert output_sha256 == "synthetic-sha256"
    if sr_registry == "ALL_CH_LST_SR":
        plans_by_category = {
            category.base_channel: category for category in plan.categories
        }
        for base_category in (
            "3l_m_offZ_1b_fwd",
            "3l_p_offZ_1b_fwd",
        ):
            assert plans_by_category[
                base_category
            ].terminal_source_physical_njets == (4, 5, 6, 7)
            assert len(plans_by_category[base_category].stored_values) == 5


def test_default_registry_is_not_rejected_by_generation_guard(monkeypatch, tmp_path):
    module = load_module()
    config = parse_config(
        module,
        "--output-file",
        str(tmp_path / "all.root"),
    )
    expected_plan = module.payload_plan(categories=())
    monkeypatch.setattr(module, "build_payload_plan", lambda observed: expected_plan)
    monkeypatch.setattr(
        module,
        "write_legacy_payload_atomic",
        lambda *_args, **_kwargs: "synthetic-sha256",
    )

    plan, output_sha256 = module.run_producer(config)

    assert plan is expected_plan
    assert output_sha256 == "synthetic-sha256"
    assert not config.output_file.exists()


def test_help_documents_legacy_and_explicit_modes_deterministically():
    module = load_module()

    help_text = module.build_arg_parser().format_help()

    for option in (
        "--years",
        "--time",
        "--output-path",
        "--var",
        "--central-card-dir",
        "--private-card-dir",
        "--output-file",
        "--dry-run",
        "--overwrite",
        "--sr-registry",
    ):
        assert option in help_text
    assert "registry-selected missing-parton ROOT payload" in help_text
    assert "default: ALL_CH_LST_SR" in " ".join(help_text.split())
