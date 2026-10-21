import ast
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import awkward as ak
import hist
import numpy as np
import pytest
from coffea.analysis_tools import Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "topeft_run2"))

from analysis.topeft_run2.analysis_processor import (
    AnalysisProcessor,
    calculate_sm_sumw2_weights,
)
from analysis.topeft_run2.make_cr_and_sr_plots import (
    CR_GRP_PATTERNS,
    SR_GRP_PATTERNS,
    populate_group_map,
)
from analysis.topeft_run2.run_analysis_excluded_mc import (
    _configure_wq_categories,
    _resolve_category_group_selection,
)
from topcoffea.modules.utils import read_cfg_file
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.nonprompt_policy import (
    EXPLICIT_NONPROMPT_EXCLUSION,
    certify_active_nonprompt_policy,
)
from topeft.modules.sumw2_policy import resolve_sumw2_storage_policy, sumw2_target
from topeft.modules.ttgamma_photon_history import classify_conversion_overlap_sample
from topcoffea.modules.get_param_from_jsons import GetParam
from topeft.modules.paths import topeft_path


CFG_DIR = REPO_ROOT / "input_samples" / "cfgs"

RUN2_ERAS = {"2016APV", "2016", "2017", "2018"}
RUN3_ERAS = {"2022", "2022EE", "2023", "2023BPix"}


def load_probe_cfg(name):
    return read_cfg_file(str(CFG_DIR / name), cfg={})


def cfg_references(path):
    """Resolve active JSON rows with read_cfg_file's comment and path rules."""
    references = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        literal = raw.strip().split("#")[0].strip()
        if not literal or literal.startswith(("root:", "http:", "https:", "file://")):
            continue
        resolved = Path(os.path.join(path.parent, literal)).resolve(strict=True)
        with resolved.open() as stream:
            year = json.load(stream)["year"]
        references.append((resolved, year, line_number))
    return references


def background_difference(era):
    if era in RUN2_ERAS:
        cr_names = ("mc_background_samples_NDSkim.cfg", "mc_background_samples_cr_NDSkim.cfg")
        sr_names = ("mc_background_samples_NDSkim.cfg",)
    else:
        cr_names = (f"NDSkim_{era}_background_samples_cr.cfg",)
        sr_names = (f"NDSkim_{era}_background_samples.cfg",)
    cr_paths = {path for name in cr_names for path, year, _ in cfg_references(CFG_DIR / name) if year == era}
    sr_paths = {path for name in sr_names for path, year, _ in cfg_references(CFG_DIR / name) if year == era}
    return cr_paths - sr_paths


def signal_cfg_references(directory):
    return {
        path for cfg in directory.glob("*.cfg") if "signal" in cfg.name.lower()
        for path, _, _ in cfg_references(cfg)
    }


def reject_signal_origin(excluded_paths, signal_paths):
    shared = set(excluded_paths) & set(signal_paths)
    if shared:
        raise ValueError(f"Excluded CFG has signal-CFG JSON paths: {sorted(map(str, shared))}")


@pytest.mark.parametrize(
    ("cfg_name", "expected_eras"),
    (
        ("excluded_mc_run2_sr.cfg", RUN2_ERAS),
        ("excluded_mc_run3_sr.cfg", RUN3_ERAS),
    ),
)
def test_cfg_selects_exact_excluded_mc_universe(cfg_name, expected_eras):
    samples = load_probe_cfg(cfg_name)

    assert {sample["year"] for sample in samples.values()} == expected_eras
    assert all(sample["isData"] is False for sample in samples.values())
    assert all(sample["WCnames"] == [] for sample in samples.values())

    counts_by_era = {
        era: sum(sample["year"] == era for sample in samples.values())
        for era in expected_eras
    }
    assert len(set(counts_by_era.values())) == 1

    active_paths = {path for path, _, _ in cfg_references(CFG_DIR / cfg_name)}
    for era in expected_eras:
        assert {path for path, year, _ in cfg_references(CFG_DIR / cfg_name) if year == era} == background_difference(era)
    reject_signal_origin(active_paths, signal_cfg_references(CFG_DIR))


def test_prior_run3_cfg_snapshot_omitted_current_cr_only_members():
    old = subprocess.run(
        ["git", "show", "8e8eb23dfc2a3a421062ed68516a3e92dd574893:input_samples/cfgs/excluded_mc_run3_sr_probe.cfg"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    prior_path = CFG_DIR / "excluded_mc_run3_sr_probe.cfg"
    old_paths = {
        Path(os.path.join(prior_path.parent, line.strip().split("#")[0].strip())).resolve(strict=True)
        for line in old.splitlines()
        if line.strip().split("#")[0].strip()
        and not line.strip().split("#")[0].strip().startswith("file://")
    }
    omitted = set().union(*(background_difference(era) for era in RUN3_ERAS)) - old_paths
    assert omitted == {
        (REPO_ROOT / f"input_samples/sample_jsons/background_samples/ND_skim{era}/TTto4Q_NDSkim_{era}.json").resolve()
        for era in RUN3_ERAS
    }


def test_signal_origin_resolves_same_path_and_preserves_era_separation(tmp_path):
    first = tmp_path / "2022" / "plain_background.json"
    second = tmp_path / "2023" / "plain_background.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"year":"2022","histAxisName":"ordinary"}')
    second.write_text('{"year":"2023","histAxisName":"ordinary"}')
    signal_cfg = tmp_path / "plain_signal.cfg"
    excluded_cfg = tmp_path / "excluded_mc_test.cfg"
    signal_cfg.write_text(f"# ignored {second}\n{first}\n")
    excluded_cfg.write_text("2022/plain_background.json # same absolute path\n2023/plain_background.json\n")
    excluded_rows = cfg_references(excluded_cfg)
    signal_paths = signal_cfg_references(tmp_path)
    with pytest.raises(ValueError, match="signal-CFG JSON paths"):
        reject_signal_origin(
            {path for path, year, _ in excluded_rows if year == "2022"}, signal_paths
        )
    assert {path for path, year, _ in excluded_rows if year == "2023"}.isdisjoint(signal_paths)
    assert len({path for path, _, _ in excluded_rows}) == 2
    excluded_cfg.write_text(excluded_cfg.read_text() + "# 2022/plain_background.json\n2022/plain_background.json\n")
    assert len(cfg_references(excluded_cfg)) == 3


def test_added_ttto4q_uses_existing_inclusive_ttbar_split_role():
    samples = load_probe_cfg("excluded_mc_run3_sr.cfg")
    for era in RUN3_ERAS:
        dataset = f"TTto4Q_NDSkim_{era}"
        assert dataset in samples
        assert classify_conversion_overlap_sample(dataset, sample_role_policy="split") == "inclusive_ttbar"
        assert classify_conversion_overlap_sample(samples[dataset]["histAxisName"], sample_role_policy="split") == "inclusive_ttbar"


@pytest.mark.parametrize(
    "cfg_name",
    ("excluded_mc_run2_sr.cfg", "excluded_mc_run3_sr.cfg"),
)
def test_sumw2_policy_covers_every_dataset_and_reaches_processor(cfg_name):
    samples = load_probe_cfg(cfg_name)
    config = {
        "mode": "full_custom",
        "rules": [{"variables": ["njets"]}],
    }
    policy = resolve_sumw2_storage_policy(
        config,
        samples=samples,
        runtime_families=["njets"],
        axes_info=axes_info,
        axes_info_2d=axes_info_2d,
        sumw2_storage_present=True,
    )

    assert len(policy.resolved_targets) == len(samples)
    assert {target.dataset for target in policy.resolved_targets} == set(samples)
    assert {target.family for target in policy.resolved_targets} == {"njets"}
    if cfg_name == "excluded_mc_run2_sr.cfg":
        for dataset, process in (
            ("UL17_DY10to50_NDSkim", "DY10to50_centralUL17"),
            ("UL17_DY50_NDSkim", "DY50_centralUL17"),
        ):
            assert samples[dataset]["histAxisName"] == process
            assert sumw2_target(dataset, process, "njets") in policy.resolved_targets

    processor = AnalysisProcessor(
        samples,
        hist_lst=["njets"],
        fill_sumw2_hist=True,
        skip_control_regions=True,
        all_analysis=True,
        sumw2_policy=policy,
    )
    assert processor._selected_sumw2_families == frozenset({"njets"})
    assert "njets_sumw2" in processor.accumulator


def test_sumw2_policy_rejects_empty_and_incomplete_targets():
    samples = load_probe_cfg("excluded_mc_run2_sr.cfg")
    with pytest.raises(ValueError, match="SUMW2-E002"):
        resolve_sumw2_storage_policy(
            {"mode": "full_custom", "rules": []},
            samples=samples,
            runtime_families=["njets"],
            axes_info=axes_info,
            axes_info_2d=axes_info_2d,
            sumw2_storage_present=True,
        )

    dataset_names = sorted(samples)
    required = [
        sumw2_target(dataset, samples[dataset]["histAxisName"], "njets")
        for dataset in dataset_names
    ]
    with pytest.raises(ValueError, match="SUMW2-E010"):
        resolve_sumw2_storage_policy(
            {
                "mode": "full_custom",
                "rules": [
                    {
                        "dataset_names": [dataset_names[0]],
                        "variables": ["njets"],
                    }
                ],
            },
            samples=samples,
            runtime_families=["njets"],
            axes_info=axes_info,
            axes_info_2d=axes_info_2d,
            sumw2_storage_present=True,
            consumer_requirements=required,
        )


@pytest.mark.parametrize(
    ("groups", "expected_groups"),
    (
        (["2l", "2los_1tau"], {"2l", "2los_1tau"}),
        (["3l_onZ_tau"], {"3l_onZ_tau"}),
    ),
)
def test_reference_block_categories_use_native_sr_groups(groups, expected_groups):
    resolved = _resolve_category_group_selection(
        category_groups=groups,
        offz_3l_split=False,
        tau_h_analysis=False,
        fwd_analysis=False,
        all_analysis=True,
        skip_sr=False,
        skip_cr=True,
    )
    assert resolved["resolved_sr_block_name"] == "ALL_CH_LST_SR"
    assert set(resolved["sr_category_dict"]) == expected_groups
    if "2los_1tau" in expected_groups:
        p0 = resolved["sr_category_dict"]["2los_1tau"]
        assert p0["jet_lst"] == [">3"]
        assert p0["lep_chan_lst"][0][0] == "2los_onZ_1tau"


def test_ul17_dy_distinct_labels_preserve_nominal_and_second_moments():
    samples = load_probe_cfg("excluded_mc_run2_sr.cfg")
    low = samples["UL17_DY10to50_NDSkim"]
    high = samples["UL17_DY50_NDSkim"]
    assert low["histAxisName"] == "DY10to50_centralUL17"
    assert high["histAxisName"] == "DY50_centralUL17"

    low_weights = Weights(2)
    low_weights.add("norm", np.array([2.0, -1.0]) * 2.0)
    high_weights = Weights(2)
    high_weights.add("norm", np.array([3.0, -4.0]) * 0.5)
    contributions = np.concatenate((low_weights.weight(), high_weights.weight()))

    expected_nominal = 1.5
    expected_sumw2 = 26.25
    assert np.sum(contributions) == pytest.approx(expected_nominal)
    assert np.sum(calculate_sm_sumw2_weights(contributions)) == pytest.approx(
        expected_sumw2
    )
    assert expected_sumw2 != pytest.approx(expected_nominal**2)

    nominal = hist.Hist(
        hist.axis.StrCategory([], growth=True, name="process"),
        storage=hist.storage.Double(),
    )
    nominal.fill(process=[low["histAxisName"]] * 2, weight=low_weights.weight())
    nominal.fill(process=[high["histAxisName"]] * 2, weight=high_weights.weight())

    second_moment = hist.Hist(
        hist.axis.StrCategory([], growth=True, name="process"),
        storage=hist.storage.Double(),
    )
    second_moment.fill(
        process=[low["histAxisName"]] * 2,
        weight=calculate_sm_sumw2_weights(low_weights.weight()),
    )
    second_moment.fill(
        process=[high["histAxisName"]] * 2,
        weight=calculate_sm_sumw2_weights(high_weights.weight()),
    )

    assert set(nominal.axes[0]) == {low["histAxisName"], high["histAxisName"]}
    assert nominal[low["histAxisName"]] == pytest.approx(2.0)
    assert nominal[high["histAxisName"]] == pytest.approx(-0.5)
    assert second_moment[low["histAxisName"]] == pytest.approx(20.0)
    assert second_moment[high["histAxisName"]] == pytest.approx(6.25)
    assert np.sum(nominal.values()) == pytest.approx(expected_nominal)
    assert np.sum(second_moment.values()) == pytest.approx(expected_sumw2)


def _native_sample_type(processor_path, process, is_data=False):
    source = processor_path.read_text()
    tree = ast.parse(source)
    process_method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process"
    )
    for index, node in enumerate(process_method.body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "sampleType"
            and isinstance(node.value, ast.Constant)
            and node.value.value == "prompt"
        ):
            branch = process_method.body[index + 1]
            assert isinstance(branch, ast.If)
            break
    else:
        pytest.fail(f"Native sampleType branch missing from {processor_path}")

    # Compile the complete original assignment and conditional, with no rewritten predicate.
    block = ast.fix_missing_locations(ast.Module(body=[node, branch], type_ignores=[]))
    lookup = GetParam(topeft_path("params/params.json"))
    scope = {
        "histAxisName": process,
        "isData": is_data,
        "get_te_param": lookup,
    }
    exec(compile(block, str(processor_path), "exec"), scope)
    return scope["sampleType"]


def _native_matching_mask(lepton_count, sample_type):
    source_path = REPO_ROOT / "topeft" / "modules" / "event_selection.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == f"add{lepton_count}lMaskAndSFs"
    )
    branch = next(
        node for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "sampleType"
    )
    block = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    flavors = ([1] * lepton_count, [1] * (lepton_count - 1) + [22], [5] * lepton_count)
    leptons = ak.Array([
        [
            {"genPartFlav": flavor, "gen_pdgId": 11, "pdgId": 11}
            for flavor in event_flavors
        ]
        for event_flavors in flavors
    ])
    scope = {
        "sampleType": sample_type,
        "padded_FOs": leptons,
        "mask": np.ones(3, dtype=bool),
        "ak": ak,
    }
    exec(compile(block, str(source_path), "exec"), scope)
    return ak.to_list(scope["mask"])


@pytest.mark.parametrize(
    "processor_name",
    ("analysis_processor.py", "analysis_processor_diboson.py"),
)
def test_ul17_dy_nominal_classification_and_matching(processor_name):
    samples = load_probe_cfg("excluded_mc_run2_sr.cfg")
    processor_path = REPO_ROOT / "analysis" / "topeft_run2" / processor_name
    old_type = _native_sample_type(processor_path, "DYJetsToLL_centralUL17")
    new_types = [
        _native_sample_type(processor_path, samples[dataset]["histAxisName"])
        for dataset in ("UL17_DY10to50_NDSkim", "UL17_DY50_NDSkim")
    ]
    assert new_types == [old_type, old_type]

    for control in (
        "DY10to50_centralUL16APV",
        "DY50_centralUL16",
        "DY10to50_centralUL18",
        "DY50_centralUL18",
        "DYJetsToLL_MLL-50_central2022",
    ):
        assert _native_sample_type(processor_path, control) == old_type
    prompt_type = _native_sample_type(processor_path, "WJetsToLNu_centralUL17")
    conversion_type = _native_sample_type(processor_path, "ZGToLLG_centralUL17")
    assert prompt_type != old_type
    assert conversion_type != old_type
    assert _native_sample_type(processor_path, "DY10to50_centralUL17", True) == "data"

    for lepton_count in (1, 2, 3):
        old_mask = _native_matching_mask(lepton_count, old_type)
        assert old_mask == [True, True, False]
        assert all(
            _native_matching_mask(lepton_count, sample_type) == old_mask
            for sample_type in new_types
        )
        assert _native_matching_mask(lepton_count, prompt_type) == [True, False, False]
        assert _native_matching_mask(lepton_count, conversion_type) == [False, True, False]
        assert _native_matching_mask(lepton_count, "data") == [True, True, True]


def test_ul17_dy_nonprompt_policy_accepts_split_labels():
    samples = load_probe_cfg("excluded_mc_run2_sr.cfg")
    certificate = certify_active_nonprompt_policy(
        samples,
        configuration_source="excluded_mc_run2_sr.cfg",
    )
    rows = {row.raw_process_label: row for row in certificate.resolutions}
    for dataset in ("UL17_DY10to50_NDSkim", "UL17_DY50_NDSkim"):
        assert rows[samples[dataset]["histAxisName"]].policy_role == EXPLICIT_NONPROMPT_EXCLUSION
    legacy = certify_active_nonprompt_policy(
        ("DYJetsToLL_centralUL17",),
        configuration_source="legacy_ul17_dy_label",
    )
    assert legacy.resolutions[0].policy_role == EXPLICIT_NONPROMPT_EXCLUSION


def test_ul17_dy_cr_plotter_routing_recognizes_distinct_labels():
    samples = load_probe_cfg("excluded_mc_run2_sr.cfg")
    processes = [
        samples[dataset]["histAxisName"]
        for dataset in ("UL17_DY10to50_NDSkim", "UL17_DY50_NDSkim")
    ]
    cr_map = populate_group_map(processes, CR_GRP_PATTERNS)

    assert cr_map["DY"] == processes
    assert all(sum(process in members for members in cr_map.values()) == 1 for process in processes)


def test_excluded_mc_runtime_category_reservations():
    class queue_recorder:
        def __init__(self):
            self.calls = []

        def specify_category_max_resources(self, category, resources):
            self.calls.append(("maximum", category, resources))

        def specify_category_min_resources(self, category, resources):
            self.calls.append(("minimum", category, resources))

    queue = queue_recorder()
    _configure_wq_categories(
        queue,
        processing_cores=1,
        accumulating_memory_mb=32000,
        accumulating_cores=2,
    )
    assert queue.calls == [
        ("maximum", "processing", {"cores": 1}),
        ("minimum", "accumulating", {"memory": 32000, "cores": 2}),
    ]


def test_dedicated_native_wq_constructor_receives_manager_after_guard():
    """Execute the dedicated source's WQ setup nodes with inert native doubles."""
    source_path = REPO_ROOT / "analysis/topeft_run2/run_analysis_excluded_mc.py"
    tree = ast.parse(source_path.read_text())
    guard = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.wq_guard"
        and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "start_guard" for child in ast.walk(node))
    )
    executor_args_assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "executor_args" for target in node.targets)
        and isinstance(node.value, ast.Dict)
    )
    resource_setup = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "args.wq_processing_cores == 1" in ast.unparse(node.test)
    )
    constructor = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "WorkQueueExecutor"
    )
    calls = []

    def fake_guard(*_):
        calls.append(("guard",))
        return ("/synthetic/staging", 1, None, None)

    class inert_queue:
        def specify_category_max_resources(self, category, resources):
            calls.append(("maximum", category, resources))

        def specify_category_min_resources(self, category, resources):
            calls.append(("minimum", category, resources))

    def fake_constructor(**kwargs):
        calls.append(("constructor", kwargs["manager_name"], kwargs["port"]))
        kwargs["custom_init"](inert_queue())
        return object()

    scope = {
        "args": SimpleNamespace(
            wq_guard=True, wq_invocation_id="synthetic", wq_quota_root="/synthetic/quota",
            wq_free_warning_bytes=100, wq_min_free_bytes=50, wq_poll_seconds=1,
            wq_processing_cores=1, wq_accumulating_memory_mb=32000,
            wq_accumulating_cores=2,
        ),
        "executor_name": "work_queue", "wq_filepath": "/synthetic/staging",
        "start_guard": fake_guard, "staging_guard_error": RuntimeError,
        "wq_guard_state": None, "wq_manager_name": "synthetic-excludedmc-run3-block-a",
        "port": 31001, "wq_log_paths": {}, "environment_file": None,
        "_configure_wq_categories": _configure_wq_categories,
        "processor": SimpleNamespace(WorkQueueExecutor=fake_constructor),
    }
    for node in (guard, executor_args_assignment, resource_setup, constructor):
        block = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(block, str(source_path), "exec"), scope)
    assert calls == [
        ("guard",),
        ("constructor", "synthetic-excludedmc-run3-block-a", 31001),
        ("maximum", "processing", {"cores": 1}),
        ("minimum", "accumulating", {"memory": 32000, "cores": 2}),
    ]
