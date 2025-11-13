from __future__ import annotations

from pathlib import Path
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.topeft_run2.workflow import (
    ChannelPlanner,
    HistogramCombination,
    HistogramPlanner,
)
from analysis.topeft_run2.run_analysis_helpers import RunConfig
from topeft.modules.channel_metadata import ChannelMetadataHelper
from topeft.modules.systematics import SystematicsHelper


VARIABLE_DEFINITIONS = {
    "mass": {"label": "m"},
    "ht": {"label": "H_T"},
}


CHANNEL_METADATA = {
    "groups": {
        "BASIC_SR": {
            "description": "Signal regions",
            "regions": [
                {
                    "lepton_category": "2l",
                    "lepton_flavors": ["ee", "mm"],
                    "jet_bins": ["=2"],
                    "application_tags": {
                        "mc": ["isSR2l"],
                        "data": ["isSR2l"],
                    },
                    "histogram_variables": {
                        "include": ["mass"],
                        "exclude": ["ht"],
                    },
                    "region_definitions": [
                        {
                            "name": "2l_sr",
                            "channel": "2l",
                            "subchannel": "2l_sr",
                            "tags": [],
                        }
                    ],
                }
            ],
        }
    },
    "scenarios": [
        {"name": "test", "groups": ["BASIC_SR"]},
    ],
}


SYSTEMATICS_METADATA = {
    "systematics": {
        "nominal": {
            "type": "nominal",
            "applies_to": ["all"],
            "variations": [{"value": "nominal"}],
        },
        "isr": {
            "type": "theory",
            "applies_to": ["mc"],
            "variations": [
                {"value": "ISRUp", "direction": "Up", "sum_of_weights": "nSumOfWeights_ISRUp"},
                {"value": "ISRDown", "direction": "Down", "sum_of_weights": "nSumOfWeights_ISRDown"},
            ],
        },
    },
}


SAMPLES_FIXTURE = {
    "mc_sample": {"isData": False, "year": "2018"},
    "data_sample": {"isData": True, "year": "2018"},
}


@pytest.fixture
def run_config():
    return RunConfig(do_systs=True, split_lep_flavor=False, scenario_names=["test"])


@pytest.fixture
def channel_planner(run_config):
    helper = ChannelMetadataHelper(CHANNEL_METADATA)
    return ChannelPlanner(
        helper,
        skip_sr=run_config.skip_sr,
        skip_cr=run_config.skip_cr,
        scenario_names=run_config.scenario_names,
    )


@pytest.fixture
def systematics_helper():
    return SystematicsHelper(SYSTEMATICS_METADATA, sample_years=["2018"])


@pytest.fixture
def histogram_planner(run_config, channel_planner):
    return HistogramPlanner(
        config=run_config,
        variable_definitions=VARIABLE_DEFINITIONS,
        channel_planner=channel_planner,
    )


def test_histogram_plan_matches_expected_structure(histogram_planner, systematics_helper):
    plan = histogram_planner.plan(SAMPLES_FIXTURE, systematics_helper)

    assert list(plan.histogram_names) == ["mass", "ht"]
    assert len(plan.tasks) == 3

    task_summaries = [
        (task.sample, task.variable, task.clean_channel, task.application)
        for task in plan.tasks
    ]
    assert task_summaries == [
        ("mc_sample", "mass", "2l_sr_2j", "isSR2l"),
        ("mc_sample", "mass", "2l_sr_2j", "isSR2l"),
        ("data_sample", "mass", "2l_sr_2j", "isSR2l"),
    ]

    nominal_task = plan.tasks[0]
    assert tuple(nominal_task.hist_keys.keys()) == ("nominal",)
    assert nominal_task.hist_keys["nominal"] == (
        ("mass", "2l_sr_2j", "isSR2l", "mc_sample", "nominal"),
    )

    isr_task = plan.tasks[1]
    assert set(isr_task.hist_keys.keys()) == {"ISRUp", "ISRDown"}
    assert ("mass", "2l_sr_2j", "isSR2l", "mc_sample", ("isr", "ISRUp")) in isr_task.hist_keys["ISRUp"]

    data_task = plan.tasks[2]
    assert tuple(data_task.hist_keys.keys()) == ("nominal",)
    assert data_task.hist_keys["nominal"] == (
        ("mass", "2l_sr_2j", "isSR2l", "data_sample", "nominal"),
    )

    expected_summary = (
        HistogramCombination(
            sample="mc_sample",
            channel="2l_sr_2j",
            variable="mass",
            application="isSR2l",
            systematic="nominal",
        ),
        HistogramCombination(
            sample="mc_sample",
            channel="2l_sr_2j",
            variable="mass",
            application="isSR2l",
            systematic="isr:ISRDown",
        ),
        HistogramCombination(
            sample="mc_sample",
            channel="2l_sr_2j",
            variable="mass",
            application="isSR2l",
            systematic="isr:ISRUp",
        ),
        HistogramCombination(
            sample="data_sample",
            channel="2l_sr_2j",
            variable="mass",
            application="isSR2l",
            systematic="nominal",
        ),
    )
    assert tuple(plan.summary) == expected_summary
