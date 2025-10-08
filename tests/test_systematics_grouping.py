from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topeft.modules.systematics import SystematicsHelper


def _build_metadata():
    return {
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
            "pileup": {
                "type": "weight",
                "applies_to": ["mc"],
                "variations": [
                    {"value": "PUUp", "direction": "Up"},
                    {"value": "PUDown", "direction": "Down"},
                ],
            },
            "fake_factors": {
                "type": "data_weight",
                "applies_to": ["all"],
                "variations": [
                    {"value": "FFUp", "direction": "Up"},
                    {"value": "FFDown", "direction": "Down"},
                ],
            },
        },
    }


def test_grouped_variations_for_sample_mc():
    metadata = _build_metadata()
    helper = SystematicsHelper(metadata, sample_years=["2018"])
    sample = {"year": "2018", "isData": False}

    grouped = helper.grouped_variations_for_sample(sample, include_systematics=True)
    variations = helper.variations_for_sample(sample, include_systematics=True)

    assert sum(len(members) for members in grouped.values()) == len(variations)

    isr_groups = [item for item in grouped.items() if item[0].name == "isr"]
    assert len(isr_groups) == 1
    _, isr_variations = isr_groups[0]
    assert sorted(variation.name for variation in isr_variations) == ["ISRDown", "ISRUp"]

    first_group = next(iter(grouped.keys()))
    assert "nominal" in first_group.members


def test_grouped_variations_for_data_sample_excludes_mc_theory():
    metadata = _build_metadata()
    helper = SystematicsHelper(metadata, sample_years=["2018"])
    sample = {"year": "2018", "isData": True}

    grouped = helper.grouped_variations_for_sample(sample, include_systematics=True)

    for _, variations in grouped.items():
        for variation in variations:
            assert variation.type != "theory"
