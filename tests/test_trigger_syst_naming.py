import numpy as np
import hist
from coffea.analysis_tools import Weights


def _build_trigger_weights(year="2022"):
    weights = Weights(3, storeIndividual=True)
    nominal = np.array([1.0, 1.1, 0.9], dtype=float)
    up = nominal * 1.02
    down = nominal * 0.98
    weights.add(f"triggerSF_{year}", nominal, up, down)
    return weights


def test_trigger_variation_names_are_canonical():
    year = "2022"
    weights = _build_trigger_weights(year)

    assert f"triggerSF_{year}Up" in weights.variations
    assert f"triggerSF_{year}Down" in weights.variations
    assert f"{year}Up" not in weights.variations
    assert f"{year}Down" not in weights.variations


def test_trigger_variations_fill_systematic_axis():
    year = "2022"
    weights = _build_trigger_weights(year)
    requested_variations = [
        "nominal",
        f"triggerSF_{year}Up",
        f"triggerSF_{year}Down",
    ]

    histogram = hist.Hist(
        hist.axis.Regular(1, 0, 1, name="x"),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
    )

    x_values = np.zeros(3, dtype=float)
    for variation in requested_variations:
        if variation == "nominal":
            weight = weights.weight(None)
        elif variation in weights.variations:
            weight = weights.weight(variation)
        else:
            continue

        histogram.fill(
            x=x_values,
            channel="2lss_p_4j",
            appl="isSR_2lSS",
            process="ttH_UL22",
            systematic=variation,
            weight=weight,
        )

    systematic_labels = {str(label) for label in histogram.axes["systematic"]}
    assert systematic_labels == set(requested_variations)
