"""SR prefit response adapter for the maintained datacard nuisance definitions.

This module consumes card definitions without constructing or writing cards.
Display groups are deliberately absent from its interface.
"""

from collections import defaultdict
from functools import lru_cache

import numpy as np

from topeft.modules.datacard_tools import (
    DatacardMaker,
    PRECISION,
    resolve_missing_parton_response_index,
    resolve_shape_nuisance_identity,
    run2_luminosity_shape_factors,
    run3_luminosity_shape_factors,
    run2_luminosity_excluded_processes,
    tau_run2_year_correlation_bases,
)
from topeft.modules.nominal_schema import canonical_process_year, year_independent_process


# These card __init__ rules have no standalone public accessor. Keep the
# adapter small and check the complete copies against a live card in tests.
RUN_DECORRELATE = (
    "FES", "FF", "FFeta", "FFpt", "btagSFbc_corr", "btagSFlight_corr",
    "charge_flips", "TES", "lepSF_elec", "lepSF_muon", "lepSF_taus_fake",
    "lepSF_taus_real", "lepSF_elec_mva", "lepSF_muon_mva",
    "lepSF_elec_non_mva", "lepSF_muon_non_mva",
)
YEAR_CORRELATE_2016 = {
    "FFcloseEl", "FFcloseMu", *tau_run2_year_correlation_bases,
}
YEAR_CORRELATE_RUN3 = {
    "FFcloseEl", "FFcloseMu",
    *(f"CMS_fake_t_DeepTau2018v2p5_VSmu_{eta}"
      for eta in ("abseta0to0p4", "abseta0p4to0p8", "abseta0p8to1p2",
                  "abseta1p2to1p7", "abseta1p7to2p4")),
    *(f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron"
      for dm in (0, 1, 10, 11)),
}


def card_process_name(process):
    """Apply the card's process cleanup and exact GROUP membership in source order."""
    cleaned = process.replace("private", "").replace("central", "").replace("_4F", "")
    year = canonical_process_year(cleaned)
    if year is None:
        raise ValueError(f"SR process has no card-compatible era: {process!r}")
    for group, members in DatacardMaker.GROUP.items():
        if cleaned in (f"{member}{year}" for member in members):
            return year_independent_process(f"{group}{year}")
    return year_independent_process(cleaned)


def is_excluded_auxiliary(process):
    """Identify the scoped plot-only MC families present in the SR auxiliaries."""
    return process.startswith(("DY", "ST_", "WJets", "ZG", "tW_", "tbarW_", "TTto4Q_"))


def _rate_factor_pair(value):
    if isinstance(value, str) and "/" in value:
        low, high = value.split("/", 1)
        return float(low), float(high)
    high = float(value)
    return 1.0 / high, high


@lru_cache(maxsize=4)
def _card_rates(run, include_jet_rates=False):
    """Reuse the card's JSON reader with a lightweight, uninitialized receiver."""
    card = object.__new__(DatacardMaker)
    card.do_nuisance = True
    card.skip_missing_parton_rate_syst = not include_jet_rates
    card.year_lst = ["UL18" if run == "run2" else "2022"]
    payload_path = (
        card.resolve_missing_parton_payload_path(card.year_lst)
        if include_jet_rates else None
    )
    return card.load_systematics(f"params/rate_systs_{run}.json", payload_path)


def _channel_jet_count(channel):
    """Use the card's channel parser, requiring a physical jet-labelled leaf."""
    if not isinstance(channel, str) or not channel:
        raise ValueError("SR non-njets card rate requires a physical channel")
    try:
        return DatacardMaker.get_jet_mults(channel)[0]
    except (AttributeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"SR non-njets card rate cannot resolve jets for channel {channel!r}"
        ) from exc


def _special_rate_factor(source, card_name, channel, jet_count):
    """Read a channel/jet factor in the format written by the card."""
    if source.name == "diboson_njets":
        return source.get_process(card_name, jet_count)
    channel_rates = source.get_process(card_name)
    if channel_rates == "-":
        return "-"
    channel_key = channel.replace(f"_{jet_count}j", "")
    if channel_key not in channel_rates:
        return "-"  # The card treats an absent payload channel as inapplicable.
    response_index = resolve_missing_parton_response_index(channel, jet_count)
    try:
        high = float(channel_rates[channel_key][response_index])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Missing card {source.name} factor for {card_name!r} "
            f"channel={channel!r} jets={jet_count}"
        ) from exc
    low = max(0.01, 2 - high)
    return f"{low:.{PRECISION}f}/{high:.{PRECISION}f}"


def rate_responses(process, nominal, *, channel=None, variable="njets"):
    """Return signed Down/Up responses keyed by card rate nuisance name."""
    year = canonical_process_year(process)
    if year is None:
        raise ValueError(f"SR rate process has no era: {process!r}")
    run = "run2" if year.startswith("UL") else "run3"
    card_name = card_process_name(process)
    if card_name in {"data_obs", "data"}:
        return {}
    include_jet_rates = (
        variable not in (None, "njets")
        and card_name in {"Diboson", "tllq", "tHq"}
    )
    rates = _card_rates(run, include_jet_rates)
    known_card_names = {
        *(group.rstrip("_") for group in DatacardMaker.GROUP),
        *DatacardMaker.SIGNALS,
        *(name for source in rates.values() for name in source.corrs),
    }
    if card_name not in known_card_names and not is_excluded_auxiliary(process):
        raise ValueError(f"Unknown SR card process identity: {process!r}")
    responses = {}
    for nuisance, source in rates.items():
        special_rate = nuisance == "diboson_njets" or DatacardMaker.is_missing_parton_nuisance_name(nuisance)
        if special_rate and not include_jet_rates:
            continue  # The card omits these rates in the njets view.
        if special_rate and not source.has_process(card_name):
            continue
        if special_rate:
            jet_count = _channel_jet_count(channel)
            factor = _special_rate_factor(source, card_name, channel, jet_count)
        else:
            factor = source.get_process(card_name)
        if factor == "-":
            continue
        low, high = _rate_factor_pair(factor)
        responses[nuisance] = (np.asarray(nominal) * (low - 1),
                               np.asarray(nominal) * (high - 1))
    return responses


def shape_nuisance_name(base, process):
    """Resolve the card's run and process decorrelation on a source shape."""
    year = canonical_process_year(process)
    if year is None:
        raise ValueError(f"SR shape process has no era: {process!r}")
    run_suffix = "_run2" if year.startswith("UL") else "_run3"
    name, _ = resolve_shape_nuisance_identity(base, run_suffix, RUN_DECORRELATE)
    if name.endswith("_2016APV") and name[:-8] in YEAR_CORRELATE_2016:
        name = name[:-8] + "_2016"
    for later, earlier in (("_2022EE", "_2022"), ("_2023BPix", "_2023")):
        if name.endswith(later):
            source = name[:-len(later)]
            if source in YEAR_CORRELATE_RUN3:
                name = source + earlier
    card_name = card_process_name(process)
    if base in {"renorm", "fact"}:
        return f"{name}_{card_name}"
    if base == "ISR":
        if card_name in {"ttH", "ttll", "tttt", "convs"}:
            return f"{name}_gg"
        if card_name in {"ttlnu", "tllq", "Diboson", "Triboson"}:
            return f"{name}_qq"
        if card_name == "tHq":
            return f"{name}_qg"
    return name


def luminosity_responses(process, nominal):
    """Use the card's annual luminosity shape factors before era aggregation."""
    year = canonical_process_year(process)
    if year is None:
        raise ValueError(f"SR luminosity process has no era: {process!r}")
    if card_process_name(process) in run2_luminosity_excluded_processes:
        return {}
    factors = run2_luminosity_shape_factors if year.startswith("UL") else run3_luminosity_shape_factors
    return {
        nuisance: (np.asarray(nominal) * (1.0 / year_factors[year] - 1),
                   np.asarray(nominal) * (year_factors[year] - 1))
        for nuisance, year_factors in factors.items()
        if year_factors[year] != 1.0
    }


def combine_responses(responses, nominal):
    """Add signed responses per nuisance, then form two-sided quadrature."""
    coherent = defaultdict(lambda: [np.zeros_like(nominal, dtype=float),
                                    np.zeros_like(nominal, dtype=float)])
    for nuisance, down, up in responses:
        coherent[nuisance][0] += np.asarray(down, dtype=float)
        coherent[nuisance][1] += np.asarray(up, dtype=float)
    low_square = np.zeros_like(nominal, dtype=float)
    high_square = np.zeros_like(nominal, dtype=float)
    for down, up in coherent.values():
        high_square += np.maximum(np.maximum(down, up), 0) ** 2
        low_square += np.maximum(np.maximum(-down, -up), 0) ** 2
    return low_square, high_square
