from analysis.topeft_run2 import make_cr_and_sr_plots


DD_PROCESS_LABELS = (
    "charge_flips_sm_2022",
    "charge_flips_sm_2022EE",
    "charge_flips_sm_2023",
    "charge_flips_sm_2023BPix",
    "nonprompt_2022",
    "nonprompt_2022EE",
    "nonprompt_2023",
    "nonprompt_2023BPix",
    "nonprompt",
)


def _filter_dd_labels(dd_year_tokens):
    return [
        label
        for label in DD_PROCESS_LABELS
        if make_cr_and_sr_plots._dd_label_matches_selected_years(
            label, dd_year_tokens
        )
    ]


def test_detect_dd_year_token_prefers_longest_match_and_supports_case_and_separators():
    assert make_cr_and_sr_plots._detect_dd_year_token("charge_flips_sm_2022") == "2022"
    assert make_cr_and_sr_plots._detect_dd_year_token("charge_flips_sm-2022EE") == "2022EE"
    assert (
        make_cr_and_sr_plots._detect_dd_year_token("nonprompt2023bpix")
        == "2023BPix"
    )
    assert make_cr_and_sr_plots._detect_dd_year_token("nonprompt_2023") == "2023"
    assert make_cr_and_sr_plots._detect_dd_year_token("nonprompt") is None


def test_dd_year_filter_keeps_only_2022_labels():
    kept = _filter_dd_labels(("2022",))
    assert kept == ["charge_flips_sm_2022", "nonprompt_2022"]


def test_dd_year_filter_keeps_requested_pair():
    kept = _filter_dd_labels(("2022", "2022EE"))
    assert kept == [
        "charge_flips_sm_2022",
        "charge_flips_sm_2022EE",
        "nonprompt_2022",
        "nonprompt_2022EE",
    ]


def test_dd_year_filter_prevents_2023_collision_with_2023bpix():
    kept = _filter_dd_labels(("2023BPix",))
    assert "charge_flips_sm_2023BPix" in kept
    assert "nonprompt_2023BPix" in kept
    assert "charge_flips_sm_2023" not in kept
    assert "nonprompt_2023" not in kept


def test_dd_year_filter_none_keeps_baseline_behavior():
    assert _filter_dd_labels(None) == list(DD_PROCESS_LABELS)
