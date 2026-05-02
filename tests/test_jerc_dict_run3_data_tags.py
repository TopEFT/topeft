from pathlib import Path

import yaml

from topeft.modules import corrections as cor


JERC_DICT_PATH = Path(__file__).resolve().parents[1] / "topeft" / "modules" / "jerc_dict.yml"


def _jerc_dict():
    with JERC_DICT_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_run3_2022_data_tags_match_correctionlib_payload_names():
    cfg = _jerc_dict()

    assert cfg["2022"]["jec_data"] == {
        "C": "Summer22_22Sep2023_V3_DATA",
        "D": "Summer22_22Sep2023_V3_DATA",
    }
    assert cfg["2022EE"]["jec_data"] == {
        "E": "Summer22EE_22Sep2023_V3_DATA",
        "F": "Summer22EE_22Sep2023_V3_DATA",
        "G": "Summer22EE_22Sep2023_V3_DATA",
    }


def test_run3_2022_data_tags_flow_through_get_jerc_keys_without_run_suffixes():
    for era in ("C", "D"):
        _, jec_tag, _, _, _ = cor.get_jerc_keys("2022", isdata=True, era=era)
        assert jec_tag == "Summer22_22Sep2023_V3_DATA"
        assert "RunCD" not in jec_tag

    for era in ("E", "F", "G"):
        _, jec_tag, _, _, _ = cor.get_jerc_keys("2022EE", isdata=True, era=era)
        assert jec_tag == "Summer22EE_22Sep2023_V3_DATA"
        assert all(run_token not in jec_tag for run_token in ("RunE", "RunF", "RunG"))


def test_unrelated_run3_jerc_tags_remain_unchanged():
    cfg = _jerc_dict()

    assert cfg["2022"]["jec_mc"] == "Summer22_22Sep2023_V3_MC"
    assert cfg["2022EE"]["jec_mc"] == "Summer22EE_22Sep2023_V3_MC"
    assert cfg["2023"]["jec_data"] == {
        "C1": "Summer23Prompt23_V3_DATA",
        "C2": "Summer23Prompt23_V3_DATA",
        "C3": "Summer23Prompt23_V3_DATA",
        "C4": "Summer23Prompt23_V3_DATA",
    }
    assert cfg["2023BPix"]["jec_data"] == {
        "D1": "Summer23BPixPrompt23_V3_DATA",
        "D2": "Summer23BPixPrompt23_V3_DATA",
    }
