from pathlib import Path

import correctionlib
import pytest
import yaml

from topeft.modules import corrections as cor
from topcoffea.modules.paths import topcoffea_path


JERC_DICT_PATH = Path(__file__).resolve().parents[1] / "topeft" / "modules" / "jerc_dict.yml"


def _load_jerc_dict():
    with JERC_DICT_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _runtime_correction_names(year, isdata, era=None):
    jet_algo, jec_tag, jec_levels, jer_tag, junc_types = cor.get_jerc_keys(
        year,
        isdata=isdata,
        era=era,
    )
    names = [f"{jec_tag}_{level}_{jet_algo}" for level in jec_levels]
    if jer_tag is not None:
        names.extend(
            (
                f"{jer_tag}_ScaleFactor_{jet_algo}",
                f"{jer_tag}_PtResolution_{jet_algo}",
            )
        )
    if junc_types:
        names.extend(f"{jec_tag}_{junc_type}_{jet_algo}" for junc_type in junc_types)
    return names


@pytest.mark.parametrize(
    "year",
    ("2016", "2016APV", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"),
)
def test_configured_runtime_jerc_corrections_exist_in_selected_payload(year):
    config = _load_jerc_dict()
    payload_path = topcoffea_path(
        f"data/POG/JME/{cor.clib_year_map[year]}/jet_jerc.json.gz"
    )
    available_names = set(correctionlib.CorrectionSet.from_file(payload_path).keys())

    required_names = _runtime_correction_names(year, isdata=False)
    for era in config[year]["jec_data"]:
        required_names.extend(_runtime_correction_names(year, isdata=True, era=era))

    missing_names = sorted(set(required_names) - available_names)
    assert not missing_names, (
        f"{year} requests correction names absent from {payload_path}: "
        f"{missing_names}"
    )
