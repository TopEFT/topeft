from topeft.modules.axes import info as axes_info, info_2d as axes_info_2d
from topeft.modules.dataDrivenEstimation import DataDrivenProducer
from topeft.modules.nominal_schema import (
    eft_nominal_key,
    scalar_nominal_key,
    sumw2_key,
)


def test_known_legacy_uniform_one_dimensional_family_resolves_as_scalar():
    assert "met" in axes_info
    assert "met" not in axes_info_2d
    assert DataDrivenProducer._family_from_nominal_key("met") == ("met", "scalar")


def test_unknown_legacy_uniform_key_remains_unresolved():
    assert DataDrivenProducer._family_from_nominal_key("nominal") == (None, None)


def test_split_and_two_dimensional_resolution_remain_explicit():
    assert DataDrivenProducer._family_from_nominal_key(scalar_nominal_key("met")) == (
        "met",
        "scalar",
    )
    assert DataDrivenProducer._family_from_nominal_key(eft_nominal_key("met")) == (
        "met",
        "eft",
    )
    assert DataDrivenProducer._family_from_nominal_key(sumw2_key("met")) == (
        "met",
        "sumw2",
    )

    two_dimensional_family = next(iter(axes_info_2d))
    assert DataDrivenProducer._family_from_nominal_key(two_dimensional_family) == (
        two_dimensional_family,
        "scalar",
    )
