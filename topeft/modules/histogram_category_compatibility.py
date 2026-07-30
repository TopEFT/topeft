"""Fail-fast histogram/category compatibility checks.

The compatibility contract is intentionally based on structured category
selection tokens and exact processor channel semantics.  Histogram or category
name substrings are not used to infer capabilities.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


FAKE_TAU_OBJECT = "fake_tau_object"
TIGHT_TAU_OBJECT = "tight_tau_object"
PTZ_WTAU_CHANNEL_FILL = "ptz_wtau_channel_fill"

HISTOGRAM_CAPABILITY_REQUIREMENTS = {
    "ptz_wtau": PTZ_WTAU_CHANNEL_FILL,
    "tau0Fpt": FAKE_TAU_OBJECT,
    "tau0Tpt": TIGHT_TAU_OBJECT,
}

_TIGHT_TAU_SELECTION = "1tau"
_FAKE_TAU_SELECTION = "1Ftau"
_PTZ_WTAU_EXACT_CHANNELS = frozenset(
    {
        "1l_dy_tautau_CR",
        "2lss_m_1tau_onZ",
        "2lss_p_1tau_onZ",
    }
)


@dataclass(frozen=True)
class incompatible_histogram:
    histogram_family: str
    required_capability: str


class histogram_category_compatibility_error(ValueError):
    """Raised before processing for a guaranteed-empty requested family."""


def _channel_capabilities(channel_definition):
    if (
        not isinstance(channel_definition, Sequence)
        or isinstance(channel_definition, (str, bytes))
        or not channel_definition
    ):
        return frozenset()

    channel_name = str(channel_definition[0])
    selection_tokens = frozenset(map(str, channel_definition[1:]))
    capabilities = set()

    if _FAKE_TAU_SELECTION in selection_tokens:
        capabilities.add(FAKE_TAU_OBJECT)
    if _TIGHT_TAU_SELECTION in selection_tokens:
        capabilities.update({FAKE_TAU_OBJECT, TIGHT_TAU_OBJECT})
    if channel_name in _PTZ_WTAU_EXACT_CHANNELS:
        capabilities.add(PTZ_WTAU_CHANNEL_FILL)

    return frozenset(capabilities)


def category_capabilities(category_definition):
    """Return fill capabilities derived from a structured ch_lst category."""

    if not isinstance(category_definition, Mapping):
        return frozenset()

    capabilities = set()
    for channel_definition in category_definition.get("lep_chan_lst", ()):
        capabilities.update(_channel_capabilities(channel_definition))
    return frozenset(capabilities)


def _selected_categories(selected_category_dicts):
    selected = []
    for category_dict in selected_category_dicts:
        if not isinstance(category_dict, Mapping):
            continue
        for category_name, category_definition in category_dict.items():
            selected.append((str(category_name), category_definition))
    return selected


def find_incompatible_histograms(
    histogram_families,
    *,
    selected_category_dicts,
):
    """Return required families that no selected category can fill."""

    selected_categories = _selected_categories(selected_category_dicts)
    available_capabilities = set()
    for _category_name, category_definition in selected_categories:
        available_capabilities.update(category_capabilities(category_definition))

    incompatible = []
    for histogram_family in dict.fromkeys(map(str, histogram_families)):
        required_capability = HISTOGRAM_CAPABILITY_REQUIREMENTS.get(
            histogram_family
        )
        if (
            required_capability is not None
            and required_capability not in available_capabilities
        ):
            incompatible.append(
                incompatible_histogram(
                    histogram_family=histogram_family,
                    required_capability=required_capability,
                )
            )
    return tuple(incompatible)


def validate_histogram_category_compatibility(
    histogram_families,
    *,
    selected_category_dicts,
    histogram_selection_explicit,
    requested_data_driven_products=(),
):
    """Reject guaranteed-empty families without silently pruning requests.

    Explicit histogram selections always fail when incompatible.  Implicit
    processor-default selections retain historical non-product behavior, but
    fail when a requested derived product would require the empty source
    family.
    """

    incompatible = find_incompatible_histograms(
        histogram_families,
        selected_category_dicts=selected_category_dicts,
    )
    requested_products = tuple(
        dict.fromkeys(map(str, requested_data_driven_products))
    )
    product_sensitive = bool(requested_products)
    if not incompatible or (
        not histogram_selection_explicit and not product_sensitive
    ):
        return incompatible

    selected_categories = list(
        dict.fromkeys(
            category_name
            for category_name, _definition in _selected_categories(
                selected_category_dicts
            )
        )
    )
    incompatible_summary = ", ".join(
        "{} (requires {})".format(
            item.histogram_family,
            item.required_capability,
        )
        for item in incompatible
    )
    selection_kind = (
        "explicit" if histogram_selection_explicit else "implicit/default"
    )
    product_summary = ", ".join(requested_products) or "<none>"
    category_summary = ", ".join(selected_categories) or "<none>"

    raise histogram_category_compatibility_error(
        "Histogram/category compatibility preflight failed before processor "
        "construction, executor setup, output staging, or event processing. "
        f"incompatible_histograms=[{incompatible_summary}]; "
        f"selected_categories=[{category_summary}]; "
        f"histogram_selection={selection_kind}; "
        f"requested_data_driven_products=[{product_summary}]; "
        f"product_required_empty_family={'yes' if product_sensitive else 'no'}. "
        "Processing was not started. Explicitly requested histograms are never "
        "silently removed."
    )
