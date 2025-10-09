"""Utilities for expanding and filtering systematic variations."""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
import json
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import topcoffea.modules.corrections as tc_cor

from topeft.modules.paths import topeft_path

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SystematicVariationGroup:
    """Descriptor capturing the metadata defining a variation group."""

    name: str
    members: Tuple[str, ...]
    metadata: Tuple[
        Tuple[Tuple[str, Optional[str], Optional[str]], Tuple[Tuple[str, object], ...]],
        ...,
    ]

    @staticmethod
    def _freeze_value(value: object) -> object:
        """Return a hashable representation of ``value`` suitable for grouping."""

        if isinstance(value, dict):
            return tuple(
                (key, SystematicVariationGroup._freeze_value(val))
                for key, val in sorted(value.items())
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(
                SystematicVariationGroup._freeze_value(item)
                for item in value
            )
        return value

    @classmethod
    def _freeze_mapping(
        cls, mapping: Dict[str, object]
    ) -> Tuple[Tuple[str, object], ...]:
        return tuple(
            (key, cls._freeze_value(value))
            for key, value in sorted(mapping.items())
        )

    @classmethod
    def from_variation(cls, variation: "SystematicVariation") -> "SystematicVariationGroup":
        """Create a group descriptor for the provided variation."""

        if variation.group:
            group_key = (variation.base, variation.component, variation.year)
            grouped_variations = variation.group.get(group_key, {})
            metadata = (
                (
                    group_key,
                    tuple(
                        (member_name, cls._freeze_mapping(info))
                        for member_name, info in sorted(grouped_variations.items())
                    ),
                ),
            )
            name = variation.base
            members = tuple(sorted(grouped_variations.keys()))
        else:
            group_key = (variation.name, None, None)
            metadata = ((group_key, cls._freeze_mapping(variation.metadata)),)
            name = variation.name
            members = (variation.name,)

        return cls(name=name, members=members, metadata=metadata)


@dataclass
class SystematicVariation:
    """Representation of a single systematic variation.

    Attributes
    ----------
    name:
        Unique identifier for the variation (e.g. ``JER_2018Up``).
    base:
        Logical group the variation belongs to (``jer``, ``isr``, etc.).
    type:
        Category of the systematic (``object``, ``weight``, ``theory``, ...).
    applies_to:
        Samples this variation applies to (``mc``, ``data`` or ``all``).
    tau_only:
        Whether the variation should only be considered for tau analyses.
    year:
        Optional year tied to the variation.
    direction:
        Direction label (typically ``Up`` or ``Down``).
    component:
        Optional component label (e.g. JES regrouping component).
    metadata:
        Additional metadata associated with the variation (sum-of-weights keys,
        etc.).
    group:
        Mapping describing the groups of variations defined for the base.  Each
        key is a ``(base, component, year)`` tuple whose value maps variation
        names to metadata dictionaries.  The dictionaries include the
        variation's metadata along with ``component`` and ``year`` entries. This
        is mainly used for theory uncertainties that require the up/down sum of
        weights simultaneously.
    """

    name: str
    base: str
    type: str
    applies_to: Set[str]
    tau_only: bool = False
    year: Optional[str] = None
    direction: Optional[str] = None
    component: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    group: Dict[
        Tuple[str, Optional[str], Optional[str]], Dict[str, Dict[str, object]]
    ] = field(default_factory=dict)

    def matches_sample(self, is_data: bool, sample_year: Optional[str], tau_analysis: bool) -> bool:
        """Return ``True`` if this variation should be run for the sample."""

        if self.tau_only and not tau_analysis:
            return False

        if is_data:
            if "all" not in self.applies_to and "data" not in self.applies_to:
                return False
        else:
            if "all" not in self.applies_to and "mc" not in self.applies_to:
                return False

        if self.year is not None and sample_year is not None and self.year != sample_year:
            return False

        return True


class SystematicsHelper:
    """Helper responsible for expanding and filtering systematic variations."""

    def __init__(
        self,
        metadata: Dict[str, object],
        sample_years: Optional[Iterable[str]] = None,
        tau_analysis: bool = False,
    ) -> None:
        self._metadata = metadata or {}
        self._tau_analysis = tau_analysis
        self._sample_years = {str(y) for y in sample_years or []}
        self._jerc_info = self._load_jerc_info()
        self._variations = self._build_variations()
        self._variations_by_name = {var.name: var for var in self._variations}

    @staticmethod
    def _load_jerc_info() -> Dict[str, Dict[str, object]]:
        path = topeft_path("modules/jerc_dict.json")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def variations(self) -> Sequence[SystematicVariation]:
        """Return the list of expanded systematic variations."""

        return self._variations

    @property
    def variation_names(self) -> List[str]:
        return [var.name for var in self._variations]

    def get(self, name: str) -> Optional[SystematicVariation]:
        return self._variations_by_name.get(name)

    def _get_sample_years(self) -> List[str]:
        if self._sample_years:
            return sorted(self._sample_years)
        # Fall back to all years known to the JERC configuration so that we do
        # not silently drop variations when the helper is instantiated before
        # the samples are read.
        return sorted(self._jerc_info.keys())

    def _build_variations(self) -> List[SystematicVariation]:
        syst_config = self._metadata.get("systematics", {})
        if not isinstance(syst_config, dict):
            raise TypeError("metadata['systematics'] must be a mapping of definitions")

        sample_years = self._get_sample_years()

        variations: List[SystematicVariation] = []

        for base_name, entry in syst_config.items():
            if not isinstance(entry, dict):
                raise TypeError(f"Systematic '{base_name}' definition must be a mapping")

            syst_type = entry.get("type", "weight")
            applies_to = set(entry.get("applies_to", ["all"]))
            tau_only = bool(entry.get("tau_only", False))
            base_metadata = dict(entry.get("metadata", {}))

            if "variations" in entry:
                raw_variations = entry["variations"]
                if not isinstance(raw_variations, list):
                    raise TypeError(f"Systematic '{base_name}' variations must be a list")
                for raw in raw_variations:
                    if isinstance(raw, str):
                        value = raw
                        var_meta: Dict[str, object] = {}
                    elif isinstance(raw, dict):
                        if "value" in raw:
                            value = str(raw["value"])
                        elif "name" in raw:
                            value = str(raw["name"])
                        else:
                            raise ValueError(
                                f"Systematic '{base_name}' variation entry must have a 'value' or 'name'"
                            )
                        var_meta = {k: v for k, v in raw.items() if k not in {"value", "name"}}
                    else:
                        raise TypeError(
                            f"Systematic '{base_name}' variations must contain strings or mappings"
                        )

                    variation = SystematicVariation(
                        name=value,
                        base=base_name,
                        type=syst_type,
                        applies_to=applies_to,
                        tau_only=tau_only,
                        year=str(var_meta.pop("year")) if "year" in var_meta else None,
                        direction=var_meta.pop("direction", None),
                        component=var_meta.pop("component", None),
                        metadata={**base_metadata, **var_meta},
                    )
                    variations.append(variation)
                continue

            template = entry.get("template")
            if not template:
                raise ValueError(f"Systematic '{base_name}' requires either 'variations' or 'template'")

            directions = entry.get("directions", [None])
            if not isinstance(directions, list):
                raise TypeError(f"Systematic '{base_name}' directions must be a list when provided")

            include_years: List[Optional[str]]
            if entry.get("year_dependent") or "{year}" in template:
                requested_years = entry.get("years")
                if requested_years is None:
                    include_years = sample_years
                else:
                    include_years = [str(y) for y in requested_years if str(y) in sample_years]
            else:
                include_years = [None]

            components: List[Optional[str]]
            if entry.get("components_from") == "jerc":
                components_set: Set[str] = set()
                strip_prefix = entry.get("component_strip", "")
                for year in sample_years:
                    year_info = self._jerc_info.get(year, {})
                    for comp in year_info.get("junc", []) or []:
                        if strip_prefix and isinstance(comp, str) and comp.startswith(strip_prefix):
                            comp_name = comp[len(strip_prefix) :]
                        else:
                            comp_name = comp
                        components_set.add(str(comp_name))
                components = sorted(components_set)
            else:
                raw_components = entry.get("components")
                if raw_components is None:
                    components = [None]
                elif isinstance(raw_components, list):
                    components = [str(comp) for comp in raw_components]
                else:
                    raise TypeError(
                        f"Systematic '{base_name}' components must be a list when provided"
                    )

            for year in include_years:
                for component in components:
                    for direction in directions:
                        fmt_args: Dict[str, object] = {}
                        if year is not None:
                            fmt_args["year"] = year
                        if component is not None:
                            fmt_args["component"] = component
                        if direction is not None:
                            fmt_args["direction"] = direction
                        try:
                            value = template.format(**fmt_args)
                        except KeyError as exc:
                            missing_key = exc.args[0]
                            raise KeyError(
                                f"Missing template argument '{missing_key}' while expanding systematic '{base_name}'"
                            ) from exc
                        variations.append(
                            SystematicVariation(
                                name=value,
                                base=base_name,
                                type=syst_type,
                                applies_to=applies_to,
                                tau_only=tau_only,
                                year=year,
                                direction=direction,
                                component=component,
                                metadata=dict(base_metadata),
                            )
                        )

        # Populate group metadata for variations that provide sum-of-weights
        # information so that processors can request the relevant pairs without
        # hard coding their names.
        variations_by_base: Dict[str, List[SystematicVariation]] = {}
        for variation in variations:
            variations_by_base.setdefault(variation.base, []).append(variation)

        for base_variations in variations_by_base.values():
            grouped_by_key: Dict[
                Tuple[str, Optional[str], Optional[str]], Dict[str, Dict[str, object]]
            ] = {}

            for var in base_variations:
                group_key = (var.base, var.component, var.year)
                members = grouped_by_key.setdefault(group_key, {})
                member_metadata = {**var.metadata}
                member_metadata.setdefault("component", var.component)
                member_metadata.setdefault("year", var.year)
                members[var.name] = member_metadata

            if not grouped_by_key:
                continue

            for var in base_variations:
                var.group = grouped_by_key

        # Ensure nominal is always first for convenience and stable ordering of
        # the rest.
        variations.sort(key=lambda item: (item.name != "nominal", item.name))
        return variations

    def variations_for_sample(
        self, sample: Dict[str, object], include_systematics: bool
    ) -> List[SystematicVariation]:
        """Return the list of variations to run for the given sample."""

        sample_year = str(sample.get("year")) if sample.get("year") is not None else None
        is_data = bool(sample.get("isData"))

        applicable = [
            variation
            for variation in self._variations
            if variation.matches_sample(is_data, sample_year, self._tau_analysis)
        ]

        if not include_systematics:
            # ``nominal`` is guaranteed to be present thanks to the metadata
            # definition.
            return [variation for variation in applicable if variation.name == "nominal"]

        return applicable

    def grouped_variations_for_sample(
        self, sample: Dict[str, object], include_systematics: bool
    ) -> "OrderedDict[SystematicVariationGroup, List[SystematicVariation]]":
        """Return systematic variations grouped by their metadata descriptors."""

        grouped: "OrderedDict[SystematicVariationGroup, List[SystematicVariation]]" = OrderedDict()

        for variation in self.variations_for_sample(sample, include_systematics):
            descriptor = SystematicVariationGroup.from_variation(variation)
            grouped.setdefault(descriptor, []).append(variation)

        return grouped

    def names_by_type(
        self,
        applies_to: Optional[str] = None,
        include_systematics: bool = True,
    ) -> Dict[str, Sequence[str]]:
        """Return variation names grouped by type.

        Parameters
        ----------
        applies_to:
            Optional sample category filter.  When set to ``"mc"`` or
            ``"data"`` only variations applying to that sample type are
            returned.
        include_systematics:
            If ``False`` only the nominal variation is returned.
        """

        if applies_to is not None:
            normalized = applies_to.lower()
            if normalized not in {"mc", "data"}:
                raise ValueError(
                    "applies_to must be either 'mc', 'data' or None"
                )
            applies_to = normalized

        grouped: Dict[str, List[str]] = {}

        for variation in self._variations:
            if not include_systematics and variation.name != "nominal":
                continue

            if variation.tau_only and not self._tau_analysis:
                continue

            if applies_to is not None:
                allowed_targets = {applies_to, "all"}
                if not (variation.applies_to & allowed_targets):
                    continue

            grouped.setdefault(variation.type, []).append(variation.name)

        # Return immutable sequences to prevent callers from mutating the
        # cached information by mistake.
        return {key: tuple(names) for key, names in grouped.items()}


def build_fake_factor_specs(channel_prefix: str, year: str) -> Dict[str, Tuple[str, str, str, str]]:
    """Return the fake-factor attribute specifications for the given channel."""

    base_attr = f"fakefactor_{channel_prefix}"
    return {
        "FF": (
            "absolute",
            base_attr,
            f"{base_attr}_up",
            f"{base_attr}_down",
        ),
        "FFpt": (
            "ratio",
            "nom",
            f"{base_attr}_pt1",
            f"{base_attr}_pt2",
        ),
        "FFeta": (
            "ratio",
            "nom",
            f"{base_attr}_be1",
            f"{base_attr}_be2",
        ),
        f"FFcloseEl_{year}": (
            "ratio",
            "nom",
            f"{base_attr}_elclosureup",
            f"{base_attr}_elclosuredown",
        ),
        f"FFcloseMu_{year}": (
            "ratio",
            "nom",
            f"{base_attr}_muclosureup",
            f"{base_attr}_muclosuredown",
        ),
    }


def add_fake_factor_weights(
    weights_object,
    events,
    channel_prefix: str,
    year: str,
    requested_data_weight_label: Optional[str] = None,
) -> None:
    """Register fake-factor weights for the requested lepton category."""

    # Preserve compatibility with callers that still pass the full channel name
    # (e.g. "2lss") by only using the leading lepton-multiplicity prefix when
    # building attribute names.
    channel_prefix = channel_prefix[:2]

    fake_factor_specs = build_fake_factor_specs(channel_prefix, year)

    requested_variations = None
    if requested_data_weight_label:
        spec = fake_factor_specs.get(requested_data_weight_label)
        if spec is None:
            raise ValueError(
                f"Unsupported fake-factor label '{requested_data_weight_label}' for '{channel_prefix}'"
            )

        mode, central_attr, up_attr, down_attr = spec
        if mode == "absolute":
            up_values = getattr(events, up_attr)
            down_values = getattr(events, down_attr)
        elif mode == "ratio":
            denominator = getattr(events, central_attr)
            up_values = getattr(events, up_attr) / denominator
            down_values = getattr(events, down_attr) / denominator
        else:
            raise ValueError(
                f"Unsupported fake-factor mode '{mode}' for '{requested_data_weight_label}'"
            )

        requested_variations = (
            copy.deepcopy(up_values),
            copy.deepcopy(down_values),
        )

    for label, (_, central_attr, _, _) in fake_factor_specs.items():
        central_values = getattr(events, central_attr)
        if requested_variations is not None and label == requested_data_weight_label:
            variations = requested_variations
        else:
            variations = ()
        weights_object.add(label, central_values, *variations)


def validate_data_weight_variations(
    weights_object,
    expected_variations: Iterable[str],
    requested_data_weight_label: Optional[str],
    current_variation_name: str,
) -> None:
    """Ensure data weights only contain supported variations."""

    variation_set = set(weights_object.variations)
    expected_variation_set = set(expected_variations)
    unexpected_variations = variation_set - expected_variation_set
    if unexpected_variations:
        raise Exception(
            "Error: Unexpected wgt variations for data! "
            f"Unexpected variations: {sorted(unexpected_variations)}"
        )

    if requested_data_weight_label:
        required_variations = {
            f"{requested_data_weight_label}Up",
            f"{requested_data_weight_label}Down",
        }
        missing_variations = required_variations - variation_set
        if missing_variations:
            raise Exception(
                "Error: Missing expected fake-factor variations for data! "
                f"Requested '{current_variation_name}' but did not find {sorted(missing_variations)}"
            )


def apply_theory_weight_variations(
    *,
    events,
    variation,
    variation_base: Optional[str],
    have_systematics: bool,
    sow: np.ndarray,
    sow_variations: MutableMapping[str, np.ndarray],
    sow_variation_key_map: MutableMapping[str, str],
    is_lo_sample: bool,
    hist_axis_name: str,
    sample: Mapping[str, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Return the coffea weight arguments for the theory variations."""

    if variation is not None:
        group_mapping = variation.group or {}
        group_key = (variation.base, variation.component, variation.year)
        group_info = group_mapping.get(group_key, {})

        if not group_info and variation.metadata.get("sum_of_weights"):
            group_info = {
                variation.name: {
                    "sum_of_weights": variation.metadata["sum_of_weights"]
                }
            }

        for member_name, info in group_info.items():
            sumw_key = info.get("sum_of_weights")
            if sumw_key:
                sow_variation_key_map.setdefault(member_name, sumw_key)

    include_flags = {
        "ISR": have_systematics and variation_base == "isr",
        "FSR": have_systematics and variation_base == "fsr",
        "renorm": have_systematics and variation_base == "renorm",
        "fact": have_systematics and variation_base == "fact",
        "renormfact": have_systematics and variation_base == "renormfact",
    }

    if include_flags["ISR"] or include_flags["FSR"]:
        tc_cor.AttachPSWeights(events)

    result: Dict[str, Tuple[np.ndarray, ...]] = {label: (events.nom,) for label in include_flags}

    variation_field_map = {
        "ISR": ("ISRUp", "ISRDown"),
        "FSR": ("FSRUp", "FSRDown"),
        "renorm": ("renormUp", "renormDown"),
        "fact": ("factUp", "factDown"),
        "renormfact": ("renormfactUp", "renormfactDown"),
    }

    current_variation_name = variation.name if variation is not None else "nominal"

    def get_sow_value(label: str) -> np.ndarray:
        if label in sow_variations:
            return sow_variations[label]

        if is_lo_sample:
            sow_variations[label] = sow
            return sow

        key = sow_variation_key_map.get(label)
        if key is None:
            raise KeyError(
                f"Unsupported sum-of-weights variation '{label}' requested while processing '{current_variation_name}'"
            )

        if key not in sample:
            raise KeyError(
                f"Sample '{hist_axis_name}' is missing required sum-of-weights entry '{key}' for '{label}' variation"
            )

        value = sample[key]
        sow_variations[label] = value
        return value

    for label, (up_field, down_field) in variation_field_map.items():
        if not include_flags[label]:
            continue

        sow_up = get_sow_value(up_field)
        sow_down = get_sow_value(down_field)

        result[label] = (
            events.nom,
            getattr(events, up_field) * (sow / sow_up),
            getattr(events, down_field) * (sow / sow_down),
        )

    return result


def register_weight_variation(
    weights,
    label: str,
    central,
    up=None,
    down=None,
    *,
    active: bool = False,
) -> None:
    """Register a weight with optional variations."""

    def _materialize(value, copy_value: bool):
        if value is None:
            return None
        result = value() if callable(value) else value
        return copy.deepcopy(result) if copy_value else result

    central_value = _materialize(central, copy_value=False)

    variation_values = ()
    if active:
        up_value = _materialize(up, copy_value=True)
        down_value = _materialize(down, copy_value=True)
        variation_values = tuple(value for value in (up_value, down_value) if value is not None)

    weights.add(label, central_value, *variation_values)


def register_lepton_sf_weight(
    weights_object,
    events,
    label: str,
    central_attr: str,
    up_attr: str,
    down_attr: str,
    include_variations: bool,
    *,
    variation_name: Optional[str] = None,
    logger_obj: Optional[logging.Logger] = None,
) -> None:
    """Register a lepton scale-factor weight, warning about missing variations."""

    central_values = getattr(events, central_attr)
    variation_values = ()

    if include_variations:
        missing_attrs = [
            attr_name
            for attr_name in (up_attr, down_attr)
            if not hasattr(events, attr_name)
        ]
        if missing_attrs:
            (logger_obj or logger).warning(
                "Requested lepton SF variation '%s' for weight '%s' but missing arrays: %s",
                variation_name,
                label,
                ", ".join(missing_attrs),
            )
        else:
            variation_values = (
                copy.deepcopy(getattr(events, up_attr)),
                copy.deepcopy(getattr(events, down_attr)),
            )

    weights_object.add(label, central_values, *variation_values)


__all__ = [
    "SystematicVariation",
    "SystematicVariationGroup",
    "SystematicsHelper",
    "add_fake_factor_weights",
    "apply_theory_weight_variations",
    "build_fake_factor_specs",
    "register_lepton_sf_weight",
    "register_weight_variation",
    "validate_data_weight_variations",
]
