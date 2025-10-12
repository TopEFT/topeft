"""Utilities for working with channel metadata from params/metadata.yml.

The helper classes defined here convert the structured YAML representation
of channel groups into convenient Python objects that can be consumed by the
Run 2 analysis scripts.  They intentionally mirror the information that used to
live in ``channels/ch_lst.json`` so downstream code can continue to operate on a
similar set of fields while benefiting from the richer metadata schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


def _normalize_histogram_list(
    values: Optional[object], *, context: str, field_name: str
) -> Tuple[str, ...]:
    """Return ``values`` as a tuple of strings with simple validation."""

    if values is None:
        return ()
    if isinstance(values, str):
        iterable = [values]
    elif isinstance(values, Sequence):
        iterable = values
    else:
        raise TypeError(
            f"{context} {field_name} must be a sequence of strings when provided"
        )

    normalized: List[str] = []
    for value in iterable:
        if not isinstance(value, str):
            raise TypeError(
                f"{context} {field_name} entries must be strings; got {type(value)!r}"
            )
        stripped = value.strip()
        if stripped:
            normalized.append(stripped)

    return tuple(normalized)


@dataclass(frozen=True)
class ChannelRegionDefinition:
    """Metadata describing a single region within a lepton category."""

    group_name: str
    lepton_category: str
    name: str
    channel: Optional[str]
    subchannel: Optional[str]
    tags: Tuple[str, ...]
    include_histograms: Tuple[str, ...]
    exclude_histograms: Tuple[str, ...]

    def to_legacy_list(self) -> List[str]:
        """Return the region definition in the legacy ``ch_lst.json`` format."""

        parts: List[str] = [self.name]
        if self.channel:
            parts.append(self.channel)
        if self.subchannel:
            parts.append(self.subchannel)
        parts.extend(self.tags)
        return parts


@dataclass
class ChannelCategory:
    """Container bundling the configuration for a lepton category."""

    group_name: str
    lepton_category: str
    lepton_flavors: List[str]
    jet_bins: List[str]
    application_tags_mc: List[str]
    application_tags_data: List[str]
    region_definitions: List[ChannelRegionDefinition]
    histogram_includes: Tuple[str, ...]
    histogram_excludes: Tuple[str, ...]

    def application_tags(self, include_data: bool) -> List[str]:
        """Return MC application tags plus data tags when requested."""

        tags = list(self.application_tags_mc)
        if include_data:
            tags.extend(self.application_tags_data)
        return tags


class ChannelGroup:
    """A named collection of lepton categories and their regions."""

    def __init__(self, name: str, group_metadata: Mapping[str, object]):
        self.name = name
        self.description = group_metadata.get("description", "")
        features = group_metadata.get("features", [])
        if features is None:
            features = []
        if isinstance(features, str) or not isinstance(features, Sequence):
            raise TypeError(
                f"Channel group {name!r} features must be a sequence of strings"
            )
        self._features: Tuple[str, ...] = tuple(str(feature) for feature in features)
        self._categories: List[ChannelCategory] = []
        for category in group_metadata.get("regions", []):
            lepton_category = category["lepton_category"]
            application_tags = category.get("application_tags", {})
            histogram_variables = category.get("histogram_variables", {})
            if histogram_variables is None:
                histogram_variables = {}
            if histogram_variables and not isinstance(histogram_variables, Mapping):
                raise TypeError(
                    f"Category {lepton_category!r} in group {name!r} must define"
                    " histogram_variables as a mapping"
                )
            category_hist_include = _normalize_histogram_list(
                histogram_variables.get("include"),
                context=f"Category {lepton_category!r} in group {name!r}",
                field_name="histogram_variables.include",
            )
            category_hist_exclude = _normalize_histogram_list(
                histogram_variables.get("exclude"),
                context=f"Category {lepton_category!r} in group {name!r}",
                field_name="histogram_variables.exclude",
            )
            region_definitions = [
                ChannelRegionDefinition(
                    group_name=name,
                    lepton_category=lepton_category,
                    name=region_def["name"],
                    channel=region_def.get("channel"),
                    subchannel=region_def.get("subchannel"),
                    tags=tuple(region_def.get("tags", [])),
                    include_histograms=_normalize_histogram_list(
                        (region_def.get("histogram_variables") or {}).get("include"),
                        context=(
                            f"Region {region_def.get('name', '<unknown>')!r}"
                            f" in group {name!r}"
                        ),
                        field_name="histogram_variables.include",
                    ),
                    exclude_histograms=_normalize_histogram_list(
                        (region_def.get("histogram_variables") or {}).get("exclude"),
                        context=(
                            f"Region {region_def.get('name', '<unknown>')!r}"
                            f" in group {name!r}"
                        ),
                        field_name="histogram_variables.exclude",
                    ),
                )
                for region_def in category.get("region_definitions", [])
            ]
            self._categories.append(
                ChannelCategory(
                    group_name=name,
                    lepton_category=lepton_category,
                    lepton_flavors=list(category.get("lepton_flavors", [])),
                    jet_bins=list(category.get("jet_bins", [])),
                    application_tags_mc=list(application_tags.get("mc", [])),
                    application_tags_data=list(application_tags.get("data", [])),
                    region_definitions=region_definitions,
                    histogram_includes=category_hist_include,
                    histogram_excludes=category_hist_exclude,
                )
            )
        self._category_map: Dict[str, ChannelCategory] = {
            category.lepton_category: category for category in self._categories
        }

    @property
    def features(self) -> Tuple[str, ...]:
        """Return the feature tags declared for this group."""

        return self._features

    def categories(self) -> Sequence[ChannelCategory]:
        """Return the lepton categories belonging to this group."""

        return tuple(self._categories)

    def category(self, lepton_category: str) -> Optional[ChannelCategory]:
        """Return the category metadata for ``lepton_category`` if present."""

        return self._category_map.get(lepton_category)


class ChannelMetadataHelper:
    """Helper exposing the grouped channel information from the metadata."""

    def __init__(self, channels_metadata: Mapping[str, object]):
        if not isinstance(channels_metadata, Mapping):
            raise TypeError("channels_metadata must be a mapping")

        groups_metadata = channels_metadata.get("groups", {})
        if not isinstance(groups_metadata, Mapping):
            raise TypeError("metadata['channels']['groups'] must be a mapping")
        self._groups: Dict[str, ChannelGroup] = {
            name: ChannelGroup(name, group_metadata)
            for name, group_metadata in groups_metadata.items()
        }

        scenarios_metadata = channels_metadata.get("scenarios", [])
        self._scenarios: Dict[str, ChannelScenario] = {}
        for scenario in scenarios_metadata:
            name = scenario.get("name")
            if not name:
                continue
            groups = tuple(scenario.get("groups", []))
            self._scenarios[name] = ChannelScenario(
                name=name,
                groups=groups,
            )

    def group(self, name: str) -> ChannelGroup:
        """Return the :class:`ChannelGroup` registered under ``name``."""

        try:
            return self._groups[name]
        except KeyError as exc:
            raise KeyError(f"Channel group {name!r} not found in metadata") from exc

    def iter_groups(self, names: Iterable[str]) -> Iterator[ChannelGroup]:
        """Yield groups in the order provided by ``names``."""

        for name in names:
            yield self.group(name)

    def group_names(self) -> Sequence[str]:
        """Return the available group names."""

        return tuple(self._groups.keys())

    def scenario_names(self) -> Sequence[str]:
        """Return the scenario names enumerated in the metadata."""

        return tuple(self._scenarios.keys())

    def scenario_groups(self, scenario_name: str) -> Tuple[str, ...]:
        """Return the group names participating in ``scenario_name``."""

        try:
            return self._scenarios[scenario_name].groups
        except KeyError as exc:
            raise KeyError(f"Scenario {scenario_name!r} not found in metadata") from exc

@dataclass(frozen=True)
class ChannelScenario:
    """Metadata describing the channel groups associated with a scenario."""

    name: str
    groups: Tuple[str, ...]
