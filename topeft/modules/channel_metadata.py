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


@dataclass(frozen=True)
class ChannelRegionDefinition:
    """Metadata describing a single region within a lepton category."""

    group_name: str
    lepton_category: str
    name: str
    channel: Optional[str]
    subchannel: Optional[str]
    tags: Tuple[str, ...]

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
            region_definitions = [
                ChannelRegionDefinition(
                    group_name=name,
                    lepton_category=lepton_category,
                    name=region_def["name"],
                    channel=region_def.get("channel"),
                    subchannel=region_def.get("subchannel"),
                    tags=tuple(region_def.get("tags", [])),
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
        self._scenarios: Dict[str, Tuple[str, ...]] = {}
        for scenario in scenarios_metadata:
            name = scenario.get("name")
            if not name:
                continue
            groups = tuple(scenario.get("groups", []))
            self._scenarios[name] = groups

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
            return self._scenarios[scenario_name]
        except KeyError as exc:
            raise KeyError(f"Scenario {scenario_name!r} not found in metadata") from exc
