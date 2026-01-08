#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

CH_LST_PATH = Path("topeft/channels/ch_lst.json")
META_YAML_PATH = Path("topeft/params/cr_sr_plots_metadata.yml")

REGION_SPECS = {
    "CR": {
        "ch_lst_keys": ("CH_LST_CR", "TAU_CH_LST_CR"),
        "yaml_key": "CR_CHAN_DICT",
        # For CRs we assume the processor was run with --split-lep-flavor
        "split_lep_flavor": True,
    },
    "SR": {
        "ch_lst_keys": (),
        "ch_lst_match": "CH_LST_SR",
        "yaml_key": "SR_CHAN_DICT",
        # For SRs we assume the processor was run without --split-lep-flavor
        "split_lep_flavor": False,
    },
}

# Semantic SR groups derived from channels.
# No 1b/2b categorization at group level: 3l_1tau collects all 3l+1tau channels.
SR_TAG_GROUP_RULES = {
    "2lss_SR": {
        "base": "2lss",
        "require": [],
        "forbid": ["fwd"],
    },
    "2lss_fwd": {
        "base": "2l",
        "require": ["fwd"],
        "forbid": [],
    },
    "2lss_1tau_onZ": {
        "base": "2lss_1tau",
        "require": ["onZ"],
        "forbid": [],
    },
    "2lss_1tau_offZ": {
        "base": "2lss_1tau",
        "require": ["offZ"],
        "forbid": [],
    },
    "2los_onZ_1tau": {
        "base": "2los_1tau",
        "require": ["onZ", "1tau"],
        "forbid": [],
    },
    "3l_onZ_SR": {
        "base": "3l",
        "require": ["onZ"],
        "forbid": ["1tau", "fwd"],
    },
    "3l_offZ_SR": {
        "base": "3l",
        "require": ["offZ"],
        "forbid": ["1tau", "fwd"],
    },
    # Combined 3l+1tau group (no 1b/2b split at group level)
    "3l_1tau": {
        "base": "3l",
        "require": ["1tau"],
        "forbid": ["fwd"],
    },
    "4l_SR": {
        "base": "4l",
        "require": [],
        "forbid": [],
    },
}


def _postprocess_sr_groups(sr_dict):
    """
    Apply higher-level physics/plotting conventions to SR_CHAN_DICT:
      * rename 3l_fwd -> 3l_onZ_fwd (it only contains onZ channels);
      * drop global base groups that are fully covered by more semantic ones.

    This function must NOT drop any unique channel labels: only rename or
    redistribute them into more specific groups.
    """
    sr_dict = copy.deepcopy(sr_dict)

    # 1) Rename 3l_fwd -> 3l_onZ_fwd (it only contains onZ channels)
    if "3l_fwd" in sr_dict:
        if "3l_onZ_fwd" in sr_dict:
            raise ValueError(
                "3l_onZ_fwd already exists when trying to rename 3l_fwd"
            )
        sr_dict["3l_onZ_fwd"] = sr_dict.pop("3l_fwd")

    # 2) Drop global / redundant SR bases which are now covered by semantic groups.
    #    We now also drop '2l' after introducing the 2lss_fwd semantic group,
    #    so that every 2l(SS) channel lives in exactly one high-level category.
    for key in ("3l", "4l", "2los_1tau", "2l"):
        sr_dict.pop(key, None)

    return sr_dict


def _construct_cat_name(chan_str, njet_str=None, flav_str=None):
    """Match analysis_processor.construct_cat_name for channel labels.

    For CRs (split_lep_flavor=True) we expect flav_str to be non-None and
    produce labels like:  3l_eee_CR_0j
    For SRs (split_lep_flavor=False) we pass flav_str=None and produce:
      3l_CR_0j, 3l_m_offZ_1b_2j, etc.
    """
    nlep_str = chan_str.split("_")[0]
    chan_str = "_".join(chan_str.split("_")[1:])
    if chan_str == "":
        chan_str = None

    if njet_str is not None:
        njet_str = njet_str[-2:]
        if "j" not in njet_str:
            raise ValueError(
                f"Invalid njet string '{njet_str}' derived from '{njet_str}'"
            )

    ret_str = nlep_str
    for component in [flav_str, chan_str, njet_str]:
        if component is None:
            continue
        ret_str = "_".join([ret_str, component])
    return ret_str


def _jet_cat_to_key(jet_cat):
    jet_cat = str(jet_cat)
    if jet_cat.startswith("="):
        jettag = "exactly_"
    elif jet_cat.startswith("<"):
        jettag = "atmost_"
    elif jet_cat.startswith(">"):
        jettag = "atleast_"
    else:
        raise ValueError(f"jet_cat {jet_cat} misses =,<,>!")

    return jettag + jet_cat.replace("=", "").replace("<", "").replace(">", "") + "j"


def _iter_channel_defs(cat_def):
    lep_chan_lst = cat_def.get("lep_chan_lst", [])
    for entry in lep_chan_lst:
        if isinstance(entry, (list, tuple)):
            if entry:
                yield entry[0]
        else:
            yield entry


def _resolve_ch_lst_keys(region_name, ch_cfg):
    spec = REGION_SPECS[region_name]
    explicit_keys = list(spec.get("ch_lst_keys", ()))
    if explicit_keys:
        missing = [key for key in explicit_keys if key not in ch_cfg]
        if missing:
            raise KeyError(
                f"Missing expected {region_name} ch_lst keys: {', '.join(missing)}"
            )
        return explicit_keys

    match_token = spec.get("ch_lst_match")
    if match_token:
        matched = sorted(key for key in ch_cfg if match_token in key)
        if matched:
            return matched

    raise KeyError(
        f"Unable to resolve {region_name} ch_lst keys from {', '.join(ch_cfg.keys())}"
    )


def _matches_tags(ch, require=(), forbid=()):
    """Return True when all require tags are present and no forbid tags match."""
    return all(tag in ch for tag in require) and not any(tag in ch for tag in forbid)


def build_channel_labels_from_ch_cfg(
    ch_cfg: dict, ch_lst_keys, split_lep_flavor: bool
) -> OrderedDict:
    """
    Reverse-engineered from topeft/analysis_processor.py, but implemented here.

    If split_lep_flavor is True (CR), we expand over lep_flav_lst and include
    the flavor in the label (e.g. 3l_eee_CR_0j).

    If split_lep_flavor is False (SR), we ignore lep_flav_lst and build labels
    without explicit flavor (e.g. 3l_CR_0j, 3l_m_offZ_1b_2j).

    Returns a mapping:
        base_name -> ordered list of full channel labels

    The returned labels must match the histogram channel labels
    that CR_CHAN_DICT or SR_CHAN_DICT refer to.
    """

    out = OrderedDict()
    seen_by_base = {}

    for key in ch_lst_keys:
        cat_block = ch_cfg.get(key) or {}
        for base_name, cat_def in cat_block.items():
            lep_flavs = list(cat_def.get("lep_flav_lst", []) or [None])
            jet_lst = cat_def.get("jet_lst", [])
            labels = out.setdefault(base_name, [])
            seen = seen_by_base.setdefault(base_name, set(labels))

            for jet_cat in jet_lst:
                jet_key = _jet_cat_to_key(jet_cat)

                # Decide whether to expand over lep_flav_lst
                if split_lep_flavor:
                    flav_loop = lep_flavs
                else:
                    flav_loop = [None]

                for lep_chan in _iter_channel_defs(cat_def):
                    for lep_flav in flav_loop:
                        label = _construct_cat_name(
                            lep_chan,
                            njet_str=jet_key,
                            flav_str=lep_flav,
                        )
                        if label in seen:
                            continue
                        seen.add(label)
                        labels.append(label)

    return out


def _build_semantic_sr_groups(sr_proc_bases):
    groups = {}
    for group_name, rule in SR_TAG_GROUP_RULES.items():
        base = rule["base"]
        if base not in sr_proc_bases:
            continue
        base_channels = sr_proc_bases[base]
        require = tuple(rule.get("require", ()))
        forbid = tuple(rule.get("forbid", ()))
        matches = [
            ch
            for ch in base_channels
            if _matches_tags(ch, require=require, forbid=forbid)
        ]
        groups[group_name] = sorted(matches)
    return groups


def _collect_region_data(region_name, ch_cfg, meta_cfg):
    spec = REGION_SPECS[region_name]
    ch_lst_keys = _resolve_ch_lst_keys(region_name, ch_cfg)
    split_lep_flavor = bool(spec.get("split_lep_flavor", False))

    proc_map = build_channel_labels_from_ch_cfg(
        ch_cfg, ch_lst_keys, split_lep_flavor=split_lep_flavor
    )

    yaml_key = spec["yaml_key"]
    if yaml_key not in meta_cfg:
        raise KeyError(f"Missing expected YAML key '{yaml_key}'")

    yaml_block = meta_cfg.get(yaml_key) or {}
    yaml_labels_by_base = {
        base: list(labels or []) for base, labels in yaml_block.items()
    }

    proc_labels = sorted({lab for labs in proc_map.values() for lab in labs})
    yaml_labels = sorted(
        {lab for labels in yaml_labels_by_base.values() for lab in labels}
    )

    proc_set = set(proc_labels)
    yaml_set = set(yaml_labels)

    return {
        "region": region_name,
        "yaml_key": yaml_key,
        "ch_lst_keys": ch_lst_keys,
        "proc_map": proc_map,
        "yaml_map": yaml_labels_by_base,
        "proc_labels": proc_labels,
        "yaml_labels": yaml_labels,
        "only_in_processor": sorted(proc_set - yaml_set),
        "only_in_yaml": sorted(yaml_set - proc_set),
    }


def _print_region_report(region_data):
    region = region_data["region"]
    yaml_key = region_data["yaml_key"]

    print(
        f"=== {region} channel labels that the PROCESSOR can build, but are MISSING in YAML {yaml_key} ==="
    )
    for lab in region_data["only_in_processor"]:
        print(f"  - {lab}")

    print(
        f"\n=== YAML {yaml_key} labels that the PROCESSOR would NEVER produce ==="
    )
    for lab in region_data["only_in_yaml"]:
        print(f"  - {lab}")

    print(f"\n=== Debug: processor {region} bases and their first few channels ===")
    for base, labs in sorted(region_data["proc_map"].items()):
        labs_sorted = sorted(labs)
        preview = ", ".join(labs_sorted[:5])
        more = "" if len(labs_sorted) <= 5 else f" ... (+{len(labs_sorted)-5} more)"
        print(f"  {base}: {preview}{more}")


def _augment_channel_dict(proc_map, yaml_block):
    updated = copy.deepcopy(yaml_block)
    labels_added = 0
    bases_created = 0

    for base, proc_labels in proc_map.items():
        if base not in updated:
            updated[base] = list(proc_labels)
            bases_created += 1
            labels_added += len(proc_labels)
            continue

        existing = list(updated.get(base) or [])
        existing_set = set(existing)
        additions = [label for label in proc_labels if label not in existing_set]
        if additions:
            existing.extend(additions)
            updated[base] = existing
            labels_added += len(additions)

    return updated, labels_added, bases_created


def _exit_with_error(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _read_json(path: Path):
    try:
        with path.open() as f:
            return json.load(f)
    except OSError as exc:
        _exit_with_error(f"unable to read JSON from {path}: {exc}")
    except json.JSONDecodeError as exc:
        _exit_with_error(f"invalid JSON in {path}: {exc}")


def _read_yaml(path: Path):
    try:
        with path.open() as f:
            return yaml.safe_load(f)
    except OSError as exc:
        _exit_with_error(f"unable to read YAML from {path}: {exc}")
    except yaml.YAMLError as exc:
        _exit_with_error(f"invalid YAML in {path}: {exc}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit CR/SR channel labels from ch_lst.json against the YAML metadata."
        )
    )
    parser.add_argument(
        "--ch-lst",
        default=str(CH_LST_PATH),
        help="Path to ch_lst.json (default: topeft/channels/ch_lst.json)",
    )
    parser.add_argument(
        "--meta-yaml",
        default=str(META_YAML_PATH),
        help="Path to cr_sr_plots_metadata.yml (default: topeft/params/cr_sr_plots_metadata.yml)",
    )
    parser.add_argument(
        "--out-meta-yaml",
        default=None,
        help="Output path for the augmented YAML (required with --write-augmented-yaml)",
    )
    parser.add_argument(
        "--regions",
        choices=("CR", "SR", "both"),
        default="both",
        help="Which region(s) to audit (default: both)",
    )
    parser.add_argument(
        "--write-augmented-yaml",
        action="store_true",
        help="Write an augmented YAML with missing labels filled in",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.write_augmented_yaml and not args.out_meta_yaml:
        raise ValueError("--out-meta-yaml is required with --write-augmented-yaml")

    ch_lst_path = Path(args.ch_lst)
    meta_yaml_path = Path(args.meta_yaml)

    ch_cfg = _read_json(ch_lst_path)
    meta_cfg = _read_yaml(meta_yaml_path)

    regions = ["CR", "SR"] if args.regions == "both" else [args.regions]

    region_data = []
    for region in regions:
        data = _collect_region_data(region, ch_cfg, meta_cfg)
        region_data.append(data)
        _print_region_report(data)

    if not args.write_augmented_yaml:
        return

    out_meta = copy.deepcopy(meta_cfg)
    for data in region_data:
        yaml_key = data["yaml_key"]

        # Do NOT modify CR_CHAN_DICT: keep it identical to the input YAML.
        if data["region"] != "SR":
            out_meta[yaml_key] = meta_cfg.get(yaml_key, {})
            print(
                f"\n=== Skipping augmentation for {data['region']} "
                f"(leaving {yaml_key} unchanged) ==="
            )
            continue

        # For SR, augment + apply semantic/post-processing rules.
        updated, labels_added, bases_created = _augment_channel_dict(
            data["proc_map"], out_meta.get(yaml_key, {})
        )

        semantic_groups = _build_semantic_sr_groups(data["proc_map"])
        for name, chans in semantic_groups.items():
            updated[name] = chans

        # Drop obsolete fixed-jet 4l groups if present
        for obsolete in ("4l_2j", "4l_3j", "4l_4j"):
            updated.pop(obsolete, None)

        # Apply higher-level SR post-processing: rename and clean up groups
        updated = _postprocess_sr_groups(updated)

        out_meta[yaml_key] = updated

        proc_count = sum(len(labels) for labels in data["proc_map"].values())
        yaml_count = sum(len(labels) for labels in data["yaml_map"].values())
        print(f"\n=== Augmented YAML summary ({data['region']}) ===")
        print(f"  processor labels: {proc_count}")
        print(f"  yaml labels: {yaml_count}")
        print(f"  labels added: {labels_added}")
        print(f"  bases created: {bases_created}")

    out_path = Path(args.out_meta_yaml)
    with out_path.open("w") as f:
        yaml.safe_dump(out_meta, f, sort_keys=False)

    print(f"\nWrote augmented YAML to {out_path}")


if __name__ == "__main__":
    main()