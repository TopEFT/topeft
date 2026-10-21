#!/usr/bin/env python3
"""Project a combined datacard package and fit config into portable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

import yaml

_analysis = "TOP-26-006"
_schema = "TOP26006_fit_input_v1"
_metadata_hashes = {
    "scalings.json": "scalings_sha256",
    "ordered_card_inputs.txt": "ordered_card_inputs_sha256",
    "combined_mapping_manifest.json": "manifest_sha256",
}
_source_top = set(_metadata_hashes) | {"cards", "package-provenance.json"}
_mapping_fields = (
    "era", "physical_name", "per_era_chN", "combined_chN",
    "combined_order_index", "destination_txt_name", "destination_root_name",
)
_provenance_fields = (
    "package_version", "package_date", "assembler_commit", "assembler_source_sha256",
    "source_mapping_sha256", "source_scalings_sha256",
)


class FitInputError(ValueError):
    """A selected input or destination violates the fit-input contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitInputError(message)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _plain_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def _read_regular_bytes(path: Path) -> bytes:
    _require(_regular_file(path), f"missing or non-regular input: {path}")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                 f"input changed type: {path}")
        return stream.read()


def _validate_source_members(package: Path) -> None:
    _require(_plain_dir(package), "combined package must be a plain directory")
    _require(_plain_dir(package / "cards"), "cards must be a plain directory")
    for name in _source_top - {"cards"}:
        _require(_regular_file(package / name), f"non-regular source metadata: {name}")


def _valid_basename(value: object, suffix: str) -> bool:
    return (isinstance(value, str) and value not in {"", ".", ".."}
            and "/" not in value and "\\" not in value and value.endswith(suffix))


def _source_contract(package: Path) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    try:
        provenance = json.loads(_read_regular_bytes(package / "package-provenance.json"))
        mapping = json.loads(_read_regular_bytes(package / "combined_mapping_manifest.json"))
        json.loads(_read_regular_bytes(package / "scalings.json"))
        ordered = _read_regular_bytes(package / "ordered_card_inputs.txt").decode("utf-8").splitlines()
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FitInputError(f"invalid combined package metadata: {error}") from error
    _require(isinstance(provenance, dict) and provenance.get("schema") == "TOP26006_v1"
             and provenance.get("artifact_type") == "package_provenance"
             and provenance.get("analysis") == _analysis, "combined provenance identity is invalid")
    _require(isinstance(mapping, dict) and mapping.get("schema") == "TOP26006_v1"
             and mapping.get("artifact_type") == "combined_mapping_manifest"
             and isinstance(mapping.get("rows"), list) and bool(mapping["rows"]),
             "combined mapping metadata is invalid")
    _require(bool(ordered) and len(ordered) == len(set(ordered)),
             "ordered card inputs missing or duplicate")
    for entry in ordered:
        _require(entry.startswith("cards/") and _valid_basename(entry.removeprefix("cards/"), ".txt"),
                 f"invalid ordered card entry: {entry}")
    for name, field in _metadata_hashes.items():
        expected = provenance.get(field)
        _require(isinstance(expected, str) and bool(re.fullmatch(r"[0-9a-f]{64}", expected)),
                 f"missing declared metadata hash: {field}")
        _require(_sha256_bytes(_read_regular_bytes(package / name)) == expected,
                 f"declared metadata hash mismatch: {name}")
    rows = mapping["rows"]
    _require(len(rows) == len(ordered), "mapping and ordered card counts differ")
    projected = []
    selected_names = set()
    for index, (row, entry) in enumerate(zip(rows, ordered), start=1):
        _require(isinstance(row, dict) and all(field in row for field in _mapping_fields),
                 f"mapping row {index} lacks consumer fields")
        _require(row["combined_order_index"] == index
                 and row["destination_txt_name"] == entry.removeprefix("cards/")
                 and _valid_basename(row["destination_txt_name"], ".txt")
                 and _valid_basename(row["destination_root_name"], ".root")
                 and all(isinstance(row[field], str) and bool(row[field])
                         for field in _mapping_fields if field != "combined_order_index"),
                 f"mapping row {index} disagrees with ordered card inputs")
        for name in (row["destination_txt_name"], row["destination_root_name"]):
            _require(name not in selected_names, f"duplicate mapped card member: {name}")
            selected_names.add(name)
            _require(_regular_file(package / "cards" / name),
                     f"missing or non-regular mapped card member: {name}")
        projected.append({field: row[field] for field in _mapping_fields})
    source_provenance = {}
    for field in _provenance_fields:
        value = provenance.get(field)
        if field in {"package_version", "package_date"}:
            valid = isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))
        elif field == "assembler_commit":
            valid = isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))
        elif field == "assembler_source_sha256":
            valid = isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
        else:
            valid = isinstance(value, dict) and all(
                key in {"run2", "run3"} and isinstance(digest, str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", digest))
                for key, digest in value.items()
            )
        if valid:
            source_provenance[field] = value
    return projected, source_provenance, sorted(f"cards/{name}" for name in selected_names)


def _validate_commit(value: str | None, field: str) -> None:
    _require(value is None or (isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))),
             f"invalid provenance commit syntax: {field}")


def _config_bytes(config: Path) -> bytes:
    content = _read_regular_bytes(config)
    try:
        parsed = yaml.safe_load(content)
    except (yaml.YAMLError, UnicodeError) as error:
        raise FitInputError(f"malformed fit config: {error}") from error
    _require(isinstance(parsed, dict) and parsed.get("analysis") == "TOP_26_006"
             and isinstance(parsed.get("wc_population"), dict)
             and isinstance(parsed["wc_population"].get("scientific"), list)
             and bool(parsed["wc_population"]["scientific"])
             and isinstance(parsed.get("wc_ranges"), dict) and bool(parsed["wc_ranges"]),
             "fit config lacks required analysis, WC population, or ranges")
    return content


def _copy_and_hash(source: Path, destination: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as input_stream, \
            destination.open("xb") as output_stream:
        _require(stat.S_ISREG(os.fstat(input_stream.fileno()).st_mode),
                 f"source member changed type: {source}")
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(block)
            digest.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _hash_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def _readme() -> str:
    return (
        "# TOP-26-006 fit input package\n\n"
        "`datacards/ordered_card_inputs.txt` is the sole card-order authority. "
        "Its order must stay aligned with `datacards/scalings.json` and the "
        "manifest channel mapping. Never discover cards with a wildcard or glob.\n\n"
        "EFTFit materializes a writable cards directory containing copies of "
        "`datacards/cards/**`, including the ROOT templates. It reads the order "
        "file, validates every line as exactly `cards/<name>.txt`, removes only "
        "the leading `cards/` in memory, and preserves the order. Then, from "
        "inside that writable cards directory, EFTFit executes:\n\n"
        "```bash\n"
        "combineCards.py \"${ordered_basenames[@]}\" > combinedcard.txt\n"
        "```\n\n"
        "`ordered_basenames` is the ordered array obtained by that validation "
        "and prefix removal. `combinedcard.txt` is co-located with individual "
        "cards and ROOT templates in the EFTFit-owned writable cards directory. "
        "TOPEFT documents this command but does not execute it or create the "
        "combined card. EFTFit also owns workspaces and all Combine and "
        "CombineHarvester products. `config/TOP_26_006.yml` is the package-local "
        "fit configuration; any selectedWCs.txt is derived downstream.\n"
    )


def build_fit_input_package(
    package: Path, fit_config: Path, output: Path, *,
    package_creation_topeft_commit: str | None = None,
    input_production_topeft_commit: str | None = None,
    input_production_topcoffea_commit: str | None = None,
) -> dict[str, object]:
    package, fit_config, output = Path(package), Path(fit_config), Path(output)
    staging = output.parent / f".{output.name}.staging"
    _require(output.is_absolute() and output.name not in {"", ".", ".."}, "output must be absolute")
    _require(_plain_dir(output.parent), "output parent must be a plain existing directory")
    _require(not output.exists() and not output.is_symlink(), "output already exists")
    _require(not staging.exists() and not staging.is_symlink(), "staging already exists")
    for first, second in ((package, output), (package, staging),
                          (fit_config, output), (fit_config, staging), (staging, output)):
        _require(not _overlap(first, second),
                 "output and staging must not overlap either input or each other")
    for field, value in (
        ("package_creation.topeft_commit", package_creation_topeft_commit),
        ("input_production.topeft_commit", input_production_topeft_commit),
        ("input_production.topcoffea_commit", input_production_topcoffea_commit),
    ):
        _validate_commit(value, field)
    _validate_source_members(package)
    mapping_rows, source_provenance, card_members = _source_contract(package)
    config_content = _config_bytes(fit_config)
    members = sorted(card_members + ["ordered_card_inputs.txt", "scalings.json"])
    staging.mkdir(mode=0o700)
    try:
        datacards = staging / "datacards"
        datacards.mkdir()
        source_inventory = []
        for relative in members:
            destination = datacards / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_inventory.append({"path": f"datacards/{relative}",
                                     **_copy_and_hash(package / relative, destination)})
        config_destination = staging / "config" / "TOP_26_006.yml"
        config_destination.parent.mkdir()
        config_destination.write_bytes(config_content)
        destination_inventory = [
            {"path": f"datacards/{relative}", **_hash_file(datacards / relative)}
            for relative in members
        ]
        _require(source_inventory == destination_inventory, "datacard copy identity mismatch")
        config_sha = _sha256_bytes(config_content)
        _require(config_sha == _hash_file(config_destination)["sha256"],
                 "fit config copy identity mismatch")
        manifest = {
            "schema": _schema, "artifact_type": "fit_input_manifest", "analysis": _analysis,
            "datacards": {
                "cards_path": "datacards/cards",
                "ordered_card_inputs_path": "datacards/ordered_card_inputs.txt",
                "scalings_path": "datacards/scalings.json",
                "input_snapshot_sha256": _sha256_bytes(_json_bytes({
                    "files": source_inventory,
                    "channel_mapping": {"rows": mapping_rows},
                })),
                "files": source_inventory,
                "channel_mapping": {"rows": mapping_rows},
            },
            "fit_config": {"path": "config/TOP_26_006.yml", "sha256": config_sha},
            "provenance": {
                "package_creation": {"topeft_commit": package_creation_topeft_commit},
                "input_production": {"topeft_commit": input_production_topeft_commit,
                                     "topcoffea_commit": input_production_topcoffea_commit},
                "source_datacard_package": source_provenance,
            },
        }
        (staging / "fit-input-manifest.json").write_bytes(_json_bytes(manifest))
        (staging / "README.md").write_text(_readme(), encoding="utf-8")
        _require(not output.exists() and not output.is_symlink(), "output appeared before publication")
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-package", type=Path, required=True)
    parser.add_argument("--fit-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-creation-topeft-commit")
    parser.add_argument("--input-production-topeft-commit")
    parser.add_argument("--input-production-topcoffea-commit")
    args = parser.parse_args(argv)
    manifest = build_fit_input_package(
        args.combined_package, args.fit_config, args.output,
        package_creation_topeft_commit=args.package_creation_topeft_commit,
        input_production_topeft_commit=args.input_production_topeft_commit,
        input_production_topcoffea_commit=args.input_production_topcoffea_commit,
    )
    print(json.dumps({"output": str(args.output), "input_snapshot_sha256":
                      manifest["datacards"]["input_snapshot_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
