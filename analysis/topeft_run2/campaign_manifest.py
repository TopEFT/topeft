"""Canonical plan, state, receipt, and recovery engine for component campaigns.

The launcher owns scientific profile selection and child execution. This module
owns the durable schema-9 state and checks every transition against that state.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 9
RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
CHANNELS_PATH = REPOSITORY_ROOT / "topeft/channels/ch_lst.json"
DEFAULT_SR_PLAN = RUN_DIRECTORY / "sr_reco_default_plan.json"
UNIT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
PROFILE_PATTERN = re.compile(r"(run2|run3)_(prod_2los_1tau|prod|stat_diagnostic|sr_ext)(_CR)?\Z")
RUN2_YEARS = ("2016APV", "2016", "2017", "2018")
RUN3_YEARS = ("2022", "2022EE", "2023", "2023BPix")
DATA_DRIVEN_SCIENTIFIC_PATHS = (
    "topeft/modules/dataDrivenEstimation.py",
    "topeft/modules/nonprompt_policy.py",
)
DATA_DRIVEN_SCIENTIFIC_SYMBOLS = {
    "analysis/topeft_run2/run_data_driven.py": (
        "_filter_to_flips", "_filter_to_allowed_processes",
    ),
    "topeft/modules/data_driven_products.py": (
        "DATA_DRIVEN_PRODUCT_NAMES", "generated_process_name",
    ),
}
# The accepted schema-9 DD receipts predate the symbol-level wrapper identity.
# This exact historical file had the same two science-bearing filter functions;
# its later edits only added output-safety CLI handling.
LEGACY_DATA_DRIVEN_WRAPPER_SYMBOLS = {
    "724a46b07e1d9d7b17888fa79acd2fddb1a8cd057f9989a12029c516c4538746":
        "05b7f4dbca8bd027af2d4d2cdebb087580f14761b79547f5144c427375585530",
}
SR_EXTENSION_VARIABLES = frozenset(
    ("fwd0pt", "j0pt", "b0pt", "met", "l0conept", "l1conept", "invmass", "tau0Tpt")
)


def fail(message):
    raise ValueError(message)


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def campaign_lock(state_path):
    state_path = Path(state_path)
    lock_path = state_path.with_name(re.sub(r"_state(?:_v9)?\.json$", ".lock", state_path.name))
    if lock_path.is_symlink():
        fail(f"campaign writer lock is a symbolic link: {lock_path}")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("campaign writer lock is held")
        yield
    finally:
        os.close(descriptor)


def profile_contract(profile, *, region=None, mode=None, run_scope=None,
                     do_systs=None, do_np=None, ttgamma_policy="split"):
    match = PROFILE_PATTERN.fullmatch(profile)
    if match is None or (match.group(2) in {"sr_ext", "prod_2los_1tau"} and match.group(3)):
        fail(f"unsupported component profile: {profile}")
    expected_scope, expected_mode, suffix = match.groups()
    expected_region = "CR" if suffix else "SR"
    if region is not None and region != expected_region:
        fail("profile name and region contradict")
    if mode is not None and mode != expected_mode:
        fail("profile name and mode contradict")
    if run_scope is not None and run_scope != expected_scope:
        fail("profile name and run scope contradict")
    expected_systs = expected_mode != "stat_diagnostic"
    if do_systs is not None and do_systs != expected_systs:
        fail("profile name and systematics policy contradict")
    if do_np is not None and do_np is not True:
        fail("maintained profile requires its data-driven contract")
    if ttgamma_policy != "split":
        fail("unsupported ttgamma sample-role policy")
    return {
        "run_scope": expected_scope,
        "region": expected_region,
        "mode": expected_mode,
        "systematics": expected_systs,
        "sumw2_policy": "full_diagnostics" if not expected_systs else "implicit_production",
        "raw_counts": not expected_systs,
        "data_driven_policy": "nonprompt_charge_flip",
        "ttgamma_sample_role_policy": ttgamma_policy,
        "stages": ["source", "data_driven"],
    }


def expanded_years(years):
    expanded = []
    for year in years:
        expanded.extend(RUN2_YEARS if year == "run2" else RUN3_YEARS if year == "run3" else (year,))
    if not expanded or len(expanded) != len(set(expanded)):
        fail("unit years must be unique and nonempty")
    if not set(expanded) <= set((*RUN2_YEARS, *RUN3_YEARS)):
        fail("unit declares unsupported years")
    return sorted(expanded)


def validate_unit(unit, profile):
    if not isinstance(unit, dict) or not {"id", "years", "category_groups", "histograms"} <= set(unit):
        fail("plan unit requires id, years, category_groups, histograms")
    if not isinstance(unit["id"], str) or not UNIT_ID_PATTERN.fullmatch(unit["id"]):
        fail("unit id must be snake_case")
    for name in ("years", "category_groups", "histograms"):
        values = unit[name]
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
            fail(f"unit {name} must be a unique nonempty string list")
    years = expanded_years(unit["years"])
    allowed = set(RUN2_YEARS if profile.startswith("run2_") else RUN3_YEARS)
    if not set(years) <= allowed:
        fail("unit years contradict the profile run scope")
    return unit


def sr_physical_pairs():
    channels = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))["ALL_CH_LST_SR"]
    approved = {"2l", "2lss_1tau", "2los_1tau", "4l", "3l_m_offZ", "3l_p_offZ", "3l_onZ_tau", "3l_fwd"}
    result = {}
    for group in approved:
        definition = channels[group]
        result[group] = {(group, channel[0], jet) for channel in definition["lep_chan_lst"] for jet in definition["jet_lst"]}
    return result


def sr_applicable(pair, variable):
    channel = pair[1]
    if variable == "fwd0pt":
        return channel.startswith("2lss_fwd_") or channel.endswith("_fwd")
    if variable == "tau0Tpt":
        return "_1tau" in channel
    return True


def unit_coverage(unit, profile):
    validate_unit(unit, profile)
    years = expanded_years(unit["years"])
    if profile.endswith("_sr_ext"):
        pairs = sr_physical_pairs()
        if not set(unit["category_groups"]) <= pairs.keys() or not set(unit["histograms"]) <= SR_EXTENSION_VARIABLES:
            fail("SR-extension unit declares unknown category or histogram")
        items = {
            (year, *pair, variable)
            for year in years
            for group in unit["category_groups"]
            for pair in pairs[group]
            for variable in unit["histograms"]
            if sr_applicable(pair, variable)
        }
        if any(not any(item[-1] == variable for item in items) for variable in unit["histograms"]):
            fail("SR-extension unit contains an inapplicable histogram")
    else:
        items = {
            (year, group, variable)
            for year in years
            for group in unit["category_groups"]
            for variable in unit["histograms"]
        }
    return {"|".join(item) for item in items}


def partition_coverage(units, profile):
    counts = Counter(key for unit in units for key in unit_coverage(unit, profile))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    if duplicate:
        fail(f"plan overlaps scientific coverage: {duplicate[:3]}")
    return sorted(counts)


def verify_partition(units, profile, approved_coverage):
    observed = partition_coverage(units, profile)
    if observed != list(approved_coverage):
        missing = sorted(set(approved_coverage) - set(observed))
        added = sorted(set(observed) - set(approved_coverage))
        fail(f"plan changes scientific coverage: missing={missing[:3]} added={added[:3]}")
    return observed


def verify_recorded_partition(specs, profile, approved_coverage):
    """Check frozen schema-9 coverage without consulting the host channel registry."""
    if (not isinstance(approved_coverage, list) or not approved_coverage
            or any(not isinstance(row, str) or not row for row in approved_coverage)
            or approved_coverage != sorted(set(approved_coverage))):
        fail("recorded scientific coverage is not a unique sorted list")
    counts = Counter()
    for spec in specs:
        validate_unit(spec, profile)
        rows = spec.get("coverage")
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, str) or not row for row in rows)
                or rows != sorted(set(rows))):
            fail(f"{spec['id']} recorded coverage is invalid")
        years = set(expanded_years(spec["years"]))
        groups = set(spec["category_groups"])
        variables = set(spec["histograms"])
        for row in rows:
            fields = row.split("|")
            if profile.endswith("_sr_ext"):
                if (len(fields) != 5 or not fields[2] or not fields[3]
                        or not sr_applicable((fields[1], fields[2], fields[3]), fields[4])):
                    fail(f"{spec['id']} recorded SR coverage row is invalid")
                year, group, variable = fields[0], fields[1], fields[4]
            else:
                if len(fields) != 3:
                    fail(f"{spec['id']} recorded coverage row is invalid")
                year, group, variable = fields
            if year not in years or group not in groups or variable not in variables:
                fail(f"{spec['id']} recorded coverage contradicts its unit")
            counts[row] += 1
    duplicate = sorted(row for row, count in counts.items() if count != 1)
    if duplicate:
        fail(f"plan overlaps scientific coverage: {duplicate[:3]}")
    observed = sorted(counts)
    if observed != approved_coverage:
        missing = sorted(set(approved_coverage) - set(observed))
        added = sorted(set(observed) - set(approved_coverage))
        fail(f"plan changes scientific coverage: missing={missing[:3]} added={added[:3]}")
    return observed


def read_default_tsv(path, profile):
    units = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 8:
            fail("default plan row requires eight fields")
        unit = {
            "id": fields[0],
            "years": fields[1].split(),
            "category_groups": fields[2].split(),
            "histograms": fields[3].split(),
        }
        units.append(validate_unit(unit, profile))
    if not units or len({unit["id"] for unit in units}) != len(units):
        fail("default plan must contain distinct units")
    return units


def default_units(profile, default_tsv):
    if profile.endswith("_sr_ext"):
        source = DEFAULT_SR_PLAN if default_tsv == "-" else Path(default_tsv)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if set(raw) != {"units"} or not isinstance(raw["units"], list):
            fail("invalid SR-extension scientific default plan")
        years = ["run2"] if profile.startswith("run2_") else list(RUN3_YEARS)
        units = [{**unit, "years": years} for unit in raw["units"]]
    else:
        units = read_default_tsv(default_tsv, profile)
    for unit in units:
        validate_unit(unit, profile)
    return units


def supplied_units(path, profile):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {"units"} or not isinstance(raw["units"], list) or not raw["units"]:
        fail("supplied plan requires one nonempty units list")
    default_years = ["run2"] if profile.startswith("run2_") else list(RUN3_YEARS)
    units = []
    for supplied in raw["units"]:
        if not isinstance(supplied, dict) or not {"id", "category_groups", "histograms"} <= supplied.keys() or set(supplied) - {"id", "years", "category_groups", "histograms"}:
            fail("supplied plan unit has unsupported fields")
        units.append(validate_unit({**supplied, "years": supplied.get("years", default_years)}, profile))
    if len({unit["id"] for unit in units}) != len(units):
        fail("supplied plan contains duplicate unit ids")
    return units


def plan_partition_sha256(units):
    return canonical_digest(units)


def make_revision(number, units):
    return f"r{number}_{plan_partition_sha256(units)[:12]}"


def make_specs(units, profile, tag, root, revision):
    contract = profile_contract(profile)
    region = contract["region"]
    prefix = profile.lower()
    specs = []
    for unit in units:
        unit_id = unit["id"] if unit["id"].startswith(f"{prefix}_") else f"{prefix}_{unit['id']}"
        output_tag = f"{tag}_{revision}_{unit_id}"
        years = expanded_years(unit["years"])
        year_label = "UL16-UL16APV-UL17-UL18" if years == sorted(RUN2_YEARS) else "-".join(unit["years"])
        output_name = f"{year_label}{region}s_{output_tag}"
        source = str(Path(root) / f"{output_name}.pkl.gz")
        data_driven = str(Path(root) / f"{output_name}_np.pkl.gz")
        specs.append({
            **unit, "id": unit_id, "output_tag": output_tag,
            "output_name": output_name, "outputs": {"source": source, "data_driven": data_driven},
            "coverage": sorted(unit_coverage(unit, profile)), "plan_revision": revision,
        })
    validate_output_namespace(specs)
    return specs


def validate_output_namespace(specs):
    owners = {}
    for spec in specs:
        for path in spec["outputs"].values():
            for candidate in (path, f"{path}.metadata.json"):
                if candidate in owners:
                    fail(f"output path collision between {owners[candidate]} and {spec['id']}")
                owners[candidate] = spec["id"]


def plan_tsv(specs):
    return "".join("\t".join((
        spec["id"], " ".join(spec["years"]), " ".join(spec["category_groups"]),
        " ".join(spec["histograms"]), spec["output_tag"], spec["output_name"],
        spec["outputs"]["source"], spec["outputs"]["data_driven"],
    )) + "\n" for spec in specs)


def parse_plan_tsv(path, profile):
    specs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 8:
            fail("rendered plan row requires eight fields")
        unit = {"id": fields[0], "years": fields[1].split(), "category_groups": fields[2].split(), "histograms": fields[3].split()}
        validate_unit(unit, profile)
        match = re.search(r"_(r[0-9]+_[0-9a-f]{12})_", fields[4])
        if match is None:
            fail("rendered unit lacks revision identity")
        specs.append({
            **unit, "output_tag": fields[4], "output_name": fields[5],
            "outputs": {"source": fields[6], "data_driven": fields[7]},
            "coverage": sorted(unit_coverage(unit, profile)), "plan_revision": match.group(1),
        })
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        fail("rendered plan must have distinct units")
    validate_output_namespace(specs)
    return specs


def sample_cfg_names(contract):
    if contract["run_scope"] == "run2":
        names = ["mc_signal_samples_NDSkim.cfg", "mc_background_samples_NDSkim.cfg", "data_samples_NDSkim.cfg"]
        if contract["region"] == "CR":
            names.append("mc_background_samples_cr_NDSkim.cfg")
    else:
        kinds = ("background_samples_cr", "data_samples", "mc_signal_samples") if contract["region"] == "CR" else ("background_samples", "data_samples", "mc_signal_samples_sr")
        names = [f"NDSkim_{year}_{kind}.cfg" for year in RUN3_YEARS for kind in kinds]
    return names


def scientific_files(contract, stage="source"):
    names = set(DATA_DRIVEN_SCIENTIFIC_PATHS if stage == "data_driven" else ())
    if stage == "source":
        for cfg_name in sample_cfg_names(contract):
            cfg = REPOSITORY_ROOT / "input_samples/cfgs" / cfg_name
            names.add(str(cfg.relative_to(REPOSITORY_ROOT)))
            for line in cfg.read_text(encoding="utf-8").splitlines():
                entry = line.strip()
                if entry.startswith("../../input_samples/sample_jsons/") and entry.endswith(".json"):
                    sample = (RUN_DIRECTORY / entry).resolve()
                    if not sample.is_relative_to(REPOSITORY_ROOT):
                        fail("sample JSON escapes the repository")
                    names.add(str(sample.relative_to(REPOSITORY_ROOT)))
    digests = {
        name: digest_file(REPOSITORY_ROOT.parent / name if name.startswith("topcoffea/") else REPOSITORY_ROOT / name)
        for name in sorted(names)
    }
    if stage == "data_driven":
        from topeft.modules.histogram_artifact import (
            FLIPS_APPLICATION_REGION, NONPROMPT_APPLICATION_REGIONS,
        )

        parameters = json.loads((REPOSITORY_ROOT / "topeft/params/params.json").read_text(encoding="utf-8"))
        digests["topeft/params/params.json:prompt_subtraction_samples"] = canonical_digest(
            sorted(set(parameters["prompt_subtraction_samples"]))
        )
        digests["data_driven_application_regions"] = canonical_digest(
            {"nonprompt": sorted(NONPROMPT_APPLICATION_REGIONS),
             "flips": [FLIPS_APPLICATION_REGION]}
        )
        for name, symbols in DATA_DRIVEN_SCIENTIFIC_SYMBOLS.items():
            path = REPOSITORY_ROOT.parent / name if name.startswith("topcoffea/") else REPOSITORY_ROOT / name
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            selected = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbol_names = (node.name,)
                elif isinstance(node, ast.Assign):
                    symbol_names = tuple(target.id for target in node.targets if isinstance(target, ast.Name))
                else:
                    continue
                for symbol in symbol_names:
                    if symbol in symbols:
                        if symbol in selected:
                            fail(f"duplicate scientific dependency symbol: {name}:{symbol}")
                        selected[symbol] = ast.get_source_segment(source, node)
            if set(selected) != set(symbols) or any(value is None for value in selected.values()):
                fail(f"scientific dependency symbol is missing: {name}")
            digests[name] = canonical_digest(selected)
    return digests


def scientific_identity(contract, environment_file_sha256, _topcoffea_source_fingerprint, coverage):
    campaign_contract = {key: value for key, value in contract.items() if key != "stages"}
    source_payload = {"frozen_environment_tarball_sha256": environment_file_sha256}
    data_driven_payload = {"scientific_files": scientific_files(contract, "data_driven")}
    sample_files = {
        name: digest for name, digest in scientific_files(contract, "source").items()
        if name.startswith("input_samples/")
    }
    payload = {
        "campaign_science_contract": campaign_contract,
        "sample_universe_sha256": canonical_digest(sample_files),
        "approved_scientific_coverage_sha256": canonical_digest(list(coverage)),
    }
    return {
        "scientific_contract_payload": payload,
        "scientific_contract_sha256": canonical_digest(payload),
        "source_stage_scientific_contract": source_payload,
        "source_stage_scientific_contract_sha256": canonical_digest(source_payload),
        "data_driven_stage_scientific_contract": data_driven_payload,
        "data_driven_stage_scientific_contract_sha256": canonical_digest(data_driven_payload),
        "scientific_coverage": list(coverage),
        "scientific_coverage_sha256": canonical_digest(list(coverage)),
    }


def data_driven_science_matches(recorded, current):
    recorded_files = recorded["data_driven_stage_scientific_contract"]["scientific_files"]
    current_files = current["data_driven_stage_scientific_contract"]["scientific_files"]
    wrapper = "analysis/topeft_run2/run_data_driven.py"
    recorded_files = dict(recorded_files)
    recorded_wrapper = recorded_files.get(wrapper)
    if recorded_wrapper in LEGACY_DATA_DRIVEN_WRAPPER_SYMBOLS:
        recorded_files[wrapper] = LEGACY_DATA_DRIVEN_WRAPPER_SYMBOLS[recorded_wrapper]
    elif recorded_wrapper == digest_file(REPOSITORY_ROOT / wrapper):
        # A legacy full-file binding remains valid when the exact file is still
        # present; no historical receipt or state identity is rewritten.
        recorded_files[wrapper] = current_files[wrapper]
    return recorded_files == current_files


def artifact_receipt(path, *, stored_receipt=None, verify_payload_sha256=True):
    path = Path(path)
    sidecar_path = Path(f"{path}.metadata.json")
    if path.is_symlink() or sidecar_path.is_symlink() or not path.is_file() or not sidecar_path.is_file() or path.stat().st_size == 0:
        fail(f"required artifact or sidecar missing: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact = sidecar.get("artifact", {})
    if not isinstance(artifact, dict):
        fail(f"invalid artifact sidecar: {path}")
    if not verify_payload_sha256 and not isinstance(stored_receipt, dict):
        fail(f"stored receipt required for fast artifact readback: {path}")
    recorded_sha256 = stored_receipt.get("sha256") if stored_receipt is not None else None
    if stored_receipt is not None and (not isinstance(recorded_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", recorded_sha256) is None):
        fail(f"invalid stored artifact SHA-256: {path}")
    observed = {
        "path": str(path), "sha256": digest_file(path) if verify_payload_sha256 else recorded_sha256,
        "size_bytes": path.stat().st_size, "sidecar_sha256": digest_file(sidecar_path),
    }
    if artifact.get("pkl_basename") != path.name or artifact.get("pkl_sha256") != observed["sha256"] or artifact.get("pkl_size_bytes") != observed["size_bytes"]:
        fail(f"artifact sidecar identity mismatch: {path}")
    lineage = sidecar.get("lineage", {})
    if not sidecar.get("synthetic") and not (isinstance(lineage, dict) and isinstance(lineage.get("inputs"), list)):
        fail(f"artifact sidecar lacks lineage: {path}")
    if stored_receipt is not None and stored_receipt != observed:
        fail(f"stored artifact receipt changed: {path}")
    return observed


def unit_spec(unit):
    return {key: unit[key] for key in (
        "id", "years", "category_groups", "histograms", "output_tag",
        "output_name", "outputs", "coverage", "plan_revision",
    )}


def new_unit(spec, timestamp):
    return {
        **spec,
        "stages": {
            "source": {"status": "pending", "attempts": [], "receipt": None},
            "data_driven": {"status": "blocked", "attempts": [], "receipt": None},
        },
        "transitions": [{"at_utc": timestamp, "stage": "campaign", "status": "planned", "detail": "campaign_initialized"}],
        "native_work_queue_logs": {},
    }


def overall_status(unit):
    source = unit["stages"]["source"]["status"]
    downstream = unit["stages"]["data_driven"]["status"]
    states = {
        ("pending", "blocked"): "planned",
        ("running", "blocked"): "source_running",
        ("failed", "blocked"): "source_failed",
        ("success", "pending"): "source_ready",
        ("success", "running"): "data_driven_running",
        ("success", "failed"): "data_driven_failed",
        ("success", "success"): "success",
    }
    result = states.get((source, downstream))
    if result is None:
        fail(f"contradictory stage statuses for {unit.get('id')}")
    return result


def classify(state):
    statuses = [overall_status(unit) for unit in state["units"]]
    if any(status.endswith("_running") for status in statuses):
        return "blocked_ambiguous"
    if any(status in {"planned", "source_ready"} for status in statuses):
        return "blocked_incomplete"
    if any(status.endswith("_failed") for status in statuses):
        return "complete_with_known_failures"
    if statuses and all(status == "success" for status in statuses):
        return "success"
    return "blocked_state_contradiction"


def new_state(specs, approved_coverage, identity, contract, provenance):
    profile = identity["production_profile"]
    verify_partition(specs, profile, approved_coverage)
    revisions = {spec["plan_revision"] for spec in specs}
    if len(revisions) != 1:
        fail("initial plan requires one revision")
    revision = revisions.pop()
    units = [{key: spec[key] for key in ("id", "years", "category_groups", "histograms")} for spec in specs]
    if not re.fullmatch(r"r1_[0-9a-f]{12}", revision):
        fail("initial revision identity is invalid")
    timestamp = now_utc()
    state = {
        "schema_version": SCHEMA_VERSION,
        "campaign_identity": identity,
        "profile_contract": contract,
        "scientific_identity": scientific_identity(
            contract, provenance["environment_file_sha256"],
            provenance["topcoffea_relevant_source_fingerprint"], approved_coverage,
        ),
        "plan": {
            "active_revision": revision,
            "plan_partition_sha256": plan_partition_sha256(units),
            "revisions": [{
                "id": revision, "partition_sha256": plan_partition_sha256(units),
                "snapshot_sha256": canonical_digest(specs), "units": specs,
            }],
        },
        "units": [new_unit(spec, timestamp) for spec in specs],
        "provenance": {"initial": provenance, "last_resolved": provenance},
        "history": [{"at_utc": timestamp, "action": "campaign_initialized", "revision": revision}],
        "summary": {},
        "created_at_utc": timestamp,
        "updated_at_utc": timestamp,
    }
    validate_state(state, verify_artifacts=False)
    return state


def validate_receipt(state, unit, stage, *, verify_payload_sha256=False):
    record = unit["stages"][stage]
    receipt = record.get("receipt")
    if record["status"] != "success" or not isinstance(receipt, dict):
        fail(f"{unit['id']} lacks successful {stage} receipt")
    bound = receipt.get("binding")
    expected = binding(state, unit, stage)
    if bound != expected:
        fail(f"{unit['id']} {stage} receipt binding changed")
    path = effective_output_path(unit, stage)
    artifact = artifact_receipt(
        path, stored_receipt=receipt.get("artifact"),
        verify_payload_sha256=verify_payload_sha256,
    )
    if stage == "data_driven":
        source = unit["stages"]["source"]["receipt"]["artifact"]
        sidecar = json.loads(Path(f"{path}.metadata.json").read_text(encoding="utf-8"))
        if not sidecar.get("synthetic") and not any(
            isinstance(item, dict) and item.get("pkl_sha256") == source["sha256"]
            for item in sidecar["lineage"]["inputs"]
        ):
            fail(f"{unit['id']} data-driven lineage does not bind source")
    return artifact


def validate_state(state, *, verify_artifacts=True):
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        fail("normal maintained resume requires schema 9")
    identity = state.get("campaign_identity")
    if not isinstance(identity, dict) or set(identity) != {"production_profile", "campaign_tag", "output_dir"}:
        fail("campaign identity is incomplete or contradictory")
    profile = identity["production_profile"]
    if state.get("profile_contract") != profile_contract(profile):
        fail("profile contract contradicts public profile")
    science = state.get("scientific_identity")
    if not isinstance(science, dict) or set(science) != {
        "scientific_contract_payload", "scientific_contract_sha256",
        "scientific_coverage", "scientific_coverage_sha256",
        "source_stage_scientific_contract", "source_stage_scientific_contract_sha256",
        "data_driven_stage_scientific_contract", "data_driven_stage_scientific_contract_sha256",
    }:
        fail("scientific identity is incomplete")
    if science["scientific_contract_sha256"] != canonical_digest(science["scientific_contract_payload"]):
        fail("scientific contract digest changed")
    for stage in ("source", "data_driven"):
        if science[f"{stage}_stage_scientific_contract_sha256"] != canonical_digest(science[f"{stage}_stage_scientific_contract"]):
            fail(f"{stage} stage scientific contract digest changed")
    frozen_sha = science["source_stage_scientific_contract"].get("frozen_environment_tarball_sha256")
    if not isinstance(frozen_sha, str) or re.fullmatch(r"[0-9a-f]{64}", frozen_sha) is None:
        fail("source stage lacks a frozen environment tarball identity")
    provenance = state.get("provenance", {})
    if (not isinstance(provenance.get("initial"), dict)
            or provenance["initial"].get("environment_file_sha256") != frozen_sha
            or not isinstance(provenance.get("last_resolved"), dict)):
        fail("NEW_CAMPAIGN_REQUIRED: campaign provenance differs from its frozen tarball")
    validate_environment_migrations(state)
    supported_environment_modes = {"current_resolved", "explicit_snapshot"}
    if any(
        provenance[key].get("environment_mode") not in supported_environment_modes
        for key in ("initial", "last_resolved")
    ):
        fail("campaign provenance has an unsupported environment mode")
    if science["scientific_coverage_sha256"] != canonical_digest(science["scientific_coverage"]):
        fail("scientific coverage digest changed")
    if (not isinstance(science["scientific_contract_payload"], dict)
            or science["scientific_contract_payload"].get("approved_scientific_coverage_sha256")
            != science["scientific_coverage_sha256"]):
        fail("scientific contract coverage differs from recorded coverage")
    if science["scientific_contract_payload"].get("campaign_science_contract") != {
            key: value for key, value in state["profile_contract"].items() if key != "stages"}:
        fail("scientific contract profile differs from recorded profile")
    plan = state.get("plan")
    units = state.get("units")
    if not isinstance(plan, dict) or not isinstance(units, list) or not units:
        fail("campaign plan is incomplete")
    revisions = plan.get("revisions")
    if not isinstance(revisions, list) or not revisions or len({item.get("id") for item in revisions}) != len(revisions):
        fail("plan revisions are inconsistent")
    for number, revision in enumerate(revisions, 1):
        snapshot = revision.get("units")
        if not isinstance(snapshot, list) or not snapshot or not re.fullmatch(rf"r{number}_[0-9a-f]{{12}}", revision["id"]):
            fail("plan revision sequence is inconsistent")
        if revision.get("snapshot_sha256") != canonical_digest(snapshot):
            fail("immutable plan revision snapshot changed")
        snapshot_units = [
            {key: spec[key] for key in ("id", "years", "category_groups", "histograms")}
            for spec in snapshot
        ]
        if revision.get("partition_sha256") != plan_partition_sha256(snapshot_units):
            fail("plan revision partition digest changed")
        verify_recorded_partition(snapshot, profile, science["scientific_coverage"])
        validate_output_namespace(snapshot)
    if plan.get("active_revision") != revisions[-1]["id"]:
        fail("active plan revision is inconsistent")
    active_specs = revisions[-1].get("units")
    if not isinstance(active_specs, list) or [unit_spec(unit) for unit in units] != active_specs:
        fail("active units differ from immutable plan snapshot")
    if len({unit["id"] for unit in units}) != len(units):
        fail("active unit ids are not unique")
    plan_units = [{key: unit[key] for key in ("id", "years", "category_groups", "histograms")} for unit in units]
    if plan.get("plan_partition_sha256") != plan_partition_sha256(plan_units):
        fail("active plan partition digest changed")
    verify_recorded_partition(units, profile, science["scientific_coverage"])
    validate_output_namespace(units)
    revision_ids = {item["id"] for item in revisions}
    for unit in units:
        if unit["plan_revision"] not in revision_ids:
            fail(f"{unit['id']} coverage or revision changed")
        if not isinstance(unit.get("transitions"), list) or not unit["transitions"]:
            fail(f"{unit['id']} transition history missing")
        stages = unit.get("stages")
        if not isinstance(stages, dict) or set(stages) != set(state["profile_contract"]["stages"]):
            fail(f"{unit['id']} stage roles contradict profile contract")
        overall_status(unit)
        for stage in state["profile_contract"]["stages"]:
            record = stages[stage]
            attempts = record.get("attempts")
            if not isinstance(attempts, list) or any(
                item.get("number") != index or item.get("stage") != stage
                for index, item in enumerate(attempts, 1)
            ):
                fail(f"{unit['id']} {stage} attempt history is inconsistent")
            if attempts and isinstance(attempts[0].get("provenance"), dict):
                first_attempt_mode = attempts[0]["provenance"].get("environment_mode")
                expected_mode = provenance["initial"]["environment_mode"]
                for migration in state.get("environment_migrations", []):
                    boundary = migration["effective_after_attempts"].get(unit["id"], {}).get(stage, {}).get("count", 0)
                    if boundary == 0:
                        expected_mode = migration["to"]["environment_mode"]
                if first_attempt_mode is not None and first_attempt_mode != expected_mode:
                    fail(f"{unit['id']} {stage} initial environment mode contradicts first attempt provenance")
            for attempt in attempts:
                effective_output_path(unit, stage, attempt)
            if record["status"] in {"running", "failed", "success"} and not attempts:
                fail(f"{unit['id']} {stage} lacks an attempt")
            if record["status"] == "running" and attempts[-1]["result"] != "running":
                fail(f"{unit['id']} running attempt was not recorded")
            if record["status"] == "success":
                if attempts[-1]["result"] not in {"success", "reconciled_success", "imported_success"}:
                    fail(f"{unit['id']} successful attempt is inconsistent")
                if (record.get("receipt") or {}).get("artifact", {}).get("path") != effective_output_path(unit, stage):
                    fail(f"{unit['id']} {stage} receipt path contradicts recorded attempt")
                if verify_artifacts:
                    validate_receipt(state, unit, stage)
            elif record.get("receipt") is not None:
                fail(f"{unit['id']} has a premature {stage} receipt")
    return state


def load_state(path, *, verify_artifacts=True):
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_state(state, verify_artifacts=verify_artifacts)


def desired_state_inputs(arguments, *, approved_coverage=None):
    if len(arguments) < 17:
        fail("campaign initializer lacks required profile and provenance fields")
    (
        rendered_plan, profile, default_plan, tag, output_dir, commit,
        env_file, env_sha, env_fingerprint, env_mode, topcoffea_commit,
        topcoffea_source, ttgamma, do_systs, do_np, region, np_mode,
    ) = arguments[:17]
    if np_mode != "separate":
        fail("new maintained campaigns require separate data-driven realization")
    contract = profile_contract(
        profile, region=region, do_systs=do_systs == "true",
        do_np=do_np == "true", ttgamma_policy=ttgamma,
    )
    if env_mode not in {"current_resolved", "explicit_snapshot"} or not re.fullmatch(r"[0-9a-f]{64}", env_sha):
        fail("invalid resolved scientific environment")
    identity = {"production_profile": profile, "campaign_tag": tag, "output_dir": output_dir}
    provenance = {
        "topeft_git_commit": commit,
        "environment_file": env_file,
        "environment_file_sha256": env_sha,
        "environment_fingerprint": env_fingerprint,
        "environment_mode": env_mode,
        "topcoffea_git_commit": topcoffea_commit,
        "topcoffea_relevant_source_fingerprint": topcoffea_source,
    }
    approved = (partition_coverage(default_units(profile, default_plan), profile)
                if approved_coverage is None else approved_coverage)
    return rendered_plan, identity, contract, provenance, approved


def validate_resume(state, identity, contract, provenance, approved, *, allow_environment_migration=False):
    validate_state(state)
    if state["campaign_identity"] != identity or state["profile_contract"] != contract:
        fail("campaign identity or profile contract differs")
    current = scientific_identity(
        contract, provenance["environment_file_sha256"],
        provenance["topcoffea_relevant_source_fingerprint"], approved,
    )
    recorded = state["scientific_identity"]
    if current["scientific_coverage_sha256"] != recorded["scientific_coverage_sha256"]:
        fail("scientific_coverage_incompatible")
    if current["scientific_contract_sha256"] != recorded["scientific_contract_sha256"]:
        fail("scientific_contract_incompatible")
    active = state["provenance"]["last_resolved"]
    migrating = provenance["environment_file_sha256"] != active["environment_file_sha256"]
    if migrating:
        if not allow_environment_migration:
            fail("NEW_CAMPAIGN_REQUIRED: frozen environment tarball differs")
        if provenance["environment_mode"] != "explicit_snapshot":
            fail("environment migration requires an explicit snapshot")
        if any(overall_status(unit).endswith("_running") for unit in state["units"]):
            fail("environment migration requires an unambiguous attempt boundary")
        for unit in state["units"]:
            for stage in contract["stages"]:
                pending = pending_artifact_path(state, unit, stage)
                if pending is not None and os.path.lexists(pending.parent):
                    fail("environment migration has unresolved pending evidence")
        from topcoffea.modules.remote_environment import validate_environment_archive
        archive = validate_environment_archive(provenance["environment_file"])
        if (archive["status"] != "valid"
                or archive["archive_sha256"] != provenance["environment_file_sha256"]
                or archive["environment_fingerprint"] != provenance["environment_fingerprint"]):
            fail("environment migration archive identity is invalid")
        if not data_driven_science_matches(recorded, current):
            fail("data_driven_scientific_contract_incompatible")
    if (not migrating and not state.get("environment_migrations")
            and state["provenance"]["initial"]["environment_mode"] == "current_resolved"
            and provenance["topcoffea_relevant_source_fingerprint"]
            != state["provenance"]["initial"]["topcoffea_relevant_source_fingerprint"]):
        fail("NEW_CAMPAIGN_REQUIRED: frozen topcoffea source fingerprint differs")
    if (not migrating and state.get("environment_migrations")
            and (provenance["environment_fingerprint"] != active["environment_fingerprint"]
                 or provenance["environment_mode"] != "explicit_snapshot")):
        fail("active migrated environment identity differs")
    if (not migrating and not state.get("environment_migrations")
            and current["source_stage_scientific_contract_sha256"] != recorded["source_stage_scientific_contract_sha256"]):
        fail("source_scientific_contract_incompatible")
    return state


def validate_environment_migrations(state):
    """Validate append-only environment epochs without changing historical science."""
    migrations = state.get("environment_migrations", [])
    if not isinstance(migrations, list):
        fail("environment migration history is invalid")
    previous = state["provenance"]["initial"]
    previous_frontier = {}
    for migration in migrations:
        payload = {key: value for key, value in migration.items() if key != "migration_sha256"}
        if migration.get("migration_sha256") != canonical_digest(payload):
            fail("environment migration record changed")
        old, new = migration["from"], migration["to"]
        if (old["environment_file_sha256"] != previous["environment_file_sha256"]
                or new["environment_file_sha256"] == old["environment_file_sha256"]
                or new["environment_mode"] != "explicit_snapshot"):
            fail("environment migration chain is inconsistent")
        for provenance in (old, new):
            if (not Path(provenance["environment_file"]).is_absolute()
                    or any(re.fullmatch(r"[0-9a-f]{64}", provenance[key] or "") is None
                           for key in ("environment_file_sha256", "environment_fingerprint"))):
                fail("environment migration identity is incomplete")
        frontier = migration["effective_after_attempts"]
        if not isinstance(frontier, dict) or not previous_frontier.keys() <= frontier.keys():
            fail("environment migration frontier is inconsistent")
        for unit_id, stages in frontier.items():
            if set(stages) != set(state["profile_contract"]["stages"]):
                fail("environment migration frontier stages differ")
            for stage, boundary in stages.items():
                count = boundary["count"]
                prior = previous_frontier.get(unit_id, {}).get(stage, {}).get("count", 0)
                if type(count) is not int or count < prior:
                    fail("environment migration frontier moved backwards")
                unit = next((u for u in state["units"] if u["id"] == unit_id), None)
                if unit is not None:
                    attempts = unit["stages"][stage]["attempts"]
                    if count > len(attempts) or canonical_digest(attempts[:count]) != boundary["sha256"]:
                        fail("historical attempts changed across environment migration")
        previous, previous_frontier = new, frontier
    active = state["provenance"]["last_resolved"]
    if active["environment_file_sha256"] != previous["environment_file_sha256"]:
        fail("NEW_CAMPAIGN_REQUIRED: campaign provenance differs from its frozen tarball")
    if migrations and (active["environment_fingerprint"] != previous["environment_fingerprint"]
                       or active["environment_mode"] != "explicit_snapshot"):
        fail("active migrated environment identity differs")
    if migrations:
        for unit in state["units"]:
            for stage, record in unit["stages"].items():
                for attempt in record["attempts"]:
                    expected = state["provenance"]["initial"]
                    migrated_attempt = False
                    for migration in migrations:
                        count = migration["effective_after_attempts"].get(unit["id"], {}).get(stage, {}).get("count", 0)
                        if attempt["number"] > count:
                            expected = migration["to"]
                            migrated_attempt = True
                    if (not migrated_attempt and "schema8_import" in state["provenance"]
                            and attempt["provenance"].get("legacy_schema8_scientific_contract_sha256")):
                        # The native importer verified the frozen archive and
                        # preserved the older receipt's own binding contract.
                        continue
                    if attempt["provenance"]["environment_file_sha256"] != expected["environment_file_sha256"]:
                        fail("attempt environment contradicts migration frontier")
                    if migrated_attempt and any(attempt["provenance"][key] != expected[key] for key in (
                            "environment_fingerprint", "environment_mode")):
                        fail("attempt environment identity contradicts migration frontier")


def attempt_identity(attempt):
    return canonical_digest({key: attempt[key] for key in (
        "number", "stage", "command_argv", "chunksize", "resources", "provenance",
    )})


def binding(state, unit, stage):
    attempt = unit["stages"][stage]["attempts"][-1]
    historical_contract = attempt["provenance"].get("legacy_schema8_scientific_contract_sha256")
    result = {
        "scientific_coverage_sha256": state["scientific_identity"]["scientific_coverage_sha256"],
        "plan_revision": attempt["provenance"].get("legacy_schema8_plan_revision", unit["plan_revision"]),
        "unit_coverage_sha256": canonical_digest(unit["coverage"]),
        "attempt_sha256": attempt_identity(attempt),
    }
    if historical_contract:
        result["legacy_schema8_scientific_contract_sha256"] = historical_contract
    else:
        result["scientific_contract_sha256"] = state["scientific_identity"]["scientific_contract_sha256"]
        result["stage_scientific_contract_sha256"] = state["scientific_identity"][f"{stage}_stage_scientific_contract_sha256"]
        if stage == "source" and state.get("environment_migrations"):
            result["stage_scientific_contract_sha256"] = canonical_digest({
                "frozen_environment_tarball_sha256": attempt["provenance"]["environment_file_sha256"],
            })
    if stage == "data_driven":
        result["source_artifact_sha256"] = unit["stages"]["source"]["receipt"]["artifact"]["sha256"]
    return result


def output_exists(path):
    return os.path.lexists(path) or os.path.lexists(f"{path}.metadata.json")


def effective_output_path(unit, stage, attempt=None):
    """Validate recorded argv against the plan-owned durable output path."""
    if attempt is None:
        attempts = unit["stages"][stage]["attempts"]
        attempt = attempts[-1] if attempts else None
    fallback = unit["outputs"][stage]
    if attempt is None:
        return fallback
    command = attempt.get("command_argv")
    if not isinstance(command, list):
        fail("attempt command is not an argv list")
    option = "-p" if stage == "source" else "--output-pkl"
    if option not in command:
        return fallback
    values = command_values(command, option)
    if len(values) != 1 or not values[0]:
        fail(f"attempt {option} is not one path")
    path = str(Path(values[0]) / Path(fallback).name) if stage == "source" else values[0]
    if path != fallback:
        fail("attempt durable path contradicts unit.outputs")
    return fallback


def output_safety_from_command(command):
    """Read the recorded executor policy without owning filesystem admission."""
    from topeft.modules.histogram_artifact import (
        histogram_artifact_error, output_capacity_thresholds,
    )

    names = ("--pkl-min-free-bytes", "--pkl-min-free-inodes",
             "--pkl-safety-reserve-bytes", "--pkl-safety-reserve-inodes")
    present = [name in command for name in names]
    if not any(present):
        return None
    if not all(present):
        fail("persistent output requires all four capacity options")
    values = []
    for name in names:
        option_values = command_values(command, name)
        if len(option_values) != 1 or not option_values[0].isdigit():
            fail(f"invalid persistent output option {name}")
        values.append(int(option_values[0]))
    try:
        return output_capacity_thresholds(*values)
    except histogram_artifact_error as error:
        fail(str(error))


def pending_artifact_id(state, unit, stage, attempt=None):
    if attempt is None:
        attempts = unit["stages"][stage]["attempts"]
        attempt = attempts[-1] if attempts else None
    if attempt is None:
        return None
    effective_output_path(unit, stage, attempt)
    if type(attempt["number"]) is not int or attempt["number"] < 1:
        fail("invalid pending artifact attempt number")
    return canonical_digest({
        "campaign_identity": state["campaign_identity"],
        "unit_id": unit["id"],
        "logical_output": unit["outputs"][stage],
        "attempt_sha256": attempt_identity(attempt),
    })


def pending_artifact_path(state, unit, stage, attempt=None):
    pending_id = pending_artifact_id(state, unit, stage, attempt)
    if pending_id is None:
        return None
    path = Path(effective_output_path(unit, stage, attempt))
    return path.parent / f".topeft-pending-{pending_id}" / path.name


def publish_pending_artifact(state, unit, stage):
    """Request publication with the recorded attempt identity and validate its semantics."""
    from topeft.modules.histogram_artifact import (
        histogram_artifact_error, recover_pending_artifact,
    )

    pending = pending_artifact_path(state, unit, stage)
    if pending is None:
        return False
    final = Path(effective_output_path(unit, stage))
    attempt = unit["stages"][stage]["attempts"][-1]
    try:
        return recover_pending_artifact(
            pending, final,
            validate_sidecar=lambda sidecar, path: validate_recovery_sidecar(
                state, unit, stage, sidecar, artifact_path=path),
            thresholds=output_safety_from_command(attempt["command_argv"]),
        )
    except histogram_artifact_error as error:
        fail(str(error))


def cleanup_published_pending(state, unit, stage):
    from topeft.modules.histogram_artifact import cleanup_pending_artifact

    pending = pending_artifact_path(state, unit, stage)
    if pending is not None:
        cleanup_pending_artifact(pending, effective_output_path(unit, stage))


def data_driven_command_artifact_kind(unit, command):
    if command.count("--output-pkl") != 1:
        fail("data-driven command requires one output path")
    output_values = command_values(command, "--output-pkl")
    if len(output_values) != 1:
        fail("data-driven command requires one output path")
    expected = [
        "python", "./run_data_driven.py",
        "--input-pkl", unit["stages"]["source"]["receipt"]["artifact"]["path"],
        "--output-pkl", output_values[0],
    ]
    # The ordinary consumer accepts this no-value observability switch before
    # the output-safety suffix. Exact comparison still rejects duplicates,
    # unknown options and switches selecting a different scientific product.
    if "--phase-telemetry" in command:
        expected.append("--phase-telemetry")
    policy_suffix = ["--pkl-no-clobber"] if "--pkl-no-clobber" in command else []
    for option in ("--pkl-min-free-bytes", "--pkl-min-free-inodes",
                   "--pkl-safety-reserve-bytes", "--pkl-safety-reserve-inodes"):
        if option in command:
            policy_suffix.extend((option, *command_values(command, option)))
    if command != expected + policy_suffix:
        fail("data-driven command contradicts the ordinary deferred nonprompt artifact contract")
    effective_output_path(unit, "data_driven", {"command_argv": command})
    return "nonprompt_output"


def validate_recovery_sidecar(state, unit, stage, sidecar, *, artifact_path=None):
    """Use fields emitted by the maintained artifact writers, not a campaign-only field."""
    from topeft.modules.histogram_artifact import read_histogram_sidecar
    from topeft.modules.data_driven_products import validate_requested_product_input

    attempt = unit["stages"][stage]["attempts"][-1]
    command = attempt["command_argv"]
    if not isinstance(command, list) or not command:
        fail("recovery_evidence_insufficient: recorded command is missing")
    if "synthetic" in sidecar:
        fail("recovery_evidence_insufficient: synthetic metadata is not producer evidence")
    if stage == "source":
        try:
            validated = read_histogram_sidecar(artifact_path or effective_output_path(unit, "source"))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            fail(f"recovery_evidence_insufficient: invalid processor sidecar: {error}")
        if validated != sidecar:
            fail("recovery_evidence_insufficient: processor sidecar is not canonical")
        for option, expected in (
            ("-y", unit["years"]),
            ("--category-groups", unit["category_groups"]),
            ("--hist-vars", unit["histograms"]),
            ("-t", [unit["output_tag"]]),
            ("-p", [str(Path(effective_output_path(unit, "source")).parent)]),
            ("--env-file", [attempt["provenance"].get("environment_file") or state["provenance"]["initial"]["environment_file"]]),
        ):
            if command_values(command, option) != expected:
                fail(f"recovery_evidence_insufficient: recorded source {option} differs")
        if output_exists(effective_output_path(unit, "data_driven")):
            fail("recovery_evidence_insufficient: downstream artifact precedes source receipt")
        if sidecar.get("artifact", {}).get("artifact_kind") != "processor_output" or sidecar.get("lineage", {}).get("inputs") != []:
            fail("recovery_evidence_insufficient: source producer lineage differs")
        artifact = sidecar["artifact"]
        if artifact["merged"] is not False:
            fail("recovery_evidence_insufficient: source artifact is unexpectedly merged")
        applicability = sidecar.get("histogram_applicability", {})
        expected_mode = "all" if "--all-analysis" in command else "offz" if "--offZ-3l-split" in command else "tau" if "--tau-h-analysis" in command else "fwd" if "--fwd-analysis" in command else "default"
        if (
            applicability.get("analysis_mode") != expected_mode
            or list(applicability.get("families", {})) != unit["histograms"]
        ):
            fail("recovery_evidence_insufficient: source histogram applicability differs")
    else:
        try:
            validated = read_histogram_sidecar(artifact_path or effective_output_path(unit, "data_driven"))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            fail(f"recovery_evidence_insufficient: invalid data-driven sidecar: {error}")
        if validated != sidecar:
            fail("recovery_evidence_insufficient: data-driven sidecar is not canonical")
        expected_kind = data_driven_command_artifact_kind(unit, command)
        if sidecar.get("artifact", {}).get("artifact_kind") != expected_kind:
            fail("recovery_evidence_insufficient: downstream artifact kind differs")
        source = unit["stages"]["source"]["receipt"]
        expected_lineage = [{
            "pkl_basename": Path(effective_output_path(unit, "source")).name,
            "artifact_kind": "processor_output",
            "pkl_sha256": source["artifact"]["sha256"],
        }] if source is not None else None
        if sidecar.get("lineage", {}).get("inputs") != expected_lineage:
            fail("recovery_evidence_insufficient: downstream lineage mismatch")
        if sidecar["artifact"]["merged"] is not False:
            fail("recovery_evidence_insufficient: downstream artifact is unexpectedly merged")
        source_sidecar = read_histogram_sidecar(effective_output_path(unit, "source"))
        try:
            effective_source = validate_requested_product_input(
                source_sidecar, artifact_kind=expected_kind,
            )["effective_sidecar"]
        except (ValueError, KeyError, TypeError) as error:
            fail(f"recovery_evidence_insufficient: source data-driven contract is invalid: {error}")
        for field in (
            "sumw2_storage_provenance", "production_sample_contract",
            "requested_data_driven_products", "resolved_data_driven_contract",
            "histogram_applicability",
        ):
            if field not in sidecar or sidecar[field] != effective_source.get(field):
                fail(f"recovery_evidence_insufficient: downstream {field} differs from source")
        for product in ("nonprompt", "flips"):
            if sidecar["requested_data_driven_products"]["products"][product]["enabled"] is not True:
                fail(f"recovery_evidence_insufficient: downstream {product} product is disabled")
    sumw2 = sidecar.get("sumw2_storage_provenance", {})
    expected_mode = "production" if state["profile_contract"]["systematics"] else "full_diagnostics"
    if sumw2.get("requested_mode") != expected_mode or sumw2.get("resolved_mode") != expected_mode or sumw2.get("runtime_histogram_families") != unit["histograms"]:
        fail("recovery_evidence_insufficient: sidecar histogram or sumw2 policy differs")
    sample = sidecar.get("production_sample_contract", {})
    cfg_items = sample.get("cfg_identities", []) if isinstance(sample, dict) else []
    if not isinstance(cfg_items, list) or not cfg_items:
        fail("recovery_evidence_insufficient: sample contract is absent")
    science_files = scientific_files(state["profile_contract"], "source")
    if "schema8_import" in state["provenance"] and attempt["provenance"].get("legacy_schema8_scientific_contract_sha256"):
        science_files = state["provenance"]["schema8_import"]["historical_scientific_files"]
    expected_cfg = {
        Path(name).name: digest for name, digest in science_files.items()
        if name.startswith("input_samples/cfgs/")
    }
    observed_cfg = {}
    for item in cfg_items:
        if not isinstance(item, dict):
            fail("recovery_evidence_insufficient: invalid sample cfg identity")
        basename = item.get("basename")
        if item.get("input_kind") != "cfg" or basename in observed_cfg or expected_cfg.get(basename) != item.get("content_sha256"):
            fail("recovery_evidence_insufficient: sample cfg differs from recorded science")
        observed_cfg[basename] = item["content_sha256"]
    if observed_cfg != expected_cfg:
        fail("recovery_evidence_insufficient: sample cfg set is incomplete or extra")
    if stage == "source":
        for product in ("nonprompt", "flips"):
            if (
                sidecar.get("requested_data_driven_products", {}).get("products", {}).get(product, {}).get("enabled") is not True
                or sidecar.get("resolved_data_driven_contract", {}).get("products", {}).get(product, {}).get("enabled") is not True
            ):
                fail(f"recovery_evidence_insufficient: producer {product} contract differs")
    if sidecar.get("metadata_schema_version") != 2:
        fail("recovery_evidence_insufficient: maintained processor metadata is incomplete")


def command_values(command, name):
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1:
        fail(f"command requires exactly one {name}")
    values = []
    for value in command[positions[0] + 1:]:
        if value.startswith("-"):
            break
        values.append(value)
    return values


def validate_source_command(state, unit, command):
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        fail("source command must be an argv list")
    contract = state["profile_contract"]
    if command.count("--do-systs") != int(contract["systematics"]):
        fail("source systematics switch contradicts science")
    if command.count("--do-np") != 1 or command.count("--defer-np") != 1:
        fail("source data-driven switch contradicts science")
    if command.count("--record-raw-count") != int(contract["raw_counts"]):
        fail("source raw-count switch contradicts science")
    if command.count("--sr") != int(contract["region"] == "SR") or command.count("--cr") != int(contract["region"] == "CR"):
        fail("source region switch contradicts science")
    if command.count("--all-analysis") != 1 or command_values(command, "-x") != ["work_queue"]:
        fail("source analysis/executor mode contradicts profile")
    if command.count("--snapshot") != int(state["provenance"]["last_resolved"]["environment_mode"] == "explicit_snapshot"):
        fail("source environment realization contradicts resolved attempt")
    for option, expected in (
        ("-y", unit["years"]), ("--category-groups", unit["category_groups"]),
        ("--hist-vars", unit["histograms"]), ("-t", [unit["output_tag"]]),
        ("-p", [state["campaign_identity"]["output_dir"]]),
        ("--env-file", [state["provenance"]["last_resolved"]["environment_file"]]),
        ("--ttgamma-sample-role-policy", [contract["ttgamma_sample_role_policy"]]),
    ):
        if option == "-p":
            expected = [str(Path(unit["outputs"]["source"]).parent)]
        observed = command_values(command, option)
        if (option == "-p" and observed not in ([value] for value in expected)) or (option != "-p" and observed != expected):
            fail(f"source command {option} contradicts unit or science")
    chunksize_values = command_values(command, "-s")
    if len(chunksize_values) != 1 or not chunksize_values[0].isdigit() or int(chunksize_values[0]) <= 0:
        fail("chunksize must be a positive integer")
    return int(chunksize_values[0])


def record_transition(state, unit, stage, status, detail, exit_code=None):
    timestamp = now_utc()
    unit["transitions"].append({
        "at_utc": timestamp, "stage": stage, "status": status,
        "exit_code": exit_code, "detail": detail,
    })
    state["history"].append({
        "at_utc": timestamp, "action": f"{stage}_{status}",
        "unit_id": unit["id"], "revision": unit["plan_revision"],
    })
    state["updated_at_utc"] = timestamp


def transition(state, unit_id, stage, status, exit_code, detail, extra=()):
    validate_state(state)
    matches = [unit for unit in state["units"] if unit["id"] == unit_id]
    if len(matches) != 1 or stage not in state["profile_contract"]["stages"]:
        fail("unknown unit or stage")
    unit = matches[0]
    record = unit["stages"][stage]
    parsed_exit = None if exit_code == "none" else int(exit_code)
    if status == "running":
        previous_pending = pending_artifact_path(state, unit, stage)
        if previous_pending is not None and os.path.lexists(previous_pending.parent):
            fail("unresolved pending evidence prevents a new attempt")
        if stage == "source":
            if record["status"] not in {"pending", "failed"}:
                fail("source is not eligible for an attempt")
            command = list(extra[1:] if extra[:1] == ("--",) else extra)
            chunksize = validate_source_command(state, unit, command)
            path = effective_output_path(unit, stage, {"command_argv": command})
            if (output_exists(unit["outputs"]["source"])
                    or output_exists(unit["outputs"]["data_driven"])
                    or output_exists(path)):
                fail("unverified output collision prevents source retry")
        else:
            if record["status"] not in {"pending", "failed"} or unit["stages"]["source"]["status"] != "success":
                fail("data-driven stage lacks verified reusable source")
            current_science = scientific_identity(
                state["profile_contract"],
                state["provenance"]["last_resolved"]["environment_file_sha256"],
                state["provenance"]["last_resolved"]["topcoffea_relevant_source_fingerprint"],
                state["scientific_identity"]["scientific_coverage"],
            )
            if not data_driven_science_matches(state["scientific_identity"], current_science):
                fail("data_driven_scientific_contract_incompatible")
            command = list(extra)
            data_driven_command_artifact_kind(unit, command)
            path = effective_output_path(unit, stage, {"command_argv": command})
            if output_exists(unit["outputs"]["data_driven"]) or output_exists(path):
                fail("unverified data-driven output collision")
            chunksize = None
        from topeft.modules.histogram_artifact import (
            check_output_capacity, histogram_artifact_error,
        )
        try:
            check_output_capacity(path, output_safety_from_command(command))
        except histogram_artifact_error as error:
            fail(str(error))
        attempt = {
            "number": len(record["attempts"]) + 1, "stage": stage,
            "started_at_utc": now_utc(), "ended_at_utc": None,
            "command_argv": command, "chunksize": chunksize,
            "resources": {option: command_values(command, option) if option in command else None for option in (
                "--wq-treereduction", "--wq-processing-cores",
                "--wq-accumulating-memory-mb", "--wq-accumulating-cores",
                "--wq-staging-base",
            )},
            "provenance": dict(state["provenance"]["last_resolved"]),
            "result": "running", "exit_code": None,
        }
        record["attempts"].append(attempt)
        record["status"] = "running"
    elif status in {"ready", "success", "failed"}:
        if record["status"] != "running":
            fail("stage completion requires its recorded running attempt")
        attempt = record["attempts"][-1]
        if status == "failed":
            pending = pending_artifact_path(state, unit, stage)
            if pending is not None and (
                os.path.lexists(pending.parent)
                or output_exists(effective_output_path(unit, stage))
            ):
                fail("unresolved output evidence prevents failed transition")
        attempt["ended_at_utc"] = now_utc()
        attempt["exit_code"] = parsed_exit
        attempt["result"] = "success" if status in {"ready", "success"} else "failed"
        if status in {"ready", "success"}:
            if parsed_exit != 0:
                fail("successful stage requires zero exit code")
            artifact = artifact_receipt(effective_output_path(unit, stage))
            record["receipt"] = {"artifact": artifact, "binding": binding(state, unit, stage)}
            record["status"] = "success"
            if stage == "source":
                unit["stages"]["data_driven"]["status"] = "pending"
            else:
                validate_receipt(state, unit, stage)
        else:
            record["status"] = "failed"
            if stage == "source":
                unit["stages"]["data_driven"]["status"] = "blocked"
    else:
        fail("unsupported stage transition")
    record_transition(state, unit, stage, status, detail, parsed_exit)
    validate_state(state, verify_artifacts=status not in {"running", "failed"})
    return state


def reconcile_running(state, selected_unit=None):
    """Reconcile interrupted attempts only after the caller proves exclusive writer ownership."""
    validate_state(state)
    if selected_unit and not any(unit["id"] == selected_unit for unit in state["units"]):
        fail("unknown selected recovery unit")
    changed = False
    for unit in state["units"]:
        if selected_unit and unit["id"] != selected_unit:
            continue
        for stage in state["profile_contract"]["stages"]:
            record = unit["stages"][stage]
            if record["status"] != "running":
                continue
            path = effective_output_path(unit, stage)
            pending = pending_artifact_path(state, unit, stage)
            if pending is not None and os.path.lexists(pending.parent):
                publish_pending_artifact(state, unit, stage)
            if output_exists(path):
                try:
                    artifact = artifact_receipt(path)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    fail(f"recovery_evidence_insufficient for {unit['id']} {stage}: {error}")
                sidecar = json.loads(Path(f"{path}.metadata.json").read_text(encoding="utf-8"))
                validate_recovery_sidecar(state, unit, stage, sidecar)
                record["receipt"] = {"artifact": artifact, "binding": binding(state, unit, stage)}
                record["status"] = "success"
                record["attempts"][-1]["result"] = "reconciled_success"
                if stage == "source":
                    unit["stages"]["data_driven"]["status"] = "pending"
                result = "reconciled_success"
            else:
                record["status"] = "failed"
                record["attempts"][-1]["result"] = "interrupted"
                if stage == "source":
                    unit["stages"]["data_driven"]["status"] = "blocked"
                result = "interrupted"
            record["attempts"][-1]["ended_at_utc"] = now_utc()
            record_transition(state, unit, stage, result, "exclusive_writer_reconciliation")
            changed = True
    if changed:
        validate_state(state)
    return changed


def activate_plan(state, supplied_plan, expected_revision):
    validate_state(state)
    if state["plan"]["active_revision"] != expected_revision:
        fail("stale expected plan revision")
    if any(overall_status(unit).endswith("_running") for unit in state["units"]):
        fail("recovery_evidence_insufficient: running unit must be reconciled")
    profile = state["campaign_identity"]["production_profile"]
    supplied = supplied_units(supplied_plan, profile)
    approved = state["scientific_identity"]["scientific_coverage"]
    verify_partition(supplied, profile, approved)
    frozen = [
        unit for unit in state["units"]
        if unit["stages"]["source"]["status"] == "success"
    ]
    frozen_coverage = set()
    by_id = {unit["id"]: unit for unit in state["units"]}
    for unit in frozen:
        frozen_coverage.update(unit["coverage"])
        matching = [
            candidate for candidate in supplied
            if (candidate["id"] == unit["id"] or f"{profile.lower()}_{candidate['id']}" == unit["id"])
        ]
        if len(matching) != 1 or unit_coverage(matching[0], profile) != set(unit["coverage"]):
            fail(f"completed or reusable unit {unit['id']} coverage is frozen")
    pending = []
    for candidate in supplied:
        prefix = profile.lower()
        unit_id = candidate["id"] if candidate["id"].startswith(f"{prefix}_") else f"{prefix}_{candidate['id']}"
        if unit_id in {unit["id"] for unit in frozen}:
            continue
        if unit_coverage(candidate, profile) & frozen_coverage:
            fail("new pending unit overlaps frozen coverage")
        pending.append(candidate)
    if not pending and len(frozen) == len(state["units"]):
        fail("no pending coverage to repartition")
    next_number = len(state["plan"]["revisions"]) + 1
    revision = make_revision(next_number, supplied)
    new_specs = make_specs(
        pending, profile, state["campaign_identity"]["campaign_tag"],
        state["campaign_identity"]["output_dir"], revision,
    )
    historical_paths = {
        path for item in state["plan"]["revisions"] for spec in item["units"]
        for path in spec["outputs"].values()
    }
    if any(path in historical_paths for spec in new_specs for path in spec["outputs"].values()):
        fail("new plan reuses a historical output path")
    if any(output_exists(path) for spec in new_specs for path in spec["outputs"].values()):
        fail("new plan collides with an existing artifact")
    timestamp = now_utc()
    active = [*frozen, *(new_unit(spec, timestamp) for spec in new_specs)]
    validate_output_namespace(active)
    plan_units = [{key: unit[key] for key in ("id", "years", "category_groups", "histograms")} for unit in active]
    verify_partition(plan_units, profile, approved)
    state["units"] = active
    state["plan"]["active_revision"] = revision
    state["plan"]["plan_partition_sha256"] = plan_partition_sha256(plan_units)
    state["plan"]["revisions"].append({
        "id": revision, "partition_sha256": plan_partition_sha256(plan_units),
        "snapshot_sha256": canonical_digest([unit_spec(unit) for unit in active]),
        "units": [unit_spec(unit) for unit in active],
    })
    state["history"].append({
        "at_utc": timestamp, "action": "plan_revision_activated",
        "from_revision": expected_revision, "to_revision": revision,
    })
    state["updated_at_utc"] = timestamp
    validate_state(state)
    return revision


def assert_writer_lock(state_path):
    value = os.environ.get("CAMPAIGN_LOCK_FD")
    if value is None or not value.isdigit():
        fail("exclusive campaign writer lock evidence missing")
    descriptor = int(value)
    state_path = Path(state_path)
    lock_path = state_path.with_name(re.sub(r"_state(?:_v9)?\.json$", ".lock", state_path.name))
    if lock_path.is_symlink() or os.fstat(descriptor).st_ino != lock_path.stat().st_ino:
        fail("campaign writer lock descriptor does not match campaign")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fail("campaign writer lock is held by another process")


def read_campaign_identity(state, profile, tag, output_dir):
    identity = {"production_profile": profile, "campaign_tag": tag, "output_dir": output_dir}
    if state["campaign_identity"] != identity:
        fail("campaign identity differs from requested profile/tag/output")


def next_attempt(state, unit_id, stage):
    matches = [unit for unit in state["units"] if unit["id"] == unit_id]
    if len(matches) != 1 or stage not in state["profile_contract"]["stages"]:
        fail("unknown unit or stage")
    return len(matches[0]["stages"][stage]["attempts"]) + 1


def next_source_attempt(state, unit_id):
    matches = [unit for unit in state["units"] if unit["id"] == unit_id]
    if len(matches) != 1 or "source" not in state["profile_contract"]["stages"]:
        fail("unknown source unit or stage")
    source = matches[0]["stages"]["source"]
    if source["status"] not in {"pending", "failed"}:
        fail("source is not eligible for a new attempt")
    attempts = source["attempts"]
    if not isinstance(attempts, list) or any(
        not isinstance(item, dict) or item.get("number") != index or item.get("stage") != "source"
        for index, item in enumerate(attempts, 1)
    ):
        fail("source attempt history is inconsistent")
    return len(attempts) + 1


def record_attempt_provenance(state, provenance, *, allow_environment_migration=False):
    validate_resume(
        state, state["campaign_identity"], state["profile_contract"], provenance,
        state["scientific_identity"]["scientific_coverage"],
        allow_environment_migration=allow_environment_migration,
    )
    if state["provenance"]["last_resolved"] != provenance:
        if state["provenance"]["last_resolved"]["environment_file_sha256"] != provenance["environment_file_sha256"]:
            migration = {
                "at_utc": now_utc(), "from": dict(state["provenance"]["last_resolved"]),
                "to": dict(provenance),
                "effective_after_attempts": {
                    **(state.get("environment_migrations") or [{}])[-1].get("effective_after_attempts", {}),
                    **{
                    unit["id"]: {stage: {"count": len(record["attempts"]),
                                           "sha256": canonical_digest(record["attempts"])}
                                 for stage, record in unit["stages"].items()}
                    for unit in state["units"]
                    },
                },
            }
            migration["migration_sha256"] = canonical_digest(migration)
            state.setdefault("environment_migrations", []).append(migration)
        state["provenance"]["last_resolved"] = provenance
        timestamp = now_utc()
        state["history"].append({"at_utc": timestamp, "action": "operational_provenance_changed"})
        state["updated_at_utc"] = timestamp
        validate_state(state)
        return True
    return False


def receipt_summary(state):
    return {
        "active_revision": state["plan"]["active_revision"],
        "target_coverage_count": len(state["scientific_identity"]["scientific_coverage"]),
        "units": [
            {"id": unit["id"], "revision": unit["plan_revision"], "status": overall_status(unit)}
            for unit in state["units"]
        ],
        "complete": all(overall_status(unit) == "success" for unit in state["units"]),
    }


def report_text(state):
    classification = classify(state)
    headers = (
        "profile", "campaign_tag", "era", "unit_id", "plan_revision",
        "categories", "variables", "source_status", "data_driven_status",
        "final_unit_status", "expected_source_path", "expected_data_driven_path",
    )
    def clean(value):
        return str(value).replace("\t", " ").replace("\n", " ")
    rows = []
    for unit in state["units"]:
        rows.append((
            state["campaign_identity"]["production_profile"],
            state["campaign_identity"]["campaign_tag"],
            " ".join(unit["years"]), unit["id"], unit["plan_revision"],
            " ".join(unit["category_groups"]), " ".join(unit["histograms"]),
            unit["stages"]["source"]["status"], unit["stages"]["data_driven"]["status"],
            overall_status(unit), effective_output_path(unit, "source"), effective_output_path(unit, "data_driven"),
        ))
    tsv = "\t".join(headers) + "\n" + "".join("\t".join(clean(value) for value in row) + "\n" for row in rows)
    markdown = "\n".join((
        f"# Campaign summary: {state['campaign_identity']['production_profile']}",
        "",
        f"- campaign_tag: {state['campaign_identity']['campaign_tag']}",
        f"- active_plan_revision: {state['plan']['active_revision']}",
        f"- scientific_contract_sha256: {state['scientific_identity']['scientific_contract_sha256']}",
        f"- scientific_coverage_sha256: {state['scientific_identity']['scientific_coverage_sha256']}",
        f"- final_classification: {classification}",
        f"- successful_units: {sum(overall_status(unit) == 'success' for unit in state['units'])}",
        f"- configured_units: {len(state['units'])}",
        "",
    ))
    return tsv, markdown


def render_new(default_path, supplied_path, profile, tag, root, output):
    profile_contract(profile)
    approved = partition_coverage(default_units(profile, default_path), profile)
    units = default_units(profile, default_path) if supplied_path == "-" else supplied_units(supplied_path, profile)
    verify_partition(units, profile, approved)
    revision = make_revision(1, units)
    specs = make_specs(units, profile, tag, root, revision)
    atomic_text(output, plan_tsv(specs))
    return revision


def status_line(unit):
    source = unit["stages"]["source"]["status"]
    data_driven = unit["stages"]["data_driven"]["status"]
    return "\t".join((
        overall_status(unit),
        {"pending": "planned", "running": "running", "failed": "failed", "success": "ready"}[source],
        {"pending": "planned", "blocked": "blocked", "running": "running", "failed": "failed", "success": "success"}[data_driven],
    ))


def archive_logs(state, unit_id, stage, archive_dir):
    matches = [unit for unit in state["units"] if unit["id"] == unit_id]
    if len(matches) != 1 or stage not in state["profile_contract"]["stages"]:
        fail("unknown unit or stage for native logs")
    unit = matches[0]
    names = {
        "wq_debug_log": "debug.log", "wq_transactions_log": "tr.log",
        "wq_stats_log": "stats.log", "wq_tasks_accum_log": "tasks.log",
    }
    metadata = {}
    for key, name in names.items():
        path = Path(archive_dir) / name
        if path.is_symlink():
            fail("native Work Queue log is a symbolic link")
        regular = path.is_file()
        observed = path.stat() if regular else None
        metadata[key] = {
            "path": str(path), "exists": path.exists(), "regular_file": regular,
            "nonempty": bool(observed and observed.st_size),
            "size_bytes": observed.st_size if observed else None,
            "mtime_ns": observed.st_mtime_ns if observed else None,
        }
    unit["native_work_queue_logs"][stage] = metadata
    record_transition(state, unit, stage, "logs_archived", "native_work_queue_logs_archived")


def finalize_state(state, selected_fragment=""):
    validate_state(state)
    classification = classify(state)
    selected_complete = bool(selected_fragment) and any(
        unit["id"] == selected_fragment and overall_status(unit) == "success"
        for unit in state["units"]
    )
    statuses = [overall_status(unit) for unit in state["units"]]
    exit_code = 0 if classification == "success" or selected_complete else 1
    state["summary"] = {
        "classification": classification, "configured_unit_count": len(statuses),
        "attempted_unit_count": sum(status not in {"planned", "source_ready"} for status in statuses),
        "successful_unit_count": statuses.count("success"),
        "known_failed_unit_count": sum(status.endswith("_failed") for status in statuses),
        "not_attempted_unit_count": sum(status in {"planned", "source_ready"} for status in statuses),
        "final_process_exit_code": exit_code, "selected_fragment": selected_fragment,
        "at_utc": now_utc(),
    }
    state["updated_at_utc"] = state["summary"]["at_utc"]
    return classification, exit_code


def failure_snapshot(state, unit_id, stage):
    matches = [unit for unit in state["units"] if unit["id"] == unit_id]
    if len(matches) != 1 or stage not in state["profile_contract"]["stages"]:
        fail("unknown failed unit or stage")
    unit = matches[0]
    record = unit["stages"][stage]
    attempt = record["attempts"][-1] if record["attempts"] else {}
    rows = {
        "unit_id": unit_id, "stage": stage, "result": attempt.get("result"),
        "exit_code": attempt.get("exit_code"), "start_time_utc": attempt.get("started_at_utc"),
        "end_time_utc": attempt.get("ended_at_utc"),
        "campaign_updated_at_utc": state["updated_at_utc"],
    }
    for role in unit["outputs"]:
        path = effective_output_path(unit, role)
        rows[f"{role}_path"] = path
        rows[f"{role}_exists"] = output_exists(path)
    return "field\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in rows.items())


def combined_summary(profile, tag, root, run2_path, run3_path, run2_exit, run3_exit):
    components = []
    for label, path, exit_code in (
        ("run2", run2_path, run2_exit), ("run3", run3_path, run3_exit),
    ):
        state_path = Path(path)
        if state_path.is_file():
            state = load_state(state_path)
            classification = classify(state)
            units = state["units"]
        else:
            state = None
            classification = "not_attempted"
            units = []
        components.append({
            "label": label, "classification": classification,
            "exit_code": int(exit_code), "state": state, "units": units,
        })
    headers = ("component", "component_status", "profile", "campaign_tag", "era", "unit_id",
               "source_status", "data_driven_status", "final_unit_status",
               "expected_source_path", "expected_data_driven_path")
    rows = []
    for item in components:
        state = item["state"]
        if state is None:
            rows.append((item["label"], item["classification"], "", "", "", "", "", "", "", "", ""))
            continue
        for unit in item["units"]:
            rows.append((
                item["label"], item["classification"], state["campaign_identity"]["production_profile"],
                state["campaign_identity"]["campaign_tag"], " ".join(unit["years"]), unit["id"],
                unit["stages"]["source"]["status"], unit["stages"]["data_driven"]["status"],
                overall_status(unit), effective_output_path(unit, "source"), effective_output_path(unit, "data_driven"),
            ))
    classification = (
        "success" if all(item["classification"] == "success" for item in components)
        else "complete_with_known_failures" if all(item["classification"] in {"success", "complete_with_known_failures"} for item in components)
        else "completed_with_component_blockers"
    )
    tsv = "\t".join(headers) + "\n" + "".join(
        "\t".join(str(value).replace("\t", " ").replace("\n", " ") for value in row) + "\n"
        for row in rows
    )
    markdown = "\n".join((
        f"# Combined campaign summary: {profile}", "",
        f"- campaign_tag: {tag}",
        f"- run2_component_status: {components[0]['classification']}",
        f"- run3_component_status: {components[1]['classification']}",
        f"- configured_count: {sum(len(item['units']) for item in components)}",
        f"- successful_count: {sum(overall_status(unit) == 'success' for item in components for unit in item['units'])}",
        f"- final_classification: {classification}", "",
    ))
    atomic_text(Path(root) / "campaign_summary.tsv", tsv)
    atomic_text(Path(root) / "campaign_summary.md", markdown)
    return classification


def old_schema8_contract_digest(legacy, *, current_policy):
    fields = (
        "production_profile", "campaign_tag", "output_dir", "env_file_sha256",
        "environment_fingerprint", "topcoffea_git_commit",
        "topcoffea_relevant_source_fingerprint", "ttgamma_sample_role_policy",
        "do_systs", "do_np", "region",
    )
    payload = {key: legacy[key] for key in fields}
    payload["files"] = legacy["sr_reco_scientific_files"]
    payload["coverage"] = legacy["sr_reco_target_coverage"]
    if not current_policy:
        payload["nonprompt_mode"] = legacy["nonprompt_mode"]
    return canonical_digest(payload)


def legacy_attempt(stage, old_block, old_state, result):
    prefix = "source" if stage == "source" else "nonprompt"
    runtime = old_block.get("source_runtime", {}) if stage == "source" else {}
    return {
        "number": 1, "stage": stage,
        "started_at_utc": old_block.get(f"{prefix}_started_at_utc") or old_state["created_at_utc"],
        "ended_at_utc": old_block.get(f"{prefix}_ended_at_utc"),
        "command_argv": old_block.get(f"{prefix}_command_argv") or [],
        "chunksize": runtime.get("chunksize") if stage == "source" else None,
        "resources": {},
        "provenance": {
            "imported_schema8_runtime": runtime,
            "environment_file": runtime.get("environment_file", old_state.get("env_file")),
            "legacy_schema8_scientific_contract_sha256": (
                runtime.get("scientific_contract_sha256")
                if stage == "source" and runtime else
                old_state.get("sr_reco_current_scientific_contract_sha256")
            ),
            "legacy_schema8_plan_revision": old_block["plan_revision"],
            "legacy_topeft_git_commit": old_state.get("topeft_git_commit"),
            "legacy_topcoffea_git_commit": old_state.get("topcoffea_git_commit"),
        },
        "result": result,
        "exit_code": old_block.get(f"{prefix}_exit_code"),
    }


def import_schema8_state(legacy, legacy_path, desired):
    rendered, identity, contract, provenance, approved = desired
    if legacy.get("schema_version") != 8 or identity["production_profile"] not in {"run2_sr_ext", "run3_sr_ext"}:
        fail("explicit import accepts only the immediate SR-extension schema-8 predecessor")
    expected_old = Path(identity["output_dir"]) / f".{identity['production_profile']}_campaign_state.json"
    if Path(legacy_path).resolve() != expected_old.resolve():
        fail("schema-8 import source is outside its campaign namespace")
    for key, expected in identity.items():
        if legacy.get(key) != expected:
            fail(f"schema-8 {key} differs from requested campaign")
    if legacy.get("env_file_sha256") != provenance["environment_file_sha256"]:
        fail("NEW_CAMPAIGN_REQUIRED: schema-8 frozen environment tarball differs")
    if (
        legacy.get("region") != contract["region"]
        or legacy.get("do_systs") != contract["systematics"]
        or legacy.get("do_np") is not True
        or legacy.get("ttgamma_sample_role_policy") != contract["ttgamma_sample_role_policy"]
    ):
        fail("schema-8 scientific policy or runtime differs")
    if legacy.get("sr_reco_current_scientific_contract_sha256") != old_schema8_contract_digest(legacy, current_policy=True):
        fail("schema-8 scientific contract digest is inconsistent")
    if legacy.get("sr_reco_scientific_contract_sha256") != old_schema8_contract_digest(legacy, current_policy=False):
        fail("schema-8 historical scientific contract digest is inconsistent")
    policy = legacy.get("sr_stage_policy", {})
    if (
        policy.get("scientific_contract_sha256") != legacy["sr_reco_current_scientific_contract_sha256"]
        or policy.get("effective_nonprompt_mode") != "separate"
    ):
        fail("schema-8 stage policy is inconsistent")
    old_identity_fields = (
        "production_profile", "campaign_tag", "output_dir", "topeft_git_commit",
        "env_file_sha256", "environment_fingerprint", "topcoffea_git_commit",
        "topcoffea_relevant_source_fingerprint", "ttgamma_sample_role_policy",
        "do_systs", "do_np", "region", "nonprompt_mode",
    )
    old_identity = {key: legacy[key] for key in old_identity_fields}
    old_identity["files"] = legacy["sr_reco_contract_files"]
    old_identity["chunk_size"] = legacy.get("sr_reco_chunk_size")
    if canonical_digest(old_identity) != legacy.get("sr_reco_contract_sha256"):
        fail("schema-8 original contract digest is inconsistent")
    current_files = scientific_files(contract, "source")
    historical_files = legacy.get("sr_reco_contract_files", {})
    for name, observed in current_files.items():
        historical = historical_files.get(name, legacy["sr_reco_scientific_files"].get(name))
        if historical is not None and historical != observed:
            fail(f"scientific source or sample identity differs: {name}")
    years = RUN2_YEARS if contract["run_scope"] == "run2" else RUN3_YEARS
    legacy_coverage = sorted(f"{year}|{item}" for year in years for item in legacy["sr_reco_target_coverage"])
    if legacy_coverage != approved:
        fail("schema-8 scientific coverage differs")
    old_blocks = legacy.get("blocks")
    if not isinstance(old_blocks, list) or not old_blocks:
        fail("schema-8 blocks are absent")
    active_revisions = [
        item for item in legacy.get("sr_reco_plan_revisions", [])
        if item.get("id") == legacy.get("active_sr_reco_revision")
    ]
    if len(active_revisions) != 1:
        fail("schema-8 active revision is inconsistent")
    historical_specs = {item.get("id"): item for item in active_revisions[0].get("blocks", [])}
    for block in old_blocks:
        historical = historical_specs.get(block.get("id"))
        if historical is None or any(
            block.get(key) != historical.get(key)
            for key in (
                "id", "years", "category_groups", "histograms", "output_tag",
                "output_name", "expected_outputs", "expected_nominal_path",
                "expected_np_path", "plan_revision",
            )
        ):
            fail("schema-8 block differs from active immutable revision")
    units = []
    for block in old_blocks:
        unit = {
            "id": block["id"], "years": block["years"],
            "category_groups": block["category_groups"], "histograms": block["histograms"],
        }
        validate_unit(unit, identity["production_profile"])
        units.append(unit)
    verify_partition(units, identity["production_profile"], approved)
    revision = make_revision(1, units)
    specs = []
    for unit, old_block in zip(units, old_blocks):
        specs.append({
            **unit,
            "output_tag": old_block["output_tag"],
            "output_name": old_block["output_name"],
            "outputs": {
                "source": old_block["expected_nominal_path"],
                "data_driven": old_block["expected_np_path"],
            },
            "coverage": sorted(unit_coverage(unit, identity["production_profile"])),
            "plan_revision": revision,
        })
    state = new_state(specs, approved, identity, contract, provenance)
    source_map = {"planned": "pending", "running": "running", "ready": "success", "failed": "failed"}
    downstream_map = {"blocked": "blocked", "planned": "pending", "running": "running", "failed": "failed", "success": "success"}
    for unit, old_block in zip(state["units"], old_blocks):
        source_status = source_map[old_block["source_status"]]
        downstream_status = downstream_map[old_block["nonprompt_status"]]
        unit["stages"]["source"]["status"] = source_status
        unit["stages"]["data_driven"]["status"] = downstream_status
        unit["legacy_transitions"] = old_block.get("transitions", [])
        unit["legacy_source_runtime_attempts"] = old_block.get("source_runtime_attempts", [])
        for stage, result in (
            ("source", source_status), ("data_driven", downstream_status),
        ):
            if result in {"pending", "blocked"}:
                continue
            attempt_result = (
                "running" if result == "running" else
                "imported_success" if result == "success" else "failed"
            )
            unit["stages"][stage]["attempts"].append(legacy_attempt(stage, old_block, legacy, attempt_result))
            if result == "success":
                old_receipt = old_block["source_receipt" if stage == "source" else "nonprompt_receipt"]
                artifact = artifact_receipt(unit["outputs"][stage], stored_receipt=old_receipt, verify_payload_sha256=True)
                unit["stages"][stage]["receipt"] = {"artifact": artifact, "binding": binding(state, unit, stage)}
        if source_status == "success" and old_block.get("receipt_contract_sha256") not in {
            legacy["sr_reco_scientific_contract_sha256"], legacy["sr_reco_current_scientific_contract_sha256"],
        }:
            fail("schema-8 source receipt contract is unknown")
        if source_status in {"success", "running"}:
            runtime = old_block.get("source_runtime")
            if (
                not isinstance(runtime, dict)
                or old_block.get("source_runtime_attempts", [])[-1:] != [runtime]
                or runtime.get("scientific_contract_sha256") not in {
                    legacy["sr_reco_scientific_contract_sha256"],
                    legacy["sr_reco_current_scientific_contract_sha256"],
                }
                or runtime.get("plan_revision") != old_block["plan_revision"]
                or runtime.get("environment_fingerprint") != legacy["environment_fingerprint"]
                or runtime.get("environment_file_sha256") != legacy["env_file_sha256"]
                or runtime.get("source_command_sha256") != canonical_digest(
                    old_block.get("source_command_argv")
                )
            ):
                fail("schema-8 source receipt runtime binding is inconsistent")
            if source_status == "success" and (
                old_block.get("receipt_plan_revision") != old_block["plan_revision"]
                or old_block.get("receipt_runtime_sha256") != canonical_digest(runtime)
            ):
                fail("schema-8 completed source receipt is inconsistent")
        overall_status(unit)
    state["provenance"]["schema8_import"] = {
        "original_state_path": str(legacy_path),
        "original_state_sha256": digest_file(legacy_path),
        "original_active_revision": legacy.get("active_sr_reco_revision"),
        "original_contract_sha256": legacy.get("sr_reco_contract_sha256"),
        "original_scientific_contract_sha256": legacy.get("sr_reco_current_scientific_contract_sha256"),
        "historical_scientific_files": legacy["sr_reco_scientific_files"],
    }
    state["history"].append({
        "at_utc": now_utc(), "action": "schema8_imported",
        "original_state_sha256": state["provenance"]["schema8_import"]["original_state_sha256"],
    })
    validate_state(state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    action = options.action
    values = options.arguments
    allow_environment_migration = False
    if action in {"validate", "record-provenance"} and values[-1:] == ["--allow-environment-migration"]:
        allow_environment_migration = True
        values = values[:-1]
    try:
        if action == "render-new":
            print(render_new(*values))
            return
        if action == "render-active":
            state = load_state(values[0])
            atomic_text(values[1], plan_tsv(state["units"]))
            print(state["plan"]["active_revision"])
            return
        if action == "initialize":
            state_path = values[0]
            assert_writer_lock(state_path)
            rendered, identity, contract, provenance, approved = desired_state_inputs(values[1:])
            if Path(state_path).exists():
                fail("refusing to overwrite existing campaign state")
            state = new_state(parse_plan_tsv(rendered, identity["production_profile"]), approved, identity, contract, provenance)
            atomic_json(state_path, state)
            return
        if action == "validate":
            state = load_state(values[0])
            rendered, identity, contract, provenance, approved = desired_state_inputs(
                values[1:], approved_coverage=state["scientific_identity"]["scientific_coverage"]
            )
            validate_resume(state, identity, contract, provenance, approved,
                            allow_environment_migration=allow_environment_migration)
            return
        if action == "reconcile":
            state_path = values[0]
            assert_writer_lock(state_path)
            state = load_state(state_path)
            selected_unit = values[1] if len(values) > 1 else None
            if reconcile_running(state, selected_unit):
                atomic_json(state_path, state)
                for unit in state["units"]:
                    if selected_unit and unit["id"] != selected_unit:
                        continue
                    for stage in state["profile_contract"]["stages"]:
                        if unit["stages"][stage]["status"] == "success":
                            cleanup_published_pending(state, unit, stage)
            return
        if action in {"env_file", "environment_mode", "historical_source_commit"}:
            state = load_state(values[0], verify_artifacts=False)
            read_campaign_identity(state, *values[1:4])
            initial = state["provenance"]["last_resolved"] if state.get("environment_migrations") else state["provenance"]["initial"]
            print({
                "env_file": initial["environment_file"],
                "environment_mode": initial["environment_mode"],
                "historical_source_commit": state["provenance"]["initial"]["topeft_git_commit"],
            }[action])
            return
        if action == "status":
            state = load_state(values[0])
            matches = [unit for unit in state["units"] if unit["id"] == values[1]]
            if len(matches) != 1:
                fail("unknown campaign unit")
            print(status_line(matches[0]))
            return
        if action == "effective-path":
            state = load_state(values[0])
            matches = [unit for unit in state["units"] if unit["id"] == values[1]]
            if len(matches) != 1 or values[2] not in state["profile_contract"]["stages"]:
                fail("unknown campaign unit or stage")
            print(effective_output_path(matches[0], values[2]))
            return
        if action == "next-attempt":
            state = load_state(values[0])
            print(next_attempt(state, values[1], values[2]))
            return
        if action == "pending-path":
            state = load_state(values[0], verify_artifacts=False)
            matches = [unit for unit in state["units"] if unit["id"] == values[1]]
            if len(matches) != 1 or values[2] not in state["profile_contract"]["stages"]:
                fail("unknown unit or stage")
            print(pending_artifact_path(state, matches[0], values[2]) or "")
            return
        if action == "pending-id":
            state = load_state(values[0], verify_artifacts=False)
            matches = [unit for unit in state["units"] if unit["id"] == values[1]]
            if len(matches) != 1 or values[2] not in state["profile_contract"]["stages"]:
                fail("unknown unit or stage")
            print(pending_artifact_id(state, matches[0], values[2]) or "")
            return
        if action == "next-source-attempt":
            state = load_state(values[0])
            print(next_source_attempt(state, values[1]))
            return
        if action == "record-provenance":
            (
                state_path, commit, env_file, env_sha, env_fingerprint,
                env_mode, topcoffea_commit, topcoffea_source,
            ) = values
            assert_writer_lock(state_path)
            state = load_state(state_path)
            provenance = {
                "topeft_git_commit": commit, "environment_file": env_file,
                "environment_file_sha256": env_sha,
                "environment_fingerprint": env_fingerprint,
                "environment_mode": env_mode,
                "topcoffea_git_commit": topcoffea_commit,
                "topcoffea_relevant_source_fingerprint": topcoffea_source,
            }
            if record_attempt_provenance(state, provenance, allow_environment_migration=allow_environment_migration):
                atomic_json(state_path, state)
            return
        if action == "mark":
            state_path, unit_id, stage, status, exit_code, detail, *extra = values
            assert_writer_lock(state_path)
            state = load_state(state_path)
            transition(state, unit_id, stage, status, exit_code, detail, tuple(extra))
            atomic_json(state_path, state)
            if status in {"ready", "success"}:
                matches = [unit for unit in state["units"] if unit["id"] == unit_id]
                cleanup_published_pending(state, matches[0], stage)
            return
        if action == "archive":
            state_path, unit_id, stage, archive_dir = values
            assert_writer_lock(state_path)
            state = load_state(state_path)
            archive_logs(state, unit_id, stage, archive_dir)
            atomic_json(state_path, state)
            return
        if action == "activate":
            state_path, plan_path, expected_revision = values
            assert_writer_lock(state_path)
            state = load_state(state_path)
            revision = activate_plan(state, plan_path, expected_revision)
            atomic_json(state_path, state)
            print(revision)
            return
        if action == "classification":
            state = load_state(values[0])
            print(classify(state))
            return
        if action == "finalize":
            state_path = values[0]
            assert_writer_lock(state_path)
            state = load_state(state_path)
            classification, exit_code = finalize_state(state, values[1] if len(values) > 1 else "")
            atomic_json(state_path, state)
            print(f"{classification}\t{exit_code}")
            return
        if action == "report":
            state_path, tsv_path, markdown_path = values
            state = load_state(state_path)
            if state["summary"].get("classification") != classify(state):
                fail("campaign state must be finalized before reporting")
            tsv, markdown = report_text(state)
            atomic_text(tsv_path, tsv)
            atomic_text(markdown_path, markdown)
            return
        if action == "failure_snapshot":
            state = load_state(values[0], verify_artifacts=False)
            atomic_text(values[3], failure_snapshot(state, values[1], values[2]))
            return
        if action == "combined-summary":
            print(combined_summary(*values))
            return
        if action == "import-schema8":
            legacy_path, target_path, *input_fields = values
            if Path(target_path).exists():
                fail("refusing to overwrite existing schema-9 state")
            desired = desired_state_inputs(input_fields)
            expected_target = Path(desired[1]["output_dir"]) / f".{desired[1]['production_profile']}_campaign_state_v9.json"
            if Path(target_path).resolve() != expected_target.resolve():
                fail("schema-9 import target is outside its campaign namespace")
            legacy = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
            with campaign_lock(target_path):
                state = import_schema8_state(legacy, legacy_path, desired)
                atomic_json(target_path, state)
            print(state["scientific_identity"]["scientific_contract_sha256"])
            return
        if action == "consumer-summary":
            print(json.dumps(receipt_summary(load_state(values[0])), sort_keys=True))
            return
        if action == "audit-integrity":
            state = load_state(values[0])
            for unit in state["units"]:
                for stage in state["profile_contract"]["stages"]:
                    if unit["stages"][stage]["status"] == "success":
                        validate_receipt(state, unit, stage, verify_payload_sha256=True)
            print("Full campaign payload integrity audit passed")
            return
        fail(f"unsupported campaign operation: {action}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, IndexError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
