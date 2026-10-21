"""Plan and receipt checks for the SR extension owned by run_cr.sh.

The campaign state is the only durable plan owner. Supplied plans are inputs;
the original and replacement snapshots are retained in that one state file.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


RUN_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUN_DIRECTORY.parents[1]
CHANNELS_PATH = REPOSITORY_ROOT / "topeft/channels/ch_lst.json"
DEFAULT_PLAN_PATH = RUN_DIRECTORY / "sr_reco_default_plan.json"
VARIABLES = frozenset(
    ("fwd0pt", "j0pt", "b0pt", "met", "l0conept", "l1conept", "invmass", "tau0Tpt")
)
CONTRACT_PATHS = (
    "topeft/modules/axes.py",
    "topeft/channels/ch_lst.json",
    "analysis/topeft_run2/run_cr.sh",
    "analysis/topeft_run2/sr_reco_manifest.py",
    "analysis/topeft_run2/run_analysis.py",
    "analysis/topeft_run2/analysis_processor.py",
    "analysis/topeft_run2/fullR3_run.sh",
)
RUNTIME_PATHS = frozenset({
    "analysis/topeft_run2/run_cr.sh",
    "analysis/topeft_run2/sr_reco_manifest.py",
    "analysis/topeft_run2/run_analysis.py",
    "analysis/topeft_run2/fullR3_run.sh",
})
SCIENTIFIC_INPUT_PATHS = frozenset({
    "topeft/modules/axes.py", "topeft/channels/ch_lst.json",
})
UNIT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")


def fail(message):
    raise ValueError(message)


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


def contract_files(profile):
    if profile == "run2_sr_ext":
        cfg_names = [
            "mc_signal_samples_NDSkim.cfg", "mc_background_samples_NDSkim.cfg",
            "data_samples_NDSkim.cfg",
        ]
    elif profile == "run3_sr_ext":
        cfg_names = [
            f"NDSkim_{year}_{kind}.cfg"
            for year in ("2022", "2022EE", "2023", "2023BPix")
            for kind in ("background_samples", "data_samples", "mc_signal_samples_sr")
        ]
    else:
        fail("unsupported SR extension profile")
    names = set(CONTRACT_PATHS)
    for cfg_name in cfg_names:
        cfg_path = REPOSITORY_ROOT / "input_samples/cfgs" / cfg_name
        names.add(str(cfg_path.relative_to(REPOSITORY_ROOT)))
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry.startswith("../../input_samples/sample_jsons/") and entry.endswith(".json"):
                sample_path = (RUN_DIRECTORY / entry).resolve()
                if not sample_path.is_relative_to(REPOSITORY_ROOT):
                    fail(f"sample JSON escapes repository: {entry}")
                names.add(str(sample_path.relative_to(REPOSITORY_ROOT)))
    return {name: digest_file(REPOSITORY_ROOT / name) for name in sorted(names)}


def legacy_contract(state):
    contract = {key: state[key] for key in (
        "production_profile", "campaign_tag", "output_dir", "topeft_git_commit", "env_file_sha256",
        "environment_fingerprint", "topcoffea_git_commit", "topcoffea_relevant_source_fingerprint",
        "ttgamma_sample_role_policy", "do_systs", "do_np", "region", "nonprompt_mode",
    )}
    contract["files"] = state["sr_reco_contract_files"]
    contract["chunk_size"] = state.get("sr_reco_chunk_size")
    if canonical_digest(contract) != state.get("sr_reco_contract_sha256"):
        fail("historical campaign contract identity mismatch")


def scientific_files(state):
    names = {
        name for name in state["sr_reco_contract_files"]
        if name.startswith("input_samples/") or name in SCIENTIFIC_INPUT_PATHS
    }
    return {name: digest_file(REPOSITORY_ROOT / name) for name in sorted(names)}


def scientific_contract(state, files, *, include_execution_mode=True):
    contract = {key: state[key] for key in (
        "production_profile", "campaign_tag", "output_dir", "env_file_sha256",
        "environment_fingerprint", "topcoffea_git_commit", "topcoffea_relevant_source_fingerprint",
        "ttgamma_sample_role_policy", "do_systs", "do_np", "region",
    )}
    if include_execution_mode:
        contract["nonprompt_mode"] = state["nonprompt_mode"]
    contract["files"] = files
    contract["coverage"] = state["sr_reco_target_coverage"]
    return canonical_digest(contract)


def effective_nonprompt_mode(state):
    return state["sr_stage_policy"]["effective_nonprompt_mode"] if state["schema_version"] == 8 else state["nonprompt_mode"]


def validate_stage_policy(state):
    if state["schema_version"] != 8:
        if "sr_stage_policy" in state or "sr_reco_current_scientific_contract_sha256" in state:
            fail("stage policy requires SR schema 8")
        return
    policy = state.get("sr_stage_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "version", "effective_nonprompt_mode", "historical_nonprompt_mode",
        "historical_contract_sha256", "scientific_contract_sha256",
        "activated_from_schema", "activated_at_utc",
    }:
        fail("unsupported or inconsistent SR stage policy")
    if (
        policy["version"] != 1
        or policy["effective_nonprompt_mode"] != "separate"
        or policy["historical_nonprompt_mode"] != state["nonprompt_mode"]
        or policy["historical_contract_sha256"] != state["sr_reco_contract_sha256"]
        or policy["activated_from_schema"] not in {6, 7, 8}
        or not isinstance(policy["activated_at_utc"], str)
    ):
        fail("unsupported or inconsistent SR stage policy")
    try:
        activated_at = datetime.fromisoformat(policy["activated_at_utc"])
    except ValueError:
        fail("SR stage policy activation time is invalid")
    if activated_at.tzinfo is None:
        fail("SR stage policy activation time lacks a timezone")
    current = state.get("sr_reco_current_scientific_contract_sha256")
    if policy["scientific_contract_sha256"] != current or current != scientific_contract(
        state, state["sr_reco_scientific_files"], include_execution_mode=False
    ):
        fail("current scientific contract identity mismatch")


def validate_source_command_contract(state, command_argv, *, nonprompt_mode=None):
    """Bind a new SR source command's execution switches to its campaign."""
    if not isinstance(command_argv, list) or any(not isinstance(item, str) for item in command_argv):
        fail("source command is not a string argument vector")

    def flag_count(name):
        if any(item.startswith(f"{name}=") for item in command_argv):
            fail(f"malformed source command option: {name}")
        return command_argv.count(name)

    def require_flag(name, expected):
        count = flag_count(name)
        if count > 1 or (count == 1) != expected:
            fail(f"source command switch contradicts campaign contract: {name}")

    def require_value(name, expected):
        positions = [index for index, item in enumerate(command_argv) if item == name]
        if len(positions) != 1 or positions[0] + 1 >= len(command_argv):
            fail(f"missing or ambiguous source command option: {name}")
        value = command_argv[positions[0] + 1]
        if value.startswith("-") or value != expected:
            fail(f"source command option contradicts campaign contract: {name}")

    if state.get("region") != "SR":
        fail("source command contract applies only to SR campaigns")
    require_flag("--do-systs", state.get("do_systs") is True)
    do_np = state.get("do_np")
    if not isinstance(do_np, bool):
        fail("campaign nonprompt switch is not boolean")
    require_flag("--do-np", do_np)

    np_modes = [item for item in command_argv if item.startswith("--np-postprocess")]
    defer_count = flag_count("--defer-np")
    mode = effective_nonprompt_mode(state) if nonprompt_mode is None else nonprompt_mode
    if mode == "inline" and do_np:
        if np_modes != ["--np-postprocess=inline"] or defer_count:
            fail("source command nonprompt realization contradicts campaign contract")
    elif mode == "separate" and do_np:
        if np_modes or defer_count != 1:
            fail("source command nonprompt realization contradicts campaign contract")
    elif not do_np:
        if np_modes or defer_count:
            fail("source command nonprompt realization contradicts campaign contract")
    else:
        fail("campaign nonprompt realization is invalid")

    require_flag("--sr", True)
    require_flag("--cr", False)
    require_flag("--all-analysis", True)
    if any(flag_count(name) for name in (
        "--offZ-3l-split", "--tau-h-analysis", "--fwd-analysis", "--skip-sr",
    )):
        fail("source command contains a conflicting SR analysis selector")
    require_value("-x", "work_queue")

    environment_mode = state.get("environment_mode")
    if environment_mode not in {"explicit_snapshot", "current_resolved"}:
        fail("campaign environment mode is invalid")
    require_flag("--snapshot", environment_mode == "explicit_snapshot")


def source_runtime(state, block, command_argv):
    if block.get("status") not in {"planned", "source_failed"} or block.get("plan_revision") != state["active_sr_reco_revision"]:
        fail("runtime selection requires an active uncompleted unit")
    validate_source_command_contract(state, command_argv)
    def option_value(name, default=None):
        positions = [index for index, item in enumerate(command_argv) if item == name]
        if len(positions) > 1 or (positions and positions[0] + 1 >= len(command_argv)):
            fail(f"ambiguous source runtime option: {name}")
        return command_argv[positions[0] + 1] if positions else default
    def option_values(name):
        positions = [index for index, item in enumerate(command_argv) if item == name]
        if len(positions) != 1:
            fail(f"missing or ambiguous source selection: {name}")
        values = []
        for item in command_argv[positions[0] + 1:]:
            if item.startswith("-"):
                break
            values.append(item)
        return values
    if (
        option_values("-y") != block["years"]
        or option_values("--category-groups") != block["category_groups"]
        or option_values("--hist-vars") != block["histograms"]
        or option_value("-t") != block["output_tag"]
        or option_value("-p") != state["output_dir"]
        or option_value("--env-file") != state["env_file"]
        or option_value("--ttgamma-sample-role-policy") != state["ttgamma_sample_role_policy"]
    ):
        fail("source selection differs from the bound unit or campaign inputs")
    chunksize = int(option_value("-s"))
    treereduction = int(option_value("--wq-treereduction", "10"))
    processing_cores = option_value("--wq-processing-cores")
    accumulation_memory = option_value("--wq-accumulating-memory-mb")
    accumulation_cores = option_value("--wq-accumulating-cores")
    if chunksize not in (25000, 50000, 75000, 100000) or treereduction not in (5, 10, 20):
        fail("invalid resolved source runtime")
    if processing_cores not in (None, "1") or (accumulation_memory is None) != (accumulation_cores is None):
        fail("invalid Work Queue category runtime")
    if accumulation_memory is not None and (int(accumulation_memory) <= 0 or int(accumulation_cores) <= 0):
        fail("invalid accumulating category minimum")
    return {
        "attempt": len(block.get("source_runtime_attempts", [])) + 1,
        "source_command_sha256": canonical_digest(command_argv),
        "chunksize": chunksize,
        "treereduction": treereduction,
        "processing_cores_max": int(processing_cores) if processing_cores else None,
        "accumulating_memory_min_mb": int(accumulation_memory) if accumulation_memory else None,
        "accumulating_cores_min": int(accumulation_cores) if accumulation_cores else None,
        "manager_source_files": {name: digest_file(REPOSITORY_ROOT / name) for name in sorted(RUNTIME_PATHS)},
        "environment_file": state["env_file"],
        "environment_file_sha256": state["env_file_sha256"],
        "environment_fingerprint": state["environment_fingerprint"],
        "topcoffea_git_commit": state["topcoffea_git_commit"],
        "topcoffea_relevant_source_fingerprint": state["topcoffea_relevant_source_fingerprint"],
        "scientific_contract_sha256": (
            state["sr_reco_current_scientific_contract_sha256"]
            if state["schema_version"] == 8 else state["sr_reco_scientific_contract_sha256"]
        ),
        "plan_revision": block["plan_revision"],
    }


def physical_pairs():
    groups = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))["ALL_CH_LST_SR"]
    approved = {"2l", "2lss_1tau", "2los_1tau", "4l", "3l_m_offZ", "3l_p_offZ", "3l_onZ_tau", "3l_fwd"}
    result = {}
    for group in approved:
        definition = groups[group]
        result[group] = {
            (group, channel[0], jet)
            for channel in definition["lep_chan_lst"]
            for jet in definition["jet_lst"]
        }
    return result


def applicable(pair, variable):
    channel = pair[1]
    if variable == "fwd0pt":
        return channel.startswith("2lss_fwd_") or channel.endswith("_fwd")
    if variable == "tau0Tpt":
        return "_1tau" in channel
    return True


def target_coverage():
    pairs = physical_pairs()
    return {
        (*pair, variable)
        for group_pairs in pairs.values()
        for pair in group_pairs
        for variable in VARIABLES
        if applicable(pair, variable)
    }


def unit_coverage(unit):
    pairs = physical_pairs()
    groups = unit.get("category_groups")
    variables = unit.get("histograms")
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        fail("unit category_groups must be a unique nonempty list")
    if not isinstance(variables, list) or not variables or len(variables) != len(set(variables)):
        fail("unit histograms must be a unique nonempty list")
    if not set(groups) <= pairs.keys() or not set(variables) <= VARIABLES:
        fail("unit declares an unknown SR group or histogram")
    result = {
        (*pair, variable)
        for group in groups
        for pair in pairs[group]
        for variable in variables
        if applicable(pair, variable)
    }
    if any(not any(item[3] == variable for item in result) for variable in variables):
        fail("unit contains an inapplicable histogram")
    return result


def coverage_key(item):
    return "|".join(item)


def verify_partition(units, expected):
    counts = Counter(coverage_key(item) for unit in units for item in unit_coverage(unit))
    wanted = {coverage_key(item) for item in expected}
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(wanted - counts.keys())
    unexpected = sorted(counts.keys() - wanted)
    if duplicate or missing or unexpected:
        fail(f"coverage partition invalid: duplicate={duplicate[:3]} missing={missing[:3]} unexpected={unexpected[:3]}")
    return sorted(wanted)


def load_plan(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"units"} or not isinstance(payload["units"], list) or not payload["units"]:
        fail("SR plan requires one nonempty units list")
    seen = set()
    units = []
    for unit in payload["units"]:
        if not isinstance(unit, dict) or set(unit) != {"id", "category_groups", "histograms"}:
            fail("SR unit requires id, category_groups and histograms only")
        unit_id = unit["id"]
        if not isinstance(unit_id, str) or not UNIT_ID_PATTERN.fullmatch(unit_id) or unit_id in seen:
            fail("SR unit ids must be unique snake_case names")
        seen.add(unit_id)
        unit_coverage(unit)
        units.append(unit)
    return units


def years_for(profile):
    return ["run2"] if profile == "run2_sr_ext" else ["2022", "2022EE", "2023", "2023BPix"]


def validate_output_namespace(blocks):
    owners = {}
    for block in blocks:
        outputs = block["expected_outputs"]
        paths = [*outputs, *(f"{output}.metadata.json" for output in outputs)]
        for path in paths:
            previous = owners.get(path)
            if previous is not None:
                fail(f"campaign output path collision between {previous} and {block['id']}: {path}")
            owners[path] = block["id"]


def plan_blocks(units, profile, tag, root, revision):
    years = years_for(profile)
    year_label = "UL16-UL16APV-UL17-UL18" if profile == "run2_sr_ext" else "2022-2022EE-2023-2023BPix"
    blocks = []
    for unit in units:
        block_id = f"{profile}_{unit['id']}"
        output_tag = f"{tag}_{revision}_{unit['id']}"
        output_name = f"{year_label}SRs_{output_tag}"
        nominal = str(Path(root) / f"{output_name}.pkl.gz")
        nonprompt = str(Path(root) / f"{output_name}_np.pkl.gz")
        blocks.append({
            "id": block_id,
            "years": years,
            "category_groups": unit["category_groups"],
            "histograms": unit["histograms"],
            "output_tag": output_tag,
            "output_name": output_name,
            "expected_outputs": [nominal, nonprompt],
            "expected_nominal_path": nominal,
            "expected_np_path": nonprompt,
            "plan_revision": revision,
        })
    validate_output_namespace(blocks)
    return blocks


def plan_tsv(blocks):
    rows = []
    for block in blocks:
        rows.append("\t".join((
            block["id"], " ".join(block["years"]), " ".join(block["category_groups"]),
            " ".join(block["histograms"]), block["output_tag"], block["output_name"],
            block["expected_nominal_path"], block["expected_np_path"],
        )))
    return "\n".join(rows) + "\n"


def artifact_receipt(path, *, stored_receipt=None, verify_payload_sha256=True):
    path = Path(path)
    sidecar_path = Path(f"{path}.metadata.json")
    if path.is_symlink() or sidecar_path.is_symlink() or not path.is_file() or not sidecar_path.is_file() or path.stat().st_size == 0:
        fail(f"required artifact or sidecar missing: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        fail(f"invalid artifact sidecar: {path}")
    artifact = sidecar.get("artifact", {})
    if not isinstance(artifact, dict):
        fail(f"invalid artifact sidecar: {path}")
    if not verify_payload_sha256 and not isinstance(stored_receipt, dict):
        fail(f"stored receipt required for fast artifact readback: {path}")
    recorded_sha256 = stored_receipt.get("sha256") if stored_receipt is not None else None
    if stored_receipt is not None and (
        not isinstance(recorded_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", recorded_sha256) is None
    ):
        fail(f"invalid stored artifact SHA-256: {path}")
    observed = {
        "path": str(path),
        "sha256": digest_file(path) if verify_payload_sha256 else recorded_sha256,
        "size_bytes": path.stat().st_size,
        "sidecar_sha256": digest_file(sidecar_path),
    }
    if artifact.get("pkl_basename") != path.name or artifact.get("pkl_sha256") != observed["sha256"] or artifact.get("pkl_size_bytes") != observed["size_bytes"]:
        fail(f"artifact sidecar identity mismatch: {path}")
    lineage = sidecar.get("lineage", {})
    if not sidecar.get("synthetic") and not (
        isinstance(lineage, dict) and isinstance(lineage.get("inputs"), list)
    ):
        fail(f"artifact sidecar lacks lineage: {path}")
    if stored_receipt is not None and stored_receipt != observed:
        fail(f"stored artifact receipt changed: {path}")
    return observed


def capture_receipt(state, block, stage):
    if not state.get("sr_reco_plan_revisions"):
        fail("SR campaign lacks revision history")
    if stage == "source":
        block["source_receipt"] = artifact_receipt(block["expected_nominal_path"])
    elif stage == "nonprompt":
        if "source_receipt" not in block:
            fail("source receipt absent before nonprompt completion")
        block["nonprompt_receipt"] = artifact_receipt(block["expected_np_path"])
    else:
        fail("unknown receipt stage")
    if stage == "source" or "source_runtime" in block:
        block["receipt_contract_sha256"] = (
            block["source_runtime"]["scientific_contract_sha256"]
            if "source_runtime" in block else state["sr_reco_contract_sha256"]
        )
        block["receipt_plan_revision"] = block["plan_revision"]
    if state["schema_version"] in {7, 8} and "source_runtime" in block:
        block["receipt_runtime_sha256"] = canonical_digest(block["source_runtime"])


def verify_receipt(state, block, *, verify_payload_sha256=False):
    revision = block.get("plan_revision")
    snapshots = [item for item in state.get("sr_reco_plan_revisions", []) if item.get("id") == revision]
    if len(snapshots) != 1 or not any(
        all(block.get(key) == historical.get(key) for key in (
            "id", "years", "category_groups", "histograms", "output_tag", "output_name",
            "expected_outputs", "expected_nominal_path", "expected_np_path", "plan_revision"
        )) for historical in snapshots[0]["blocks"]
    ):
        fail(f"block {block.get('id')} does not match its immutable revision")
    expected_contract = (block["source_runtime"]["scientific_contract_sha256"]
                         if "source_runtime" in block else state["sr_reco_contract_sha256"])
    if block.get("receipt_contract_sha256") != expected_contract or block.get("receipt_plan_revision") != revision:
        fail(f"block {block['id']} lacks bound contract/revision receipt")
    if "source_runtime" in block:
        runtime = block["source_runtime"]
        command_mode = state["nonprompt_mode"]
        if state["schema_version"] == 8:
            if expected_contract == state["sr_reco_current_scientific_contract_sha256"]:
                command_mode = "separate"
            elif expected_contract != state.get("sr_reco_scientific_contract_sha256"):
                fail(f"block {block['id']} has unknown runtime contract")
        validate_source_command_contract(state, block.get("source_command_argv"), nonprompt_mode=command_mode)
        if (
            block.get("receipt_runtime_sha256") != canonical_digest(runtime)
            or block.get("source_runtime_attempts", [])[-1:] != [runtime]
            or runtime.get("plan_revision") != revision
            or runtime.get("source_command_sha256") != canonical_digest(block.get("source_command_argv"))
            or runtime.get("scientific_contract_sha256") != expected_contract
            or runtime.get("environment_file_sha256") != state["env_file_sha256"]
            or runtime.get("topcoffea_git_commit") != state["topcoffea_git_commit"]
            or runtime.get("chunksize") not in (25000, 50000, 75000, 100000)
        ):
            fail(f"block {block['id']} runtime receipt changed")
    if block.get("source_status") != "ready" or block.get("status") not in {
        "source_ready", "nonprompt_running", "nonprompt_failed", "success"
    }:
        fail(f"block {block['id']} has no completed source")
    adoption = block.get("source_adoption")
    if adoption is not None:
        if state["schema_version"] != 8 or not isinstance(adoption, dict) or set(adoption) != {
            "version", "accepted_evidence", "accepted_evidence_sha256", "original_source_attempt",
            "historical_composite_outcome", "source_only_exit_code",
        }:
            fail(f"block {block['id']} has invalid source adoption")
        if (
            adoption["version"] != 1
            or not isinstance(adoption["accepted_evidence"], dict)
            or adoption["source_only_exit_code"] is not None
            or adoption["accepted_evidence_sha256"] != canonical_digest(adoption["accepted_evidence"])
            or adoption["original_source_attempt"] != block.get("source_runtime", {}).get("attempt")
        ):
            fail(f"block {block['id']} has invalid source adoption")
        outcome = adoption["historical_composite_outcome"]
        evidence = adoption["accepted_evidence"]
        if not isinstance(outcome, dict) or set(outcome) != {"exit_code", "signal", "cause"}:
            fail(f"block {block['id']} lost historical composite outcome")
        if outcome != evidence.get("historical_composite_outcome") or block.get("source_exit_code") != outcome["exit_code"] or block.get("source_signal") != outcome["signal"] or not any(
            item.get("stage") == "source" and item.get("status") == "failed"
            and item.get("exit_code") == outcome["exit_code"] and item.get("signal") == outcome["signal"]
            for item in block.get("transitions", [])
        ):
            fail(f"block {block['id']} lost historical composite outcome")
        if any(evidence.get(key) != value for key, value in {
            "campaign_tag": state["campaign_tag"], "production_profile": state["production_profile"],
            "output_dir": state["output_dir"], "block_id": block["id"],
            "plan_revision": revision, "source_attempt": block["source_runtime"]["attempt"],
            "source_command_sha256": canonical_digest(block["source_command_argv"]),
            "source_runtime_sha256": canonical_digest(block["source_runtime"]),
            "scientific_contract_sha256": expected_contract,
            "environment_file_sha256": state["env_file_sha256"],
            "source_receipt": block.get("source_receipt"),
        }.items()):
            fail(f"block {block['id']} source adoption binding changed")
    elif block.get("source_exit_code") != 0:
        fail(f"block {block['id']} source did not exit successfully")
    artifact_receipt(
        block["expected_nominal_path"], stored_receipt=block.get("source_receipt"),
        verify_payload_sha256=verify_payload_sha256,
    )
    if block["status"] == "success":
        if block.get("nonprompt_status") != "success" or block.get("nonprompt_exit_code") != 0:
            fail(f"block {block['id']} nonprompt did not exit successfully")
        artifact_receipt(
            block["expected_np_path"], stored_receipt=block.get("nonprompt_receipt"),
            verify_payload_sha256=verify_payload_sha256,
        )
        nonprompt_sidecar = json.loads(
            Path(f"{block['expected_np_path']}.metadata.json").read_text(encoding="utf-8")
        )
        if not nonprompt_sidecar.get("synthetic") and not any(
            isinstance(item, dict) and item.get("pkl_sha256") == block["source_receipt"]["sha256"]
            for item in nonprompt_sidecar["lineage"]["inputs"]
        ):
            fail(f"block {block['id']} nonprompt lineage does not bind source")
    else:
        if effective_nonprompt_mode(state) != "separate" or block.get("nonprompt_status") not in {"planned", "running", "failed"}:
            fail(f"block {block['id']} is an invalid deferred source")
        if block.get("nonprompt_receipt") is not None:
            fail(f"block {block['id']} has a premature nonprompt receipt")
        if block["status"] == "source_ready" and (
            os.path.lexists(block["expected_np_path"]) or
            os.path.lexists(f"{block['expected_np_path']}.metadata.json")
        ):
            fail(f"block {block['id']} has ambiguous deferred nonprompt output")


def load_state(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_contract(state):
    legacy_contract(state)
    if state["schema_version"] == 6:
        if state["sr_reco_contract_files"] != contract_files(state["production_profile"]):
            fail("frozen scientific/software source contract changed")
    else:
        files = scientific_files(state)
        if state.get("sr_reco_scientific_files") != files:
            fail("frozen scientific/input source contract changed")
        if state.get("sr_reco_scientific_contract_sha256") != scientific_contract(state, files):
            fail("scientific campaign contract identity mismatch")
    validate_stage_policy(state)


def check_state(state, *, require_contract=True, verify_payload_sha256=False):
    if state.get("schema_version") not in {6, 7, 8} or state.get("production_profile") not in {"run2_sr_ext", "run3_sr_ext"}:
        fail("legacy or non-SR campaign cannot be adopted into revision state")
    validate_stage_policy(state)
    if require_contract:
        current_contract(state)
    revisions = state.get("sr_reco_plan_revisions")
    if not isinstance(revisions, list) or not revisions or len({item["id"] for item in revisions}) != len(revisions):
        fail("invalid plan revision history")
    if state.get("active_sr_reco_revision") != revisions[-1]["id"]:
        fail("active revision pointer is inconsistent")
    for index, revision in enumerate(revisions, 1):
        snapshot = revision.get("blocks")
        if not isinstance(snapshot, list) or not snapshot:
            fail("empty or invalid revision snapshot")
        validate_output_namespace(snapshot)
        units = [{
            "id": block["id"].removeprefix(f"{state['production_profile']}_"),
            "category_groups": block["category_groups"],
            "histograms": block["histograms"],
        } for block in snapshot]
        if revision["id"] != f"r{index}_{canonical_digest(units)[:12]}":
            fail("revision snapshot identity mismatch")
    validate_output_namespace([
        block for revision in revisions for block in revision["blocks"]
    ])
    target = target_coverage()
    if set(state.get("sr_reco_target_coverage", [])) != {coverage_key(item) for item in target}:
        fail("desired physical coverage changed")
    blocks = state.get("blocks", [])
    if len({block.get("id") for block in blocks}) != len(blocks):
        fail("duplicate campaign unit id")
    if len({block.get("expected_nominal_path") for block in blocks}) != len(blocks):
        fail("duplicate campaign output path")
    validate_output_namespace(blocks)
    for block in blocks:
        if any(Path(output).parent != Path(state["output_dir"]) for output in block["expected_outputs"]):
            fail("campaign output escapes its root")
    if any(block.get("plan_revision") not in {item["id"] for item in revisions} for block in blocks):
        fail("block has an unknown plan revision")
    for block in blocks:
        historical = [
            item for revision in revisions for item in revision["blocks"]
            if item.get("id") == block.get("id")
        ]
        if len(historical) != 1 or block_spec(block) != historical[0]:
            fail(f"block {block.get('id')} differs from immutable plan snapshot")
        if block["plan_revision"] != state["active_sr_reco_revision"] and block.get("status") not in {"success", "source_ready"}:
            fail("inactive revision contains unfinished unit")
        expected_stages = {
            "planned": ("planned", "blocked"),
            "source_running": ("running", "blocked"),
            "source_ready": ("ready", "planned"),
            "source_failed": ("failed", "blocked"),
            "nonprompt_running": ("ready", "running"),
            "nonprompt_failed": ("ready", "failed"),
            "success": ("ready", "success"),
        }.get(block.get("status"))
        if expected_stages is None or (block.get("source_status"), block.get("nonprompt_status")) != expected_stages:
            fail(f"block {block['id']} has contradictory stage state")
        if (
            state["schema_version"] in {7, 8}
            and block.get("status") != "planned"
            and "source_runtime" not in block
            and block.get("receipt_contract_sha256") != state["sr_reco_contract_sha256"]
        ):
            fail(f"block {block['id']} lacks execution runtime identity")
        if "source_runtime" in block and block.get("source_runtime_attempts", [])[-1:] != [block["source_runtime"]]:
            fail(f"block {block['id']} runtime attempt history changed")
        if block.get("expected_outputs") != [block.get("expected_nominal_path"), block.get("expected_np_path")]:
            fail(f"block {block['id']} has contradictory output paths")
    verify_partition(blocks, target)
    for block in blocks:
        if block.get("status") in {"success", "source_ready", "nonprompt_running", "nonprompt_failed"}:
            verify_receipt(state, block, verify_payload_sha256=verify_payload_sha256)
    return blocks


def upgrade_runtime_policy(state_path, expected_old_commit):
    state_path = Path(state_path)
    profile = state_path.name.removeprefix(".").removesuffix("_campaign_state.json")
    lock_path = state_path.parent / f".{profile}_campaign.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        fail("existing campaign writer lock is unavailable")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("campaign writer lock is held")
        state = load_state(state_path)
        if state.get("schema_version") != 6 or state.get("topeft_git_commit") != expected_old_commit:
            fail("runtime transition requires exact historical schema and source commit")
        legacy_contract(state)
        check_state(state, require_contract=False)
        for block in state["blocks"]:
            if block["status"] == "planned":
                for output in block["expected_outputs"]:
                    if os.path.lexists(output) or os.path.lexists(f"{output}.metadata.json"):
                        fail(f"pending unit has an output collision: {output}")
            elif block["status"] != "success" and not (
                block["status"] == "source_ready" and state["nonprompt_mode"] == "separate"
            ):
                fail(f"unit {block['id']} is in-flight, failed or ambiguous")
        recorded_files = state["sr_reco_contract_files"]
        files = scientific_files(state)
        if set(files) != {name for name in recorded_files if name.startswith("input_samples/") or name in SCIENTIFIC_INPUT_PATHS}:
            fail("scientific input file set changed")
        for name, observed_digest in files.items():
            if observed_digest != recorded_files[name]:
                fail(f"historical scientific/input file changed: {name}")
        candidate = dict(state)
        candidate["schema_version"] = 7
        candidate["sr_reco_scientific_files"] = files
        candidate["sr_reco_scientific_contract_sha256"] = scientific_contract(candidate, files)
        candidate["runtime_policy_transition"] = {
            "historical_commit": expected_old_commit,
            "transition_utc": datetime.now(timezone.utc).isoformat(),
        }
        check_state(candidate)
        atomic_json(state_path, candidate)
        return candidate["sr_reco_scientific_contract_sha256"]


def campaign_lock(state_path):
    state_path = Path(state_path)
    profile = state_path.name.removeprefix(".").removesuffix("_campaign_state.json")
    lock_path = state_path.parent / f".{profile}_campaign.lock"
    if state_path.is_symlink() or lock_path.is_symlink() or not lock_path.is_file():
        fail("existing campaign state or writer lock is unavailable")
    return lock_path


def validate_state_location(state_path, state):
    output_root = Path(state["output_dir"])
    if (
        not output_root.is_absolute()
        or state_path.parent != output_root
        or state_path.name != f".{state['production_profile']}_campaign_state.json"
        or any(parent.is_symlink() for parent in (output_root, *output_root.parents))
    ):
        fail("campaign state path is foreign or symbolic")


def activate_stage_policy(state_path, expected_revision):
    state_path = Path(state_path)
    with campaign_lock(state_path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("campaign writer lock is held")
        state = load_state(state_path)
        validate_state_location(state_path, state)
        if state.get("active_sr_reco_revision") != expected_revision:
            fail("stale expected plan revision")
        legacy_contract(state)
        check_state(state, require_contract=state.get("schema_version") != 6)
        if state["schema_version"] == 8:
            return state["sr_reco_current_scientific_contract_sha256"]
        if state["nonprompt_mode"] not in {"inline", "separate"}:
            fail("unsupported historical nonprompt realization")
        for block in state["blocks"]:
            if block["status"] in {"source_running", "nonprompt_running"}:
                fail("in-flight or ambiguous unit prevents stage activation")
            if block["status"] == "planned" and any(
                os.path.lexists(path) or os.path.lexists(f"{path}.metadata.json")
                for path in block["expected_outputs"]
            ):
                fail("pending unit has an output collision")
            if block["status"] == "source_failed" and any(
                os.path.lexists(path) or os.path.lexists(f"{path}.metadata.json")
                for path in (block["expected_np_path"],)
            ):
                fail("failed unit has an ambiguous nonprompt output")
        files = scientific_files(state)
        if any(state["sr_reco_contract_files"].get(name) != value for name, value in files.items()):
            fail("historical scientific/input file changed")
        candidate = dict(state)
        candidate["schema_version"] = 8
        candidate["sr_reco_scientific_files"] = files
        if state["schema_version"] == 6:
            candidate["sr_reco_scientific_contract_sha256"] = scientific_contract(state, files)
        current = scientific_contract(state, files, include_execution_mode=False)
        candidate["sr_reco_current_scientific_contract_sha256"] = current
        candidate["sr_stage_policy"] = {
            "version": 1,
            "effective_nonprompt_mode": "separate",
            "historical_nonprompt_mode": state["nonprompt_mode"],
            "historical_contract_sha256": state["sr_reco_contract_sha256"],
            "scientific_contract_sha256": current,
            "activated_from_schema": state["schema_version"],
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        check_state(candidate)
        atomic_json(state_path, candidate)
        return current


def validate_adoption_evidence(state, block, evidence, *, verify_payload_sha256=False):
    required = {
        "version", "accepted_by", "evidence_ref", "validation_basis", "campaign_tag",
        "production_profile", "output_dir", "block_id", "plan_revision", "source_attempt",
        "source_command_sha256", "source_runtime_sha256", "scientific_contract_sha256",
        "environment_file_sha256", "source_receipt", "historical_composite_outcome",
        "source_publication_complete",
    }
    if not isinstance(evidence, dict) or set(evidence) != required or evidence["version"] != 1:
        fail("unsupported or incomplete source adoption evidence")
    if any(not isinstance(evidence[key], str) or not evidence[key].strip() for key in (
        "accepted_by", "evidence_ref"
    )) or evidence["validation_basis"] not in {"accepted_prior_validation", "full_payload_sha256"} or evidence["source_publication_complete"] is not True:
        fail("source adoption lacks accepted completion authority")
    runtime = block.get("source_runtime")
    if not isinstance(runtime, dict) or block.get("source_runtime_attempts", [])[-1:] != [runtime]:
        fail("source adoption lacks original runtime attempt")
    validate_source_command_contract(state, block.get("source_command_argv"), nonprompt_mode="inline")
    if runtime.get("scientific_contract_sha256") != state.get("sr_reco_scientific_contract_sha256") or runtime.get("plan_revision") != block["plan_revision"] or runtime.get("topcoffea_git_commit") != state["topcoffea_git_commit"]:
        fail("source adoption historical scientific/runtime contract changed")
    bound = {
        "campaign_tag": state["campaign_tag"],
        "production_profile": state["production_profile"],
        "output_dir": state["output_dir"],
        "block_id": block["id"],
        "plan_revision": block["plan_revision"],
        "source_attempt": runtime["attempt"],
        "source_command_sha256": canonical_digest(block["source_command_argv"]),
        "source_runtime_sha256": canonical_digest(runtime),
        "scientific_contract_sha256": runtime["scientific_contract_sha256"],
        "environment_file_sha256": state["env_file_sha256"],
    }
    if any(evidence.get(key) != value for key, value in bound.items()) or runtime["source_command_sha256"] != bound["source_command_sha256"] or runtime["environment_file_sha256"] != state["env_file_sha256"]:
        fail("source adoption campaign, command, runtime or scientific identity mismatch")
    outcome = evidence["historical_composite_outcome"]
    if not isinstance(outcome, dict) or set(outcome) != {"exit_code", "signal", "cause"} or not isinstance(outcome["exit_code"], int) or outcome["exit_code"] <= 0 or outcome["signal"] != (outcome["exit_code"] - 128 if outcome["exit_code"] > 128 else None) or not isinstance(outcome["cause"], str) or not outcome["cause"] or not any(
        item.get("stage") == "source" and item.get("status") == "failed"
        and item.get("exit_code") == outcome["exit_code"] and item.get("signal") == outcome["signal"]
        for item in block.get("transitions", [])
    ):
        fail("source adoption historical composite outcome mismatch")
    source_path = Path(block["expected_nominal_path"])
    output_root = Path(state["output_dir"])
    if not output_root.is_absolute() or any(parent.is_symlink() for parent in (output_root, *output_root.parents)) or source_path.parent != output_root or any(
        path.is_symlink() for path in (source_path, Path(f"{source_path}.metadata.json"))
    ) or source_path.resolve().parent != output_root.resolve():
        fail("source adoption path is foreign or symbolic")
    sidecar_path = Path(f"{source_path}.metadata.json")
    source_stat = source_path.stat()
    receipt = artifact_receipt(
        source_path, stored_receipt=evidence["source_receipt"],
        verify_payload_sha256=verify_payload_sha256 or evidence["validation_basis"] == "full_payload_sha256",
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact = sidecar.get("artifact", {})
    if sidecar.get("synthetic") or sidecar.get("metadata_schema_version") != 2 or artifact.get("artifact_kind") != "processor_output" or artifact.get("merged") is not False or artifact.get("nominal_container_schema_version") != 2 or artifact.get("nominal_container_layout") != "split_sibling_v1" or sidecar.get("lineage", {}).get("inputs") != [] or not isinstance(sidecar.get("histogram_applicability"), dict) or not isinstance(sidecar.get("production_sample_contract"), dict):
        fail("source adoption sidecar lacks the production contract")
    applicability = sidecar["histogram_applicability"]
    if applicability.get("contract_version") != 1 or applicability.get("analysis_mode") != "all" or applicability.get("producer_query") != "AnalysisProcessor.histogram_fill_is_applicable" or not isinstance(applicability.get("families"), dict) or set(applicability["families"]) != set(block["histograms"]):
        fail("source adoption histogram applicability differs from the unit")
    sample_contract = sidecar["production_sample_contract"]
    if sample_contract.get("contract_version") != 1 or sample_contract.get("resolved_mode") != "production":
        fail("source adoption production sample contract is invalid")
    cfg_identities = sample_contract.get("cfg_identities")
    if not isinstance(cfg_identities, list) or any(not isinstance(item, dict) for item in cfg_identities):
        fail("source adoption CFG identities are invalid")
    cfg_recorded = {item.get("basename"): item.get("content_sha256") for item in cfg_identities}
    cfg_expected = {Path(name).name: digest for name, digest in state["sr_reco_contract_files"].items() if name.startswith("input_samples/cfgs/")}
    if len(cfg_identities) != len(cfg_expected) or cfg_recorded != cfg_expected or sample_contract.get("wrapper_identity") != "run_cr.sh -> fullR3_run.sh":
        fail("source adoption CFG or sample contract mismatch")
    stable_stat = source_path.stat()
    if any(getattr(source_stat, field) != getattr(stable_stat, field) for field in (
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
    )) or receipt != evidence["source_receipt"]:
        fail("source adoption artifact changed during validation")
    return receipt


def adopt_source(state_path, block_id, evidence_path, expected_revision):
    state_path = Path(state_path)
    evidence_path = Path(evidence_path)
    if evidence_path.is_symlink() or not evidence_path.is_file():
        fail("source adoption evidence path is not a regular file")
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    with campaign_lock(state_path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("campaign writer lock is held")
        state = load_state(state_path)
        validate_state_location(state_path, state)
        check_state(state)
        if state["schema_version"] != 8 or state["active_sr_reco_revision"] != expected_revision:
            fail("source adoption requires an activated exact SR revision")
        matches = [block for block in state["blocks"] if block["id"] == block_id]
        if len(matches) != 1:
            fail("source adoption requires one existing unit")
        block = matches[0]
        if block.get("source_adoption") is not None:
            if block["source_adoption"]["accepted_evidence"] == evidence:
                verify_receipt(state, block)
                return block["source_receipt"]["sha256"]
            fail("conflicting source adoption evidence")
        if block["status"] != "source_failed" or block["source_status"] != "failed" or block["nonprompt_status"] != "blocked" or block["plan_revision"] != expected_revision:
            fail("source adoption requires a failed active composite source")
        if state["nonprompt_mode"] != "inline" or "--np-postprocess=inline" not in block.get("source_command_argv", []):
            fail("source adoption requires a historical inline source")
        if any(os.path.lexists(path) for path in (
            block["expected_np_path"], f"{block['expected_np_path']}.metadata.json"
        )):
            fail("source adoption has a nonprompt output collision")
        receipt = validate_adoption_evidence(state, block, evidence)
        candidate = json.loads(json.dumps(state))
        adopted = next(item for item in candidate["blocks"] if item["id"] == block_id)
        outcome = evidence["historical_composite_outcome"]
        adopted["source_adoption"] = {
            "version": 1, "accepted_evidence": evidence,
            "accepted_evidence_sha256": canonical_digest(evidence),
            "original_source_attempt": adopted["source_runtime"]["attempt"],
            "historical_composite_outcome": outcome,
            "source_only_exit_code": None,
        }
        adopted["source_status"] = "ready"
        adopted["status"] = "source_ready"
        adopted["nonprompt_status"] = "planned"
        adopted["source_receipt"] = receipt
        adopted["receipt_contract_sha256"] = adopted["source_runtime"]["scientific_contract_sha256"]
        adopted["receipt_plan_revision"] = adopted["plan_revision"]
        adopted["receipt_runtime_sha256"] = canonical_digest(adopted["source_runtime"])
        timestamp = datetime.now(timezone.utc).isoformat()
        adopted["last_transition_utc"] = timestamp
        adopted["last_transition_detail"] = "accepted_historical_source_adoption"
        adopted["transitions"].append({
            "timestamp_utc": timestamp, "stage": "source", "status": "ready",
            "exit_code": None, "signal": None, "detail": "accepted_historical_source_adoption",
            "accepted_evidence_sha256": canonical_digest(evidence),
        })
        candidate["updated_at_utc"] = timestamp
        check_state(candidate)
        if evidence_path.is_symlink() or evidence_path.read_bytes() != evidence_bytes:
            fail("source adoption evidence changed during validation")
        atomic_json(state_path, candidate)
        return receipt["sha256"]


def decorate_initial_state(state, chunk_size):
    if state.get("schema_version") not in {6, 8} or "sr_reco_plan_revisions" in state:
        fail("SR state already has revisions or unsupported schema")
    target = target_coverage()
    verify_partition(state["blocks"], target)
    validate_output_namespace(state["blocks"])
    revision = state["blocks"][0].get("plan_revision")
    if not revision or any(block.get("plan_revision") != revision for block in state["blocks"]):
        fail("initial plan lacks one revision identity")
    files = contract_files(state["production_profile"])
    contract = {key: state[key] for key in (
        "production_profile", "campaign_tag", "output_dir", "topeft_git_commit", "env_file_sha256",
        "environment_fingerprint", "topcoffea_git_commit", "topcoffea_relevant_source_fingerprint",
        "ttgamma_sample_role_policy", "do_systs", "do_np", "region", "nonprompt_mode",
    )}
    contract["files"] = files
    contract["chunk_size"] = chunk_size
    state["sr_reco_chunk_size"] = chunk_size
    state["sr_reco_contract_files"] = files
    state["sr_reco_contract_sha256"] = canonical_digest(contract)
    state["sr_reco_target_coverage"] = sorted(coverage_key(item) for item in target)
    state["sr_reco_plan_revisions"] = [{"id": revision, "blocks": [block_spec(block) for block in state["blocks"]]}]
    state["active_sr_reco_revision"] = revision
    if state["schema_version"] == 8:
        if state["nonprompt_mode"] != "separate":
            fail("new SR stage policy requires separate nonprompt")
        scientific_inputs = scientific_files(state)
        state["sr_reco_scientific_files"] = scientific_inputs
        state["sr_reco_scientific_contract_sha256"] = scientific_contract(state, scientific_inputs)
        current = scientific_contract(state, scientific_inputs, include_execution_mode=False)
        state["sr_reco_current_scientific_contract_sha256"] = current
        state["sr_stage_policy"] = {
            "version": 1, "effective_nonprompt_mode": "separate",
            "historical_nonprompt_mode": "separate",
            "historical_contract_sha256": state["sr_reco_contract_sha256"],
            "scientific_contract_sha256": current,
            "activated_from_schema": 8,
            "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    return state


def initialize_state(state_path, chunk_size):
    state = decorate_initial_state(load_state(state_path), chunk_size)
    atomic_json(state_path, state)


SPEC_KEYS = (
    "id", "years", "category_groups", "histograms", "output_tag", "output_name",
    "expected_outputs", "expected_nominal_path", "expected_np_path", "plan_revision",
)


def block_spec(block):
    return {key: block[key] for key in SPEC_KEYS}


def new_pending_block(spec):
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        **spec, "status": "planned", "exit_code": None,
        "source_status": "planned", "source_exit_code": None, "source_signal": None,
        "source_command_argv": None, "source_started_at_utc": None, "source_ended_at_utc": None,
        "source_duration_seconds": None, "nonprompt_status": "blocked",
        "nonprompt_exit_code": None, "nonprompt_signal": None, "nonprompt_command_argv": None,
        "nonprompt_started_at_utc": None, "nonprompt_ended_at_utc": None,
        "nonprompt_duration_seconds": None, "native_work_queue_logs": {},
        "last_transition_utc": timestamp, "last_transition_detail": "pending_plan_activated",
        "transitions": [{"timestamp_utc": timestamp, "stage": "campaign", "status": "planned", "exit_code": None, "detail": "pending_plan_activated"}],
    }


def activate(state_path, plan_path, expected_revision):
    state = load_state(state_path)
    check_state(state)
    if state["active_sr_reco_revision"] != expected_revision:
        fail("stale expected plan revision")
    frozen = []
    for block in state["blocks"]:
        if block["status"] in {"success", "source_ready"}:
            if block["status"] == "source_ready" and effective_nonprompt_mode(state) != "separate":
                fail("incomplete inline unit cannot be frozen")
            frozen.append(block)
        elif block["status"] == "planned" and block["source_status"] == "planned" and block["nonprompt_status"] == "blocked":
            for output in block["expected_outputs"]:
                if os.path.lexists(output) or os.path.lexists(f"{output}.metadata.json"):
                    fail(f"untouched pending unit has an output collision: {output}")
        else:
            fail(f"unit {block['id']} is in-flight, failed or ambiguous")
    covered = {item for block in frozen for item in unit_coverage(block)}
    remaining = target_coverage() - covered
    units = load_plan(plan_path)
    verify_partition(units, remaining)
    next_number = len(state["sr_reco_plan_revisions"]) + 1
    revision = f"r{next_number}_{canonical_digest(units)[:12]}"
    specs = plan_blocks(units, state["production_profile"], state["campaign_tag"], state["output_dir"], revision)
    historical_ids = {block["id"] for item in state["sr_reco_plan_revisions"] for block in item["blocks"]}
    historical_paths = {path for item in state["sr_reco_plan_revisions"] for block in item["blocks"] for path in block["expected_outputs"]}
    if any(block["id"] in historical_ids for block in specs):
        fail("replacement unit id collides with historical unit")
    for block in specs:
        for output in block["expected_outputs"]:
            if output in historical_paths or os.path.lexists(output) or os.path.lexists(f"{output}.metadata.json"):
                fail(f"replacement output collides: {output}")
    replacement = [new_pending_block(spec) for spec in specs]
    candidate = dict(state)
    candidate["blocks"] = frozen + replacement
    candidate["sr_reco_plan_revisions"] = state["sr_reco_plan_revisions"] + [{"id": revision, "blocks": specs}]
    candidate["active_sr_reco_revision"] = revision
    candidate["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    for summary_key in (
        "campaign_status", "completed_at_utc", "configured_block_count",
        "attempted_block_count", "successful_block_count", "known_failed_block_count",
        "not_attempted_block_count", "final_process_exit_code",
    ):
        candidate.pop(summary_key, None)
    check_state(candidate)
    atomic_json(state_path, candidate)
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    render = subparsers.add_parser("render-new")
    for field in ("plan", "profile", "tag", "root", "output"):
        render.add_argument(field)
    active = subparsers.add_parser("render-active")
    active.add_argument("state")
    active.add_argument("output")
    active.add_argument("chunk_size")
    init = subparsers.add_parser("initialize")
    init.add_argument("state")
    init.add_argument("chunk_size")
    activate_parser = subparsers.add_parser("activate")
    for field in ("state", "plan", "expected_revision"):
        activate_parser.add_argument(field)
    upgrade = subparsers.add_parser("upgrade-runtime-policy")
    for field in ("state", "expected_old_commit"):
        upgrade.add_argument(field)
    stage_policy = subparsers.add_parser("activate-stage-policy")
    for field in ("state", "expected_revision"):
        stage_policy.add_argument(field)
    adoption = subparsers.add_parser("adopt-source")
    for field in ("state", "block_id", "evidence", "expected_revision"):
        adoption.add_argument(field)
    summary = subparsers.add_parser("consumer-summary")
    summary.add_argument("state")
    audit = subparsers.add_parser("audit-integrity", help="rehash all accepted SR PKL payloads")
    audit.add_argument("state")
    args = parser.parse_args()
    try:
        if args.action == "render-new":
            units = load_plan(args.plan)
            verify_partition(units, target_coverage())
            revision = f"r1_{canonical_digest(units)[:12]}"
            Path(args.output).write_text(plan_tsv(plan_blocks(units, args.profile, args.tag, args.root, revision)), encoding="utf-8")
            print(revision)
        elif args.action == "render-active":
            state = load_state(args.state)
            check_state(state)
            if effective_nonprompt_mode(state) != "separate":
                fail("inline SR campaign requires explicit stage-policy activation before resume")
            if state["schema_version"] == 6 and state.get("sr_reco_chunk_size") != args.chunk_size:
                fail("resume chunk size differs from frozen campaign contract")
            if state["schema_version"] == 8 and effective_nonprompt_mode(state) != "separate":
                fail("unsupported effective SR stage policy")
            if args.chunk_size not in {"25000", "50000", "75000", "100000"}:
                fail("unsupported SR runtime chunk size")
            Path(args.output).write_text(plan_tsv(state["blocks"]), encoding="utf-8")
            print(state["active_sr_reco_revision"])
        elif args.action == "initialize":
            initialize_state(args.state, args.chunk_size)
        elif args.action == "activate":
            print(activate(args.state, args.plan, args.expected_revision))
        elif args.action == "upgrade-runtime-policy":
            print(upgrade_runtime_policy(args.state, args.expected_old_commit))
        elif args.action == "activate-stage-policy":
            print(activate_stage_policy(args.state, args.expected_revision))
        elif args.action == "adopt-source":
            print(adopt_source(args.state, args.block_id, args.evidence, args.expected_revision))
        elif args.action == "consumer-summary":
            state = load_state(args.state)
            check_state(state)
            complete = all(block["status"] == "success" for block in state["blocks"])
            print(json.dumps({
                "active_revision": state["active_sr_reco_revision"],
                "target_coverage_count": len(state["sr_reco_target_coverage"]),
                "units": [{"id": block["id"], "revision": block["plan_revision"], "status": block["status"]} for block in state["blocks"]],
                "complete": complete,
            }, sort_keys=True))
        elif args.action == "audit-integrity":
            state = load_state(args.state)
            check_state(state, verify_payload_sha256=True)
            print("Full SR payload integrity audit passed")
        else:
            fail("unknown operation")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
