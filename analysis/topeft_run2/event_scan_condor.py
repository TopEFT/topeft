"""Small event-scan adapter to the existing per-execution Condor protocol.

Target resolution, submit ambiguity, identity and reconciliation remain owned
by plot_campaign_condor. Its plot-specific descriptor/worker is not reused.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from analysis.topeft_run2 import event_scan as scan
from analysis.topeft_run2 import plot_campaign_backend as durable
from analysis.topeft_run2 import plot_campaign_condor as scheduler

REPO = Path(__file__).resolve().parents[2]


def plan_manifest(manifest):
    """Structural, zero-effect readback; does not requalify scientific inputs."""
    raw = Path(manifest).read_bytes()
    data = json.loads(raw)
    scan.require(data.get("schema_version") == 1 and isinstance(data.get("units"), list)
                 and bool(data["units"]), "unsupported or empty execution manifest")
    units = sorted(data["units"], key=lambda u: u["unit_id"])
    ids = [u["unit_id"] for u in units]
    scan.require(len(ids) == len(set(ids)), "duplicate execution unit")
    for unit in units:
        body = {k: v for k, v in unit.items() if k != "unit_id"}
        scan.require(unit["unit_id"] == "unit_" + durable.identity(body)[:24],
                     "execution unit identity changed")
        scan.entry_range(unit["entry_start"], unit["entry_stop"],
                         unit["source_identity"]["tree_entries"])
    return dict(manifest_sha256=hashlib.sha256(raw).hexdigest(), units=units,
                unit_count=len(units))


def _campaign_json(path):
    scan.require(path.is_file() and not path.is_symlink(), "campaign evidence unavailable")
    return json.loads(path.read_text())


def read_campaign(campaign_root):
    root = Path(campaign_root).resolve(strict=True)
    campaign = _campaign_json(root / "campaign.json")
    body = {k: v for k, v in campaign.items() if k != "token"}
    scan.require(campaign["token"] == durable.identity(body) and campaign["version"] == 1
                 and campaign["campaign_root"] == str(root), "campaign identity changed")
    frozen = root / "manifest.json"
    plan = plan_manifest(frozen)
    scan.require(plan["manifest_sha256"] == campaign["manifest_sha256"] ==
                 scan.native.sha256(campaign["manifest"]), "campaign manifest changed")
    scan.require(campaign["unit_ids"] == [u["unit_id"] for u in plan["units"]],
                 "campaign unit membership changed")
    scan.require({p.name for p in (root / "units").iterdir()} == set(campaign["unit_ids"]),
                 "campaign attempt membership changed")
    for unit in plan["units"]:
        path = root / "units" / unit["unit_id"] / "_metadata/execution.json"
        descriptor = read_descriptor(path)
        scan.require((path.parent / "execution.sub").read_text() ==
                     submit_description(path, descriptor), "campaign submit description changed")
        scan.require(descriptor["unit"] == unit and descriptor["manifest"] == str(frozen)
                     and descriptor["attempt_root"] == str(root / "units" / unit["unit_id"])
                     and all(descriptor[k] == campaign[k] for k in
                             ("manifest_sha256", "wrapper", "python", "request_memory_mb",
                              "max_runtime")), "campaign execution binding changed")
    return campaign


def prepare_manifest(manifest, campaign_root, wrapper, python, *, request_memory_mb=8192,
                     max_runtime=3600):
    """Prepare all units, publishing campaign.json last; partial preparation fails closed."""
    plan = plan_manifest(manifest)
    root = Path(campaign_root).absolute()
    scan.require(not root.is_symlink(), "campaign namespace occupied")
    root = root.resolve()
    body = dict(version=1, campaign_root=str(root), manifest=str(Path(manifest).resolve()),
                manifest_sha256=plan["manifest_sha256"],
                unit_ids=[u["unit_id"] for u in plan["units"]],
                wrapper=str(Path(wrapper).absolute()), python=str(Path(python).absolute()),
                request_memory_mb=request_memory_mb, max_runtime=max_runtime)
    campaign = dict(body, token=durable.identity(body))
    if root.exists():
        scan.require(read_campaign(root) == campaign, "campaign re-entry identity mismatch")
        return campaign
    root.mkdir(mode=0o700)
    with (root / "manifest.json").open("xb") as stream:
        raw = Path(manifest).read_bytes()
        scan.require(hashlib.sha256(raw).hexdigest() == plan["manifest_sha256"],
                     "manifest changed during preparation")
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    (root / "units").mkdir(mode=0o700)
    for unit in plan["units"]:
        prepare(root / "manifest.json", unit["unit_id"], root / "units" / unit["unit_id"],
                wrapper, python, request_memory_mb=request_memory_mb, max_runtime=max_runtime)
    durable.exclusive_json(root / "campaign.json", campaign)
    return read_campaign(root)


def _submission_state(path, descriptor, campaign):
    """Local evidence only. Contradictions/partial attempts never become eligibility."""
    metadata = path.parent
    names = ("campaign_submit_intent.json", "submission_intent.json", "submission_response.json",
             "scheduler_identity.json", "worker_started.json", "worker_receipt.json")
    present = {name for name in names if (metadata / name).exists() or
               (metadata / name).is_symlink()}
    if not present:
        science = Path(descriptor["attempt_root"]) / "science"
        scan.require(not science.exists() and not science.is_symlink(),
                     "science output without submission evidence")
        return "never_attempted"
    required = set(names[:4])
    scan.require(required <= present, "ambiguous submission; no resubmit")
    claim = _campaign_json(metadata / names[0])
    scan.require(claim == dict(campaign_token=campaign["token"], token=descriptor["token"]),
                 "campaign submit identity mismatch")
    intent = scheduler._read_intent(metadata / names[1], descriptor["token"])
    identity = scheduler._read_identity(metadata / names[3], intent)
    response = _campaign_json(metadata / names[2])
    match = re.fullmatch(r"([0-9]+)\.0(?:\s*-\s*\1\.0)?\s*", response["stdout"])
    scan.require(response["returncode"] == 0 and match is not None and
                 match[1] + ".0" == identity["job"], "submission response contradiction")
    for name in names[4:]:
        if name in present:
            receipt = _campaign_json(metadata / name)
            scan.require(receipt["token"] == descriptor["token"] and
                         receipt["job"] == identity["job"], "worker identity contradiction")
            if name == "worker_receipt.json":
                scan.require(receipt.get("status") in ("COMPLETE", "STOPPED_NO_RETRY"),
                             "worker receipt status contradiction")
    return "submitted"


def submit_manifest(campaign_root):
    """Serialize controllers; call the canonical submit once per never-attempted unit."""
    root = Path(campaign_root).resolve(strict=True)
    # NFS exclusive flock requires a descriptor opened for writing.
    with (root / "campaign.json").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = read_campaign(root)
        executions = []
        # Check the entire campaign before any new effect.
        for unit_id in campaign["unit_ids"]:
            path = root / "units" / unit_id / "_metadata/execution.json"
            descriptor = read_descriptor(path)
            executions.append((path, descriptor, _submission_state(path, descriptor, campaign)))
        results = []
        for path, descriptor, state in executions:
            if state == "never_attempted":
                durable.exclusive_json(path.parent / "campaign_submit_intent.json",
                    dict(campaign_token=campaign["token"], token=descriptor["token"]))
                result = submit(path)
                scan.require(_submission_state(path, descriptor, campaign) == "submitted",
                             "submission evidence incomplete")
            else:
                result = dict(state="previously_submitted")
            results.append(dict(unit_id=descriptor["unit"]["unit_id"], submission=result))
        return dict(campaign_token=campaign["token"], units=results)


def status_manifest(campaign_root):
    """Read-only observation; preserve scheduler ads and worker receipts verbatim."""
    campaign = read_campaign(campaign_root)
    results = []
    for unit_id in campaign["unit_ids"]:
        path = Path(campaign["campaign_root"]) / "units" / unit_id / "_metadata/execution.json"
        descriptor = read_descriptor(path)
        try:
            local = _submission_state(path, descriptor, campaign)
        except (ValueError, KeyError, OSError) as exc:
            local = dict(state="unresolved", reason=str(exc))
        try:
            observation = scheduler.observe(path, descriptor, read_only=True)
        except (durable.ActiveChildUnresolved, OSError, ValueError) as exc:
            observation = dict(state="unresolved", reason=str(exc))
        receipt = path.parent / "worker_receipt.json"
        try:
            worker_receipt = (_campaign_json(receipt) if receipt.exists() or receipt.is_symlink()
                              else None)
        except (ValueError, OSError) as exc:
            worker_receipt = dict(state="unresolved", reason=str(exc))
        results.append(dict(unit_id=unit_id, local_submission=local, scheduler=observation,
                            worker_receipt=worker_receipt))
    return dict(campaign_token=campaign["token"], units=results)


def read_descriptor(path):
    path = Path(path).resolve(strict=True)
    descriptor = json.loads(path.read_text())
    body = {k: v for k, v in descriptor.items() if k != "token"}
    scan.require(descriptor["token"] == durable.identity(body), "execution descriptor changed")
    scan.require(path == Path(descriptor["attempt_root"]) / "_metadata/execution.json",
                 "execution descriptor path changed")
    scan.require(scan.native.sha256(descriptor["manifest"]) == descriptor["manifest_sha256"] and
                 scan.select_unit(descriptor["manifest"], descriptor["unit"]["unit_id"]) ==
                 descriptor["unit"], "execution manifest changed")
    for name, digest in descriptor["code_sha256"].items():
        scan.require(scan.native.sha256(name) == digest, "execution code changed: " + name)
    return descriptor


def prepare(manifest, unit_id, output_dir, wrapper, python, *, request_memory_mb=8192,
            max_runtime=3600):
    unit = scan.select_unit(manifest, unit_id)
    scan.validate_unit(unit, revalidate_source=True)
    scan.require(type(request_memory_mb) is int and request_memory_mb > 0 and
                 type(max_runtime) is int and 0 < max_runtime <= 108000, "invalid resource bound")
    wrapper, python = (str(Path(p).absolute()) for p in (wrapper, python))
    scan.require(all(Path(p).is_file() and os.access(p, os.X_OK) for p in (wrapper, python)),
                 "runtime not executable")
    root = Path(output_dir).resolve()
    scan.require(not root.exists() and not root.is_symlink(), "execution namespace occupied")
    code = [Path(__file__).resolve(), Path(scan.__file__).resolve(),
            Path(scheduler.__file__).resolve(), Path(durable.__file__).resolve(), Path(wrapper)]
    body = dict(version=1, manifest=str(Path(manifest).resolve()),
        manifest_sha256=scan.native.sha256(manifest), unit=unit, wrapper=wrapper, python=python,
        attempt_root=str(root), request_memory_mb=request_memory_mb, max_runtime=max_runtime,
        code_sha256={str(p): scan.native.sha256(p) for p in code})
    descriptor = dict(body, token=durable.identity(body))
    root.mkdir(mode=0o700)
    metadata = root / "_metadata"
    metadata.mkdir(mode=0o700)
    path = metadata / "execution.json"
    durable.exclusive_json(path, descriptor)
    with (metadata / "execution.sub").open("x") as stream:
        stream.write(submit_description(path, descriptor))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def submit_description(path, descriptor):
    command = ['/bin/bash', '--noprofile', '--norc', '-c',
               'cd "$1" && exec "$2" -m analysis.topeft_run2.event_scan_condor --worker "$3"',
               'event_scan_worker', str(REPO), descriptor["python"], str(path)]
    # Same HTCondor argument codec as the maintained per-execution helper.
    arguments = '"' + ' '.join("'" + a.replace('"', '""').replace("'", "''") + "'" for a in command) + '"'
    metadata = Path(path).parent
    lines = ['universe = vanilla', 'executable = ' + scheduler._filename(descriptor["wrapper"]),
        'arguments = ' + arguments, 'initialdir = ' + scheduler._filename(REPO),
        'getenv = False', 'environment = "PATH=/usr/bin:/bin"', 'should_transfer_files = NO',
        'request_cpus = 1', 'request_memory = ' + str(descriptor["request_memory_mb"]),
        'request_disk = 1024MB', 'allowed_execute_duration = ' + str(descriptor["max_runtime"]),
        '+PlotExecutionToken = ' + scheduler._quoted(descriptor["token"]),
        'max_retries = 0', 'on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)',
        'periodic_hold = (NumJobStarts > 1)',
        'output = ' + scheduler._filename(metadata / 'worker.out'),
        'error = ' + scheduler._filename(metadata / 'worker.err'),
        'log = ' + scheduler._filename(metadata / 'scheduler.log'), 'queue 1', '']
    return "\n".join(lines)


def submit(path):
    path = Path(path).resolve(strict=True)
    descriptor = read_descriptor(path)
    metadata = path.parent
    scan.require(all(not (metadata / name).exists() and not (metadata / name).is_symlink()
                     for name in ("submission_intent.json", "scheduler_identity.json",
                                  "worker_started.json")), "existing execution; no resubmit")
    scan.require((metadata / "execution.sub").read_text() == submit_description(path, descriptor),
                 "submit description changed")
    scan.validate_unit(descriptor["unit"], revalidate_source=True)
    target = scheduler.resolve_target()
    intent = dict(token=descriptor["token"], **target)
    durable.exclusive_json(metadata / "submission_intent.json", intent)
    # Any ambiguity is handled by observation, never by another submit.
    return scheduler._submit_once(metadata / "execution.sub", intent,
                                  metadata / "submission_response.json",
                                  metadata / "scheduler_identity.json")


def worker_identity(token):
    path = os.environ.get("_CONDOR_JOB_AD")
    scan.require(path is not None, "missing worker scheduler identity")
    text = Path(path).read_text()
    values = {}
    for name, pattern in (("ClusterId", r"([0-9]+)"), ("ProcId", r"([0-9]+)"),
                          ("PlotExecutionToken", r'"([a-f0-9]{64})"')):
        matches = re.findall(r"(?mi)^" + name + r"[ \t]*=[ \t]*" + pattern + r"[ \t]*$", text)
        scan.require(len(matches) == 1, "ambiguous worker scheduler attribute: " + name)
        values[name] = matches[0]
    scan.require(values["PlotExecutionToken"] == token and values["ProcId"] == "0",
                 "worker token/job contradiction")
    return values["ClusterId"] + ".0"


def worker(path):
    path = Path(path).resolve(strict=True)
    descriptor = read_descriptor(path)
    job = worker_identity(descriptor["token"])
    durable.exclusive_json(path.parent / "worker_started.json",
                           dict(token=descriptor["token"], job=job))
    # One exclusive marker fences evictions/restarts before scientific execution.
    try:
        receipt = scan.run_unit(descriptor["unit"], Path(descriptor["attempt_root"]) / "science",
                                time_limit=descriptor["max_runtime"])
        result = dict(token=descriptor["token"], job=job, status=receipt["status"],
            scientific_receipt_sha256=scan.native.sha256(
                Path(descriptor["attempt_root"]) / "science/receipt.json"))
    except Exception as exc:
        result = dict(token=descriptor["token"], job=job, status="STOPPED_NO_RETRY",
                      error=dict(type=type(exc).__name__, message=str(exc)))
    durable.exclusive_json(path.parent / "worker_receipt.json", result)
    return 0 if result["status"] == "COMPLETE" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    actions = parser.add_subparsers(dest="action")
    plan = actions.add_parser("plan-manifest")
    plan.add_argument("--manifest", type=Path, required=True)
    campaign_make = actions.add_parser("prepare-manifest")
    for name in ("manifest", "campaign-root", "wrapper", "python"):
        campaign_make.add_argument("--" + name, type=Path, required=True)
    campaign_make.add_argument("--request-memory-mb", type=int, default=8192)
    campaign_make.add_argument("--max-runtime", type=int, default=3600)
    for name in ("submit-manifest", "status-manifest"):
        command = actions.add_parser(name)
        command.add_argument("--campaign-root", type=Path, required=True)
    make = actions.add_parser("prepare")
    for name in ("manifest", "output-dir", "wrapper", "python"):
        make.add_argument("--"+name, type=Path, required=True)
    make.add_argument("--unit-id", required=True)
    make.add_argument("--request-memory-mb", type=int, default=8192)
    make.add_argument("--max-runtime", type=int, default=3600)
    for name in ("submit", "status"):
        command = actions.add_parser(name)
        command.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args(argv)
    scan.require(bool(args.worker) != bool(args.action), "choose worker or one controller action")
    if args.worker:
        return worker(args.worker)
    if args.action == "plan-manifest":
        print(json.dumps(plan_manifest(args.manifest), sort_keys=True))
    elif args.action == "prepare-manifest":
        print(json.dumps(prepare_manifest(args.manifest, args.campaign_root, args.wrapper,
            args.python, request_memory_mb=args.request_memory_mb, max_runtime=args.max_runtime)))
    elif args.action == "submit-manifest":
        print(json.dumps(submit_manifest(args.campaign_root)))
    elif args.action == "status-manifest":
        print(json.dumps(status_manifest(args.campaign_root), sort_keys=True))
    elif args.action == "prepare":
        path = prepare(args.manifest, args.unit_id, args.output_dir, args.wrapper, args.python,
                       request_memory_mb=args.request_memory_mb, max_runtime=args.max_runtime)
        print(json.dumps(dict(execution=str(path), submitted=False)))
    elif args.action == "submit":
        print(json.dumps(submit(args.execution)))
    else:
        descriptor = read_descriptor(args.execution)
        print(json.dumps(scheduler.observe(args.execution.resolve(), descriptor), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
