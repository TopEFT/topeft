"""Small event-scan adapter to the existing per-execution Condor protocol.

Target resolution, submit ambiguity, identity and reconciliation remain owned
by plot_campaign_condor. Its plot-specific descriptor/worker is not reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from analysis.topeft_run2 import event_scan as scan
from analysis.topeft_run2 import plot_campaign_backend as durable
from analysis.topeft_run2 import plot_campaign_condor as scheduler

REPO = Path(__file__).resolve().parents[2]


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
    if args.action == "prepare":
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
