"""Owned manager staging, free-space warnings, and post-success cleanup."""

import json
import argparse
from datetime import datetime, timezone
import math
import multiprocessing
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time


class staging_guard_error(RuntimeError):
    pass


def _process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None
    except (OSError, IndexError) as error:
        raise staging_guard_error("manager process identity cannot be read") from error


def _quota_available(quota_root):
    if not quota_root or not quota_root.startswith(("/groups/", "/users/")):
        raise staging_guard_error("an exact supported quota root is required")
    try:
        result = subprocess.run(
            ["quota", quota_root], capture_output=True, text=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise staging_guard_error("quota headroom could not be measured") from error
    if result.returncode or len(result.stdout.splitlines()) != 2:
        raise staging_guard_error("quota headroom could not be measured")
    fields = result.stdout.splitlines()[1].split()
    if len(fields) != 5 or fields[0] != quota_root:
        raise staging_guard_error("quota result does not match the declared root")
    value = fields[4]
    # The site prints rounded human units. Decimal multipliers provide a
    # conservative lower estimate if its display uses binary units.
    suffixes = {"K": 1000, "M": 1000 ** 2, "G": 1000 ** 3, "T": 1000 ** 4}
    try:
        if value[-1:] in suffixes:
            return int(float(value[:-1]) * suffixes[value[-1]])
        return int(value)
    except ValueError as error:
        raise staging_guard_error("quota headroom is not numeric") from error


def _space(base, quota_root):
    values = os.statvfs(base)
    free_bytes = values.f_bavail * values.f_frsize
    free_inodes = values.f_favail
    if quota_root is None:
        return free_bytes, free_inodes
    quota_bytes = _quota_available(quota_root)
    return min(free_bytes, quota_bytes), free_inodes


def _validate_base(base):
    path = Path(base)
    if not path.is_absolute() or not path.is_dir():
        raise staging_guard_error("staging base must be an existing absolute directory")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise staging_guard_error("staging path has a symlink component")
    real = path.resolve(strict=True)
    if real == Path("/tmp") or real == Path("/var/tmp") or any(
        parent in (Path("/tmp"), Path("/var/tmp")) for parent in real.parents
    ):
        raise staging_guard_error("root temporary staging is forbidden")
    if os.stat(real).st_uid != os.getuid() or not os.access(real, os.W_OK | os.X_OK):
        raise staging_guard_error("staging base ownership or write access is invalid")
    mount = real
    while mount.parent != mount and os.stat(mount.parent).st_dev == os.stat(mount).st_dev:
        mount = mount.parent
    if mount == Path("/"):
        raise staging_guard_error("root-backed staging is forbidden")
    return real


def _local_readers(path, max_processes=32768):
    scanned = 0
    prefix = str(path) + os.sep
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        scanned += 1
        if scanned > max_processes:
            raise staging_guard_error("local reader scan limit exceeded")
        try:
            if item.stat().st_uid != os.getuid():
                continue
            for descriptor in (item / "fd").iterdir():
                target = os.readlink(descriptor)
                if target == str(path) or target.startswith(prefix):
                    return True
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise staging_guard_error("local reader state is inaccessible") from error
    return False


def _verify_owned_tree(path, marker, owner_pid, owner_start, owner_inode, marker_inode):
    root_info = os.lstat(path)
    marker_info = os.lstat(marker)
    if (not stat.S_ISDIR(root_info.st_mode)
            or stat.S_ISLNK(root_info.st_mode)
            or root_info.st_ino != owner_inode
            or root_info.st_uid != os.getuid()
            or not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_ino != marker_inode
            or marker_info.st_uid != os.getuid()):
        raise staging_guard_error("staging root or marker identity changed")
    metadata = json.loads(marker.read_text())
    if not isinstance(metadata, dict):
        raise staging_guard_error("staging owner marker is not an object")
    if (metadata.get("path") != os.fspath(path)
            or metadata.get("owner_inode") != owner_inode
            or metadata.get("owner_uid") != os.getuid()
            or metadata.get("owner_pid") != owner_pid
            or metadata.get("owner_start") != owner_start
            or metadata.get("status") != "running"):
        raise staging_guard_error("staging owner marker changed")


def _critical_signal_identity(path, marker, owner_pid, owner_start,
                              owner_inode, marker_inode, telemetry_degraded):
    try:
        _verify_owned_tree(path, marker, owner_pid, owner_start,
                           owner_inode, marker_inode)
    except OSError as error:
        if telemetry_degraded is not None:
            telemetry_degraded.set()
        print(f"WARNING: Work Queue critical-space signal deferred because "
              f"staging identity could not be read: {type(error).__name__}: "
              f"{str(error)[:240]}", flush=True)
        return False
    try:
        observed_start = _process_start(owner_pid)
    except staging_guard_error as error:
        if telemetry_degraded is not None:
            telemetry_degraded.set()
        print(f"WARNING: Work Queue critical-space signal deferred because "
              f"manager identity could not be read: {str(error)[:240]}", flush=True)
        return False
    if observed_start != owner_start or os.getppid() != owner_pid:
        raise staging_guard_error(
            "manager process identity changed before critical-space signal")
    return True


def _format_decimal_gb(byte_count):
    whole_gb, remaining_bytes = divmod(byte_count, 1_000_000_000)
    fractional_milli_gb = remaining_bytes // 1_000_000
    return f"{whole_gb}.{fractional_milli_gb:03d} GB"


def _watch(path, base, quota_root, owner_pid, owner_start, warning_bytes,
           min_free_bytes,
           poll_seconds, stop_event, owner_inode,
           marker_inode, telemetry_degraded=None, shutdown_ack=None):
    marker = Path(path) / "guard.json"
    critical_signal_sent = False
    last_health_state = None
    active_degraded_reasons = set()
    while not stop_event.is_set():
        if os.getppid() != owner_pid:
            raise staging_guard_error("manager parent exited during monitoring")
        identity_detail = "process entry absent"
        try:
            observed_start = _process_start(owner_pid)
        except staging_guard_error as error:
            observed_start = None
            identity_detail = str(error)
        if observed_start is not None and observed_start != owner_start:
            raise staging_guard_error("manager process identity changed during monitoring")
        current_identity_verified = observed_start == owner_start
        if observed_start is None:
            if telemetry_degraded is not None:
                telemetry_degraded.set()
            if "manager_identity" not in active_degraded_reasons:
                print(f"WARNING: Work Queue manager identity telemetry unavailable: "
                      f"{identity_detail}; monitoring continues", flush=True)
            active_degraded_reasons.add("manager_identity")
        else:
            active_degraded_reasons.discard("manager_identity")
        try:
            _verify_owned_tree(path, marker, owner_pid, owner_start, owner_inode, marker_inode)
        except OSError as error:
            if telemetry_degraded is not None:
                telemetry_degraded.set()
            if "staging_identity" not in active_degraded_reasons:
                print(f"WARNING: Work Queue staging identity read unavailable: "
                      f"{type(error).__name__}: {str(error)[:240]}; monitoring continues",
                      flush=True)
            active_degraded_reasons.add("staging_identity")
            stop_event.wait(poll_seconds)
            continue
        active_degraded_reasons.discard("staging_identity")
        measurement = None
        try:
            measurement = _space(base, quota_root)
        except Exception as error:
            if telemetry_degraded is not None:
                telemetry_degraded.set()
            if "capacity_telemetry" not in active_degraded_reasons:
                print(f"WARNING: Work Queue staging telemetry unavailable: "
                      f"{type(error).__name__}: {str(error)[:240]}; manager remains running",
                      flush=True)
            active_degraded_reasons.add("capacity_telemetry")
        if measurement is not None:
            active_degraded_reasons.discard("capacity_telemetry")
            available_bytes, free_inodes = measurement
            if available_bytes <= min_free_bytes:
                health_state = "critical"
            elif available_bytes < warning_bytes or free_inodes < 1000:
                health_state = "warning"
            else:
                health_state = "healthy"

            if health_state == "critical" and last_health_state != "critical":
                print(
                    "CRITICAL: Work Queue staging effective available space "
                    f"{_format_decimal_gb(available_bytes)} is at or below "
                    f"{_format_decimal_gb(min_free_bytes)}; verifying manager "
                    "identity before SIGTERM",
                    flush=True,
                )
            elif health_state == "warning" and last_health_state != "warning":
                print(
                    "WARNING: Work Queue staging free space or inodes are low; "
                    f"available={_format_decimal_gb(available_bytes)} "
                    f"free_inodes={free_inodes}; manager remains running",
                    flush=True,
                )
            elif health_state == "healthy" and last_health_state == "warning":
                print(
                    "RECOVERY: Work Queue staging free space and inodes recovered; "
                    f"available={_format_decimal_gb(available_bytes)} "
                    f"free_inodes={free_inodes}",
                    flush=True,
                )
            last_health_state = health_state

            if available_bytes <= min_free_bytes and not critical_signal_sent:
                try:
                    safe_identity = current_identity_verified
                    if safe_identity:
                        safe_identity = _critical_signal_identity(
                            path, marker, owner_pid, owner_start, owner_inode,
                            marker_inode, telemetry_degraded)
                    if safe_identity:
                        try:
                            metadata = json.loads(marker.read_text())
                        except OSError as error:
                            if telemetry_degraded is not None:
                                telemetry_degraded.set()
                            print(f"WARNING: Work Queue critical-space signal "
                                  f"deferred because guard marker could not be "
                                  f"read: {type(error).__name__}: "
                                  f"{str(error)[:240]}", flush=True)
                            safe_identity = False
                    if safe_identity:
                        observed_at = datetime.now(timezone.utc).isoformat()
                        metadata.update(
                            reason="critical_free_space",
                            critical_free_space={
                                "available_bytes": available_bytes,
                                "min_free_bytes": min_free_bytes,
                                "observed_at": observed_at,
                                "observed_at_unix_ns": time.time_ns(),
                            },
                        )
                        with marker.open("w") as marker_file:
                            json.dump(metadata, marker_file, sort_keys=True)
                            marker_file.write("\n")
                            marker_file.flush()
                            os.fsync(marker_file.fileno())
                        safe_identity = _critical_signal_identity(
                            path, marker, owner_pid, owner_start, owner_inode,
                            marker_inode, telemetry_degraded)
                    if safe_identity:
                        os.kill(owner_pid, signal.SIGTERM)
                        critical_signal_sent = True
                except (OSError, ValueError, staging_guard_error) as error:
                    raise staging_guard_error(
                        "critical free-space signal was not safely completed") from error
        try:
            _verify_owned_tree(path, marker, owner_pid, owner_start, owner_inode, marker_inode)
        except OSError as error:
            if telemetry_degraded is not None:
                telemetry_degraded.set()
            if "staging_identity" not in active_degraded_reasons:
                print(f"WARNING: Work Queue staging identity read unavailable: "
                      f"{type(error).__name__}: {str(error)[:240]}; monitoring continues",
                      flush=True)
            active_degraded_reasons.add("staging_identity")
        else:
            active_degraded_reasons.discard("staging_identity")
        stop_event.wait(poll_seconds)
    if shutdown_ack is not None:
        shutdown_ack.set()


def start_guard(base, invocation, quota_root, warning_bytes, min_free_bytes,
                poll_seconds):
    if not invocation or invocation in (".", "..") or Path(invocation).name != invocation:
        raise staging_guard_error("invocation must be one path component")
    if not (10000000000 <= min_free_bytes <= warning_bytes
            and math.isfinite(poll_seconds) and poll_seconds > 0):
        raise staging_guard_error("invalid free-space warning/minimum/poll configuration")
    real_base = _validate_base(base)
    if quota_root is not None:
        if not quota_root or not quota_root.startswith(("/groups/", "/users/")):
            raise staging_guard_error("an exact site quota root is required")
        if os.path.commonpath((str(real_base), quota_root)) != quota_root:
            raise staging_guard_error("quota root does not own the selected staging base")
    available, free_inodes = _space(real_base, quota_root)
    if available <= min_free_bytes or free_inodes < 1000:
        raise staging_guard_error("minimum free space or protective inode reserve unavailable")
    path = real_base / invocation
    path.mkdir(mode=0o700)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        path.rmdir()
        raise staging_guard_error("invocation staging mode is not 0700")
    owner_inode = path.stat().st_ino
    owner_pid = os.getpid()
    owner_start = _process_start(owner_pid)
    if owner_start is None:
        raise staging_guard_error("manager process identity unavailable")
    marker = path / "guard.json"
    marker.write_text(json.dumps({"status": "running", "path": str(path),
                                  "owner_uid": os.getuid(), "owner_pid": owner_pid,
                                  "owner_start": owner_start, "owner_inode": owner_inode,
                                  "invocation": invocation}) + "\n")
    marker_inode = marker.stat().st_ino
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    telemetry_degraded = context.Event()
    shutdown_ack = context.Event()
    watchdog = context.Process(target=_watch, args=(str(path), str(real_base), quota_root,
        owner_pid, owner_start, warning_bytes, min_free_bytes,
        poll_seconds, stop_event, owner_inode, marker_inode, telemetry_degraded,
        shutdown_ack), daemon=False)
    watchdog.telemetry_degraded = telemetry_degraded
    watchdog.shutdown_ack = shutdown_ack
    watchdog.marker_inode = marker_inode
    watchdog.owner_pid = owner_pid
    watchdog.owner_start = owner_start
    try:
        watchdog.start()
    except Exception as error:
        marker.write_text(json.dumps({"status": "cleanup_needed", "reason": "watchdog_start_failed",
                                      "path": str(path), "owner_inode": owner_inode,
                                      "owner_uid": os.getuid(), "owner_pid": owner_pid,
                                      "owner_start": owner_start}) + "\n")
        raise staging_guard_error("watchdog failed before Work Queue startup") from error
    if not watchdog.is_alive():
        raise staging_guard_error("watchdog failed before Work Queue startup")

    return path, owner_inode, stop_event, watchdog


def finish_guard(path, owner_inode, stop_event, watchdog, succeeded):
    stop_event.set()
    watchdog.join(timeout=30)
    marker = path / "guard.json"
    try:
        metadata = json.loads(marker.read_text())
        if not isinstance(metadata, dict):
            raise staging_guard_error("staging owner marker is not an object")
        _verify_owned_tree(path, marker,
                           getattr(watchdog, "owner_pid", metadata["owner_pid"]),
                           getattr(watchdog, "owner_start", metadata["owner_start"]),
                           owner_inode, getattr(watchdog, "marker_inode", marker.stat().st_ino))
    except (OSError, ValueError, KeyError, staging_guard_error) as error:
        raise staging_guard_error(f"staging ownership marker invalid; cleanup_needed: {error}") from error
    if watchdog.is_alive() or watchdog.exitcode != 0:
        raise staging_guard_error("watchdog exited or did not stop cleanly; cleanup_needed")
    if getattr(watchdog, "shutdown_ack", None) is not None and not watchdog.shutdown_ack.is_set():
        raise staging_guard_error("watchdog exited before requested shutdown; cleanup_needed")
    metadata["telemetry_degraded"] = (getattr(watchdog, "telemetry_degraded", None) is not None
                                      and watchdog.telemetry_degraded.is_set())
    if metadata["telemetry_degraded"]:
        print("WARNING: Work Queue staging telemetry degraded during this invocation", flush=True)
    if not succeeded:
        metadata.update(status="cleanup_needed", reason="manager_failed")
        marker.write_text(json.dumps(metadata) + "\n")
        return
    metadata.update(status="completed_pending_receipt")
    marker.write_text(json.dumps(metadata) + "\n")


def closeout_invocation_staging(base, path, owner_inode):
    """Remove transient files from this finished invocation, keeping PKLs and sidecars."""
    base = Path(base)
    path = Path(path)
    if (not base.is_absolute() or not path.is_absolute()
            or path.parent != base or path.name in ("", ".", "..")):
        raise staging_guard_error("closeout path is not one invocation inside its base")
    for component in (base, *base.parents):
        if component.is_symlink():
            raise staging_guard_error("closeout base has a symlink component")
    if not base.is_dir() or base.resolve(strict=True) != base:
        raise staging_guard_error("closeout base identity changed")
    if not os.path.lexists(path):
        return  # Already closed; never widen the scope to the base.
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_ino != owner_inode or info.st_dev != base.stat().st_dev):
        raise staging_guard_error("closeout invocation identity changed")
    marker = path / "guard.json"
    if os.path.lexists(marker):
        marker_info = marker.lstat()
        if (not stat.S_ISREG(marker_info.st_mode)
                or marker_info.st_uid != os.getuid() or marker_info.st_nlink != 1):
            raise staging_guard_error("closeout marker identity changed")
        try:
            metadata = json.loads(marker.read_text())
        except (OSError, ValueError) as error:
            raise staging_guard_error("closeout marker is unreadable") from error
        if (not isinstance(metadata, dict) or metadata.get("path") != str(path)
                or metadata.get("owner_inode") != owner_inode
                or metadata.get("owner_uid") != os.getuid()
                or metadata.get("owner_pid") != os.getpid()
                or metadata.get("owner_start") != _process_start(os.getpid())
                or metadata.get("status") not in
                ("completed_pending_receipt", "cleanup_needed")
                or (metadata.get("status") == "cleanup_needed"
                    and metadata.get("reason") != "manager_failed")):
            raise staging_guard_error("closeout marker does not identify this invocation")
    if _local_readers(path):
        raise staging_guard_error("local staging reader remains active")

    root_device = info.st_dev

    def remove_transients(directory_fd, *, root=False):
        preserved = False
        with os.scandir(directory_fd) as entries:
            names = [entry.name for entry in entries]
        for name in names:
            if root and name == "guard.json":
                continue
            entry_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if entry_info.st_uid != os.getuid() or entry_info.st_dev != root_device:
                raise staging_guard_error("foreign staging entry blocks closeout")
            if stat.S_ISDIR(entry_info.st_mode):
                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=directory_fd)
                try:
                    if os.fstat(child_fd).st_ino != entry_info.st_ino:
                        raise staging_guard_error("staging directory changed during closeout")
                    child_preserved = remove_transients(child_fd)
                finally:
                    os.close(child_fd)
                if child_preserved:
                    preserved = True
                else:
                    os.rmdir(name, dir_fd=directory_fd)
            elif stat.S_ISREG(entry_info.st_mode):
                is_pkl = name.endswith((".pkl", ".pkl.gz"))
                is_sidecar = name.endswith((".pkl.metadata.json", ".pkl.gz.metadata.json"))
                if is_sidecar:
                    pkl_name = name.removesuffix(".metadata.json")
                    try:
                        pkl_info = os.stat(pkl_name, dir_fd=directory_fd,
                                           follow_symlinks=False)
                    except FileNotFoundError:
                        pkl_info = None
                    is_sidecar = (pkl_info is not None
                                  and stat.S_ISREG(pkl_info.st_mode)
                                  and pkl_info.st_uid == os.getuid())
                if is_pkl or is_sidecar:
                    preserved = True
                else:
                    os.unlink(name, dir_fd=directory_fd)
            elif stat.S_ISLNK(entry_info.st_mode):
                os.unlink(name, dir_fd=directory_fd)  # Never follow an escape.
            else:
                raise staging_guard_error("special staging entry blocks closeout")
        return preserved

    root_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if os.fstat(root_fd).st_ino != owner_inode:
            raise staging_guard_error("closeout invocation changed before traversal")
        preserved = remove_transients(root_fd, root=True)
        if os.path.lexists(marker):
            os.unlink("guard.json", dir_fd=root_fd)
    finally:
        os.close(root_fd)
    if not preserved:
        path.rmdir()


def cleanup_owned_staging(path):
    """Called only after campaign state and native logs record source completion."""
    path = Path(path)
    marker = path / "guard.json"
    if not os.path.lexists(path):
        return  # run_analysis already closed the exact invocation.
    try:
        metadata = json.loads(marker.read_text())
        if not isinstance(metadata, dict):
            raise staging_guard_error("staging owner marker is not an object")
        info = path.lstat()
        marker_info = marker.lstat()
    except (OSError, ValueError) as error:
        raise staging_guard_error("staging ownership evidence unavailable") from error
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or not stat.S_ISREG(marker_info.st_mode) or marker_info.st_uid != os.getuid()
            or marker_info.st_nlink != 1
            or info.st_ino != metadata.get("owner_inode")
            or metadata.get("owner_uid") != os.getuid()
            or metadata.get("path") != str(path)
            or metadata.get("status") != "completed_pending_receipt"
            or not isinstance(metadata.get("owner_pid"), int)
            or not isinstance(metadata.get("owner_start"), str)):
        raise staging_guard_error("staging owner/path/status does not match the completed invocation")
    for component in path.parents:
        if component.is_symlink():
            raise staging_guard_error("staging parent gained a symlink; cleanup_needed")
    try:
        owner_start = _process_start(metadata.get("owner_pid"))
    except staging_guard_error:
        metadata.update(status="cleanup_needed", reason="manager_liveness_unknown")
        marker.write_text(json.dumps(metadata) + "\n")
        raise
    if owner_start == metadata.get("owner_start"):
        metadata.update(status="cleanup_needed", reason="manager_still_live")
        marker.write_text(json.dumps(metadata) + "\n")
        raise staging_guard_error("manager process is still live")
    try:
        readers_active = _local_readers(path)
    except staging_guard_error as error:
        metadata.update(status="cleanup_needed", reason="local_reader_state_unknown")
        marker.write_text(json.dumps(metadata) + "\n")
        raise staging_guard_error("local staging reader state is unknown") from error
    if readers_active:
        metadata.update(status="cleanup_needed", reason="local_reader_active")
        marker.write_text(json.dumps(metadata) + "\n")
        raise staging_guard_error("local staging reader remains active")
    def reject_cleanup(reason):
        metadata.update(status="cleanup_needed", reason=reason)
        marker.write_text(json.dumps(metadata) + "\n")
        raise staging_guard_error(f"{reason}; cleanup_needed")

    # Coffea normally removes its TemporaryDirectory on exit. If it remains,
    # remove only its recognized, owned regular files after checking the whole
    # directory. Unknown entries keep the invocation available for review.
    children = [child for child in path.iterdir() if child.name != "guard.json"]
    if children:
        if (len(children) != 1 or not re.fullmatch(r"tmp[a-z0-9_]+wq-tmp-", children[0].name)):
            reject_cleanup("unexpected owned staging entry")
        temporary = children[0]
        temporary_info = temporary.lstat()
        if (not stat.S_ISDIR(temporary_info.st_mode) or temporary_info.st_uid != os.getuid()
                or stat.S_IMODE(temporary_info.st_mode) != 0o700):
            reject_cleanup("Coffea temporary directory identity is uncertain")
        def recognized_owned_file(entry):
            info = entry.stat(follow_symlinks=False)
            name = entry.name
            recognized = (re.fullmatch(r"(?:args_|out_|stdout_)[A-Za-z0-9_]+\.p", name)
                          or re.fullmatch(r"fn_wrapper[a-z0-9_]+", name)
                          or re.fullmatch(r"(?:preproc|proc|accum)[a-z0-9_]+\.p", name))
            return (recognized and stat.S_ISREG(info.st_mode)
                    and info.st_uid == os.getuid() and info.st_nlink == 1)

        # Validate the whole directory before removing anything. Repeat the
        # recognition check during deletion without retaining a file list.
        with os.scandir(temporary) as entries:
            for entry in entries:
                if not recognized_owned_file(entry):
                    reject_cleanup("unknown or foreign Coffea staging file")
        if _local_readers(temporary):
            reject_cleanup("local Coffea staging reader remains active")
        with os.scandir(temporary) as entries:
            for entry in entries:
                if not recognized_owned_file(entry):
                    reject_cleanup("Coffea staging file identity changed")
                Path(entry.path).unlink()
        if temporary.lstat().st_ino != temporary_info.st_ino:
            reject_cleanup("Coffea temporary directory identity changed")
        temporary.rmdir()
    marker.unlink()
    path.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize one completed owned Work Queue staging directory")
    parser.add_argument("action", choices=("cleanup",))
    parser.add_argument("path")
    arguments = parser.parse_args()
    try:
        cleanup_owned_staging(arguments.path)
    except staging_guard_error as error:
        parser.exit(1, f"cleanup_needed: {error}\n")
