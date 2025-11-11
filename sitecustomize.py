"""Test environment helpers."""
import getpass
import os

# Ensure tests that expect a USER variable have a sensible default in CI.
# getpass.getuser() falls back to LOGNAME/USER/..., but may raise in
# containerized environments. Using os.getuid provides a stable suffix.
_default_user = f"user{os.getuid()}"
os.environ.setdefault("USER", getpass.getuser() if os.environ.get("USER") else _default_user)
