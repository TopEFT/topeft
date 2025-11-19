"""Test environment helpers."""
import getpass
import os

from topeft.compat.topcoffea import ensure_histEFT_py39_compat

# Ensure tests that expect a USER variable have a sensible default in CI.
# getpass.getuser() falls back to LOGNAME/USER/..., but may raise in
# containerized environments. Using os.getuid provides a stable suffix.
_default_user = f"user{os.getuid()}"
os.environ.setdefault("USER", getpass.getuser() if os.environ.get("USER") else _default_user)

# Ensure ``topcoffea.modules.histEFT`` is patched for Python 3.9 before any
# modules import it at module scope.
ensure_histEFT_py39_compat()
