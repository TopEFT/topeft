#!/usr/bin/env bash
set -euo pipefail

# Install topcoffea and topeft together from sibling checkouts.
pip install -e ../topcoffea -e .
