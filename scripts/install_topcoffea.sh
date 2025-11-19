#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_DIR="${REPO_ROOT}/external/topcoffea"

TOPCOFFEA_REPO_URL=${TOPCOFFEA_REPO_URL:-https://github.com/TopEFT/topcoffea.git}
TOPCOFFEA_GIT_REF=${TOPCOFFEA_GIT_REF:-}
TOPCOFFEA_DIR=${TOPCOFFEA_DIR:-${DEFAULT_DIR}}
PARENT_DIR=$(dirname "${TOPCOFFEA_DIR}")

mkdir -p "${PARENT_DIR}"

if [[ -d "${TOPCOFFEA_DIR}/.git" ]]; then
    echo "[install_topcoffea] Updating existing checkout at ${TOPCOFFEA_DIR}" >&2
    git -C "${TOPCOFFEA_DIR}" remote set-url origin "${TOPCOFFEA_REPO_URL}"
    git -C "${TOPCOFFEA_DIR}" fetch origin
else
    echo "[install_topcoffea] Cloning ${TOPCOFFEA_REPO_URL} into ${TOPCOFFEA_DIR}" >&2
    git clone "${TOPCOFFEA_REPO_URL}" "${TOPCOFFEA_DIR}"
fi

if [[ -z "${TOPCOFFEA_GIT_REF}" ]]; then
    DEFAULT_BRANCH=$(git -C "${TOPCOFFEA_DIR}" remote show origin | awk '/HEAD branch:/ {print $NF}')
    TOPCOFFEA_GIT_REF=${DEFAULT_BRANCH:-master}
fi

echo "[install_topcoffea] Checking out ${TOPCOFFEA_GIT_REF}" >&2
git -C "${TOPCOFFEA_DIR}" checkout "${TOPCOFFEA_GIT_REF}"
git -C "${TOPCOFFEA_DIR}" pull --ff-only origin "${TOPCOFFEA_GIT_REF}" >/dev/null 2>&1 || true

echo "[install_topcoffea] Installing topcoffea in editable mode" >&2
python -m pip install -e "${TOPCOFFEA_DIR}"

echo "[install_topcoffea] topcoffea is now importable from $(python -c 'import sys; print(sys.executable)')" >&2
