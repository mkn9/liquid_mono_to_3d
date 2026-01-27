#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PROOF-BUNDLE GATE
# The ONLY definition of "done" for any task.
# ============================================================================

# ---- config you may edit ----
PYTEST_CMD=${PYTEST_CMD:-"pytest -q"}
PROOF_ROOT=${PROOF_ROOT:-"artifacts/proof"}
CONTRACTS_DIR=${CONTRACTS_DIR:-"contracts"}
# -----------------------------

# Determine git commit (works even if detached HEAD in CI)
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'no-git')"
TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
OUT_DIR="${PROOF_ROOT}/${GIT_SHA}"
LOG="${OUT_DIR}/prove.log"
META="${OUT_DIR}/meta.txt"
MANIFEST="${OUT_DIR}/manifest.txt"

mkdir -p "${OUT_DIR}"

# Helper to run a command and capture full output + exit code
run_capture () {
  local label="$1"
  shift
  echo "=== ${label} ===" | tee -a "${LOG}"
  echo "cmd: $*" | tee -a "${LOG}"
  set +e
  "$@" 2>&1 | tee -a "${LOG}"
  status=${PIPESTATUS[0]}
  set -e
  echo "exit_code: ${status}" | tee -a "${LOG}"
  echo "" | tee -a "${LOG}"
  return "${status}"
}

# --- provenance ---
{
  echo "timestamp_utc: ${TS_UTC}"
  echo "git_sha: ${GIT_SHA}"
  echo "pwd: $(pwd)"
  echo "python: $(python3 -V 2>&1 || echo 'python3 not found')"
  echo "pip: $(pip3 -V 2>&1 || echo 'pip3 not found')"
  echo "pytest: $(pytest --version 2>&1 || echo 'pytest not found')"
  echo "platform: $(uname -a 2>&1 || true)"
} | tee "${META}"

# (Optional but helpful) freeze environment for reproducibility
pip3 freeze > "${OUT_DIR}/pip_freeze.txt" 2>/dev/null || true

# --- run tests (required) ---
run_capture "TESTS" bash -lc "${PYTEST_CMD}"

# --- optional: run component contracts if present ---
if [ -d "${CONTRACTS_DIR}" ] && compgen -G "${CONTRACTS_DIR}/*.yaml" > /dev/null; then
  run_capture "CONTRACTS" python3 -m scripts.prove_component --contracts "${CONTRACTS_DIR}" --outdir "${OUT_DIR}"
fi

# --- create a manifest of what was produced (ties evidence to this commit) ---
# List all files in the proof bundle with sha256 checksums
(
  cd "${OUT_DIR}"
  # sha256sum exists on Linux/macOS; if not, you can swap for shasum -a 256
  if command -v sha256sum >/dev/null 2>&1; then
    find . -type f -maxdepth 2 -print0 | sort -z | xargs -0 sha256sum
  else
    find . -type f -maxdepth 2 -print0 | sort -z | xargs -0 shasum -a 256
  fi
) | tee "${MANIFEST}"

echo ""
echo "========================================================================"
echo "✓ PROOF BUNDLE CREATED"
echo "========================================================================"
echo "Location: ${OUT_DIR}"
echo "Git SHA:  ${GIT_SHA}"
echo "Time:     ${TS_UTC}"
echo ""
echo "Contents:"
echo "  - prove.log      (full test output)"
echo "  - meta.txt       (environment & provenance)"
echo "  - manifest.txt   (file checksums)"
echo "  - pip_freeze.txt (python dependencies)"
echo ""
echo "This proof bundle ties all evidence to git commit ${GIT_SHA}"
echo "========================================================================"

