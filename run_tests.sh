#!/usr/bin/env bash
# Runs every test module in its own process so each gets a fresh, isolated
# temp data directory (see tests/_env.py) - avoids cross-test contamination
# of the sqlite DB / vector store singletons.
set -uo pipefail
cd "$(dirname "$0")"

PASS=0
FAIL=0

for f in tests/test_*.py; do
  module="${f%.py}"
  module="${module//\//.}"
  echo "=== $module ==="
  if python3 -m unittest "$module" -v; then
    PASS=$((PASS+1))
  else
    FAIL=$((FAIL+1))
  fi
  echo
done

echo "================================"
echo "Test modules passed: $PASS"
echo "Test modules failed: $FAIL"
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
