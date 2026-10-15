#!/usr/bin/env bash

# Backward-compatible entry point for the Qwen3 mathematics example.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/../scripts_run/qwen3-4b-instruct-vaspo-math-deepmath.sh" "$@"
