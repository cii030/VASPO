#!/usr/bin/env bash

set -euo pipefail

# Model/task values follow the variable-oriented reference launcher format.
# Shared paper hyperparameters are explicit in _vaspo_launch.sh.
method=vaspo
model_path="${MODEL_PATH:-${MODEL_ROOT:-/path/to/pretrained_models}/Qwen3-4B-Instruct-2507}"
critic_model_path="${CRITIC_MODEL_PATH:-${OUTPUT_ROOT:-/path/to/outputs}/critic_output/qwen3_4b_instruct_kk_solution}"
train_dataset=kk_4to8
val_dataset=kk_4to8_test,kk_9,kk_10,kk_11
val_n=16

# Native Instruct templates; no Qwen thinking mode is enabled.
export STUDENT_ENABLE_THINKING="${STUDENT_ENABLE_THINKING:-false}"
export TEACHER_ENABLE_THINKING="${TEACHER_ENABLE_THINKING:-false}"
export VAL_ENABLE_THINKING="${VAL_ENABLE_THINKING:-false}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_vaspo_launch.sh" "$@"
