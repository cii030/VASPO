#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRITIC_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Explicit assets replace the original cluster-specific paths. Legacy lowercase
# names are accepted when the uppercase variable is not supplied.
model_name_or_path="${MODEL_PATH:-${model_name_or_path:-}}"
train_file="${TRAIN_FILE:-${train_file:-}}"
output_dir="${OUTPUT_DIR:-${output_dir:-}}"
: "${model_name_or_path:?set MODEL_PATH to the matching actor/value backbone}"
: "${train_file:?set TRAIN_FILE to the prepared critic JSONL}"
: "${output_dir:?set OUTPUT_DIR to the critic checkpoint directory}"

deepspeed="${DEEPSPEED_CONFIG:-${CRITIC_ROOT}/configs/zero1.json}"
if [[ ! -f "${train_file}" ]]; then
    echo "TRAIN_FILE does not exist: ${train_file}" >&2
    exit 1
fi
if [[ ! -f "${deepspeed}" ]]; then
    echo "DeepSpeed configuration does not exist: ${deepspeed}" >&2
    exit 1
fi

# Paper record: two epochs, lr 5e-6, 256-step accumulation, bfloat16, and
# ZeRO-1 on eight GPUs. Micro-batch/context sizes are inherited runtime defaults.
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
master_port="${MASTER_PORT:-29501}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-2}"
learning_rate="${LEARNING_RATE:-5e-6}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-256}"
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
max_length="${MAX_LENGTH:-8192}"

distributed_args=(
    --nproc_per_node="${NGPUS}"
    --nnodes="${NNODES}"
    --master_port="${master_port}"
)
if [[ "${NNODES}" -gt 1 ]]; then
    node_rank="${NODE_RANK:-${RANK:-}}"
    : "${node_rank:?set NODE_RANK (or RANK) for multi-node training}"
    : "${MASTER_ADDR:?set MASTER_ADDR for multi-node training}"
    distributed_args+=(--node_rank="${node_rank}" --master_addr="${MASTER_ADDR}")
fi

export PYTHONUNBUFFERED=1

exec torchrun "${distributed_args[@]}" \
    "${CRITIC_ROOT}/runner/train.py" \
    --deepspeed "${deepspeed}" \
    --model_name_or_path "${model_name_or_path}" \
    --train_file "${train_file}" \
    --output_dir "${output_dir}" \
    --do_train \
    --save_strategy epoch \
    --logging_strategy steps \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --learning_rate "${learning_rate}" \
    --lr_scheduler_type constant \
    --num_train_epochs "${num_train_epochs}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS:-8}" \
    --preprocessing_batch_size "${PREPROCESSING_BATCH_SIZE:-256}" \
    --overwrite_cache "${OVERWRITE_CACHE:-False}" \
    --use_length_grouping "${USE_LENGTH_GROUPING:-True}" \
    --warmup_ratio 0.0 \
    --max_length "${max_length}" \
    --train_value_head_only "${TRAIN_VALUE_HEAD_ONLY:-False}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --gradient_checkpointing \
    --bf16 \
    --ignore_data_skip True \
    --report_to none \
    "$@"
