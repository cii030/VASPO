#!/usr/bin/env bash

set -euo pipefail

# Shared direct launcher. Model/task examples set their variables, then source
# this file in the same style as the reference _rlcsd_launch.sh.
project_name=${1:?"usage: $0 <project_name> <experiment_name> [Hydra overrides...]"}
experiment_name=${2:?"missing experiment_name"}
shift 2

: "${method:?launcher requires method}"
: "${model_path:?launcher requires model_path}"
: "${critic_model_path:?launcher requires critic_model_path}"
: "${train_dataset:?launcher requires train_dataset}"
: "${val_dataset:?launcher requires val_dataset}"

# Paper: Experimental Setup, Training Details, and the two configuration tables.
# Global batch size stays 128 regardless of the GPU count.
train_batch_size="${train_batch_size:-128}"
ppo_mini_batch_size="${ppo_mini_batch_size:-32}"
group_size="${group_size:-8}"
learning_rate="${learning_rate:-1e-6}"
critic_lr="${critic_lr:-5e-6}"
weight_decay="${weight_decay:-0.01}"
warmup_steps="${warmup_steps:-50}"
ppo_epochs="${ppo_epochs:-1}"
num_epochs="${num_epochs:-30}"
total_training_steps="${total_training_steps:-200}"
gamma="${gamma:-1.0}"
lam="${lam:-1.0}"
epsilon="${epsilon:-0.2}"
# Recorded generic actor field; the VASPO loss uses policy_loss.epsilon.
clip_ratio_high="${clip_ratio_high:-0.28}"
epsilon_w="${VASPO_EPSILON_W:-${epsilon_w:-0.2}}"
tanh_tau="${tanh_tau:-0.02}"
token_mask_threshold="${token_mask_threshold:-0.02}"
ema_decay="${ema_decay:-0.95}"
normalize_advantages="${NORMALIZE_ADVANTAGES:-false}"
pre_update_value_before_adv="${PRE_UPDATE_VALUE_BEFORE_ADV:-true}"
max_prompt_length="${max_prompt_length:-2048}"
max_response_length="${max_response_length:-8192}"
temperature="${temperature:-1.0}"
top_p="${top_p:-0.95}"
top_k="${top_k:-20}"
val_n="${VAL_N:-${val_n:-32}}"
val_temperature="${val_temperature:-0.6}"
val_top_p="${val_top_p:-0.95}"
val_top_k="${val_top_k:-20}"
val_do_sample="${val_do_sample:-true}"

# Runtime choices inherited from the submission launcher, rather than paper
# hyperparameters. Adjust memory budgets for the training machine if needed.
grad_clip="${grad_clip:-1.0}"
ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu:-4}"
tensor_model_parallel_size="${tensor_model_parallel_size:-1}"
vllm_gpu_memory_utilization="${vllm_gpu_memory_utilization:-0.90}"
critic_max_prompt_length="${CRITIC_MAX_PROMPT_LENGTH:-2048}"
save_freq="${SAVE_FREQ:-50}"
test_freq="${test_freq:-50}"
checkpoint_contents='[model,optimizer,extra]'
policy_loss_args=(
    +actor_rollout_ref.actor.policy_loss.vaspo_use_token_mask="${VASPO_USE_TOKEN_MASK:-true}"
    +actor_rollout_ref.actor.policy_loss.vaspo_token_mask_threshold=${token_mask_threshold}
    +actor_rollout_ref.actor.policy_loss.vaspo_tanh_tau=${tanh_tau}
    +actor_rollout_ref.actor.policy_loss.vaspo_student_prob_source="${VASPO_STUDENT_PROB_SOURCE:-old}"
    +actor_rollout_ref.actor.policy_loss.vaspo_teacher_in_actor_update="${VASPO_TEACHER_IN_ACTOR_UPDATE:-false}"
    +actor_rollout_ref.actor.policy_loss.vaspo_whiten_before_fusion=false
    +actor_rollout_ref.actor.policy_loss.vaspo_diagnostics="${VASPO_DIAGNOSTICS:-false}"
    +actor_rollout_ref.actor.policy_loss.vaspo_position_log="${VASPO_POSITION_LOG:-null}"
    +actor_rollout_ref.actor.policy_loss.epsilon=${epsilon}
    +actor_rollout_ref.actor.policy_loss.epsilon_w=${epsilon_w}
    +actor_rollout_ref.actor.policy_loss.vaspo_correct_only=true
    +actor_rollout_ref.actor.policy_loss.teacher_mode=ema
    +actor_rollout_ref.actor.policy_loss.ema_decay=${ema_decay}
)

export TORCH_COMPILE_DISABLE=1
export TORCH_COMPILE_BACKEND=eager
export TRITON_MAX_CONCURRENT_COMPILATIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export NCCL_DEBUG=WARN
export MASTER_PORT="${MASTER_PORT:-29500}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export MODEL_ROOT="${MODEL_ROOT:-/path/to/pretrained_models}"
export DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${OUTPUT_ROOT}/swanlog}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${VERL_ROOT}"

set -x

train_files="${TRAIN_FILES:-${DATA_ROOT%/}/verl/${train_dataset}/train.parquet}"
val_dataset_path=${val_dataset//,/+}
val_files="${VAL_FILES:-${DATA_ROOT%/}/verl/${val_dataset_path}/val.parquet}"

export EXP_DIR="${OUTPUT_ROOT}/${project_name}/${experiment_name}/checkpoints"
export rollout_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}/rollout"
mkdir -p "${EXP_DIR}" "${rollout_dir}"

# Hardware used in the paper: 8 H200 GPUs on one node.
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

student_enable_thinking="${STUDENT_ENABLE_THINKING:-false}"
teacher_enable_thinking="${TEACHER_ENABLE_THINKING:-false}"
val_enable_thinking="${VAL_ENABLE_THINKING:-false}"

# Both readers use the first reference solution and final answer.
privileged_text_mode="${PRIVILEGED_TEXT_MODE:-solution_answer}"
teacher_wrapper_variant="${TEACHER_WRAPPER_VARIANT:-neutral}"

# VASPO uses one with-answer critic; it does not use HEM's second clean critic.
adv_estimator=decoupled_gae
critic_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

use_solution_for_critic=true
case "${privileged_text_mode}" in
    solution_answer|answer_only) ;;
    *) echo "PRIVILEGED_TEXT_MODE must be solution_answer or answer_only" >&2; exit 1 ;;
esac
if [[ "${privileged_text_mode}" == answer_only ]]; then
    use_solution_for_critic=false
fi

critic_args=(
    data.use_answer_for_critic=true
    data.use_solution_for_critic=${use_solution_for_critic}
    data.standard_answer_key=reward_model.ground_truth
    data.max_prompt_length_with_answer=${critic_max_prompt_length}
    algorithm.pre_update_value_before_adv=${pre_update_value_before_adv}
    algorithm.use_hem=false
    algorithm.use_dynamic_lambda_by_seq_len=false
    critic.use_value_sigmoid=true
)

python3 -m verl.trainer.main_ppo \
    algorithm.normalize_advantages="${normalize_advantages}" \
    algorithm.use_dppo=false \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gamma} \
    algorithm.lam=${lam} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.train_max_samples=-1 \
    data.val_batch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    +data.apply_chat_template_kwargs.enable_thinking=${student_enable_thinking} \
    +data.val_apply_chat_template_kwargs.enable_thinking=${val_enable_thinking} \
    +data.thinking_system_prompt=False \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${warmup_steps} \
    actor_rollout_ref.actor.optim.optimizer=AdamW \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=${epsilon} \
    actor_rollout_ref.actor.clip_ratio_low=${epsilon} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${checkpoint_contents}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${checkpoint_contents}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=${method} \
    "${policy_loss_args[@]}" \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${group_size} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    +actor_rollout_ref.rollout.custom.privileged_text_mode=${privileged_text_mode} \
    +actor_rollout_ref.rollout.custom.teacher_wrapper_variant=${teacher_wrapper_variant} \
    +actor_rollout_ref.rollout.custom.teacher_enable_thinking=${teacher_enable_thinking} \
    +actor_rollout_ref.rollout.custom.thinking_system_prompt=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    critic.model.path="${critic_model_path}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_activation_offload=True \
    critic.optim.lr=${critic_lr} \
    critic.optim.optimizer=AdamW \
    critic.optim.weight_decay=${weight_decay} \
    critic.optim.lr_warmup_steps=${warmup_steps} \
    critic.optim.lr_scheduler_type=constant \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.use_dynamic_bsz=True \
    critic.loss_agg_mode=token-mean \
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len} \
    reward_model.reward_manager=naive \
    custom_reward_function.path=rlcsd/verl_reward.py \
    custom_reward_function.name=compute_score \
    trainer.use_legacy_worker_impl=enable \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.default_local_dir="${EXP_DIR}" \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir="${rollout_dir}" \
    trainer.validation_data_dir="${rollout_dir}" \
    trainer.val_before_train=True \
    trainer.log_val_generations=1 \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=${num_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    "${critic_args[@]}" \
    "$@"
