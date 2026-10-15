# VASPO launch examples

The examples follow the reference launcher's structure: a small model/task
script defines variables and sources `_vaspo_launch.sh`. The shared launcher
defines the paper hyperparameters and calls `python3 -m verl.trainer.main_ppo`.
Run these scripts on the training machine after installing the GPU stack.

## Model/task examples

| Script | Actor | Training data | Validation samples per problem |
|---|---|---|---|
| `qwen3-4b-instruct-vaspo-math-deepmath.sh` | Qwen3-4B-Instruct-2507 | Full DeepMath-103K | 32 |
| `olmo3-7b-instruct-vaspo-math-deepmath.sh` | OLMo3-7B-Instruct-DPO | Full DeepMath-103K | 32 |
| `qwen3-4b-instruct-vaspo-logic-kk.sh` | Qwen3-4B-Instruct-2507 | KK 4–8 roles | 16 |

Set paths to your prepared assets. Each example needs a solution critic
checkpoint matching its actor backbone and training task.

```bash
export MODEL_ROOT=/path/to/pretrained_models
export DATA_ROOT=/path/to/dataset
export OUTPUT_ROOT=/path/to/outputs

# Qwen3 mathematics
CRITIC_MODEL_PATH=/path/to/qwen3_math_solution_critic \
  bash scripts_run/qwen3-4b-instruct-vaspo-math-deepmath.sh vaspo qwen3_math

# OLMo3 mathematics
CRITIC_MODEL_PATH=/path/to/olmo3_math_solution_critic \
  bash scripts_run/olmo3-7b-instruct-vaspo-math-deepmath.sh vaspo olmo3_math

# Qwen3 Knights and Knaves
CRITIC_MODEL_PATH=/path/to/qwen3_kk_solution_critic \
  bash scripts_run/qwen3-4b-instruct-vaspo-logic-kk.sh vaspo qwen3_kk
```

The first two arguments name the project and experiment. Checkpoints go to
`$OUTPUT_ROOT/<project>/<experiment>/checkpoints`; rollout and validation logs
go to the adjacent `rollout/` directory. The scripts resolve the repository root
from their own location, so they can also be invoked by absolute path.

## Prepared data paths

The default paths are naming conventions for locally prepared Parquet files,
not files shipped in this repository:

```text
$DATA_ROOT/verl/deepmath_103k/train.parquet
$DATA_ROOT/verl/aime24+aime25+aime26+hmmt25/val.parquet
$DATA_ROOT/verl/kk_4to8/train.parquet
$DATA_ROOT/verl/kk_4to8_test+kk_9+kk_10+kk_11/val.parquet
```

Mathematics uses all 103,022 DeepMath training problems; KK uses the complete
5,000-problem 4–8-role training split described in the paper. Supply your
prepared files via `TRAIN_FILES` and `VAL_FILES` if their names differ. The
launcher sets no training sample cap and does not enable dynamic group filtering.

Training rows need `prompt`, `data_source`, `reward_model.ground_truth`, and
`extra_info.solution`. The actor sees the original prompt; the teacher and
critic receive the reference solution and answer. KK rows must use a supported
`data_source`, such as `kk_4to8` or `kk_9`, for assignment-based scoring.
The launcher selects the included `rlcsd/verl_reward.py` scorer, which routes
mathematics to its math-verify grading path and KK to the assignment verifier.

Qwen3 uses its native non-thinking Instruct template. OLMo3 uses the
Instruct-DPO checkpoint and its native instruction template, rather than a
Think checkpoint.

## Paper parameter mapping

Values below come from Experimental Setup, Appendix Training Details, and the
paper's training/decoding and reader-configuration tables.

| Setting | Value | Launcher variable or Hydra field |
|---|---|---|
| Hardware | 8 H200 GPUs, one node | `NGPUS=8`, `NNODES=1` |
| Full-parameter training | Enabled | `actor_rollout_ref.model.lora_rank=0`, critic default rank 0 |
| Optimizer | AdamW | actor/critic `optim.optimizer` |
| Actor / critic learning rate | 1e-6 / 5e-6 | `learning_rate`, `critic_lr` |
| Weight decay | 0.01 | `weight_decay` |
| Learning-rate schedule | 50-step linear warmup, then constant | `warmup_steps`, `optim.lr_scheduler_type` |
| Global / PPO mini-batch size | 128 / 32 | `train_batch_size`, `ppo_mini_batch_size` |
| Rollout updates / PPO epochs | 200 / 1 | `total_training_steps`, `ppo_epochs` |
| Responses per prompt | 8 | `group_size` |
| Gamma / GAE lambda | 1 / 1 | `gamma`, `lam` |
| VASPO PPO clip / weight clip | 0.2 / 0.2 | `epsilon`, `epsilon_w` |
| Generic actor upper clip field | 0.28 | `clip_ratio_high` |
| Token selection tau / delta | 0.02 / 0.02 | `tanh_tau`, `token_mask_threshold` |
| Teacher | EMA, decay 0.95 | `teacher_mode=ema`, `ema_decay` |
| Student probability denominator | Behavior policy | `vaspo_student_prob_source=old` |
| Teacher scoring | Cached before actor updates | `vaspo_teacher_in_actor_update=false` |
| Privileged evidence | Solution and answer | `privileged_text_mode=solution_answer` |
| Critic pre-update | Enabled | `algorithm.pre_update_value_before_adv=true` |
| Advantage normalization | None before or after fusion | `normalize_advantages=false`, `vaspo_whiten_before_fusion=false` |
| Actor loss reduction | Valid-token mean per response, then response mean | `loss_agg_mode=seq-mean-token-mean` |
| Reference KL / entropy coefficient | 0 / 0 | `kl_loss_coef`, `entropy_coeff` |
| Rollout correction | Token importance weighting | `algorithm.rollout_correction.rollout_is=token` |
| Training / evaluation temperature | 1.0 / 0.6 | `temperature`, `val_temperature` |
| Top-p / top-k | 0.95 / 20 | `top_p`, `top_k`, validation counterparts |
| Actor prompt / response limit | 2048 / 8192 | `max_prompt_length`, `max_response_length` |
| Validation sampling | Enabled | `val_do_sample=true` |

The paper records both a PPO clip of 0.2 and a generic higher-clip field of
0.28. These are separate configuration fields: the supplied VASPO loss reads
`policy_loss.epsilon=0.2` and clips symmetrically to `[0.8, 1.2]`; it does not
read `actor.clip_ratio_high`. The example preserves both recorded values.

The value loss includes the existing factor of 0.5 inside
`compute_value_loss`; the critic has a separate optimizer. There is no extra
joint actor/value loss multiplier to set in the launch command. Offline critic
pretraining (two epochs in the paper) is a separate stage; the examples load
its checkpoint rather than setting RL critic epochs to two.
Use the included [critic-pretraining pipeline](../critic_init/README.md) to
prepare that checkpoint, then pass its output directory as `CRITIC_MODEL_PATH`.
The critic retains the source's token-mean loss reduction independently of
the actor's response-mean reduction.

Micro-batch size 4, gradient clipping 1.0, tensor parallel size 1, vLLM memory
fraction 0.90, checkpoint/validation intervals 50, and token memory budgets
are runtime defaults inherited from the source. The critic prompt budget
defaults to 2048 and can be changed with `CRITIC_MAX_PROMPT_LENGTH`; this is
separate from the paper's clean actor prompt limit. Prepare the complete
privileged evidence to fit the reader's context budget before training.

## Overrides

Use environment variables for paths, GPU layout, and the existing ablations.
Hydra overrides after the first two arguments take precedence over defaults.

```bash
MODEL_PATH=/path/to/actor \
CRITIC_MODEL_PATH=/path/to/solution_critic \
TRAIN_FILES=/path/to/deepmath_train.parquet \
VAL_FILES=/path/to/math_validation.parquet \
  bash scripts_run/qwen3-4b-instruct-vaspo-math-deepmath.sh vaspo custom_math \
    trainer.save_freq=25
```

Math examples default to the four avg@32 validation benchmarks. For a
training-time validation file containing only Minerva Math or OlympiadBench,
set `VAL_N=8` and supply that file via `VAL_FILES`. KK uses 16 samples per
problem. All examples retain the 8192-token response limit for validation.
The training-time verifier does not implement the paper's final two-stage
math evaluation with CompassVerifier-3B; final benchmark evaluation remains a
separate pipeline.

`scripts/run_vaspo.sh` remains an alias for the Qwen3 mathematics example.
