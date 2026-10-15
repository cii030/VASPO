import ast
import json
import numpy as np
import os
from pathlib import Path
import subprocess
import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode,source,whiten', [('solution_answer','old','false'), ('answer_only','old','false'), ('solution_answer','current','false'), ('solution_answer','old','true')])
def test_launcher_hydra_config(tmp_path, mode, source, whiten):
    # Replace only the training process; run the real shell launcher and Hydra composition.
    stub = tmp_path / 'python3'
    stub.write_text('#!/bin/sh\nexec "' + os.sys.executable + '" -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+':'+os.environ['PATH'], OUTPUT_ROOT=str(tmp_path/'output'),
               PRIVILEGED_TEXT_MODE=mode, VASPO_STUDENT_PROB_SOURCE=source, NORMALIZE_ADVANTAGES=whiten)
    run = subprocess.run(['bash', str(ROOT/'scripts/run_vaspo.sh'), 'test', 'test'], env=env, text=True, capture_output=True, check=True)
    args = json.loads(run.stdout.strip())
    assert args[:2] == ['-m', 'verl.trainer.main_ppo']
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'verl/trainer/config')):
        cfg = compose(config_name='ppo_trainer', overrides=args[2:])
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == 'vaspo'
    assert cfg.actor_rollout_ref.actor.policy_loss.vaspo_correct_only
    assert cfg.actor_rollout_ref.actor.policy_loss.ema_decay == .95
    assert cfg.algorithm.pre_update_value_before_adv
    assert not cfg.algorithm.use_dynamic_lambda_by_seq_len
    assert cfg.data.use_solution_for_critic == (mode == 'solution_answer')
    assert cfg.data.train_batch_size == 128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 32
    assert cfg.trainer.total_training_steps == 200
    validate = load_validator()
    validate(cfg)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.actor_rollout_ref.actor.policy_loss.vaspo_fusion_mode = "add"
    with pytest.raises(ValueError, match="Removed non-paper"):
        validate(cfg)


def load_validator():
    path = ROOT / 'rlcsd/self_distill_main.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_validate_vaspo_config')
    ns = {'np': np, 'normalize_privileged_text_mode': lambda mode: mode}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['_validate_vaspo_config']


def run_example(tmp_path, script, overrides=(), **settings):
    stub = tmp_path / 'python3'
    stub.write_text('#!/bin/sh\nexec "' + os.sys.executable + '" -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+':'+os.environ['PATH'],
               MODEL_ROOT=str(tmp_path/'models with spaces'),
               DATA_ROOT=str(tmp_path/'data'), OUTPUT_ROOT=str(tmp_path/'output'))
    env.update(settings)
    run = subprocess.run(
        ['bash', str(ROOT/'scripts_run'/script), 'vaspo', 'example', *overrides],
        env=env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    args = json.loads(run.stdout.strip())
    assert args[:2] == ['-m', 'verl.trainer.main_ppo']
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'verl/trainer/config')):
        return compose(config_name='ppo_trainer', overrides=args[2:])


@pytest.mark.parametrize('script,model,dataset,val_n', [
    ('qwen3-4b-instruct-vaspo-math-deepmath.sh', 'Qwen3-4B-Instruct-2507', 'deepmath_103k', 32),
    ('olmo3-7b-instruct-vaspo-math-deepmath.sh', 'Olmo-3-7B-Instruct-DPO', 'deepmath_103k', 32),
    ('qwen3-4b-instruct-vaspo-logic-kk.sh', 'Qwen3-4B-Instruct-2507', 'kk_4to8', 16),
])
def test_paper_examples(tmp_path, script, model, dataset, val_n):
    cfg = run_example(tmp_path, script)
    load_validator()(cfg)
    actor = cfg.actor_rollout_ref.actor
    rollout = cfg.actor_rollout_ref.rollout
    assert cfg.actor_rollout_ref.model.path == str(tmp_path/'models with spaces'/model)
    assert cfg.data.train_files == str(tmp_path/'data'/'verl'/dataset/'train.parquet')
    assert cfg.data.train_batch_size == 128 and cfg.data.train_max_samples == -1
    assert cfg.data.max_prompt_length == 2048 and cfg.data.max_response_length == 8192
    assert 'filter_groups' not in cfg.algorithm and 'gen_batch_size' not in cfg.data
    assert actor.ppo_mini_batch_size == 32 and actor.ppo_epochs == 1
    assert actor.optim.lr == 1e-6 and cfg.critic.optim.lr == 5e-6
    for optim in [actor.optim, cfg.critic.optim]:
        assert optim.optimizer == 'AdamW' and optim.weight_decay == .01
        assert optim.lr_warmup_steps == 50 and optim.lr_scheduler_type == 'constant'
    assert actor.policy_loss.epsilon == .2 and actor.policy_loss.epsilon_w == .2
    assert actor.clip_ratio_high == .28
    assert actor.policy_loss.vaspo_tanh_tau == .02
    assert actor.policy_loss.vaspo_token_mask_threshold == .02
    assert actor.policy_loss.ema_decay == .95
    assert actor.policy_loss.vaspo_student_prob_source == 'old'
    assert not actor.policy_loss.vaspo_teacher_in_actor_update
    assert not actor.policy_loss.vaspo_whiten_before_fusion
    assert not cfg.algorithm.normalize_advantages and cfg.algorithm.pre_update_value_before_adv
    assert cfg.algorithm.gamma == cfg.algorithm.lam == 1
    assert actor.loss_agg_mode == 'seq-mean-token-mean'
    assert cfg.critic.loss_agg_mode == 'token-mean'
    assert not actor.use_kl_loss and actor.kl_loss_coef == actor.entropy_coeff == 0
    assert cfg.actor_rollout_ref.model.lora_rank == cfg.critic.model.lora_rank == 0
    assert rollout.n == 8 and rollout.temperature == 1.0
    assert rollout.top_p == rollout.val_kwargs.top_p == .95
    assert rollout.top_k == rollout.val_kwargs.top_k == 20
    assert rollout.val_kwargs.n == val_n and rollout.val_kwargs.temperature == .6
    assert rollout.val_kwargs.do_sample
    assert not cfg.data.apply_chat_template_kwargs.enable_thinking
    assert not cfg.data.val_apply_chat_template_kwargs.enable_thinking
    assert not rollout.custom.teacher_enable_thinking
    assert cfg.custom_reward_function.path == 'rlcsd/verl_reward.py'
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.n_gpus_per_node == 8 and cfg.trainer.nnodes == 1


def test_example_overrides(tmp_path):
    cfg = run_example(
        tmp_path, 'qwen3-4b-instruct-vaspo-math-deepmath.sh',
        overrides=['trainer.total_training_steps=2', 'actor_rollout_ref.actor.optim.lr_warmup_steps=0'],
        MODEL_PATH=str(tmp_path/'custom actor'), CRITIC_MODEL_PATH=str(tmp_path/'custom critic'),
        TRAIN_FILES=str(tmp_path/'custom train.parquet'), VAL_FILES=str(tmp_path/'minerva.parquet'),
        VAL_N='8', NGPUS='4', CRITIC_MAX_PROMPT_LENGTH='4096',
    )
    assert cfg.actor_rollout_ref.model.path == str(tmp_path/'custom actor')
    assert cfg.critic.model.path == str(tmp_path/'custom critic')
    assert cfg.data.train_files == str(tmp_path/'custom train.parquet')
    assert cfg.data.val_files == str(tmp_path/'minerva.parquet')
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 8
    assert cfg.data.max_prompt_length_with_answer == 4096
    assert cfg.data.train_batch_size == 128 and cfg.trainer.n_gpus_per_node == 4
    assert cfg.trainer.total_training_steps == 2
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
