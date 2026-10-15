# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.kpop import apply_kpop, configure_kpop
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           compute_variance_proxy_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (Role, WorkerType, need_critic,
                                    need_reference_policy, need_reward_model)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import (find_latest_ckpt_path,
                                                      should_save_ckpt_esi)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (calculate_workload,
                                         get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import (left_right_2_no_padding,
                                        no_padding_2_padding)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    sample_weights: torch.Tensor | None = None,
    whiten_advantages: bool = True,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.
        sample_weights: Optional sequence-level `(bs,)` or token-level `(bs, response_length)`
            weights applied before advantage whitening by supported estimators.
        whiten_advantages: Whether GAE-family estimators should whiten before returning. VASPO passes
            false and lets its fusion hook apply optional global whitening before or after fusion.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    policy_lam: float | torch.Tensor = lam
    if config is not None and config.get("use_dynamic_lambda_by_seq_len", False):
        alpha = float(config.get("dynamic_lambda_alpha", 0.05))
        if alpha <= 0:
            raise ValueError(f"algorithm.dynamic_lambda_alpha must be > 0, got {alpha}")
        seq_len = data.batch["response_mask"].sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)
        # lambda_policy = 1 - 1 / (alpha * l), clamp to valid [0, 1) range for short responses.
        policy_lam = 1.0 - 1.0 / (alpha * seq_len)
        policy_lam = torch.clamp(policy_lam, min=0.0, max=0.999999)
        if "values" in data.batch:
            policy_lam = policy_lam.to(dtype=data.batch["values"].dtype, device=data.batch["values"].device)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=policy_lam,
            sample_weights=sample_weights,
            whiten_advantages=whiten_advantages,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.DECOUPLED_GAE:
        # Decoupled GAE: lambda_critic=1.0 for V_target, lambda_actor=0.99 for policy advantage
        advantages, returns = core_algos.compute_decoupled_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam_actor=policy_lam,
            sample_weights=sample_weights,
            whiten_advantages=whiten_advantages,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    RESPONSE_THIRD_BUCKET_NAMES = ("first_1_3", "middle_1_3", "last_1_3")

    @staticmethod
    def _response_third_slices(valid_len: int) -> list[tuple[str, int, int]]:
        first_end = valid_len // 3
        second_end = (2 * valid_len) // 3
        return [
            ("first_1_3", 0, first_end),
            ("middle_1_3", first_end, second_end),
            ("last_1_3", second_end, valid_len),
        ]

    @staticmethod
    def _explained_variance_from_tensors(values: torch.Tensor, returns: torch.Tensor) -> float | None:
        if values.numel() <= 1 or returns.numel() <= 1:
            return None
        values = values.float()
        returns = returns.float()
        return_diff_var = torch.var(returns - values)
        return_var = torch.var(returns)
        return (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # BPCO switches. DPPO lives in the shared vanilla PPO loss so custom
        # PPO-based methods (including HEM and VASPO) can opt in without changing
        # their loss_mode. Advantage normalization is controlled independently.
        self.use_dppo = bool(self.config.algorithm.get("use_dppo", False))
        self.normalize_advantages = bool(self.config.algorithm.get("normalize_advantages", True))
        with open_dict(self.config.actor_rollout_ref.actor.policy_loss):
            self.config.actor_rollout_ref.actor.policy_loss.use_dppo = self.use_dppo

        self.kpop_config = configure_kpop(self.config)
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None and rollout_corr_config.get("loss_type", "ppo_clip") == "dis":
            # DIS must bind before workers initialize. Load its optional helper
            # only when selected so KPop/PPO also work with pre-DIS helpers.
            from verl.trainer.ppo.rollout_corr_helper import configure_dis

            configure_dis(self.config)

        self.use_hem = self.config.algorithm.get("use_hem", False)
        # Solution context is intentionally subordinate to answer context.  If
        # use_solution_for_critic is set in an old/incomplete config without
        # use_answer_for_critic, it is ignored rather than changing the default
        # standard critic input.
        self.use_answer_for_critic = bool(self.config.data.get("use_answer_for_critic", False))
        self.use_solution_for_critic = bool(self.config.data.get("use_solution_for_critic", False)) and bool(
            self.use_answer_for_critic
        )
        self.hem_alpha = self.config.algorithm.get("hem_alpha", 1.0)
        self.hem_eps = self.config.algorithm.get("hem_eps", 1e-6)
        self.hem_clip_min = self.config.algorithm.get("hem_clip_min", 0.5)
        self.hem_clip_max = self.config.algorithm.get("hem_clip_max", 2.0)
        self.hem_weight_mode = str(self.config.algorithm.get("hem_weight_mode", "sequence")).lower()
        self.hem_apply_before_whiten = self.config.algorithm.get("hem_apply_before_whiten", False)
        self.hem_samples_dir = self.config.algorithm.get("hem_samples_dir", None)
        if self.hem_samples_dir is None:
            rollout_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_dir:
                self.hem_samples_dir = os.path.join(rollout_dir, "hem_samples")
        self.hem_pre_update_oracle_only = self.config.algorithm.get("hem_pre_update_oracle_only", False)
        self.hem_invert_weight = self.config.algorithm.get("hem_invert_weight", False)
        self.hem_skip_truncated = self.config.algorithm.get("hem_skip_truncated", False)
        self.hem_correct_only = self.config.algorithm.get("hem_correct_only", False)
        # Reject removed config keys with a migration hint so silently-stale launch scripts fail loudly.
        for legacy_key in ("hem_oracle_build_mode", "hem_oracle_connector", "hem_oracle_content_field"):
            if legacy_key in self.config.algorithm:
                raise ValueError(
                    f"algorithm.{legacy_key} has been removed. Customize oracle critic inputs "
                    f"via data.standard_answer_key (content field) and data.answer_template "
                    f"(connector phrase) instead, with data.use_answer_for_critic=True."
                )
        self.pre_update_value_before_adv = self.config.algorithm.get("pre_update_value_before_adv", False)
        self.critic_standard_model_path = self.config.critic.get("standard_model_path", None)
        self.critic_oracle_model_path = self.config.critic.get("oracle_model_path", None)
        self.critic_standard_model_init_path = self.config.critic.get("standard_model_init_path", None)
        self.critic_oracle_model_init_path = self.config.critic.get("oracle_model_init_path", None)
        self.critic_oracle_wg = None
        self._hem_standard_bootstrapped = False
        runtime_root = os.path.expanduser(self.config.trainer.default_local_dir)
        self._hem_runtime_standard_ckpt_path = os.path.join(runtime_root, "_hem_runtime_standard_critic")
        self._hem_runtime_oracle_ckpt_path = os.path.join(runtime_root, "_hem_runtime_oracle_critic")
        self._hem_oracle_runtime_ready = False
        self._hem_runtime_standard_from_hf_ckpt_path = os.path.join(runtime_root, "_hem_runtime_standard_from_hf")
        self._hem_runtime_oracle_from_hf_ckpt_path = os.path.join(runtime_root, "_hem_runtime_oracle_from_hf")
        self._hem_runtime_backup_ckpt_path = os.path.join(runtime_root, "_hem_runtime_backup_critic")
        self._hem_standard_load_path = self.critic_standard_model_path
        self._hem_oracle_load_path = self.critic_oracle_model_path
        self._hem_model_paths_ready = False
        self._hem_rollout_dump_cache: dict = {}

        if self.use_hem:
            if self.hem_weight_mode not in ("sequence", "token"):
                raise ValueError(
                    f"algorithm.hem_weight_mode must be either 'sequence' or 'token', got {self.hem_weight_mode!r}."
                )
            if not self.use_critic:
                raise ValueError("algorithm.use_hem=True requires critic to be enabled.")
            if not self.use_answer_for_critic:
                raise ValueError(
                    "algorithm.use_hem=True requires data.use_answer_for_critic=True so the "
                    "dataset constructs answer-augmented oracle critic inputs."
                )
            if not self.critic_standard_model_path or not self.critic_oracle_model_path:
                raise ValueError(
                    "algorithm.use_hem=True requires both critic.standard_model_path and critic.oracle_model_path."
                )
            self.critic_standard_model_path = os.path.expanduser(self.critic_standard_model_path)
            self.critic_oracle_model_path = os.path.expanduser(self.critic_oracle_model_path)
            if self.critic_standard_model_init_path:
                self.critic_standard_model_init_path = os.path.expanduser(self.critic_standard_model_init_path)
            if self.critic_oracle_model_init_path:
                self.critic_oracle_model_init_path = os.path.expanduser(self.critic_oracle_model_init_path)
            if self.use_legacy_worker_impl == "disable":
                raise ValueError(
                    "algorithm.use_hem=True now uses separate standard/oracle critic workers and currently "
                    "requires the legacy FSDP/Megatron worker implementation. Set "
                    "trainer.use_legacy_worker_impl=auto or enable."
                )

        if self.pre_update_value_before_adv and not self.use_critic:
            raise ValueError("algorithm.pre_update_value_before_adv=True requires critic to be enabled.")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=True,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=False,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        import numpy as np

        def _to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: _to_json_serializable(v[i]) for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _per_sample_segmented_means(
        self, token_tensor: torch.Tensor, response_mask: torch.Tensor
    ) -> list[list[float]]:
        """Compute per-sample [first_1_3, middle_1_3, last_1_3, total] means.

        Empty / out-of-range segments are emitted as None so the consumer can tell
        them apart from a true zero mean.
        """
        if token_tensor.shape != response_mask.shape:
            return []
        results: list[list[float]] = []
        with torch.no_grad():
            mask_bool = response_mask.bool()
            for idx in range(token_tensor.shape[0]):
                valid = token_tensor[idx][mask_bool[idx]]
                valid_len = int(valid.numel())
                row: list = []
                if valid_len == 0:
                    results.append([None] * len(self.RESPONSE_THIRD_BUCKET_NAMES) + [None])
                    continue
                for _, s, e in self._response_third_slices(valid_len):
                    if e <= s:
                        row.append(None)
                    else:
                        row.append(float(valid[s:e].mean().item()))
                row.append(float(valid.mean().item()))
                results.append(row)
        return results

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            # Per-sample HEM error, segmented entropy, and segmented advantage.
            n = len(inputs)
            cache = getattr(self, "_hem_rollout_dump_cache", {}) or {}

            hem_error_list = cache.get("hem_error")
            if hem_error_list is not None and len(hem_error_list) == n:
                reward_extra_infos_to_dump["hem_error"] = hem_error_list

            entropy_segments = cache.get("entropy_segments")
            if entropy_segments is not None and len(entropy_segments) == n:
                reward_extra_infos_to_dump["entropy_segments"] = entropy_segments

            adv_segments = cache.get("advantage_segments")
            if adv_segments is not None and len(adv_segments) == n:
                reward_extra_infos_to_dump["advantage_segments"] = adv_segments

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

            # Clear the cache so a later step that doesn't repopulate it does not leak stale data.
            self._hem_rollout_dump_cache = {}

    def _get_rollout_log_dir(self):
        return os.path.join(self.config.trainer.default_local_dir, "rollout_details")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_response_lengths = []
        sample_value_tensors = []
        sample_value_oracle_tensors = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", None)
            val_response_length = self.config.actor_rollout_ref.rollout.response_length
            if rollout_custom is not None:
                val_response_length = int(rollout_custom.get("val_response_length", val_response_length))
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "max_tokens": val_response_length,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.val_kwargs.temperature
            # Match training metrics: count non-padding response tokens, including EOS when present.
            sample_response_lengths.extend(compute_response_mask(test_batch).sum(dim=-1).cpu().tolist())

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            # evaluate value model on validation batch
            if self.use_critic:
                if "position_ids" not in test_batch.batch:
                    test_batch.batch["position_ids"] = compute_position_id_with_mask(test_batch.batch["attention_mask"])
                if "response_mask" not in test_batch.batch:
                    test_batch.batch["response_mask"] = compute_response_mask(test_batch)

                # Build actor/critic routing (sets input_ids_for_critic_standard / _oracle).
                self._prepare_actor_critic_inputs(test_batch)

                resp_mask = test_batch.batch["response_mask"]

                def _compute_values_with_source(source: str) -> torch.Tensor:
                    self._activate_critic_inputs(test_batch, source=source)
                    critic_wg = self._critic_wg_for_source(source)
                    padded, pad_sz = pad_dataproto_to_divisor(test_batch, critic_wg.world_size)
                    out = self._compute_values(padded, source=source)
                    out = unpad_dataproto(out, pad_size=pad_sz)
                    return out.batch["values"]

                if self.use_hem:
                    # 1) Standard value model on original critic inputs.
                    std_values = _compute_values_with_source("standard")

                    # 2) Oracle value model on with-answer critic inputs.
                    oracle_values = _compute_values_with_source("oracle")

                    for i in range(std_values.size(0)):
                        m = resp_mask[i].bool()
                        sample_value_tensors.append(std_values[i][m].detach().cpu().float())
                        sample_value_oracle_tensors.append(oracle_values[i][m].detach().cpu().float())
                    print(
                        f"value model evaluation (HEM): computed standard+oracle values "
                        f"for {std_values.size(0)} samples"
                    )
                else:
                    std_values = _compute_values_with_source("default")
                    for i in range(std_values.size(0)):
                        m = resp_mask[i].bool()
                        sample_value_tensors.append(std_values[i][m].detach().cpu().float())
                    print(f"value model evaluation: computed values for {std_values.size(0)} samples")

                # Restore actor input activation for any downstream use.
                self._activate_actor_inputs(test_batch)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # compute value model evaluation metrics
        value_model_eval_metrics = {}
        if self.use_critic and sample_value_tensors:
            data_sources_flat = np.concatenate(data_source_lst, axis=0)
            value_model_eval_metrics = self._evaluate_value_model_on_validation(
                sample_uids=sample_uids,
                sample_scores=sample_scores,
                sample_values=sample_value_tensors,
                data_sources=data_sources_flat,
                metric_prefix="val_value_model",
            )
            if self.use_hem and sample_value_oracle_tensors:
                oracle_metrics = self._evaluate_value_model_on_validation(
                    sample_uids=sample_uids,
                    sample_scores=sample_scores,
                    sample_values=sample_value_oracle_tensors,
                    data_sources=data_sources_flat,
                    metric_prefix="val_value_model_oracle",
                )
                value_model_eval_metrics.update(oracle_metrics)
            print(f"value model eval metrics: { {k: f'{v:.4f}' for k, v in value_model_eval_metrics.items() if 'accuracy' in k or 'gap' in k} }")

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "sample_response_lengths": sample_response_lengths,
                "reward_extra_infos_dict": reward_extra_infos_dict,
                "value_model_eval_metrics": value_model_eval_metrics,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_response_lengths,
        )
        metric_dict.update(value_model_eval_metrics)
        return metric_dict

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
        sample_response_lengths=None,
    ):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if sample_response_lengths is not None and len(sample_response_lengths) > 0:
            if len(sample_response_lengths) != len(data_sources):
                raise ValueError(
                    "Validation response lengths and data sources must have the same size: "
                    f"{len(sample_response_lengths)} != {len(data_sources)}"
                )
            data_source_to_response_lengths = defaultdict(list)
            for data_source, response_length in zip(data_sources, sample_response_lengths, strict=True):
                data_source_to_response_lengths[data_source].append(response_length)
            for data_source, response_lengths in data_source_to_response_lengths.items():
                metric_dict[f"val-aux/{data_source}/response_length/mean"] = float(np.mean(response_lengths))

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "sample_response_lengths": [],
                "reward_extra_infos_dict": {},
                "value_model_eval_metrics": {},
            }
        if result_b is None:
            result_b = {
                "data_sources": [],
                "sample_uids": [],
                "sample_turns": [],
                "sample_response_lengths": [],
                "reward_extra_infos_dict": {},
                "value_model_eval_metrics": {},
            }

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]
        sample_response_lengths = result_a.get("sample_response_lengths", []) + result_b.get(
            "sample_response_lengths", []
        )

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        metric_dict = self._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            sample_response_lengths,
        )

        # merge value model eval metrics (pre-computed per side, just combine)
        vm_a = result_a.get("value_model_eval_metrics", {})
        vm_b = result_b.get("value_model_eval_metrics", {})
        metric_dict.update(vm_a)
        metric_dict.update(vm_b)

        return metric_dict

    def _has_hf_model_weights(self, path: Optional[str]) -> bool:
        if not path:
            return False
        local_path = os.path.expanduser(path)
        if not os.path.isdir(local_path):
            return False
        weight_patterns = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "*.safetensors",
            "*.bin",
        )
        return os.path.exists(os.path.join(local_path, "config.json")) and any(
            glob(os.path.join(local_path, pattern)) for pattern in weight_patterns
        )

    def _has_any_sharded_critic_checkpoint_files(self, path: Optional[str]) -> bool:
        if not path:
            return False
        local_path = os.path.expanduser(path)
        return os.path.isdir(local_path) and bool(glob(os.path.join(local_path, "model_world_size_*_rank_*.pt")))

    def _hem_oracle_role(self) -> Role:
        return Role.CriticOracle if Role.CriticOracle in self.resource_pool_manager.mapping else Role.Critic

    def _resolve_hem_critic_init_path(self, source: str) -> str:
        """Resolve the HF/init path used to construct a HEM critic worker.

        A sharded FSDP checkpoint has the trainable weights but usually not a
        complete HF weight file. In that case we still need an HF/init path to
        build the model architecture before loading the shard.
        """
        if source == "standard":
            model_path = self.critic_standard_model_path
            explicit_init_path = self.critic_standard_model_init_path
            fallback_init_path = self.config.critic.model.path
            role = Role.Critic
        elif source == "oracle":
            model_path = self.critic_oracle_model_path
            explicit_init_path = self.critic_oracle_model_init_path
            fallback_init_path = None
            role = self._hem_oracle_role()
        else:
            raise ValueError(f"Unknown HEM critic source: {source}")

        if explicit_init_path:
            return os.path.expanduser(explicit_init_path)

        if not self._is_sharded_critic_checkpoint_dir(model_path, role=role):
            if self._has_any_sharded_critic_checkpoint_files(model_path):
                expected_world_size = self.resource_pool_manager.get_resource_pool(role).world_size
                raise ValueError(
                    f"HEM {source} critic checkpoint at {model_path} has FSDP shard files, but not a complete "
                    f"world_size={expected_world_size} shard set. Re-export it with the current critic world size "
                    f"or adjust trainer.n_gpus_per_node/trainer.nnodes."
                )
            return os.path.expanduser(model_path)

        hf_path = os.path.join(os.path.expanduser(model_path), "huggingface")
        if self._has_hf_model_weights(hf_path):
            return hf_path

        if fallback_init_path:
            print(
                f"[HEM] {source} critic checkpoint is sharded; using critic.model.path="
                f"{fallback_init_path} as the architecture init path."
            )
            return os.path.expanduser(fallback_init_path)

        raise ValueError(
            f"critic.{source}_model_path points to a sharded checkpoint ({model_path}), but no complete "
            f"HF model weights were found under {hf_path}. Set critic.{source}_model_init_path to the "
            f"HF/base value model path with the matching architecture."
        )

    def _critic_config_for_hem_source(self, source: str):
        cfg = deepcopy(self.config.critic)
        init_path = self._resolve_hem_critic_init_path(source)
        with open_dict(cfg):
            cfg.model.path = init_path
            if OmegaConf.select(cfg, "model_config") is not None:
                cfg.model_config.path = init_path
        print(f"[HEM] {source} critic init path: {init_path}")
        return cfg

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_config = self._critic_config_for_hem_source("standard") if self.use_hem else self.config.critic
            critic_cfg: CriticConfig = omega_conf_to_dataclass(critic_config)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

            if self.use_hem:
                oracle_resource_pool = self.resource_pool_manager.get_resource_pool(
                    self._hem_oracle_role()
                )
                oracle_critic_config = self._critic_config_for_hem_source("oracle")
                oracle_critic_cfg: CriticConfig = omega_conf_to_dataclass(oracle_critic_config)
                oracle_worker_cls = self.role_worker_mapping.get(Role.CriticOracle, self.role_worker_mapping[Role.Critic])
                critic_oracle_cls = RayClassWithInitArgs(cls=oracle_worker_cls, config=oracle_critic_cfg)
                self.resource_pool_to_cls[oracle_resource_pool][str(Role.CriticOracle)] = critic_oracle_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_hem:
                self.critic_oracle_wg = all_wg[str(Role.CriticOracle)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()
                if self.use_hem:
                    self.critic_oracle_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

            # When HEM is enabled, persist the independent oracle critic worker.
            if self.use_hem:
                critic_oracle_local_path = os.path.join(local_global_step_folder, "critic_oracle")
                critic_oracle_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(
                        self.config.trainer.default_hdfs_dir,
                        f"global_step_{self.global_steps}",
                        "critic_oracle",
                    )
                )
                self.critic_oracle_wg.save_checkpoint(
                    critic_oracle_local_path,
                    critic_oracle_remote_path,
                    self.global_steps,
                    max_ckpt_to_keep=max_critic_ckpt_to_keep,
                )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self._bootstrap_hem_critics_for_fresh_start()
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self._bootstrap_hem_critics_for_fresh_start()
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
            if self.use_hem:
                # Resume path already restores a trained standard critic state.
                self._hem_standard_bootstrapped = True

                # If a trained oracle critic was saved alongside, restore it into the
                # independent oracle worker. Otherwise keep the initial oracle model.
                critic_oracle_path = os.path.join(global_step_folder, "critic_oracle")
                if self._is_sharded_critic_checkpoint_dir(critic_oracle_path, role=self._hem_oracle_role()):
                    print(f"[HEM] restore oracle critic from {critic_oracle_path}")
                    self.critic_oracle_wg.load_checkpoint(
                        critic_oracle_path,
                        del_local_after_load=False,
                    )
                    self._hem_oracle_runtime_ready = True
                else:
                    print(f"[HEM] no oracle critic checkpoint at {critic_oracle_path}; using initial oracle critic.")
                    self._bootstrap_oracle_critic_for_hem()

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
                if self.use_hem and self.critic_oracle_wg is not None:
                    self.critic_oracle_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
                if self.use_hem and self.critic_oracle_wg is not None:
                    self.critic_oracle_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import \
                get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_position_ids_for_critic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch: DataProto,
    ) -> torch.Tensor:
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)

        multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs", None)
        if multi_modal_inputs is None:
            return compute_position_id_with_mask(attention_mask)

        position_ids_list = []
        for i in range(input_ids.shape[0]):
            mmi = multi_modal_inputs[i]
            if mmi is None:
                image_grid_thw = None
                video_grid_thw = None
            else:
                image_grid_thw = mmi.get("image_grid_thw")
                video_grid_thw = mmi.get("video_grid_thw")
            vision_position_ids, _ = self.processor.get_rope_index(
                input_ids=input_ids[i : i + 1],
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask[i : i + 1],
            )
            vision_position_ids = vision_position_ids.transpose(0, 1)  # (1, 3, seq_len)

            valid_mask = attention_mask[i].bool()
            text_position_ids = torch.ones((1, input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)
            text_position_ids = text_position_ids.unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_len)
            position_ids_list.append(position_ids)

        return torch.cat(position_ids_list, dim=0)

    def _prepare_actor_critic_inputs(self, batch: DataProto) -> None:
        if "input_ids_for_actor" not in batch.batch:
            batch.batch["input_ids_for_actor"] = batch.batch["input_ids"]
            batch.batch["attention_mask_for_actor"] = batch.batch["attention_mask"]
            batch.batch["position_ids_for_actor"] = batch.batch["position_ids"]
            if "prompts" in batch.batch:
                batch.batch["prompts_for_actor"] = batch.batch["prompts"]

        # Standard critic inputs always follow the original actor prompt + response.
        input_ids_for_critic_standard = batch.batch["input_ids_for_actor"]
        attention_mask_for_critic_standard = batch.batch["attention_mask_for_actor"]
        position_ids_for_critic_standard = batch.batch["position_ids_for_actor"]
        prompts_for_critic_standard = batch.batch.get("prompts_for_actor", None)

        # Oracle critic inputs come from dataset-prebuilt input_ids_with_answer when present.
        # When both switches are enabled, that tensor contains the answer-first
        # prompt followed by the reference solution:
        #   {question} The ground truth answer is {answer}.
        #
        #   A reference solution:
        #   {solution}
        # Prompt construction (field paths/templates) remains at dataset level.
        has_oracle_inputs = "input_ids_with_answer" in batch.batch
        prompt_ids = batch.batch["input_ids_with_answer"] if has_oracle_inputs else None
        prompt_attention_mask = (
            batch.batch.get("attention_mask_with_answer", None) if has_oracle_inputs else None
        )

        prompts_for_critic_oracle = prompts_for_critic_standard
        if has_oracle_inputs:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
            if pad_token_id is None:
                pad_token_id = 0

            if prompt_attention_mask is None:
                prompt_attention_mask = (prompt_ids != pad_token_id).to(batch.batch["attention_mask"].dtype)

            response_length = batch.batch["responses"].shape[-1]
            response_attention_mask = batch.batch["attention_mask_for_actor"][:, -response_length:]
            input_ids_for_critic_oracle = torch.cat([prompt_ids, batch.batch["responses"]], dim=1)
            attention_mask_for_critic_oracle = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
            position_ids_for_critic_oracle = self._compute_position_ids_for_critic(
                input_ids_for_critic_oracle, attention_mask_for_critic_oracle, batch
            )
            if prompts_for_critic_oracle is not None:
                prompts_for_critic_oracle = prompt_ids
        else:
            input_ids_for_critic_oracle = input_ids_for_critic_standard
            attention_mask_for_critic_oracle = attention_mask_for_critic_standard
            position_ids_for_critic_oracle = position_ids_for_critic_standard

        batch.batch["input_ids_for_critic_standard"] = input_ids_for_critic_standard
        batch.batch["attention_mask_for_critic_standard"] = attention_mask_for_critic_standard
        batch.batch["position_ids_for_critic_standard"] = position_ids_for_critic_standard
        batch.batch["input_ids_for_critic_oracle"] = input_ids_for_critic_oracle
        batch.batch["attention_mask_for_critic_oracle"] = attention_mask_for_critic_oracle
        batch.batch["position_ids_for_critic_oracle"] = position_ids_for_critic_oracle
        if prompts_for_critic_standard is not None:
            batch.batch["prompts_for_critic_standard"] = prompts_for_critic_standard
        if prompts_for_critic_oracle is not None:
            batch.batch["prompts_for_critic_oracle"] = prompts_for_critic_oracle

        # Backward-compatible default critic route.  The solution switch only
        # has an effect on top of use_answer_for_critic; both routes use the
        # dataset-prebuilt oracle tensor, so HEM and legacy answer-only configs
        # remain compatible.
        use_answer_for_critic = self.use_answer_for_critic
        use_solution_for_critic = self.use_solution_for_critic
        if use_solution_for_critic and not has_oracle_inputs:
            raise ValueError(
                "data.use_solution_for_critic=True requires input_ids_with_answer in batch. "
                "Check that data.use_answer_for_critic=True and the dataset constructed "
                "the answer-plus-solution critic inputs."
            )
        if self.use_hem:
            if not has_oracle_inputs:
                raise ValueError(
                    "algorithm.use_hem=True requires input_ids_with_answer in batch. "
                    "Check that data.use_answer_for_critic=True and the dataset constructed "
                    "the answer-augmented inputs."
                )
            default_source = "standard"
        elif use_answer_for_critic and has_oracle_inputs:
            default_source = "oracle"
        else:
            default_source = "standard"

        if default_source == "oracle":
            batch.batch["input_ids_for_critic"] = input_ids_for_critic_oracle
            batch.batch["attention_mask_for_critic"] = attention_mask_for_critic_oracle
            batch.batch["position_ids_for_critic"] = position_ids_for_critic_oracle
            if prompts_for_critic_oracle is not None:
                batch.batch["prompts_for_critic"] = prompts_for_critic_oracle
        else:
            batch.batch["input_ids_for_critic"] = input_ids_for_critic_standard
            batch.batch["attention_mask_for_critic"] = attention_mask_for_critic_standard
            batch.batch["position_ids_for_critic"] = position_ids_for_critic_standard
            if prompts_for_critic_standard is not None:
                batch.batch["prompts_for_critic"] = prompts_for_critic_standard

    def _activate_actor_inputs(self, batch: DataProto) -> None:
        if "input_ids_for_actor" not in batch.batch:
            return
        batch.batch["input_ids"] = batch.batch["input_ids_for_actor"]
        batch.batch["attention_mask"] = batch.batch["attention_mask_for_actor"]
        batch.batch["position_ids"] = batch.batch["position_ids_for_actor"]
        if "prompts_for_actor" in batch.batch:
            batch.batch["prompts"] = batch.batch["prompts_for_actor"]

    def _activate_critic_inputs(self, batch: DataProto, source: str = "default") -> None:
        source_to_suffix = {
            "default": "critic",
            "standard": "critic_standard",
            "oracle": "critic_oracle",
        }
        if source not in source_to_suffix:
            raise ValueError(f"Unknown critic input source: {source}")

        suffix = source_to_suffix[source]
        input_key = f"input_ids_for_{suffix}"
        attention_key = f"attention_mask_for_{suffix}"
        position_key = f"position_ids_for_{suffix}"
        prompt_key = f"prompts_for_{suffix}"

        if input_key not in batch.batch:
            return
        batch.batch["input_ids"] = batch.batch[input_key]
        batch.batch["attention_mask"] = batch.batch[attention_key]
        batch.batch["position_ids"] = batch.batch[position_key]
        if prompt_key in batch.batch:
            batch.batch["prompts"] = batch.batch[prompt_key]

    def _bootstrap_standard_critic_for_hem(self) -> None:
        if not self.use_hem:
            return
        if self._hem_standard_bootstrapped:
            return
        if self._is_sharded_critic_checkpoint_dir(self.critic_standard_model_path, role=Role.Critic):
            print(f"[HEM] load initial standard critic checkpoint: {self.critic_standard_model_path}")
            self.critic_wg.load_checkpoint(
                self.critic_standard_model_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        elif self._has_any_sharded_critic_checkpoint_files(self.critic_standard_model_path):
            raise ValueError(f"Incomplete or world-size mismatched standard critic checkpoint: {self.critic_standard_model_path}")
        else:
            print(f"[HEM] standard critic initialized from HF path: {self.critic_standard_model_path}")
        self._hem_standard_bootstrapped = True

    def _save_runtime_standard_critic_for_hem(self) -> None:
        """Compatibility no-op.

        HEM now keeps standard and oracle critics in independent workers, so no
        in-memory checkpoint swap is needed.
        """
        return None

    def _load_runtime_standard_critic_for_hem(self) -> None:
        """Compatibility no-op for the independent standard critic worker."""
        return None

    def _save_runtime_oracle_critic_for_hem(self) -> None:
        """Mark the independent oracle critic worker as initialized."""
        self._hem_oracle_runtime_ready = True

    def _load_oracle_critic_for_hem(self) -> None:
        """Ensure the independent oracle critic worker has its initial weights."""
        self._bootstrap_oracle_critic_for_hem()

    def _bootstrap_oracle_critic_for_hem(self) -> None:
        if not self.use_hem:
            return
        if self._hem_oracle_runtime_ready:
            return
        if self.critic_oracle_wg is None:
            raise RuntimeError("HEM oracle critic worker is not initialized.")
        if self._is_sharded_critic_checkpoint_dir(self.critic_oracle_model_path, role=self._hem_oracle_role()):
            print(f"[HEM] load initial oracle critic checkpoint: {self.critic_oracle_model_path}")
            self.critic_oracle_wg.load_checkpoint(
                self.critic_oracle_model_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        elif self._has_any_sharded_critic_checkpoint_files(self.critic_oracle_model_path):
            raise ValueError(f"Incomplete or world-size mismatched oracle critic checkpoint: {self.critic_oracle_model_path}")
        else:
            print(f"[HEM] oracle critic initialized from HF path: {self.critic_oracle_model_path}")
        self._hem_oracle_runtime_ready = True

    def _bootstrap_hem_critics_for_fresh_start(self) -> None:
        if not self.use_hem:
            return
        self._bootstrap_standard_critic_for_hem()
        self._bootstrap_oracle_critic_for_hem()

    def _save_hem_runtime_checkpoint(self, local_path: str) -> None:
        """Persist current critic state to disk. Used only for boot-time HF->sharded conversion
        and for save_checkpoint / load_checkpoint integration with the global trainer ckpt.
        """
        os.makedirs(local_path, exist_ok=True)
        # Do not rotate checkpoints for HEM internal runtime caches. Rotation history
        # is shared with regular training checkpoints and can accidentally delete
        # these runtime directories.
        self.critic_wg.save_checkpoint(
            local_path,
            None,
            self.global_steps,
            max_ckpt_to_keep=None,
        )
        try:
            self.critic_wg.clear_checkpoint_history()
        except Exception:
            # Backward compatible with workers without this RPC.
            pass

    def _is_sharded_critic_checkpoint_dir(self, path: Optional[str], role: Role = Role.Critic) -> bool:
        if not path:
            return False
        local_path = os.path.expanduser(path)
        if not os.path.isdir(local_path):
            return False
        shard_pattern = os.path.join(local_path, "model_world_size_*_rank_*.pt")
        model_files = glob(shard_pattern)
        if not model_files:
            return False

        expected_world_size = None
        if self.use_critic:
            expected_world_size = self.resource_pool_manager.get_resource_pool(role).world_size

        ws_to_ranks: dict[int, set[int]] = defaultdict(set)
        regex = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
        for fpath in model_files:
            matched = regex.search(os.path.basename(fpath))
            if not matched:
                continue
            ws = int(matched.group(1))
            rank = int(matched.group(2))
            ws_to_ranks[ws].add(rank)

        if not ws_to_ranks:
            return False

        for ws, ranks in ws_to_ranks.items():
            if expected_world_size is not None and ws != expected_world_size:
                continue
            if len(ranks) == ws and all(rank in ranks for rank in range(ws)):
                return True
        return False

    def _ensure_hem_model_paths_ready(self) -> None:
        """Compatibility guard for older HEM helper calls.

        HEM now constructs independent standard/oracle critic workers during
        init_workers(), so runtime HF->sharded conversion and checkpoint
        swapping are no longer needed.
        """
        if not self.use_hem or self._hem_model_paths_ready:
            return
        self._hem_standard_load_path = self.critic_standard_model_path
        self._hem_oracle_load_path = self.critic_oracle_model_path
        self._hem_model_paths_ready = True

    @staticmethod
    def _masked_sequence_mean(token_tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Summarize a token-level tensor into one scalar per response."""
        if token_tensor.dim() == 1:
            return token_tensor
        if token_tensor.shape != response_mask.shape:
            raise ValueError(f"token tensor shape {token_tensor.shape} must match response_mask {response_mask.shape}")
        mask = response_mask.to(device=token_tensor.device, dtype=token_tensor.dtype)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (token_tensor * mask).sum(dim=-1) / denom

    def _compute_token_hem_weights(
        self,
        values_standard: torch.Tensor,
        values_oracle: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute token-level discrepancy weights from per-token value gaps."""
        if values_standard.shape != values_oracle.shape:
            raise ValueError(
                f"values_standard shape {values_standard.shape} must match values_oracle shape {values_oracle.shape}"
            )
        if response_mask.shape != values_standard.shape:
            raise ValueError(f"response_mask shape {response_mask.shape} must match values shape {values_standard.shape}")

        with torch.no_grad():
            mask_bool = response_mask.bool()
            error = torch.abs(values_oracle - values_standard)
            error = torch.where(mask_bool, error, torch.zeros_like(error))

            weight = torch.ones_like(error)
            if mask_bool.any():
                valid_error = error[mask_bool]
                mu = valid_error.mean()
                std = valid_error.std(unbiased=False)
                normalized = (error - mu) / (std + self.hem_eps)
                sign = -1.0 if self.hem_invert_weight else 1.0
                raw_weight = 1 + sign * self.hem_alpha * normalized
                weight = torch.where(mask_bool, raw_weight, torch.ones_like(raw_weight))

            weight_clipped = torch.clamp(weight, min=self.hem_clip_min, max=self.hem_clip_max)
            weight_clipped = torch.nan_to_num(
                weight_clipped,
                nan=1.0,
                posinf=self.hem_clip_max,
                neginf=self.hem_clip_min,
            )

        return error, weight, weight_clipped

    def _compute_hem_weights_for_mode(
        self,
        values_standard: torch.Tensor,
        values_oracle: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hem_weight_mode == "sequence":
            return core_algos.compute_hem_weights(
                values_standard=values_standard,
                values_oracle=values_oracle,
                response_mask=response_mask,
                alpha=self.hem_alpha,
                eps=self.hem_eps,
                clip_min=self.hem_clip_min,
                clip_max=self.hem_clip_max,
                invert=self.hem_invert_weight,
            )
        if self.hem_weight_mode == "token":
            return self._compute_token_hem_weights(
                values_standard=values_standard,
                values_oracle=values_oracle,
                response_mask=response_mask,
            )
        raise ValueError(f"Unsupported HEM weight mode: {self.hem_weight_mode}")

    def _apply_hem_advantage_weight(self, batch: DataProto, apply_to_advantage: bool = True) -> dict[str, float]:
        if "values_standard" not in batch.batch or "values_oracle" not in batch.batch:
            raise ValueError("HEM requires both values_standard and values_oracle in batch.")
        if apply_to_advantage and "advantages" not in batch.batch:
            raise ValueError("HEM requires advantages in batch before reweighting.")

        values_standard = batch.batch["values_standard"]
        values_oracle = batch.batch["values_oracle"]
        response_mask = batch.batch["response_mask"]

        hem_error, hem_weight, hem_weight_clipped = self._compute_hem_weights_for_mode(
            values_standard=values_standard,
            values_oracle=values_oracle,
            response_mask=response_mask,
        )
        hem_error_sample = self._masked_sequence_mean(hem_error, response_mask)
        hem_weight_sample = self._masked_sequence_mean(hem_weight_clipped, response_mask)

        # When hem_skip_truncated is enabled, detect responses that reached the
        # length limit (used all tokens without an EOS) and override their weight to 1.0.
        if self.hem_skip_truncated:
            responses = batch.batch.get("responses", None)
            if responses is not None:
                eos_token_id = self.tokenizer.eos_token_id
                eos_token_set = (
                    eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                )
                response_length = responses.size(1)
                truncated_mask = torch.zeros(
                    responses.size(0), dtype=torch.bool, device=responses.device
                )

                for i in range(responses.size(0)):
                    valid_len = int(response_mask[i].sum().item())
                    # A response is truncated if it uses all available tokens and
                    # the last token is not EOS.
                    if valid_len == response_length:
                        last_token = int(responses[i, valid_len - 1].item())
                        if last_token not in eos_token_set:
                            truncated_mask[i] = True

                # Override weight to 1.0 for truncated responses
                truncated_weight_mask = truncated_mask.unsqueeze(-1) if hem_weight_clipped.dim() == 2 else truncated_mask
                hem_weight_clipped = torch.where(
                    truncated_weight_mask, torch.ones_like(hem_weight_clipped), hem_weight_clipped
                )
                hem_weight_sample = self._masked_sequence_mean(hem_weight_clipped, response_mask)
                num_truncated = int(truncated_mask.sum().item())
            else:
                num_truncated = 0
        else:
            num_truncated = 0

        # When hem_correct_only is enabled, only apply HEM weighting to correct
        # responses (sequence reward >= 0.5); incorrect responses keep weight=1.0.
        if self.hem_correct_only:
            token_level_scores = batch.batch.get("token_level_scores", None)
            if token_level_scores is not None:
                score_mask = response_mask.to(token_level_scores.dtype)
                seq_rewards = (token_level_scores * score_mask).sum(dim=-1)
                incorrect_mask = seq_rewards < 0.5
                incorrect_weight_mask = incorrect_mask.unsqueeze(-1) if hem_weight_clipped.dim() == 2 else incorrect_mask
                hem_weight_clipped = torch.where(
                    incorrect_weight_mask, torch.ones_like(hem_weight_clipped), hem_weight_clipped
                )
                hem_weight_sample = self._masked_sequence_mean(hem_weight_clipped, response_mask)
                num_incorrect = int(incorrect_mask.sum().item())
            else:
                num_incorrect = 0
        else:
            num_incorrect = 0

        if apply_to_advantage:
            advantage_weight = hem_weight_clipped.unsqueeze(-1) if hem_weight_clipped.dim() == 1 else hem_weight_clipped
            batch.batch["advantages"] = batch.batch["advantages"] * advantage_weight
        batch.batch["hem_error"] = hem_error
        batch.batch["hem_weight"] = hem_weight_clipped
        batch.batch["hem_error_sample"] = hem_error_sample
        batch.batch["hem_weight_sample"] = hem_weight_sample

        mask = response_mask.to(values_standard.dtype)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        value_standard_mean = (values_standard * mask).sum(dim=-1) / denom
        value_oracle_mean = (values_oracle * mask).sum(dim=-1) / denom
        if hem_error.dim() == 2:
            valid_error = torch.masked_select(hem_error, response_mask.bool())
            valid_weight = torch.masked_select(hem_weight_clipped, response_mask.bool())
        else:
            valid_sample = response_mask.bool().any(dim=-1)
            valid_error = hem_error[valid_sample]
            valid_weight = hem_weight_clipped[valid_sample]
        if valid_error.numel() == 0:
            valid_error = hem_error_sample
        if valid_weight.numel() == 0:
            valid_weight = hem_weight_sample

        metrics = {
            "hem_weight_mode_token": 1.0 if self.hem_weight_mode == "token" else 0.0,
            "hem_error_mean": valid_error.mean().detach().item(),
            "hem_error_std": valid_error.std(unbiased=False).detach().item(),
            "hem_error_sample_mean": hem_error_sample.mean().detach().item(),
            "hem_weight_mean": valid_weight.mean().detach().item(),
            "hem_weight_min": valid_weight.min().detach().item(),
            "hem_weight_max": valid_weight.max().detach().item(),
            "hem_weight_sample_mean": hem_weight_sample.mean().detach().item(),
            "value_standard_mean": value_standard_mean.mean().detach().item(),
            "value_oracle_mean": value_oracle_mean.mean().detach().item(),
        }

        if self.hem_skip_truncated and num_truncated > 0:
            metrics["hem_truncated_count"] = num_truncated
            metrics["hem_truncated_ratio"] = num_truncated / responses.size(0)

        if self.hem_correct_only:
            metrics["hem_incorrect_count"] = num_incorrect
            metrics["hem_incorrect_ratio"] = num_incorrect / response_mask.size(0)

        # Per-sample correlation between hem_error and response entropy.
        if "entropys" in batch.batch:
            with torch.no_grad():
                entropys = batch.batch["entropys"].detach()
                if entropys.shape == response_mask.shape:
                    entropy_mask = response_mask.to(entropys.dtype)
                    entropy_per_sample = (entropys * entropy_mask).sum(dim=-1) / denom
                    valid = denom > 0
                    if valid.sum().item() >= 2:
                        e = hem_error_sample[valid].float()
                        h = entropy_per_sample[valid].float()
                        e_std = e.std(unbiased=False)
                        h_std = h.std(unbiased=False)
                        if e_std.item() > 0 and h_std.item() > 0:
                            corr = torch.corrcoef(torch.stack([e, h], dim=0))[0, 1]
                            metrics["hem_error_entropy_pearson"] = corr.detach().item()
                        # Spearman: pearson on rank vectors.
                        e_rank = e.argsort().argsort().float()
                        h_rank = h.argsort().argsort().float()
                        if e_rank.std(unbiased=False).item() > 0 and h_rank.std(unbiased=False).item() > 0:
                            srho = torch.corrcoef(torch.stack([e_rank, h_rank], dim=0))[0, 1]
                            metrics["hem_error_entropy_spearman"] = srho.detach().item()
                        metrics["hem_entropy_per_sample_mean"] = h.mean().item()
                    batch.batch["entropy_per_sample"] = entropy_per_sample
        return metrics

    @staticmethod
    def _segment_means(values: list, num_segments: int = 5) -> list:
        """Split a per-token sequence into ``num_segments`` contiguous chunks by
        position and return the mean of each chunk.

        Empty chunks (when the sequence is shorter than ``num_segments``) yield
        ``None`` so the segment count is always stable.
        """
        if values is None or len(values) == 0:
            return [None] * num_segments
        chunks = np.array_split(np.asarray(values, dtype=np.float64), num_segments)
        return [float(c.mean()) if c.size > 0 else None for c in chunks]

    def _save_hem_samples_to_json(self, batch: DataProto, hem_samples_dir: str) -> None:
        """Save HEM samples with extreme errors to a per-step JSON file.

        For each step we group responses by sequence reward (0 vs 1) and, within
        each group, keep the 5 largest- and 5 smallest-error examples. To keep the
        files compact we no longer dump the full token / critic-score lists; the
        per-token ``values_standard`` / ``values_oracle`` sequences are compressed
        into the mean of 5 equal-length segments instead.

        Args:
            batch: DataProto containing batch data with HEM information
            hem_samples_dir: Directory path to save HEM sample JSON files
        """
        if "hem_error" not in batch.batch:
            return
        hem_error = batch.batch["hem_error"].detach()
        hem_error_sample = batch.batch.get("hem_error_sample", hem_error).detach()
        if hem_error_sample.dim() != 1:
            response_mask_for_summary = batch.batch.get("response_mask", None)
            if response_mask_for_summary is None:
                return
            hem_error_sample = self._masked_sequence_mean(hem_error_sample, response_mask_for_summary)
        if hem_error_sample.numel() == 0:
            return

        responses = batch.batch.get("responses", None)
        if responses is None or self.tokenizer is None:
            return

        # Prepare directory
        os.makedirs(hem_samples_dir, exist_ok=True)

        # Extract data
        hem_weight = batch.batch.get("hem_weight", None)
        hem_weight_sample = batch.batch.get("hem_weight_sample", hem_weight)
        if hem_weight_sample is not None:
            hem_weight_sample = hem_weight_sample.detach()
            if hem_weight_sample.dim() != 1:
                response_mask_for_summary = batch.batch.get("response_mask", None)
                if response_mask_for_summary is not None:
                    hem_weight_sample = self._masked_sequence_mean(hem_weight_sample, response_mask_for_summary)
        prompts = batch.batch.get("prompts", None)
        values_standard = batch.batch.get("values_standard", None)
        values_oracle = batch.batch.get("values_oracle", None)
        response_mask = batch.batch.get("response_mask", None)
        reward_model_info = batch.non_tensor_batch.get("reward_model", None)

        # Per-sample sequence reward (sum of token-level scores over the response).
        token_level_scores = batch.batch.get("token_level_scores", None)
        if token_level_scores is None:
            return
        if response_mask is not None:
            score_mask = response_mask.to(token_level_scores.dtype)
            seq_rewards = (token_level_scores * score_mask).sum(dim=-1)
        else:
            seq_rewards = token_level_scores.sum(dim=-1)
        seq_rewards = seq_rewards.detach().cpu()

        def _extract_sample_data(idx: int) -> dict:
            """Extract compact, segment-summarized data for a single sample."""
            sample_data = {
                "global_step": self.global_steps,
                "sample_idx": idx,
                "reward": float(seq_rewards[idx].item()),
                "hem_error": float(hem_error_sample[idx].item()),
                "hem_weight": float(hem_weight_sample[idx].item()) if hem_weight_sample is not None else None,
            }

            # Ground-truth / standard answer for the prompt.
            if reward_model_info is not None and idx < len(reward_model_info):
                rm_entry = reward_model_info[idx]
                if isinstance(rm_entry, dict):
                    sample_data["ground_truth"] = rm_entry.get("ground_truth", None)

            # Decode prompt (text only, no raw token list).
            if prompts is not None:
                prompt_tokens = prompts[idx].detach().cpu().tolist()
                sample_data["prompt_text"] = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            # Decode response (text only, no raw token list).
            response_tokens = responses[idx].detach().cpu().tolist()
            sample_data["response_text"] = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Value predictions over valid response tokens, summarized as the
            # mean of 5 equal-length segments instead of the full per-token list.
            if response_mask is not None:
                mask = response_mask[idx].bool()
                sample_data["response_length"] = int(mask.sum().item())
                if values_standard is not None:
                    valid_standard = values_standard[idx][mask].detach().cpu().tolist()
                    sample_data["values_standard_segment_means"] = self._segment_means(valid_standard)
                if values_oracle is not None:
                    valid_oracle = values_oracle[idx][mask].detach().cpu().tolist()
                    sample_data["values_oracle_segment_means"] = self._segment_means(valid_oracle)
                if hem_error.dim() == 2 and hem_error.shape == response_mask.shape:
                    valid_error = hem_error[idx][mask].detach().cpu().tolist()
                    sample_data["hem_error_segment_means"] = self._segment_means(valid_error)
                if hem_weight is not None and hem_weight.dim() == 2 and hem_weight.shape == response_mask.shape:
                    valid_weight = hem_weight[idx][mask].detach().cpu().tolist()
                    sample_data["hem_weight_segment_means"] = self._segment_means(valid_weight)

            return sample_data

        def _select_extremes(indices: list, k: int = 5) -> tuple:
            """Return (max_error_indices, min_error_indices) within ``indices``."""
            if not indices:
                return [], []
            errs = [(i, float(hem_error_sample[i].item())) for i in indices]
            errs_desc = sorted(errs, key=lambda x: x[1], reverse=True)
            errs_asc = sorted(errs, key=lambda x: x[1])
            max_idx = [i for i, _ in errs_desc[:k]]
            min_idx = [i for i, _ in errs_asc[:k]]
            return max_idx, min_idx

        # Group sample indices by sequence reward (0 vs 1). Use a 0.5 threshold so
        # slightly noisy scores still fall into the nearest binary bucket.
        n = hem_error_sample.shape[0]
        reward0_indices, reward1_indices = [], []
        for i in range(n):
            (reward1_indices if float(seq_rewards[i].item()) >= 0.5 else reward0_indices).append(i)

        result = {"global_step": self.global_steps}
        for group_name, group_indices in (("reward_0", reward0_indices), ("reward_1", reward1_indices)):
            max_idx, min_idx = _select_extremes(group_indices, k=5)
            result[group_name] = {
                "num_samples": len(group_indices),
                "max_error": [_extract_sample_data(i) for i in max_idx],
                "min_error": [_extract_sample_data(i) for i in min_idx],
            }

        filename = os.path.join(hem_samples_dir, f"step_{self.global_steps}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[HEM] Saved extreme samples (per reward group) to {filename}")

    def _log_hem_extreme_responses(self, batch: DataProto) -> None:
        """Print responses with max/min HEM error for quick step-level inspection."""
        if "hem_error" not in batch.batch:
            return
        hem_error = batch.batch["hem_error"].detach()
        hem_error_sample = batch.batch.get("hem_error_sample", hem_error).detach()
        if hem_error_sample.dim() != 1:
            response_mask = batch.batch.get("response_mask", None)
            if response_mask is None:
                return
            hem_error_sample = self._masked_sequence_mean(hem_error_sample, response_mask)
        if hem_error_sample.numel() == 0:
            return

        max_idx = int(torch.argmax(hem_error_sample).item())
        min_idx = int(torch.argmin(hem_error_sample).item())
        hem_weight = batch.batch.get("hem_weight_sample", batch.batch.get("hem_weight", None))
        if hem_weight is not None:
            hem_weight = hem_weight.detach()
            if hem_weight.dim() != 1:
                response_mask = batch.batch.get("response_mask", None)
                if response_mask is not None:
                    hem_weight = self._masked_sequence_mean(hem_weight, response_mask)
        max_error = float(hem_error_sample[max_idx].item())
        min_error = float(hem_error_sample[min_idx].item())
        max_weight = float(hem_weight[max_idx].item()) if hem_weight is not None else float("nan")
        min_weight = float(hem_weight[min_idx].item()) if hem_weight is not None else float("nan")

        responses = batch.batch.get("responses", None)
        if responses is None or self.tokenizer is None:
            print(
                f"[HEM][step={self.global_steps}] max idx={max_idx} err={max_error:.6f} w={max_weight:.6f}; "
                f"min idx={min_idx} err={min_error:.6f} w={min_weight:.6f}"
            )
            return

        def _decode_response(idx: int) -> str:
            text = self.tokenizer.decode(responses[idx].detach().cpu().tolist(), skip_special_tokens=True)
            text = text.replace("\n", "\\n")
            max_chars = 300
            if len(text) > max_chars:
                text = text[:max_chars] + "...(truncated)"
            return text

        max_response = _decode_response(max_idx)
        min_response = _decode_response(min_idx)
        print(
            f"[HEM][step={self.global_steps}] max idx={max_idx} err={max_error:.6f} w={max_weight:.6f} "
            f"response={max_response}"
        )
        print(
            f"[HEM][step={self.global_steps}] min idx={min_idx} err={min_error:.6f} w={min_weight:.6f} "
            f"response={min_response}"
        )

    def _compute_segmented_vf_explained_var_metrics(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor,
        metric_prefix: str,
    ) -> dict[str, float]:
        """Compute value explained variance over first/middle/last response thirds."""
        if values.shape != returns.shape or values.shape != response_mask.shape:
            return {}

        metrics: dict[str, float] = {}
        bucket_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.RESPONSE_THIRD_BUCKET_NAMES
        }
        bucket_returns: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.RESPONSE_THIRD_BUCKET_NAMES
        }

        with torch.no_grad():
            mask_bool = response_mask.bool()
            for idx in range(values.shape[0]):
                valid_values = values[idx][mask_bool[idx]]
                valid_returns = returns[idx][mask_bool[idx]]
                valid_len = int(valid_values.numel())
                if valid_len <= 0:
                    continue
                for bucket_name, start_idx, end_idx in self._response_third_slices(valid_len):
                    if end_idx <= start_idx:
                        continue
                    bucket_values[bucket_name].append(valid_values[start_idx:end_idx])
                    bucket_returns[bucket_name].append(valid_returns[start_idx:end_idx])

            for bucket_name in self.RESPONSE_THIRD_BUCKET_NAMES:
                if not bucket_values[bucket_name]:
                    continue
                valid_values = torch.cat(bucket_values[bucket_name])
                valid_returns = torch.cat(bucket_returns[bucket_name])
                explained_var = self._explained_variance_from_tensors(valid_values, valid_returns)
                if explained_var is not None:
                    metrics[f"{metric_prefix}/vf_explained_var/{bucket_name}"] = explained_var

        return metrics

    def _compute_value_monitor_metrics(self, batch: DataProto) -> dict[str, float]:
        """Compute value statistics by correctness and relative response position."""
        required_keys = ("values", "response_mask", "token_level_scores")
        if any(key not in batch.batch for key in required_keys):
            return {}

        values_for_standard_metrics = (
            batch.batch["values_standard"]
            if self.use_hem and "values_standard" in batch.batch
            else batch.batch["values"]
        )
        metrics = self._compute_segmented_token_monitor_metrics(
            token_tensor=values_for_standard_metrics.detach(),
            response_mask=batch.batch["response_mask"].bool(),
            token_level_scores=batch.batch["token_level_scores"],
            metric_prefix="critic/value_monitor",
        )
        if "returns" in batch.batch:
            metrics.update(
                self._compute_segmented_vf_explained_var_metrics(
                    values=values_for_standard_metrics.detach(),
                    returns=batch.batch["returns"].detach(),
                    response_mask=batch.batch["response_mask"].bool(),
                    metric_prefix="critic",
                )
            )

        if "values_oracle" in batch.batch:
            metrics.update(
                self._compute_segmented_token_monitor_metrics(
                    token_tensor=batch.batch["values_oracle"].detach(),
                    response_mask=batch.batch["response_mask"].bool(),
                    token_level_scores=batch.batch["token_level_scores"],
                    metric_prefix="critic_oracle/value_monitor",
                )
            )

            if "returns" in batch.batch:
                response_mask = batch.batch["response_mask"].bool()
                valid_returns = torch.masked_select(batch.batch["returns"], response_mask)
                valid_values_oracle = torch.masked_select(batch.batch["values_oracle"], response_mask)
                if valid_returns.numel() > 0:
                    return_diff_var_oracle = torch.var(valid_returns - valid_values_oracle)
                    return_var = torch.var(valid_returns)
                    metrics["critic_oracle/vf_explained_var"] = (
                        1.0 - return_diff_var_oracle / (return_var + 1e-5)
                    ).detach().item()
                    metrics.update(
                        self._compute_segmented_vf_explained_var_metrics(
                            values=batch.batch["values_oracle"].detach(),
                            returns=batch.batch["returns"].detach(),
                            response_mask=response_mask,
                            metric_prefix="critic_oracle",
                        )
                    )

        return metrics

    def _compute_advantage_monitor_metrics(self, batch: DataProto) -> dict[str, float]:
        """Compute advantage statistics by correctness and relative response position."""
        required_keys = ("advantages", "response_mask", "token_level_scores")
        if any(key not in batch.batch for key in required_keys):
            return {}

        return self._compute_segmented_token_monitor_metrics(
            token_tensor=batch.batch["advantages"].detach(),
            response_mask=batch.batch["response_mask"].bool(),
            token_level_scores=batch.batch["token_level_scores"],
            metric_prefix="actor/adv_monitor",
        )

    def _compute_hem_entropy_monitor_metrics(self, batch: DataProto) -> dict[str, float]:
        """Segment entropy and per-token value gap by response position when HEM is on.

        Uses the same response-third buckets as adv_monitor so trends can be
        compared side by side. The token-level "hem error" used here is
        |values_oracle - values_standard| per token, the unaggregated form of the
        per-sample hem_error scalar.
        """
        if not self.use_hem:
            return {}
        required_keys = ("response_mask", "token_level_scores")
        if any(key not in batch.batch for key in required_keys):
            return {}

        response_mask = batch.batch["response_mask"].bool()
        token_level_scores = batch.batch["token_level_scores"]
        metrics: dict[str, float] = {}

        if "entropys" in batch.batch:
            metrics.update(
                self._compute_segmented_token_monitor_metrics(
                    token_tensor=batch.batch["entropys"].detach(),
                    response_mask=response_mask,
                    token_level_scores=token_level_scores,
                    metric_prefix="actor/hem_entropy_monitor",
                )
            )

        if "values_standard" in batch.batch and "values_oracle" in batch.batch:
            token_value_gap = (
                batch.batch["values_oracle"].detach() - batch.batch["values_standard"].detach()
            ).abs()
            metrics.update(
                self._compute_segmented_token_monitor_metrics(
                    token_tensor=token_value_gap,
                    response_mask=response_mask,
                    token_level_scores=token_level_scores,
                    metric_prefix="critic/hem_error_monitor",
                )
            )
        return metrics

    def _compute_segmented_token_monitor_metrics(
        self,
        token_tensor: torch.Tensor,
        response_mask: torch.Tensor,
        token_level_scores: torch.Tensor,
        metric_prefix: str,
    ) -> dict[str, float]:
        """Compute per-group segmented means for a token-level tensor."""
        values = token_tensor

        if values.shape != response_mask.shape or token_level_scores.shape != response_mask.shape:
            return {}

        with torch.no_grad():
            score_mask = response_mask.to(token_level_scores.dtype)
            sample_scores = (token_level_scores * score_mask).sum(dim=-1)
            group_masks = {
                "correct": sample_scores > 0,
                "incorrect": sample_scores <= 0,
            }
            metrics: dict[str, float] = {}
            for group_name, group_mask in group_masks.items():
                sample_indices = torch.nonzero(group_mask, as_tuple=False).flatten().tolist()
                metrics[f"{metric_prefix}/{group_name}/sample_count"] = float(len(sample_indices))
                if not sample_indices:
                    continue

                bucket_values: dict[str, list[torch.Tensor]] = {
                    name: [] for name in self.RESPONSE_THIRD_BUCKET_NAMES
                }
                bucket_values["last5"] = []

                for idx in sample_indices:
                    valid_values = values[idx][response_mask[idx]]
                    valid_len = int(valid_values.numel())
                    if valid_len <= 0:
                        continue

                    for name, start_idx, end_idx in self._response_third_slices(valid_len):
                        if end_idx <= start_idx:
                            continue
                        bucket_values[name].append(valid_values[start_idx:end_idx])

                    tail_len = min(5, valid_len)
                    bucket_values["last5"].append(valid_values[-tail_len:])

                for bucket_name, segments in bucket_values.items():
                    if not segments:
                        continue
                    metrics[f"{metric_prefix}/{group_name}/{bucket_name}_mean"] = (
                        torch.cat(segments).mean().detach().item()
                    )
        return metrics

    def _evaluate_value_model_on_validation(
        self,
        sample_uids: list,
        sample_scores: list[float],
        sample_values: list[torch.Tensor],
        data_sources: np.ndarray,
        last_k_tokens: int = 5,
        metric_prefix: str = "val_value_model",
    ) -> dict[str, float]:
        """Evaluate value model ranking ability on validation data.

        References value_pretraining/analysis/evaluate_value_model.py for
        segment-based scoring and best-of-N accuracy computation.
        """
        from collections import OrderedDict

        methods = list(self.RESPONSE_THIRD_BUCKET_NAMES)
        methods.append(f"last_{last_k_tokens}_tokens")
        methods.append("ratio_0_1")

        def _segment_scores(token_scores: torch.Tensor) -> dict[str, float]:
            length = token_scores.numel()
            if length <= 0:
                return {m: 0.0 for m in methods}
            out: dict[str, float] = {}
            full_score = float(token_scores.mean().item())
            for name, start_idx, end_idx in self._response_third_slices(length):
                out[name] = (
                    float(token_scores[start_idx:end_idx].mean().item())
                    if end_idx > start_idx
                    else full_score
                )
            k = max(1, min(last_k_tokens, length))
            out[f"last_{last_k_tokens}_tokens"] = float(token_scores[-k:].mean().item())
            out["ratio_0_1"] = full_score
            return out

        score_by_idx: dict[int, dict[str, float]] = {}
        for i, vals in enumerate(sample_values):
            score_by_idx[i] = _segment_scores(vals)

        labels = [1 if s > 0 else 0 for s in sample_scores]

        uid_to_indices: OrderedDict[str, list[int]] = OrderedDict()
        for i, uid in enumerate(sample_uids):
            uid_to_indices.setdefault(uid, []).append(i)
        groups = list(uid_to_indices.values())

        mixed_groups = [
            g for g in groups
            if any(labels[i] > 0 for i in g) and any(labels[i] <= 0 for i in g)
        ]

        metrics: dict[str, float] = {
            f"{metric_prefix}/num_groups_total": float(len(groups)),
            f"{metric_prefix}/num_groups_mixed": float(len(mixed_groups)),
        }

        if not mixed_groups:
            return metrics

        # --- overall accuracy per method ---
        for method in methods:
            correct = sum(
                1 for g in mixed_groups
                if labels[max(g, key=lambda idx: score_by_idx[idx][method])] > 0
            )
            total = len(mixed_groups)
            metrics[f"{metric_prefix}/{method}/accuracy"] = correct / total
            metrics[f"{metric_prefix}/{method}/correct"] = float(correct)
            metrics[f"{metric_prefix}/{method}/total"] = float(total)

        # --- mean value for correct vs incorrect ---
        correct_vals = [score_by_idx[i]["ratio_0_1"] for i, lb in enumerate(labels) if lb > 0]
        incorrect_vals = [score_by_idx[i]["ratio_0_1"] for i, lb in enumerate(labels) if lb <= 0]
        if correct_vals:
            metrics[f"{metric_prefix}/mean_score_correct"] = float(np.mean(correct_vals))
        if incorrect_vals:
            metrics[f"{metric_prefix}/mean_score_incorrect"] = float(np.mean(incorrect_vals))
        if correct_vals and incorrect_vals:
            metrics[f"{metric_prefix}/score_gap"] = float(np.mean(correct_vals) - np.mean(incorrect_vals))

        last_method = f"last_{last_k_tokens}_tokens"
        correct_last = [score_by_idx[i][last_method] for i, lb in enumerate(labels) if lb > 0]
        incorrect_last = [score_by_idx[i][last_method] for i, lb in enumerate(labels) if lb <= 0]
        if correct_last:
            metrics[f"{metric_prefix}/last_k_mean_correct"] = float(np.mean(correct_last))
        if incorrect_last:
            metrics[f"{metric_prefix}/last_k_mean_incorrect"] = float(np.mean(incorrect_last))
        if correct_last and incorrect_last:
            metrics[f"{metric_prefix}/last_k_score_gap"] = float(np.mean(correct_last) - np.mean(incorrect_last))

        # --- per-dataset metrics (all methods + score stats) ---
        ds_to_group_indices: OrderedDict[str, list[int]] = OrderedDict()
        for gi, g in enumerate(mixed_groups):
            ds = str(data_sources[g[0]]) if g else "unknown"
            ds_to_group_indices.setdefault(ds, []).append(gi)

        ds_to_row_indices: OrderedDict[str, list[int]] = OrderedDict()
        for i, ds in enumerate(data_sources):
            ds_str = str(ds)
            ds_to_row_indices.setdefault(ds_str, []).append(i)

        for ds, gis in ds_to_group_indices.items():
            ds_total = len(gis)
            if ds_total == 0:
                continue

            for method in methods:
                ds_correct = sum(
                    1 for gi in gis
                    if labels[max(mixed_groups[gi], key=lambda idx: score_by_idx[idx][method])] > 0
                )
                metrics[f"{metric_prefix}/{ds}/{method}/accuracy"] = ds_correct / ds_total

            ds_rows = ds_to_row_indices.get(ds, [])
            ds_correct_vals = [score_by_idx[i]["ratio_0_1"] for i in ds_rows if labels[i] > 0]
            ds_incorrect_vals = [score_by_idx[i]["ratio_0_1"] for i in ds_rows if labels[i] <= 0]
            if ds_correct_vals:
                metrics[f"{metric_prefix}/{ds}/mean_score_correct"] = float(np.mean(ds_correct_vals))
            if ds_incorrect_vals:
                metrics[f"{metric_prefix}/{ds}/mean_score_incorrect"] = float(np.mean(ds_incorrect_vals))
            if ds_correct_vals and ds_incorrect_vals:
                metrics[f"{metric_prefix}/{ds}/score_gap"] = float(
                    np.mean(ds_correct_vals) - np.mean(ds_incorrect_vals)
                )

            ds_correct_last = [score_by_idx[i][last_method] for i in ds_rows if labels[i] > 0]
            ds_incorrect_last = [score_by_idx[i][last_method] for i in ds_rows if labels[i] <= 0]
            if ds_correct_last:
                metrics[f"{metric_prefix}/{ds}/last_k_mean_correct"] = float(np.mean(ds_correct_last))
            if ds_incorrect_last:
                metrics[f"{metric_prefix}/{ds}/last_k_mean_incorrect"] = float(np.mean(ds_incorrect_last))
            if ds_correct_last and ds_incorrect_last:
                metrics[f"{metric_prefix}/{ds}/last_k_score_gap"] = float(
                    np.mean(ds_correct_last) - np.mean(ds_incorrect_last)
                )

        return metrics

    def _evaluate_value_model_on_train_batch(self, batch: DataProto) -> dict[str, float]:
        """Evaluate value model path-selection accuracy on the current training batch.

        Reuses _evaluate_value_model_on_validation with training-batch data.
        When HEM is enabled, evaluates both standard and oracle value models.
        """
        required_keys = ("values", "response_mask", "token_level_scores")
        if any(key not in batch.batch for key in required_keys):
            return {}
        if "uid" not in batch.non_tensor_batch:
            return {}

        response_mask = batch.batch["response_mask"].bool()
        sample_uids = list(batch.non_tensor_batch["uid"])
        sample_scores = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().tolist()
        data_sources = batch.non_tensor_batch.get("data_source", np.array(["unknown"] * len(sample_uids)))

        # Standard value model. Under HEM, batch["values"] is the oracle
        # baseline for actor advantages, so read the standard branch explicitly.
        values_key = "values_standard" if self.use_hem and "values_standard" in batch.batch else "values"
        values = batch.batch[values_key].detach()
        sample_values = [values[i][response_mask[i]].cpu().float() for i in range(values.size(0))]

        metrics = self._evaluate_value_model_on_validation(
            sample_uids=sample_uids,
            sample_scores=sample_scores,
            sample_values=sample_values,
            data_sources=data_sources,
            metric_prefix="train_value_model",
        )

        # Oracle value model (HEM)
        if self.use_hem and "values_oracle" in batch.batch:
            values_oracle = batch.batch["values_oracle"].detach()
            sample_values_oracle = [
                values_oracle[i][response_mask[i]].cpu().float() for i in range(values_oracle.size(0))
            ]
            oracle_metrics = self._evaluate_value_model_on_validation(
                sample_uids=sample_uids,
                sample_scores=sample_scores,
                sample_values=sample_values_oracle,
                data_sources=data_sources,
                metric_prefix="train_value_model_oracle",
            )
            metrics.update(oracle_metrics)

        return metrics

    def _critic_wg_for_source(self, source: str = "standard"):
        if self.use_hem and source == "oracle":
            if self.critic_oracle_wg is None:
                raise RuntimeError("HEM oracle critic worker is not initialized.")
            return self.critic_oracle_wg
        return self.critic_wg

    def _compute_values(self, batch: DataProto, source: str = "standard") -> DataProto:
        critic_wg = self._critic_wg_for_source(source)
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            if self.config.critic.get("use_value_sigmoid", True):
                values = torch.sigmoid(values)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = critic_wg.compute_values(batch)
        return values

    def _compute_value_predictions_for_batch(self, batch: DataProto) -> DataProto:
        """Compute value predictions on the currently selected critic route."""
        if self.use_hem:
            # 1) Standard value model on original critic inputs.
            self._activate_critic_inputs(batch, source="standard")
            values_standard = self._compute_values(batch, source="standard")
            batch.batch["values"] = values_standard.batch["values"]
            batch.batch["values_standard"] = values_standard.batch["values"].clone()

            # 2) Oracle value model on with-answer critic inputs.
            self._activate_critic_inputs(batch, source="oracle")
            values_oracle = self._compute_values(batch, source="oracle")
            batch.batch["values_oracle"] = values_oracle.batch["values"].clone()

            # HEM uses the oracle critic as the actor baseline. Keep
            # values_standard for HEM error and standard-critic value clipping.
            batch.batch["values"] = batch.batch["values_oracle"]
        else:
            self._activate_critic_inputs(batch)
            values = self._compute_values(batch)
            batch.batch["values"] = values.batch["values"]
        self._activate_actor_inputs(batch)
        return batch

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto, source: str = "standard") -> DataProto:
        critic_wg = self._critic_wg_for_source(source)
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = critic_wg.update_critic(batch)
        return critic_output

    def _run_critic_update_step(self, batch: DataProto, skip_standard: bool = False) -> dict[str, float]:
        """Run one critic update step (including optional oracle branch for HEM).

        Args:
            batch: DataProto containing training batch data.
            skip_standard: If True and HEM is enabled, skip the standard critic update
                and only update the oracle critic. Requires use_hem=True.
        """
        if skip_standard and not self.use_hem:
            raise ValueError("skip_standard=True requires use_hem=True.")

        critic_output_metrics: dict[str, float] = {}
        values_to_restore = batch.batch.get("values", None)

        if not skip_standard:
            self._activate_critic_inputs(batch, source="standard" if self.use_hem else "default")
            if self.use_hem and "values_standard" in batch.batch:
                batch.batch["values"] = batch.batch["values_standard"]
            critic_output = self._update_critic(batch, source="standard")
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])

        if self.use_hem:
            try:
                # Train oracle critic on oracle inputs with the same PPO targets.
                self._activate_critic_inputs(batch, source="oracle")
                batch.batch["values"] = batch.batch["values_oracle"]

                oracle_output = self._update_critic(batch, source="oracle")
                oracle_metrics_raw = reduce_metrics(oracle_output.meta_info["metrics"])
                oracle_metrics = {}
                for key, val in oracle_metrics_raw.items():
                    if key.startswith("critic/"):
                        oracle_metrics[key.replace("critic/", "critic_oracle/", 1)] = val
                    elif key == "perf/mfu/critic":
                        oracle_metrics["perf/mfu/critic_oracle"] = val
                    else:
                        oracle_metrics[key] = val
                critic_output_metrics.update(oracle_metrics)
                self._save_runtime_oracle_critic_for_hem()
            finally:
                if values_to_restore is not None:
                    batch.batch["values"] = values_to_restore

        self._activate_actor_inputs(batch)
        return critic_output_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        filter_groups_cfg = self.config.algorithm.get("filter_groups", None)
        enable_filter_groups = filter_groups_cfg is not None and filter_groups_cfg.enable
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                new_batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # Self-distillation: cache GT privileged text while vLLM is still warm.
                    # No-op when not patched (vanilla/HEM training).
                    if hasattr(self, "_generate_rationales"):
                        self._cached_rationales = self._generate_rationales(new_batch, gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # Compute or extract reward for training
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=new_batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                new_batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                    new_batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    num_gen_batches += 1
                    if not enable_filter_groups:
                        batch = new_batch
                    else:
                        metric_name = filter_groups_cfg.metric
                        if metric_name is None:
                            raise ValueError("algorithm.filter_groups.metric must be set when filter_groups is enabled.")

                        if metric_name == "seq_final_reward":
                            metric_tensor = (
                                new_batch.batch["token_level_rewards"]
                                if "token_level_rewards" in new_batch.batch.keys()
                                else new_batch.batch["token_level_scores"]
                            )
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                metric_tensor.sum(dim=-1).detach().cpu().numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
                            )

                        if metric_name not in new_batch.non_tensor_batch:
                            raise ValueError(
                                f"filter_groups.metric={metric_name} not found in batch non_tensor keys: "
                                f"{list(new_batch.non_tensor_batch.keys())}"
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=False
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {
                            prompt_uid: np.std(metric_vals) for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
                        }
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_prompt_uid_set = set(kept_prompt_uids)
                        kept_traj_idxs = [
                            idx
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if traj_from_prompt_uid in kept_prompt_uid_set
                        ]
                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = filter_groups_cfg.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                with marked_timer("stop_profile", timing_raw):
                                    self._stop_profiling(curr_step_profile)
                                continue
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. "
                                "Please check your data."
                            )

                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = batch[:traj_bsz]

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    if self.use_critic:
                        # Decouple actor/critic inputs; critic may use standard answers if provided.
                        self._prepare_actor_critic_inputs(batch)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import \
                            apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            # Keep per-token entropys when HEM is on so we can correlate them
                            # with hem_error and segment them by response position later.
                            if not self.use_hem:
                                old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import \
                                    calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    run_pre_update_value_before_adv = self.use_critic and self.pre_update_value_before_adv

                    # compute values (first pass)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            batch = self._compute_value_predictions_for_batch(batch)

                    # VASPO prepares teacher data after value inference. By default it
                    # scores now; old-student fusion runs globally and current-student
                    # fusion runs in the actor. Fresh-teacher mode only builds inputs
                    # here and scores/fuses each micro-batch during actor optimization.
                    if hasattr(self, "_compute_vaspo_teacher_scores"):
                        with marked_timer("vaspo_teacher", timing_raw, color="blue"):
                            batch, vaspo_teacher_metrics = self._compute_vaspo_teacher_scores(batch)
                            metrics.update(vaspo_teacher_metrics)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import \
                                compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        policy_loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get(
                            "loss_mode", "vanilla"
                        )
                        defer_advantage_whitening = policy_loss_mode == "vaspo"
                        vaspo_whiten_before_fusion = defer_advantage_whitening and bool(
                            self.config.actor_rollout_ref.actor.policy_loss.get("vaspo_whiten_before_fusion", False)
                        )
                        metrics["actor/advantage_normalization_enabled"] = float(
                            self.normalize_advantages or vaspo_whiten_before_fusion
                        )

                        # When hem_apply_before_whiten is enabled, compute HEM weights first
                        # and pass them into GAE so they are applied before masked_whiten.
                        hem_sample_weights = None
                        if self.use_hem and self.hem_apply_before_whiten:
                            hem_metrics = self._apply_hem_advantage_weight(batch, apply_to_advantage=False)
                            metrics.update(hem_metrics)
                            self._log_hem_extreme_responses(batch)
                            if self.hem_samples_dir:
                                self._save_hem_samples_to_json(batch, self.hem_samples_dir)
                            hem_sample_weights = batch.batch["hem_weight"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            sample_weights=hem_sample_weights,
                            whiten_advantages=self.normalize_advantages and not defer_advantage_whitening,
                        )

                        if run_pre_update_value_before_adv:
                            # Pre-update critic once, then recompute values/advantages with the updated critic.
                            with marked_timer("update_critic_pre_adv", timing_raw, color="pink"):
                                skip_standard = self.use_hem and self.hem_pre_update_oracle_only
                                critic_output_metrics = self._run_critic_update_step(
                                    batch, skip_standard=skip_standard
                                )
                                metrics.update(critic_output_metrics)
                            with marked_timer("values_post_pre_adv", timing_raw, color="cyan"):
                                batch = self._compute_value_predictions_for_batch(batch)

                            # Recompute HEM weights after value re-prediction
                            if self.use_hem and self.hem_apply_before_whiten:
                                hem_metrics = self._apply_hem_advantage_weight(batch, apply_to_advantage=False)
                                metrics.update(hem_metrics)
                                hem_sample_weights = batch.batch["hem_weight"]

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                                sample_weights=hem_sample_weights,
                                whiten_advantages=self.normalize_advantages and not defer_advantage_whitening,
                            )

                        if defer_advantage_whitening:
                            if not hasattr(self, "_apply_vaspo_advantage_fusion"):
                                raise RuntimeError(
                                    "VASPO requires its advantage-fusion hook to be installed"
                                )
                            batch, vaspo_fusion_metrics = self._apply_vaspo_advantage_fusion(batch)
                            metrics.update(vaspo_fusion_metrics)

                        if self.use_critic:
                            metrics.update(self._compute_value_monitor_metrics(batch))
                            metrics.update(self._evaluate_value_model_on_train_batch(batch))

                        if self.use_hem and not self.hem_apply_before_whiten:
                            hem_metrics = self._apply_hem_advantage_weight(batch)
                            metrics.update(hem_metrics)
                            self._log_hem_extreme_responses(batch)
                            if self.hem_samples_dir:
                                self._save_hem_samples_to_json(batch, self.hem_samples_dir)

                        metrics.update(self._compute_advantage_monitor_metrics(batch))

                        if self.use_hem:
                            metrics.update(self._compute_hem_entropy_monitor_metrics(batch))
                            # Precompute per-sample segmented stats while entropys / advantages are
                            # still on the batch, so _log_rollout_data can read them after the
                            # actor update (which expects entropys to be popped).
                            response_mask = batch.batch.get("response_mask", None)
                            if "entropys" in batch.batch and response_mask is not None:
                                self._hem_rollout_dump_cache["entropy_segments"] = (
                                    self._per_sample_segmented_means(
                                        batch.batch["entropys"].detach(), response_mask
                                    )
                                )
                            if "advantages" in batch.batch and response_mask is not None:
                                self._hem_rollout_dump_cache["advantage_segments"] = (
                                    self._per_sample_segmented_means(
                                        batch.batch["advantages"].detach(), response_mask
                                    )
                                )
                            if "hem_error" in batch.batch:
                                hem_error_for_dump = batch.batch.get("hem_error_sample", batch.batch["hem_error"])
                                self._hem_rollout_dump_cache["hem_error"] = (
                                    hem_error_for_dump.detach().cpu().tolist()
                                )
                            # Free per-token entropy now that monitors have consumed it.
                            batch.batch.pop("entropys", None)
                            batch.batch.pop("entropy_per_sample", None)

                    # update critic
                    if self.use_critic and not run_pre_update_value_before_adv:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output_metrics = self._run_critic_update_step(batch)
                            metrics.update(critic_output_metrics)
                    elif self.use_critic and run_pre_update_value_before_adv and self.hem_pre_update_oracle_only:
                        # Standard critic was skipped during pre-update; train it now.
                        with marked_timer("update_critic_standard", timing_raw, color="pink"):
                            self._activate_critic_inputs(batch)
                            values_to_restore = batch.batch.get("values", None)
                            if self.use_hem and "values_standard" in batch.batch:
                                batch.batch["values"] = batch.batch["values_standard"]
                            critic_output = self._update_critic(batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                            if values_to_restore is not None:
                                batch.batch["values"] = values_to_restore
                            self._activate_actor_inputs(batch)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Self-distillation: compute teacher log probs before actor update
                        # (installed by the main_ppo SD integration hooks)
                        skip_actor_update = False
                        policy_loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get(
                            "loss_mode", "vanilla"
                        )
                        if hasattr(self, '_compute_teacher_log_probs') and policy_loss_mode != "vaspo":
                            teacher_lp = self._compute_teacher_log_probs(batch)
                            if teacher_lp is not None:
                                batch = batch.union(teacher_lp)
                            skip_actor_update = batch.meta_info.pop("skip_actor_update", False)

                        # update actor
                        actor_output_metrics = {}
                        if not skip_actor_update:
                            actor_batch = batch
                            if hasattr(self, "_prepare_actor_update_batch"):
                                actor_batch = self._prepare_actor_update_batch(batch)
                            if self.kpop_config is not None:
                                metrics.update(apply_kpop(actor_batch, self.kpop_config))
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(actor_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        else:
                            metrics["actor/skip_update"] = 1.0
                        metrics.update(actor_output_metrics)

                    # RLCSD/self-distillation diagnostics are monkey-patched by
                    # the main_ppo SD integration. Keep the hooks optional so the
                    # regular PPO/HEM path is unchanged.
                    if hasattr(self, "_save_train_rollout"):
                        self._save_train_rollout(batch)
                    if hasattr(self, "_judge_rationales"):
                        metrics.update(self._judge_rationales())

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                n_samples = batch.batch.batch_size[0]
                rollout_n = self.config.actor_rollout_ref.rollout.n
                metrics.update(
                    {
                        "rollout/num_queries": n_samples // rollout_n if rollout_n else 0,
                        "rollout/num_samples": n_samples,
                    }
                )
                data_metrics = {} if n_samples == 0 else compute_data_metrics(batch=batch, use_critic=self.use_critic)
                if hasattr(self, "_get_data_metric_overrides"):
                    data_metric_overrides = self._get_data_metric_overrides()
                    if data_metric_overrides:
                        data_metrics.update(data_metric_overrides)
                metrics.update(data_metrics)
                teacher_trunc_metric_keys = (
                    "teacher_prompt_truncated",
                    "teacher_prompt_truncated_tokens",
                    "teacher_prompt_tokens_before_trunc",
                    "teacher_prompt_tokens_after_trunc",
                )
                for key in teacher_trunc_metric_keys:
                    if key in batch.non_tensor_batch:
                        vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                        if vals.size > 0:
                            metric_prefix = f"teacher/{key}"
                            metrics[f"{metric_prefix}/mean"] = float(np.mean(vals))
                            metrics[f"{metric_prefix}/std"] = float(np.std(vals))
                if "teacher_prompt_truncated" in batch.non_tensor_batch:
                    vals = np.asarray(batch.non_tensor_batch["teacher_prompt_truncated"], dtype=np.float32)
                    if vals.size > 0:
                        metrics["teacher/prompt_truncation_ratio"] = float(np.mean(vals))
                if "teacher_prompt_truncated_tokens" in batch.non_tensor_batch:
                    vals = np.asarray(batch.non_tensor_batch["teacher_prompt_truncated_tokens"], dtype=np.float32)
                    if vals.size > 0:
                        metrics["teacher/prompt_truncated_tokens_mean"] = float(np.mean(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation
                metrics["train/num_gen_batches"] = num_gen_batches

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
