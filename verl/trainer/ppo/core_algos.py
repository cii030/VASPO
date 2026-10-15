# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import math
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def _apply_advantage_weights(
    advantages: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply either sequence-level or token-level weights to advantages."""
    weights = weights.to(device=advantages.device, dtype=advantages.dtype)
    if weights.dim() == 1:
        if weights.shape[0] != advantages.shape[0]:
            raise ValueError(f"weights shape {weights.shape} must match batch size {advantages.shape[0]}")
        weights = weights.unsqueeze(-1)
    elif weights.shape != advantages.shape:
        raise ValueError(f"token weights shape {weights.shape} must match advantages shape {advantages.shape}")
    return advantages * weights


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    DECOUPLED_GAE = "decoupled_gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    whiten_advantages: bool = True,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        sample_weights: `(torch.Tensor | None)`
            Optional sequence-level or token-level weights (e.g., HEM weights) to apply before masked_whiten.
            Shape is (bs,) or (bs, response_length). If provided, advantages are multiplied before whitening.
        whiten_advantages: `(bool)`
            Whether to whiten advantages before returning them. VASPO disables this so it can fuse the
            privileged-teacher signal first and apply the standard masked whitening exactly once afterward.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values

        if sample_weights is not None:
            advantages = _apply_advantage_weights(advantages, sample_weights)

        if whiten_advantages:
            advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.DECOUPLED_GAE)  # or simply: @register_adv_est("decoupled_gae")
def compute_decoupled_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam_actor: float = 0.99,
    lam_critic: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    whiten_advantages: bool = True,
):
    """Decoupled GAE: use different lambda for actor advantage and critic target.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam_actor: `(float)`
            lambda value when computing actor advantage (GAE)
        lam_critic: `(float)`
            lambda value when computing critic target V^{target}(s_t)
        sample_weights: `(torch.Tensor | None)`
            Optional sequence-level or token-level weights (e.g., HEM weights) to apply before masked_whiten.
            Shape is (bs,) or (bs, response_length). If provided, advantages are multiplied before whitening.
        whiten_advantages: `(bool)`
            Whether to whiten actor advantages before returning them. VASPO defers whitening
            to its driver hook, which selects whether to whiten before or after fusion.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam_actor = 0
        lastgaelam_critic = 0
        advantages_reversed = []
        critic_advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_actor_ = delta + gamma * lam_actor * lastgaelam_actor
            lastgaelam_critic_ = delta + gamma * lam_critic * lastgaelam_critic

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam_actor = lastgaelam_actor_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam_actor
            lastgaelam_critic = (
                lastgaelam_critic_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam_critic
            )

            advantages_reversed.append(lastgaelam_actor)
            critic_advantages_reversed.append(lastgaelam_critic)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        critic_advantages = torch.stack(critic_advantages_reversed[::-1], dim=1)
        returns = critic_advantages + values

        if sample_weights is not None:
            advantages = _apply_advantage_weights(advantages, sample_weights)

        if whiten_advantages:
            advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def _median_center_per_response(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median-center token values independently within each response."""
    if values.shape != response_mask.shape:
        raise ValueError(
            f"values shape {values.shape} must match response_mask shape {response_mask.shape}"
        )

    valid_mask = response_mask.to(device=values.device).bool()
    response_centers = torch.zeros(
        values.shape[0], device=values.device, dtype=values.dtype
    )
    for row_idx in range(values.shape[0]):
        row_values = values[row_idx][valid_mask[row_idx]]
        if row_values.numel() == 0:
            continue
        # torch.median picks the lower middle value for even-length rows;
        # quantile(0.5) gives the conventional interpolated median instead.
        row_center = torch.quantile(row_values.float(), q=0.5)
        response_centers[row_idx] = row_center.to(dtype=values.dtype)

    centered_values = values - response_centers.unsqueeze(-1)
    centered_values = torch.where(
        valid_mask, centered_values, torch.zeros_like(centered_values)
    )
    return centered_values, response_centers


def compute_vaspo_fused_advantage(
    advantages: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    tanh_tau: float = 0.02,
    positive_advantage_only: bool = True,
    whiten_advantages: bool = False,
    token_mask_enabled: bool = True,
    token_mask_threshold: float = 0.02,
    epsilon_w: float = 0.2,
    student_log_probs: torch.Tensor | None = None,
    student_prob_source: str = "old",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Detached clipped-ratio reweighting (paper Eqs. teacher-gap through fusion).

    The tanh gap gates tokens; exp(gap) sets their weights. The default gates
    positive raw GAE advantages only. Mask removal and global post-whitening
    support the reported ablations; current student probabilities support the
    appendix variant. Padding never enters gap exponentiation or statistics.
    """
    for name, value in (("positive_advantage_only", positive_advantage_only),
                        ("whiten_advantages", whiten_advantages),
                        ("token_mask_enabled", token_mask_enabled)):
        if not isinstance(value, bool):
            raise ValueError(f"VASPO {name} must be a boolean")
    if not np.isfinite(tanh_tau) or tanh_tau <= 0:
        raise ValueError("VASPO tanh_tau must be finite and > 0")
    if not np.isfinite(token_mask_threshold) or token_mask_threshold < 0:
        raise ValueError("VASPO token_mask_threshold must be finite and >= 0")
    if not np.isfinite(epsilon_w) or not 0 <= epsilon_w < 1:
        raise ValueError("VASPO epsilon_w must be finite and in [0, 1)")
    if student_prob_source not in ("old", "current"):
        raise ValueError("VASPO student_prob_source must be old or current")
    if student_prob_source == "current" and student_log_probs is None:
        raise ValueError("VASPO current denominator requires student_log_probs")
    if student_prob_source == "current" and whiten_advantages:
        raise ValueError("VASPO current denominator does not support post-fusion whitening")
    student = student_log_probs if student_prob_source == "current" else old_log_probs
    if advantages.ndim != 2 or any(t.shape != advantages.shape for t in
                                  (teacher_log_probs, student, old_log_probs, response_mask)):
        raise ValueError("VASPO tensors must have matching [batch, response_length] shapes")
    with torch.no_grad():
        valid = response_mask.bool()
        anchor = torch.where(valid, advantages.detach().float(), 0.0)
        gap = torch.where(valid, teacher_log_probs.detach().to(anchor) - student.detach().to(anchor), 0.0)
        signal = torch.tanh(gap / tanh_tau)
        token_mask = valid & (signal.abs() > token_mask_threshold) if token_mask_enabled else valid
        eligible = valid & (anchor > 0) if positive_advantage_only else valid
        selected = token_mask & eligible
        # Clamp in log space to avoid overflow in exp for extreme teacher gaps.
        evidence = gap.clamp(np.log1p(-epsilon_w), np.log1p(epsilon_w)).exp()
        weight = torch.where(selected, evidence, torch.ones_like(anchor))
        fused = anchor * weight
        fused_actor, anchor_for_loss = fused, anchor
        if whiten_advantages and valid.any():
            mean = verl_F.masked_mean(fused, valid)
            var = verl_F.masked_var(fused, valid, unbiased=bool(valid.sum() > 1))
            inv_std = torch.rsqrt(var + 1e-8)
            fused_actor = torch.where(valid, (fused - mean) * inv_std, 0.0)
            anchor_for_loss = torch.where(valid, (anchor - mean) * inv_std, 0.0)
    return fused_actor, {
        "anchor_for_loss": anchor_for_loss, "teacher_gap": gap,
        "teacher_signal_raw": signal, "teacher_signal": torch.where(selected, signal, 0.0),
        "token_modulation_raw": weight - 1, "token_mask": token_mask,
        "applied_token_mask": selected, "applied_response_mask": selected.any(-1),
        "fused_raw": fused, "fusion_weight": weight,
    }


def compute_hem_weights(
    values_standard: torch.Tensor,
    values_oracle: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float,
    eps: float,
    clip_min: float,
    clip_max: float,
    invert: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sample-level HEM weights from two value evaluations.

    Args:
        values_standard: Value predictions on standard inputs. Shape: (bs, response_len).
        values_oracle: Value predictions on oracle(with-answer) inputs. Shape: (bs, response_len).
        response_mask: Response mask. Shape: (bs, response_len).
        alpha: HEM scaling factor.
        eps: Numerical epsilon for std division.
        clip_min: Lower clip bound for weights.
        clip_max: Upper clip bound for weights.
        invert: If True, invert the relationship so larger errors get smaller weights
            (1 - alpha * normalized). Default False (1 + alpha * normalized).

    Returns:
        hem_error: Absolute value gap per sample. Shape: (bs,).
        hem_weight: Unclipped HEM weight per sample. Shape: (bs,).
        hem_weight_clipped: Clipped HEM weight per sample. Shape: (bs,).
    """
    if values_standard.shape != values_oracle.shape:
        raise ValueError(
            f"values_standard shape {values_standard.shape} must match values_oracle shape {values_oracle.shape}"
        )
    if response_mask.shape != values_standard.shape:
        raise ValueError(f"response_mask shape {response_mask.shape} must match values shape {values_standard.shape}")

    with torch.no_grad():
        mask = response_mask.to(dtype=values_standard.dtype)
        valid_tokens = mask.sum(dim=-1)  # (bs,)
        denom = valid_tokens.clamp_min(1.0)
        valid_sample_mask = valid_tokens > 0

        value_standard_mean = (values_standard * mask).sum(dim=-1) / denom
        value_oracle_mean = (values_oracle * mask).sum(dim=-1) / denom
        hem_error = torch.abs(value_oracle_mean - value_standard_mean)

        hem_weight = torch.ones_like(hem_error)
        if valid_sample_mask.any():
            valid_error = hem_error[valid_sample_mask]
            mu = valid_error.mean()
            std = valid_error.std(unbiased=False)
            normalized = (hem_error - mu) / (std + eps)
            sign = -1.0 if invert else 1.0
            hem_weight = 1 + sign * alpha * normalized
            hem_weight = torch.where(valid_sample_mask, hem_weight, torch.ones_like(hem_weight))

        hem_weight_clipped = torch.clamp(hem_weight, min=clip_min, max=clip_max)
        hem_weight_clipped = torch.nan_to_num(
            hem_weight_clipped,
            nan=1.0,
            posinf=clip_max,
            neginf=clip_min,
        )

    return hem_error, hem_weight, hem_weight_clipped


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the Binary-TV DPPO clipped surrogate objective.

    ``clip_ratio(_low/_high)`` is interpreted as an *absolute sampled-token
    probability* threshold instead of PPO's probability-ratio threshold. For
    behavior probability ``mu_t = exp(old_log_prob_t)``, this is exactly

        clip(rho_t, 1 - eps_low / mu_t, 1 + eps_high / mu_t) * A_t.

    The direct clipped-surrogate form is used here (rather than an equivalent
    zero-gradient mask) so both the optimized objective and the reported loss
    match the BPCO paper's DPPO equation.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)

    clip_divergence = config.clip_ratio
    clip_divergence_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_divergence
    clip_divergence_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_divergence
    if clip_divergence_low < 0 or clip_divergence_high < 0:
        raise ValueError(
            "DPPO absolute-probability clipping thresholds must be non-negative, "
            f"got low={clip_divergence_low}, high={clip_divergence_high}."
        )

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # old_log_prob is a fixed behavior-policy anchor. Clamp only the
    # denominator used to construct ratio bounds to avoid division by zero for
    # numerically underflowed, extremely unlikely sampled tokens.
    behavior_prob = torch.exp(old_log_prob.detach())
    min_prob = torch.finfo(behavior_prob.dtype).tiny
    safe_behavior_prob = behavior_prob.clamp_min(min_prob)
    ratio_lower_bound = 1.0 - clip_divergence_low / safe_behavior_prob
    ratio_upper_bound = 1.0 + clip_divergence_high / safe_behavior_prob
    clipped_ratio = torch.clamp(ratio, min=ratio_lower_bound, max=ratio_upper_bound)

    pg_losses_unclipped = -advantages * ratio
    pg_losses_clipped = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses_unclipped, pg_losses_clipped)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    clipped_tokens = torch.gt(pg_losses_clipped, pg_losses_unclipped)
    pg_clipfrac = verl_F.masked_mean(clipped_tokens.float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped_tokens & (advantages < 0)).float(), response_mask)
    pg_clipfrac_upper = verl_F.masked_mean((clipped_tokens & (advantages >= 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/dppo_clipfrac_upper": pg_clipfrac_upper.detach().item(),
        "actor/dppo_enabled": 1.0,
    }
    return pg_loss, pg_metrics


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # Keep DPPO orthogonal to higher-level algorithms that reuse vanilla PPO.
    # HEM uses this loss directly; OPSD-PPO delegates after advantage fusion.
    # VASPO uses clipped PPO with response-wise valid-token reduction.
    if bool(config.policy_loss.get("use_dppo", False)):
        return compute_policy_loss_dppo_tv(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


def compute_policy_loss_dis(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SAO direct double-sided IS: -E[stopgrad(r * gate) * A * log pi].

    The gate is strictly 1-epsilon_low < pi_current/pi_rollout < 1+epsilon_high,
    independently of the advantage sign. Keep the original response mask in
    the denominator: rejected tokens contribute zero, not a renormalized mean.
    IS weights are detached as in an importance-weighted policy gradient;
    differentiating r * log pi would introduce an extra (1 + log pi) factor.
    """
    if config is None:
        raise ValueError("DIS requires actor config")
    eps_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    eps_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    if not math.isfinite(eps_low) or not 0 <= eps_low <= 1:
        raise ValueError("DIS requires 0 <= clip_ratio_low <= 1")
    if not math.isfinite(eps_high) or eps_high < 0 or eps_low + eps_high == 0:
        raise ValueError("DIS requires finite clip_ratio_high >= 0 and a nonempty interval")

    with torch.no_grad():
        # At least fp32 even when model outputs use bf16. Gate before exp so
        # extreme rejected ratios cannot overflow or get clamped into range.
        log_ratio = log_prob.detach().to(torch.promote_types(log_prob.dtype, torch.float32)) - rollout_log_prob.detach()
        lower = math.log1p(-eps_low) if eps_low < 1 else -math.inf
        upper = math.log1p(eps_high)
        valid = response_mask.bool()
        keep = valid & (log_ratio > lower) & (log_ratio < upper)
        weights = torch.where(keep, log_ratio, 0.0).exp() * keep
        count = response_mask.sum().clamp_min(1)
        low_fraction = ((log_ratio <= lower) & valid).sum() / count
        high_fraction = ((log_ratio >= upper) & valid).sum() / count
        rejected_fraction = (valid & ~keep).sum() / count
        metrics = {
            "actor/dis_clipfrac": rejected_fraction,
            "actor/dis_clipfrac_low": low_fraction,
            "actor/dis_clipfrac_high": high_fraction,
            "actor/dis_weight_mean": weights.sum() / count,
            "actor/pg_clipfrac": rejected_fraction,
        }

    # Mask log probabilities before multiplying, including padding values.
    pg_losses = -weights * advantages.detach().masked_fill(~keep, 0.0) * log_prob.masked_fill(~keep, 0.0)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting REINFORCE, PPO-clip, and SAO DIS.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    DIS dispatches directly to its token gate and importance-weighted objective.
    Other modes compute optional IS weights and rejection masks before dispatch.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
        - "dis": Direct double-sided IS with actor.clip_ratio_low/high as epsilons
            L = -E[stopgrad(r * gate) * log π(a|s) * A], with no extra IS/RS

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default), "reinforce", or "dis"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    if loss_type == "dis":
        from verl.trainer.ppo.rollout_corr_helper import validate_dis_config

        validate_dis_config(rollout_corr_config, config)
        return compute_policy_loss_dis(
            rollout_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
        )
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce', 'ppo_clip', or 'dis'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


# ================================================================== #
# Self-distillation losses (OPSD / SDPO / RLSD)
# Algorithm params are read from config.policy_loss (set via Hydra overrides)
# ================================================================== #

def _sd_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


def _sd_cfg(config, key, default):
    """Read algorithm param from config.policy_loss."""
    if config is None:
        return default
    pl = getattr(config, 'policy_loss', None)
    if pl is None:
        return default
    if hasattr(pl, 'get'):
        value = pl.get(key, default)
    else:
        value = getattr(pl, key, default)
    return default if value is None else value


def _sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_entropy=None):
    """Common teacher metrics for all self-distillation losses."""
    metrics = {}
    if teacher_lp is not None:
        metrics["teacher_gap"] = _sd_token_mean((teacher_lp - log_prob).abs(), response_mask).item()
        metrics["teacher_avg_logprob"] = _sd_token_mean(teacher_lp, response_mask).item()
        metrics["student_avg_logprob"] = _sd_token_mean(log_prob, response_mask).item()
    if teacher_entropy is not None:
        metrics["teacher_entropy"] = _sd_token_mean(teacher_entropy, response_mask).item()
    return metrics


def _sd_bool_cfg(config, key, default):
    return bool(_sd_cfg(config, key, default))


def _sd_int_cfg(config, key, default):
    return int(_sd_cfg(config, key, default))


def _sd_group_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denom


def _sd_group_var(values: torch.Tensor, valid_mask: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
    if mean is None:
        mean = _sd_group_mean(values, valid_mask)
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    centered = values - mean.unsqueeze(1)
    return (centered.square() * weights).sum(dim=1) / denom


def _sd_distribution_metrics(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    prefix: str,
    include_percentiles: bool = True,
) -> dict[str, float]:
    valid_mask = response_mask.bool() & torch.isfinite(values)
    valid_values = torch.masked_select(values.float(), valid_mask)
    if valid_values.numel() == 0:
        base = {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan")}
        if include_percentiles:
            base.update({
                f"{prefix}_p05": float("nan"),
                f"{prefix}_p25": float("nan"),
                f"{prefix}_p50": float("nan"),
                f"{prefix}_p75": float("nan"),
                f"{prefix}_p95": float("nan")})
        return base

    base = {
        f"{prefix}_mean": valid_values.mean().item(),
        f"{prefix}_min": valid_values.min().item(),
        f"{prefix}_max": valid_values.max().item()}
    if not include_percentiles:
        return base
    quantiles = torch.quantile(
        valid_values,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=valid_values.device))
    p05, p25, p50, p75, p95 = [quantile.item() for quantile in quantiles]
    base.update({
        f"{prefix}_p05": p05,
        f"{prefix}_p25": p25,
        f"{prefix}_p50": p50,
        f"{prefix}_p75": p75,
        f"{prefix}_p95": p95})
    return base


def _sd_effective_lam(config, global_steps, default=0.5):
    """Linearly decay lambda over lam_decay_steps, keeping step 1 at the initial value."""
    lam = float(_sd_cfg(config, "lam", default))
    lam_decay_steps = int(_sd_cfg(config, "lam_decay_steps", 0) or 0)
    if lam_decay_steps <= 0 or global_steps is None:
        return lam

    step_index = max(int(global_steps) - 1, 0)
    decay_progress = min(step_index / lam_decay_steps, 1.0)
    return max(lam * (1.0 - decay_progress), 0.0)


def _sd_prepare_distill_log_probs(
    student_all_log_probs,
    teacher_all_log_probs,
    top_k=0,
    add_tail=True,
    top_k_source="student"):
    """Prepare aligned student/teacher log-probs for full-vocab or top-k distillation."""
    assert student_all_log_probs is not None and teacher_all_log_probs is not None, (
        "full_logit_distill=True requires student_all_log_probs and teacher_all_log_probs."
    )

    if top_k <= 0:
        return student_all_log_probs, teacher_all_log_probs

    top_k = min(top_k, student_all_log_probs.shape[-1])
    source_log_probs = student_all_log_probs if top_k_source == "student" else teacher_all_log_probs
    _, topk_indices = torch.topk(source_log_probs.detach(), k=top_k, dim=-1)
    student_topk = torch.gather(student_all_log_probs, dim=-1, index=topk_indices)
    teacher_topk = torch.gather(teacher_all_log_probs, dim=-1, index=topk_indices)

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk)
        teacher_tail = _tail_log_prob(teacher_topk)
        student_topk = torch.cat([student_topk, student_tail], dim=-1)
        teacher_topk = torch.cat([teacher_topk, teacher_tail], dim=-1)
    else:
        student_topk = student_topk - torch.logsumexp(student_topk, dim=-1, keepdim=True)
        teacher_topk = teacher_topk - torch.logsumexp(teacher_topk, dim=-1, keepdim=True)

    return student_topk, teacher_topk


def _sd_prepare_sparse_topk_log_probs(student_topk_log_probs, teacher_topk_log_probs, add_tail=True):
    """Finalize aligned sparse top-k log-probs by adding a tail bucket or renormalizing."""
    assert student_topk_log_probs is not None and teacher_topk_log_probs is not None, (
        "Sparse top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
    )

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk_log_probs)
        teacher_tail = _tail_log_prob(teacher_topk_log_probs)
        student_topk_log_probs = torch.cat([student_topk_log_probs, student_tail], dim=-1)
        teacher_topk_log_probs = torch.cat([teacher_topk_log_probs, teacher_tail], dim=-1)
    else:
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(
            teacher_topk_log_probs, dim=-1, keepdim=True
        )

    return student_topk_log_probs, teacher_topk_log_probs


def _sd_generalized_jsd(student_log_probs, teacher_log_probs, mix, vocab_clip=None):
    """Generalized JSD / KL over log-prob distributions.

    If vocab_clip is a positive float, the per-(token, vocab) divergence
    contribution is clamped at vocab_clip BEFORE summing over the vocab
    dimension. This matches the official OPSD reference implementation, where
    `F.kl_div(..., reduction="none")` (shape [B, S, V]) is clipped per element
    and then summed across both vocab and tokens. With vocab_clip=None (the
    default), behavior is unchanged: KL contributions are summed over the
    vocab dimension first and any clipping is left to the caller.
    """
    apply_vocab_clip = vocab_clip is not None and vocab_clip > 0

    if mix == 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)
    if mix == 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)

    mix_tensor = torch.tensor(mix, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-mix_tensor),
                teacher_log_probs + torch.log(mix_tensor),
            ]
        ),
        dim=0)
    if apply_vocab_clip:
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = torch.lerp(kl_student, kl_teacher, mix_tensor)
        return per_vocab.clamp(max=vocab_clip).sum(-1)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
    return torch.lerp(kl_student, kl_teacher, mix_tensor)


@register_policy_loss("vaspo")
def compute_policy_loss_vaspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    vaspo_anchor_advantages: torch.Tensor | None = None,
    vaspo_token_mask: torch.Tensor | None = None,
    vaspo_diagnostic_context: dict | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clipped PPO, mean over valid tokens per response then over responses.

    Selected and unselected tokens share the denominator and equal weights.
    Optional rollout importance weights multiply the token surrogate.
    """
    if vaspo_anchor_advantages is None or vaspo_token_mask is None:
        raise ValueError("VASPO loss requires vaspo_anchor_advantages and vaspo_token_mask")
    if any(t.shape != log_prob.shape for t in (old_log_prob, advantages, response_mask,
                                               vaspo_anchor_advantages, vaspo_token_mask)):
        raise ValueError("VASPO loss tensors must have matching shapes")
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("VASPO epsilon must be finite and in [0, 1)")
    normalization, eta = "unified", 1.0

    valid = response_mask.bool()
    selected = vaspo_token_mask.bool() & valid
    unselected = valid & ~selected
    anchor = vaspo_anchor_advantages.detach().float()
    fused = advantages.detach().float()
    effective_adv = torch.where(valid, torch.where(selected, fused, anchor), 0.0)
    # Ignore arbitrary padded log-probabilities before exponentiation.
    log_ratio = torch.where(valid, log_prob.float() - old_log_prob.detach().float(), 0.0)
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    unclipped = ratio * effective_adv
    clipped_ratio = ratio.clamp(1 - epsilon, 1 + epsilon)
    clipped = clipped_ratio * effective_adv
    surrogate = torch.minimum(unclipped, clipped)
    if rollout_is_weights is not None:
        surrogate = surrogate * rollout_is_weights.detach()

    def path_mean(mask):
        denominator = valid.sum(-1)
        return torch.where(mask, surrogate, 0.0).sum(-1) / denominator.clamp_min(1)

    unmodulated = path_mean(unselected)
    modulated = path_mean(selected)
    loss = -(unmodulated + eta * modulated).mean()
    count = valid.sum().clamp_min(1)
    metrics = {
        "actor/pg_clipfrac": ((unclipped > clipped) & valid).sum().detach().item() / count.item(),
        "actor/ppo_kl": (-log_ratio.sum() / count).detach().item(),
        "vaspo/unmodulated_loss": -unmodulated.mean().detach().item(),
        "vaspo/modulated_loss": -modulated.mean().detach().item(),
        "vaspo/eta": eta,
        "vaspo/loss_normalization_unified": float(normalization == "unified"),
    }
    if bool(_sd_cfg(config, "vaspo_diagnostics", False)):
        # Detached diagnostics only. JSON preserves per-microbatch extrema/counts;
        # dashboard reductions may average these across workers and microbatches.
        import json

        with torch.no_grad():
            v = valid.detach()
            lr = log_ratio.detach()
            adv = effective_adv.detach()
            weights = (torch.ones_like(adv) if rollout_is_weights is None
                       else torch.broadcast_to(rollout_is_weights.detach(), adv.shape))
            denominator = valid.sum(-1).clamp_min(1)
            du = denominator if normalization == "unified" else unselected.sum(-1).clamp_min(1)
            dm = denominator if normalization == "unified" else selected.sum(-1).clamp_min(1)
            coefficient = torch.where(selected, eta / dm[:, None], 1.0 / du[:, None]) / max(adv.shape[0], 1)
            contribution = torch.where(v, -surrogate.detach() * coefficient, 0.0)
            danger = v & (adv < 0) & (lr > 5)
            stats = {"valid_count": int(v.sum().item()),
                     "is_enabled": rollout_is_weights is not None,
                     "loss": float(loss.detach().item()),
                     "unmodulated_loss": metrics["vaspo/unmodulated_loss"],
                     "modulated_loss": metrics["vaspo/modulated_loss"],
                     "danger_count": int(danger.sum().item()),
                     "danger_loss_contribution": float(contribution[danger].sum().item())}

            def extrema(name, tensor, mask):
                values = tensor.detach()[mask]
                finite = values[torch.isfinite(values)]
                stats[name + "_nonfinite_count"] = int((~torch.isfinite(values)).sum().item())
                stats[name + "_min"] = float(finite.min().item()) if finite.numel() else 0.0
                stats[name + "_max"] = float(finite.max().item()) if finite.numel() else 0.0
                stats[name + "_abs_max"] = float(finite.abs().max().item()) if finite.numel() else 0.0

            extrema("log_ratio", lr, v)
            extrema("is_weight", weights, v)
            for threshold in (1, 5, 10):
                stats[f"log_ratio_abs_gt_{threshold}_count"] = int((v & (lr.abs() > threshold)).sum().item())
            stats["log_ratio_gt_20_count"] = int((v & (lr > 20)).sum().item())
            stats["log_ratio_lt_neg20_count"] = int((v & (lr < -20)).sum().item())
            stats["neg_adv_log_ratio_gt5_lt20_count"] = int((danger & (lr < 20)).sum().item())
            stats["neg_adv_log_ratio_gt20_count"] = int((v & (adv < 0) & (lr > 20)).sum().item())
            for name, mask in (("selected", selected), ("unselected", unselected)):
                stats[name + "_count"] = int(mask.sum().item())
                extrema(name + "_adv", adv, mask)
                extrema(name + "_anchor", anchor, mask)
                stats[name + "_clipfrac"] = float(((unclipped > clipped) & mask).sum().item() / max(mask.sum().item(), 1))
            metrics.update({"vaspo/diag/" + key: float(value) for key, value in stats.items()})
            spike = (not torch.isfinite(loss.detach()).item()
                     or abs(stats["loss"]) > 10
                     or stats["log_ratio_abs_max"] > 5
                     or stats["log_ratio_nonfinite_count"] > 0
                     or stats["selected_adv_nonfinite_count"] > 0
                     or stats["unselected_adv_nonfinite_count"] > 0
                     or stats["is_weight_nonfinite_count"] > 0)
            # First microbatch supplies a normal comparison each rollout/worker.
            context = vaspo_diagnostic_context or {}
            baseline = context.get("epoch", 0) == 0 and context.get("mini_batch", 0) == 0 and context.get("micro_batch", 0) == 0
            report = {"event": "spike" if spike else "baseline" if baseline else "micro_batch",
                      **context, "normalization": normalization, "eta": eta, "stats": stats}
            if spike or baseline:
                flat_indices = v.reshape(-1).nonzero().flatten()
                scores = contribution.reshape(-1)[flat_indices].abs()
                order = torch.nan_to_num(scores, nan=float("inf"), posinf=float("inf")).topk(min(5, scores.numel())).indices
                tokens = []
                for flat_index in flat_indices[order].tolist():
                    row, column = divmod(flat_index, adv.shape[1])
                    tokens.append({"row": row, "response_token_index": column,
                                   "selected": bool(selected[row, column].item()),
                                   "old_log_prob": float(old_log_prob[row, column].detach().item()),
                                   "new_log_prob": float(log_prob[row, column].detach().item()),
                                   "log_ratio": float(lr[row, column].item()),
                                   "anchor": float(anchor[row, column].item()),
                                   "fused": float(fused[row, column].item()),
                                   "is_weight": float(weights[row, column].item()),
                                   "loss_contribution": float(contribution[row, column].item())})
                report["top_tokens"] = tokens
            # Emit every microbatch so a later backward/optimizer spike can be
            # traced even when its forward values did not trigger the heuristic.
            print("VASPO_DIAG " + json.dumps(report), flush=True)
    return loss, metrics


@register_policy_loss("opsd_ppo")
def compute_policy_loss_opsd_ppo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO-clip over the sum of PPO and sampled-token OPSD advantages.

    The privileged teacher scores the student's sampled token, yielding the
    fixed per-token preference signal

        d_t = log q_teacher(a_t | h_t, z) - log pi_old(a_t | h_t),
        A_opsd = d_t - median_response({d_j}).

    We fuse it with the critic-derived PPO advantage before applying the
    unchanged verl PPO-clip objective. With correct-only gating enabled,
    c_i is one only for responses whose outcome reward passes the configured
    correctness threshold:

        A_fused_i = A_ppo_i + c_i * opsd_advantage_coef * A_opsd_i.

    Per-response median centering removes a response-wide teacher calibration
    offset while retaining the relative token preference signal. Both teacher
    and old-policy log probabilities are detached. This is the cheapest OPSD
    variant because it needs only sampled-token log probabilities, not a
    full-vocabulary or top-k distribution.
    """
    teacher_log_prob = kwargs.get("teacher_log_probs")
    assert teacher_log_prob is not None, (
        "opsd_ppo requires teacher_log_probs. Ensure the self-distillation "
        "teacher hook is enabled and privileged teacher inputs are present."
    )
    if teacher_log_prob.shape != advantages.shape:
        raise ValueError(
            f"teacher_log_probs shape {teacher_log_prob.shape} must match advantages shape {advantages.shape}"
        )

    opsd_advantage_coef = float(_sd_cfg(config, "opsd_advantage_coef", 1.0))
    opsd_correct_only = _sd_bool_cfg(config, "opsd_correct_only", False)
    with torch.no_grad():
        advantage_opsd_raw = teacher_log_prob.detach() - old_log_prob.detach()
        advantage_opsd_raw = advantage_opsd_raw.to(
            device=advantages.device,
            dtype=advantages.dtype,
        )
        advantage_opsd, response_centers = _median_center_per_response(
            advantage_opsd_raw,
            response_mask,
        )
        valid_response_rows = response_mask.to(device=advantages.device).bool().any(dim=-1)
        valid_response_centers = response_centers[valid_response_rows]

        if opsd_correct_only:
            opsd_correct_mask = kwargs.get("opsd_correct_mask")
            if opsd_correct_mask is None:
                raise ValueError(
                    "opsd_correct_only=true requires a per-response opsd_correct_mask"
                )
            if opsd_correct_mask.ndim == 2 and opsd_correct_mask.shape[-1] == 1:
                opsd_correct_mask = opsd_correct_mask.squeeze(-1)
            expected_mask_shape = (advantages.shape[0],)
            if opsd_correct_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"opsd_correct_mask must have shape {expected_mask_shape}, "
                    f"got {opsd_correct_mask.shape}"
                )
            opsd_applied_rows = opsd_correct_mask.to(device=advantages.device).bool()
            opsd_applied_rows = opsd_applied_rows & valid_response_rows
        else:
            opsd_applied_rows = valid_response_rows

        opsd_gate = opsd_applied_rows.to(dtype=advantages.dtype).unsqueeze(-1)
        advantage_opsd_effective = advantage_opsd * opsd_gate
        advantage_fused = advantages + opsd_advantage_coef * advantage_opsd_effective

        valid_response_count = valid_response_rows.sum().clamp(min=1)
        opsd_applied_response_ratio = (
            opsd_applied_rows.sum().to(dtype=torch.float32)
            / valid_response_count.to(dtype=torch.float32)
        )

    loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantage_fused,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    metrics.update(
        {
            "actor/opsd_advantage_coef": opsd_advantage_coef,
            "actor/opsd_correct_only_enabled": float(opsd_correct_only),
            "actor/opsd_applied_response_ratio": opsd_applied_response_ratio.item(),
            "actor/advantage_ppo_mean": _sd_token_mean(advantages.detach(), response_mask).item(),
            "actor/advantage_opsd_raw_mean": _sd_token_mean(
                advantage_opsd_raw, response_mask
            ).item(),
            "actor/advantage_opsd_raw_abs_mean": _sd_token_mean(
                advantage_opsd_raw.abs(), response_mask
            ).item(),
            "actor/opsd_gap_response_median_mean": valid_response_centers.mean().item(),
            "actor/opsd_gap_response_median_abs_mean": valid_response_centers.abs().mean().item(),
            "actor/advantage_opsd_mean": _sd_token_mean(advantage_opsd, response_mask).item(),
            "actor/advantage_opsd_abs_mean": _sd_token_mean(advantage_opsd.abs(), response_mask).item(),
            "actor/advantage_opsd_effective_mean": _sd_token_mean(
                advantage_opsd_effective, response_mask
            ).item(),
            "actor/advantage_opsd_effective_abs_mean": _sd_token_mean(
                advantage_opsd_effective.abs(), response_mask
            ).item(),
            "actor/advantage_fused_mean": _sd_token_mean(advantage_fused, response_mask).item(),
        }
    )
    metrics.update(
        _sd_teacher_metrics(
            teacher_log_prob,
            log_prob,
            response_mask,
            kwargs.get("teacher_entropy"),
        )
    )
    return loss, metrics


@register_policy_loss("opsd")
def compute_policy_loss_opsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """OPSD: Forward KL distillation (teacher -> student).

    The teacher's privileged context (what extra info it conditions on) is
    chosen on the driver side via privileged_text_mode (solution_answer /
    answer_only / oracle). This loss only consumes teacher_log_probs and does
    not care how that teacher forward was conditioned.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "OPSD requires teacher_log_probs but got None. "
        "Ensure teacher_mode=fixed and use_kl_loss=True so ref_log_prob is computed."
    )
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="teacher")
        # Match official OPSD: clip per-(token, vocab) KL contribution before
        # summing over vocab. The clip can break Gibbs' inequality and produce
        # negative per-token loss values; this matches the reference behavior.
        per_token_loss = _sd_generalized_jsd(
            student_distill,
            teacher_distill,
            beta,
            vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
    else:
        per_token_loss = (teacher_lp.detach() - log_prob).pow(2)
        if jsd_token_clip > 0:
            per_token_loss = per_token_loss.clamp(max=jsd_token_clip)

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("opsd_ectr")
def compute_policy_loss_opsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """OPSD with CFG-style contrastive guided target in the top-k subspace.

    Same sparse top-k machinery as OPSD: top-k vocab indices are chosen by the
    correct-context teacher, and student / correct-teacher / wrong-teacher all
    gather their log-probs at those indices. An optional tail bucket captures
    the remaining mass. In this (top-k + tail) log-space:

        guided_logits = (1 + α) · log p_c  -  α · clamp(log p_w, min=floor)
        q_target      = softmax(guided_logits)

    Student matches q_target via the same generalized-JSD machinery as OPSD.
    α = 0 recovers vanilla OPSD; α > 0 amplifies the correct-vs-wrong contrast
    across the top-k support.

    Token-level contrastive masking (controlled by ``opsd_ectr_cw_tau``):
    At each token position the KL divergence between correct and wrong teachers
    is computed.  Positions where correct ≈ wrong (stylistic tokens) receive
    near-zero loss weight so the student's policy is left unchanged there;
    positions where the teachers genuinely disagree (content tokens) receive
    full loss weight.  tau controls the sharpness of this soft mask —
    setting tau=0 disables masking and recovers the original uniform behaviour.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    student_topk = kwargs.get("student_topk_log_probs")
    teacher_correct_topk = kwargs.get("teacher_topk_log_probs")
    teacher_wrong_topk = kwargs.get("teacher_wrong_topk_log_probs")
    assert (
        student_topk is not None
        and teacher_correct_topk is not None
        and teacher_wrong_topk is not None
    ), (
        "opsd_ectr requires student/teacher_correct/teacher_wrong topk_log_probs "
        "gathered at teacher_correct's top-k indices."
    )

    alpha = float(_sd_cfg(config, "opsd_ectr_alpha", 0.5))
    log_w_floor = float(_sd_cfg(config, "opsd_ectr_log_w_floor", -30.0))
    cw_tau = float(_sd_cfg(config, "opsd_ectr_cw_tau", 1.0))
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    student_dist, correct_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_correct_topk,
        add_tail=distill_add_tail)
    _, wrong_dist = _sd_prepare_sparse_topk_log_probs(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_wrong_topk,
        add_tail=distill_add_tail)

    wrong_clamped = wrong_dist.clamp(min=log_w_floor)
    guided_logits = (1.0 + alpha) * correct_dist - alpha * wrong_clamped
    guided_log_probs = guided_logits - torch.logsumexp(guided_logits, dim=-1, keepdim=True)

    per_token_loss = _sd_generalized_jsd(
        student_dist,
        guided_log_probs,
        beta,
        vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)

    # Token-level contrastive mask: only train where correct and wrong teachers
    # genuinely disagree; leave the student's policy untouched at stylistic
    # positions where the hint carries no information.
    if cw_tau > 0:
        with torch.no_grad():
            cw_kl = F.kl_div(
                wrong_dist, correct_dist, reduction="none", log_target=True).sum(-1)  # [B, S]
            cw_kl = cw_kl.clamp(min=0.0)
            token_weight = cw_kl / (cw_kl + cw_tau)
        per_token_loss = per_token_loss * token_weight

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)

    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(teacher_correct_topk.shape[-1]),
        "distill_full_logit": 1.0,
        "opsd_ectr_alpha": alpha,
        "opsd_ectr_log_w_floor": log_w_floor,
        "opsd_ectr_cw_tau": cw_tau}
    if cw_tau > 0:
        active = _sd_token_mean(token_weight, response_mask).item()
        metrics["opsd_ectr_active_ratio"] = active
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlsd")
def compute_policy_loss_rlsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """RLSD: GRPO + evidence-ratio modulated advantages.

    The teacher's privileged context is chosen on the driver side via
    privileged_text_mode (solution_answer / answer_only / oracle). This loss
    only consumes teacher_log_probs and does not care how the teacher forward
    was conditioned.
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "RLSD requires teacher_log_probs but got None. "
        "teacher_mode=snapshot requires EMA/snapshot implementation in engine_workers."
    )
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()

    with torch.no_grad():
        # Evidence ratio: how much better the (oracle) teacher is than the
        # current student on this token, signed by the advantage direction.
        delta_t = teacher_lp - log_prob.detach()
        w_t = (advantages.sign() * delta_t).exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        # Interpolate between raw advantage (lam=0) and evidence-weighted (lam=1).
        mod_adv = advantages * ((1 - lam) + lam * w_t)

    surr1 = ratio * mod_adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * mod_adv
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {"pg_loss": loss.item(),
               "pg_clipfrac": ((ratio - 1).abs() > epsilon).float().mean().item(),
               "rlsd_epsilon": epsilon,
               "rlsd_epsilon_w": epsilon_w,
               "rlsd_lam": lam,
               "rlsd_w_t_mean": w_t.mean().item(),
               "rlsd_w_t_std": w_t.std(unbiased=False).item()}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlsd_ectr")
def compute_policy_loss_rlsd_ectr(old_log_prob, log_prob, advantages, response_mask,
                                  loss_agg_mode="token_mean", config=None,
                                  rollout_is_weights=None, **kwargs):
    """RLSD variant: contrast teacher_correct vs teacher_wrong instead of student.

    Structurally identical to RLSD but the evidence ratio uses the pairwise
    teacher contrast e_ctr = teacher_correct_lp - teacher_wrong_lp in place of
    e_c = teacher_correct_lp - student_lp. Both correct and wrong teacher
    contexts come from the single-pair data path.
    Added a -60/60 clamp on the exponent because e_ctr (teacher vs teacher) can
    be much larger in magnitude than e_c (teacher vs student); epsilon_w still
    bounds the final weight.
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_lp = kwargs.get("teacher_wrong_log_probs")
    assert teacher_correct_lp is not None and teacher_wrong_lp is not None, (
        "rlsd_ectr requires teacher_log_probs and teacher_wrong_log_probs."
    )
    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    epsilon_w = float(_sd_cfg(config, "epsilon_w", 0.2))
    lam = _sd_effective_lam(config, kwargs.get("global_steps"), default=0.5)

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()

    with torch.no_grad():
        delta_t = teacher_correct_lp - teacher_wrong_lp
        scaled = (advantages.sign() * delta_t).clamp(min=-60.0, max=60.0)
        w_t = scaled.exp().clamp(1 - epsilon_w, 1 + epsilon_w)
        mod_adv = advantages * ((1 - lam) + lam * w_t)

    surr1 = ratio * mod_adv
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * mod_adv
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": ((ratio - 1).abs() > epsilon).float().mean().item(),
        "rlsd_ectr_lam": lam,
        "rlsd_ectr_e_ctr_abs_mean": verl_F.masked_mean(delta_t.abs(), response_mask).item()}
    metrics.update(_sd_distribution_metrics(delta_t, response_mask, "rlsd_ectr_e_ctr", include_percentiles=False))
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("rlcsd")
def compute_policy_loss_rlcsd(old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode="token_mean", config=None,
                                rollout_is_weights=None, **kwargs):
    """RLCSD: K-marginal wrong-hint contrast policy loss.

    Wrong-hint term is replaced by the
    log of the uniform mixture of teacher distributions over up to K non-self
    negative siblings:

        e_ctr_t = log P_T(y_t | x, y_c*, y_<t)
                  - log[ (1/K) * sum_k P_T(y_t | x, y_w^(k), y_<t) ]

    Implemented as logsumexp(teacher_wrong_multi_lp) - log(K_valid) per row,
    with -inf masking on padded K-slots so they don't contribute. All other
    hyperparameters reuse the rlcsd_* keys (tau/beta/lam/delta/eta/clip)
    so 
    """
    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_multi_lp = kwargs.get("teacher_wrong_multi_log_probs")
    teacher_wrong_multi_valid_mask = kwargs.get("teacher_wrong_multi_valid_mask")
    assert teacher_correct_lp is not None, "rlcsd_5 requires teacher_log_probs."
    assert teacher_wrong_multi_lp is not None and teacher_wrong_multi_valid_mask is not None, (
        "rlcsd_5 requires teacher_wrong_multi_log_probs and teacher_wrong_multi_valid_mask."
    )

    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    tau = max(float(_sd_cfg(config, "rlcsd_tau", _sd_cfg(config, "tau", 0.03))), 1e-6)
    beta = float(_sd_cfg(config, "rlcsd_beta", 0.2))
    lam = float(_sd_cfg(config, "rlcsd_lam", _sd_cfg(config, "lam", 0.5)))
    delta = max(float(_sd_cfg(config, "rlcsd_delta", 0.02)), 0.0)
    eta = max(float(_sd_cfg(config, "rlcsd_eta", 1.0)), 0.0)
    residual_clip_low = float(_sd_cfg(config, "rlcsd_residual_clip_low", -0.5))
    residual_clip_high = float(_sd_cfg(config, "rlcsd_residual_clip_high", 0.5))
    if residual_clip_low > residual_clip_high:
        residual_clip_low, residual_clip_high = residual_clip_high, residual_clip_low

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    with torch.no_grad():
        # Marginalize wrong-hint logp over K valid siblings:
        # log_marginal_w = logsumexp_k(lp_k) - log(K_valid)
        valid_mask_bool = teacher_wrong_multi_valid_mask.to(dtype=torch.bool)
        masked_w_lp = teacher_wrong_multi_lp.masked_fill(
            ~valid_mask_bool.unsqueeze(-1), float("-inf"))
        K_valid = valid_mask_bool.to(dtype=teacher_wrong_multi_lp.dtype).sum(dim=1).clamp(min=1.0)
        log_marginal_w = torch.logsumexp(masked_w_lp, dim=1) - torch.log(K_valid).unsqueeze(-1)

        e_ctr = teacher_correct_lp - log_marginal_w
        s_t = torch.tanh(e_ctr / tau)
        preclip_residual = beta * lam * s_t
        residual = preclip_residual.clamp(min=residual_clip_low, max=residual_clip_high)
        selected = (residual.abs() > delta) & response_mask.bool()
        selected_float = selected.float()
        selected_count = selected_float.sum()
        residual_for_adv = selected_float * residual
        response_token_count = response_mask.sum()
        if selected_float.dim() >= 2:
            valid_sample_mask = response_mask.sum(dim=-1) > 0
            sample_count = valid_sample_mask.float().sum()
        else:
            sample_count = torch.as_tensor(1.0, dtype=selected_count.dtype, device=selected_count.device)
        raw_mod_adv = advantages + residual_for_adv
        mod_adv = torch.where(
            advantages > 0,
            raw_mod_adv.clamp_min(0.0),
            torch.where(advantages < 0, raw_mod_adv.clamp_max(0.0), torch.zeros_like(raw_mod_adv)))
        K_valid_mean = K_valid.mean()

    ratio_clamped = ratio.clamp(1 - epsilon, 1 + epsilon)
    nonselected_mask = response_mask * (1.0 - selected_float)
    selected_mask = response_mask * selected_float

    surr_nonselected1 = ratio * advantages
    surr_nonselected2 = ratio_clamped * advantages
    nonselected_per_token_loss = -torch.min(surr_nonselected1, surr_nonselected2)

    surr_selected1 = ratio * mod_adv
    surr_selected2 = ratio_clamped * mod_adv
    selected_per_token_loss = -torch.min(surr_selected1, surr_selected2)
    if rollout_is_weights is not None:
        nonselected_per_token_loss = nonselected_per_token_loss * rollout_is_weights
        selected_per_token_loss = selected_per_token_loss * rollout_is_weights
    nonselected_loss = _sd_token_mean(nonselected_per_token_loss, nonselected_mask)
    selected_loss = _sd_token_mean(selected_per_token_loss, selected_mask)
    loss = nonselected_loss + eta * selected_loss

    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": verl_F.masked_mean(((ratio - 1).abs() > epsilon).float(), response_mask).item(),
        "ppo_kl": ppo_kl.item(),
        "rlcsd_selected_token_count": selected_count.item(),
        "rlcsd_response_token_count": response_token_count.item(),
        "rlcsd_sample_count": sample_count.item(),
        "rlcsd_K_valid_mean": K_valid_mean.item()}
    metrics.update(_sd_distribution_metrics(e_ctr, response_mask, "rlcsd_e_ctr", include_percentiles=False))
    metrics.update(_sd_distribution_metrics(residual, selected_mask, "rlcsd_selected_residual"))
    return loss, metrics


@register_policy_loss("sdpo")
def compute_policy_loss_sdpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SDPO: JSD / KL distillation with optional full-vocab or top-k views and IS correction."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "SDPO requires teacher_log_probs but got None. "
        "teacher_mode=ema requires EMA implementation in engine_workers."
    )
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SDPO only supports alpha=1.0 (reverse KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    per_token_loss = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "distill_loss": loss.item(),
        "is_ratio": is_ratio.mean().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics


@register_policy_loss("srpo")
def compute_policy_loss_srpo(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """SRPO: Sample-Routed Policy Optimization (arXiv:2604.02288).

    Routes each rollout to the supervision signal best suited to its learning
    status: correct rollouts go through GRPO's PPO-clip surrogate (reward-aligned,
    sequence-level advantage), while incorrect rollouts with available teacher
    information go through SDPO's logit-level distillation, reweighted token-wise
    by teacher entropy (Dynamic-Weighted SDPO).

    The combined objective follows Eq. (Lfinal) of the paper:
        L = (Σ_{i,t} z^GRPO_i ℓ^GRPO_{i,t} + Σ_{i,t} z^SDPO_i ℓ^DW-SDPO_{i,t})
            / (Σ_{i,t} z^GRPO_i + Σ_{i,t} z^SDPO_i)
    where z^SDPO_i = (1 - c_i) m_i and z^GRPO_i = 1 - z^SDPO_i.

    The correctness flag c_i is read from the per-rollout sign of the GRPO
    advantage (a positive group-normalized advantage corresponds to a higher
    reward than the group mean, equivalent to "correct" in the binary-reward
    regime when the group is mixed). Teacher availability m_i is taken to be 1
    whenever teacher_log_probs are present.

    Hyperparameters (read from config.policy_loss):
        srpo_beta:           dynamic-weighting temperature β (default 1.0).
        alpha:               JSD mixing coefficient for the SDPO branch
                             (0 = forward KL, 1 = reverse KL, 0<α<1 = JSD).
        full_logit_distill:  use full-vocab JSD (True) or k1 reverse-KL (False).
        top_k_distill:       sparse top-k support size for the JSD branch.
        is_clip:             upper clamp on the SDPO IS ratio (default 2.0).
    """
    teacher_lp = kwargs.get("teacher_log_probs")
    teacher_ent = kwargs.get("teacher_entropy")
    assert teacher_lp is not None, (
        "SRPO requires teacher_log_probs (feedback-conditioned self-teacher). "
        "Ensure the SDPO teacher pipeline is enabled for this method."
    )
    assert teacher_ent is not None, (
        "SRPO requires teacher_entropy for entropy-aware dynamic weighting."
    )

    beta = float(_sd_cfg(config, "srpo_beta", 1.0))
    alpha = float(_sd_cfg(config, "alpha", 0.5))
    is_clip = float(_sd_cfg(config, "is_clip", 2.0))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    valid_mask = response_mask.to(log_prob.dtype)

    # ---- Sample-level routing: c_i from per-rollout advantage sign --------
    rollout_token_count = valid_mask.sum(dim=1).clamp(min=1.0)
    rollout_adv = (advantages * valid_mask).sum(dim=1) / rollout_token_count
    correct_per_rollout = (rollout_adv > 0).to(log_prob.dtype)         # c_i
    sdpo_per_rollout = 1.0 - correct_per_rollout                        # z^SDPO_i (m_i = 1)
    grpo_per_rollout = 1.0 - sdpo_per_rollout                           # z^GRPO_i

    grpo_mask = grpo_per_rollout.unsqueeze(1) * valid_mask
    sdpo_mask = sdpo_per_rollout.unsqueeze(1) * valid_mask

    # ---- GRPO branch: dual-clip PPO surrogate, per-token (no reduction) ---
    negative_approx_kl = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = negative_approx_kl.exp()
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    grpo_per_token = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        grpo_per_token = grpo_per_token * rollout_is_weights

    # Diagnostics on the GRPO branch only (clipping is meaningless for SDPO tokens).
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    grpo_diag_mask = grpo_mask if grpo_mask.sum() > 0 else response_mask
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), grpo_diag_mask)
    pg_clipfrac_lower = verl_F.masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float()),
        grpo_diag_mask)

    # ---- SDPO branch: logit-level distillation, per-token (no reduction) --
    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="student")
        per_token_kl = _sd_generalized_jsd(student_distill, teacher_distill, alpha)
    else:
        if alpha != 1.0:
            raise AssertionError("Non-full-logit SRPO/SDPO only supports alpha=1.0 (k1 reverse-KL estimator).")
        log_ratio = log_prob - teacher_lp.detach()
        per_token_kl = log_ratio.detach() * log_prob

    is_ratio = (log_prob - old_log_prob).detach().clamp(-20, 20).exp().clamp(max=is_clip)
    sdpo_per_token = per_token_kl * is_ratio
    if rollout_is_weights is not None:
        sdpo_per_token = sdpo_per_token * rollout_is_weights

    # ---- Entropy-aware Dynamic Weighting on the SDPO branch ---------------
    # w̃_{i,t} = exp(-β · H_{i,t}); normalize so the mean weight over routed
    # tokens is 1, preserving the overall loss scale (paper §3.2).
    with torch.no_grad():
        teacher_ent_d = teacher_ent.detach().to(log_prob.dtype)
        log_w = -beta * teacher_ent_d
        # Numerically stable exp: subtract the max log-weight over routed tokens.
        # Tokens outside the SDPO route get -1e30 so they don't influence the
        # max; the (log_w - max) difference is then clamped to avoid +inf when
        # no SDPO tokens exist in the batch.
        log_w_for_max = torch.where(
            sdpo_mask > 0, log_w, torch.full_like(log_w, -1.0e30)
        )
        log_w_max = log_w_for_max.amax()
        w_unnorm = ((log_w - log_w_max).clamp(max=50.0)).exp() * sdpo_mask
        n_sdpo_tokens = sdpo_mask.sum().clamp(min=1.0)
        w_mean = w_unnorm.sum() / n_sdpo_tokens
        dw_weights = w_unnorm / w_mean.clamp(min=1e-12)

    sdpo_per_token = sdpo_per_token * dw_weights

    # ---- Final routed loss with unified denominator -----------------------
    grpo_term = (grpo_per_token * grpo_mask).sum()
    sdpo_term = (sdpo_per_token * sdpo_mask).sum()
    denom = (grpo_mask.sum() + sdpo_mask.sum()).clamp(min=1.0)
    loss = (grpo_term + sdpo_term) / denom

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "srpo/correct_rollout_frac": correct_per_rollout.mean().detach().item(),
        "srpo/sdpo_token_frac": (sdpo_mask.sum() / valid_mask.sum().clamp(min=1.0)).detach().item(),
        "srpo/grpo_loss_term": (grpo_term / denom).detach().item(),
        "srpo/sdpo_loss_term": (sdpo_term / denom).detach().item(),
        "srpo/dw_beta": beta,
        "srpo/dw_weight_mean": w_mean.detach().item(),
        "srpo/is_ratio": is_ratio.mean().detach().item(),
        "distill_alpha": alpha,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_ent))
    return loss, metrics
