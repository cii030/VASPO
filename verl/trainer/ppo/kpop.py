"""KPop correction for the fixed training and rollout policies.

Reference: https://ringtech.notion.site/kpop
The gate bounds BOTH Bernoulli KL directions, rather than their sum/average.
"""

import torch

from verl import DataProto
from verl.trainer.config import KPopConfig
from verl.utils.config import omega_conf_to_dataclass


def configure_kpop(config) -> KPopConfig | None:
    """Validate KPop before workers start; leave disabled runs untouched."""
    kpop = config.algorithm.get("kpop", None)
    if kpop is None or not kpop.get("enable", False):
        return None
    kpop = omega_conf_to_dataclass(kpop, KPopConfig)
    correction = config.algorithm.get("rollout_correction", None)
    if correction is not None:
        if correction.get("bypass_mode", False):
            raise ValueError("KPop requires algorithm.rollout_correction.bypass_mode=False to recompute old_log_probs")
        if correction.get("rollout_is") is not None or correction.get("rollout_rs") is not None:
            raise ValueError("KPop supplies its own weights and mask; set rollout_is=null and rollout_rs=null")
        if correction.get("rollout_is_batch_normalize", False):
            raise ValueError("KPop requires algorithm.rollout_correction.rollout_is_batch_normalize=False")
        if correction.get("loss_type") == "dis":
            raise ValueError("KPop cannot be combined with rollout_correction.loss_type=dis")
    if not config.actor_rollout_ref.rollout.get("calculate_log_probs", False):
        raise ValueError("KPop requires actor_rollout_ref.rollout.calculate_log_probs=True")
    loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
    if loss_mode in ("bypass_mode", "geo_mean"):
        raise ValueError(
            f"KPop requires a loss that applies token rollout_is_weights; loss_mode={loss_mode} cannot do this"
        )
    return kpop


@torch.no_grad()
def apply_kpop(data: DataProto, config: KPopConfig) -> dict[str, float]:
    """Attach detached ``gate * exp(old_log_probs - rollout_log_probs)``.

    Called after advantage/critic computation and before the actor update. The
    gate is zero for padding and for tokens with either binary KL above phi.
    Keep response_mask unchanged: rejected tokens contribute zero policy loss,
    while the original loss denominator, critic targets, entropy and reference
    KL regularization retain their existing semantics.
    """
    if not config.enable:
        return {}
    required = ("old_log_probs", "rollout_log_probs", "response_mask")
    missing = [key for key in required if key not in data.batch]
    if missing:
        raise ValueError(f"KPop requires {', '.join(missing)} in batch; enable rollout.calculate_log_probs=True")
    if "rollout_is_weights" in data.batch:
        raise ValueError("KPop cannot overwrite existing rollout_is_weights (would mix correction methods)")

    train_logp, rollout_logp, mask = (data.batch[key] for key in required)
    if train_logp.shape != rollout_logp.shape or train_logp.shape != mask.shape or mask.ndim != 2:
        raise ValueError("KPop requires matching [batch, response_length] log probabilities and response_mask")
    valid = mask.bool()
    # Ignore padding before any arithmetic; some rollout engines leave NaNs there.
    train_logp = torch.where(valid, train_logp.detach().float(), 0.0)
    rollout_logp = torch.where(valid, rollout_logp.detach().float(), 0.0)
    if not (torch.isfinite(train_logp).all() and torch.isfinite(rollout_logp).all()):
        raise ValueError("KPop requires finite log probabilities on valid response tokens")

    # A user epsilon below FP32 resolution must not let 1-epsilon round to 1.
    epsilon = max(config.probability_epsilon, torch.finfo(torch.float32).eps)
    p = train_logp.exp().clamp(epsilon, 1.0 - epsilon)
    q = rollout_logp.exp().clamp(epsilon, 1.0 - epsilon)
    log_p_over_q = p.log() - q.log()
    log_complement_ratio = torch.log1p(-p) - torch.log1p(-q)
    forward_kl = (p * log_p_over_q + (1.0 - p) * log_complement_ratio).clamp_min(0.0)
    reverse_kl = (-q * log_p_over_q - (1.0 - q) * log_complement_ratio).clamp_min(0.0)
    keep = valid & (forward_kl <= config.phi) & (reverse_kl <= config.phi)

    # Use the original log probabilities for IS, not the epsilon-clamped KL inputs.
    log_ratio = train_logp - rollout_logp
    ratio = log_ratio.clamp(-config.log_ratio_clip, config.log_ratio_clip).exp()
    weights = torch.where(keep, ratio, 0.0)
    data.batch["rollout_is_weights"] = weights

    count = valid.sum().clamp_min(1)

    def valid_mean(value):
        return (torch.where(valid, value, 0.0).sum(dtype=torch.float64) / count).item()

    return {
        "kpop/filtered_fraction": valid_mean((~keep).float()),
        "kpop/kl_train_rollout_mean": valid_mean(forward_kl),
        "kpop/kl_rollout_train_mean": valid_mean(reverse_kl),
        "kpop/is_weight_mean": valid_mean(weights),
        "kpop/is_weight_max": weights.max().item() if weights.numel() else 0.0,
        "kpop/log_ratio_clipped_fraction": valid_mean((log_ratio.abs() > config.log_ratio_clip).float()),
    }
