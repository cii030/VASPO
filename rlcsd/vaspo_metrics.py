"""Read-only VASPO diagnostics by normalized response position."""

import json
from pathlib import Path

import torch


def _position_buckets(mask, bins):
    valid = mask.bool()
    counts = valid.sum(-1, keepdim=True)
    rank = valid.cumsum(-1) - 1
    return valid, ((rank * bins) // counts.clamp_min(1)).clamp(0, bins - 1)


@torch.no_grad()
def position_statistics(values, targets, mask, gap=None, signal=None, delta=None, bins=10):
    """Additive sufficient statistics; EV must be pooled from moments.

    Targets are the current training batch's returns. These are training-rollout
    diagnostics, not held-out continuation calibration.
    """
    if bins < 1 or values.ndim != 2 or values.shape != targets.shape or mask.shape != values.shape:
        raise ValueError("invalid position-statistics shape or bin count")
    valid, bucket = _position_buckets(mask, bins)
    result = []
    for b in range(bins):
        selected = valid & (bucket == b)
        target = targets[selected].double()
        residual = target - values[selected].double()
        entry = dict(bin=b, count=int(selected.sum()), target_sum=float(target.sum()),
                     target_sq_sum=float(target.square().sum()), residual_sum=float(residual.sum()),
                     residual_sq_sum=float(residual.square().sum()))
        for name, tensor in (("gap", gap), ("signal", signal), ("delta", delta)):
            if tensor is not None:
                entry[name + "_sum"] = float(tensor[selected].double().sum())
                entry[name + "_abs_sum"] = float(tensor[selected].double().abs().sum())
        result.append(entry)
    return result


def summarize_bin(entry):
    n = entry["count"]
    if not n:
        return {"mse": None, "explained_variance": None}
    target_var = max(0.0, entry["target_sq_sum"] / n - (entry["target_sum"] / n) ** 2)
    residual_var = max(0.0, entry["residual_sq_sum"] / n - (entry["residual_sum"] / n) ** 2)
    return {"mse": entry["residual_sq_sum"] / n,
            "explained_variance": 1 - residual_var / target_var if n > 1 and target_var > 1e-12 else None}


@torch.no_grad()
def signal_metrics(mask, diagnostics, anchor, prefix="vaspo_actor_position", bins=10):
    """Actor-loop diagnostics use existing scores, with no extra forwards.

    These are worker/microbatch means, reduced by the existing metric logger;
    unlike driver moments they must not be used to reconstruct pooled EV.
    """
    valid, bucket = _position_buckets(mask, bins)
    tensors = {"sampled_gap_abs": diagnostics["teacher_gap"].abs(),
               "teacher_signal_abs": diagnostics["teacher_signal"].abs(),
               "fusion_delta_abs": (diagnostics["fused_raw"] - anchor).abs()}
    result = {}
    for b in range(bins):
        selected = valid & (bucket == b)
        if selected.any():
            for name, tensor in tensors.items():
                result[f"{prefix}/bin_{b:02d}/{name}"] = tensor[selected].float().mean().item()
    return result


@torch.no_grad()
def record_diagnostics(trainer, batch, anchor=None, diagnostics=None):
    """Read the values/returns already selected by the driver; no model forwards."""
    policy = trainer.config.actor_rollout_ref.actor.policy_loss
    tensors = getattr(batch, "batch", None)
    if tensors is None or any(key not in tensors for key in ("values", "returns", "response_mask")):
        return {}
    teacher = diagnostics is not None
    rows = position_statistics(
        batch.batch["values"], batch.batch["returns"], batch.batch["response_mask"],
        gap=diagnostics["teacher_gap"] if teacher else None,
        signal=diagnostics["teacher_signal"] if teacher else None,
        delta=diagnostics["fused_raw"] - anchor if teacher else None)
    path = policy.get("vaspo_position_log", None)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {"step": int(trainer.global_steps), "teacher_scored_on_driver": teacher,
                  "privileged_critic": bool(trainer.config.data.get("use_answer_for_critic", False)), "bins": rows}
        with destination.open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    metrics = {}
    for row in rows:
        prefix = f"vaspo_position/bin_{row['bin']:02d}"
        for key, value in summarize_bin(row).items():
            if value is not None:
                metrics[f"{prefix}/{key}"] = value
        if teacher and row["count"]:
            for name, field in (("sampled_gap_abs", "gap_abs_sum"),
                                ("teacher_signal_abs", "signal_abs_sum"),
                                ("fusion_delta_abs", "delta_abs_sum")):
                metrics[f"{prefix}/{name}"] = row[field] / row["count"]
    return metrics
