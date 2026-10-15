import torch
from transformers import Trainer


class PreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._explained_variance_stats = None
        super().__init__(*args, **kwargs)
        # Custom loss does not use num_items_in_batch; disable related kwargs path
        # to avoid inaccurate loss scaling under gradient accumulation.
        self.model_accepts_loss_kwargs = False

    @torch.no_grad()
    def _update_explained_variance_stats(self, predictions, targets, response_mask):
        """Accumulate token-weighted sufficient statistics for one logging window."""
        valid_tokens = response_mask.detach().bool()
        predictions = predictions.detach()[valid_tokens].float()
        targets = targets.detach()[valid_tokens].float()
        if predictions.numel() == 0:
            return

        errors = targets - predictions

        # Keep the expensive elementwise work in FP32 and convert only the five
        # scalar sufficient statistics to FP64 for stable accumulation/all-reduce.
        count = predictions.new_tensor(predictions.numel()).to(dtype=torch.float64)
        stats = torch.stack(
            (
                count,
                targets.sum().to(dtype=torch.float64),
                targets.square().sum().to(dtype=torch.float64),
                errors.sum().to(dtype=torch.float64),
                errors.square().sum().to(dtype=torch.float64),
            )
        )
        if self._explained_variance_stats is None:
            self._explained_variance_stats = stats
        else:
            self._explained_variance_stats = self._explained_variance_stats + stats

    @torch.no_grad()
    def _consume_explained_variance(self):
        """Aggregate all ranks and return metrics for the current logging window."""
        if self._explained_variance_stats is None:
            return None

        stats = self._explained_variance_stats
        self._explained_variance_stats = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            stats = stats.clone()
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        count, target_sum, target_sq_sum, error_sum, error_sq_sum = stats
        if count.item() <= 0:
            return None

        target_mean = target_sum / count
        error_mean = error_sum / count
        target_variance = (target_sq_sum / count - target_mean.square()).clamp_min(0.0)
        error_variance = (error_sq_sum / count - error_mean.square()).clamp_min(0.0)
        explained_variance = 1.0 - error_variance / (target_variance + 1e-5)

        return {
            "critic/vf_explained_var": explained_variance.item(),
            "critic/vf_target_var": target_variance.item(),
            "critic/vf_num_tokens": count.item(),
        }

    def log(self, logs, *args, **kwargs):
        # Trainer invokes log on every rank at the configured logging_steps.
        # Add metrics only to training-loss records so all ranks participate in
        # distributed reductions at the same point.
        if "loss" in logs or "train_loss" in logs:
            explained_variance_metrics = self._consume_explained_variance()
            if explained_variance_metrics is not None:
                logs = {**logs, **explained_variance_metrics}
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        response_mask = inputs.pop("response_mask")

        outputs = model(**inputs, use_cache=False)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("value model output does not contain logits")
        predictions = torch.sigmoid(logits)
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        elif predictions.ndim != 2:
            raise ValueError(
                "value logits must have shape [batch, seq] or [batch, seq, 1], "
                f"got {tuple(logits.shape)}"
            )

        labels = labels.to(device=predictions.device, dtype=predictions.dtype)
        response_mask = response_mask.to(device=predictions.device, dtype=predictions.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=predictions.device, dtype=predictions.dtype)
            response_mask = response_mask * attention_mask

        if response_mask.shape != predictions.shape:
            raise ValueError(
                f"response_mask shape {tuple(response_mask.shape)} does not match "
                f"predictions shape {tuple(predictions.shape)}"
            )

        # Match verl's critic alignment: the value emitted at position t - 1
        # supervises response token t. For the first response token this uses
        # the value at the final prompt token.
        shifted_predictions = predictions[:, :-1]
        shifted_labels = labels[:, 1:]
        shifted_response_mask = response_mask[:, 1:]

        per_token_loss = (shifted_predictions - shifted_labels).pow(2)
        masked_loss = per_token_loss * shifted_response_mask
        final_loss = masked_loss.sum() / shifted_response_mask.sum().clamp_min(1.0)

        if model.training:
            self._update_explained_variance_stats(
                predictions=shifted_predictions,
                targets=shifted_labels,
                response_mask=shifted_response_mask,
            )

        return (final_loss, outputs) if return_outputs else final_loss
