"""
Self-distillation integration hooks for verl's standard main_ppo entry point.

Self-distillation flow per rollout step:
1. Student rollout: vLLM generates responses (verl standard)
2. Privileged text extraction: read GT solution/answer from the dataset batch
3. Teacher forward: compute log probs with privileged context on the same response tokens
4. Actor update with distillation loss

The standard ``verl.trainer.main_ppo`` TaskRunner imports this module and
applies its patches whenever a self-distillation loss mode is selected.
"""

import hashlib
import json
import os
import random
import sys

# project_root is the verl-main repo root (parent of this rlcsd/ package).
# We use the in-tree verl (with HEM) directly, so we do NOT prepend any
# vendored third_party/verl. Only the repo root is added so `import rlcsd.*`
# and `import verl.*` both resolve.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn.functional as F
import verl.utils.torch_functional as verl_F
from omegaconf import open_dict

from rlcsd.opsd_format import (
    build_rollout_messages,
    build_teacher_messages,
    RLCSD_TEACHER_TRANSITION_PROMPT,
    extract_boxed_answer,
    normalize_privileged_text_mode,
    strip_legacy_math_prompt)
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_vaspo_fused_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tensor_stats import diagnostic_quantile

RLCSD_LOSS_MODES = {"rlcsd","rlsd_ectr", "opsd_ectr"}

# rlcsd_5 averages teacher_wrong_lp over up to K non-self negative siblings.
# 4 covers the typical group_size=8 case (3-7 non-self negatives) without
# ballooning per-rollout teacher-forward cost.
RLCSD_DEFAULT_K_MAX = 4
DISTILL_LOSS_MODES = {
    "vaspo",
    "opsd",
    "opsd_ppo",
    "sdpo",
    "rlsd",
    "srpo",
    *RLCSD_LOSS_MODES,
}
MATH_DATA_SOURCES = {"dapo_math_17k", "openthoughts_114k_math_filtered", "amc23", "aime24", "aime25"}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for segment in content:
            if isinstance(segment, dict):
                if segment.get("type") == "text":
                    parts.append(str(segment.get("text", "")))
                elif "content" in segment:
                    parts.append(str(segment.get("content", "")))
            elif segment is not None:
                parts.append(str(segment))
        return " ".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)

def _raw_prompt_to_problem(raw_prompt):
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, list):
        user_parts = []
        fallback_parts = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            text = _content_to_text(msg.get("content", ""))
            if not text:
                continue
            fallback_parts.append(text)
            if msg.get("role") == "user":
                user_parts.append(text)
        parts = user_parts or fallback_parts
        return strip_legacy_math_prompt("\n".join(parts).strip())

    if raw_prompt is None:
        return ""
    return strip_legacy_math_prompt(str(raw_prompt).strip())

def _policy_loss_cfg(self):
    return self.config.actor_rollout_ref.actor.policy_loss

def _rollout_custom_cfg(self):
    rollout_cfg = self.config.actor_rollout_ref.get("rollout", {})
    custom_cfg = rollout_cfg.get("custom", None)
    return custom_cfg if custom_cfg is not None else {}

def _cfg_get(self, key, default):
    return _policy_loss_cfg(self).get(key, default)

def _custom_cfg_get(self, key, default):
    custom_cfg = _rollout_custom_cfg(self)
    if key in custom_cfg:
        return custom_cfg.get(key)
    return _cfg_get(self, key, default)

def _cfg_bool(self, key, default=False):
    value = _cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _custom_cfg_bool(self, key, default=False):
    value = _custom_cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _correct_response_mask_from_scores(
    batch: DataProto,
    *,
    threshold: float,
    feature_name: str,
) -> torch.Tensor:
    """Classify each sampled response from the unshaped verifier score."""
    if not np.isfinite(threshold):
        raise ValueError(f"{feature_name} reward threshold must be finite, got {threshold}")

    required_keys = ("token_level_scores", "response_mask")
    missing_keys = [key for key in required_keys if key not in batch.batch]
    if missing_keys:
        raise ValueError(
            f"{feature_name} requires outcome scores and response masks; "
            f"missing batch tensors: {missing_keys}"
        )

    token_level_scores = batch.batch["token_level_scores"]
    response_mask = batch.batch["response_mask"]
    if token_level_scores.shape != response_mask.shape:
        raise ValueError(
            f"{feature_name} requires token_level_scores and response_mask "
            f"to have the same shape, got {token_level_scores.shape} and {response_mask.shape}"
        )

    score_mask = response_mask.to(dtype=token_level_scores.dtype)
    sequence_scores = (token_level_scores * score_mask).sum(dim=-1)
    return sequence_scores >= threshold

def _teacher_total_token_budget(self) -> int:
    actor_cfg = self.config.actor_rollout_ref.actor
    budget = actor_cfg.get("ppo_max_token_len_per_gpu", None)
    if budget is not None:
        return int(budget)
    return int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)

def _teacher_chat_template_kwargs(self):
    kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
    kwargs["enable_thinking"] = _custom_cfg_bool(self, "teacher_enable_thinking", True)
    return kwargs

def _compute_prefilter_metric_overrides(self, batch: DataProto) -> dict[str, float]:
    """Capture trainer-side metrics that should stay on the pre-filter rollout batch."""
    required_keys = {"token_level_scores", "token_level_rewards", "advantages", "returns", "response_mask"}
    if any(key not in batch.batch.keys() for key in required_keys):
        return {}
    prefilter_metrics = compute_data_metrics(batch=batch, use_critic=getattr(self, "use_critic", True))
    kept_prefixes = ("critic/", "response_length/", "response_length_non_aborted/")
    kept_exact_keys = {"response/aborted_ratio"}
    return {
        key: value
        for key, value in prefilter_metrics.items()
        if key.startswith(kept_prefixes) or key in kept_exact_keys
    }

def _get_actor_dp_size(self) -> int:
    """Best-effort lookup for actor data-parallel size during training."""
    try:
        if hasattr(self, "_get_dp_size") and getattr(self, "actor_rollout_wg", None) is not None:
            return int(self._get_dp_size(self.actor_rollout_wg, "actor"))
    except Exception:
        pass
    return 1

def _prepare_actor_update_batch(self, batch: DataProto) -> DataProto:
    """Pad rlcsd actor batches to DP divisibility with zero-loss replicas."""
    if _cfg_get(self, "loss_mode", "vanilla") not in RLCSD_LOSS_MODES:
        return batch

    batch_size = len(batch)
    actor_dp_size = _get_actor_dp_size(self)
    if batch_size == 0 or actor_dp_size <= 1 or batch_size % actor_dp_size == 0:
        metrics = getattr(self, "_rlcsd_step_metrics", {})
        metrics["rlcsd/actor_pad_count"] = 0.0
        metrics["rlcsd/actor_padded_batch_size"] = float(batch_size)
        self._rlcsd_step_metrics = metrics
        return batch

    pad_size = actor_dp_size - (batch_size % actor_dp_size)
    pad_indices = [idx % batch_size for idx in range(pad_size)]
    padded_tail = batch.select_idxs(pad_indices)
    padded_tail.meta_info = dict(batch.meta_info)

    if "response_mask" in padded_tail.batch.keys():
        padded_tail.batch["response_mask"] = torch.zeros_like(padded_tail.batch["response_mask"])
    if "uid" in padded_tail.non_tensor_batch:
        padded_tail.non_tensor_batch["uid"] = np.asarray(
            [f"__rlcsd_pad__{i}" for i in range(pad_size)],
            dtype=object)
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    metrics["rlcsd/actor_pad_count"] = float(pad_size)
    metrics["rlcsd/actor_padded_batch_size"] = float(batch_size + pad_size)
    self._rlcsd_step_metrics = metrics
    actor_batch = DataProto.concat([batch, padded_tail])
    actor_batch.meta_info = dict(batch.meta_info)
    if "global_token_num" in batch.meta_info:
        original_token_num = list(batch.meta_info.get("global_token_num", []))
        actor_batch.meta_info["global_token_num"] = original_token_num + [original_token_num[i] for i in pad_indices]
    return actor_batch

def _collapse_repeated_to_batch(values, batch_size):
    if values is None:
        return None

    values = list(values)
    if len(values) == batch_size:
        return values
    if batch_size <= 0 or len(values) % batch_size != 0:
        raise AssertionError(f"Cannot align {len(values)} items to batch size {batch_size}")

    repeat_factor = len(values) // batch_size
    return [values[i * repeat_factor] for i in range(batch_size)]

def _extract_problems_answers_solutions(self, batch: DataProto, raw_prompts=None, reward_models=None, extra_infos=None):
    """Extract problem texts, GT answers, and GT solutions from the dataset batch."""
    batch_size = len(batch)
    if raw_prompts is None:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if reward_models is None:
        reward_models = batch.non_tensor_batch.get("reward_model")
    if extra_infos is None:
        extra_infos = batch.non_tensor_batch.get("extra_info")

    raw_prompts = _collapse_repeated_to_batch(raw_prompts, batch_size) if raw_prompts is not None else None
    reward_models = _collapse_repeated_to_batch(reward_models, batch_size) if reward_models is not None else None
    extra_infos = _collapse_repeated_to_batch(extra_infos, batch_size) if extra_infos is not None else None

    problems = []
    ground_truths = []
    solutions = []
    for i in range(batch_size):
        raw_prompt = raw_prompts[i] if raw_prompts is not None else None
        reward_model = reward_models[i] if reward_models is not None else {}
        answer = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(answer))

        extra_info = extra_infos[i] if extra_infos is not None else {}
        problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
        if not problem:
            problem = _raw_prompt_to_problem(raw_prompt)
        if not problem and "prompts" in batch.batch:
            problem = strip_legacy_math_prompt(
                self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True).strip()
            )
        problems.append(problem)

        solution = extra_info.get("solution", "") if isinstance(extra_info, dict) else ""
        if _cfg_get(self, "loss_mode", "vanilla") == "vaspo" and isinstance(solution, (list, tuple, np.ndarray)):
            solution = solution[0] if len(solution) else ""
        solutions.append(_content_to_text(solution).strip())

    return problems, ground_truths, solutions

def _build_teacher_prompt(problem, answer, solution, mode, transition_prompt=None,
                          wrapper_variant="neutral", thinking=False):
    privileged_text, messages, _ = build_teacher_messages(
        problem=problem,
        answer=answer,
        solution=solution,
        mode=mode,
        transition_prompt=transition_prompt,
        wrapper_variant=wrapper_variant,
        thinking=thinking)
    return privileged_text, messages

def _materialize_teacher_context(self, problem, answer, solution):
    """Build a teacher context for the rlcsd_* methods.

    rlcsd_* uses the lighter RLCSD_TEACHER_TRANSITION_PROMPT (does not ask
    the model to derive a separate answer). opsd/sdpo/rlsd/srpo continue
    to go through _build_teacher_context_payloads and keep the original
    TEACHER_TRANSITION_PROMPT.

    The "neutral" wrapper variant reproduces this lighter wording.  Setting
    teacher_wrapper_variant=verbose / terse in the YAML config swaps both
    the framing line and the transition prompt for the robustness probe.
    """
    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    privileged_text, messages = _build_teacher_prompt(
        problem=problem,
        answer=answer,
        solution=solution,
        mode="solution_answer",
        wrapper_variant=wrapper_variant,
        thinking=use_thinking)
    prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_teacher_chat_template_kwargs(self))
    return {
        "prompt": prompt,
        "text": privileged_text,
        "answer": answer}

def _sample_other_rollout(indices, current_idx):
    candidate_indices = [idx for idx in indices if idx != current_idx]
    if not candidate_indices:
        return None
    return random.choice(candidate_indices)

def _expand_to_rollout(values, batch_size, rollout_n):
    num_values = len(values)
    if num_values == batch_size:
        return values
    if num_values * rollout_n != batch_size:
        raise AssertionError(f"Mismatch: cached={num_values}, rollout_n={rollout_n}, batch={batch_size}")

    expanded = []
    for value in values:
        for _ in range(rollout_n):
            expanded.append(value)
    return expanded

def _build_teacher_context_payloads(self, problems, ground_truths, solutions):
    # Existing self-distillation methods default to full solution evidence.
    # VASPO's first implementation shares the ground-truth answer with its
    # answer-conditioned critic, so make that safe behavior the VASPO default
    # even when the caller does not provide a custom rollout section.
    default_mode = "solution_answer"
    requested_mode = normalize_privileged_text_mode(_custom_cfg_get(self, "privileged_text_mode", default_mode))
    privileged_texts = []
    privilege_hashes = []
    privilege_token_lengths = []
    effective_modes = []
    teacher_prompts = []

    if requested_mode == "solution_answer":
        missing_indices = [i for i, solution in enumerate(solutions) if not solution]
        if missing_indices:
            preview = ", ".join(str(i) for i in missing_indices[:8])
            if len(missing_indices) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "privileged_text_mode=solution_answer requires every sample to have a non-empty GT solution, "
                f"but {len(missing_indices)}/{len(solutions)} samples in the current batch are missing it "
                f"(batch indices: {preview})."
            )

    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    for problem, answer, solution in zip(problems, ground_truths, solutions):
        privileged_text, messages, effective_mode = build_teacher_messages(
            problem=problem,
            answer=answer,
            solution=solution,
            mode=requested_mode,
            wrapper_variant=wrapper_variant,
            thinking=use_thinking)
        privileged_texts.append(privileged_text)
        privilege_hashes.append(hashlib.sha256(privileged_text.encode("utf-8")).hexdigest())
        privilege_token_lengths.append(len(self.tokenizer.encode(privileged_text, add_special_tokens=False)))
        effective_modes.append(effective_mode)
        teacher_prompts.append(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_teacher_chat_template_kwargs(self))
        )

    return {
        "privileged_texts": privileged_texts,
        "privilege_hashes": privilege_hashes,
        "privilege_token_lengths": privilege_token_lengths,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "rationales": privileged_texts}

def _generate_rationales(self, batch: DataProto, gen_output: DataProto) -> dict | None:
    """Cache GT privileged text from the dataset batch for teacher forwarding."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return None

    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if raw_prompts is None:
        raw_prompts = gen_output.non_tensor_batch.get("raw_prompt")

    problems, ground_truths, solutions = self._extract_problems_answers_solutions(
        batch,
        raw_prompts=raw_prompts,
        reward_models=batch.non_tensor_batch.get("reward_model"),
        extra_infos=batch.non_tensor_batch.get("extra_info"))
    payloads = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    payloads.update({"problems": problems, "ground_truths": ground_truths, "solutions": solutions})
    return payloads

def _align_resp_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate the response dimension (dim=1) to match target_len."""
    current_len = tensor.shape[1]
    if current_len == target_len:
        return tensor
    if current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len), value=0.0)
    return tensor[:, :target_len]

def _sampled_response_token_lengths(responses, response_mask, pad_id) -> list[int]:
    """Return each row's sampled-token count using response_mask when available.

    The teacher packer and the actor's compact-response realignment must agree
    on one length per row. The actor derives it from ``response_mask.sum(-1)``,
    so the packer must too: filtering by ``token != pad_id`` desyncs whenever a
    sampled token legitimately equals pad_id (Qwen3 lists <|endoftext|> in both
    ``pad_token_id`` and ``eos_token_id``) or whenever response_mask has interior
    zeros (multi-turn/tool tokens). Either desync silently shifts every teacher
    log probability in that row.
    """
    if response_mask is not None:
        mask = response_mask.to(device=responses.device).bool()
        if mask.shape != responses.shape:
            raise ValueError(
                f"response_mask shape {tuple(mask.shape)} must match responses shape {tuple(responses.shape)}"
            )
        counts = mask.sum(dim=-1)
        # The actor realigns teacher outputs by taking one contiguous slice of
        # length counts[i], so only a leading run of sampled tokens can be
        # aligned. An interior zero (multi-turn / tool-response tokens) has no
        # correct contiguous packing; fail loudly instead of shifting the row.
        prefix_run = (torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0) < counts.unsqueeze(-1))
        holes = (mask != prefix_run).any(dim=-1)
        if holes.any():
            raise ValueError(
                "Teacher packing requires response_mask to be a leading run of sampled tokens per row, "
                f"but rows {holes.nonzero(as_tuple=True)[0].tolist()[:8]} have interior zeros. "
                "Teacher scoring does not support masked interior tokens (e.g. multi-turn tool responses)."
            )
        return counts.tolist()
    return [int((row != pad_id).sum()) for row in responses]

def _build_teacher_batch_from_prompts(
    self, teacher_prompts, responses, prefix="teacher", response_mask=None
) -> DataProto:
    """Pack teacher prompts and sampled responses into model-ready tensors.

    ``response_mask`` selects the sampled tokens per row. It must be the same
    mask the actor uses to realign teacher outputs; see
    ``_sampled_response_token_lengths``.
    """
    batch_size = len(teacher_prompts)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    pad_id = self.tokenizer.pad_token_id or 0
    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]

    response_lengths = _sampled_response_token_lengths(responses, response_mask, pad_id)
    teacher_input_ids_list = []
    teacher_prompt_lens = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        prompt_ids = teacher_prompt_ids_list[i]
        # Take the leading run of sampled tokens so the packed response stays
        # contiguous and its length matches response_mask.sum() exactly.
        response_ids = responses[i].tolist()[: response_lengths[i]]
        max_prompt_tokens = max(teacher_total_budget - len(response_ids), 0)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
        teacher_input_ids_list.append(prompt_ids + response_ids)
        teacher_prompt_lens.append(len(prompt_ids))

    max_prompt_len = max(teacher_prompt_lens)
    response_len = max(len(ids) - teacher_prompt_lens[i] for i, ids in enumerate(teacher_input_ids_list))
    total_len = max_prompt_len + response_len

    teacher_input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
    teacher_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long)
    teacher_position_ids = torch.zeros((batch_size, total_len), dtype=torch.long)

    for i, ids in enumerate(teacher_input_ids_list):
        prompt_len = teacher_prompt_lens[i]
        prompt_offset = max_prompt_len - prompt_len
        response_ids = ids[prompt_len:]
        response_count = len(response_ids)

        teacher_input_ids[i, prompt_offset:max_prompt_len] = torch.tensor(ids[:prompt_len], dtype=torch.long)
        if response_count > 0:
            teacher_input_ids[i, max_prompt_len:max_prompt_len + response_count] = torch.tensor(
                response_ids, dtype=torch.long
            )

        teacher_attention_mask[i, prompt_offset:max_prompt_len + response_count] = 1
        teacher_position_ids[i, prompt_offset:max_prompt_len + response_count] = torch.arange(
            prompt_len + response_count
        )

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": teacher_input_ids,
            f"{prefix}_attention_mask": teacher_attention_mask,
            f"{prefix}_position_ids": teacher_position_ids}
    )

def _build_teacher_multi_batch_from_prompt_groups(
    self, teacher_prompt_groups, responses, prefix="teacher", response_mask=None
) -> DataProto:
    """Pack per-sample teacher prompt groups into [batch, num_ctx, seq] tensors."""
    batch_size = len(teacher_prompt_groups)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    group_sizes = [len(group) for group in teacher_prompt_groups]
    max_group_size = max(group_sizes, default=0)
    if max_group_size <= 0:
        raise ValueError(f"{prefix} received no teacher prompts to pack.")

    # Replicate the first prompt of each sample to fill empty slots so every
    # (sample, ctx) carries a non-empty input. This lets the actor's
    # multi-context teacher forward run unconditionally on every column,
    # avoiding NCCL collective desync when DP ranks see different valid_mask
    # column populations. Replicas are marked invalid in valid_mask so the
    # downstream loss ignores their outputs.
    flat_prompts = []
    flat_responses = []
    flat_masks = []
    flat_indices = []
    for sample_idx, prompt_group in enumerate(teacher_prompt_groups):
        if not prompt_group:
            raise ValueError(f"{prefix} sample {sample_idx} has an empty prompt group.")
        for group_idx in range(max_group_size):
            real = group_idx < len(prompt_group)
            prompt = prompt_group[group_idx] if real else prompt_group[0]
            flat_prompts.append(prompt)
            flat_responses.append(responses[sample_idx])
            if response_mask is not None:
                flat_masks.append(response_mask[sample_idx])
            flat_indices.append((sample_idx, group_idx, real))

    flat_responses = torch.stack(flat_responses, dim=0)
    flat_prefix = f"{prefix}_flat"
    flat_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=flat_prompts,
        responses=flat_responses,
        prefix=flat_prefix,
        response_mask=torch.stack(flat_masks, dim=0) if flat_masks else None)

    flat_input_ids = flat_batch.batch[f"{flat_prefix}_input_ids"]
    flat_attention_mask = flat_batch.batch[f"{flat_prefix}_attention_mask"]
    flat_position_ids = flat_batch.batch[f"{flat_prefix}_position_ids"]
    seq_len = flat_input_ids.shape[1]

    input_ids = flat_input_ids.new_zeros((batch_size, max_group_size, seq_len))
    attention_mask = flat_attention_mask.new_zeros((batch_size, max_group_size, seq_len))
    position_ids = flat_position_ids.new_zeros((batch_size, max_group_size, seq_len))
    valid_mask = torch.zeros((batch_size, max_group_size), dtype=torch.bool)

    for flat_idx, (sample_idx, group_idx, real) in enumerate(flat_indices):
        input_ids[sample_idx, group_idx] = flat_input_ids[flat_idx]
        attention_mask[sample_idx, group_idx] = flat_attention_mask[flat_idx]
        position_ids[sample_idx, group_idx] = flat_position_ids[flat_idx]
        if real:
            valid_mask[sample_idx, group_idx] = True

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attention_mask": attention_mask,
            f"{prefix}_position_ids": position_ids,
            f"{prefix}_valid_mask": valid_mask},
        num_batch_dims=1)

def _compute_rlcsd_teacher_inputs(self, batch: DataProto) -> DataProto | None:
    """Build positive/negative teacher prompts from rollout groups identified by uid."""
    self._data_metric_overrides = _compute_prefilter_metric_overrides(self, batch)
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in RLCSD_LOSS_MODES:
        raise ValueError(f"_compute_rlcsd_teacher_inputs called with unsupported loss_mode={loss_mode}")
    batch_size = batch.batch.batch_size[0]
    if batch_size == 0:
        batch.meta_info["skip_actor_update"] = True
        self._aligned_teacher_rollout_details = {}
        self._rlcsd_step_metrics = {"rlcsd/skip_actor_update": 1.0}
        return None

    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        raise ValueError("rlcsd_* requires uid in non_tensor_batch to recover rollout groups.")

    problems, ground_truths, _ = self._extract_problems_answers_solutions(batch)
    responses = batch.batch["responses"]
    responses_text = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    reward_tensor = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch.keys() else batch.batch["rm_scores"]
    if reward_tensor.dim() > 1:
        rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    else:
        rewards = reward_tensor.detach().cpu().tolist()

    groups = {}
    for idx, uid in enumerate(uid_values.tolist()):
        groups.setdefault(str(uid), []).append(idx)

    num_aborted = 0
    all_same_outcome_group_count = 0
    self_negative_missing_boxed_count = 0
    self_excluded_sampling_skip_count = 0
    valid_mask = np.zeros(batch_size, dtype=bool)
    correct_prompt_by_idx = {}
    wrong_prompt_by_idx = {}
    correct_text_by_idx = {}
    wrong_text_by_idx = {}
    correct_answer_by_idx = {}
    wrong_answer_by_idx = {}
    correct_prompt_group_by_idx = {}
    wrong_prompt_group_by_idx = {}
    correct_text_group_by_idx = {}
    wrong_text_group_by_idx = {}
    correct_answer_group_by_idx = {}
    wrong_answer_group_by_idx = {}
    correct_pool_sizes = []
    wrong_pool_sizes = []
    valid_group_count = 0
    for group_indices in groups.values():
        positives = [idx for idx in group_indices if rewards[idx] > 0]
        negatives = [idx for idx in group_indices if rewards[idx] <= 0]
        if not positives or not negatives:
            num_aborted += 1
            all_same_outcome_group_count += 1
            continue

        problem = problems[group_indices[0]]
        positive_contexts = {}
        for idx in positives:
            correct_answer = extract_boxed_answer(responses_text[idx]) or ground_truths[idx]
            positive_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=correct_answer,
                solution=responses_text[idx])

        negative_contexts = {}
        for idx in negatives:
            wrong_answer = extract_boxed_answer(responses_text[idx])
            if wrong_answer is None:
                continue
            negative_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=wrong_answer,
                solution=responses_text[idx])

        valid_positive_indices = list(positive_contexts)
        valid_negative_indices = list(negative_contexts)
        if not valid_positive_indices or not valid_negative_indices:
            num_aborted += 1
            continue

        group_valid_count = 0
        for idx in group_indices:
            if idx in positive_contexts:
                positive_idx = _sample_other_rollout(valid_positive_indices, idx)
                if positive_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                correct_context = positive_contexts[positive_idx]
                wrong_idx = random.choice(valid_negative_indices)
                wrong_context = negative_contexts[wrong_idx]
                rlcsd_wrong_pool = list(valid_negative_indices)
            else:
                negative_idx = _sample_other_rollout(valid_negative_indices, idx)
                if negative_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                wrong_context = negative_contexts.get(negative_idx)
                if wrong_context is None:
                    self_negative_missing_boxed_count += 1
                    continue
                wrong_idx = negative_idx
                correct_context = positive_contexts[random.choice(valid_positive_indices)]
                rlcsd_wrong_pool = [c for c in valid_negative_indices if c != idx]

            valid_mask[idx] = True
            correct_prompt_by_idx[idx] = correct_context["prompt"]
            wrong_prompt_by_idx[idx] = wrong_context["prompt"]
            correct_text_by_idx[idx] = correct_context["text"]
            wrong_text_by_idx[idx] = wrong_context["text"]
            correct_answer_by_idx[idx] = correct_context["answer"]
            wrong_answer_by_idx[idx] = wrong_context["answer"]
            if loss_mode == "rlcsd":
                # K = min(rlcsd_k_max, |non-self negatives|). Always include the
                # chosen single-pair wrong sibling first so the marginal mean
                # covers it; fill remaining slots with random other non-self
                # negatives.
                k_max = int(_cfg_get(self, "rlcsd_k_max", RLCSD_DEFAULT_K_MAX) or RLCSD_DEFAULT_K_MAX)
                rest = [c for c in rlcsd_wrong_pool if c != wrong_idx]
                random.shuffle(rest)
                chosen = [wrong_idx] + rest[: max(k_max - 1, 0)]
                wrong_prompt_group_by_idx[idx] = [negative_contexts[c]["prompt"] for c in chosen]
                wrong_text_group_by_idx[idx]   = [negative_contexts[c]["text"]   for c in chosen]
                wrong_answer_group_by_idx[idx] = [negative_contexts[c]["answer"] for c in chosen]
                wrong_pool_sizes.append(len(chosen))
            group_valid_count += 1

        if group_valid_count > 0:
            valid_group_count += 1
        else:
            num_aborted += 1

    num_groups = len(groups)
    valid_sample_count = int(valid_mask.sum())
    step_metrics = {
        "rlcsd/aborted_ratio": float(num_aborted / max(num_groups, 1)),
        "rlcsd/valid_group_ratio": float(valid_group_count / max(num_groups, 1)),
        "rlcsd/valid_group_count": float(valid_group_count),
        "rlcsd/valid_sample_count": float(valid_sample_count),
        "rlcsd/group_count": float(num_groups),
        "rlcsd/all_same_outcome_group_count": float(all_same_outcome_group_count),
        "rlcsd/self_negative_missing_boxed_count": float(self_negative_missing_boxed_count),
        "rlcsd/self_excluded_sampling_skip_count": float(self_excluded_sampling_skip_count),
        "rlcsd/skipped_sample_count": float(batch_size - valid_sample_count),
        "rlcsd/actor_pad_count": 0.0,
        "rlcsd/actor_padded_batch_size": float(valid_sample_count),
        "rlcsd/correct_pool_size_mean": float(np.mean(correct_pool_sizes)) if correct_pool_sizes else 0.0,
        "rlcsd/wrong_pool_size_mean": float(np.mean(wrong_pool_sizes)) if wrong_pool_sizes else 0.0}

    if not valid_mask.any():
        filtered_batch = batch.select_idxs(valid_mask)
        batch.batch = filtered_batch.batch
        batch.non_tensor_batch = filtered_batch.non_tensor_batch
        batch.meta_info["skip_actor_update"] = True
        step_metrics["rlcsd/skip_actor_update"] = 1.0
        self._aligned_teacher_rollout_details = {
            "teacher_correct_prompts": [],
            "teacher_wrong_prompts": [],
            "privileged_texts_correct": [],
            "privileged_texts_wrong": [],
            "correct_answers": [],
            "wrong_answers": []}
        self._rlcsd_step_metrics = step_metrics
        return None

    kept_indices = [idx for idx in range(batch_size) if valid_mask[idx]]
    filtered_batch = batch.select_idxs(valid_mask)
    batch.batch = filtered_batch.batch
    batch.non_tensor_batch = filtered_batch.non_tensor_batch

    if loss_mode == "rlcsd":
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        # rlcsd_5 only marginalizes on the wrong side; correct stays single.
        teacher_correct_prompt_groups = []
        privileged_text_groups_correct = []
        correct_answer_groups = []
        teacher_wrong_prompt_groups = [wrong_prompt_group_by_idx[idx] for idx in kept_indices]
        privileged_text_groups_wrong = [wrong_text_group_by_idx[idx] for idx in kept_indices]
        wrong_answer_groups = [wrong_answer_group_by_idx[idx] for idx in kept_indices]
    else:
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        teacher_correct_prompt_groups = []
        teacher_wrong_prompt_groups = []
        privileged_text_groups_correct = []
        privileged_text_groups_wrong = []
        correct_answer_groups = []
        wrong_answer_groups = []

    self._aligned_teacher_rollout_details = {
        "teacher_correct_prompts": teacher_correct_prompts,
        "teacher_wrong_prompts": teacher_wrong_prompts,
        "privileged_texts_correct": privileged_texts_correct,
        "privileged_texts_wrong": privileged_texts_wrong,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "teacher_correct_prompt_groups": teacher_correct_prompt_groups,
        "teacher_wrong_prompt_groups": teacher_wrong_prompt_groups,
        "privileged_text_groups_correct": privileged_text_groups_correct,
        "privileged_text_groups_wrong": privileged_text_groups_wrong,
        "correct_answer_groups": correct_answer_groups,
        "wrong_answer_groups": wrong_answer_groups,
        "effective_modes": ["solution_answer"] * len(teacher_correct_prompts)}
    self._rlcsd_step_metrics = step_metrics

    rlcsd_response_mask = batch.batch.get("response_mask")
    if loss_mode == "rlcsd":
        teacher_correct_pair = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct",
            response_mask=rlcsd_response_mask)
        teacher_wrong_multi = _build_teacher_multi_batch_from_prompt_groups(
            self,
            teacher_prompt_groups=teacher_wrong_prompt_groups,
            responses=batch.batch["responses"],
            prefix="teacher_wrong_multi",
            response_mask=rlcsd_response_mask)
        return teacher_correct_pair.union(teacher_wrong_multi)
    else:
        teacher_correct = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct",
            response_mask=rlcsd_response_mask)
        teacher_wrong = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_wrong_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_wrong",
            response_mask=rlcsd_response_mask)
        return teacher_correct.union(teacher_wrong)

# ------------------------------------------------------------------ #
# Step 3: Compute teacher log probs (called during training phase)
# ------------------------------------------------------------------ #

def _compute_teacher_log_probs(self, batch: DataProto) -> DataProto | None:
    """Build teacher-side inputs from cached GT privileged prompts."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    self._data_metric_overrides = {}
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return _compute_rlcsd_teacher_inputs(self, batch)

    self._rlcsd_step_metrics = {}

    batch_size = batch.batch.batch_size[0]
    problems, ground_truths, solutions = self._extract_problems_answers_solutions(batch)
    aligned_context = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    self._aligned_teacher_rollout_details = aligned_context
    teacher_prompts = aligned_context["teacher_prompts"]

    if len(teacher_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        teacher_prompts = _expand_to_rollout(teacher_prompts, batch_size, rollout_n)
        aligned_context["privilege_hashes"] = _expand_to_rollout(
            aligned_context["privilege_hashes"], batch_size, rollout_n
        )
        aligned_context["privilege_token_lengths"] = _expand_to_rollout(
            aligned_context["privilege_token_lengths"], batch_size, rollout_n
        )

    batch.non_tensor_batch["privilege_hash"] = np.asarray(aligned_context["privilege_hashes"], dtype=object)
    batch.non_tensor_batch["privilege_token_length"] = np.asarray(
        aligned_context["privilege_token_lengths"], dtype=np.int32
    )

    responses = batch.batch["responses"]
    pad_id = self.tokenizer.pad_token_id or 0
    response_mask = batch.batch.get("response_mask")
    # Mirror the packer's length definition so the truncation budgets below
    # match the tokens actually packed.
    response_lengths = _sampled_response_token_lengths(responses, response_mask, pad_id)

    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]
    teacher_prompt_lens = []
    teacher_prompt_lens_raw = []
    teacher_prompt_tokens_truncated = []
    teacher_prompt_truncated = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        raw_prompt_len = len(teacher_prompt_ids_list[i])
        max_prompt_tokens = max(teacher_total_budget - response_lengths[i], 0)
        kept_prompt_len = min(raw_prompt_len, max_prompt_tokens)
        truncated_tokens = raw_prompt_len - kept_prompt_len
        teacher_prompt_lens.append(kept_prompt_len)
        teacher_prompt_lens_raw.append(raw_prompt_len)
        teacher_prompt_tokens_truncated.append(truncated_tokens)
        teacher_prompt_truncated.append(1.0 if truncated_tokens > 0 else 0.0)

    batch.non_tensor_batch["teacher_prompt_truncated"] = np.asarray(teacher_prompt_truncated, dtype=np.float32)
    batch.non_tensor_batch["teacher_prompt_truncated_tokens"] = np.asarray(
        teacher_prompt_tokens_truncated, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_before_trunc"] = np.asarray(
        teacher_prompt_lens_raw, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_after_trunc"] = np.asarray(
        teacher_prompt_lens, dtype=np.float32
    )
    teacher_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_prompts,
        responses=responses,
        prefix="teacher",
        response_mask=response_mask)

    if loss_mode == "opsd_ppo" and _cfg_bool(self, "opsd_correct_only", False):
        correct_reward_threshold = float(_cfg_get(self, "opsd_correct_reward_threshold", 0.5))
        opsd_correct_mask = _correct_response_mask_from_scores(
            batch,
            threshold=correct_reward_threshold,
            feature_name="opsd_correct_only",
        )
        teacher_batch.batch["opsd_correct_mask"] = opsd_correct_mask

        metrics = getattr(self, "_rlcsd_step_metrics", {})
        metrics["opsd_ppo/correct_only_enabled"] = 1.0
        metrics["opsd_ppo/correct_reward_threshold"] = correct_reward_threshold
        metrics["opsd_ppo/correct_response_ratio"] = opsd_correct_mask.float().mean().item()
        self._rlcsd_step_metrics = metrics

    return teacher_batch

def _compute_vaspo_teacher_scores(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
    """Score sampled tokens with VASPO's detached privileged self-teacher.

    This runs before any actor or critic update, using the same teacher
    forward and mask alignment as RLSD/OPSD. EMA/snapshot modes use the
    teacher shadow; fixed mode disables LoRA adapters when available, as in
    the ordinary teacher branch. Only sampled-token log probabilities are
    materialized. With vaspo_teacher_in_actor_update enabled, this hook only
    prepares inputs; scoring and fusion run in the actor loop. Optional
    pre-fusion whitening is performed globally by the driver beforehand.
    With a cached teacher and current student source, scores stay in the batch
    for fusion after each actor forward.
    """
    if _cfg_get(self, "loss_mode", "vanilla") != "vaspo":
        return batch, {}

    required_keys = ("responses", "response_mask", "old_log_probs")
    missing_keys = [key for key in required_keys if key not in batch.batch]
    if missing_keys:
        raise ValueError(f"VASPO teacher scoring is missing batch tensors: {missing_keys}")

    vaspo_correct_only = _cfg_bool(self, "vaspo_correct_only", True)

    teacher_inputs = _compute_teacher_log_probs(self, batch)
    if teacher_inputs is None:
        raise RuntimeError("VASPO could not build privileged teacher inputs")

    response_mask = batch.batch["response_mask"]
    response_length = int(batch.batch["responses"].shape[-1])
    if not response_mask.bool().any():
        raise ValueError("VASPO requires at least one valid response token")

    if _cfg_bool(self, "vaspo_teacher_in_actor_update", False):
        if (not _cfg_bool(self, "vaspo_whiten_before_fusion", False)
                and self.config.algorithm.get("normalize_advantages", True) is not False):
            raise ValueError(
                "VASPO actor-loop fusion requires algorithm.normalize_advantages=False "
                "or vaspo_whiten_before_fusion=True"
            )
        batch = batch.union(teacher_inputs)
        metrics = {
            "vaspo/teacher_in_actor_update": 1.0,
            "vaspo/correct_only_enabled": float(vaspo_correct_only),
        }
        return batch, metrics

    teacher_mode = str(_cfg_get(self, "teacher_mode", "fixed")).strip().lower()
    use_teacher_shadow = teacher_mode in {"ema", "snapshot"}
    teacher_scoring_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_inputs.batch["teacher_input_ids"],
            "attention_mask": teacher_inputs.batch["teacher_attention_mask"],
            "position_ids": teacher_inputs.batch["teacher_position_ids"],
            "responses": batch.batch["responses"],
            "response_mask": response_mask,
        },
        meta_info={
            "calculate_entropy": False,
            "use_teacher_shadow": use_teacher_shadow,
            "use_teacher_forward": True,
        },
    )
    teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_scoring_batch)
    teacher_log_probs = teacher_output.batch.get("old_log_probs", None)
    if teacher_log_probs is None:
        teacher_log_probs = teacher_output.batch.get("log_probs", None)
    if teacher_log_probs is None:
        raise RuntimeError(
            "VASPO teacher scoring expected old_log_probs or log_probs from the actor worker, "
            f"but received {list(teacher_output.batch.keys())}"
        )

    expected_shape = (len(batch), response_length)
    if teacher_log_probs.shape != expected_shape:
        raise RuntimeError(
            f"VASPO teacher scoring expected response shape {expected_shape}, got {teacher_log_probs.shape}"
        )
    teacher_log_probs = teacher_log_probs.float()
    batch.batch["vaspo_teacher_log_probs"] = teacher_log_probs

    old_log_probs = batch.batch["old_log_probs"].float()
    valid_mask = response_mask.bool()
    teacher_gap = teacher_log_probs - old_log_probs
    valid_gap = teacher_gap[valid_mask]
    privilege_token_lengths = np.asarray(batch.non_tensor_batch["privilege_token_length"], dtype=np.float32)
    privilege_hashes = np.asarray(batch.non_tensor_batch["privilege_hash"], dtype=object)
    metrics = {
        "vaspo/teacher_in_actor_update": 0.0,
        "vaspo/privilege_token_length_mean": float(privilege_token_lengths.mean()),
        "vaspo/privilege_unique_hashes": float(len(np.unique(privilege_hashes))),
        "vaspo/teacher_log_prob_mean": teacher_log_probs[valid_mask].mean().item(),
        "vaspo/teacher_gap_mean": valid_gap.mean().item(),
        "vaspo/teacher_gap_abs_mean": valid_gap.abs().mean().item(),
        "vaspo/teacher_uses_shadow": float(use_teacher_shadow),
        "vaspo/correct_only_enabled": float(vaspo_correct_only),
    }
    return batch, metrics

def _apply_vaspo_advantage_fusion(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
    """Prepare the global PPO anchor, then fuse here or defer fusion to actor."""
    if _cfg_get(self, "loss_mode", "vanilla") != "vaspo":
        return batch, {}
    normalize_advantages = self.config.algorithm.get("normalize_advantages", True)
    whiten_before_fusion = _cfg_bool(self, "vaspo_whiten_before_fusion", False)
    if whiten_before_fusion:
        # Use the same masked whitening as PPO, once on the complete rollout
        # batch after the final GAE computation (including any critic pre-update).
        # Teacher/advantage gates must not restrict the whitening statistics.
        with torch.no_grad():
            valid = batch.batch["response_mask"].bool()
            anchor = torch.where(valid, batch.batch["advantages"].detach().float(), 0.0)
            if valid.sum() > 1:
                anchor = verl_F.masked_whiten(anchor, valid)
            else:
                # A singleton has zero centered advantage; an empty batch stays zero.
                anchor = torch.zeros_like(anchor)
            batch.batch["advantages"] = torch.where(valid, anchor, 0.0)
    if (_cfg_bool(self, "vaspo_teacher_in_actor_update", False)
            or str(_cfg_get(self, "vaspo_student_prob_source", "old")).strip().lower() == "current"):
        if not whiten_before_fusion and normalize_advantages is not False:
            raise ValueError(
                "VASPO actor-loop fusion requires algorithm.normalize_advantages=False "
                "or vaspo_whiten_before_fusion=True"
            )
        # Revisited PPO epochs reuse this unchanged, globally prepared anchor.
        batch.meta_info["vaspo_raw_advantages"] = not whiten_before_fusion
        batch.meta_info["vaspo_whitened_advantages"] = whiten_before_fusion
        from rlcsd.vaspo_metrics import record_diagnostics

        metrics = record_diagnostics(self, batch)
        metrics["vaspo/advantage_normalization_enabled"] = float(whiten_before_fusion)
        metrics["vaspo/whiten_before_fusion"] = float(whiten_before_fusion)
        metrics["vaspo/student_prob_source_current"] = float(
            str(_cfg_get(self, "vaspo_student_prob_source", "old")).strip().lower() == "current"
        )
        return batch, metrics

    required_keys = ("advantages", "vaspo_teacher_log_probs", "old_log_probs", "response_mask")
    missing_keys = [key for key in required_keys if key not in batch.batch]
    if missing_keys:
        raise ValueError(f"VASPO advantage fusion is missing batch tensors: {missing_keys}")
    anchor = batch.batch["advantages"].detach().clone()
    correct_only = _cfg_bool(self, "vaspo_correct_only", True)
    fused, diagnostics = compute_vaspo_fused_advantage(
        advantages=anchor,
        teacher_log_probs=batch.batch["vaspo_teacher_log_probs"],
        old_log_probs=batch.batch["old_log_probs"],
        response_mask=batch.batch["response_mask"],

        tanh_tau=float(_cfg_get(self, "vaspo_tanh_tau", 0.02)),
        positive_advantage_only=correct_only,
        whiten_advantages=normalize_advantages and not whiten_before_fusion,
        token_mask_enabled=_cfg_bool(self, "vaspo_use_token_mask", True),
        token_mask_threshold=float(_cfg_get(self, "vaspo_token_mask_threshold", 0.02)),

        epsilon_w=float(_cfg_get(self, "epsilon_w", 0.2)),
        student_prob_source=str(_cfg_get(self, "vaspo_student_prob_source", "old")).strip().lower(),
    )
    batch.batch["vaspo_anchor_advantages"] = diagnostics["anchor_for_loss"]
    batch.batch["vaspo_token_mask"] = diagnostics["applied_token_mask"]
    batch.batch["advantages"] = fused
    from rlcsd.vaspo_metrics import record_diagnostics

    position_metrics = record_diagnostics(self, batch, anchor, diagnostics)
    batch.batch.pop("vaspo_teacher_log_probs")

    valid = batch.batch["response_mask"].bool()
    def valid_mean(tensor):
        return tensor[valid].float().mean().item() if valid.any() else 0.0

    # Modulation strength on the selected subset only; applied_token_ratio gives
    # coverage, these give magnitude. The plain mean shows the net amplify/suppress
    # bias but cancels a symmetric signal to 1.0, so report the deviation too:
    # abs_dev near 0 means the mask admits tokens the teacher barely moves.
    selected = diagnostics["applied_token_mask"].bool() & valid
    if selected.any():
        selected_weights = diagnostics["fusion_weight"][selected].float()
        weight_mean_selected = selected_weights.mean().item()
        weight_abs_dev_selected = (selected_weights - 1.0).abs().mean().item()
    else:
        weight_mean_selected, weight_abs_dev_selected = 1.0, 0.0
    # 90th percentile of |d| over valid tokens, for calibrating the mask
    # threshold: if this sits below the threshold, the mask is filtering too hard.
    gap_abs_valid = diagnostics["teacher_gap"][valid].abs().float()
    gap_abs_p90 = diagnostic_quantile(gap_abs_valid, 0.9)

    metrics = {

        "vaspo/tanh_tau": float(_cfg_get(self, "vaspo_tanh_tau", 0.02)),

        "vaspo/student_prob_source_current": 0.0,

        "vaspo/token_mask_enabled": float(_cfg_bool(self, "vaspo_use_token_mask", True)),
        "vaspo/token_mask_threshold": float(_cfg_get(self, "vaspo_token_mask_threshold", 0.02)),

        "vaspo/token_mask_ratio": valid_mean(diagnostics["token_mask"]),
        "vaspo/applied_token_ratio": valid_mean(diagnostics["applied_token_mask"]),
        "vaspo/positive_advantage_token_ratio": valid_mean(anchor > 0),
        "vaspo/weight_mean_selected": weight_mean_selected,
        "vaspo/weight_abs_dev_selected": weight_abs_dev_selected,
        "vaspo/gap_abs_p90": gap_abs_p90,
        "vaspo/advantage_normalization_enabled": float(normalize_advantages or whiten_before_fusion),
        "vaspo/whiten_before_fusion": float(whiten_before_fusion),
        "vaspo/teacher_signal_abs_mean": valid_mean(diagnostics["teacher_signal"].abs()),
        "vaspo/modulation_abs_mean": valid_mean(diagnostics["token_modulation_raw"].abs()),
        "vaspo/anchor_abs_mean": valid_mean(anchor.abs()),
        "vaspo/fused_raw_abs_mean": valid_mean(diagnostics["fused_raw"].abs()),
        "vaspo/raw_sign_flip_ratio": valid_mean((diagnostics["fused_raw"] * anchor) < 0),
        "vaspo/fusion_delta_abs_mean": valid_mean((diagnostics["fused_raw"] - anchor).abs()),
        "vaspo/actor_advantage_abs_mean": valid_mean(fused.abs()),
    }
    metrics.update(position_metrics)
    return batch, metrics

def _judge_rationales(self) -> dict:
    """Return per-step teacher-context metrics when available."""
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    self._rlcsd_step_metrics = {}
    return dict(metrics)

def _get_data_metric_overrides(self) -> dict:
    """Return per-step metric overrides that should win over post-filter stats."""
    metrics = getattr(self, "_data_metric_overrides", {})
    self._data_metric_overrides = {}
    return dict(metrics)

_ORIGINAL_RLHF_BUILD_MESSAGES = None

def _patched_rlhf_build_messages(self, example: dict):
    """Format stored raw problems into official OPSD train/val rollout prompts."""
    messages = _ORIGINAL_RLHF_BUILD_MESSAGES(self, example)

    data_source = str(example.get("data_source", "")).lower()
    if data_source not in MATH_DATA_SOURCES:
        return messages

    extra_info = example.get("extra_info") or {}
    problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
    if not problem:
        problem = _raw_prompt_to_problem(messages)

    # Read thinking_system_prompt from data config (self = RLHFDataset).
    use_thinking = False
    cfg = getattr(self, "config", None)
    if cfg is not None:
        val = cfg.get("thinking_system_prompt", False)
        use_thinking = str(val).strip().lower() in ("1", "true", "yes", "on") if isinstance(val, str) else bool(val)
    return build_rollout_messages(problem, data_source=data_source, thinking=use_thinking)

def _save_train_rollout(self, batch: DataProto):
    """Save train rollout details with teacher context aligned to the current batch order."""
    log_dir = self._get_rollout_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"train_step_{self.global_steps}.jsonl")

    batch_size = batch.batch.batch_size[0]
    prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    score_tensor = batch.batch["token_level_scores"]
    if score_tensor.dim() > 1:
        scores = score_tensor.sum(dim=-1).cpu().tolist()
    else:
        scores = score_tensor.cpu().tolist()

    ground_truths = []
    for i in range(batch_size):
        item = batch[i]
        reward_model = item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(ground_truth))

    aligned_context = getattr(self, "_aligned_teacher_rollout_details", None)
    privileged_texts = aligned_context.get("privileged_texts", []) if aligned_context is not None else []
    privilege_hashes = aligned_context.get("privilege_hashes", []) if aligned_context is not None else []
    privilege_token_lengths = (
        aligned_context.get("privilege_token_lengths", []) if aligned_context is not None else []
    )
    teacher_prompts = aligned_context.get("teacher_prompts", []) if aligned_context is not None else []
    effective_modes = aligned_context.get("effective_modes", []) if aligned_context is not None else []
    privileged_texts_correct = aligned_context.get("privileged_texts_correct", []) if aligned_context is not None else []
    privileged_texts_wrong = aligned_context.get("privileged_texts_wrong", []) if aligned_context is not None else []
    teacher_correct_prompts = aligned_context.get("teacher_correct_prompts", []) if aligned_context is not None else []
    teacher_wrong_prompts = aligned_context.get("teacher_wrong_prompts", []) if aligned_context is not None else []
    teacher_correct_prompt_groups = aligned_context.get("teacher_correct_prompt_groups", []) if aligned_context is not None else []
    teacher_wrong_prompt_groups = aligned_context.get("teacher_wrong_prompt_groups", []) if aligned_context is not None else []
    privileged_text_groups_correct = aligned_context.get("privileged_text_groups_correct", []) if aligned_context is not None else []
    privileged_text_groups_wrong = aligned_context.get("privileged_text_groups_wrong", []) if aligned_context is not None else []
    correct_answers = aligned_context.get("correct_answers", []) if aligned_context is not None else []
    wrong_answers = aligned_context.get("wrong_answers", []) if aligned_context is not None else []
    correct_answer_groups = aligned_context.get("correct_answer_groups", []) if aligned_context is not None else []
    wrong_answer_groups = aligned_context.get("wrong_answer_groups", []) if aligned_context is not None else []

    lines = []
    for i in range(batch_size):
        entry = {
            "step": self.global_steps,
            "sample_idx": i,
            "prompt": prompts[i],
            "response": responses[i],
            "score": scores[i],
            "ground_truth": ground_truths[i]}
        if i < len(privileged_texts):
            entry["privileged_text"] = privileged_texts[i]
        if i < len(privilege_hashes):
            entry["privilege_hash"] = privilege_hashes[i]
        if i < len(privilege_token_lengths):
            entry["privilege_token_length"] = privilege_token_lengths[i]
        if i < len(teacher_prompts):
            entry["teacher_prompt"] = teacher_prompts[i]
        if i < len(effective_modes):
            entry["privileged_text_mode"] = effective_modes[i]
        if i < len(privileged_texts_correct):
            entry["privileged_text_correct"] = privileged_texts_correct[i]
        if i < len(privileged_texts_wrong):
            entry["privileged_text_wrong"] = privileged_texts_wrong[i]
        if i < len(teacher_correct_prompts):
            entry["teacher_correct_prompt"] = teacher_correct_prompts[i]
        if i < len(teacher_wrong_prompts):
            entry["teacher_wrong_prompt"] = teacher_wrong_prompts[i]
        if i < len(teacher_correct_prompt_groups):
            entry["teacher_correct_prompt_pool"] = teacher_correct_prompt_groups[i]
        if i < len(teacher_wrong_prompt_groups):
            entry["teacher_wrong_prompt_pool"] = teacher_wrong_prompt_groups[i]
        if i < len(privileged_text_groups_correct):
            entry["privileged_text_correct_pool"] = privileged_text_groups_correct[i]
        if i < len(privileged_text_groups_wrong):
            entry["privileged_text_wrong_pool"] = privileged_text_groups_wrong[i]
        if i < len(correct_answers):
            entry["teacher_correct_answer"] = correct_answers[i]
        if i < len(wrong_answers):
            entry["teacher_wrong_answer"] = wrong_answers[i]
        if i < len(correct_answer_groups):
            entry["teacher_correct_answer_pool"] = correct_answer_groups[i]
        if i < len(wrong_answer_groups):
            entry["teacher_wrong_answer_pool"] = wrong_answer_groups[i]
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved train rollout details to {filename} ({batch_size} samples)")

# ------------------------------------------------------------------ #
# Apply monkey patches
# ------------------------------------------------------------------ #

def _apply_patches():
    global _ORIGINAL_RLHF_BUILD_MESSAGES
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if _ORIGINAL_RLHF_BUILD_MESSAGES is None:
        _ORIGINAL_RLHF_BUILD_MESSAGES = RLHFDataset._build_messages
    RLHFDataset._build_messages = _patched_rlhf_build_messages

    RayPPOTrainer._extract_problems_answers_solutions = _extract_problems_answers_solutions
    RayPPOTrainer._generate_rationales = _generate_rationales
    RayPPOTrainer._compute_teacher_log_probs = _compute_teacher_log_probs
    RayPPOTrainer._compute_vaspo_teacher_scores = _compute_vaspo_teacher_scores
    RayPPOTrainer._apply_vaspo_advantage_fusion = _apply_vaspo_advantage_fusion
    RayPPOTrainer._prepare_actor_update_batch = _prepare_actor_update_batch
    RayPPOTrainer._judge_rationales = _judge_rationales
    RayPPOTrainer._get_data_metric_overrides = _get_data_metric_overrides
    RayPPOTrainer._save_train_rollout = _save_train_rollout

def _validate_vaspo_config(config):
    """Fail early when a VASPO run would violate the paper's training interface."""
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    in_actor_update = policy_loss.get("vaspo_teacher_in_actor_update", False)
    if not isinstance(in_actor_update, bool):
        raise ValueError("vaspo_teacher_in_actor_update must be a boolean")
    whiten_before_fusion = policy_loss.get("vaspo_whiten_before_fusion", False)
    if not isinstance(whiten_before_fusion, bool):
        raise ValueError("vaspo_whiten_before_fusion must be a boolean")
    student_prob_source = str(policy_loss.get("vaspo_student_prob_source", "old")).strip().lower()
    if student_prob_source not in {"old", "current"}:
        raise ValueError("vaspo_student_prob_source must be 'old' or 'current'")
    in_actor_update = in_actor_update or student_prob_source == "current"
    if (in_actor_update and not whiten_before_fusion
            and config.algorithm.get("normalize_advantages", True) is not False):
        raise ValueError(
            "VASPO actor-loop fusion requires algorithm.normalize_advantages=False "
            "or vaspo_whiten_before_fusion=True"
        )
    if not isinstance(policy_loss.get("vaspo_use_token_mask", True), bool):
        raise ValueError("vaspo_use_token_mask must be a boolean")
    token_mask_threshold = float(policy_loss.get("vaspo_token_mask_threshold", 0.02))
    if not np.isfinite(token_mask_threshold) or token_mask_threshold < 0:
        raise ValueError("vaspo_token_mask_threshold must be finite and >= 0")
    tanh_tau = float(policy_loss.get("vaspo_tanh_tau", 0.02))
    epsilon = float(policy_loss.get("epsilon", 0.2))
    if not isinstance(policy_loss.get("vaspo_correct_only", True), bool):
        raise ValueError("vaspo_correct_only must be a boolean")
    if "vaspo_correct_reward_threshold" in policy_loss:
        raise ValueError(
            "vaspo_correct_reward_threshold was removed; vaspo_correct_only now selects "
            "tokens with pre-fusion PPO advantage > 0"
        )
    for obsolete_key in ("vaspo_scale_quantile", "vaspo_scale_eps"):
        if obsolete_key in policy_loss:
            raise ValueError(f"{obsolete_key} was removed; VASPO now uses fixed vaspo_tanh_tau")
    if not np.isfinite(tanh_tau) or tanh_tau <= 0:
        raise ValueError("vaspo_tanh_tau must be finite and > 0")
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("VASPO epsilon must be finite and in [0, 1)")
    epsilon_w = float(policy_loss.get("epsilon_w", 0.2))
    if not np.isfinite(epsilon_w) or not 0 <= epsilon_w < 1:
        raise ValueError("VASPO epsilon_w must be finite and in [0, 1)")
    removed = {"vaspo_fusion_mode", "vaspo_fusion_lambda", "vaspo_mask_space",
               "vaspo_gap_threshold", "vaspo_weight_clip_min", "vaspo_weight_clip_max",
               "vaspo_hybrid_kappa", "vaspo_eta", "vaspo_loss_normalization"}
    removed.update(key for key in policy_loss if key.startswith("port_"))
    supplied = removed.intersection(policy_loss)
    if supplied:
        raise ValueError(f"Removed non-paper VASPO options: {sorted(supplied)}")

    teacher_mode = str(policy_loss.get("teacher_mode", "fixed")).strip().lower()
    if teacher_mode not in {"fixed", "ema", "snapshot"}:
        raise ValueError(
            "VASPO teacher_mode must be one of fixed, ema, or snapshot, "
            f"got {teacher_mode!r}"
        )
    actor_strategy = str(config.actor_rollout_ref.actor.strategy).strip().lower()
    if actor_strategy not in {"fsdp", "fsdp2"}:
        raise ValueError(
            "VASPO's ordinary teacher forward and mask alignment require actor.strategy=fsdp or fsdp2; "
            f"got {actor_strategy!r}"
        )

    adv_estimator = str(config.algorithm.adv_estimator)
    if adv_estimator not in {"gae", "decoupled_gae"}:
        raise ValueError(
            "VASPO's privileged-critic anchor requires algorithm.adv_estimator=gae or decoupled_gae, "
            f"got {adv_estimator!r}"
        )
    if not bool(config.data.get("use_answer_for_critic", False)):
        raise ValueError(
            "VASPO requires data.use_answer_for_critic=true so GAE uses the privileged critic"
        )
    standard_answer_key = str(config.data.get("standard_answer_key", "reward_model.ground_truth"))
    if standard_answer_key != "reward_model.ground_truth":
        raise ValueError(
            "This VASPO implementation currently shares reward_model.ground_truth between teacher and critic; "
            "set data.standard_answer_key=reward_model.ground_truth"
        )
    if bool(config.algorithm.get("use_hem", False)):
        raise ValueError("VASPO must use a single privileged critic; set algorithm.use_hem=false")

    rollout_custom = config.actor_rollout_ref.rollout.get("custom", {}) or {}
    privilege_mode = normalize_privileged_text_mode(rollout_custom.get("privileged_text_mode", "solution_answer"))
    has_solution = bool(config.data.get("use_solution_for_critic", False))
    if has_solution != (privilege_mode == "solution_answer"):
        raise ValueError("VASPO teacher and critic must use the same solution/answer evidence")
    if has_solution and config.data.get("standard_solution_key", "extra_info.solution") != "extra_info.solution":
        raise ValueError("VASPO shares extra_info.solution between both readers")
    if bool(config.algorithm.get("use_dppo", False)):
        raise ValueError("VASPO uses PPO clipping; set algorithm.use_dppo=false")
    if privilege_mode not in {"oracle", "answer_only", "solution_answer"}:
        raise ValueError(
            "Unsupported VASPO privileged_text_mode. Set "
            "actor_rollout_ref.rollout.custom.privileged_text_mode to "
            "oracle, answer_only, or solution_answer."
        )

def _ensure_legacy_worker_impl(config):
    worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    if worker_impl == "disable":
        with open_dict(config):
            config.trainer.use_legacy_worker_impl = "enable"
        print(
            "[Self-Distill] Overriding trainer.use_legacy_worker_impl=disable -> enable. "
            "Privileged teacher prompting and scoring are currently implemented "
            "only in the legacy actor worker path."
        )
    return config
