# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from verl.utils.import_utils import deprecated


_SIMPLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DEGREE_SUFFIX_RE = re.compile(r"(?:\^\{?\\circ\}?|\\degree|\\deg|\u00b0)$")
_KK_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z\-']*)\s*(?:=|:|is(?:\s+(?:a|an))?)\s*(knight|knave)",
    re.IGNORECASE,
)


def _extract_rlcsd_boxed_answer(text: str | None) -> Optional[str]:
    """Extract the final ``\\boxed{...}`` using RLCSD's training logic."""
    if text is None:
        return None

    text = str(text)
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None

    num_left_braces = 0
    right_brace_idx = None
    for position in range(idx, len(text)):
        if text[position] == "{":
            num_left_braces += 1
        if text[position] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = position
                break

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith(r"\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def _normalize_rlcsd_fallback_string(text: str) -> str:
    return str(text).replace("$", "").replace(" ", "").lower().strip()


def _strip_rlcsd_answer_wrappers(text: str) -> str:
    text = str(text).strip()
    boxed = _extract_rlcsd_boxed_answer(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$").strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def _canonical_rlcsd_simple_number(text: str) -> Optional[Decimal]:
    text = _strip_rlcsd_answer_wrappers(text)
    text = _DEGREE_SUFFIX_RE.sub("", text.replace(" ", "").replace(",", "").replace("\u2212", "-"))
    if not _SIMPLE_NUMBER_RE.fullmatch(text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _rlcsd_simple_numeric_answers_match(predicted: str, ground_truth: str) -> Optional[bool]:
    pred_number = _canonical_rlcsd_simple_number(predicted)
    gt_number = _canonical_rlcsd_simple_number(ground_truth)
    if pred_number is None or gt_number is None:
        return None
    return pred_number == gt_number


def _grade_rlcsd_deepmath_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Grade a DeepMath answer with RLCSD's numeric/math_verify/fallback path."""
    if predicted is None:
        return False

    ground_truth = str(ground_truth)
    numeric_match = _rlcsd_simple_numeric_answers_match(predicted, ground_truth)
    if numeric_match is not None:
        return numeric_match

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ModuleNotFoundError(
            "math_verify is required for RLCSD-aligned DeepMath grading. "
            "Install it with `pip install math-verify`."
        ) from exc

    try:
        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        pred_parsed = parse(pred_text, fallback_mode="no_fallback")
        gt_parsed = parse(gt_text, fallback_mode="no_fallback")
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        return _normalize_rlcsd_fallback_string(predicted) == _normalize_rlcsd_fallback_string(ground_truth)


def _parse_rlcsd_kk_assignment(text: str | None) -> dict[str, str]:
    """Parse an RLCSD Knights & Knaves assignment into an order-free mapping."""
    if text is None:
        return {}

    text = str(text).strip()
    boxed = _extract_rlcsd_boxed_answer(text)
    if boxed is not None:
        text = boxed

    assignment = {}
    for name, role in _KK_PAIR_RE.findall(text):
        key = name.lower()
        if key not in assignment:
            assignment[key] = role.lower()
    return assignment


def _grade_rlcsd_kk_answer(predicted: Optional[str], ground_truth: str) -> bool:
    ground_truth_assignment = _parse_rlcsd_kk_assignment(ground_truth)
    if not ground_truth_assignment:
        return False
    return _parse_rlcsd_kk_assignment(predicted) == ground_truth_assignment


def _compute_rlcsd_deepmath_score(solution_str: str, ground_truth: str) -> dict[str, float]:
    predicted = _extract_rlcsd_boxed_answer(solution_str)
    formatted = 1.0 if predicted is not None else 0.0
    correct = _grade_rlcsd_deepmath_answer(predicted, ground_truth) if predicted is not None else False
    score = 1.0 if correct else 0.0
    return {"score": score, "acc": score, "formatted": formatted}


def _compute_rlcsd_kk_score(solution_str: str, ground_truth: str) -> dict[str, float]:
    predicted = _extract_rlcsd_boxed_answer(solution_str)
    formatted = 1.0 if predicted is not None else 0.0
    correct = _grade_rlcsd_kk_answer(predicted, ground_truth) if predicted is not None else False
    score = 1.0 if correct else 0.0
    return {"score": score, "acc": score, "formatted": formatted}


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    sandbox_timeout=10,
    sandbox_continuous=False,
    **kwargs,
):
    """Route KK/DeepMath to RLCSD rewards and preserve the existing fallback.

    Routing is case-insensitive and based on substring containment:

    - sources containing ``kk`` use RLCSD's Knights & Knaves assignment reward;
    - the exact source ``dapo_deepmath`` uses the existing boxed-integer reward;
    - sources containing ``deepmath`` use RLCSD's boxed math reward;
    - all other values, including an empty/``None`` source, use the previous
      code-or-boxed-integer behavior unchanged.

    Code routing is intentionally exact: the field selected by ``data.reward_fn_key`` must be
    the string ``"code"`` (case-insensitive). For Eurus-2-RL-Data, set
    ``data.reward_fn_key=ability`` because ``data_source`` contains provider names such as
    ``taco`` and ``apps`` while ``ability`` contains ``code``.
    """
    normalized_source = data_source.decode() if isinstance(data_source, bytes) else str(data_source)
    normalized_source = normalized_source.strip().lower()

    if "kk" in normalized_source:
        return _compute_rlcsd_kk_score(solution_str, ground_truth)

    if normalized_source == "dapo_deepmath":
        from . import boxed_integer

        return boxed_integer.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            **kwargs,
        )

    if "deepmath" in normalized_source:
        return _compute_rlcsd_deepmath_score(solution_str, ground_truth)

    is_code = normalized_source == "code"

    if is_code:
        if not sandbox_fusion_url:
            raise RuntimeError(
                "A code sample requires reward_model.sandbox_fusion.url. "
                "Set it to the Sandbox Fusion /run_code endpoint."
            )

        from . import sandbox_fusion

        score, _metadata = sandbox_fusion.compute_score(
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb or 1024,
            completion=solution_str,
            test_cases=ground_truth,
            continuous=bool(sandbox_continuous),
            timeout=int(sandbox_timeout),
        )
        return float(score)

    from . import boxed_integer

    del extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, sandbox_timeout, sandbox_continuous
    return boxed_integer.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        **kwargs,
    )


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    sandbox_timeout=10,
    sandbox_continuous=False,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
        sandbox_timeout,
        sandbox_continuous,
    )


__all__ = ["default_compute_score"]
