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
"""
Reward function for integer-answer datasets.

Rule:
1. Extract the last \\boxed{...} in model output.
2. Parse integer from the boxed content.
3. Compare with integer ground truth.
4. Correct -> score=1.0, wrong/invalid -> score=0.0.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")


def _normalize_int(text: str | int | float | None) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if _INTEGER_PATTERN.fullmatch(s):
        return int(s)
    return None


def _extract_boxed_int(solution_str: str) -> tuple[Optional[int], str]:
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None, "[NO_BOXED]"

    try:
        raw = remove_boxed(boxed).strip()
    except Exception:
        return None, "[INVALID_BOXED]"

    parsed = _normalize_int(raw)
    if parsed is None:
        return None, "[NON_INTEGER]"
    return parsed, str(parsed)


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    del data_source, extra_info, kwargs

    pred_int, pred = _extract_boxed_int(solution_str)
    gt_int = _normalize_int(ground_truth)
    correct = pred_int is not None and gt_int is not None and pred_int == gt_int

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "pred": pred,
    }

