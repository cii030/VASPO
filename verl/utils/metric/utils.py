# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Metrics utils.
"""

import re
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch

# ------------------------------------------------------------------ #
# RLCSD-specific metric reduction helpers
# ------------------------------------------------------------------ #
# Keys whose NaN values (from steps with no valid contrastive samples)
# should be silently dropped rather than propagated through np.mean/min/max.
_IGNORE_NAN_KEY = re.compile(
    r"^rlcsd_(?:w_tilde|lambda_t|snr|e_ctr|selected_residual)_(?:p\d+|mean|min|max)$"
    r"|^rlcsd_selected_tokens_per_sample_mean$"
)
# Count keys that should be summed (not averaged) across micro-batches.
_SUM_KEY = re.compile(
    r"^rlcsd_(?:selected_token_count|response_token_count|sample_count)$"
)
_RLCSD_DERIVABLE_PREFIXES = ("rlcsd",)


def _finite_values(values: list[Any]) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    return value_array[np.isfinite(value_array)]


def _reduce_mean_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.mean(finite_values))
    return np.mean(values)


def _reduce_min_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.min(finite_values))
    return np.min(values)


def _reduce_max_metric(key: str, values: list[Any]) -> float:
    if _IGNORE_NAN_KEY.fullmatch(key):
        finite_values = _finite_values(values)
        if finite_values.size == 0:
            return float("nan")
        return float(np.max(finite_values))
    return np.max(values)


def _safe_div(numerator: Any, denominator: Any) -> float:
    denominator = float(denominator)
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def _derive_rlcsd_count_metrics(metrics: dict[str, Any]) -> None:
    for prefix in _RLCSD_DERIVABLE_PREFIXES:
        selected_token_count = metrics.get(f"{prefix}_selected_token_count")
        response_token_count = metrics.get(f"{prefix}_response_token_count")
        sample_count = metrics.pop(f"{prefix}_sample_count", None)
        if selected_token_count is not None and response_token_count is not None:
            metrics[f"{prefix}_selected_token_ratio"] = _safe_div(selected_token_count, response_token_count)
        if selected_token_count is not None and sample_count is not None:
            metrics[f"{prefix}_selected_tokens_per_sample_mean"] = _safe_div(selected_token_count, sample_count)


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif _SUM_KEY.fullmatch(key):
            finite = _finite_values(val)
            metrics[key] = float(np.sum(finite)) if finite.size > 0 else float("nan")
        elif "max" in key:
            metrics[key] = _reduce_max_metric(key, val)
        elif "min" in key:
            metrics[key] = _reduce_min_metric(key, val)
        else:
            metrics[key] = _reduce_mean_metric(key, val)
    _derive_rlcsd_count_metrics(metrics)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = int, float, torch.Tensor
Numeric = int | float | torch.Tensor


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(self, aggregation: str | AggregationType, value: Optional[Numeric | list[Numeric]] = None) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        if value is not None:
            self.append(value)

    def append(self, value: Union[Numeric, "Metric"]) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        match self.aggregation:
            case AggregationType.MEAN:
                return np.mean(self.values)
            case AggregationType.SUM:
                return np.sum(self.values)
            case AggregationType.MIN:
                return np.min(self.values)
            case AggregationType.MAX:
                return np.max(self.values)

    @classmethod
    def chain(cls, metric_lists: list["Metric"]) -> "Metric":
        if len(metric_lists) == 0:
            return cls(aggregation=AggregationType.MEAN)
        aggregation = metric_lists[0].aggregation
        chained = cls(aggregation=aggregation)
        for ml in metric_lists:
            chained.extend(ml)
        return chained

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "Metric"]:
        return {key: cls(value=value, aggregation=aggregation) for key, value in data.items()}

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
