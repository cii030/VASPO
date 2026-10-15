# Copyright 2026 THU-BPM
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load VASPO fusion helpers from core_algos without importing ray or tensordict.

``verl.trainer.ppo.core_algos`` is only reachable through ``verl/__init__.py``,
which imports ray. These functions are pure tensor math, so the tests can lift
their real source out of the file and execute it against the few module globals
they actually touch. The source is never copied, so the tests still fail if the
implementation changes.
"""

import ast
import functools
import pathlib
import types
import typing

import numpy as np
import torch

_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "verl" / "trainer" / "ppo" / "core_algos.py"
)
_WANTED = ("compute_vaspo_fused_advantage", "compute_policy_loss_vaspo", "_sd_cfg")


class _MaskedTorchFunctional:
    """The masked_* helpers core_algos reaches for via ``verl_F``."""

    @staticmethod
    def masked_mean(tensor, mask, axis=None):
        mask = mask.to(tensor.dtype)
        if axis is None:
            return (tensor * mask).sum() / mask.sum().clamp_min(1)
        return (tensor * mask).sum(axis) / mask.sum(axis).clamp_min(1)

    @staticmethod
    def masked_var(tensor, mask, unbiased=True):
        mean = _MaskedTorchFunctional.masked_mean(tensor, mask)
        mask = mask.to(tensor.dtype)
        total = mask.sum()
        variance = (((tensor - mean) ** 2) * mask).sum() / total.clamp_min(1)
        if unbiased and total > 1:
            variance = variance * total / (total - 1)
        return variance

    @staticmethod
    def masked_whiten(tensor, mask, shift_mean=True):
        mean = _MaskedTorchFunctional.masked_mean(tensor, mask)
        var = _MaskedTorchFunctional.masked_var(tensor, mask)
        whitened = (tensor - mean) * torch.rsqrt(var + 1e-8)
        return whitened if shift_mean else whitened + mean


@functools.lru_cache(maxsize=1)
def _load():
    tree = ast.parse(_SOURCE.read_text())
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED
    }
    missing = set(_WANTED) - set(wanted)
    if missing:
        raise RuntimeError(f"core_algos no longer defines {sorted(missing)}")

    module = types.ModuleType("_vaspo_fusion_shim_ns")
    module.__dict__.update(
        torch=torch,
        np=np,
        verl_F=_MaskedTorchFunctional,
        Optional=typing.Optional,
        Any=typing.Any,
        ActorConfig="ActorConfig",
        register_policy_loss=lambda *_, **__: (lambda fn: fn),
    )
    ordered = [wanted[name] for name in ("_sd_cfg", *(n for n in _WANTED if n != "_sd_cfg"))]
    for node in ordered:
        # Decorators reference registries that live behind the ray import.
        node.decorator_list = []
    exec(compile(ast.Module(body=ordered, type_ignores=[]), str(_SOURCE), "exec"), module.__dict__)
    return module


def __getattr__(name):
    if name in _WANTED:
        return getattr(_load(), name)
    raise AttributeError(name)
