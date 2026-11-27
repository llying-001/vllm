# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract test of the packed-MXFP4 all-gather dispatch
(`VLLM_ROCM_AITER_MOE_FP4_DISPATCH`); AITER and the collective are faked."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)

HIDDEN = 128
NUM_RANKS = 2


class _FakeEpGroup:
    """All-gathers by repeating this rank's rows for every rank."""

    def __init__(self) -> None:
        self.hidden: torch.Tensor | None = None
        self.extras: list[torch.Tensor] | None = None

    def dispatch(
        self, a1q, topk_weights, topk_ids, is_sequence_parallel, extra_tensors
    ):
        self.hidden, self.extras = a1q, extra_tensors
        gathered = [t.repeat(NUM_RANKS, 1) for t in (a1q, topk_weights, topk_ids)]
        if extra_tensors is None:
            return tuple(gathered)
        return (*gathered, [t.repeat(NUM_RANKS, 1) for t in extra_tensors])


def _fake_per_1x32_quant(a1: torch.Tensor, shuffle: bool):
    assert not shuffle
    num_tokens = a1.shape[0]
    values = torch.arange(num_tokens * HIDDEN // 2, dtype=torch.uint8)
    # AITER pads the scale rows; prepare() keeps the first num_tokens.
    scales = torch.arange((num_tokens + 3) * HIDDEN // 32, dtype=torch.uint8)
    return (
        values.view(num_tokens, HIDDEN // 2).view(torch.float4_e2m1fn_x2),
        scales.view(num_tokens + 3, HIDDEN // 32).view(torch.float8_e8m0fnu),
    )


def _prepare(monkeypatch, *, enabled: bool, tokens_per_rank: int):
    fake_aiter = types.ModuleType("aiter")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")  # type: ignore[attr-defined]
    fake_aiter.get_hip_quant = lambda _: _fake_per_1x32_quant  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(naive_dp_ep.envs, "VLLM_ROCM_AITER_MOE_FP4_DISPATCH", enabled)

    group = _FakeEpGroup()
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(local_sizes=[tokens_per_rank] * NUM_RANKS)
    )
    hidden_states = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
    with (
        patch.object(naive_dp_ep, "get_ep_group", return_value=group),
        patch.object(naive_dp_ep, "get_forward_context", return_value=forward_context),
        patch.object(
            naive_dp_ep, "current_platform", SimpleNamespace(is_rocm=lambda: True)
        ),
        patch.object(naive_dp_ep, "dbo_enabled", return_value=False),
    ):
        result = MoEPrepareAndFinalizeNaiveDPEPModular().prepare(
            hidden_states,
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32),
            num_experts=4,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=SimpleNamespace(use_mxfp4_w4a4=True),
            defer_input_quant=True,
        )
    return group, hidden_states, result


def test_fp4_dispatch_gathers_packed_activations_and_scales(monkeypatch):
    group, hidden_states, (a1q, a1q_scale, _, topk_ids, _) = _prepare(
        monkeypatch, enabled=True, tokens_per_rank=100
    )

    # A quarter of the BF16 hidden states plus the scales cross the collective,
    # both as BF16 so it takes its usual path.
    assert group.hidden is not None and group.extras is not None
    assert group.hidden.dtype == torch.bfloat16
    assert group.hidden.shape == (3, HIDDEN // 4)
    assert [(t.dtype, t.shape) for t in group.extras] == [
        (torch.bfloat16, (3, HIDDEN // 64))
    ]

    values, scales = _fake_per_1x32_quant(hidden_states, shuffle=False)
    assert a1q.dtype == torch.float4_e2m1fn_x2
    assert torch.equal(
        a1q.view(torch.uint8), values.view(torch.uint8).repeat(NUM_RANKS, 1)
    )
    assert a1q_scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(
        a1q_scale.view(torch.uint8), scales[:3].view(torch.uint8).repeat(NUM_RANKS, 1)
    )
    assert topk_ids.shape == (3 * NUM_RANKS, 2)


@pytest.mark.parametrize(
    ("enabled", "tokens_per_rank"),
    [
        (False, 100),
        # AITER's kernels for up to 128 tokens read BF16 activations.
        (True, 64),
        (True, 2000),
    ],
)
def test_fp4_dispatch_keeps_bf16_outside_its_range(
    monkeypatch, enabled, tokens_per_rank
):
    group, hidden_states, (a1q, a1q_scale, *_) = _prepare(
        monkeypatch, enabled=enabled, tokens_per_rank=tokens_per_rank
    )

    assert group.hidden is hidden_states
    assert group.extras is None
    assert a1q.dtype == torch.bfloat16
    assert a1q.shape == (3 * NUM_RANKS, HIDDEN)
    assert a1q_scale is None
