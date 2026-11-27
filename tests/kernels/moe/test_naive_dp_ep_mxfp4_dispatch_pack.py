# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed MXFP4 naive DP dispatch (VLLM_ROCM_USE_AITER_MOE_FP4_DISPATCH_PACK):
one all-gather of values, scales, topk ids and weights per MoE layer, with
the same expert inputs as the four separate gathers.

The CPU tests run the packed prepare() on a torch reference of the pack
kernels; the GPU tests check the Triton kernels against that reference.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    mxfp4_dispatch_pack as pack,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.platforms import current_platform

HIDDEN = 512
NUM_RANKS = 2


def _fields(values, scales, topk_ids):
    m = topk_ids.shape[0]
    return values.view(torch.uint8), scales.view(torch.uint8)[:m]


def _pack_reference(values, scales, topk_weights, topk_ids, layout):
    q, s = _fields(values, scales, topk_ids)
    m, t = topk_ids.shape
    out = torch.zeros((m, layout.row_bytes), dtype=torch.uint8, device=q.device)
    out[:, : layout.value_bytes] = q
    so = layout.scale_offset
    out[:, so : so + layout.scale_bytes] = s
    io, wo = layout.ids_offset, layout.weights_offset
    out[:, io : io + 4 * t] = topk_ids.contiguous().view(torch.uint8)
    out[:, wo : wo + 4 * t] = topk_weights.contiguous().view(torch.uint8)
    return out


def _unpack_reference(packed, layout, values_dtype, scales_dtype):
    u8 = packed.view(torch.uint8)
    t, so = layout.topk, layout.scale_offset
    io, wo = layout.ids_offset, layout.weights_offset
    q = u8[:, : layout.value_bytes].contiguous().view(values_dtype)
    s = u8[:, so : so + layout.scale_bytes].contiguous().view(scales_dtype)
    ids = u8[:, io : io + 4 * t].contiguous().view(torch.int32)
    w = u8[:, wo : wo + 4 * t].contiguous().view(torch.float32)
    return q, s, w, ids


def _fake_quant(x):
    m, k = x.shape
    q = (x.float().abs() * 7).to(torch.uint8)[:, : k // 2].contiguous()
    s = torch.arange(m * (k // 32), dtype=torch.int32).remainder(251)
    s = s.to(torch.uint8).view(m, k // 32)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e8m0fnu)


def _gather(hidden, weights, ids, extra_tensors=None, **kwargs):
    out = tuple(torch.cat([t] * NUM_RANKS) for t in (hidden, weights, ids))
    return (*out, [torch.cat([t] * NUM_RANKS) for t in extra_tensors])


@pytest.mark.parametrize(
    "hidden, topk, offsets",
    [
        # GLM-5 / DeepSeek-V3: 8 routed experts, + 1 fused shared expert.
        (6144, 8, (3072, 3264, 3296, 3328)),
        (6144, 9, (3072, 3264, 3300, 3344)),
        (7168, 9, (3584, 3808, 3844, 3888)),
    ],
)
def test_layout(hidden, topk, offsets):
    values = torch.empty((5, hidden // 2), dtype=torch.uint8)
    scales = torch.empty((8, hidden // 32), dtype=torch.uint8)
    weights = torch.empty((5, topk), dtype=torch.float32)
    ids = torch.empty((5, topk), dtype=torch.int32)
    layout = pack.mxfp4_dispatch_layout(values, scales, weights, ids)
    assert layout is not None
    got = (
        layout.scale_offset,
        layout.ids_offset,
        layout.weights_offset,
        layout.row_bytes,
    )
    assert got == offsets
    assert layout.row_bytes % 16 == 0
    assert layout.value_bytes == hidden // 2 and layout.scale_bytes == hidden // 32


@pytest.mark.parametrize(
    "change",
    ["ids_int64", "weights_bf16", "hidden_not_256", "strided_ids", "few_scales"],
)
def test_layout_rejects(change):
    hidden = 6144 + (32 if change == "hidden_not_256" else 0)
    values = torch.empty((5, hidden // 2), dtype=torch.uint8)
    scales = torch.empty((4 if change == "few_scales" else 5, hidden // 32))
    scales = scales.to(torch.uint8)
    weights = torch.empty(
        (5, 8), dtype=torch.bfloat16 if change == "weights_bf16" else torch.float32
    )
    ids = torch.empty(
        (5, 16 if change == "strided_ids" else 8),
        dtype=torch.int64 if change == "ids_int64" else torch.int32,
    )
    if change == "strided_ids":
        ids = ids[:, ::2]
    assert pack.mxfp4_dispatch_layout(values, scales, weights, ids) is None


def _prepare(packed: bool, manager_ok: bool = True):
    torch.manual_seed(0)
    m, topk = 3, 9
    a1 = torch.randn(m, HIDDEN, dtype=torch.bfloat16)
    weights = torch.rand(m, topk)
    ids = torch.randint(0, 16, (m, topk), dtype=torch.int32)
    ep_group = MagicMock()
    ep_group.dispatch.side_effect = _gather
    manager = MagicMock()
    manager.dispatch_packed.side_effect = lambda t, sp: torch.cat([t] * NUM_RANKS)
    localize = MagicMock(side_effect=lambda t: t)
    pf = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular()
    ops = rocm_aiter_moe.rocm_aiter_ops
    with ExitStack() as stack:
        for target, attr, value in (
            (naive_dp_ep, "get_ep_group", lambda: ep_group),
            (naive_dp_ep, "_num_dispatched_tokens", lambda: 512),
            (naive_dp_ep, "_localize_shared_expert_ids", localize),
            (naive_dp_ep, "current_platform", SimpleNamespace(is_rocm=lambda: True)),
            (
                naive_dp_ep,
                "_packed_dispatch_manager",
                lambda: manager if manager_ok else None,
            ),
            (rocm_aiter_moe, "quantize_mxfp4_for_dispatch", _fake_quant),
            (pack, "pack_mxfp4_dispatch", _pack_reference),
            (pack, "unpack_mxfp4_dispatch", _unpack_reference),
        ):
            stack.enter_context(patch.object(target, attr, value))
        for attr, value in (
            ("is_fused_moe_fp4_dispatch_enabled", True),
            ("is_fused_moe_fp4_dispatch_pack_enabled", packed),
        ):
            stack.enter_context(patch.object(ops, attr, return_value=value))
        result = pf.prepare(
            a1,
            weights,
            ids,
            num_experts=16,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=SimpleNamespace(use_mxfp4_w4a4=True),
            defer_input_quant=True,
        )
    return result, ep_group, manager, localize


def test_packed_prepare_matches_unpacked():
    packed, ep_group, manager, localize = _prepare(packed=True)
    ep_group.dispatch.assert_not_called()
    manager.dispatch_packed.assert_called_once()
    (sent, _), _ = manager.dispatch_packed.call_args
    assert sent.dtype == torch.bfloat16
    # The shared-expert ids are still localized on the gathered rows.
    localize.assert_called_once()

    unpacked, ep_group, manager, _ = _prepare(packed=False)
    ep_group.dispatch.assert_called_once()
    manager.dispatch_packed.assert_not_called()

    assert packed[2] is None and unpacked[2] is None
    for i in (0, 1, 3, 4):
        assert packed[i].dtype == unpacked[i].dtype
        assert packed[i].shape == unpacked[i].shape
        assert torch.equal(packed[i].view(torch.uint8), unpacked[i].view(torch.uint8))


def test_packed_prepare_needs_agrs_manager():
    _, ep_group, manager, _ = _prepare(packed=True, manager_ok=False)
    ep_group.dispatch.assert_called_once()
    manager.dispatch_packed.assert_not_called()


def _random_inputs(m, hidden, topk, scale_rows):
    q = torch.randint(0, 256, (m, hidden // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(
        0, 256, (scale_rows, hidden // 32), dtype=torch.uint8, device="cuda"
    )
    w = torch.rand(m, topk, device="cuda")
    ids = torch.randint(0, 257, (m, topk), dtype=torch.int32, device="cuda")
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e8m0fnu), w, ids


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("m", [1, 3, 4, 7, 64, 129, 256])
@pytest.mark.parametrize("hidden, topk", [(6144, 8), (6144, 9), (7168, 9)])
@torch.inference_mode()
def test_pack_kernels_match_reference(m, hidden, topk):
    torch.manual_seed(m)
    # AITER's scales may have padded rows.
    q, s, w, ids = _random_inputs(m, hidden, topk, scale_rows=m + 5)
    layout = pack.mxfp4_dispatch_layout(q, s, w, ids)
    assert layout is not None
    got = pack.pack_mxfp4_dispatch(q, s, w, ids, layout)
    want = _pack_reference(q, s, w, ids, layout)
    used = layout.weights_offset + 4 * topk
    # The row padding is left unwritten.
    assert torch.equal(got[:, :used], want[:, :used])

    # Four ranks with different row counts, gathered.
    ranks = [_random_inputs(r, hidden, topk, scale_rows=r) for r in (m, 1, 5, m)]
    gathered = torch.cat([pack.pack_mxfp4_dispatch(*r, layout) for r in ranks]).view(
        torch.bfloat16
    )
    out = pack.unpack_mxfp4_dispatch(gathered, layout, q.dtype, s.dtype)
    expected = (
        torch.cat([r[0].view(torch.uint8) for r in ranks]),
        torch.cat([r[1].view(torch.uint8) for r in ranks]),
        torch.cat([r[2] for r in ranks]),
        torch.cat([r[3] for r in ranks]),
    )
    assert out[0].dtype == q.dtype and out[1].dtype == s.dtype
    assert out[0].is_contiguous() and out[3].is_contiguous()
    assert torch.equal(out[0].view(torch.uint8), expected[0])
    assert torch.equal(out[1].view(torch.uint8), expected[1])
    assert torch.equal(out[2].view(torch.int32), expected[2].view(torch.int32))
    assert torch.equal(out[3], expected[3])
