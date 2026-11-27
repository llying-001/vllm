# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 naive DP dispatch for ROCm AITER MoE
(VLLM_ROCM_USE_AITER_MOE_FP4_DISPATCH).

The CPU tests mock the collective and the AITER quantization; the GPU test
checks on gfx950 that quantizing each rank's rows before the gather gives the
bytes of quantizing the gathered rows.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.platforms import current_platform

HIDDEN = 512
TOPK = 4
NUM_RANKS = 2


def _fake_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic stand-in for AITER's per-1x32 MXFP4 quantization."""
    m, k = x.shape
    q = (x.float().abs() * 7).to(torch.uint8)[:, : k // 2].contiguous()
    s = torch.arange(m * (k // 32), dtype=torch.int32).remainder(251)
    s = s.to(torch.uint8).view(m, k // 32)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e8m0fnu)


def _gather(hidden, weights, ids, extra_tensors=None, **kwargs):
    out = tuple(torch.cat([t] * NUM_RANKS) for t in (hidden, weights, ids))
    if extra_tensors is None:
        return out
    return (*out, [torch.cat([t] * NUM_RANKS) for t in extra_tensors])


def _run_prepare(num_tokens: int | None, enabled: bool = True, **prepare_kwargs):
    m = 3
    a1 = torch.randn(m, HIDDEN, dtype=torch.bfloat16)
    weights = torch.rand(m, TOPK)
    ids = torch.randint(0, 16, (m, TOPK), dtype=torch.int32)
    ep_group = MagicMock()
    ep_group.dispatch.side_effect = _gather
    localize = MagicMock(side_effect=lambda t: t)
    # The fields the non-FP4 path reads; quant_dtype=None leaves activations BF16.
    quant_config = SimpleNamespace(
        use_mxfp4_w4a4=prepare_kwargs.pop("use_mxfp4_w4a4", True),
        use_nvfp4_w4a4=False,
        a1_scale=None,
        a1_gscale=None,
        quant_dtype=None,
        per_act_token_quant=False,
        block_shape=None,
        mx_alignment=None,
    )
    pf = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular()
    if prepare_kwargs.pop("lora", False):
        mapping = torch.zeros(8, dtype=torch.int32)
        meta = SimpleNamespace(token_lora_mapping=mapping)
        pf.set_lora_context(
            SimpleNamespace(punica_wrapper=SimpleNamespace(token_mapping_meta=meta))
        )
    with ExitStack() as stack:
        for target, attr, value in (
            (naive_dp_ep, "get_ep_group", lambda: ep_group),
            (naive_dp_ep, "_num_dispatched_tokens", lambda: num_tokens),
            (naive_dp_ep, "_localize_shared_expert_ids", localize),
            (naive_dp_ep, "current_platform", SimpleNamespace(is_rocm=lambda: True)),
            (rocm_aiter_moe, "quantize_mxfp4_for_dispatch", _fake_quant),
        ):
            stack.enter_context(patch.object(target, attr, value))
        stack.enter_context(
            patch.object(
                rocm_aiter_moe.rocm_aiter_ops,
                "is_fused_moe_fp4_dispatch_enabled",
                return_value=enabled,
            )
        )
        result = pf.prepare(
            a1,
            weights,
            ids,
            num_experts=16,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
            defer_input_quant=prepare_kwargs.pop("defer_input_quant", True),
        )
    return a1, weights, ids, ep_group, localize, result


def test_mxfp4_dispatch_gathers_packed_values_and_scales():
    a1, weights, ids, ep_group, localize, result = _run_prepare(num_tokens=512)
    a1q, a1q_scale, _, out_ids, out_weights = result

    (sent, *_), kwargs = ep_group.dispatch.call_args
    assert sent.dtype == torch.bfloat16 and sent.shape == (3, HIDDEN // 4)
    (sent_scale,) = kwargs["extra_tensors"]
    assert sent_scale.dtype == torch.bfloat16
    assert sent_scale.shape == (3, HIDDEN // 64)

    q, s = _fake_quant(a1)
    assert a1q.dtype == torch.float4_e2m1fn_x2
    assert a1q.shape == (3 * NUM_RANKS, HIDDEN // 2)
    assert torch.equal(a1q.view(torch.uint8), torch.cat([q.view(torch.uint8)] * 2))
    assert a1q_scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(
        a1q_scale.view(torch.uint8), torch.cat([s.view(torch.uint8)] * 2)
    )
    torch.testing.assert_close(out_ids, torch.cat([ids, ids]))
    torch.testing.assert_close(out_weights, torch.cat([weights, weights]))
    localize.assert_called_once()


@pytest.mark.parametrize(
    "num_tokens, kwargs",
    [
        (128, {}),  # AITER's block_m 16 kernels read BF16
        (4096, {}),
        (None, {}),  # no DP metadata
        (512, {"enabled": False}),
        (512, {"defer_input_quant": False}),
        (512, {"use_mxfp4_w4a4": False}),
        (512, {"lora": True}),
    ],
)
def test_mxfp4_dispatch_falls_back_to_bf16(num_tokens, kwargs):
    a1, _, _, ep_group, localize, result = _run_prepare(num_tokens, **kwargs)
    (sent, *_), _ = ep_group.dispatch.call_args
    assert sent is a1
    assert result[0].dtype == torch.bfloat16
    assert result[1] is None
    localize.assert_called_once()


@pytest.mark.parametrize(
    "num_tokens, hidden, expected",
    [
        (128, 6144, False),
        (129, 6144, True),
        (2048, 6144, True),
        (2049, 6144, False),
        (512, 6176, False),  # K / 32 scale bytes not BF16-viewable
    ],
)
def test_use_mxfp4_dispatch_window(num_tokens, hidden, expected):
    with patch.object(
        rocm_aiter_moe.rocm_aiter_ops,
        "is_fused_moe_fp4_dispatch_enabled",
        return_value=True,
    ):
        assert rocm_aiter_moe.use_mxfp4_dispatch(num_tokens, hidden) is expected


def test_num_dispatched_tokens_without_forward_context():
    with patch.object(naive_dp_ep, "is_forward_context_available", return_value=False):
        assert naive_dp_ep._num_dispatched_tokens() is None
    context = SimpleNamespace(dp_metadata=SimpleNamespace(local_sizes=[100, 60, 3]))
    with (
        patch.object(naive_dp_ep, "is_forward_context_available", return_value=True),
        patch.object(naive_dp_ep, "get_forward_context", return_value=context),
    ):
        assert naive_dp_ep._num_dispatched_tokens() == 163


@pytest.mark.parametrize("enabled", [True, False])
def test_workspace_output_width_with_packed_input(enabled):
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(use_mori_epv1_kernels=False, hidden_dim=6144),
        quant_config=SimpleNamespace(use_mxfp4_w4a4=True),
    )

    def output(k: int) -> tuple[int, ...]:
        return rocm_aiter_moe.AiterExperts.workspace_shapes(
            experts, 256, 1024, k, 9, 257, 257, None, None
        )[2]

    with patch.object(
        rocm_aiter_moe.rocm_aiter_ops,
        "is_fused_moe_fp4_dispatch_enabled",
        return_value=enabled,
    ):
        assert output(6144) == (256, 6144)
        assert output(3072) == ((256, 6144) if enabled else (256, 3072))
        experts.quant_config.use_mxfp4_w4a4 = False
        assert output(3072) == (256, 3072)


def _on_gfx950_with_aiter() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import IS_AITER_FOUND
    from vllm.platforms.rocm import on_gfx950

    return IS_AITER_FOUND and on_gfx950()


@pytest.mark.skipif(not _on_gfx950_with_aiter(), reason="needs gfx950 and AITER")
@pytest.mark.parametrize("rows", [[1, 7, 64, 3], [64, 64, 64, 64], [300, 1, 5, 211]])
@pytest.mark.parametrize("hidden", [2048, 6144, 7168])
@torch.inference_mode()
def test_quantize_before_gather_matches_after(rows, hidden):
    torch.manual_seed(0)
    parts = [
        torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda") * 3 for m in rows
    ]
    gathered_q, gathered_s = rocm_aiter_moe.quantize_mxfp4_for_dispatch(
        torch.cat(parts)
    )
    assert gathered_q.dtype == torch.float4_e2m1fn_x2
    assert gathered_s.shape == (sum(rows), hidden // 32)

    per_rank = [rocm_aiter_moe.quantize_mxfp4_for_dispatch(p) for p in parts]
    # What the all-gather moves and the prepare views back.
    q = torch.cat([q.view(torch.bfloat16) for q, _ in per_rank])
    s = torch.cat([s.view(torch.bfloat16) for _, s in per_rank])
    assert torch.equal(
        q.view(gathered_q.dtype).view(torch.uint8), gathered_q.view(torch.uint8)
    )
    assert torch.equal(
        s.view(gathered_s.dtype).view(torch.uint8), gathered_s.view(torch.uint8)
    )
