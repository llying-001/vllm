# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton DSA indexer decode top-k on ROCm
(VLLM_ROCM_USE_TRITON_INDEXER_DECODE_TOPK).

The guard tests run on CPU; the kernel tests need gfx950.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_triton_indexer_topk as tk

K = 2048


def _guard_inputs(rows=4, width=4096, topk=K):
    logits = torch.empty((rows, width), dtype=torch.float32)
    lengths = torch.full((rows,), width, dtype=torch.int32)
    out = torch.empty((rows, 2 * topk), dtype=torch.int32)[:, :topk]
    return logits, lengths, out, topk


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_TRITON_INDEXER_DECODE_TOPK", "1")
    with patch.object(tk, "_platform_supported", return_value=True):
        yield


def test_guard_accepts_decode_shapes(enabled):
    assert tk.use_triton_decode_top_k(*_guard_inputs())
    assert tk.use_triton_decode_top_k(*_guard_inputs(rows=128, topk=512))


def test_guard_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_TRITON_INDEXER_DECODE_TOPK", raising=False)
    with patch.object(tk, "_platform_supported", return_value=True):
        assert not tk.use_triton_decode_top_k(*_guard_inputs())


def test_guard_needs_platform(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_TRITON_INDEXER_DECODE_TOPK", "1")
    with patch.object(tk, "_platform_supported", return_value=False):
        assert not tk.use_triton_decode_top_k(*_guard_inputs())


@pytest.mark.parametrize(
    "case",
    [
        "too_many_rows",
        "fp16_logits",
        "strided_columns",
        "width_not_4",
        "topk_not_pow2",
        "topk_too_large",
        "int64_lengths",
        "lengths_per_request",
        "out_strided",
        "int64_out",
    ],
)
def test_guard_falls_back(enabled, case):
    rows = 129 if case == "too_many_rows" else 4
    width = 4098 if case == "width_not_4" else 4096
    topk = {"topk_not_pow2": 1000, "topk_too_large": 4096}.get(case, K)
    logits, lengths, out, topk = _guard_inputs(rows, width, topk)
    if case == "fp16_logits":
        logits = logits.half()
    elif case == "strided_columns":
        logits = torch.empty((rows, 2 * width))[:, ::2]
    elif case == "int64_lengths":
        lengths = lengths.long()
    elif case == "lengths_per_request":
        lengths = lengths[:2]
    elif case == "out_strided":
        out = torch.empty((rows, 2 * topk), dtype=torch.int32)[:, ::2]
    elif case == "int64_out":
        out = out.long()
    assert not tk.use_triton_decode_top_k(logits, lengths, out, topk)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(not _on_gfx950(), reason="needs gfx950")


def _rows(lengths: list[int], width: int, ties: dict[int, float]) -> torch.Tensor:
    """fp32 [rows, width] logits, -inf past each row's length. Rows in `ties`
    are 0 except the given fraction of positive values, the shape of indexer
    logits where every head's ReLU is 0."""
    g = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.full((len(lengths), width), float("-inf"), device="cuda")
    for r, n in enumerate(lengths):
        x = torch.randn(n, device="cuda", generator=g)
        if r in ties:
            hit = torch.rand(n, device="cuda", generator=g) < ties[r]
            x = torch.where(hit, x.abs() + 0.1, torch.zeros_like(x))
        logits[r, :n] = x
    return logits


def _check_exact(logits, lengths, out, topk):
    for r, n in enumerate(lengths.tolist()):
        kk = min(n, topk)
        sel, pad = out[r, :kk], out[r, kk:]
        assert (pad == -1).all(), r
        assert (sel >= 0).all() and (sel < n).all(), r
        assert bool((sel[1:] > sel[:-1]).all()), f"row {r} not ascending"
        want = torch.topk(logits[r, :n], kk).values.sort().values
        got = logits[r, sel.long()].sort().values
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_gfx950
@torch.inference_mode()
def test_exact_on_long_and_tied_rows():
    lengths = [1000, 1001, 2048, 2049, 150_000, 150_003, 700_000, 700_003]
    lengths += [700_000, 700_001, 5_000, 300_000, 1, 4096, 33_333, 64]
    # Mostly-zero rows: fewer positives than k on the long ones, all zero, many.
    ties = {8: 0.0015, 9: 0.0015, 10: 0.0, 11: 0.02}
    width = 1 << 20
    logits = _rows(lengths, width, ties)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    # The decode top-k buffer is wider than k.
    out = torch.full((len(lengths), 2 * K), -7, dtype=torch.int32, device="cuda")
    out = out[:, :K]
    tk.triton_decode_top_k(logits, lens, out, K)
    _check_exact(logits, lens, out, K)


@requires_gfx950
@pytest.mark.parametrize("rows", [1, 5, 128])
@pytest.mark.parametrize("topk", [512, 2048])
@torch.inference_mode()
def test_exact_random(rows, topk):
    torch.manual_seed(rows)
    width = 262_144
    lens = torch.randint(1, width + 1, (rows,), dtype=torch.int32, device="cuda")
    logits = torch.randn(rows, width, device="cuda")
    out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    tk.triton_decode_top_k(logits, lens, out, topk)
    _check_exact(logits, lens, out, topk)


@requires_gfx950
@torch.inference_mode()
def test_cuda_graph_replay():
    rows, width = 8, 131_072
    logits = torch.randn(rows, width, device="cuda")
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, K), dtype=torch.int32, device="cuda")
    tk.triton_decode_top_k(logits, lens, out, K)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tk.triton_decode_top_k(logits, lens, out, K)
    for seed in range(3):
        torch.manual_seed(seed)
        logits.copy_(torch.randn_like(logits))
        lens.copy_(torch.randint(1, width + 1, (rows,), device="cuda"))
        out.fill_(-7)
        graph.replay()
        _check_exact(logits, lens, out, K)
