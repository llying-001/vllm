# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single-pass Triton decode top-k of the DSA indexer on gfx950 returns
an exact top-k for rows shaped like a decode step's."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-specific tests", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950

if not on_gfx950():
    pytest.skip("gfx950-specific kernel", allow_module_level=True)

from vllm.v1.attention.ops import rocm_triton_topk_decode

TOPK = 2048
MAX_MODEL_LEN = 1048576
# Shorter than k, long, and very long rows, as MTP verify rows of one request.
LENGTHS = [
    *range(1000, 1004),
    *range(150_000, 150_004),
    *range(700_000, 700_004),
]
# Rows that are 0 except for a share of positive entries: the k-th value of a
# long row is then a 0 tied hundreds of thousands of times.
TIED = [(700_000, 0.0015), (700_001, 0.0015), (5_000, 0.0), (300_000, 0.02)]


def _decode_logits() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    lengths = LENGTHS + [length for length, _ in TIED]
    logits = torch.full((len(lengths), MAX_MODEL_LEN), float("-inf"), device="cuda")
    for row, length in enumerate(lengths):
        values = torch.randn(length, device="cuda", generator=generator)
        if row >= len(LENGTHS):
            positive = torch.rand(length, device="cuda", generator=generator)
            values = torch.where(
                positive < TIED[row - len(LENGTHS)][1],
                values.abs() + 0.1,
                torch.zeros_like(values),
            )
        logits[row, :length] = values
    return logits, torch.tensor(lengths, dtype=torch.int32, device="cuda")


def test_decode_top_k_is_exact():
    logits, lengths = _decode_logits()
    # The indexer's top-k buffer is wider than k.
    out = torch.full((len(lengths), 2 * TOPK), -7, dtype=torch.int32, device="cuda")
    out = out[:, :TOPK]

    assert rocm_triton_topk_decode.supported(logits, TOPK)
    rocm_triton_topk_decode.top_k_per_row_decode(logits, lengths, out, TOPK)

    for row, length in enumerate(lengths.tolist()):
        k = min(length, TOPK)
        selected, padding = out[row, :k], out[row, k:]
        assert (padding == -1).all()
        assert (selected >= 0).all() and (selected < length).all()
        assert (selected[1:] > selected[:-1]).all(), f"row {row} is not ascending"
        kth = torch.topk(logits[row, :length], k).values[-1]
        assert (logits[row, selected.long()] >= kth).all(), f"row {row}"


@pytest.mark.parametrize(
    ("rows", "dtype", "topk", "expected"),
    [
        (4, torch.float32, 2048, True),
        (4, torch.float32, 512, True),
        (rocm_triton_topk_decode.MAX_ROWS + 1, torch.float32, 2048, False),
        (4, torch.bfloat16, 2048, False),
        (4, torch.float32, 4096, False),
        (4, torch.float32, 1000, False),
    ],
)
def test_supported(rows, dtype, topk, expected):
    logits = torch.empty((rows, 4096), dtype=dtype)
    assert rocm_triton_topk_decode.supported(logits, topk) is expected
