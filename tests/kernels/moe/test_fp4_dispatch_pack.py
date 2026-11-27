# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed row of the MXFP4 DP dispatch: one all-gather instead of four."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize import fp4_dispatch_pack
from vllm.model_executor.layers.fused_moe.prepare_finalize.fp4_dispatch_pack import (
    PackedRow,
    packed_row,
)
from vllm.triton_utils import HAS_TRITON

HIDDEN = 512
TOPK = 6


def _inputs(num_tokens: int, device: str):
    generator = torch.Generator().manual_seed(num_tokens)

    def random_bytes(width: int) -> torch.Tensor:
        return torch.randint(
            0, 256, (num_tokens, width), generator=generator, dtype=torch.uint8
        ).to(device)

    return (
        random_bytes(HIDDEN // 2).view(torch.float4_e2m1fn_x2),
        random_bytes(HIDDEN // 32).view(torch.float8_e8m0fnu),
        torch.randint(
            0, 256, (num_tokens, TOPK), generator=generator, dtype=torch.int32
        ).to(device),
        torch.rand((num_tokens, TOPK), generator=generator).to(device),
    )


def _fields(rows: torch.Tensor, layout: PackedRow) -> list[torch.Tensor]:
    """The four fields of packed rows as bytes; padding is left out."""
    topk_bytes = 4 * layout.topk
    return [
        rows[:, : layout.values],
        rows[:, layout.values : layout.values + layout.scales],
        rows[:, layout.ids_offset : layout.ids_offset + topk_bytes],
        rows[:, layout.weights_offset : layout.weights_offset + topk_bytes],
    ]


def test_packed_row_layout():
    # GLM-5.2: 6144 hidden, top-8 routing.
    assert packed_row(6144, 8) == PackedRow(
        values=3072, scales=192, ids_offset=3264, weights_offset=3296, size=3328, topk=8
    )
    # The ids start on an int32 boundary and the row is a multiple of 16 bytes.
    layout = packed_row(HIDDEN, TOPK)
    assert layout.ids_offset % 4 == 0 and layout.size % 16 == 0
    assert layout.size >= layout.weights_offset + 4 * TOPK


@pytest.mark.parametrize(
    ("hidden", "ids_dtype", "weights_dtype", "expected"),
    [
        (HIDDEN, torch.int32, torch.float32, True),
        (HIDDEN + 64, torch.int32, torch.float32, False),
        (HIDDEN, torch.int64, torch.float32, False),
        (HIDDEN, torch.int32, torch.bfloat16, False),
    ],
)
def test_can_pack(hidden, ids_dtype, weights_dtype, expected):
    hidden_states = torch.empty(3, hidden, dtype=torch.bfloat16)
    topk_ids = torch.zeros(3, TOPK, dtype=ids_dtype)
    topk_weights = torch.zeros(3, TOPK, dtype=weights_dtype)
    assert fp4_dispatch_pack.can_pack(hidden_states, topk_ids, topk_weights) is expected


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="needs Triton and a GPU"
)
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 7, 129, 256])
def test_pack_unpack_round_trip(num_tokens):
    layout = packed_row(HIDDEN, TOPK)
    fp4, fp4_scale, topk_ids, topk_weights = _inputs(num_tokens, "cuda")

    rows = fp4_dispatch_pack.pack(fp4, fp4_scale, topk_ids, topk_weights, layout)
    assert rows.shape == (num_tokens, layout.size)
    expected = [
        t.contiguous().view(torch.uint8).reshape(num_tokens, -1)
        for t in (fp4, fp4_scale, topk_ids, topk_weights)
    ]
    for packed, field in zip(_fields(rows, layout), expected):
        assert torch.equal(packed, field)

    # Two ranks' rows, gathered as BF16 like the dispatch does.
    gathered = torch.cat([rows, rows]).view(torch.bfloat16)
    unpacked = fp4_dispatch_pack.unpack(gathered, layout)
    assert unpacked[0].dtype == torch.float4_e2m1fn_x2
    assert unpacked[1].dtype == torch.float8_e8m0fnu
    for out, field in zip(unpacked, (fp4, fp4_scale, topk_ids, topk_weights)):
        assert out.is_contiguous() and out.dtype == field.dtype
        assert torch.equal(out.view(torch.uint8), field.view(torch.uint8).repeat(2, 1))
