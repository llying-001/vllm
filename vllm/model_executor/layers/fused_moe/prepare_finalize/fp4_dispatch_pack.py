# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One row per token for the packed-MXFP4 DP dispatch: the activations, their
e8m0 scales, the top-k ids and the top-k weights, so that a single all-gather
moves what the dispatch otherwise gathers as four tensors."""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

_BLOCK_M = 4
_BLOCK_N = 256
_UNPACK_BLOCK = 1024


class PackedRow(NamedTuple):
    """Byte layout of a row: `[0, values)` packed MXFP4, then `scales` bytes
    of e8m0 scales, `topk` int32 ids at `ids_offset` and `topk` float32 weights
    at `weights_offset`, padded to `size`."""

    values: int
    scales: int
    ids_offset: int
    weights_offset: int
    size: int
    topk: int


def packed_row(hidden_size: int, topk: int) -> PackedRow:
    values, scales = hidden_size // 2, hidden_size // 32
    ids_offset = (values + scales + 3) // 4 * 4
    weights_offset = ids_offset + 4 * topk
    # A multiple of 16 bytes, which AITER's custom all-gather needs.
    size = (weights_offset + 4 * topk + 15) // 16 * 16
    return PackedRow(values, scales, ids_offset, weights_offset, size, topk)


def can_pack(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> bool:
    return (
        hidden_states.dim() == 2
        and hidden_states.shape[1] % _BLOCK_N == 0
        and topk_ids.shape[0] == hidden_states.shape[0]
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.dim() == 2
        and topk_ids.shape == topk_weights.shape
        and topk_ids.stride(-1) == 1
        and topk_weights.stride(-1) == 1
    )


@triton.jit
def _pack_kernel(
    values_ptr,
    scales_ptr,
    ids_ptr,
    weights_ptr,
    out_ptr,
    out_i32_ptr,
    out_f32_ptr,
    M,
    stride_values,
    stride_scales,
    stride_ids,
    stride_weights,
    ROW: tl.constexpr,
    SCALES_OFFSET: tl.constexpr,
    IDS_OFFSET: tl.constexpr,
    WEIGHTS_OFFSET: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    row_mask = rows < M
    value_cols = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    scale_cols = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    values = tl.load(
        values_ptr + rows[:, None] * stride_values + value_cols[None, :],
        mask=row_mask[:, None],
    )
    scales = tl.load(
        scales_ptr + rows[:, None] * stride_scales + scale_cols[None, :],
        mask=row_mask[:, None],
    )
    tl.store(
        out_ptr + rows[:, None] * ROW + value_cols[None, :],
        values,
        mask=row_mask[:, None],
    )
    tl.store(
        out_ptr + rows[:, None] * ROW + SCALES_OFFSET + scale_cols[None, :],
        scales,
        mask=row_mask[:, None],
    )
    if pid_n == 0:
        k = tl.arange(0, TOPK_P2)
        mask = row_mask[:, None] & (k < TOPK)[None, :]
        ids = tl.load(ids_ptr + rows[:, None] * stride_ids + k[None, :], mask=mask)
        weights = tl.load(
            weights_ptr + rows[:, None] * stride_weights + k[None, :], mask=mask
        )
        tl.store(
            out_i32_ptr + rows[:, None] * (ROW // 4) + (IDS_OFFSET // 4) + k[None, :],
            ids,
            mask=mask,
        )
        tl.store(
            out_f32_ptr
            + rows[:, None] * (ROW // 4)
            + (WEIGHTS_OFFSET // 4)
            + k[None, :],
            weights,
            mask=mask,
        )


@triton.jit
def _unpack_kernel(
    in_ptr,
    values_ptr,
    scales_ptr,
    ids_ptr,
    weights_ptr,
    ROW: tl.constexpr,
    VALUES: tl.constexpr,
    SCALES: tl.constexpr,
    IDS_OFFSET: tl.constexpr,
    WEIGHTS_OFFSET: tl.constexpr,
    TOPK_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    src = in_ptr + row * ROW
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        values_ptr + row * VALUES + cols,
        tl.load(src + cols, mask=cols < VALUES),
        mask=cols < VALUES,
    )
    if pid_n == 0:
        tail = tl.arange(0, BLOCK_TAIL)
        tl.store(
            scales_ptr + row * SCALES + tail,
            tl.load(src + VALUES + tail, mask=tail < SCALES),
            mask=tail < SCALES,
        )
        tl.store(
            ids_ptr + row * TOPK_BYTES + tail,
            tl.load(src + IDS_OFFSET + tail, mask=tail < TOPK_BYTES),
            mask=tail < TOPK_BYTES,
        )
        tl.store(
            weights_ptr + row * TOPK_BYTES + tail,
            tl.load(src + WEIGHTS_OFFSET + tail, mask=tail < TOPK_BYTES),
            mask=tail < TOPK_BYTES,
        )


def pack(
    fp4: torch.Tensor,
    fp4_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: PackedRow,
) -> torch.Tensor:
    """This rank's rows as uint8 `[M, layout.size]`."""
    num_tokens = fp4.shape[0]
    out = torch.empty((num_tokens, layout.size), dtype=torch.uint8, device=fp4.device)
    if num_tokens == 0:
        return out
    values, scales = fp4.view(torch.uint8), fp4_scale.view(torch.uint8)
    grid = (triton.cdiv(num_tokens, _BLOCK_M), 2 * layout.values // _BLOCK_N)
    _pack_kernel[grid](
        values,
        scales,
        topk_ids,
        topk_weights,
        out,
        out.view(torch.int32),
        out.view(torch.float32),
        num_tokens,
        values.stride(0),
        scales.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        ROW=layout.size,
        SCALES_OFFSET=layout.values,
        IDS_OFFSET=layout.ids_offset,
        WEIGHTS_OFFSET=layout.weights_offset,
        TOPK=layout.topk,
        TOPK_P2=triton.next_power_of_2(layout.topk),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
    )
    return out


def unpack(
    gathered: torch.Tensor, layout: PackedRow
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gathered rows as contiguous (MXFP4 activations, e8m0 scales,
    top-k ids, top-k weights)."""
    rows = gathered.view(torch.uint8)
    assert rows.dim() == 2 and rows.shape[1] == layout.size
    num_tokens, device = rows.shape[0], rows.device
    values = torch.empty((num_tokens, layout.values), dtype=torch.uint8, device=device)
    scales = torch.empty((num_tokens, layout.scales), dtype=torch.uint8, device=device)
    ids = torch.empty((num_tokens, layout.topk), dtype=torch.int32, device=device)
    weights = torch.empty((num_tokens, layout.topk), dtype=torch.float32, device=device)
    if num_tokens:
        grid = (num_tokens, triton.cdiv(layout.values, _UNPACK_BLOCK))
        _unpack_kernel[grid](
            rows,
            values,
            scales,
            ids.view(torch.uint8),
            weights.view(torch.uint8),
            ROW=layout.size,
            VALUES=layout.values,
            SCALES=layout.scales,
            IDS_OFFSET=layout.ids_offset,
            WEIGHTS_OFFSET=layout.weights_offset,
            TOPK_BYTES=4 * layout.topk,
            BLOCK=_UNPACK_BLOCK,
            BLOCK_TAIL=triton.next_power_of_2(max(layout.scales, 4 * layout.topk)),
        )
    return (
        values.view(torch.float4_e2m1fn_x2),
        scales.view(torch.float8_e8m0fnu),
        ids,
        weights,
    )
