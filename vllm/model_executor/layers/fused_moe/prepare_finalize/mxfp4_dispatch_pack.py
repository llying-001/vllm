# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One buffer for the MXFP4 naive DP dispatch.

The MXFP4 dispatch of `naive_dp_ep` all-gathers four tensors per MoE layer:
the packed FP4 hidden states, their e8m0 scales, the topk ids and the topk
weights. Packing them into one row per token makes it a single all-gather.
Row layout, in bytes, for hidden size K and topk T:

    [0, K / 2)              packed MXFP4 values
    [K / 2, K / 2 + K / 32) e8m0 scales
    [ids, ids + 4 T)        int32 topk ids, `ids` = the above rounded up to 4
    [weights, weights + 4T) float32 topk weights, `weights` = ids + 4 T
    padding to a multiple of 16 bytes

The copies in and out are bit-exact, so the experts see the same inputs as
with the four gathers.
"""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# Hidden columns per pack program; K must be a multiple of it.
_PACK_BLOCK_N = 256
_PACK_BLOCK_M = 4
_UNPACK_BLOCK = 1024


class Mxfp4DispatchLayout(NamedTuple):
    hidden_size: int
    topk: int
    scale_offset: int
    ids_offset: int
    weights_offset: int
    row_bytes: int

    @property
    def value_bytes(self) -> int:
        return self.hidden_size // 2

    @property
    def scale_bytes(self) -> int:
        return self.hidden_size // 32


def mxfp4_dispatch_layout(
    values: torch.Tensor,
    scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> Mxfp4DispatchLayout | None:
    """The packed row layout for MXFP4 `values` [M, K / 2], e8m0 `scales`
    [>= M, K / 32] and their routing, or None when they cannot be packed.
    Depends only on shapes, dtypes and strides."""
    m, k = values.shape[0], values.shape[-1] * 2
    if not (
        values.dim() == 2
        and values.element_size() == 1
        and values.stride(-1) == 1
        and k % _PACK_BLOCK_N == 0
        and scales.dim() == 2
        and scales.element_size() == 1
        and scales.shape[0] >= m
        and scales.shape[1] == k // 32
        and scales.stride(-1) == 1
        and topk_ids.dim() == 2
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.shape == topk_weights.shape
        and topk_ids.shape[0] == m
        and topk_ids.stride(-1) == 1
        and topk_weights.stride(-1) == 1
    ):
        return None
    t = topk_ids.shape[1]
    scale_offset = k // 2
    ids_offset = round_up(scale_offset + k // 32, 4)
    weights_offset = ids_offset + 4 * t
    return Mxfp4DispatchLayout(
        hidden_size=k,
        topk=t,
        scale_offset=scale_offset,
        ids_offset=ids_offset,
        weights_offset=weights_offset,
        # The AITER custom all-gather takes multiples of 16 bytes.
        row_bytes=round_up(weights_offset + 4 * t, 16),
    )


@triton.jit
def _pack_kernel(
    q_ptr,
    s_ptr,
    ids_ptr,
    w_ptr,
    out_ptr,
    out_i32_ptr,
    out_f32_ptr,
    M,
    stride_q,
    stride_s,
    stride_ids,
    stride_w,
    ROW: tl.constexpr,
    SOFF: tl.constexpr,
    IOFF: tl.constexpr,
    WOFF: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_N hidden columns of BLOCK_M rows (their values and scales) into
    the packed rows; the pid_n == 0 programs also copy the topk ids and
    weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rmask = rows < M
    qc = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    sc = pid_n * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    q = tl.load(q_ptr + rows[:, None] * stride_q + qc[None, :], mask=rmask[:, None])
    s = tl.load(s_ptr + rows[:, None] * stride_s + sc[None, :], mask=rmask[:, None])
    tl.store(out_ptr + rows[:, None] * ROW + qc[None, :], q, mask=rmask[:, None])
    tl.store(out_ptr + rows[:, None] * ROW + SOFF + sc[None, :], s, mask=rmask[:, None])
    if pid_n == 0:
        k = tl.arange(0, TOPK_P2)
        m2 = rmask[:, None] & (k < TOPK)[None, :]
        ids = tl.load(ids_ptr + rows[:, None] * stride_ids + k[None, :], mask=m2)
        w = tl.load(w_ptr + rows[:, None] * stride_w + k[None, :], mask=m2)
        row32 = rows[:, None] * (ROW // 4)
        tl.store(out_i32_ptr + row32 + (IOFF // 4) + k[None, :], ids, mask=m2)
        tl.store(out_f32_ptr + row32 + (WOFF // 4) + k[None, :], w, mask=m2)


@triton.jit
def _unpack_kernel(
    in_ptr,
    q_ptr,
    s_ptr,
    ids_ptr,
    w_ptr,
    ROW: tl.constexpr,
    QB: tl.constexpr,
    SB: tl.constexpr,
    IOFF: tl.constexpr,
    WOFF: tl.constexpr,
    TB: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One gathered row into the contiguous values, scales, ids and weights
    (all copied as bytes)."""
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    src = in_ptr + row * ROW
    c = pid_n * BLOCK_Q + tl.arange(0, BLOCK_Q)
    tl.store(q_ptr + row * QB + c, tl.load(src + c, mask=c < QB), mask=c < QB)
    if pid_n == 0:
        e = tl.arange(0, BLOCK_E)
        tl.store(s_ptr + row * SB + e, tl.load(src + QB + e, mask=e < SB), mask=e < SB)
        tl.store(
            ids_ptr + row * TB + e, tl.load(src + IOFF + e, mask=e < TB), mask=e < TB
        )
        tl.store(
            w_ptr + row * TB + e, tl.load(src + WOFF + e, mask=e < TB), mask=e < TB
        )


def pack_mxfp4_dispatch(
    values: torch.Tensor,
    scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layout: Mxfp4DispatchLayout,
) -> torch.Tensor:
    """uint8 [M, row_bytes]: `values`, `scales` and the routing, one row per
    token, in one launch. `layout` is `mxfp4_dispatch_layout` of the same
    tensors."""
    m = topk_ids.shape[0]
    out = torch.empty((m, layout.row_bytes), dtype=torch.uint8, device=values.device)
    if m == 0:
        return out
    q = values.view(torch.uint8)
    s = scales.view(torch.uint8)
    grid = (triton.cdiv(m, _PACK_BLOCK_M), layout.hidden_size // _PACK_BLOCK_N)
    _pack_kernel[grid](
        q,
        s,
        topk_ids,
        topk_weights,
        out,
        out.view(torch.int32),
        out.view(torch.float32),
        m,
        q.stride(0),
        s.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        ROW=layout.row_bytes,
        SOFF=layout.scale_offset,
        IOFF=layout.ids_offset,
        WOFF=layout.weights_offset,
        TOPK=layout.topk,
        TOPK_P2=triton.next_power_of_2(layout.topk),
        BLOCK_M=_PACK_BLOCK_M,
        BLOCK_N=_PACK_BLOCK_N,
    )
    return out


def unpack_mxfp4_dispatch(
    packed: torch.Tensor,
    layout: Mxfp4DispatchLayout,
    values_dtype: torch.dtype,
    scales_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gathered rows split into contiguous values [N, K / 2], scales
    [N, K / 32], topk weights and topk ids [N, T]. Contiguous copies, since
    AITER's fused MoE does not take strided views of the rows."""
    u8 = packed.view(torch.uint8)
    assert u8.dim() == 2 and u8.shape[1] == layout.row_bytes
    n, dev = u8.shape[0], u8.device
    q = torch.empty((n, layout.value_bytes), dtype=torch.uint8, device=dev)
    s = torch.empty((n, layout.scale_bytes), dtype=torch.uint8, device=dev)
    ids = torch.empty((n, layout.topk), dtype=torch.int32, device=dev)
    w = torch.empty((n, layout.topk), dtype=torch.float32, device=dev)
    if n:
        _unpack_kernel[(n, triton.cdiv(layout.value_bytes, _UNPACK_BLOCK))](
            u8,
            q,
            s,
            ids.view(torch.uint8),
            w.view(torch.uint8),
            ROW=layout.row_bytes,
            QB=layout.value_bytes,
            SB=layout.scale_bytes,
            IOFF=layout.ids_offset,
            WOFF=layout.weights_offset,
            TB=4 * layout.topk,
            BLOCK_Q=_UNPACK_BLOCK,
            BLOCK_E=triton.next_power_of_2(max(layout.scale_bytes, 4 * layout.topk)),
        )
    return q.view(values_dtype), s.view(scales_dtype), w, ids
