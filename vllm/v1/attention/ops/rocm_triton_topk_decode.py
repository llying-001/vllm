# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-pass top-k (k <= 2048) for the DSA indexer's decode rows on gfx950.

AITER's decode top-k, which the ROCm indexer otherwise uses at k = 2048, gives
every row to one block, so a call takes as long as its longest row. This one
passes over all the rows' logits once, on a fixed grid:

- est: per row, sample 8192 logits (2048 probes of 4) and bit-search the
  sample key that leaves about `4 * topk` elements above it in the full row:
  a threshold per row, rounded down so that it only adds candidates. Program 0
  also builds the chunk prefix.
- collect: one pass over the logits on a persistent grid of row-aligned
  chunks; every key at or above its row's threshold is appended to that row's
  candidate buffer.
- select: per row, the exact k-th largest key among the candidates, then
  exactly `min(length, topk)` positions. A row with too few candidates, or
  more than the buffer holds, is selected exactly from the whole row instead
  (slow, rare), so the result is always an exact top-k.

Which of the elements tied at the k-th value are chosen depends on candidate
arrival order. The launch shape depends only on the row count, so the call is
CUDA graph safe.

Adapted from sgl-project/sglang#47872.
"""

import torch

from vllm.triton_utils import gl, gluon, tl, triton

_T = 8192  # logits per collect chunk
_GRID = 2048  # persistent collect programs
_CAND = 16384  # candidates per row
_OVER = 4  # candidate target, x topk
# Indexer logits are correlated along the row, so the sample is many short
# probes: with a few long ones the threshold is off by more than 2x on a
# large share of long rows.
_SMP_CH = 4  # probe length
_SMP_NCH = 2048  # probes per row
MAX_ROWS = 128
_CSTR = 64  # int32 gap between rows' counters
_COLLECT_NW = 4
_FALLBACK_BLOCK = 4096
# Tie mode when the t-th sample key is expected more than this often in the row.
_TIE_MIN = _CAND // 4


@triton.jit
def _sum_combine(a, b):
    return a + b


@triton.jit
def _okey(b):
    # order-preserving int32 key of a float32 bit pattern
    return tl.where(b >= 0, b, (~b) ^ (-2147483648))


@triton.jit
def _topk_est(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    stride_l,
    rows,
    CH: tl.constexpr,
    NCH: tl.constexpr,
    TOPK: tl.constexpr,
    OVER: tl.constexpr,
    T: tl.constexpr,
    RMAX: tl.constexpr,
    CSTR: tl.constexpr,
    TIE_MIN: tl.constexpr,
):
    r = tl.program_id(0)
    # program 0 also builds the chunk prefix
    if r == 0:
        ro = tl.arange(0, RMAX)
        msk = ro < rows
        La = tl.load(lengths_ptr + ro, mask=msk, other=0)
        cp = tl.cumsum((La + (T - 1)) // T, 0)
        tl.store(pref_ptr + 1 + ro, cp, mask=msk)
        tl.store(pref_ptr, 0)
    tl.store(ctr_ptr + r * CSTR + 1, 0)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    span = tl.maximum(CH, L // NCH)
    o = tl.arange(0, CH * NCH)
    p = (o // CH) * span + (o % CH)
    m = p < L
    b = tl.load(logits_ptr + r.to(tl.int64) * stride_l + p, mask=m, other=-1)
    key = _okey(b)
    ns = tl.sum(m.to(tl.int32), 0)
    t = tl.maximum(1, tl.minimum((OVER * k * ns) // tl.maximum(L, 1), ns))
    thr = 0
    if tl.sum((key >= 0).to(tl.int32), 0) < t:
        thr = -2147483648
    # bits 30..15 only: rounding the sampled threshold down can only add
    # candidates. Sampling can still overestimate the full-row k-th key;
    # the selector handles that case with its exact underflow fallback.
    for i in tl.static_range(0, 16):
        cand = thr + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            thr = cand
    # Tie guard: refine to the exact t-th sample key kt. If that one value is
    # expected to fill a large part of the row (indexer logits are exactly 0
    # wherever every head's ReLU is 0), collecting `>= thr` would overflow the
    # candidate buffer. Then collect only `> kt` and let the selector take the
    # missing `== kt` elements from the row: tied elements are interchangeable.
    kt = thr
    for i in tl.static_range(16, 31):
        cand = kt + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            kt = cand
    n_eq = tl.sum((m & (key == kt)).to(tl.int32), 0)
    strict = ((n_eq.to(tl.int64) * L) // tl.maximum(ns, 1) > TIE_MIN) & (
        kt < 2147483647
    )
    tl.store(ctr_ptr + r * CSTR, tl.where(strict, kt + 1, thr))
    tl.store(ctr_ptr + r * CSTR + 2, kt)
    tl.store(ctr_ptr + r * CSTR + 3, strict.to(tl.int32))


@gluon.jit
def _topk_collect(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    ckey_ptr,
    cidx_ptr,
    rows,
    stride_l,
    T: gl.constexpr,
    CAND: gl.constexpr,
    RMAX: gl.constexpr,
    GRID: gl.constexpr,
    CSTR: gl.constexpr,
    NW: gl.constexpr,
):
    # Thread-local compaction with a pinned layout: thread t of the CTA owns the R
    # 16-byte vectors (a, t, 0..3) of the chunk. The slot of a selected element is
    # base + (count of the threads before t) + (its rank inside thread t): per-thread
    # sums and in-thread ranks are register ops, the NT-wide scan (one value per
    # thread) is the only cross-thread step, and the scattered stores keep the load
    # layout (no LDS round trip). Within-chunk order is a permutation -- free, since
    # the selector's output order is unconstrained and cross-chunk order was already
    # atomic arrival order.
    NT: gl.constexpr = 64 * NW
    R: gl.constexpr = T // (NT * 4)
    L3: gl.constexpr = gl.BlockedLayout([R, 1, 4], [1, 64, 1], [1, NW, 1], [2, 1, 0])
    LRT: gl.constexpr = gl.SliceLayout(2, L3)
    L2: gl.constexpr = gl.BlockedLayout([R, 1], [1, 64], [1, NW], [1, 0])
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    ro = gl.arange(0, RMAX, layout=L1)
    pref = gl.load(pref_ptr + 1 + ro, mask=ro < rows, other=2147483647)
    total = gl.load(pref_ptr + rows)
    a = gl.arange(0, R, layout=gl.SliceLayout(1, LRT))
    t = gl.arange(0, NT, layout=gl.SliceLayout(0, LRT))
    v = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    va = gl.expand_dims(gl.expand_dims(a, 1) * (NT * 4) + gl.expand_dims(t, 0) * 4, 2)
    loc = va + gl.expand_dims(gl.expand_dims(v, 0), 1)
    for c in range(pid, total, GRID):
        r = gl.sum((c >= pref).to(gl.int32), 0)
        start = (c - gl.load(pref_ptr + r)) * T
        L = gl.load(lengths_ptr + r)
        thr = gl.load(ctr_ptr + r * CSTR)
        idx = start + loc
        # 16-byte-granular load mask (rows are 4-aligned and >= L wide, so the tail
        # vector's extra elements are in-bounds); the exact `idx < L` gates selection.
        # Buffer base = the kernel argument (uniform), row folded into the 32-bit offset
        # (host asserts the logits fit in 2**31 bytes): a base derived from the reduced
        # `r` is not provably uniform and every buffer op becomes a waterfall loop.
        b = gl.amd.cdna3.buffer_load(
            ptr=logits_ptr, offsets=r * stride_l + idx, mask=(start + va) < L, other=-1
        )
        key = gl.where(b >= 0, b, (~b) ^ (-2147483648))
        si = ((idx < L) & (key >= thr)).to(gl.int32)
        s_v = gl.sum(si, 2)
        cnt = gl.sum(s_v, 0)
        n = gl.sum(cnt, 0)
        if n > 0:
            base = gl.atomic_add(ctr_ptr + r * CSTR + 1, n, sem="relaxed")
            # tt.scan only lowers on blocked layouts: L1 / L2 are the same thread maps
            # as the slice layouts (no data movement).
            cnt_b = gl.convert_layout(cnt, L1)
            tb = gl.convert_layout(
                gl.associative_scan(cnt_b, 0, _sum_combine) - cnt_b,
                gl.SliceLayout(0, LRT),
            )
            s_vb = gl.convert_layout(s_v, L2)
            rv = gl.convert_layout(
                gl.associative_scan(s_vb, 0, _sum_combine) - s_vb, LRT
            )
            rk = gl.expand_dims(rv, 2) + (gl.associative_scan(si, 2, _sum_combine) - si)
            p = base + gl.expand_dims(gl.expand_dims(tb, 0), 2) + rk
            mm = (si != 0) & (p < CAND)
            gl.amd.cdna3.buffer_store(key, ptr=ckey_ptr, offsets=r * CAND + p, mask=mm)
            gl.amd.cdna3.buffer_store(idx, ptr=cidx_ptr, offsets=r * CAND + p, mask=mm)


@triton.jit
def _row_threshold(row_ptr, L, k, BLOCK: tl.constexpr):
    """Exact k-th largest key of a whole row, by 4 radix passes of 8 bits
    over u = key ^ INT_MIN, whose unsigned order is the key order."""
    bins = tl.arange(0, 512)
    # int32 scalars throughout: shifts into bit 31 wrap to the wanted pattern
    zero = L * 0
    ff = zero + 255
    prefix = zero
    pmask = zero
    need = k
    for p in tl.static_range(0, 4):
        h = tl.zeros((512,), tl.int32)
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            u = _okey(tl.load(row_ptr + o, mask=o < L, other=-1)) ^ (-2147483648)
            hit = (o < L) & ((u & pmask) == prefix)
            d = tl.where(hit, (u >> (24 - 8 * p)) & ff, 511)
            h += tl.histogram(d, 512)
        h = tl.where(bins < 256, h, 0)
        ge = tl.sum(h, 0) - tl.cumsum(h, 0) + h  # elements in bins >= b
        b = tl.max(tl.where(ge >= need, bins, -1), 0)
        need -= tl.sum(tl.where(bins > b, h, 0), 0)
        prefix = prefix | (b.to(tl.int32) << (24 - 8 * p))
        pmask = pmask | (ff << (24 - 8 * p))
    return prefix ^ (-2147483648)


@triton.jit
def _topk_select(
    ckey_ptr,
    cidx_ptr,
    ctr_ptr,
    lengths_ptr,
    logits_ptr,
    stride_l,
    out_ptr,
    stride_o,
    CAND: tl.constexpr,
    TOPK: tl.constexpr,
    CSTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    n_all = tl.load(ctr_ptr + r * CSTR + 1)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    out_cols = tl.arange(0, TOPK)
    strict = tl.load(ctr_ptr + r * CSTR + 3)
    tie = tl.load(ctr_ptr + r * CSTR + 2)
    zero = n_all * 0
    tie_ok = zero
    if (strict != 0) & (n_all < k):
        # Tie mode (see _topk_est): every candidate (> tie) is in the top-k;
        # the other k - n_all slots go to elements equal to the tie value,
        # taken in row order until enough are found.
        oc = tl.arange(0, CAND)
        mc = oc < n_all
        cb0 = r.to(tl.int64) * CAND
        idc = tl.load(cidx_ptr + cb0 + oc, mask=mc, other=0)
        tl.store(out_ptr + r * stride_o + oc, idc, mask=mc)
        need = k - n_all
        rowt = logits_ptr + r.to(tl.int64) * stride_l
        filled = zero
        s = zero
        while (filled < need) & (s < L):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(rowt + o, mask=o < L, other=-1))
            eq = (o < L) & (key == tie)
            eqi = eq.to(tl.int32)
            rank = filled + tl.cumsum(eqi, 0) - eqi
            take = eq & (rank < need)
            tl.store(out_ptr + r * stride_o + n_all + rank, o, mask=take)
            filled += tl.sum(take.to(tl.int32), 0)
            s += BLOCK
        tie_ok = (filled == need).to(tl.int32)
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=(out_cols >= k) & (tie_ok != 0),
        )
    if ((n_all < k) & (tie_ok == 0)) | (n_all > CAND):
        # The sampled threshold was too high (underflow) or too low (overflow):
        # select exactly from the whole row instead. Rare and slow.
        row = logits_ptr + r.to(tl.int64) * stride_l
        thr = _row_threshold(row, L, k, BLOCK)
        base = 0
        above = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            above += tl.sum(((o < L) & (key > thr)).to(tl.int32), 0)
        room = k - above
        eq_seen = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            gt = (o < L) & (key > thr)
            eq = (o < L) & (key == thr)
            eq_rank = eq_seen + tl.cumsum(eq.to(tl.int32), 0) - eq.to(tl.int32)
            take = gt | (eq & (eq_rank < room))
            ti = take.to(tl.int32)
            pos = base + tl.cumsum(ti, 0) - ti
            tl.store(out_ptr + r * stride_o + pos, o, mask=take)
            base += tl.sum(ti, 0)
            eq_seen += tl.sum(eq.to(tl.int32), 0)
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= base,
        )
    elif n_all >= k:
        n = n_all
        o = tl.arange(0, CAND)
        m = o < n
        cb = r.to(tl.int64) * CAND
        key = tl.load(ckey_ptr + cb + o, mask=m, other=-2147483648)
        thr = 0
        if tl.sum((key >= 0).to(tl.int32), 0) < k:
            thr = -2147483648
        for i in tl.static_range(0, 31):
            cand = thr + (1 << (30 - i))
            if tl.sum((key >= cand).to(tl.int32), 0) >= k:
                thr = cand
        gt = m & (key > thr)
        gti = gt.to(tl.int32)
        eq = m & (key == thr)
        eqi = eq.to(tl.int32)
        # one packed CAND-wide scan carries both ranks (each field < CAND <= 65536)
        cs = tl.cumsum(gti + (eqi * 65536), 0)
        rank_gt = (cs & 65535) - gti
        rank_eq = (cs // 65536) - eqi
        above = tl.sum(gti, 0)
        room = k - above
        take = gt | (eq & (rank_eq < room))
        n_take = tl.sum(take.to(tl.int32), 0)
        pos = rank_gt + tl.minimum(rank_eq, room)
        idx = tl.load(cidx_ptr + cb + o, mask=take, other=0)
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= n_take,
        )
        tl.store(out_ptr + r * stride_o + pos, idx, mask=take)


@triton.jit
def _sort_rows(out_ptr, stride_o, TOPK: tl.constexpr):
    """Sort one row of the output ascending, -1 padding last."""
    r = tl.program_id(0)
    c = tl.arange(0, TOPK)
    v = tl.load(out_ptr + r * stride_o + c)
    v = tl.sort(tl.where(v < 0, 2147483647, v))
    tl.store(out_ptr + r * stride_o + c, tl.where(v == 2147483647, -1, v))


_SCRATCH: dict[tuple[str, int | None], tuple[torch.Tensor, ...]] = {}


def _scratch(device: torch.device):
    """Fixed-address scratch (CUDA graph safe), sized for MAX_ROWS rows."""
    key = (device.type, device.index)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty((MAX_ROWS, _CSTR), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS + 1,), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS, _CAND), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS, _CAND), dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = buf
    return buf


def supported(logits: torch.Tensor, topk: int) -> bool:
    rows = logits.shape[0]
    return (
        0 < rows <= MAX_ROWS
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and rows * logits.stride(0) * 4 < 2**31
        and 0 < topk <= 2048
        and topk & (topk - 1) == 0
    )


def top_k_per_row_decode(
    logits: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor, topk: int
) -> torch.Tensor:
    """Write the positions of the `topk` largest of `logits[row, :lengths[row]]`
    into `out[row, :topk]`, ascending and padded with -1.

    Args:
        logits: fp32 `[rows, >= max(lengths)]` with unit column stride; see
            `supported`.
        lengths: int32 `[rows]`.
        out: int32 with unit column stride and at least `rows` rows.
        topk: A power of two, at most 2048.

    """
    rows = logits.shape[0]
    assert supported(logits, topk), (tuple(logits.shape), logits.stride(), topk)
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[0] >= rows
    assert lengths.dtype == torch.int32 and lengths.numel() >= rows
    ctr, pref, ckey, cidx = _scratch(logits.device)
    li = logits.view(torch.int32)
    sl = li.stride(0)
    _topk_est[(rows,)](
        li,
        lengths,
        pref,
        ctr,
        sl,
        rows,
        CH=_SMP_CH,
        NCH=_SMP_NCH,
        TOPK=topk,
        OVER=_OVER,
        T=_T,
        RMAX=MAX_ROWS,
        CSTR=_CSTR,
        TIE_MIN=_TIE_MIN,
        num_warps=8,
    )
    _topk_collect[(_GRID,)](
        li,
        lengths,
        pref,
        ctr,
        ckey,
        cidx,
        rows,
        sl,
        T=_T,
        CAND=_CAND,
        RMAX=MAX_ROWS,
        GRID=_GRID,
        CSTR=_CSTR,
        NW=_COLLECT_NW,
        num_warps=_COLLECT_NW,
    )
    _topk_select[(rows,)](
        ckey,
        cidx,
        ctr,
        lengths,
        li,
        sl,
        out,
        out.stride(0),
        CAND=_CAND,
        TOPK=topk,
        CSTR=_CSTR,
        BLOCK=_FALLBACK_BLOCK,
        num_warps=16,  # CAND-wide (16K) register arrays
    )
    # Ascending positions, like the other decode top-k kernels: the sparse MLA
    # kernel reads the KV cache in this order.
    _sort_rows[(rows,)](out, out.stride(0), TOPK=topk, num_warps=8)
    return out
