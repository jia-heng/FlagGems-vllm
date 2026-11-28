# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sparse MLA-absorb attention over an FP8 per-tile quantised KV cache (A5).

Replaces ``torch_npu.npu_kv_quant_sparse_flash_attention`` in the decode shape
vllm-ascend calls it with
(``vllm_ascend/device/device_op.py:634``, ``_execute_kv_quant_sparse_flash_attention_vllm``)::

    torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query, key=kv, value=kv, sparse_indices=topk_indices,
        scale_value=..., sparse_block_size=1, block_table=block_table,
        actual_seq_lengths_query=..., actual_seq_lengths_kv=...,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2, quant_scale_repo_mode=1,
        tile_size=128, key_quant_mode=2, value_quant_mode=2, rope_head_dim=64,
    )

The AscendC implementation this is aligned against lives in
``ops-transformer/attention/kv_quant_sparse_flash_attention/op_kernel/arch35/``
(the 950 takes the arch35 path) and in
``vllm-ascend/csrc/attention/kv_quant_sparse_flash_attention_vllm/``.

Call convention::

    query          [T, Q_N, 576]                    bf16    TND, 576 = 512 + 64
    kv_cache       [block_num, block_size, 1, 656]  fp8_e4m3fn   PA_BSND
    sparse_indices [T, 1, sparse]                   int32   valid first, -1 pad
    block_table    [B, max_blocks]                  int32
    -> out         [T, Q_N, 512]                    bf16

The 656-byte cache slot (``quant_scale_repo_mode=1``, quantisation parameters
stored next to the data) holds three dtypes per KV token:

    byte [0,   512)   512 x fp8_e4m3   nope (latent K; under MLA-absorb this is
                                       also V, there is no separate V tensor)
    byte [512, 640)   64  x bf16       rope
    byte [640, 656)   4   x fp32       per-tile dequantisation scale

Four scales rather than 512: ``tile_size=128`` and 512 / 128 = 4, so every 128
nope channels share one scale. Guessing per-channel or per-token is wrong.
Because one slot mixes three dtypes, the kernel takes three zero-copy views of
the same storage instead of bitcasting inside the kernel; see ``cache_views``.

Numerics (identical to the official A5 fp8 reference, ``test_a5_c8_lse`` in
``vllm-ascend/tests/e2e/nightly/.../test_kv_quant_sparse_flash_attention_vllm.py``)::

    k_deq  = (k_nope_fp8.float() * scale.repeat_interleave(128)).to(bf16)
    k      = cat(k_deq, k_rope)
    scores = q.float() @ k.T * scale_value
    out    = softmax(scores) @ k_deq.float()

A slot takes part in the attention only when both conditions hold: the index is
not the ``-1`` sentinel, and its position is below
``valid_count = min(sparse_size, ceil(threshold / sparse_block_size))``, which
for ``sparse_mode=3`` with ``Q_S=1`` makes the threshold
``actual_seq_lengths_kv``. The two are independent judgements, which is why
``actual_seq_lengths_kv`` is a required input here and not a cross-check.

Splitting: a decode step has ``T=1``, so the natural parallelism is the
``Q_N=64`` heads against 32 AI cores. The main kernel therefore also splits the
sparse dimension ``splits`` ways (flash-decoding): each program accumulates the
unnormalised partial sum of one KV segment plus that segment's ``(m, l)``, and
a combine kernel merges them. Program count has to oversubscribe the cores by a
lot, because the kernel is dominated by the discrete gather and a program
waiting on DMA lets another compute: 16 / 32 / 64 programs measure 50.7 / 29.6 /
18.5 us on the GLM-W4 shape. ``splits=1`` skips the combine kernel.

Measured on an idle 950DT (device kernel time, best of several rounds), against
the vendor operator on the two shapes it is called with:

    GLM-W4  (18.81% of model time)   vendor 47.92 us   triton 25.46 us  1.882x
    GLM-W8  (17.25% of model time)   vendor 47.80 us   triton 25.22 us  1.895x

Accuracy sits at the vendor's own level: against an fp32 reference this
operator measures 4.4e-03 and the vendor 4.0e-03 (bf16 output of a 2048-term
softmax-weighted sum).

Two fragile points worth knowing before changing the tiling:

  * ``block_h=64`` is a discrete feasible point, not a continuous knob. It only
    compiles while ``sparse // splits == block_s``, which makes the S loop a
    single iteration that gets folded away, so the compiler does not
    double-buffer ``k_raw`` / ``k_deq``. With a real loop the UB requirement is
    470-486 KB against the 216 KB available, and turning auto multi-buffer off
    at launch does not close a gap that size (measured: an empty knob here, it
    changes neither the feasible set nor the time). The planner below therefore
    only emits single-iteration plans.
  * ``combine_h`` of 2 or 4 is 4x slower (30-32 us) while 1 and 8 behave
    normally, which is not monotonic and looks like a 3D tile degenerating at
    those widths. The default is the fastest clean point, ``combine_h=1`` with
    ``block_sp=32``.

Covered: decode shapes (``T == B``, one query token per batch),
``attention_mode=2``, ``sparse_block_size=1``, ``key/value_quant_mode=2``,
``quant_scale_repo_mode=1``, ``tile_size=128``, ``rope_head_dim=64``,
``sparse_mode=3``, ``layout_query="TND"``, ``layout_kv="PA_BSND"``, fp8_e4m3fn
KV. Not covered, and raising instead of guessing: prefill (``Q_S > 1``) needs a
per-query causal threshold, where this implementation relies on the identity
that the threshold equals ``actual_seq_lengths_kv`` when ``Q_S=1``;
``return_softmax_lse`` (the partials already carry ``(m, l)``, the combine
kernel just does not write it out); ``rope_head_dim=0`` (528-byte slot); int8 /
hifloat8 / TQ4 KV; ``sparse_block_size > 1``; ``sparse_mode=0``.
"""

import torch
import triton
import triton.language as tl

__all__ = [
    "kv_quant_sparse_flash_attention",
    "cache_views",
    "plan_tiling",
    "NOPE",
    "ROPE",
    "TILE",
    "NTILE",
    "SLOT",
]

NOPE = 512  # latent / nope width
ROPE = 64  # rope width
TILE = 128  # the operator's tile_size attribute; only 128 exists
NTILE = NOPE // TILE  # 4 per-tile scales
SLOT = NOPE + ROPE * 2 + NTILE * 4  # 656 bytes per KV token

# Defaults of the measured best plan on the deployment shapes.
_BLOCK_S = 32
_SPLITS_MAX = 64
_BLOCK_H_MAX = 64
_COMBINE_H = 1
_BLOCK_SP = 32


@triton.jit
def _qsfa_kernel(
    q_ptr,  # [T, Q_N, 576] bf16, nope and rope in one row
    k_nope_ptr,  # fp8  view [NROW, 512]
    k_rope_ptr,  # bf16 view [NROW, 64]
    k_scl_ptr,  # fp32 view [NROW, 4]
    sparse_ptr,  # [T, 1, S] int32
    bt_ptr,  # [B, MAXBLK] int32
    seqkv_ptr,  # [B] int32/int64, actual_seq_lengths_kv
    out_ptr,  # [T, Q_N, 512] bf16, written when SPLITS == 1
    po_ptr,  # [T, SPLITS, Q_N, 512] partial sums
    pm_ptr,  # [T, SPLITS, Q_N] f32, per-segment max
    pl_ptr,  # [T, SPLITS, Q_N] f32, per-segment sum
    scale_value,
    QN: tl.constexpr,
    S: tl.constexpr,
    MAXBLK: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_R: tl.constexpr,
    SLOT_S: tl.constexpr,
    LOG2_BS: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SPLITS: tl.constexpr,
    LOG2_SPLITS: tl.constexpr,
    NOPE_C: tl.constexpr,
    ROPE_C: tl.constexpr,
    TILE_C: tl.constexpr,
    NTILE_C: tl.constexpr,
    DEQ_F32: tl.constexpr,
    DEQ3D: tl.constexpr,
    PART_BF16: tl.constexpr,
):
    # pid splits into (head block, KV segment). SPLITS is a power of two so
    # this uses a shift and a mask: // and % on runtime values measure 17-37x
    # slower on this backend.
    pid = tl.program_id(0)
    t = tl.program_id(1)  # batch / query token
    hb = pid >> LOG2_SPLITS
    sp = pid & (SPLITS - 1)

    CHUNK: tl.constexpr = S // SPLITS

    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    offs_d = tl.arange(0, NOPE_C)  # [512]
    offs_r = tl.arange(0, ROPE_C)  # [64]
    offs_t = tl.arange(0, NTILE_C)  # [4]

    qrow = q_ptr + t * QN * (NOPE_C + ROPE_C) + offs_h[:, None] * (NOPE_C + ROPE_C)
    q_nope = tl.load(qrow + offs_d[None, :])  # [BH, 512] bf16
    q_rope = tl.load(qrow + NOPE_C + offs_r[None, :])  # [BH, 64] bf16

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, NOPE_C], dtype=tl.float32)

    # valid_count = min(S, threshold); with sparse_mode=3 and Q_S=1 the
    # threshold is actual_seq_lengths_kv. One scalar load, hoisted.
    valid_cnt = tl.minimum(tl.load(seqkv_ptr + t), S)

    for s0 in range(0, CHUNK, BLOCK_S):
        offs_s = sp * CHUNK + s0 + tl.arange(0, BLOCK_S)
        idx = tl.load(sparse_ptr + t * S + offs_s)  # [BS] int32

        # A slot counts only if both hold:
        #
        #   (a) idx >= 0            the -1 sentinel; deployment left-aligns the
        #                           valid indices and pads with -1 (vllm-ascend
        #                           sfa_v1.py:176-178 sorts them on the A5 path)
        #   (b) offs_s < valid_cnt  the operator contract's valid_count
        #
        # Both are needed. The vendor builtin honours (b) but not (a): with
        # seq_kv left at kv_len and only half the slots filled it reads the -1
        # slots, which disagrees with the official reference
        # (``if sparse_id == -1: break``); setting seq_kv to the valid count
        # makes the two agree again.
        #
        # The loads stay unmasked: a masked load loses the regular access
        # pattern on this backend (the scale load alone measures 13x slower),
        # so the index is clamped to 0 and read as usual (slot 0, harmless) and
        # the score is set to -inf instead.
        valid = (idx >= 0) & (offs_s < valid_cnt)
        idxc = tl.maximum(idx, 0)

        # block_size is a power of two, so >> and & again.
        logical = idxc >> LOG2_BS
        off = idxc & (BS - 1)
        phys = tl.load(bt_ptr + t * MAXBLK + logical)  # [BS] gather
        row = phys * BS + off  # [BS] physical row

        k_raw = tl.load(
            k_nope_ptr + row[:, None] * SLOT_N + offs_d[None, :]
        )  # [BS, 512] fp8
        k_rope = tl.load(
            k_rope_ptr + row[:, None] * SLOT_R + offs_r[None, :]
        )  # [BS, 64] bf16
        sc4 = tl.load(
            k_scl_ptr + row[:, None] * SLOT_S + offs_t[None, :]
        )  # [BS, 4] f32

        # Expand the per-tile scale 4 -> 512 and dequantise. The three paths
        # measure the same max_rel (6.29e-03) on the deployment shapes; they
        # differ in how much UB the temporaries take, and UB is what caps
        # block_h.
        #
        # DEQ_F32 multiplies in fp32 like the reference, but k_raw.to(f32) and
        # the expanded scale are two [BS, 512] fp32 temporaries (128 KB at
        # BS=32 against 216 KB of UB), which pins block_h at 16.
        # The bf16 paths halve that: fp8_e4m3 -> bf16 is lossless (bf16's 8
        # exponent bits cover e4m3's 4, its 7 mantissa bits cover 3), so the
        # extra error is only the <=2^-8 relative rounding of the scale
        # (measured 3.887e-03), the same order as rounding the fp32 product,
        # not a systematic bias. DEQ3D also drops the expanded scale tensor by
        # viewing [BS, 512] as [BS, 4, 128] and multiplying [BS, 4, 1].
        if DEQ_F32:
            sc_e = tl.reshape(
                tl.broadcast_to(sc4[:, :, None], (BLOCK_S, NTILE_C, TILE_C)),
                (BLOCK_S, NOPE_C),
            )
            k_deq = (k_raw.to(tl.float32) * sc_e).to(tl.bfloat16)
        elif DEQ3D:
            k3 = tl.reshape(k_raw.to(tl.bfloat16), (BLOCK_S, NTILE_C, TILE_C))
            k_deq = tl.reshape(k3 * sc4.to(tl.bfloat16)[:, :, None], (BLOCK_S, NOPE_C))
        else:
            sc_b = tl.reshape(
                tl.broadcast_to(
                    sc4.to(tl.bfloat16)[:, :, None], (BLOCK_S, NTILE_C, TILE_C)
                ),
                (BLOCK_S, NOPE_C),
            )
            k_deq = k_raw.to(tl.bfloat16) * sc_b  # [BS, 512] bf16, this is V

        s = tl.dot(q_nope, tl.trans(k_deq), out_dtype=tl.float32)
        s += tl.dot(q_rope, tl.trans(k_rope), out_dtype=tl.float32)
        s = s * scale_value
        s = tl.where(valid[None, :], s, float("-inf"))  # [BH, BS]

        # Online softmax. A segment with no valid slot leaves m_new at -inf,
        # where exp(-inf - -inf) is nan, so the max is replaced by 0: every p
        # is then exp(-inf) = 0 and l and acc stay as they were.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.bfloat16), k_deq, out_dtype=tl.float32
        )
        m_i = m_new

    if SPLITS == 1:
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / safe_l[:, None]
        tl.store(
            out_ptr + t * QN * NOPE_C + offs_h[:, None] * NOPE_C + offs_d[None, :],
            o.to(tl.bfloat16),
        )
    else:
        base = sp * (QN * NOPE_C) + t * (SPLITS * QN * NOPE_C)
        # Partial traffic is SPLITS * QN * 512 * itemsize, 8 MB written plus
        # 8 MB read in fp32 at splits=64, against 1.34 MB of unique input and
        # 64 KB of output: this is what the combine kernel is bound by, and
        # PART_BF16 halves it.
        #
        # The bf16 store keeps acc unnormalised. Its magnitude is bounded
        # (exp(s - m) <= 1 over CHUNK tokens, so |acc| <= CHUNK * max|V|) and
        # bf16's 8 exponent bits cover that; only mantissa is lost. Dividing by
        # l first to regularise the magnitude needs one more [BH, 512] fp32
        # temporary, which overflows UB at block_h=64 (272 KB of 216 KB).
        if PART_BF16:
            tl.store(
                po_ptr + base + offs_h[:, None] * NOPE_C + offs_d[None, :],
                acc.to(tl.bfloat16),
            )
        else:
            tl.store(po_ptr + base + offs_h[:, None] * NOPE_C + offs_d[None, :], acc)
        mbase = sp * QN + t * (SPLITS * QN)
        tl.store(pm_ptr + mbase + offs_h, m_i)
        tl.store(pl_ptr + mbase + offs_h, l_i)


@triton.jit
def _qsfa_combine_kernel(
    po_ptr,
    pm_ptr,
    pl_ptr,
    out_ptr,
    QN: tl.constexpr,
    NOPE_C: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PART_BF16: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    """Merge the SPLITS unnormalised partial sums.

    ``out = sum_sp acc_sp * exp(m_sp - M) / sum_sp l_sp * exp(m_sp - M)`` with
    ``M = max_sp m_sp``.

    Two passes rather than an online softmax: ``(m, l)`` is only
    ``[SPLITS, BLOCK_H]`` (256 B at splits=64, bh=1), so it can be read in one
    go to get the global ``M`` and a whole row of weights ``w[sp]``. What is
    left is a single weighted reduction over ``po`` with no serial dependence
    across ``sp``, so the ``sp`` direction can read ``BLOCK_SP`` partials per
    iteration and the loop count drops to ``SPLITS / BLOCK_SP``.

    The online version read only ``[BLOCK_H, 512]`` per step (4 KB at bh=4) and
    needed 64 steps: 11.86 us for 4 MB is 340 GB/s, so it was bound by the
    per-step overhead rather than bandwidth, the same lesson as ``block_s`` in
    the main kernel. Two passes with ``BLOCK_SP=32`` measure 7.28 us.
    """
    hb = tl.program_id(0)
    t = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, NOPE_C)
    offs_a = tl.arange(0, SPLITS)

    # First pass: read every (m, l), take the global max and the weights.
    ml_base = t * (SPLITS * QN) + offs_a[:, None] * QN + offs_h[None, :]
    pm_all = tl.load(pm_ptr + ml_base)  # [SPLITS, BH]
    pl_all = tl.load(pl_ptr + ml_base)  # [SPLITS, BH]

    M = tl.max(pm_all, axis=0)  # [BH]
    M_safe = tl.where(M == float("-inf"), 0.0, M)
    w = tl.where(pm_all == float("-inf"), 0.0, tl.exp(pm_all - M_safe[None, :]))
    l_tot = tl.sum(pl_all * w, axis=0)  # [BH]

    # Second pass: weighted reduction over po, BLOCK_SP partials at a time.
    acc = tl.zeros([BLOCK_H, NOPE_C], dtype=tl.float32)
    for s0 in range(0, SPLITS, BLOCK_SP):
        offs_s = s0 + tl.arange(0, BLOCK_SP)
        po = tl.load(
            po_ptr
            + t * (SPLITS * QN * NOPE_C)
            + offs_s[:, None, None] * (QN * NOPE_C)
            + offs_h[None, :, None] * NOPE_C
            + offs_d[None, None, :]
        )  # [BSP, BH, 512]
        wb = tl.load(
            pm_ptr + t * (SPLITS * QN) + offs_s[:, None] * QN + offs_h[None, :]
        )  # [BSP, BH]
        wb = tl.where(wb == float("-inf"), 0.0, tl.exp(wb - M_safe[None, :]))
        acc += tl.sum(po.to(tl.float32) * wb[:, :, None], axis=0)

    safe_l = tl.where(l_tot > 0.0, l_tot, 1.0)
    o = acc / safe_l[:, None]
    tl.store(
        out_ptr + t * QN * NOPE_C + offs_h[:, None] * NOPE_C + offs_d[None, :],
        o.to(tl.bfloat16),
    )


def cache_views(cache: torch.Tensor):
    """Three zero-copy views of the packed KV cache: fp8 nope, bf16 rope, fp32 scale.

    ``.view(dtype)`` cannot express these: it needs the last dimension to be
    contiguous and the whole tensor to be reinterpretable, while a slot holds
    three dtypes and none of the three segments is a suffix of the tensor.
    ``Tensor.set_`` takes (storage_offset, size, stride) directly. The byte
    offsets 0 / 512 / 640 divide by the respective itemsizes 1 / 2 / 4, so the
    conversion to element offsets is exact.
    """
    st = cache.untyped_storage()
    nrow = cache.numel() // SLOT
    dev = cache.device

    def mk(dtype, elem_off, width, slot_elems):
        t = torch.empty(0, dtype=dtype, device=dev)
        t.set_(st, storage_offset=elem_off, size=(nrow, width), stride=(slot_elems, 1))
        return t

    nope = mk(torch.float8_e4m3fn, 0, NOPE, SLOT)  # 656 / 1
    rope = mk(torch.bfloat16, NOPE // 2, ROPE, SLOT // 2)  # 512/2, 656/2
    scl = mk(torch.float32, (NOPE + ROPE * 2) // 4, NTILE, SLOT // 4)  # 640/4, 656/4
    return nope, rope, scl


def _prev_pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1) if n > 0 else 1


def plan_tiling(
    sparse: int,
    heads: int,
    splits: int | None = None,
    block_h: int | None = None,
    block_s: int | None = None,
    combine_h: int | None = None,
    block_sp: int | None = None,
):
    """Pick ``(splits, block_h, block_s, combine_h, block_sp)`` for a shape.

    The rule is one S-loop iteration per program, ``sparse // splits ==
    block_s``, because that is the only region where ``block_h`` can reach 64:
    the loop folds away and the compiler stops double-buffering the KV tiles,
    which otherwise needs 470-486 KB of the 216 KB UB. On the deployment shape
    (sparse=2048, Q_N=64) this reproduces the measured best point, splits=64
    with block_h=64 and block_s=32; a larger ``sparse`` raises ``splits``
    rather than widening the tile.

    Any argument given explicitly is kept as is, which is how the sweeps
    explore other points.
    """
    bs = block_s if block_s is not None else min(_BLOCK_S, _prev_pow2(sparse))
    if splits is None:
        if sparse & (sparse - 1) or sparse % bs:
            raise NotImplementedError(
                f"the automatic plan needs a power-of-two sparse size, got "
                f"{sparse}; pass splits and block_s explicitly to override"
            )
        splits = sparse // bs
    bh = block_h
    if bh is None:
        bh = min(_BLOCK_H_MAX, _prev_pow2(heads))
        while heads % bh:
            bh //= 2
    ch = combine_h if combine_h is not None else _COMBINE_H
    bsp = block_sp if block_sp is not None else min(_BLOCK_SP, splits)
    return splits, bh, bs, ch, min(bsp, splits)


def kv_quant_sparse_flash_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_kv,
    scale_value: float | None = None,
    *,
    splits: int | None = None,
    block_h: int | None = None,
    block_s: int | None = None,
    combine_h: int | None = None,
    block_sp: int | None = None,
    deq_f32: bool = False,
    deq3d: bool = True,
    part_bf16: bool = True,
    num_cores: int | None = None,
):
    """Sparse MLA-absorb attention over an fp8 per-tile quantised KV cache.

    Arguments shared with ``torch_npu.npu_kv_quant_sparse_flash_attention``
    keep that operator's meaning::

        query                 [T, Q_N, 576]  bf16, TND (512 nope + 64 rope)
        kv_cache              [block_num, block_size, 1, 656] fp8_e4m3fn
        sparse_indices        [T, 1, sparse] int32, valid first, -1 padded
        block_table           [B, max_blocks] int32
        actual_seq_lengths_kv [B] int32/int64, or a sequence of ints
        -> out                [T, Q_N, 512]  bf16

    ``actual_seq_lengths_kv`` is required rather than a cross-check: it is the
    threshold in the contract's ``valid_count = min(sparse_size, ceil(threshold
    / sparse_block_size))``, which with ``sparse_mode=3`` and ``Q_S=1`` is this
    value. Positions at or above ``valid_count`` never take part, independently
    of the ``-1`` sentinel.

    The tiling arguments default to ``plan_tiling``, which reproduces the
    measured best point on the deployment shapes. ``deq_f32`` / ``deq3d``
    select the dequantisation path (see the kernel), ``part_bf16`` halves the
    partial-sum traffic, and ``num_cores`` is passed through to the launch.
    """
    if query.dtype != torch.bfloat16:
        raise ValueError(f"query must be bf16, got {query.dtype}")
    if kv_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(f"kv_cache must be fp8_e4m3fn, got {kv_cache.dtype}")
    if query.ndim != 3 or query.shape[-1] != NOPE + ROPE:
        raise ValueError(
            f"query must be [T, Q_N, {NOPE + ROPE}], got {tuple(query.shape)}"
        )
    if kv_cache.ndim != 4 or kv_cache.shape[-1] != SLOT or kv_cache.shape[-2] != 1:
        raise ValueError(
            f"kv_cache must be [block_num, block_size, 1, {SLOT}], got "
            f"{tuple(kv_cache.shape)}"
        )
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D, got {tuple(block_table.shape)}")

    T, QN, _ = query.shape
    block_size = kv_cache.shape[1]
    S = sparse_indices.shape[-1]
    B, MAXBLK = block_table.shape

    if block_size & (block_size - 1):
        raise ValueError(
            f"block_size={block_size} must be a power of two (the kernel uses "
            f">> and & instead of // and %)"
        )
    if T != B:
        raise NotImplementedError(
            f"only decode shapes are supported, one query token per batch; got "
            f"T={T} B={B}"
        )

    nsplits, bh, bs, ch, bsp = plan_tiling(
        S, QN, splits, block_h, block_s, combine_h, block_sp
    )
    if nsplits & (nsplits - 1):
        raise ValueError(f"splits={nsplits} must be a power of two")
    if S % nsplits or (S // nsplits) % bs:
        raise ValueError(
            f"sparse={S} must divide by splits={nsplits}, and the segment "
            f"{S // nsplits} by block_s={bs}"
        )
    if QN % bh or QN % ch:
        raise ValueError(f"Q_N={QN} must divide by block_h={bh} and combine_h={ch}")

    if isinstance(actual_seq_lengths_kv, (list, tuple)):
        seqkv = torch.tensor(
            actual_seq_lengths_kv, dtype=torch.int32, device=query.device
        )
    else:
        seqkv = actual_seq_lengths_kv.reshape(-1).contiguous()
        if seqkv.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"actual_seq_lengths_kv must be int32 or int64, got {seqkv.dtype}"
            )
        if seqkv.device != query.device:
            raise ValueError(
                f"actual_seq_lengths_kv is on {seqkv.device}, query on "
                f"{query.device}"
            )
    if seqkv.numel() != B:
        raise ValueError(
            f"actual_seq_lengths_kv must hold B={B} entries, got {seqkv.numel()}"
        )
    if scale_value is None:
        scale_value = (NOPE + ROPE) ** -0.5

    q = query.contiguous()
    si = sparse_indices.contiguous()
    bt = block_table.contiguous()
    k_nope, k_rope, k_scl = cache_views(kv_cache)

    out = torch.empty(T, QN, NOPE, device=query.device, dtype=torch.bfloat16)
    nh = QN // bh

    if nsplits == 1:
        # Unused by the kernel on this path; pass a real pointer anyway.
        po = pm = pl = out
    else:
        po = torch.empty(
            T,
            nsplits,
            QN,
            NOPE,
            device=query.device,
            dtype=torch.bfloat16 if part_bf16 else torch.float32,
        )
        pm = torch.empty(T, nsplits, QN, device=query.device, dtype=torch.float32)
        pl = torch.empty(T, nsplits, QN, device=query.device, dtype=torch.float32)

    extra = {} if num_cores is None else {"num_cores": num_cores}

    _qsfa_kernel[(nh * nsplits, T)](
        q,
        k_nope,
        k_rope,
        k_scl,
        si,
        bt,
        seqkv,
        out,
        po,
        pm,
        pl,
        scale_value,
        QN=QN,
        S=S,
        MAXBLK=MAXBLK,
        SLOT_N=SLOT,
        SLOT_R=SLOT // 2,
        SLOT_S=SLOT // 4,
        LOG2_BS=block_size.bit_length() - 1,
        BS=block_size,
        BLOCK_H=bh,
        BLOCK_S=bs,
        SPLITS=nsplits,
        LOG2_SPLITS=nsplits.bit_length() - 1,
        NOPE_C=NOPE,
        ROPE_C=ROPE,
        TILE_C=TILE,
        NTILE_C=NTILE,
        DEQ_F32=deq_f32,
        DEQ3D=deq3d,
        PART_BF16=part_bf16,
        **extra,
    )

    if nsplits > 1:
        # The combine head block is decoupled from the main kernel's: block_h
        # there is capped by UB, while this kernel only moves partial sums, so
        # a finer split means a larger grid and better core occupancy.
        _qsfa_combine_kernel[(QN // ch, T)](
            po,
            pm,
            pl,
            out,
            QN=QN,
            NOPE_C=NOPE,
            SPLITS=nsplits,
            BLOCK_H=ch,
            PART_BF16=part_bf16,
            BLOCK_SP=bsp,
        )

    return out
