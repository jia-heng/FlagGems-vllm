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

"""Accuracy tests for the A5 sparse MLA-absorb attention over an fp8 KV cache.

Three references per case: the vendor operator
``torch_npu.npu_kv_quant_sparse_flash_attention`` called the way vllm-ascend
calls it, and a CPU fp32 implementation of the official A5 numerics
(``test_a5_c8_lse`` in vllm-ascend). The CPU reference is what decides whether
an input was built correctly: with the 656-byte slot laid out right the vendor
lands at 4.0e-03 of it, and a wrong layout shows up as O(1).

Run on an idle 950DT, for example:

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q \\
        tests/test_kv_quant_sparse_flash_attention.py
"""

import os

# TMPDIR must point to /tmp before importing torch_npu, so that the benchmark
# importing this module can still profile.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.kv_quant_sparse_flash_attention import (  # noqa: E402
    NOPE,
    NTILE,
    ROPE,
    SLOT,
    TILE,
    kv_quant_sparse_flash_attention,
)

DEV = "npu"
SCALE_VALUE = (NOPE + ROPE) ** -0.5

# bf16 output of a softmax-weighted sum over up to 2048 dequantised fp8 rows;
# the vendor itself measures 4.0e-03 against the fp32 reference.
TOL = 2e-2

# The two shapes the operator is called with in deployment
# (op_list_glm53_w4a4_graph_meta.csv); they differ only in block_num and
# kv_len, the query and sparse shapes are the same.
SHAPES = {
    "glm_w4": dict(
        heads=64,
        kv_len=8192,
        sparse=2048,
        block_size=128,
        block_num=4194,
        name="GLM-W4 (18.81%)",
    ),
    "glm_w8": dict(
        heads=64,
        kv_len=4096,
        sparse=2048,
        block_size=128,
        block_num=8778,
        name="GLM-W8 (17.25%)",
    ),
}

# (name, heads, kv_len, sparse, n_valid). Where n_valid < sparse the tail is
# padded with -1 and seq_kv is set to n_valid: the contract's valid_count is
# min(sparse_size, threshold) and the threshold is actual_seq_lengths_kv, so
# the valid count has to be expressed through it. Leaving seq_kv at kv_len and
# marking the tail only with -1 keeps those slots inside valid_count, where
# the vendor reads them and drifts to 6e-01 - that is a malformed case, not an
# implementation difference.
EDGES = [
    ("all valid", 64, 8192, 2048, 2048),
    ("half valid, -1 padded", 64, 8192, 2048, 1024),
    ("one valid", 64, 8192, 2048, 1),
    ("none valid", 64, 8192, 2048, 0),
    ("heads=16", 16, 4096, 2048, 2048),
    ("heads=32, kv=2048", 32, 2048, 2048, 2048),
    ("sparse=512", 64, 8192, 512, 512),
]


def build_inputs(heads, kv_len, sparse, block_size, block_num, seed=0):
    """Build the packed fp8 per-tile quantised KV cache and the sparse indices.

    The cache is allocated as uint8 and written through the three segment
    offsets of the 656-byte slot: ``torch.cat`` on fp8 goes through aclnnCat
    and wants one dtype, while a slot mixes three. Writing through the byte
    view keeps the layout exactly as the operator defines it.

    ``block_table`` holds a random permutation of physical blocks rather than
    an arange: pages are scattered in real serving, and a contiguous arange
    would measure the discrete gather away.
    """
    g = torch.Generator().manual_seed(seed)
    n_blk = kv_len // block_size

    q_nope = torch.randn(1, heads, NOPE, generator=g).to(torch.bfloat16)
    q_rope = torch.randn(1, heads, ROPE, generator=g).to(torch.bfloat16)
    query = torch.cat([q_nope, q_rope], dim=-1).npu()

    # Logical KV: only the first kv_len tokens exist.
    k_nope = (torch.randn(kv_len, NOPE, generator=g) * 2.0).to(torch.float8_e4m3fn)
    k_rope = torch.randn(kv_len, ROPE, generator=g).to(torch.bfloat16)
    scales = (torch.rand(kv_len, NTILE, generator=g) * 0.1 + 0.05).float()

    cache_u8 = torch.zeros(block_num, block_size, 1, SLOT, dtype=torch.uint8)
    flat = cache_u8.view(-1, SLOT)  # [block_num * block_size, 656]

    perm = torch.randperm(block_num, generator=g)[:n_blk].to(torch.int32)
    block_table = perm.reshape(1, n_blk)

    # Logical token t sits at physical row perm[t // bs] * bs + t % bs.
    rows = (
        perm.long()[torch.arange(kv_len) // block_size] * block_size
        + torch.arange(kv_len) % block_size
    )
    flat[rows, 0:NOPE] = k_nope.view(torch.uint8)
    flat[rows, NOPE : NOPE + ROPE * 2] = k_rope.contiguous().view(torch.uint8)
    flat[rows, NOPE + ROPE * 2 : SLOT] = scales.contiguous().view(torch.uint8)

    cache = cache_u8.view(torch.float8_e4m3fn).npu()

    # sparse_indices: sparse distinct tokens out of kv_len, all valid.
    idx = torch.randperm(kv_len, generator=g)[:sparse].to(torch.int32)
    sparse_indices = idx.reshape(1, 1, sparse).npu()

    meta = dict(
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
        scales=scales,
        idx=idx,
    )
    return query, cache, sparse_indices, block_table.npu(), meta


def vendor_kwargs(query, cache, sparse_indices, block_table, seq_kv, seq_q=None):
    """Vendor arguments as vllm-ascend passes them (``device_op.py:634``).

    ``seq_kv`` takes an int or the ready int32 tensor, and ``seq_q`` the ready
    tensor; passing tensors in keeps the benchmark from creating one per call,
    which would add a host-to-device copy to the vendor's measured time.
    """
    if not torch.is_tensor(seq_kv):
        seq_kv = torch.tensor([seq_kv], dtype=torch.int32, device=DEV)
    if seq_q is None:
        seq_q = torch.tensor([1] * query.shape[0], dtype=torch.int32, device=DEV)
    return dict(
        query=query,
        key=cache,
        value=cache,
        sparse_indices=sparse_indices,
        scale_value=SCALE_VALUE,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=seq_q,
        actual_seq_lengths_kv=seq_kv,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=TILE,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=ROPE,
    )


def vendor(query, cache, sparse_indices, block_table, seq_kv):
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        **vendor_kwargs(query, cache, sparse_indices, block_table, seq_kv)
    )


def torch_ref(meta, valid_cnt=None):
    """CPU fp32 reference, the same numerics as the official A5 reference.

    V is the dequantised nope half, there is no separate V tensor, and the
    dequantisation multiplies in fp32 before rounding to bf16.
    """
    q = torch.cat([meta["q_nope"], meta["q_rope"]], dim=-1)[0].float()
    sel = meta["idx"].long()
    if valid_cnt is not None:
        sel = sel[:valid_cnt]
    sel = sel[sel >= 0]
    deq = (meta["k_nope"].float() * meta["scales"].repeat_interleave(TILE, dim=-1)).to(
        torch.bfloat16
    )
    k = torch.cat([deq, meta["k_rope"]], dim=-1)[sel].float()
    scores = q @ k.T * SCALE_VALUE
    return torch.softmax(scores, dim=-1) @ deq[sel].float()


def max_rel(got, ref):
    return ((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()


@pytest.mark.kv_quant_sparse_flash_attention
@pytest.mark.parametrize("shape", ["glm_w4", "glm_w8"])
def test_kv_quant_sparse_flash_attention(shape, capsys):
    cfg = SHAPES[shape]
    query, cache, si, bt, meta = build_inputs(
        cfg["heads"], cfg["kv_len"], cfg["sparse"], cfg["block_size"], cfg["block_num"]
    )
    kw = vendor_kwargs(query, cache, si, bt, cfg["kv_len"])
    print(
        f"== {cfg['name']}: query[1,{cfg['heads']},{NOPE + ROPE}] "
        f"cache[{cfg['block_num']},{cfg['block_size']},1,{SLOT}] "
        f"sparse[1,1,{cfg['sparse']}] kv_len={cfg['kv_len']} =="
    )

    got = kv_quant_sparse_flash_attention(
        query,
        cache,
        si,
        bt,
        kw["actual_seq_lengths_kv"],
        scale_value=SCALE_VALUE,
    )[0]
    got = got.cpu().float()
    base = torch_npu.npu_kv_quant_sparse_flash_attention(**kw)[0].cpu().float()
    ref = torch_ref(meta)

    r_base = max_rel(got, base)
    r_ref = max_rel(got, ref)
    r_vendor_ref = max_rel(base, ref)
    print(f"  triton vs vendor   max_rel {r_base:.3e}")
    print(f"  triton vs fp32 ref max_rel {r_ref:.3e}")
    print(f"  vendor vs fp32 ref max_rel {r_vendor_ref:.3e}   <- the same order")

    assert not got.isnan().any(), capsys.readouterr().out
    assert r_base <= TOL, capsys.readouterr().out
    assert r_ref <= TOL, capsys.readouterr().out


@pytest.mark.kv_quant_sparse_flash_attention
@pytest.mark.parametrize("case", EDGES, ids=[c[0] for c in EDGES])
def test_edge_shapes(case, capsys):
    name, heads, kv_len, sparse, n_valid = case
    block_size, block_num = 128, max(64, kv_len // 128 + 8)
    query, cache, si, bt, meta = build_inputs(
        heads, kv_len, sparse, block_size, block_num
    )

    if n_valid < sparse:
        # Indices out of [0, n_valid), ascending (the A5 path sorts them, see
        # vllm-ascend sfa_v1.py:176-178), then -1 padding.
        g = torch.Generator().manual_seed(11)
        ids = torch.randperm(max(n_valid, 1), generator=g)[:n_valid].to(torch.int32)
        padded = torch.full((1, 1, sparse), -1, dtype=torch.int32)
        if n_valid:
            padded[0, 0, :n_valid] = ids.sort().values
        si = padded.npu()
        meta = dict(meta, idx=padded[0, 0, :n_valid])
    seq_kv = max(n_valid, 1)

    got = kv_quant_sparse_flash_attention(
        query, cache, si, bt, [seq_kv], scale_value=SCALE_VALUE
    )[0]
    got = got.cpu().float()
    assert not got.isnan().any(), f"{name}: nan in the output"

    if n_valid == 0:
        # The official reference expects all zeros for an empty selection. The
        # vendor returns nan for this input, so it is not compared here.
        print(f"  {name}: max|out| {got.abs().max().item():.3g} (expected 0)")
        assert got.abs().max().item() == 0.0
        return

    base = vendor(query, cache, si, bt, seq_kv)[0].cpu().float()
    r = max_rel(got, base)
    print(f"  {name}: triton vs vendor max_rel {r:.3e}")
    assert r <= TOL, capsys.readouterr().out


# The fp32 dequantisation path needs two [BLOCK_S, 512] fp32 temporaries, which
# only fits UB at a small block_h; the bf16 paths are what the default picks.
_DEQ_PATHS = {
    "deq3d": dict(deq3d=True),
    "bf16_expanded": dict(deq3d=False),
    "fp32": dict(deq_f32=True, block_h=16),
}


@pytest.mark.kv_quant_sparse_flash_attention
@pytest.mark.parametrize("path", sorted(_DEQ_PATHS))
def test_dequant_paths(path, capsys):
    """All three dequantisation paths answer the vendor within tolerance.

    fp8_e4m3 -> bf16 is lossless, so the bf16 paths only add the <=2^-8
    rounding of the scale, the same order as rounding the fp32 product.
    """
    query, cache, si, bt, meta = build_inputs(64, 4096, 2048, 128, 64)
    got = kv_quant_sparse_flash_attention(
        query, cache, si, bt, [4096], scale_value=SCALE_VALUE, **_DEQ_PATHS[path]
    )[0]
    got = got.cpu().float()
    base = vendor(query, cache, si, bt, 4096)[0].cpu().float()
    r_base, r_ref = max_rel(got, base), max_rel(got, torch_ref(meta))
    print(f"  {path}: vs vendor {r_base:.3e}   vs fp32 ref {r_ref:.3e}")
    assert not got.isnan().any(), capsys.readouterr().out
    assert r_base <= TOL and r_ref <= TOL, capsys.readouterr().out


@pytest.mark.kv_quant_sparse_flash_attention
def test_single_split_skips_combine(capsys):
    """``splits=1`` writes the normalised result straight out of the main kernel.

    The planner reaches it on its own once the sparse size equals one tile.
    """
    query, cache, si, bt, meta = build_inputs(16, 2048, 32, 128, 64)
    got = kv_quant_sparse_flash_attention(
        query, cache, si, bt, [2048], scale_value=SCALE_VALUE
    )[0]
    got = got.cpu().float()
    base = vendor(query, cache, si, bt, 2048)[0].cpu().float()
    r_base, r_ref = max_rel(got, base), max_rel(got, torch_ref(meta))
    print(f"  splits=1: vs vendor {r_base:.3e}   vs fp32 ref {r_ref:.3e}")
    assert not got.isnan().any(), capsys.readouterr().out
    assert r_base <= TOL and r_ref <= TOL, capsys.readouterr().out


@pytest.mark.kv_quant_sparse_flash_attention
def test_unsupported_inputs():
    """Unsupported shapes raise instead of guessing."""
    query, cache, si, bt, _ = build_inputs(16, 2048, 2048, 128, 64)

    with pytest.raises(ValueError, match="bf16"):
        kv_quant_sparse_flash_attention(query.float(), cache, si, bt, [2048])
    with pytest.raises(ValueError, match="int32 or int64"):
        kv_quant_sparse_flash_attention(query, cache, si, bt, torch.ones(1, device=DEV))
    with pytest.raises(ValueError, match="block_h"):
        kv_quant_sparse_flash_attention(query, cache, si, bt, [2048], block_h=24)
    with pytest.raises(NotImplementedError, match="power-of-two"):
        kv_quant_sparse_flash_attention(query, cache, si[..., :1536], bt, [2048])
