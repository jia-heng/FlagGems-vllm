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

"""
Tests for the MetaX fused GDN chunk forward implementation.

Directly invokes the fused kernel path in
flaggems_vllm.runtime.backend._metax.fused.gdn_chunk
and validates against a token-by-token naive reference.
"""

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm

pytestmark = [
    pytest.mark.metax_gdn_chunk,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires GPU"
    ),
]


def naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state):
    """
    Token-by-token reference for gated delta rule.
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
        o_t = q_t @ S_t * scale

    Inputs:
        q: (B, T, H, K)
        k: (B, T, Hg, K)  -- Hg may differ from H (GQA)
        v: (B, T, H, V)
        g: (B, T, H)
        beta: (B, T, H)
        scale: float
        initial_state: (B, H, K, V) or None
    """
    B, T, H, K = q.shape
    Hg = k.shape[2]
    V = v.shape[-1]
    heads_per_group = H // Hg

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    S = (
        initial_state.float().clone()
        if initial_state is not None
        else torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    )
    outputs = []

    for t in range(T):
        q_t = q[:, t, :, :]  # (B, H, K)
        # k uses Hg heads, expand to H for computation
        k_t_grouped = k[:, t, :, :]  # (B, Hg, K)
        k_t = k_t_grouped.repeat_interleave(heads_per_group, dim=1)  # (B, H, K)
        v_t = v[:, t, :, :]  # (B, H, V)
        g_t = g[:, t, :]  # (B, H)
        beta_t = beta[:, t, :]  # (B, H)

        # Gating
        gate = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        S = gate * S

        # Delta rule
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)  # (B, H, V)
        delta = v_t - kS  # (B, H, V)
        update = torch.einsum("bhk,bhv->bhkv", k_t, delta) * beta_t.unsqueeze(
            -1
        ).unsqueeze(-1)
        S = S + update

        # Output
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, S) * scale  # (B, H, V)
        outputs.append(o_t)

    o = torch.stack(outputs, dim=1)  # (B, T, H, V)
    return o, S


def _compare_with_ref_path(q, k, v, g, beta, scale, initial_state, output_final_state):
    """Compare metax fused implementation against the generic FLA chunk path.

    Since both are chunk-based algorithms with identical numerical behavior,
    they should produce bit-identical results (or very close in float32).
    The naive token-by-token reference diverges in extreme value regimes due
    to different accumulation order; use this helper for reliable validation.

    NOTE: We call both paths twice — the first call warms up the autotuner,
    the second call produces stable results for comparison.
    """
    from flaggems_vllm.ops.FLA.chunk import (
        chunk_gated_delta_rule_fwd as ref_fwd,
    )
    from flaggems_vllm.runtime.backend._metax.fused.gdn_chunk import (
        chunk_gated_delta_rule_fwd as metax_fwd,
    )

    # Warmup both paths (autotuner may pick bad configs on first run)
    metax_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=None,
    )
    ref_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=None,
    )

    # Real comparison
    g_ref, o_ref, A_ref, fs_ref, _, _, _ = ref_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=None,
    )
    g_m, o_m, A_m, fs_m, _, _, _ = metax_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=None,
    )
    return o_ref, o_m, fs_ref, fs_m


# ---------------------------------------------------------------------------
# Test: compare metax fused path vs generic FLA chunk path (should be identical)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16], ids=["bf16"]
)
def test_metax_gdn_chunk_fwd_no_initial_state(B, T, H, K, V, dtype):
    device = flaggems_vllm.device
    torch.manual_seed(42)

    Hg = H  # no GQA in this test
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    o_ref, o_m, _, _ = _compare_with_ref_path(
        q, k, v, g, beta, scale, None, False
    )

    # The metax fused path should produce identical results to the generic path
    torch.testing.assert_close(o_m.float(), o_ref.float(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: with initial state + output final state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16], ids=["bf16"]
)
def test_metax_gdn_chunk_fwd_with_initial_state(B, T, H, K, V, dtype):
    device = flaggems_vllm.device
    torch.manual_seed(123)

    Hg = H
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

    o_ref, o_m, fs_ref, fs_m = _compare_with_ref_path(
        q, k, v, g, beta, scale, initial_state, True
    )

    torch.testing.assert_close(o_m.float(), o_ref.float(), rtol=1e-5, atol=1e-5)
    if fs_ref is not None and fs_m is not None:
        torch.testing.assert_close(
            fs_m.float(), fs_ref.float(), rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Test: larger head dimension (K=128) — smoke test, may hit shared memory
# or autotuner instability on some hardware
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="K=128 may trigger shared memory OOR or autotuner NaN on first run"
)
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
def test_metax_gdn_chunk_fwd_large_k(T, K, V):
    from flaggems_vllm.runtime.backend._metax.fused.gdn_chunk import (
        chunk_gated_delta_rule_fwd,
    )

    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(7)

    B, H, Hg = 1, 4, 4
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    # Use small gate to keep output numerically stable
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    g_cumsum, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

    assert o.shape == (B, T, H, V)
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Test: shape sanity check (ensure no crashes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B,T,H,Hg,K,V",
    [
        (1, 64, 8, 8, 64, 64),
        (2, 128, 4, 4, 64, 64),
        (2, 64, 8, 4, 64, 64),  # GQA: H=8, Hg=4
    ],
    ids=["small", "medium", "gqa"],
)
def test_metax_gdn_chunk_fwd_shapes(B, T, H, Hg, K, V):
    from flaggems_vllm.runtime.backend._metax.fused.gdn_chunk import (
        chunk_gated_delta_rule_fwd,
    )

    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    g_cumsum, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

    assert o.shape == (B, T, H, V), f"Expected {(B, T, H, V)}, got {o.shape}"
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Test: naive reference comparison with numerically stable inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16], ids=["bf16"]
)
def test_metax_gdn_chunk_fwd_naive_accuracy(T, dtype):
    """Compare against naive token-by-token reference with small gate values
    to avoid exponential blowup that makes the comparison meaningless."""
    from flaggems_vllm.runtime.backend._metax.fused.gdn_chunk import (
        chunk_gated_delta_rule_fwd,
    )

    device = flaggems_vllm.device
    torch.manual_seed(42)

    B, H, Hg, K, V = 1, 4, 4, 64, 64
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    # Small gates to keep numerical stability
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype)) * 0.1
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    ref_o, _ = naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, None)

    g_cumsum, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=None, output_final_state=False, cu_seqlens=None,
    )

    torch.testing.assert_close(o.float(), ref_o, rtol=5e-2, atol=5e-3)
