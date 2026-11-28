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

"""Benchmark for the A5 sparse MLA-absorb attention over an fp8 KV cache.

gems op: ``flaggems_vllm.kv_quant_sparse_flash_attention`` (A5 registrar entry).
vendor baseline: ``torch_npu.npu_kv_quant_sparse_flash_attention`` in the decode
shape vllm-ascend calls it with
(``_execute_kv_quant_sparse_flash_attention_vllm``, ``device_op.py:634``).
SpeedUp = latency_vendor / latency_flaggems (repo convention).

  ASCEND_RT_VISIBLE_DEVICES=7 pytest -q \\
      benchmark/test_kv_quant_sparse_flash_attention.py

``--mode kernel`` is the default and the only mode whose SpeedUp describes this
operator: device kernel time, summed over every kernel one call launches. Host
launch overhead cannot be optimized and is gone once the model runs as a
captured graph, so the host-timed modes measure something deployment never
pays. They still run through the common runner, but read the kernel-mode rows.

The shared ``do_bench_npu`` cannot produce that number. It needs a constant
kernel count per iteration and an exact multiple of them in the trace, while
one call launches the main kernel plus the combine kernel, and the vendor
operator has its own set.

``benchmark/bench_kv_quant_sparse_flash_attention.py`` stays alongside this
file for the per-kernel breakdown and the tiling knobs, which this runner does
not expose; ``device_time_us`` below is the reader from that script, including
its discard-one-measurement-per-call rule.
"""

import os

# TMPDIR must point to /tmp before importing torch_npu, or the NPU profiler
# cannot write kernel_details.csv and the device time silently comes back None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
torch_npu = pytest.importorskip("torch_npu")

import flaggems_vllm  # noqa: E402
from tests.test_kv_quant_sparse_flash_attention import (  # noqa: E402
    SCALE_VALUE,
    TOL,
    build_inputs,
    max_rel,
    vendor_kwargs,
)

from . import base  # noqa: E402
from .bench_kv_quant_sparse_flash_attention import device_time_us  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import BenchMode  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    or not torch.npu.is_available()
    or not hasattr(flaggems_vllm, "kv_quant_sparse_flash_attention"),
    reason="kv_quant_sparse_flash_attention is registered on Ascend 950 (A5) only",
)

# The two shapes the operator is called with, with the share of model time the
# deployment profile measures for it. They differ only in kv_len and the number
# of cache blocks.
SHAPES = [
    (64, 8192, 2048, 128, 4194),  # GLM-W4, 18.81%
    (64, 4096, 2048, 128, 8778),  # GLM-W8, 17.25%
]

# This machine is shared and the measurement is noisy, so each latency is the
# minimum of several rounds: interference only ever makes a round slower.
_ROUNDS = 3


class KvQuantSparseFlashAttentionBenchmark(base.GenericBenchmark):
    """Use the common benchmark runner with the fixed deployment shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "Q_N, kv_len, sparse, block_size, block_num"

    def get_latency(self, op, *args, **kwargs):
        """Device kernel time per call, summed over the kernels of one call.

        Device time is the metric: host launch overhead is not optimizable and
        does not exist once the model runs as a captured graph.
        ``do_bench_npu`` cannot report it here because one call launches the
        main kernel and the combine kernel, which leaves the trace row count
        short of an exact multiple of the iteration count and makes it answer
        ``inf``. Summing the profiler's kernel durations keeps the combine
        kernel, and the vendor operator's own extra kernels, inside both
        numbers.
        """
        if Config.mode != BenchMode.KERNEL:
            return super().get_latency(op, *args, **kwargs)
        # The common runner's warmup/repetition are durations in other modes;
        # here they bound the iteration counts, as on the other NPU benchmarks.
        warmup = min(int(Config.warm_up), 5)
        iters = min(int(Config.repetition), 20)
        total_us, _ = device_time_us(
            lambda: op(*args, **kwargs), warmup=warmup, iters=iters, rounds=_ROUNDS
        )
        if total_us is None:
            raise RuntimeError("the NPU profiler produced no kernel_details.csv")
        return total_us / 1e3


def _qsfa_vendor(query, cache, sparse_indices, block_table, seq_kv, *, seq_q=None):
    """Vendor baseline. The sequence lengths arrive as ready device tensors, so
    that no host-to-device copy lands inside the measured call."""
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        **vendor_kwargs(query, cache, sparse_indices, block_table, seq_kv, seq_q)
    )


def _qsfa_gems(query, cache, sparse_indices, block_table, seq_kv, *, seq_q=None):
    """Production defaults: the tiling ``plan_tiling`` picks for the shape, the
    bf16 3-D dequantisation path and bf16 partial sums."""
    _ = seq_q
    return flaggems_vllm.kv_quant_sparse_flash_attention(
        query,
        cache,
        sparse_indices,
        block_table,
        seq_kv,
        scale_value=SCALE_VALUE,
    )


def _qsfa_input_fn(shape, dtype, device):
    """Yield one deployment shape, after checking both sides agree.

    ``build_inputs`` writes the 656-byte cache slot segment by segment and
    fills ``block_table`` with a random permutation of physical blocks: pages
    are scattered in real serving, and a contiguous arange would measure the
    discrete gather away. ``dtype`` is the query dtype, which the operator
    fixes at bf16.
    """
    _ = device
    if dtype != torch.bfloat16:
        raise ValueError(f"query is bf16 in this operator, not {dtype}")
    heads, kv_len, sparse, block_size, block_num = shape
    query, cache, si, bt, _ = build_inputs(heads, kv_len, sparse, block_size, block_num)
    seq_kv = torch.tensor([kv_len], dtype=torch.int32, device=query.device)
    seq_q = torch.tensor([1], dtype=torch.int32, device=query.device)
    args = (query, cache, si, bt, seq_kv)
    kwargs = {"seq_q": seq_q}

    # A row whose output does not match the vendor is not timing the same
    # computation. The tolerance is the bf16 output of a softmax-weighted sum
    # over `sparse` dequantised fp8 rows; the accuracy test additionally pins
    # both sides against an fp32 reference.
    got = _qsfa_gems(*args, **kwargs)
    base = _qsfa_vendor(*args, **kwargs)
    rel = max_rel(got.float(), base.float())
    del got, base
    assert rel <= TOL, (
        f"Q_N={heads} kv_len={kv_len} sparse={sparse}: max_rel {rel:.3e} "
        f"against the vendor operator exceeds {TOL:.0e}"
    )
    yield (*args, kwargs)


@pytest.mark.kv_quant_sparse_flash_attention
def test_kv_quant_sparse_flash_attention_perf():
    bench = KvQuantSparseFlashAttentionBenchmark(
        input_fn=_qsfa_input_fn,
        op_name="kv_quant_sparse_flash_attention",
        torch_op=_qsfa_vendor,
        gems_op=_qsfa_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
