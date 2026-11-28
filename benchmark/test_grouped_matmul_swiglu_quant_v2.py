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

"""Benchmark for the A5 MX-FP8 grouped matmul + SwiGLU + MX quant fusion.

gems op: ``flaggems_vllm.grouped_matmul_swiglu_quant_v2`` (A5 registrar entry).
vendor baseline: ``torch_npu.npu_grouped_matmul_swiglu_quant_v2`` with e8m0
scales, called the way vllm-ascend's ``apply_gmm1_act_quant`` calls it on the
MXFP8 path (``w8a8_mxfp8.py:562``).  SpeedUp = latency_vendor /
latency_flaggems (repo convention).

  ASCEND_RT_VISIBLE_DEVICES=7 pytest -q \\
      benchmark/test_grouped_matmul_swiglu_quant_v2.py

``--mode kernel`` is the default and the only mode whose SpeedUp describes this
operator: device kernel time, summed over every kernel one call launches.  Host
launch overhead cannot be optimized and is gone once the model runs as a
captured graph, so the host-timed modes measure something deployment never
pays.  They still run through the common runner, but read the kernel-mode rows.

The shared ``do_bench_npu`` cannot produce that number.  It needs a constant
kernel count per iteration and an exact multiple of them in the trace, while
one call on the default split path is the fused gmm+SwiGLU kernel plus the
MX-quant kernel, and the vendor operator has its own set.

``benchmark/bench_grouped_matmul_swiglu_quant_v2.py`` stays alongside this file
for the per-kernel breakdown and the tiling report, which this runner does not
print; ``device_time_us`` below is the reader from that script.
"""

import os

# TMPDIR must point to /tmp before importing torch_npu, or the NPU profiler
# cannot write kernel_details.csv and the device time silently comes back None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")

import flaggems_vllm  # noqa: E402
from tests.test_grouped_matmul_swiglu_quant_v2 import (  # noqa: E402
    build_inputs,
    scale_bytes,
    vendor,
)

from . import base  # noqa: E402
from .bench_grouped_matmul_swiglu_quant_v2 import device_time_us  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import BenchMode  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    or not torch.npu.is_available()
    or not hasattr(flaggems_vllm, "grouped_matmul_swiglu_quant_v2"),
    reason="grouped_matmul_swiglu_quant_v2 is registered on Ascend 950 (A5) only",
)

# (M, K, N, E) as the operator receives them; N is the concatenated gate|up
# width, so the output is [M, N/2].  The comment is the operator's share of
# model time measured on the deployment profile.
SHAPES = [
    (64, 2048, 6144, 32),  # GLM W8A8C8, 7.37%
    (16, 512, 2048, 128),  # Qwen3.6, 8.22%
]


class GroupedMatmulSwigluQuantV2Benchmark(base.GenericBenchmark):
    """Use the common benchmark runner with the fixed deployment shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "M, K, N, E"

    def get_latency(self, op, *args, **kwargs):
        """Device kernel time per call, summed over the kernels of one call.

        Device time is the metric: host launch overhead is not optimizable and
        does not exist once the model runs as a captured graph.
        ``do_bench_npu`` cannot report it here because the default split path
        launches two kernels per call, which leaves the trace row count short
        of an exact multiple of the iteration count and makes it answer
        ``inf``.  Summing the profiler's kernel durations keeps the MX-quant
        kernel, and the vendor operator's own extra kernels, inside both
        numbers.
        """
        if Config.mode != BenchMode.KERNEL:
            return super().get_latency(op, *args, **kwargs)
        # The common runner's warmup/repetition are durations in other modes;
        # here they bound the iteration counts, as on the other NPU benchmarks.
        warmup = min(int(Config.warm_up), 5)
        iters = min(int(Config.repetition), 30)
        total_us, _ = device_time_us(
            lambda: op(*args, **kwargs), warmup=warmup, iters=iters
        )
        if total_us is None:
            raise RuntimeError("the NPU profiler produced no kernel_details.csv")
        return total_us / 1e3


def _gmm_swiglu_quant_v2_vendor(x, weight, weight_scale, x_scale, group_list, **kwargs):
    """Vendor baseline. It takes the weight as a one-element list."""
    _ = kwargs
    return vendor(x, weight, weight_scale, x_scale, group_list)


def _gmm_swiglu_quant_v2_gems(
    x, weight, weight_scale, x_scale, group_list, *, group_list_type=0
):
    """Production defaults: the OCP scale algorithm (what V2 exposes), the
    two-kernel split path, and the tiling the wrapper picks for the shape."""
    return flaggems_vllm.grouped_matmul_swiglu_quant_v2(
        x,
        weight,
        weight_scale,
        x_scale,
        group_list,
        group_list_type=group_list_type,
    )


def _gmm_swiglu_quant_v2_input_fn(shape, dtype, device):
    """Yield one deployment shape with a cumulative group_list.

    ``build_inputs`` quantises through ``npu_dynamic_mx_quant``, so the scale
    exponents keep the distribution of real inference.  ``dtype`` is the BF16
    the SwiGLU rounds through before quantisation; both outputs are FP8.
    """
    _ = device
    if dtype != torch.bfloat16:
        raise ValueError(f"this operator rounds the SwiGLU through bf16, not {dtype}")
    M, K, N, E = shape
    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    args = (x, w, sb, sa, gl)
    kwargs = {"group_list_type": 0}

    # Both paths must reach the same MX-MAC instruction and the same scale
    # algorithm: a row that is not bitwise equal to the vendor is not timing
    # the same computation.
    valid = int(gl[-1])
    y, ys = _gmm_swiglu_quant_v2_gems(*args, **kwargs)
    vy, vys = _gmm_swiglu_quant_v2_vendor(*args, **kwargs)
    nh = N // 2
    # Compare FP8 as bytes: aclnnNeTensor has no FP8_E4M3FN kernel, and bytes
    # are the strictest comparison anyway.
    bad_y = int(
        (y[:valid].view(torch.uint8) != vy[:valid].view(torch.uint8)).sum().item()
    )
    bad_s = int(
        (scale_bytes(ys, valid, nh) != scale_bytes(vys, valid, nh)).sum().item()
    )
    del y, ys, vy, vys
    assert bad_y == 0 and bad_s == 0, (
        f"[M,K]=[{M},{K}] [E,K,N]=[{E},{K},{N}]: {bad_y} output and {bad_s} scale "
        f"elements differ from the vendor operator"
    )
    yield (*args, kwargs)


@pytest.mark.grouped_matmul_swiglu_quant_v2
def test_grouped_matmul_swiglu_quant_v2_perf():
    bench = GroupedMatmulSwigluQuantV2Benchmark(
        input_fn=_gmm_swiglu_quant_v2_input_fn,
        op_name="grouped_matmul_swiglu_quant_v2",
        torch_op=_gmm_swiglu_quant_v2_vendor,
        gems_op=_gmm_swiglu_quant_v2_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
