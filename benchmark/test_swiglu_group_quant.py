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

"""Benchmark for the A5 SwiGLU group-quant operator.

gems op: ``flaggems_vllm.swiglu_group_quant`` (A5 registrar entry).
vendor baseline: ``aclnnSwigluGroupQuant`` through
``torch.ops.cann_ops_nn.swiglu_group_quant``, which needs the CANN
``cann_ops_nn`` package on PYTHONPATH; the module skips without it.
SpeedUp = latency_vendor / latency_flaggems (repo convention).

  ASCEND_RT_VISIBLE_DEVICES=7 pytest -q benchmark/test_swiglu_group_quant.py

Both sides measure the production MX configuration: BF16 input, per-32 MX
dynamic quantization to FP8_E4M3FN with a rounded E8M0 scale, no weight, no
group_index, no clamp.  The other modes are covered by
``tests/test_swiglu_group_quant.py``.

``--mode kernel`` is the default and the one to read: device kernel time from
the shared ``do_bench_npu``, which works here because either side launches
exactly one kernel per call.  Host launch overhead cannot be optimized and is
gone once the model runs as a captured graph, so the host-timed modes measure
something deployment never pays; on the small rows they would report the
Python wrapper, not the kernel.

``benchmark/bench_swiglu_group_quant.py`` stays alongside this file: it sweeps
the full shape list with a pre-bound launcher, which this runner does not do.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_nn")

import flaggems_vllm  # noqa: E402
from tests.test_swiglu_group_quant import (  # noqa: E402
    _check_bytes,
    _check_scale,
    _vendor,
)

from . import base  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    or not torch.npu.is_available()
    or not hasattr(flaggems_vllm, "swiglu_group_quant"),
    reason="swiglu_group_quant is registered on Ascend 950 (A5) only",
)

# (T, D): T tokens of a D-wide gate|up activation, so the output is [T, D/2].
# Decode (T=1..64) and prefill (T>=256) at the common hidden width, plus two
# other widths from the full sweep in bench_swiglu_group_quant.py.
SHAPES = [
    (1, 4096),
    (16, 4096),
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (4096, 4096),
    (4096, 2560),
    (4096, 14336),
]

# The production MX path: quant_mode=1 rounds the E8M0 scale, which is what the
# MXFP8 model config asks for.
_MX = dict(
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    block_size=0,
    round_scale=True,
    clamp_limit=-1.0,
    output_origin=False,
)


class SwigluGroupQuantBenchmark(base.GenericBenchmark):
    """Use the common benchmark runner with the fixed deployment shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "T, D"


def _swiglu_group_quant_vendor(x):
    """Vendor baseline in the form the accuracy test calls it."""
    return _vendor(x, None, None, **_MX)


def _swiglu_group_quant_gems(x):
    """Production defaults; the tile is chosen from the shape by the wrapper."""
    return flaggems_vllm.swiglu_group_quant(x, None, None, **_MX)


def _swiglu_group_quant_input_fn(shape, dtype, device):
    """Yield one activation tensor per shape, checked against the vendor first.

    The checks are the accuracy test's own: output bytes must match except for
    a bounded rate of exact FP8 ties, and the E8M0 scale bytes must match
    exactly.  A row that fails them is not timing the same computation.
    """
    T, D = shape
    torch.manual_seed(0)
    x = torch.randn(T, D, device=device, dtype=dtype) * 3
    ours = _swiglu_group_quant_gems(x)
    expected = _swiglu_group_quant_vendor(x)
    _check_bytes(f"y [{T},{D}]", ours[0], expected[0], fp8=True)
    _check_scale(ours[1], expected[1])
    del ours, expected
    yield (x,)


@pytest.mark.swiglu_group_quant
def test_swiglu_group_quant_perf():
    bench = SwigluGroupQuantBenchmark(
        input_fn=_swiglu_group_quant_input_fn,
        op_name="swiglu_group_quant",
        torch_op=_swiglu_group_quant_vendor,
        gems_op=_swiglu_group_quant_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
