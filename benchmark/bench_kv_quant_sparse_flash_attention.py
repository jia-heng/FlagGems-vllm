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

"""Single-shape debug script for the A5 sparse fp8-KV MLA attention.

Default shape (GLM-W4, 18.81% of model time, the first hotspot of that model):

  query          [1, 64, 576]        bf16        TND
  kv_cache       [4194, 128, 1, 656] fp8_e4m3fn  PA_BSND, block_size=128
  sparse_indices [1, 1, 2048]        int32
  block_table    [1, 64]             int32       kv_len = 8192
  out            [1, 64, 512]        bf16

Usage:
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=.:src python \\
      benchmark/bench_kv_quant_sparse_flash_attention.py
  ... --shape glm_w8
  ... --splits 32 --block-h 32

This is the iteration script: one deployment shape, per-kernel breakdown, and
the tiling knobs on the command line.  ``benchmark/test_kv_quant_sparse_flash_attention.py``
is the repo-standard runner over both shapes.

Measurement:

  * Device kernel time only, summed over the kernels of one call (profiler
    ``kernel_details.csv``).  Host time inflates the gap and disappears once
    the model runs as a captured graph.
  * The first profiler measurement of any function is not trustworthy, even
    when the function is warm and even when the process has already profiled
    something else: the same vendor baseline measures 89.61 us on its first
    measurement in a fresh process and 47.8-48.1 us from then on.  So
    ``device_time_us`` throws away one measurement per call.  Discarding once
    per process is not enough: whichever side is measured first then spends the
    allowance, and the other one gets the inflated number.
  * Several rounds, keeping the minimum rather than the median.  Interference
    only ever makes a measurement slower, so the minimum is the clean lower
    bound; the vendor baseline has shown a 44.78 us spread over five rounds
    (median 48.24), which a median does not survive.  The median and the
    spread are printed alongside.
"""

import argparse
import csv
import glob
import os
import shutil
import statistics
import sys
import tempfile

# TMPDIR must point to /tmp before importing torch_npu; otherwise the profiler
# cannot write kernel_details.csv and device_time_us silently returns None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

# The input construction, the vendor call and the fp32 reference live in the
# accuracy test, so both scripts measure exactly the same cache layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flaggems_vllm.runtime.backend._ascend.a5.ops.kv_quant_sparse_flash_attention import (  # noqa: E402
    NOPE,
    ROPE,
    SLOT,
    kv_quant_sparse_flash_attention,
)
from tests.test_kv_quant_sparse_flash_attention import (  # noqa: E402
    SCALE_VALUE,
    SHAPES,
    build_inputs,
    max_rel,
    torch_ref,
    vendor_kwargs,
)

_DEQ = {
    "3d": dict(deq3d=True),
    "bf16": dict(deq3d=False),
    "f32": dict(deq_f32=True),
}


def device_time_us(fn, warmup=5, iters=20, rounds=1, _settle=True):
    """Device kernel time per call in us, as ``(min_total, {kernel: us})``.

    The returned breakdown belongs to the round that produced the minimum.
    See the module docstring for why one measurement is discarded per call and
    why rounds are reduced with ``min``.
    """
    if _settle:
        device_time_us(fn, warmup=2, iters=3, _settle=False)

    totals, breakdowns = [], []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()

        out_dir = tempfile.mkdtemp(prefix="qsfa_prof_")
        try:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
                record_shapes=False,
            ):
                for _ in range(iters):
                    fn()
                torch.npu.synchronize()

            cands = glob.glob(
                os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
            )
            if not cands:
                return None, {}
            per_kernel = {}
            with open(cands[0]) as f:
                for row in csv.DictReader(f):
                    name = row.get("Name") or row.get("Op Name") or "?"
                    dur = (
                        row.get("Duration(us)")
                        or row.get("Duration (us)")
                        or row.get("Duration")
                    )
                    if dur is None:
                        continue
                    try:
                        per_kernel[name] = per_kernel.get(name, 0.0) + float(dur)
                    except ValueError:
                        continue
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        totals.append(sum(per_kernel.values()) / iters)
        breakdowns.append({k: v / iters for k, v in per_kernel.items()})

    best = min(range(len(totals)), key=totals.__getitem__)
    return totals[best], breakdowns[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="glm_w4", choices=sorted(SHAPES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="rounds per side, reduced with min; one round can be off by 15%%",
    )
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--block-h", type=int, default=None)
    ap.add_argument("--block-s", type=int, default=None)
    ap.add_argument("--combine-h", type=int, default=None)
    ap.add_argument("--block-sp", type=int, default=None)
    ap.add_argument("--deq", default="3d", choices=sorted(_DEQ))
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--part-f32", action="store_true", help="fp32 partial sums")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    cfg = SHAPES[args.shape]
    heads, kv_len = cfg["heads"], cfg["kv_len"]
    sparse, block_size, block_num = (
        cfg["sparse"],
        cfg["block_size"],
        cfg["block_num"],
    )
    print(f"== {cfg['name']} ==")
    print(
        f"   query[1,{heads},{NOPE + ROPE}] bf16 · cache[{block_num},{block_size},"
        f"1,{SLOT}] fp8 · sparse[1,1,{sparse}] · kv_len={kv_len} "
        f"-> out[1,{heads},{NOPE}] bf16"
    )

    query, cache, si, bt, meta = build_inputs(
        heads, kv_len, sparse, block_size, block_num
    )
    kw = vendor_kwargs(query, cache, si, bt, kv_len)
    print(
        f"   cache {cache.numel() / 2**20:.0f} MB, block_table is a random "
        f"permutation (pages are scattered in real serving)"
    )

    def vendor_fn():
        return torch_npu.npu_kv_quant_sparse_flash_attention(**kw)

    tkw = dict(
        splits=args.splits,
        block_h=args.block_h,
        block_s=args.block_s,
        combine_h=args.combine_h,
        block_sp=args.block_sp,
        num_cores=args.cores,
        part_bf16=not args.part_f32,
        **_DEQ[args.deq],
    )

    def triton_fn():
        return kv_quant_sparse_flash_attention(
            query,
            cache,
            si,
            bt,
            kw["actual_seq_lengths_kv"],
            scale_value=SCALE_VALUE,
            **tkw,
        )

    base = vendor_fn()[0].cpu().float()
    ref = torch_ref(meta)
    print(
        f"\n[accuracy] vendor vs fp32 ref max_rel {max_rel(base, ref):.3e}"
        f"   <- the vendor's own error, the bar to match"
    )
    if not args.baseline_only:
        got = triton_fn()[0].cpu().float()
        r_base, r_ref = max_rel(got, base), max_rel(got, ref)
        print(f"           triton vs vendor   max_rel {r_base:.3e}")
        print(f"           triton vs fp32 ref max_rel {r_ref:.3e}")
        ok = r_ref <= 2e-2 and not got.isnan().any()
        print(f"           {'PASS' if ok else 'FAIL'} (max_rel <= 2e-2)")

    print(f"\n[perf] device kernel time, {args.rounds} rounds, min")
    vt = [device_time_us(vendor_fn, iters=args.iters)[0] for _ in range(args.rounds)]
    t_vendor = min(vt)
    if not args.baseline_only:
        res = [device_time_us(triton_fn, iters=args.iters) for _ in range(args.rounds)]
        tt = [r[0] for r in res]
        t_triton = min(tt)
        kernels = res[tt.index(t_triton)][1]
        main_us = sum(v for k, v in kernels.items() if "combine" not in k)
        comb_us = sum(v for k, v in kernels.items() if "combine" in k)
        print(
            f"  triton {t_triton:8.2f} us   (median {statistics.median(tt):.2f}, "
            f"spread {max(tt) - min(tt):.2f})   main {main_us:.2f} + combine "
            f"{comb_us:.2f}"
        )
        for name, us in sorted(kernels.items(), key=lambda kv: -kv[1]):
            print(f"         {us:8.2f} us  {name}")
    print(
        f"  vendor {t_vendor:8.2f} us   (median {statistics.median(vt):.2f}, "
        f"spread {max(vt) - min(vt):.2f})"
    )
    if not args.baseline_only:
        print(f"  SpeedUp {t_vendor / t_triton:.3f}x")


if __name__ == "__main__":
    main()
