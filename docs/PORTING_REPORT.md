# GDN Chunk 算子移植报告

## 概述

将 vLLM-metax 中已验证跑通的 GDN (Gated Delta Rule) chunk forward 算子移植到 FlagGems-vllm 的 metax 后端。

- **源文件**: `/data/jianheng/vLLM-metax/vllm_metax/patch/triton_support/gdn_chunk.py`
- **目标文件**: `/data/jianheng/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_metax/ops/gdn_chunk.py`
- **注册入口**: `/data/jianheng/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_metax/ops/__init__.py`

---

## 移植内容

### 1. Fused cumsum + KKT + solve_tril 内核

| 项目 | 说明 |
|------|------|
| 内核函数 | `chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel` |
| Python wrapper | `chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril` |
| 优化点 | 将原本 3 个 kernel launch + 2 次全局内存往返合并为 1 个 kernel |
| 功能 | cumsum(g) → KKT 计算下三角矩阵 L → solve_tril 求逆得到 A_inv |
| 约束 | 仅支持 BT=64（solve_tril 硬编码为 4×4 个 16×16 块） |

### 2. K-major chunk_delta_h 内核

| 项目 | 说明 |
|------|------|
| 内核函数 | `chunk_gated_delta_rule_fwd_kernel_h_kmajor` |
| Python wrapper | `chunk_gated_delta_rule_fwd_h` |
| 优化点 | 状态块保持 `[64, BV]` (K-major) 布局，消除递推循环中所有 `tl.trans` 操作 |
| h 快照布局 | `(B, NT, H, K, V)` — 与通用版的 `(B, NT, H, K, V)` 一致 |
| initial_state / final_state | 保持外部 `[N, H, K, V]` 布局 |

### 3. 配套 chunk_o 内核

| 项目 | 说明 |
|------|------|
| 内核函数 | `chunk_fwd_kernel_o` |
| Python wrapper | `chunk_fwd_o` |
| 说明 | 专为 K-major h 布局设计，直接以 `(K, V)` 读取 h，与上面的 chunk_delta_h 配对使用 |
| 配置空间 | BK/BV ∈ {32, 64}，num_warps ∈ {2, 4, 8}，num_stages ∈ {1, 2} |

### 4. 顶层前向函数

| 项目 | 说明 |
|------|------|
| 函数 | `chunk_gated_delta_rule_fwd` |
| 流水线 | fused_cumsum_kkt_solve_tril → recompute_w_u_fwd → chunk_delta_h (K-major) → chunk_o |
| 注册方式 | 通过 metax 后端 ops 模块自动覆盖通用实现 |

---

## 适配改动

| 方面 | vLLM-metax 原版 | FlagGems-vllm 移植版 |
|------|----------------|---------------------|
| 导入路径 | `vllm.triton_utils`、`vllm.model_executor.layers.fla.ops.*` | `flaggems_vllm.ops.FLA.*` |
| Triton 装饰器 | `@triton.autotune` | `@libentry()` + `@libtuner()` |
| `exp2` 支持 | 条件参数 `USE_EXP2`，支持 `exp2` 和 `exp` 两条路径 | 移除 — 统一使用 `exp()`（与 FlagGems 现有内核保持一致） |
| 生效机制 | Monkey-patch 覆盖 `vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule_fwd` | 通过 `SpecOpRegistrar` 后端 ops 机制自动注册 |
| `num_stages` | `[1]`（MACA 共享内存限制） | 同样限制为 `[1]`（chunk_o 放宽到 `[1, 2]`） |
| Final state 布局 | `[N, H, V, K]`（从 K-major 转置） | `[N, H, K, V]`（匹配 FlagGems 约定，无需额外转置） |
| `recompute_w_u_fwd` 调用 | 传 `chunk_indices`、`use_exp2` 参数 | 去掉这两个参数（FlagGems 版本内部派生 chunk_indices，不支持 exp2） |
| K-loop 配置裁剪 | `_prune_unsafe_kloop_configs` | 保留，防止 `num_warps > 2` + `BK < K` 组合导致错误结果 |

---

## MACA 平台特殊约束

1. **num_stages 限制为 1** — MACA 共享内存容量不足以支持多级流水线（与 vLLM-metax 现有 chunk_delta_h patch 同一约束）
2. **不使用 Triton TLE** — MACA 不支持 TLE (Triton Language Extensions)，因此 TLE 相关路径全部跳过
3. **K-loop miscompile 规避** — 保留 `_prune_unsafe_kloop_configs`，在 K > BK 时禁止 num_warps > 2 的配置

---

## 文件结构

```
src/flaggems_vllm/runtime/backend/_metax/ops/
├── __init__.py      # 导出 chunk_gated_delta_rule_fwd
└── gdn_chunk.py     # 完整的 metax 优化 GDN chunk forward 实现 (994 行)
```

---

## 验证状态

- [x] Python 语法检查通过
- [x] 导入路径与 FlagGems-vllm 模块结构匹配
- [x] 函数签名与上层调用方 (`FLA/chunk.py`) 返回值格式一致（7 元组）
- [ ] 运行时功能测试（需在 MetaX 硬件环境执行）
