# FastIV SVD 现状报告（2026-08-29）

## 1. 实现概况

- 实现文件：`src/mat/fiv_mat_svd.c`（新增）、测试 `src/test/test_mat_svd.c`（新增）。
- 公开接口：`api/fiv_matrix.h` 中 `fiv_matrix_svd(fiv_mat* mat_a, ivf32* sing_vals, fiv_mat* mat_u, fiv_mat* mat_v)`。
- 已提交本地 git：`9df6e91`「feat(mat): 分块 SVD + 对称特征分解 + real64 GEMM 命名规范」（同一笔提交还含 `fiv_mat_eig_sym` 与 real64 GEMM 命名清理）。

## 2. 算法

- 阻塞式（panel-based）双对角化 + Golub-Kahan QR 迭代，移植自 Eigen 5.0.0 `UpperBidiagonalization`。
- 流程：
  1. 混合双对角化：前导块走 panel（blocked，阈值 48），其余窄/尾块走 unblocked；输出上双对角带，反射器就地打包进矩阵。
  2. 对上双对角带做 Golub-Kahan QR 迭代，求出奇异值并就地更新 U/V。
  3. 由打包的 Householder 反射器重建 U、V。
  4. 负奇异值搬到 U 对应列取反，再降序排序。
- 宽矩阵（`rows < cols`）转置成 `Aᵀ`（tall）后复用同一条上双对角路径，并交换 U/V 角色：`mat_v` 收 `Aᵀ` 的左奇异向量（=A 的右奇异向量），`mat_u` 收 `Aᵀ` 的右奇异向量（=A 的左奇异向量）。

## 3. 已修复的 bug（均已验证）

`test_mat_svd` 当前 **pass=121 fail=0**（修复前 panel 形状 64×64 / 65×70 / 127×100 / 50×200 / 600×300 失败，为 106/15）。三处修复：

1. **panel step-3 `y_k` 计算**：原误用标量 `Σ v_k[t]²` 坍缩；改为长度 col 的向量 `tmp(t)=V_k1ᵀ·v_k`、`tmp2(t)=X_k1ᵀ·v_k`（每元素是 packed 左反射器列 / compact-WY 列与 `v_k` 的内积）。
2. **阻塞驱动恢复**：前序为验证 QR 曾临时改成单次 unblocked，已恢复 panel + unblocked 混合驱动。
3. **右反射器 tau 存储 off-by-one（决定性 bug）**：原代码先在 `if (col+1<bcols)` 块内 `tau_u_prev = tau_u` 再存上一列 tau，导致当前列 tau 右移一列（`tau_p[col]==tau_u[col+1]`）；对照 Eigen `UpperBidiagonalization.h:235-238/243` 修正为「先存上一列 tau 再做 `tau_u_prev=tau_u`」，并补 `else if` 处理最后列 + 尾部 `if (bs<bcols)` 兜底。

构建：`build/` 下 `make test_mat_svd` 零警告。同批提交前实跑：`test_mat_eig_sym` 170/0、`test_mat_mul_db` 96/0、`test_mat_mul` 95/0。

## 4. 内存与缓冲设计（已选 B：统一保留输入，已实现）

- **临时缓冲已是 block 尺寸**：`xmat = work_rows × MAX_BLOCK`、`ymat = work_cols × MAX_BLOCK`（compact-WY，block 维 ≤48）；`diag / superdiag / vec_v / vec_w / vec_p / vec_w2` 是长度 dim/big 的小向量。
- **`work`（被约化的矩阵本体）必须全程保持 full 尺寸**：打包的 Householder 反射器就存在其中，bidiag → QR 迭代 → 重建 U/V 全程依赖它，不能缩成小块。分块只是 panel-by-panel 消元，矩阵本体一直在。
- **契约（已定，方案 B）**：`fiv_matrix_svd` 现在**始终分配独立 `work` 并拷贝**（tall/square 用 `memcpy` 原样拷贝、wide 经 `fiv_matrix_transpose` 转置拷贝），`work` 不再别名 `mat_a`，**输入 `mat_a` 在所有形状下均不被改动**。测试已加 `input matrix preserved` 断言锁定该契约。
- 代价：tall/square 也比原来多付一次 `O(mn)` 拷贝（相对 `O(mn²)` 的 SVD 可忽略），峰值内存为 `mat_a` + `work` 两份 full 矩阵。
- 备选（未采用）：方案 A 统一 destructive（wide 原地转置别名，删 718 malloc，零额外 full 分配但 tall/wide 都破坏输入）。当初讨论结论——wide 下「零额外 full 分配」与「保留输入」不能兼得。

## 5. 未提交 / 遗留

- 调试产物 `build/bench_vec_dot`、`build/debug_qr`、`build/neon_disasm_analysis.html` 与 `docs/` 未纳入提交（提交时已刻意排除）。

## 6. 下一步建议

- 方案 B 已实现并经测试锁定（`input matrix preserved` 断言）。如需节约内存（牺牲「保留输入」）可改回方案 A；当前实现以安全优先。
- 若后续要把 SVD 接入流水线，注意峰值内存为两份 full 矩阵。
