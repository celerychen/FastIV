# P0 回归基线（yolo26 端口起点，2026-09-05 记录）

> 运行方式（从 `build/` 目录）：`make -C build <test> && (cd build && ./<test>)`
> 其中 `make test_nn test_nn_conv2d test_nn_mnist test_nn_mnist_conv test_face_api
>  test_landmark_ops test_landmark_net test_landmark_api test_landmark_e2e
>  test_yunet_api test_yunet_resize`

之后每次改动 `src/nn/` 或 `api/` 后，**重跑同一组，逐项与下表比对**，任何新增失败都必须查清。

| 测试 | pass | fail | exit | 备注 |
| --- | --- | --- | --- | --- |
| test_nn | 71 | 0 | 0 | |
| test_nn_conv2d | 658 | 0 | 0 | |
| test_nn_mnist | 23 | 0 | 0 | |
| test_nn_mnist_conv | 13 | 0 | 0 | |
| test_face_api | 14 | 0 | 0 | |
| test_landmark_ops | — | 0 | 0 | 打印 `all PASS`，无计数值 |
| test_landmark_net | — | 0 | 0 | 输出 `out_lm.raw`，无 FAIL |
| test_landmark_api | 9 | 1 | 1 | **已知历史问题**：`.lm_truth/` 参考数据缺失，非回归 |
| test_landmark_e2e | 16 | 0 | 0 | `RESULT: ALL PASS` |
| test_yunet_api | 16 | 0 | 0 | |
| test_yunet_resize | — | 0 | 0 | 输出 FPS，无 FAIL |

**基线结论**：除 `test_landmark_api` 的 1 个已知失败外，其余全部通过。

## 真值基线（gen_ref.py）

- 确定性：连跑两次 `--size 64,64 --nc 80`，`ref/MD5SUMS.txt` **完全一致**。
- 内部自检（脚本内 assert）：
  - SPPF replay（`module` 作用于上一层输出）`max_abs_err = 0`；
  - 两个 Attention（attn0/attn1）逐步版 forward vs 原 forward `max_abs_err = 0`。
- 产物：75 个张量 + 594 个权重 blob（2591848 floats）；
  64×64 输入下 `detect_out` 形状 `(1, 84, 6)`。
