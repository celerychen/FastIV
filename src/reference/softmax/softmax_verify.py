#!/usr/bin/env python3
# 验证脚本（非 C 交付物，仅用于在本沙箱无编译器时验证算法数学正确性）。
# 它用纯 Python 复刻三个 C 函数的计算逻辑，并与 double 参考实现比对。
import math
import random

# ---------- 复刻 C 的 exp256_ps 多项式近似（用于验证 AVX2 版精度） ----------
def exp256_ps(x):
    HI, LO = 88.3762626647949, -88.3762626647949
    if x > HI: x = HI
    if x < LO: x = LO
    LOG2EF = 1.44269504088896341
    fx = x * LOG2EF + 0.5
    n = int(fx)            # cvttps_epi32: 向零截断
    fx = float(n)
    C1 = 0.693359375
    C2 = -2.12194440e-4
    z = x - fx * C1
    z = z - fx * C2
    y = 1.9875691500E-4
    y = y * z + 1.3981999507E-3
    y = y * z + 8.3334519073E-3
    y = y * z + 4.1665795894E-2
    y = y * z + 1.6666665459E-1
    y = y * z + 5.0000001201E-1
    y = y * z
    y = y * z
    y = y + z
    y = y + 1.0
    y = y * (2.0 ** n)
    return y

# ---------- 复刻三个 C 函数（原地语义的 Python 版） ----------
def softmax_basic(x):
    n = len(x)
    if n == 0: return []
    m = max(x)
    s = 0.0
    y = [0.0] * n
    for i in range(n):
        y[i] = math.exp(x[i] - m); s += y[i]
    for i in range(n):
        y[i] /= s
    return y

def softmax_online(x):
    n = len(x)
    if n == 0: return []
    m = x[0]; d = 1.0
    for i in range(1, n):
        if x[i] > m:
            scale = math.exp(m - x[i]); d = d * scale + 1.0; m = x[i]
        else:
            d += math.exp(x[i] - m)
    inv = 1.0 / d
    return [math.exp(x[i] - m) * inv for i in range(n)]

def softmax_avx2(x):
    n = len(x)
    if n == 0: return []
    m = max(x)
    s = 0.0
    y = [0.0] * n
    for i in range(n):
        e = exp256_ps(x[i] - m); y[i] = e; s += e
    for i in range(n):
        y[i] /= s
    return y

def softmax_ref(x):
    n = len(x)
    if n == 0: return []
    m = max(x)
    s = 0.0
    y = [0.0] * n
    for i in range(n):
        e = math.exp(x[i] - m); y[i] = e; s += e
    return [e / s for e in y]

# ---------- 测试 ----------
def check_case(label, x):
    ref = softmax_ref(x)
    for name, fn, tol in [
        ("basic", softmax_basic, 1e-6),
        ("online", softmax_online, 1e-6),
        ("avx2 ", softmax_avx2, 1e-3),
    ]:
        if len(x) == 0:
            fn(x[:])  # 仅验证不崩溃
            print(f"  PASS {name} {label} (n=0 no-crash)")
            continue
        got = fn(x[:])
        maxd = max(abs(g - r) for g, r in zip(got, ref))
        s = sum(got)
        ok = (maxd <= tol) and (abs(s - 1.0) < 1e-5)
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {name} {label}: maxdiff={maxd:.2e} sum={s:.6f}")
        if not ok:
            global g_fail
            g_fail = True

g_fail = False
random.seed(42)
for n in [1, 7, 256, 4096, 65536]:
    x = [random.uniform(-10, 10) for _ in range(n)]
    check_case(f"random n={n}", x)

check_case("all equal=3.0", [3.0] * 1000)
check_case("large pos~1000", [1000.0 + random.uniform(-0.1, 0.1) for _ in range(500)])
check_case("large neg~-1000", [-1000.0 + random.uniform(-0.1, 0.1) for _ in range(500)])
check_case("mixed +/-50", [50.0 if i % 2 == 0 else -50.0 for i in range(2000)])
check_case("empty", [])

print("\n==== RESULT:", "ALL PASS ====" if not g_fail else "FAIL ====")
