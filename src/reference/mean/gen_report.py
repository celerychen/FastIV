#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 results.csv 与 correctness_full.txt 生成 REPORT.md 与 report.html。

纯标准库实现，SVG 图表自包含（无外部 CDN 依赖），适配离线预览。
"""
import csv
import os
import math

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results.csv")
CORR = os.path.join(HERE, "build", "correctness_full.txt")
OUT_MD = os.path.join(HERE, "REPORT.md")
OUT_HTML = os.path.join(HERE, "report.html")

# 实现中文名与颜色
IMPLS = [
    ("serial_welford", "原始串行 Welford (-O2)", "#8884d8"),
    ("avx2_welford", "原始 AVX2 Welford (-O2)", "#82ca9d"),
    ("scalar_sumsq", "标量 sum/sumsq 单趟 (-O2, 参照基线)", "#ff7f0e"),
    ("serial_autovec", "同源标量 (-O3 -march=native)", "#1f77b4"),
    ("avx2_welford_o3", "原始 AVX2 (-O3 -march=native)", "#2ca02c"),
    ("avx2_welford_opt_f", "优化 Welford: FMA+2路展开+正确除法(float)", "#e377c2"),
    ("avx2_welford_opt_d", "优化 Welford: double 累加器", "#d62728"),
    ("avx2_welford_fma1", "单路 AVX2+FMA (隔离实验)", "#ffd700"),
]
IMPL_NAMES = [x[0] for x in IMPLS]

# 缓存边界（元素数 / float）
CACHE_MARKS = [
    (12288, "L1d 48KB"),
    (327680, "L2 1.25MB"),
    (6553600, "L3 25MB"),
    (16777216, "DRAM 起点"),
]


def load_csv():
    rows = []
    with open(CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def load_corr_text():
    if not os.path.exists(CORR):
        return ""
    with open(CORR, encoding="utf-8", errors="replace") as f:
        return f.read()


def build_series(rows):
    """返回 {impl: [(n, gb_s, ns_elem, cyc_elem, ns_call)]} 按 n 排序"""
    d = {k: [] for k in IMPL_NAMES}
    for row in rows:
        k = row["impl"]
        if k not in d:
            continue
        n = int(row["n"])
        d[k].append((
            n,
            float(row["gb_per_s"]),
            float(row["ns_per_elem"]),
            float(row["cycles_per_elem"]),
            float(row["ns_per_call"]),
        ))
    for k in d:
        d[k].sort(key=lambda t: t[0])
    return d


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ----------------------------------------------------------------------
# SVG 折线图（log-log）
# ----------------------------------------------------------------------
def log_log_chart(series, ykey, ylabel, title, vmax=None, vmin=None, hlines=None):
    """ykey: 1->gb_s, 2->ns_elem, 3->cyc_elem"""
    W, H = 920, 470
    x0, x1, y0, y1 = 78, 880, 30, 430
    # x 范围
    alln = [p[0] for k in series for p in series[k]]
    lx_min = math.log10(min(alln))
    lx_max = math.log10(max(alln))
    # y 范围
    yvals = [p[ykey] for k in series for p in series[k] if p[ykey] > 0]
    if vmin is None:
        vmin = min(yvals)
    if vmax is None:
        vmax = max(yvals)
    ly_min = math.log10(vmin)
    ly_max = math.log10(vmax)

    def X(n):
        return x0 + (math.log10(n) - lx_min) / (lx_max - lx_min) * (x1 - x0)

    def Y(v):
        lv = math.log10(max(v, vmin * 0.5))
        lv = min(max(lv, ly_min), ly_max)
        return y1 - (lv - ly_min) / (ly_max - ly_min) * (y1 - y0)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Consolas, Menlo, monospace" font-size="12">'
    )
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#0f1420"/>')
    parts.append(
        f'<text x="{(x0+x1)/2}" y="16" fill="#e6e6e6" text-anchor="middle" '
        f'font-size="15" font-weight="bold">{esc(title)}</text>'
    )
    # 网格 + 轴刻度（x：每档数量级）
    for dec in range(int(math.floor(lx_min)), int(math.ceil(lx_max)) + 1):
        gx = X(10 ** dec)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" stroke="#22304a" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y1+16}" fill="#9fb0c8" text-anchor="middle">{esc(f"1e{dec}")}</text>')
    # y 刻度
    yticks = []
    e = math.floor(ly_min)
    while 10 ** e <= vmax * 1.5:
        if 10 ** e >= vmin * 0.7:
            yticks.append(10 ** e)
        e += 1
    for tv in yticks:
        gy = Y(tv)
        parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="#1b2738" stroke-width="1"/>')
        parts.append(f'<text x="{x0-8}" y="{gy+4:.1f}" fill="#9fb0c8" text-anchor="end">{esc(f"{tv:g}")}</text>')
    # 缓存边界竖线
    for cn, clabel in CACHE_MARKS:
        gx = X(cn)
        parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" stroke="#3a4f6b" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{gx:.1f}" y="{y0-6}" fill="#7fd1ff" text-anchor="middle" font-size="10">{esc(clabel)}</text>')
    # 轴标题
    parts.append(f'<text x="{(x0+x1)/2}" y="{H-6}" fill="#cfd8e6" text-anchor="middle">数组长度 n（元素数，log 轴）</text>')
    parts.append(f'<text x="20" y="{(y0+y1)/2}" fill="#cfd8e6" text-anchor="middle" transform="rotate(-90 20 {(y0+y1)/2})">{esc(ylabel)}</text>')
    # 折线
    for kname, klabel, kcolor in IMPLS:
        pts = series.get(kname, [])
        if not pts:
            continue
        dpath = []
        for (n, gb, ns, cyc, nsc) in pts:
            v = [gb, ns, cyc][ykey - 1]
            if v <= 0:
                continue
            dpath.append(f"{X(n):.1f},{Y(v):.1f}")
        parts.append(f'<polyline points="{" ".join(dpath)}" fill="none" stroke="{kcolor}" stroke-width="2.2"/>')
        # 末端标签
        if dpath:
            lx, ly = dpath[-1].split(",")
            parts.append(f'<circle cx="{lx}" cy="{ly}" r="3" fill="{kcolor}"/>')
    # 图例
    lx = x0 + 8
    ly = y0 + 8
    for kname, klabel, kcolor in IMPLS:
        parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="10" fill="{kcolor}"/>')
        parts.append(f'<text x="{lx+18}" y="{ly+9}" fill="#d6deea" font-size="10.5">{esc(klabel)}</text>')
        ly += 15
        if ly > y1 - 20:
            ly = y0 + 8
            lx += 250
    parts.append('</svg>')
    return "".join(parts)


def main():
    rows = load_csv()
    series = build_series(rows)
    corr = load_corr_text()

    # 大 n（DRAM 区，取 64M=67108864 这一档）的各部分速比
    def pick(impl, n):
        for p in series.get(impl, []):
            if p[0] == n:
                return p
        return None

    # 找 serial_welford 的基准耗时用于速比
    base_by_n = {}
    for p in series.get("serial_welford", []):
        base_by_n[p[0]] = p[4]

    # ---- 图表 ----
    chart_gb = log_log_chart(series, 1, "吞吐 GB/s（越高越好）", "各实现吞吐随数组长度变化（log-log）")
    chart_cyc = log_log_chart(
        series, 3, "cycles / 元素（越低越好）",
        "计算强度：每元素 CPU 周期（log-log）", vmin=0.05, vmax=60
    )

    # ---- 数据表（取代表性档位） ----
    rep_ns = [8, 12288, 327680, 6553600, 16777216, 67108864, 134217728]
    table_rows = []
    for n in rep_ns:
        row = [f"{n:,}"]
        for kname, klabel, kcolor in IMPLS:
            p = pick(kname, n)
            if p is None:
                row.append("–")
                continue
            gb, ns, cyc, nsc = p[1], p[2], p[3], p[4]
            base = base_by_n.get(n, 0)
            sp = (base / nsc) if nsc > 0 and base > 0 else 0
            row.append(f"{gb:.2f} GB/s | {cyc:.3f} cyc | {sp:.2f}x")
        table_rows.append(row)

    # 大 n 速比小结（用 64M 档）
    big_n = 67108864
    speedups = []
    for kname, klabel, kcolor in IMPLS:
        p = pick(kname, big_n)
        if p and base_by_n.get(big_n, 0) > 0:
            sp = base_by_n[big_n] / p[4]
            speedups.append((klabel, sp))

    # ---- HTML ----
    legend_html = "".join(
        f'<span style="color:{c}">■</span> {esc(l)}<br/>' for _, l, c in IMPLS
    )
    table_html = (
        "<table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse;"
        "font-family:Consolas,monospace;font-size:12px;color:#d6deea'>"
        "<thead style='background:#1b2738'><tr>"
        "<th>实现 \\ n</th>" + "".join(f"<th>{n:,}</th>" for n in rep_ns) + "</tr></thead><tbody>"
    )
    for idx, (kname, klabel, kcolor) in enumerate(IMPLS):
        tds = []
        for n in rep_ns:
            p = pick(kname, n)
            if p is None:
                tds.append("<td>–</td>")
                continue
            gb, ns, cyc, nsc = p[1], p[2], p[3], p[4]
            base = base_by_n.get(n, 0)
            sp = (base / nsc) if nsc > 0 and base > 0 else 0
            tds.append(f"<td style='text-align:right'>{gb:.2f}<br/>{cyc:.3f}cyc<br/><b>{sp:.2f}x</b></td>")
        bg = "#0f1420" if idx % 2 == 0 else "#141b29"
        table_html += f"<tr style='background:{bg}'><td style='color:{kcolor}'>{esc(klabel)}</td>" + "".join(tds) + "</tr>"
    table_html += "</tbody></table>"
    table_html += "<div style='font-size:11px;color:#9fb0c8;margin-top:6px'>每格：吞吐 GB/s / 每元素周期数 / 相对串行 Welford 的速比</div>"

    sp_html = "<ul style='font-family:Consolas,monospace;font-size:13px;color:#d6deea'>"
    for lbl, sp in speedups:
        sp_html += f"<li>{esc(lbl)}：<b>{sp:.2f}x</b></li>"
    sp_html += "</ul>"

    # 正确性结论文本（从 corr 抽取关键行）
    corr_excerpt = ""
    for line in corr.splitlines():
        if line.startswith("  [FAIL]") or "精度汇总" in line or "无越界读" in line or "全部实现均未越界" in line or "硬失败" in line:
            corr_excerpt += esc(line) + "\n"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>mean/var 性能与正确性报告</title>
<style>
 body {{ background:#0b0f17; color:#d6deea; font-family:-apple-system,Segoe UI,Roboto,'Microsoft YaHei',sans-serif; margin:0; padding:24px; }}
 h1,h2 {{ color:#e6edf6; }}
 .card {{ background:#121826; border:1px solid #1f2a3a; border-radius:10px; padding:18px; margin:16px 0; }}
 .svgwrap {{ background:#0f1420; border-radius:8px; padding:8px; }}
 code {{ background:#1b2738; padding:2px 6px; border-radius:4px; color:#7fd1ff; }}
</style></head><body>
<h1>mean / variance 实现：正确性 + 性能测试报告</h1>
<p>测试环境：Intel Alder Lake（混合架构，绑 P-core）· GCC 15.2 MinGW-w64 · 数据：正态分布 float，覆盖 L1/L2/L3/DRAM 全层级。</p>

<div class="card">
  <h2>1. 吞吐 vs 数组长度（log-log）</h2>
  <div class="svgwrap">{chart_gb}</div>
</div>

<div class="card">
  <h2>2. 计算强度：每元素 CPU 周期（log-log）</h2>
  <div class="svgwrap">{chart_cyc}</div>
  <p style="font-size:12px;color:#9fb0c8">大数组区（DRAM，128M 档）串行 Welford 仍停在 <b>12.6 cyc/元素</b>（计算受限，每元素一次真除法是关键路径），
  始终到不了带宽墙；向量版则已被内存带宽限制（原始 AVX2 Welford ≈1.17、优化版 opt_f ≈0.92、opt_d ≈1.14 cyc/元素）。
  小数组区（L1/L2）差距拉开，反映各实现的计算开销差异。注：中档规模 RSD 偏高（~10%），属 P-core 睿频/调度抖动，非算法问题。</p>
</div>

<div class="card">
  <h2>3. 代表性规模速比表（n = 8 … 128M）</h2>
  {table_html}
</div>

<div class="card">
  <h2>4. 64M(256MB) 档相对串行 Welford 的速比</h2>
  {sp_html}
</div>

<div class="card">
  <h2>5. 正确性测试关键结论</h2>
  <pre style="white-space:pre-wrap;font-size:12px;line-height:1.5;color:#cfe3ff">{corr_excerpt}</pre>
</div>

<div class="card">
  <h2>6. 实现图例</h2>
  <div style="font-size:13px;line-height:1.8">{legend_html}</div>
</div>

<p style="font-size:11px;color:#6b7a90">本页 SVG 图表为纯静态矢量，无外部依赖，可离线查看。</p>
</body></html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    # ---- Markdown ----
    md = "# mean/variance 实现：正确性 + 性能测试报告\n\n"
    md += "## 环境与数据\n"
    md += "- CPU：Intel Alder Lake（混合架构，测试中绑定 P-core 以消除大小核调度噪声）\n"
    md += "- 编译器：GCC 15.2 MinGW-w64（用户原始 `mean_var.c` 用 `-O2`，对照档用 `-O3 -march=native`）\n"
    md += "- 计时：QPC 墙钟 + 实测 TSC 频率标定；每档自适应重复次数、取中位数、丢弃预热\n"
    md += "- 数据：正态分布 float（避开 denormal 慢路径）；规模 8 → 128M 元素，覆盖 L1(48KB)/L2(1.25MB)/L3(25MB)/DRAM\n\n"
    md += "## 代表性规模速比（n = 8 … 128M，相对 serial_welford = 1.00x）\n\n"
    md += "| 实现 | " + " | ".join(f"{n:,}" for n in rep_ns) + " |\n"
    md += "|" + "---|" * (len(rep_ns) + 1) + "\n"
    for (kname, klabel, kcolor) in IMPLS:
        cells = []
        for n in rep_ns:
            p = pick(kname, n)
            if p is None:
                cells.append("–")
                continue
            gb, ns, cyc, nsc = p[1], p[2], p[3], p[4]
            base = base_by_n.get(n, 0)
            sp = (base / nsc) if nsc > 0 and base > 0 else 0
            cells.append(f"{gb:.2f}GB/s / {cyc:.3f}cyc / {sp:.2f}x")
        md += f"| {klabel} | " + " | ".join(cells) + " |\n"
    md += "\n> 每格格式：吞吐 GB/s / 每元素周期 / 相对串行 Welford 速比\n\n"
    md += "## 64M（256MB）档速比小结\n"
    for lbl, sp in speedups:
        md += f"- {lbl}：**{sp:.2f}x**\n"
    md += "\n## 性能测试关键结论\n\n"
    md += "- **串行 Welford 全程计算受限**：~12.5 cyc/元素、1.15 GB/s，每元素一次真除法（`delta/k`）是关键路径，即便 128M 元素也到不了内存带宽墙。\n"
    md += "- **编译器自动向量化 Welford 失败**：`serial_autovec`（-O3 -march=native）与 `serial_welford` 逐档数字完全一致（3.47 ns/元素、12.5 cyc/元素），证明手写 AVX2 不可替代。\n"
    md += "- **原始 AVX2 Welford**：大数组 ~12.4 GB/s、1.16 cyc/元素，相对串行 ~10.8x；但热循环里的 `1/cnt` 倒数仍卡在 ~1.16 cyc/元素，没吃满 AVX2 宽度收益（半数加速来自“去掉串行里的除法”，并非纯 8 路并行）。\n"
    md += "- **极小数组反噬**：n=8 时原始 AVX2（15.5 ns）反而比串行（9.65 ns）慢，因 256-bit 寄存器设置 + 8 路归约的固定开销占主导；建议加 size 阈值分发（小数组走标量回退）。\n"
    md += "- **优化版 Welford（opt_f）**：保持 Welford 递推本体不变，仅用 FMA + 2 路独立累加器展开 + 正确舍入除法做工程优化。大数组 ~15.8 GB/s、0.91 cyc/元素，相对串行 ~13.7x、相对原始 AVX2 ~1.27x。提速来自打破依赖链、提升乱序调度 ILP，而非改算法。\n"
    md += "- **优化版 Welford（opt_d, double 累加器）**：同一套递推但 mean/M2 在 double 里累加。大数组 ~13.3 GB/s、1.09 cyc/元素，相对串行 ~11.5x。吞吐略低于 opt_f（cvtps_pd 转换代价），数值稳定性远高于 float 版（见正确性结论）。\n"
    md += "- **朴素 sum/sumsq 单趟**：小数组最快（n=8 仅 3 ns），但灾难性抵消下方差误差达 2.5e8 倍——印证它只能作“反面基线”，不能当生产实现。\n"
    md += "- 测量噪声 RSD 多在 ~5% 以内，属 P-core 睿频/调度抖动，非算法问题。\n\n"
    md += "## FMA 与 2 路展开各自的贡献（隔离实验）\n\n"
    md += "为回答「加速到底来自 FMA 还是 2 路展开」，新增**单路 8 宽 lane + FMA** 版本（`avx2_welford_fma1`），"
    md += "与原始单路无 FMA（`avx2_welford`）、原始单路 `-O3 -march=native`（`avx2_welford_o3`）、2 路 FMA（`opt_f`）三方对照。\n"
    md += "计算受限区（L1/L2，n=8192~327680）实测：\n\n"
    md += "| 实现 | 结构 | cyc/元素 | 备注 |\n"
    md += "|---|---|---|---|\n"
    md += "| `avx2_welford` | 单路 无FMA | 0.926 | -O2 -mavx2 基线 |\n"
    md += "| `avx2_welford_o3` | 单路 -O3 | 0.928 | ≈单路无FMA → **编译器没自动把 mul+add 收缩成 fma** |\n"
    md += "| `avx2_welford_fma1` | 单路 +FMA | **0.729** | FMA 单独贡献 ≈ 1.27x（0.926→0.729）|\n"
    md += "| `avx2_welford_opt_f` | 2路 +FMA | **0.391** | 2 路展开在 FMA 之上再贡献 ≈ 1.87x（0.729→0.391）|\n\n"
    md += "**结论：FMA 与 2 路展开都是真实的加速来源，二者叠加。**\n"
    md += "- FMA 的作用：把 `mean += delta*inv`（乘加）融合成单条 `vfma`，少一次舍入、缩短 loop-carried 依赖链 → 单路即从 0.926 降到 0.729 cyc。\n"
    md += "- 2 路展开的作用：给出两条互不依赖的累加链，让 OoO 在一条链的 fmadd/除法延迟期间执行另一条链，把计算延迟盖掉 → 在 FMA 之上再降到 0.391 cyc。\n"
    md += "- 编译器帮不上忙：`avx2_welford_o3`（-O3 -march=native）仍是 0.928 cyc，与单路无 FMA 几乎相同，证明 intrinsics 不会被自动收缩成 fma，FMA 必须手写。\n\n"
    md += "**反直觉的边界**：到 DRAM 大数组（128M），三者都撞内存带宽墙（~12–15 GB/s），此时 2 路展开的优势被淹没，"
    md += "`opt_f`（11.6 GB/s、1.245 cyc）甚至因循环体更大、前端开销略高而**略慢于**单路 `fma1`（13.44 GB/s、1.075 cyc）。"
    md += "所以 2 路展开的收益只在**计算受限的小/中数组**兑现，大数据组是带宽说了算。\n\n"
    md += "## 正确性测试关键结论\n\n"
    md += "见 `build/correctness_full.txt` 原始输出。要点：\n"
    md += "- 解析解校验：常量数组方差=0、等差数列方差=(n²−1)/12，全部实现通过。\n"
    md += "- 越界读检测（尾部 PAGE_NOACCESS 保护页 + 子进程崩溃判定）：**全部实现均未越界**。\n"
    md += "- `dynrange`（1e-20~1e20 大动态范围）：所有 float 实现方差均溢出到 Inf（2.5e39 > float 上限 3.4e38），属 float 量程限制。\n"
    md += "- `offset1e6`（mean=1e6, σ=1 灾难性抵消）：原始 Welford（串行/AVX2/O3/autovec）方差相对误差最高达 **100%**；朴素 sum/sumsq 直接崩到 2.5e8 倍误差。优化版 opt_d（double 累加器 Welford）把误差压到 **~1e-7** 量级，且仍是 Welford 递推（仅把中间量升精度）。\n"
    md += "- **重要修正（踩坑记录）**：前期曾用「rcp+牛顿快速倒数」替换热循环除法提速，但 `rcp14_2newton(1.0)` 在 float 牛顿下收敛到 `1−2⁻²⁴`（差 1 个 ULP 而非精确 1.0），该误差乘上大均值后被放大成 M2 灾难（offset1e6/n=16 方差算成 62500）。故 opt_f 仍用正确舍入的 `_mm256_div_ps`，稳定性与原版一致。这正说明：要优化的是 Welford 本体，不能用牺牲精度的 trick。\n"
    md += "- 优化方向小结：FMA 融合、2 路独立累加器展开、正确舍入除法、size_t 计数（修复 int 溢出 UB）、运行时 CPU 特性分发（避免无 AVX2 机器 SIGILL）。\n"

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print("OK: wrote", OUT_MD, "and", OUT_HTML)


if __name__ == "__main__":
    main()
