# 运行 bench：PowerShell 侧强制 UTF-8 编解码，确保中文表头正确显示
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location "D:\aicode\aicode\conv2d\build_avx2"
$lines = & ".\bench.exe" 2>&1
# 直接输出到控制台（UTF-8 渲染）
$lines | ForEach-Object { [Console]::Out.WriteLine($_) }
# 同时保存为 UTF-8 日志，便于后续查看
$lines | Out-File -Encoding utf8 "D:\aicode\aicode\conv2d\build_avx2\bench_results.log"
