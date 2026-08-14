# 用 cmd 运行 bench，避免 PowerShell 管道对 UTF-8 字节的二次编码
# cmd 重定向把 bench 的 UTF-8 输出字节原样写入文件
cmd /c "cd /d D:\aicode\aicode\conv2d\build_avx2 && bench.exe > bench_utf8.txt 2>&1"
"CMD_DONE"
