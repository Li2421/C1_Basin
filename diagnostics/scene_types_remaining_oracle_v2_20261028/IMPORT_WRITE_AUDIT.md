# Import 写入边界审计

旧 harness 的 experiment.py 在 import 时 mkdir TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME/CUDA_CACHE_PATH；common.evidence 在调用时追加 HERE/EVIDENCE_INPUTS.jsonl。二者与 bootstrap 先复制到本轮 generated/，内容逐字节不变，HERE 自然指向本轮目录。backend 中的结果写入函数通过 experiment.write 进入同一边界。旧 runner/oracle/basis_select 的顶层路径修改与 mkdir 不直接 import。

guard.py 在任何研究模块之前启用：禁用 pycache，CPU-only、BLAS 单线程，重定向缓存和临时路径；审计 Python open、目录/文件变更、SQLite 连接。写入仅放行本轮目录与本实验的官方 journal；仅协调进程可连接官方 SQLite 写入。审计钩子不能拦截第三方本地库自行执行的系统调用，故也记录导入文件哈希与结束完整性核对，不能把 Python 钩子称为完整 OS 沙箱。

复用上一轮200局逐位认证；数值函数逐字节复制，新增命名空间另做初步动作回放核对。旧实验结果只读，已有运行时/名义控制器/安全层定义通过 SHA256 留痕复用。
