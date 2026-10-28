# 开工导入写入审计

在导入研究代码前启用guard.py。写入仅允许本轮目录和本实验官方journal；只有协调者可写官方SQLite。旧代码、作者代码及虚拟环境只读。

已定位的C1导入写入点：旧harness experiment.py导入时创建临时与绘图/缓存目录；common.evidence调用时追加HERE/EVIDENCE_INPUTS.jsonl；backend结果写入经experiment.write；Python可能写pycache；JAX/CUDA/Matplotlib可能创建缓存。先将harness逐字节复制到本轮generated/，使HERE自然落在本轮，禁用pycache，CPU-only、BLAS单线程，重定向TMPDIR、MPLCONFIGDIR、XDG_CACHE_HOME、CUDA_CACHE_PATH。每次进程退出记录所有实际导入文件哈希。

论文端口尚未导入或运行；进入该阶段前须完成作者源码的绘图、日志、缓存和临时路径审计并追加本文件。Chen作者异常分支plot_pre_traj、结果绘图/规划路径保存，旋转木马图形/视频输出，以及作者依赖的日志/缓存路径均须检查。仅在本轮适配层重定向，不修改vendor/deps。

Python审计钩子不能覆盖本地库自行执行的全部系统调用，不称其为OS沙箱；结合导入SHA、源文件完整性和已知C扩展输出路径核验。发现保护事件按prompt整轮停止。
