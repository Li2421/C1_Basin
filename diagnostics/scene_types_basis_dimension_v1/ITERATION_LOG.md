
2026-10-28 01:46：S1/S2 的 role_history 含 numpy.bool_，数值控制已运行但诊断 JSON 序列化失败；已有完整结果保留。停止协调进程和当前 stb_ 作业，修复输出编码后仅恢复未完成 UID。该错误不是 solver unknown。控制器、CBF、参数、种子、数值计算不变。

修复方案：只在新 worker 的 recording_fix.py 将诊断 meta 转成 JSON 原生标量；runtime.py、实际列数值、两次安全投影、随机键与所有计算输入均不改。每条新原始结果记录编码适配器 SHA256，原失败作业记录存 incidents/diagnostic_json_bool_20261028/。S4 在读取完整 S1/S2 结果前补充明确的选族次序：优先覆盖所要求的低维维度，再纳入上一轮冻结的二维族，剩余名额依有限见证覆盖排序；仍在六族、2000局原预算内，不读取留出。
