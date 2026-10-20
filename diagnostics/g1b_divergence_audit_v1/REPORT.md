# Round25 G1b divergence audit

**ROOT_CAUSE_FOUND**：G1b 的一例分叉由全局 JAX FP64 导入副作用导致。历史路径显式恢复 FP32 后，与新适配层的首动作及全部272步轨迹逐字节一致；新适配层强制 FP64 后，与历史路径的首动作及全部677步轨迹逐字节一致。

事实：四条路径的观测字节、RNG key、四组归一化数组及第一步网络参数 hash 完全相同；首次可观测数值差异是 Flow 输出。FP32 路径成功，FP64 路径 timeout。`poc/field_controller.py:25` 在导入时执行 `jax.config.update('jax_enable_x64', True)`；原 g1b.py 先导入新适配层（间接导入 PoC），运行历史路径前没有复位。`sample_actions` 的 `jax.random.normal` 没指定 dtype，全局精度会改变同 key 的抽样及后续网络计算。证据为 old/old32/new32/new64.json 和四份完整 trace、results.json。

解释：这是当前复现入口的导入顺序差异，不能据此把历史 intervention 标签标为错误。Round19b 主数值使用明确恢复 FP32 的新适配层，且 G1a 已逐位匹配32条历史结局/长度/最小距离指纹；本轮定位不削弱其以 G1a 为准的结果。历史 intervention 系列中未显式记录精度的条目仍有 provenance 缺口，见 IMPACT_AUDIT.json；不把一例复现推广为所有历史记录精度已确认。

仅运行四个排查单例，不写共享 DB、不替换历史记录、不修改历史代码或环境、不做 commit。可操作建议是调用层在全部导入之后显式恢复协议精度；本轮仅在本目录的排查脚本执行该对照。

恢复后的FP32单例还匹配原历史记录的成功结局、绝对终止步695及两项float64最小距离指纹，见HISTORICAL_FINGERPRINT_RECOVERY.json；272是本次从中间初态继续执行的物理步数。
