# Ring：失败与 correction 成功对比

本目录共5个完整仿真MP4：4组左右对比，另附1个独立的安全成功案例。

- 左侧：原controller仅开启safe filter，本例最终timeout。
- 右侧：同controller、同初态、同seed，学习双η与切换时间表（pipline），并开启相同safe filter，最终无碰撞成功。
- Ring没有正式deadlock判据，因此这些是失败/停滞/超时与成功的对比，不标作已证实deadlock。
- 全程原速20fps，每个仿真状态保留一帧；较早结束的一侧停留在真实终态，另一侧播放到终止。
- 只读取已有仿真轨迹，未补跑、未改变物理、controller、事件或成功标准。
- 这是为展示挑选的案例，不代表总体成功率。筛选依据及逐条输入、checkpoint、η、切换步、原始终止事件和SHA256见manifest.json。
- source_traces/包含完整导出证据，可追溯到manifest中的原始记录。
