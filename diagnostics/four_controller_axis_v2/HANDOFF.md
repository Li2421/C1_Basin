# Round19b handoff

工作根目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。所有本轮文件在 `diagnostics/four_controller_axis_v2/`；仅按官方生命周期写入共享 rollout DB，没有 git commit。

结论：Four η regime SUPPORTED；H_compress_Four REJECTED；E learning REJECTED。G0/G1a/G2 PASS，G1b MISMATCH（非阻断）。数值完成，视频缺两个正式死锁角色；probe 官方事前 preflight 收据缺失。

核心文件：`PREREGISTRATION.md`、`G1A_SELECTION.json`、`G0_REPORT.json`、`G1A_REPORT.json`、`G1B_REPORT.json`、`G2_REPORT.json`、`FRESH_STATES.json`、`R5_FROZEN.npz`、`R5_FREEZE.receipt.json`、`FIXED_PROTOCOL.json`、`SCHEDULE_PROTOCOL.json`、`FIXED_RESULTS.json`、`SCHEDULE_RESULTS.json`、`results.json`、`REPORT.md`、`videos/manifest.json`。

Slurm arrays: probe 21020 (25 tasks, max18 concurrent CPU), fixed 21045 (18 CPU), schedule 21063 (18 CPU); phases sequential. Exact records: fixed 4,800, schedule 8,000; unknown fixed 36, schedule 57。视频入口失败的缺口不能用 timeout 补齐。

只读复算：用上面的 Python 运行 `diagnostics/four_controller_axis_v2/analyze_fixed.py`、`analyze_schedule.py`、`plot_results.py`、`finalize.py`。前两者会核对 DB 与原始 trace SHA；不要重新提交 rollout。若需正式视频 COMPLETE，须另有事前定义并记录的 Four 死锁判据和同初态基线/成功干预，不得改旧 outcome。
