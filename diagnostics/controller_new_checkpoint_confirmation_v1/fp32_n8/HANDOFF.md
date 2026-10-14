# Round12 FP32/N8 handoff

Active directory: /home/zhihan/research/Basin_C1/diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8

# Round12：8 个新参数 final 的前瞻确认

C1 **SUPPORTED**（字面点估计门 通过）；C3 **SUPPORTED**。C2 的 headroom 门 满足：3/8 个真实下界点 E<5pp（unknown 可取范围 [3, 3]），C2 **SUPPORTED**。这次只确认 8 个参数不同的 final；6 个参数完全相同的 final 排除。全部 12,288 条主 rollout 与 256 次修复后无标签探测已完成。批次状态 **NUMERICAL_COMPLETE_VIDEO_INCOMPLETE**；视频严格门未通过时不宣称整批交付完成。


Read REPORT.md and results.json for endpoints. Original parent phase is invalid for confirmation; do not resume old queued jobs or use its features/choices. Retained original prereg is superseded only by PREREGISTRATION_ADDENDUM_N8_FP32.md. Eligible list fixed in FINAL_BASE_PROFILES.json; six exclusions in semantic parameter audit. No training or additional rollout is needed. Unknowns must not be rerun. Reuse journals/DB only with exact semantic identities. All output writes restricted to parent experiment and explicit SQLite lifecycle.

Numerical postflight PASS; strict video gate exit 2, missing ['deadlock_show', 'deadlock_resolution']. No deadlock event exists in frozen Ring runtime. Do not fabricate roles or create COMPLETE marker. Source14 best, primary13 excluding brother, target8; original C1-C4/C1b/headroom thresholds unchanged. Corrected probe256, main12288, gate32. Reproduction: core.py guards prereg; probe.py/features receipt; models.py/selector receipt; design.py/cache; run.py binds immutable R9 loop and restores FP32; gate.py exact32bytes; merge.py official single merger; analyze.py frozen endpoints; videos.py unified Ring-scoped entry; report.py immutable/chronology audit.
