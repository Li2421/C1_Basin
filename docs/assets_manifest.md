# C1 Toy GiveWay 大型资产清单

> 生成于 2026-09-21。大型资产只做内容寻址和用途登记；本轮未删除、改写或按实验表现筛选这些资产。


## 哈希定义

- `file` 行为普通文件 SHA256。
- `tree` 行为确定性目录 SHA256：按相对路径 UTF-8 字典序，对每个文件计算 SHA256，再哈希全部 `文件SHA256␠␠相对路径\n`。
- 大小与文件数是在删除 `.log`、Python cache 和重复 `source_snapshot/` 后计算；这些删除项已在清理前 inventory 中计数。
- 本清单不包含两个本地虚拟环境；它们属于可再生开发环境，并由 `.gitignore` 排除。

## 资产

| path | type | files | size | SHA256 | current use |
|---|---|---:|---:|---|---|
| `baseline_309_314/CLEANUP.md` | `file` | 1 | 1.88 KiB | `92849735793ead4855ffa1393b94b94abf7863901d28b94b98cd2c7bc98e3570` | 共享规划/配置资产 |
| `baseline_309_314/checkpoints/seed0/benchmark.json` | `file` | 1 | 5.31 KiB | `f060bb78374555bc16b109a911003828be07d47e4184c62235f656a7eeda96a6` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl` | `file` | 1 | 1.60 MiB | `8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed0/config.json` | `file` | 1 | 2.23 KiB | `fb4c7193fe7bdb36030397e79c6d7edb86aa9a312c3f8fe4fb9bb8b29d92ed2d` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed0/metrics.jsonl` | `file` | 1 | 1.93 KiB | `884eeabd60cdd8df14d6d12571ef1dbb47ea2ac43c2918616797addb48db2f8e` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed1/benchmark.json` | `file` | 1 | 5.31 KiB | `134979a8d039e1a38954e2da35e813ae61f109c4d6d1448195e7ccd5757a3aa5` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed1/ckpt_0025000.pkl` | `file` | 1 | 1.60 MiB | `3f58b12266d19358a8ef5673e3e4018c1248c526de7d71877218ea176bd53383` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed1/config.json` | `file` | 1 | 2.23 KiB | `82db4c1eb2c946a0a2a82f1848674bde69c57cb177e9335b158cce5390d0c44e` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/checkpoints/seed1/metrics.jsonl` | `file` | 1 | 714.00 B | `8e05c84beebdc8de24b7d9d7443b5e7cf4c425576c3512c0261c7f74fa5911e7` | 冻结 Flow-BC / 历史模型检查点；保留供可复现加载 |
| `baseline_309_314/planning/PROTOCOL.md` | `file` | 1 | 985.00 B | `4743874609d558a9928f3db1a75b28e34fae42e5eb2ad8fbbefeb1cb5e1c471c` | 共享规划/配置资产 |
| `baseline_309_314/planning/RESULT_ZH.md` | `file` | 1 | 1.97 KiB | `d169aced04cedd5d74166b55c46a9849a65d72241f83ec35a7cdc93354570fae` | 共享规划/配置资产 |
| `baseline_309_314/planning/outcome_distribution.png` | `file` | 1 | 127.97 KiB | `4a8c0bc59ec280e117061cf928d5a3af5a8da4b3498eb445baf8dffd95393d3a` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/comparison.json` | `file` | 1 | 986.00 B | `c8b8f563bb4666825f9d01abe597572eb74d95169532f9b32e8ba53b1fc2a239` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/complete.json` | `file` | 1 | 29.00 B | `54c27cffd1c90cdbf195adba2da9a428695083657d92c150f4b1f658af638126` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/config.json` | `file` | 1 | 40.44 KiB | `9f1894478a628c5be059ecb53a3b7b04449fe59d92a473eea052300d6b8dec19` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/debug.json` | `file` | 1 | 81.00 B | `d65f6379f146299901c9cf3703cb7597eded869a314d73415220e54514e47480` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/mac_cbf/summary.json` | `file` | 1 | 239.13 KiB | `e86af3896ee8b4c3004b40a60f327426614b2fdb527b0c16899c1716fce514bb` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed0_paired/mac_only/summary.json` | `file` | 1 | 232.41 KiB | `c4bde60d0f90ead3531eefdbe8f3a25f4058cb4709ed7f35d76770f5dbf4b0b7` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/comparison.json` | `file` | 1 | 995.00 B | `7ba15c3f714023e8fdc83df0dea1e77054a855d557037293fe2ee2bf2ef8115c` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/complete.json` | `file` | 1 | 29.00 B | `54c27cffd1c90cdbf195adba2da9a428695083657d92c150f4b1f658af638126` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/config.json` | `file` | 1 | 40.44 KiB | `ab1ae8a97a4a7734966e0345d8e570a33ad572b25dad9d543a36c1682541809d` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/debug.json` | `file` | 1 | 81.00 B | `d65f6379f146299901c9cf3703cb7597eded869a314d73415220e54514e47480` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/mac_cbf/summary.json` | `file` | 1 | 239.19 KiB | `450dd435c1f222a20005b3fc4bd518f9a96006a475ace8d61d6b16ef63e07fc6` | 共享规划/配置资产 |
| `baseline_309_314/planning/seed1_paired/mac_only/summary.json` | `file` | 1 | 232.08 KiB | `e1d1a14de0f58e146ed2ff4455e8c191c9bbe8b86a5755a0dd694aefa82b6090` | 共享规划/配置资产 |
| `baseline_309_314/planning/stalled_outcomes_v3.json` | `file` | 1 | 166.47 KiB | `03ab192bf282856387331f2da75a2c133a13deb100e637fb0faa02a028435cda` | 共享规划/配置资产 |
| `baseline_309_314/planning/wide_initial_states_200.npz` | `file` | 1 | 6.95 KiB | `30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb` | 共享规划/配置资产 |
| `datasets/give_way_si_short_uniform_state_v1/` | `tree` | 451 | 28.54 MiB | `dfd905db23d5e203daefcfcd1ee5d06a013c05b421a37ae60fec797ca9d56db2` | GiveWay 冻结数据集或初态集合 |
| `datasets/give_way_si_short_v1/` | `tree` | 503 | 28.97 MiB | `19947d59da8213032f0f8cb0e6735784906f5e24c7db5b912438b9c0e6c7c7e9` | GiveWay 冻结数据集或初态集合 |
| `papers/tro_deadlock.pdf` | `file` | 1 | 6.74 MiB | `7664838f8ea1fc62bd042d9643f3fe7466eb21ea803a85a43280b6ad189ff033` | 研究参考资料 |
| `papers/tro_deadlock.txt` | `file` | 1 | 148.88 KiB | `f977c0fa5e714020ce29e02edda2ede88ba383c69ab7686992619459a6f27bd1` | 研究参考资料 |
| `results/C1_CALIBRATION_RESUMPTION.md` | `file` | 1 | 1.48 KiB | `b08707417e97489f92d76b6b8e143bc2f1a44c77603337c9912cff56f93435db` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_CONSOLIDATED_FINDINGS.md` | `file` | 1 | 9.17 KiB | `396c293163f0549aeea299084447c0fb0223a90021a48063c5df909a52ad99b2` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_DATASET_LOGIC_AUDIT.md` | `file` | 1 | 7.94 KiB | `5909a21c0d6299779c9419218349722b82b8e3deb0063e160d57255298d83ac8` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_GUARDED_OPTIMIZATION_PROTOCOL.md` | `file` | 1 | 2.15 KiB | `b212e1e9799dae12eff25f4447da645d30866871b5dee2350f0a91d985218736` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_GUARDED_OPTIMIZATION_RESULTS.md` | `file` | 1 | 5.52 KiB | `7ead0c534d17565ac38c2b6e6c01db466006f415d2f15311309db9ba6bc56521` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_LOGIC_AUDIT.md` | `file` | 1 | 6.32 KiB | `4b45c6888769fe2bf3ca5e6bb5326ce85f48888f08d947604b0a144f1db8876e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_RECOVERY_20260917.md` | `file` | 1 | 2.33 KiB | `417b7228fccb0f73978af21c86718fa538c66e026922bc18e6d85115cccd677a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_RISK_ITERATIONS.md` | `file` | 1 | 5.25 KiB | `b7f5cfcd3c1862b33bf001b58c80964105a85b2d315fb4daae68dccd7bbe110e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_THEORY_AND_REGRESSION_DIAGNOSIS.md` | `file` | 1 | 9.73 KiB | `445e66d033704f4f2955c666e971ddbc79b2ad1b871ea97acfd53812e7222b7f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_V1_RESUMPTION.md` | `file` | 1 | 4.70 KiB | `1a2c27cc2c8ff654ea01bfdf87cf8e6b433a56086b3b7b3c94d9be35da9de2d6` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/C1_V2_FROZEN_PROTOCOL.md` | `file` | 1 | 4.17 KiB | `711febf1dce9035ad7c8acb4876850d691c38eda5168d14bf2bfcee6c906fa61` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_action_probe_parity_v1/` | `tree` | 8 | 464.47 KiB | `80f9447107a62a03a2f37bd9ffd0e9aa34f93909c5b998ef89c2b16a3a103184` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_arithmetic_harmonic_adapter_v1/` | `tree` | 2 | 23.98 KiB | `51d3161adb2956f028151fd45a5e12487c3059a472d279ae346221490f3f56fc` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_arithmetic_harmonic_kernel_v1/` | `tree` | 1 | 557.00 B | `a2f4c6403210c8b879d7ea7adba0763a750034627708a35b4872c7afedfd1216` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_arithmetic_harmonic_probe_v1/` | `tree` | 234 | 17.68 MiB | `7ffb72de2e4710d45594f9c9297474af8634c4ca91b27b2d2c9d6560ce5f3daa` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_bellman_certificate_audit_v1.json` | `file` | 1 | 960.00 B | `2942e880120c4262f8a50feaa2060388f1f477641ff051105142fe2165c6704c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_boundary_importance_v1/` | `tree` | 232 | 14.34 MiB | `657b8cc475584dbc2b8e3505d0a5ee3d4a84255d3bf53f27e0eb4f4b50f02414` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_cached_directional_v1.json` | `file` | 1 | 137.65 KiB | `34aafce2c0befe4fa82e580f4dd6d1ed667c2815a9c3c137135801572ae6714c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_candidate_boundary_probe.json` | `file` | 1 | 2.51 KiB | `88eda41eea711e8cfca529b661bc5a314a219aa79cf481d62e3c992640145ecb` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_candidate_coverage_expansion/` | `tree` | 651 | 26.39 MiB | `c49f5653f5d0c77dca8afeeccacd10392dacb55c20ff7e0448ce703bd228014a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_certificate_development/` | `tree` | 82 | 8.77 MiB | `a2cfc9799c5f2ea52bfc0b97753a6fa8167382f8bf9bbd72bb4030a82a87dfd3` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_clean_geometry_audit/` | `tree` | 115 | 151.02 KiB | `88992a80a2549446c56a553772a509c5555c4337754e6c68ac93e007bf4536ac` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_completed_waiting_risk_audit/` | `tree` | 12 | 578.96 KiB | `c749e96b6b0071256b2c1e8ec944a9dee15d8433a142ed93341b4c9cf02c295d` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_completion_development/` | `tree` | 18 | 7.65 MiB | `26c08c1568e777caff8d4c1055b27b873e03390d07ed6201a61bc6ba4640072b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_conditional_event_sections_v1/` | `tree` | 59 | 2.43 MiB | `cd2c9346f3416f78e44069fb20ebf53399b2f1118f6db07d3b70dd0cb49f2d3a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_conditional_gaussian_kernel_v1/` | `tree` | 1 | 512.00 B | `4ea6c21e687fd835f7a202f91e0955eddf76e0187981a3cca12b8acc43c37a5f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_conditional_gpu_v2/` | `tree` | 406 | 31.09 MiB | `ce49143e0d5b742c05afadbc2013ccd59780133fbe8a2db8c4b4049835d690e1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_conditional_risk_cpu_v1/` | `tree` | 195 | 15.00 MiB | `cfd9c9e6f7e2085cd5f3058f83ff62032a7b6bc364b58e15b46f153173fd3a9f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_conditional_transversality_v1.json` | `file` | 1 | 3.42 KiB | `996a610a52c2a0b9d30e538d172a4fb4d52ea72c87516c802fe62d4e4a0c67cf` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_critical_cone_kernel_v1/` | `tree` | 2 | 539.00 B | `8b63b9ca58e0bd489d85c62747c1d9c6ede2826d799ebfb60717c4aee632d8e5` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_cross_noise_audit_v1.json` | `file` | 1 | 7.86 KiB | `f15bce7fb796e50b6c1605b52736dffa5f3078c7cb7675aa400307579525256b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_audit_seed0/` | `tree` | 7 | 1.63 MiB | `b7e44da26b66bd4a3b3eeb3d60a73ef316b3a3423a80e884e4a9c18559da8eaf` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_extended_h120_comparison/` | `tree` | 2 | 24.31 KiB | `4f2618bfe120649de3ae7c3ce7ab86bda0735182d2a60b2b23e9fd9ac72f1e46` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_extended_h120_seed0_baselines/` | `tree` | 56 | 12.65 MiB | `59a6e7455c2db1d5c132ed8a8dabd7bcefabb4dd9e3f655fb43e2e695b11c920` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_extended_h120_seed0_c1/` | `tree` | 29 | 18.47 MiB | `57733a887f1beab4f57d6bb21c23f72de36491905ea83068b506b20ffb591f0a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_frozen_sets/` | `tree` | 3 | 2.71 MiB | `9fa60938f5dd5ddcd8a0cf4dd10693b669e9efad1ef64052e0147bae3cc0fed6` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_paired_seed0_baselines/` | `tree` | 406 | 95.21 MiB | `deac3ec6b3f124a5e54477f228c3ab0c6ff549b107a57ef918972ff7235b83c4` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_dataset_paired_seed0_c1/` | `tree` | 144 | 101.94 MiB | `226de7f63b8bc874ead2718ce3ff7ccf0ca4603d71aea512e80044d660367d00` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_deadlock_primary/` | `tree` | 338 | 33.64 MiB | `d827ec260d367bee396e1b6ce9c55f84bdaf7e37247bcabeb9ec984a11088e2e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_deadlock_union/` | `tree` | 8,289 | 359.45 MiB | `6105355e7a5c3ade1125b7489caa2488082af18484455a24d507f2a8719fbad9` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_diagnosis_training/` | `tree` | 3 | 85.31 KiB | `f3713d8f635ce9609f85814c69f04f3d8592e2f5394005443a2066b85bd5286b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_diagnosis_trajectories/` | `tree` | 12 | 797.25 KiB | `fe6ecf648304a88f84dcba9b05d5124769f079591df5fe949bd3efe5d03de006` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_difficulty_calibration_v1/` | `tree` | 53 | 3.22 MiB | `45a926a0279ed3257996fa288a770a7796d38413c300a74b5408ec2d4bb84a0d` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_direction_a_minimal_v1/` | `tree` | 7 | 124.51 KiB | `549ca154dd4cfbbbd001c9f30184f76f4168b25f67f62f61e2840bfd223221e9` | Direction A 实现审计/独立诊断证据 |
| `results/c1_directional_closed_loop_5sec_capture_v1/` | `tree` | 3 | 6.84 KiB | `ab798556ace3973dbe4ce2f3efa3370249c4743b6f13c40bd120ec3d3dcb6550` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_closed_loop_5sec_v1/` | `tree` | 2 | 4.03 KiB | `d386f2c440693a0927d3261e2a2374ec6fbf2a7101492592fbaa31cc4a79f21c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_closed_loop_5sec_v2/` | `tree` | 22 | 644.97 KiB | `4013ac952d73c3545639fc0b32c730c84b92e2fdb78049a46a994e824aafcf50` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_closed_loop_v1/` | `tree` | 22 | 138.60 KiB | `35abffeeda62a22a5e556aefa5b792dd77d301b3aa5f314392ea0a967d873aae` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_derivative_cpu_v1/` | `tree` | 4 | 7.03 KiB | `35b47f9be649467320c3ed6eb58e378e8a7fc37c137feeafa22b2b207da4561c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_derivative_cpu_v2/` | `tree` | 4 | 11.87 KiB | `3ab81618f148bc7f1b7e35ac549e867c7c9ceef97333062d7d5218ec79c3b612` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_localization_v1/` | `tree` | 2 | 50.10 KiB | `7aa932cc725e981b005231310d547b9a671e3dd4192afa72d04952efaf426e23` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_localization_v2/` | `tree` | 0 | 0.00 B | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_directional_localization_v3/` | `tree` | 3 | 70.65 KiB | `ed888493d31e3d696ebecb4dc672268039e456a93d29c04b8124d6717e912975` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_double_projection_boundary_v1.json` | `file` | 1 | 24.77 KiB | `f2d9f5505d705731044e847c0f6827b959470de0924c14999d2660fc4bd09328` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_early_forward_cpu_v1/` | `tree` | 1 | 975.00 B | `f8cce2ace8d3f97e1b0fcb0365cb1db71def009b88b1c6dd3fec5edfe9aa5ec6` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_early_forward_cpu_v2/` | `tree` | 2 | 4.02 KiB | `6d110f3b54e05d79b969a8954cf08ce35327f60919b9b42049fe0aa78df1656f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_early_gradient_v1/` | `tree` | 2 | 8.40 KiB | `3db7795f643afa1f93eb6faf1d0ef73059070485338aa9b27f0d56cfa5859bf2` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_early_gradient_v2/` | `tree` | 409 | 30.36 MiB | `38d6c5aa78d46db3ba5f97c33e927fafee2cd132090cf39f1aef1c8a91a6a3ae` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_episode_pool_benchmark.json` | `file` | 1 | 686.00 B | `2bd06c6701de905310a0a59926bcca0125af01b7cdda622c34b495c7ce206a6e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_episode_risk_c1_audit.json` | `file` | 1 | 7.55 KiB | `30747269444848919446deb1ffa3ca0bc0d490a0bb7adfc950b4adefd98c5326` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_episode_risk_c1_balanced.json` | `file` | 1 | 7.71 KiB | `965cec691943befb85c8fbaa0b34eb915fe8a1bf5394f9f4cddc1251da094e50` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_episode_risk_safety_audit.json` | `file` | 1 | 7.60 KiB | `18201ef9b7c04158118fea4b8afef9871fb35dd6d76812bdeadcfc81e6375e79` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_episode_risk_safety_balanced.json` | `file` | 1 | 7.77 KiB | `7b46e3e3117b9f46fe0a54e71b3be08873cdbdc23990ec617436b306815aac2e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_event_history_prefix_v1/` | `tree` | 9 | 1.26 MiB | `9d8424e3f8c97b55732e30ee693df71a066e61583a3f6dce82a5b49cb2d9bf60` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_event_history_probe_v1/` | `tree` | 234 | 17.50 MiB | `5255d4872df9814177ba3f2d86ea93c4e14ca9801acf10982dd9e54ae02442f1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_event_probability_difference_v1/` | `tree` | 54 | 2.63 MiB | `9d30fd3845f45c2be8f05ac1d47db038be41dd013a4d77f68b83e4a0e750e80d` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_exact_margin_v1/` | `tree` | 9 | 3.65 MiB | `6e2b97da3812fa10f58db1e9996e74de853466c9dd6e72c532e69c0aa0d94050` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_exact_pipeline_monitor.json` | `file` | 1 | 13.18 KiB | `abc44d8f26dcee31227af2e1ca700915200d85f830df52d184204329117459f0` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_fixed_calibration_smoke/` | `tree` | 3 | 1.61 MiB | `611a41e9b32737e88c50f96ec0f52f12a8e396d07baeb82ca87649c32f594cda` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_fixed_horizon_waiting_audit/` | `tree` | 59 | 7.64 MiB | `a3b13b5798e789d0497684d12893e0c12ff05142af109c6eba8b2484f0f62783` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_focused_p_vs_pg_6seeds/` | `tree` | 3,973 | 166.54 MiB | `0d07e010cce97df4bfecc3355d42f2974211358da8ef9a49e57900b3e7661a33` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_formal_h4_seed0/` | `tree` | 3 | 1.61 MiB | `1c60eb0f8ba7e286d52b379951384ed9bd51c9a5a0cff44c79b0a87ba511a0fb` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_formal_seed0/` | `tree` | 3 | 559.43 KiB | `8856ef734ec97cb06eb5d4e4921fb452a0ced33d2db9d1137bf3de0d93777a77` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_four_objectives_multiseed/` | `tree` | 5,062 | 233.22 MiB | `7dab71320726875df71a3008b81f079497c340883ce28b4cf0eac876b540981d` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_frozen_unseen_64/` | `tree` | 20,678 | 675.81 MiB | `1ab275627bc3c2c6ddf114fadab5c9700a8f355e574e9ce074a0138b91a92286` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_gaussian_transport_kernel_v1/` | `tree` | 2 | 1.13 KiB | `97cff3189fc52e53de3576d181dd98a4ee70f06c4e2d4999ead256610851dac8` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_geometry_ranking_analysis/` | `tree` | 42 | 4.14 MiB | `51765b5d015c1c1635458741f2a5a15728ce46449154050725a698f84bf83a30` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_geometry_solver_safety_audit/` | `tree` | 20 | 562.21 KiB | `e1842ec15d3783350f791003ed278913b5d7bf5a088d7473bcd6801cb6f92108` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_gradient_closed_loop_v1/` | `tree` | 373 | 30.90 MiB | `587f384761f3ddb740f88f16964a1c1a4f3aa7157c084843d1b88f15688d4748` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_gradient_exact_v1/` | `tree` | 372 | 29.73 MiB | `0888ce7dbc427e54d287184387b3338848c23a84ca9ef6ce9a03de7ef7bf28e1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_gradient_ordered_v1/` | `tree` | 369 | 29.72 MiB | `99085e1beb2f42b3c4b3a44859faf9003de7fd19c50bd2e06f630c77377b61ba` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_gradient_paths_guarded/` | `tree` | 4 | 478.58 KiB | `dbc0e703904f0c61327e63df23a530c2c8e1eb06dd7aa0de92e442b63ef66ccd` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_guard_factorial_v1/` | `tree` | 428 | 95.54 MiB | `227e42e999694a1a345993886d0a8a6c3cabe27b84f425f7b1410ecc9559f8d5` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_guard_trace_audit_v1/` | `tree` | 2 | 154.79 KiB | `227522d1a40e57b8b7c0b0de66ea4fdc02a113582e7e9df49a54f7da7c9d1b02` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_guarded_h120_comparison/` | `tree` | 2 | 24.84 KiB | `9dccd4ff9c8488f00bb1a4d123f102c476d879c7036c508a8ca26ae252d45d46` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_guarded_h120_seed0/` | `tree` | 29 | 18.82 MiB | `9473daa93a1b17af0c27ec0071b61b956828b0afea99f7d7fd52da27c6e97928` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_guarded_seed0/` | `tree` | 6 | 1.75 MiB | `1a009402ac87a772f62b21d0c81da6504efa59586432ab7bb5ebb2079fd859ee` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_independent_distribution/` | `tree` | 533 | 30.73 MiB | `9fc58aa70aec7513213672d4c943637bf562757e5e9b222cd65d153534c6022a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_jax_training_audit/` | `tree` | 24 | 2.65 MiB | `a3e54e680357ff2c1ba2c6807b6acf88a0b13da6b1b16b22873612755e07307a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_joint_closed_loop_audit/` | `tree` | 62 | 6.68 MiB | `d147b5fed19a1ed33cd85f27342c56b19e47b2830e6d0ed921e7532f0bf20855` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_joint_witness_risk_audit/` | `tree` | 263 | 20.27 MiB | `060aa35c7786a264c1105386737fd47bf4fda4b648f2fdca702d4fac6f38be46` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_large_risk_association_400/` | `tree` | 406 | 37.83 MiB | `d2d63d1ebe2cfa6a3d87d9d7ff68005b4a29ffcbe07a88557debac874e276246` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_logic_audit_replay/` | `tree` | 5 | 1.35 MiB | `3474fba36db79558878e03caca8f38dec9761509df84cf3115f721d1850d4397` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_logic_smoke_validation/` | `tree` | 1 | 1.36 KiB | `7be97810c4a19c2c7e8ad75065a26d6bf74c0c16320c0901fdd09f4d184b1e2c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_logic_v11_smoke/` | `tree` | 4 | 1.61 MiB | `135dba7b673ba2f98a5c516a1b6fbd3db86d2904ce7a6f0ac24bf0584130bfc1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_matched_scene_development/` | `tree` | 71 | 2.61 MiB | `28c2eea23081d2790443da309f4839b0ced5cf05a1fd02e20b4e589b2561039e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_matched_scene_stage_i_smoke/` | `tree` | 56 | 1.98 MiB | `5950676728d0e46c5fa17655fa7a7f271ffe3f9d862ccf6b043ce7fde748a81a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_matched_scene_stage_i_v1/` | `tree` | 7,798 | 617.14 MiB | `7e6148f1273794e39d3b7a1c2fc0bf2f26664e4424a31d415b836232b0595e8a` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_matched_scene_stage_i_wide_smoke/` | `tree` | 56 | 2.26 MiB | `a2c8bffdfb3ea8f8cb1bb92d8ba6f424ead7afb4d3cf515ddb59d6fb1562538f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_matched_scene_stage_i_wide_v1/` | `tree` | 7,812 | 686.10 MiB | `9de57d7228189f1d1ed65af4d6e9de37469b0861993c035a1efbf50b4a066456` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_moment_probability_probe_v1/` | `tree` | 232 | 12.39 MiB | `928ee7e062a025ca5d671076e3ba849d6020e59eff86251d9255be7b98c3f08b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_multistep_objective_audit/` | `tree` | 280 | 10.24 MiB | `d2e24fa44b371725333102de9e2eba4f81475769eb5f391a971bc4bc17a1a027` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_nominal_localization_v1/` | `tree` | 3 | 48.24 KiB | `49f2ee5b8223738bd5703029e69eb19dcfafe9945490f9f730ad821a0a81d4a4` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_optimizer_diagnostic_v1/` | `tree` | 18 | 4.73 MiB | `83c16aab14050ca53f3464ae37a5574cddfd5a858907425704e7ca62a4de7b2c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_ordered_parallel_v1/` | `tree` | 4 | 125.88 KiB | `8f06b04b5c8b3c03bcf8ea11fffe5ff8018f50fb31df1125088b8fc4afeedb9b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_ordered_parallel_v2/` | `tree` | 8 | 3.45 MiB | `4ad257f9ef1514a2d9ea2cccc3c07c33022315e165ac945dfb134431b2d57c97` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_pause_after_first_large_20260917/` | `tree` | 6 | 8.79 KiB | `4df46e859f70f04994a2a7651a0fd0db716ef06dffa42b51b538e5db6c9887d1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_prefix_physics_cpu_v1/` | `tree` | 10 | 455.61 KiB | `305639803ec03853a99ef091841dd1eee6b8da74decff15dbc76d9e047443f8e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_pretraining_audits/` | `tree` | 2,218 | 6.45 MiB | `1ebae9d5f8922681c6933151c61ced6a11c830863e0c89bdb02c3412bd67aab0` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_primal_dual_seed0/` | `tree` | 3 | 561.91 KiB | `7d960d6e80a0f707b4c8fe46bdf2937613f5a2fa1c1b68637ee62b8460857221` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_probability_gradient_limits_v1.json` | `file` | 1 | 3.97 KiB | `be58381d2230a37e5bde3c78394479f9e184bf1b8a28914f4d947060b2d49035` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_progress_conditioned_wait_audit/` | `tree` | 224 | 2.14 MiB | `07d9683cfab9e6e313918f554915deadfb5e5f8eaeb2f5573ad1c54e588e91c8` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_progress_debt_audit/` | `tree` | 5 | 298.88 KiB | `7afa278d8504ad4cf36e1be386a978c3bb761e9270b6cc5b5220f600279601e0` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_projection_derivative_audit_v1.json` | `file` | 1 | 19.19 KiB | `12028d7c19dd3275b3a08dd07d1d070eb5c1c1e1a51f895ea1f468ccb042564b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_projection_derivative_audit_v1_source.py` | `file` | 1 | 2.97 KiB | `5df852f36aa59423acf0f8b9b8ff4d5bc319bc3bf1047e561d0cbb981759befe` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_projection_derivative_audit_v2.json` | `file` | 1 | 39.12 KiB | `39729d4ea37972a74f94ea78d95398bf5069bb2533d62a50568c467cf1899a64` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_projection_derivative_tighter_v1.json` | `file` | 1 | 39.50 KiB | `5f2553ca3e40756e2487ad0073dc7242a902a2e116d2f24c3b7451b8dc6d6e0b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_r_cert_aggregate_audit_v1.json` | `file` | 1 | 2.46 KiB | `715fae056fb550eeb178cad460736d7040f2e1f860424053d703d4ad736121da` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_risk_deadlock_correspondence/` | `tree` | 776 | 5.15 MiB | `cc48b0637013ff5e74063379c5ab789e448bab696172ed292e2c3a02101bcb1e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_risk_two_tests_v1/` | `tree` | 434 | 32.39 MiB | `0b921aaddeffb5e8468be531bbabb20d27aed31c25ceb9a63e13d642b07d333b` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_scene_deadlock_union/` | `tree` | 6,475 | 265.43 MiB | `c272c94b47eacb32206701c28ea8c64b3d0b6517623fb0a3ffcbad3bb289768f` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_seed0_risk_audit.json` | `file` | 1 | 357.76 KiB | `48442086f67b52a660a2d5b361600ffd8ef2de03a70b4859cfcc7ed2270f6115` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_soft_activity_edge/` | `tree` | 12 | 590.43 KiB | `5e423785a6a74977dede1dc9d878c945c4e6d0d3e2e69e2b556110eb1bd1a9c5` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_soft_activity_experiment/` | `tree` | 3 | 262.64 KiB | `0d966a97ab4adf67dd875c0cedf0884d711bd00de13c206a6aa7511b4c33e48c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_soft_v01_h120_seed0/` | `tree` | 3 | 1.61 MiB | `451fe6b646d2d4d3e5869a60c1b60e00b77b230375f9bd0cab02ed4225c4b848` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_two_horizon_wait_audit/` | `tree` | 222 | 3.00 MiB | `7c946e567f56910e12bd99e7703e50288dfa374aebd824e41f73ac2e2739d7c8` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v0_smoke/` | `tree` | 3 | 15.32 KiB | `a537fa75ada93179aea438f11b537e419af481b7617640d8dd6728a3de339ead` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v1_environment_freeze.txt` | `file` | 1 | 926.00 B | `e5090eac0e56b18306274325abecc32297f5e4d008ec023f1c3de2b1c1d10ce1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v1_fixed_validation/` | `tree` | 1 | 8.17 KiB | `50b5582b5dabb9394aa18e1390fb555a3a6a652f5a60ba5d052d6345e1360831` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v1_h120_seed0/` | `tree` | 4 | 1.61 MiB | `af8fe988390522e543819942e35fddea3b243d1cbdd23896c2a3b9ffeb3eb5c8` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v1_timeout_challenge/` | `tree` | 7 | 1.80 MiB | `315c05fb6d9623aba56899151343f5ae0f9f2d63b4f6a01c90d43a1733a29063` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_episode_eval/` | `tree` | 8 | 3.39 MiB | `f5f79a83ca3783dc58c0ef764fa3a53c0708e0dba7d8f1e4f9b2fbc959373a6c` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_episode_scan_resources.txt` | `file` | 1 | 969.00 B | `7de6f5c9221461710ebe28ad8fd3a12bdf94b4ac6c38c889fd33b8fb6449dc0e` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_episode_seed0_scan/` | `tree` | 4 | 1.61 MiB | `0d03246c6c08567ae5b9bb0f919a41e9b33ec81911bcf60874c7ffcadec701eb` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_episode_seed0_smoke/` | `tree` | 2 | 8.55 KiB | `46d2897ae42d8b6b22c52a010c7ad44a17604b028d208ca795082f38b7b7efb1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_episode_validation/` | `tree` | 1 | 25.76 KiB | `9dc33da39fc424f9f906fae3201dde5e098dbe37cf3068bb38c2a58e8a27dbfb` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_frozen_seed0/` | `tree` | 4 | 1.62 MiB | `185a25b034dfd6091f53a638a869e43d768704aef688b495cc64e3d73ed9ee22` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v2_frozen_sets/` | `tree` | 3 | 1.55 MiB | `90a34a6a97efdc2d77478123a091e1dcb7fdac5e756bfc6425aab36893375fb4` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v3_audit/` | `tree` | 8 | 203.02 KiB | `d51ddf77dc65907c0d8a748dba01f71023dc45e124fbfe6a60351d3cc8b867e1` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v3_risk_outcome_association/` | `tree` | 18 | 180.89 KiB | `35d57ac49a146fe0887c78587d7fdf7a6f04ce41e69ae2b87559a29ef4ca0063` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_v3_smoke/` | `tree` | 35 | 6.94 MiB | `92b78785beab5de65d82e67370dccd00be0abdf745d0020e00fa5546b062ea77` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_vi_adfd_layers_v1/` | `tree` | 1 | 93.36 KiB | `a8b3607f8f18451ef81b045430d611ae5b22fc26ca0e6a641828ee5891451e71` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_vi_adfd_localize_v1/` | `tree` | 1 | 103.94 KiB | `e157c71a804844d1c1ee37e02d695f41eedda5ac8c198a53d69868b1b2fb5909` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_vi_r_cert_check_v1/` | `tree` | 1 | 4.31 KiB | `d946130a60ccfb245fd9782d5c647db9fdbb74a670658e4ac33664ab402428b0` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_vi_r_cert_counterexamples_v1.json` | `file` | 1 | 3.56 KiB | `23f02d6859e1dd186a5c5124b7dd835be8ef6392655b30b91854aca2311dd3b0` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_vi_r_cert_probe_v1/` | `tree` | 482 | 23.63 MiB | `56cf854b9eccf9cfda5bb8c03e08a035b4880ad7c981c6dc4cb97014dc8f3ef5` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_vi_vjp_one_sided_v2/` | `tree` | 3 | 1.85 MiB | `84064af3f556127c2669a26cd46047e042dba3924beaefa266e9043c5a449f2b` | explicit R_risk / VI-R_CERT 冻结审计证据 |
| `results/c1_wait_horizon_sweep/` | `tree` | 6 | 666.05 KiB | `8b7768fa7e568a06c650041645748369b711ce9628a21c3e86a8527d8659f8e5` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_weak_activity_counterexample.json` | `file` | 1 | 645.00 B | `4a225e188eb401648e04746ee7c14f14dccbacdb328766f2917d66b70c0acdb2` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/c1_zero_event_confidence_audit.json` | `file` | 1 | 521.00 B | `312fcac19e0399173cb6af798dabf6eb2d9d7b8e13d6c7452be653333254ecdf` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/eval_c1_primal_dual_seed0/` | `tree` | 203 | 55.53 MiB | `71f17cd11d48306eb0adddca5948f08d4940b8557c73341b28d64aff75d5a529` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/extended_h120_COMPARISON.md` | `file` | 1 | 2.81 KiB | `196cfa8ee4b91c376142e70386dd8aac44243dd935eb5652025bfa2306a76398` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/extended_h120_comparison_soft_risk.json` | `file` | 1 | 28.79 KiB | `67dfb42a5700c1c4cacc4b0d8cfef18399f009680bab83dc44cea4cdf01c178d` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/extended_h120_eval_seed0_baselines/` | `tree` | 56 | 12.65 MiB | `ddb4b6e65bdef1a092cf1b7f1e1d0296ef8732d35fcc864bf4441b988c4292aa` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/extended_h120_eval_seed0_c1/` | `tree` | 28 | 9.35 MiB | `ce4831a66cf1eab95225f0423fc5963ced8ea509634cbd801ea44f68fdda6218` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |
| `results/giveway_residual_effect_pilot_v1/` | `tree` | 134 | 19.12 MiB | `9e75eb5fdd012ff2365b3f81eecf51d85c60a817de0793ded86e229fa5589b75` | Direction A 实现审计/独立诊断证据 |
| `results/risk_audit_seed0_baselines/` | `tree` | 406 | 95.20 MiB | `37c4cce50bc7fd88b2cf78b556d7924020a98c59efa50e0419bb198c1e0b15cc` | 历史 C1 实验/诊断资产；不作为 main 默认方法 |

资产条目数：187。

所有大型资产保留在原路径；Git 只显式纳入路线所需的小型 JSON/Markdown 摘要，二进制 rollout、checkpoint 与 dataset 不提交。
