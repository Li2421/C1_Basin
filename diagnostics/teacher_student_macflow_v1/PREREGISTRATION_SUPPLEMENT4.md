# Round30 第四次运行补充预注册

UTC: 2026-10-15T07:02:24.148985+00:00

此前原预注册与第一次补充均保持内容不变。本次保存第三次REPORT/HANDOFF/results及审计到attempt3_db_registration/。未训练或读取任何学生结果。复用第三次噪声门1,008/1,008及完整重建数据的hash核验；未完成数据重做。

本次修改严格来自新版任务：① design允许用官方connect/transaction对experiment/state/controller_config/eta（需要时scenario）INSERT OR IGNORE新身份，不UPDATE/DELETE或直接写rollout；uid冲突停止；记录新增行及hash，验证原行hash与已有experiment rollout计数不变。② pipeline.py、episode与core复制到本目录，逻辑不变，仅依文件位置将路径和环境变量定位到新目录；不直接import round28 pipeline。先用复制版复现16条D-local记录：教师按round28顺序取前三个global及首个local教师，每教师F10/S500×前两个state×seed0。首动作、终止类型、步数、完整位置逐位一致，否则停止。复制源码hash及diff见COPY_PROVENANCE.json/COPY_DIFF.patch。

学生通过runtime缓存参数接入，copied decentralized_action不改。safety=local eta0；nominal-only及1.5m B-RHR在单独新函数中实现并测试。private noise=fold_in(key_for(state,seed,step),i)，local clock。所有其余训练参数、14seed、能力门与统计终点保持原预注册。每阶段preflight先于提交，unknown不重跑。
