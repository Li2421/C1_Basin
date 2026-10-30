# 相关工作的外部基线可行性

核查日期：2026-10-30。本次未找到可核验为该论文完整官方实现的代码与模型，因此不自行实现、不接入；若后续找到，先原环境复现，再在接入前等用户决定。

论文：[Deadlock-Free Hybrid RL-MAPF Framework for Zero-Shot Multi-Robot Navigation](https://arxiv.org/abs/2511.22685)，当前可访问页面为v2。项目页：[作者项目页面](https://wanghaoyi518.github.io/rl-mapf-project-page/)。项目页代码按钮标注“coming soon”，实际指向占位链接 https://github.com/USERNAME/REPO ，访问返回404。作者个人主页根路径当前不可访问，已保存HTTP错误；[作者GitHub](https://github.com/wanghaoyi518)与公开仓库API可访问。

公开仓库中存在[rl_rvo_nav](https://github.com/wanghaoyi518/rl_rvo_nav)，是2022年RL导航工作的fork，不能据仓库名称认定为这篇混合方法实现。已保存完整目录树和41个Python/Markdown/YAML文件，目录含deadlock配置和GIF，但所检索Python与Markdown未出现MAPF、Push-and-Rotate、该论文编号或deadlock实现关键词。此项是有限来源核查，不是“作者绝无其他代码”的证明。

目前证据不足以核验混合接口、匹配的RL模型、原论文复现和在本环境是否需要重训；不编造工作量与rollout估计。原论文复现次数为0。用户注明的PMLR2026版本本次尚未找到可直接核验的正式页面，暂不宣称已读该版本。

来源、时间与快照哈希见 RELATED_BASELINE_SOURCE_CHECK.json 及 inputs/related_baseline_web/。后者含官方项目HTML、作者仓库API、相关fork的固定Git树与文本文件清单。
