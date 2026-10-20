# 集中式模型预测路径积分控制的来源

Williams, Drews, Goldfain, Rehg, Theodorou. Aggressive Driving with Model Predictive Path Integral Control. ICRA 2016, 1433–1440. DOI: 10.1109/ICRA.2016.7487277。

作者所属机构的原始出版信息及摘要：[Illinois Experts](https://experts.illinois.edu/en/publications/aggressive-driving-with-model-predictive-path-integral-control/)。作者实验室的研究索引将2016 ICRA列为首版：[Georgia Tech ACDS](https://sites.gatech.edu/acds/mppi/)。出版信息与作者摘要已检索；实验室网页直接抓取遇到502或连接关闭，未据此声称获得该会议论文全文。

本轮严格实现用户prompt指定的Ring适配：联合动作序列、0.8×最大速度的采样标准差、逐字调用round27 shooting_cost、exp(−cost/λ)稳定归一化加权均值、每步平移热启动。代价、动力学、温度网格由prompt固定。这是指定的Ring优化控制器，不声称复现论文中的车辆动力学或原车辆实验。零温度极限与同键逐位重现测试见MIXED_CONTROLLER_TESTS.json。
