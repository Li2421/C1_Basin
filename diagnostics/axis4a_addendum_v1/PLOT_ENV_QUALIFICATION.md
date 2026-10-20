# 绘图环境补充

首次最终作业22136在导入matplotlib时退出，没有执行任何新仿真，也没有改变统计结果。原错误记录与日志保留。原指定 .venv-c1 未安装/升级任何包；按共同协议缺依赖时在本轮另建环境的规则，在本轮plot_env中安装了matplotlib及绘图依赖，pip缓存/tmp均在本轮目录。

独立环境仅运行冻结plot_curves.py，读取CONTROLLER_RADIUS_CURVES.json生成14个controller的逐半径曲线PNG/PDF，不读取或更改控制器/物理。全部仿真、统计、DB、审计和交付复核仍用原指定Python -B。精确环境版本见PLOT_ENV_PACKAGES.json。本补充不改变GA1/GA2/GA3、B的未决定状态或C的结论。
